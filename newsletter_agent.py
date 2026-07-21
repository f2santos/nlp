"""
newsletter_agent.py
--------------------
Agentic pipeline behind the "Newsletter pessoal" page: turns what the user
searches in "Explorar tendências" into a personalised weekly digest, using a
local instruction-tuned LLM (no paid API key — same "open models only"
principle as the distilBART summariser in nlp_pipeline.py).

Pipeline (deterministic multi-step, with a bounded quality-control loop):

    compact_context -> plan_queries -> fetch_candidates -> select_top10
        -> summarize_and_preview -> critique -> (refine, up to MAX_REFINE_LOOPS)

Results are cached per ISO week in data/newsletter_cache/, so the LLM and
the arXiv API are only hit when the week changes or the user forces a
regeneration — not on every page view.
"""

import datetime
import functools
import json
import re
from collections import Counter
from pathlib import Path

import arxiv_api
import nlp_pipeline as nlp

# Fast/light model for validating the workflow end-to-end on CPU. Same
# family as Qwen2.5-1.5B-Instruct (identical chat template), ~3x smaller —
# once the pipeline is confirmed working, swap this one constant to scale
# up to "Qwen/Qwen2.5-1.5B-Instruct" (or larger) for better output quality.
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_REFINE_LOOPS = 1
TOP_N = 10
MAX_CANDIDATES = 16  # hard cap on candidates fed to the LLM — keeps prompts
                     # (and CPU prefill time) bounded regardless of query count

DATA_DIR = Path(__file__).parent / "data"
HISTORY_PATH = DATA_DIR / "search_history.jsonl"
CACHE_DIR = DATA_DIR / "newsletter_cache"
PROFILES_DIR = DATA_DIR / "profiles"  # one JSON file per person (by CV name)

MONTHS_PT = ["", "janeiro", "fevereiro", "março", "abril", "maio", "junho",
             "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]

# Bootstrap profile used only when there is no search history yet, so the
# newsletter is never empty on first use.
DEFAULT_PROFILE = {
    "categories": ["cs.CL", "cs.AI"],
    "keywords": ["retrieval augmented generation"],
}


# ==========================================================================
#  Local LLM — lazy singleton, chat-template prompting, JSON-tolerant parsing
# ==========================================================================
@functools.lru_cache(maxsize=1)
def _load_llm():
    from transformers import pipeline
    # No device_map="auto" here: on this CPU-only machine, accelerate's
    # automatic dispatch mis-estimates available memory and tries to
    # disk-offload part of the model, which errors without an
    # offload_folder. The model comfortably fits in RAM, so just load it
    # straight onto the default (CPU) device.
    return pipeline("text-generation", model=LLM_MODEL_NAME, dtype="auto")


def _try_parse_json(text):
    match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _chat(system, user, max_new_tokens=300, expect_json=False):
    """Run one chat-formatted generation. If expect_json, try to parse the
    output as JSON, with one corrective retry if parsing fails — this retry
    is the module's cheapest, most common form of self-correction."""
    gen = _load_llm()
    tok = gen.tokenizer
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False,
              return_full_text=False, pad_token_id=tok.eos_token_id)
    text = out[0]["generated_text"].strip()
    if not expect_json:
        return text

    data = _try_parse_json(text)
    if data is not None:
        return data

    messages.append({"role": "assistant", "content": text})
    messages.append({"role": "user", "content": (
        f"A tua resposta anterior não era JSON válido:\n{text}\n\n"
        "Responde APENAS com JSON válido, sem texto à volta.")})
    prompt2 = tok.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True)
    out2 = gen(prompt2, max_new_tokens=max_new_tokens, do_sample=False,
               return_full_text=False, pad_token_id=tok.eos_token_id)
    return _try_parse_json(out2[0]["generated_text"].strip())


# ==========================================================================
#  Search history — what feeds the compacted context
# ==========================================================================
def record_search(subject_label, category, keyword, mode, profile_slug=None):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
             "mode": mode, "subject_label": subject_label,
             "category": category, "keyword": keyword or "",
             "profile_slug": profile_slug or ""}
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_recent_history(days=30, profile_slug=None):
    """If profile_slug is given, only entries recorded under that profile
    are returned; otherwise all recent entries are returned (this is also
    the behaviour when no profile has ever been created)."""
    if not HISTORY_PATH.exists():
        return []
    cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
    out = []
    with open(HISTORY_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.datetime.fromisoformat(entry["ts"])
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            if ts < cutoff:
                continue
            if profile_slug and entry.get("profile_slug") != profile_slug:
                continue
            out.append(entry)
    return out


# ==========================================================================
#  Profile — who the newsletter is for (extracted from an uploaded CV)
# ==========================================================================
def extract_text_from_pdf(file_bytes):
    from pypdf import PdfReader
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _sanitize_cv_profile(result):
    if not isinstance(result, dict):
        return None
    name = (result.get("name") or "").strip()
    summary = (result.get("summary") or "").strip()
    if not name or not summary:
        return None

    valid_categories = set(arxiv_api.CATEGORIES.values())
    categories = [c for c in (result.get("categories") or [])
                 if c in valid_categories][:3] or DEFAULT_PROFILE["categories"]
    keywords = [k for k in (result.get("keywords") or [])
               if isinstance(k, str) and k.strip()][:3] or \
        DEFAULT_PROFILE["keywords"]
    return {"name": name, "summary": summary, "categories": categories,
           "keywords": keywords}


def analyze_cv(pdf_bytes):
    """Extract a name + interest profile from an uploaded CV PDF via the
    local LLM. Returns None if the PDF has no usable text or the model's
    output can't be sanitized into a usable profile — callers should show
    an error rather than silently falling back, since this is a direct,
    explicit user action."""
    text = extract_text_from_pdf(pdf_bytes).strip()
    if len(text) < 50:
        return None

    valid_categories = ", ".join(sorted(set(arxiv_api.CATEGORIES.values())))
    user_prompt = (
        f"Currículo (texto extraído de PDF):\n{text[:6000]}\n\n"
        "A partir deste currículo, identifica:\n"
        "1. O nome completo da pessoa.\n"
        "2. Um resumo curto (1-2 frases, português) das áreas de "
        "investigação/trabalho que parecem ser do interesse desta pessoa, "
        "com base na experiência, competências e projetos descritos.\n"
        "3. Até 3 categorias arXiv relevantes, escolhidas APENAS desta "
        f"lista: {valid_categories}\n"
        "4. Até 3 palavras-chave em inglês para pesquisar artigos "
        "científicos sobre a área desta pessoa.\n\n"
        "Responde APENAS com JSON válido. Exemplo do FORMATO (inventa "
        "sempre conteúdo novo, a partir do currículo acima — nunca copies "
        "este exemplo):\n"
        '{"name": "Maria Silva", "summary": "Trabalha em visão '
        'computacional aplicada a robótica, com foco recente em modelos '
        'de deteção de objetos em tempo real.", "categories": ["cs.CV", '
        '"cs.RO"], "keywords": ["object detection", "robotics"]}'
    )
    result = _chat(
        system=("Extrais informação estruturada de currículos para "
               "personalizar recomendações de artigos científicos."),
        user=user_prompt, max_new_tokens=250, expect_json=True,
    )
    return _sanitize_cv_profile(result)


def _slugify(name):
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "utilizador"


def save_profile(profile):
    """Saves one JSON file per person (keyed by their CV name), so multiple
    people can each have their own profile on the same local install.
    Returns the slug, which callers use to mark this profile as active."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    slug = _slugify(profile["name"])
    profile = {**profile, "slug": slug}
    (PROFILES_DIR / f"{slug}.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    return slug


def list_profiles():
    """All saved profiles, most recently updated first."""
    if not PROFILES_DIR.exists():
        return []
    files = sorted(PROFILES_DIR.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for fp in files:
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def load_profile(slug=None):
    """The profile matching slug, or the most recently updated one if slug
    is None/not found, or None if no profile has been created yet."""
    profiles = list_profiles()
    if not profiles:
        return None
    if slug:
        for p in profiles:
            if p.get("slug") == slug:
                return p
    return profiles[0]


# ==========================================================================
#  Step 1 — compact context
# ==========================================================================
def compact_context(history, profile_slug=None):
    profile = load_profile(profile_slug)

    if not history:
        if profile:
            return {"summary": profile["summary"],
                   "top_categories": profile["categories"],
                   "top_keywords": profile["keywords"], "n_searches": 0}
        return {
            "summary": ("Ainda sem pesquisas nem CV carregado — a mostrar "
                        "sugestões iniciais alinhadas com RAG e IA aplicada."),
            "top_categories": DEFAULT_PROFILE["categories"],
            "top_keywords": DEFAULT_PROFILE["keywords"],
            "n_searches": 0,
        }

    base_categories = profile["categories"] if profile \
        else DEFAULT_PROFILE["categories"]
    base_keywords = profile["keywords"] if profile else []
    cat_counts = Counter(h["category"] for h in history if h.get("category"))
    kw_counts = Counter(h["keyword"] for h in history if h.get("keyword"))
    top_categories = [c for c, _ in cat_counts.most_common(3)] or base_categories
    top_keywords = [k for k, _ in kw_counts.most_common(3)] or base_keywords

    stats_text = (f"{len(history)} pesquisas recentes. Categorias mais "
                 f"pesquisadas: {', '.join(top_categories)}."
                 + (f" Palavras-chave: {', '.join(top_keywords)}."
                    if top_keywords else ""))
    if profile:
        stats_text += f" Perfil profissional (do CV): {profile['summary']}"

    summary = _chat(
        system=("És um assistente que resume interesses de investigação em "
               "português europeu, em 1-2 frases curtas e naturais, sem "
               "introduções como 'Aqui está'."),
        user=f"Estatísticas de um utilizador:\n{stats_text}\n\n"
            "Escreve um resumo curto (1-2 frases) do que este utilizador "
            "parece estar a investigar ultimamente, combinando o perfil "
            "profissional com as pesquisas recentes (quando ambos existirem).",
        max_new_tokens=100,
    )
    return {"summary": (summary or "").strip() or stats_text,
           "top_categories": top_categories, "top_keywords": top_keywords,
           "n_searches": len(history)}


# ==========================================================================
#  Step 2 — plan arXiv queries
# ==========================================================================
def _sanitize_queries(result, context):
    valid_categories = set(arxiv_api.CATEGORIES.values())
    queries = []
    if isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue
            cat = item.get("category")
            if cat in valid_categories:
                queries.append({"category": cat,
                               "keyword": (item.get("keyword") or "").strip()})
    if not queries:
        cats = context["top_categories"] or DEFAULT_PROFILE["categories"]
        kw = (context["top_keywords"][0] if context["top_keywords"]
             else DEFAULT_PROFILE["keywords"][0])
        queries = [{"category": c, "keyword": kw} for c in cats[:2]]
    return queries[:3]


def plan_queries(context):
    user_prompt = (
        f"Contexto do utilizador: {context['summary']}\n"
        f"Categorias mais pesquisadas: "
        f"{', '.join(context['top_categories']) or 'nenhuma'}\n"
        f"Palavras-chave: {', '.join(context['top_keywords']) or 'nenhuma'}\n"
        f"Categorias arXiv válidas: {', '.join(arxiv_api.CATEGORIES.values())}\n\n"
        "Sugere entre 1 e 3 pesquisas no arXiv para encontrar os papers mais "
        "relevantes para este utilizador esta semana. Responde APENAS com "
        'JSON: uma lista de objetos {"category": "<código arXiv válido>", '
        '"keyword": "<palavra-chave ou vazio>"}.'
    )
    result = _chat(
        system=("És um assistente de investigação que decide que pesquisas "
               "fazer no arXiv para satisfazer os interesses de um "
               "utilizador."),
        user=user_prompt, max_new_tokens=120, expect_json=True,
    )
    return _sanitize_queries(result, context)


def fetch_candidates(queries, per_query=8):
    seen = set()
    candidates = []
    for q in queries:
        docs = arxiv_api.collect_papers(q["category"], max_results=per_query,
                                        keyword=q["keyword"] or None)
        for d in docs:
            key = d.get("link") or d.get("title")
            if key in seen:
                continue
            seen.add(key)
            candidates.append(d)
    return candidates[:MAX_CANDIDATES]


# ==========================================================================
#  Step 3 — select top 10 (topic clustering + LLM judgment for diversity)
# ==========================================================================
def _sanitize_indices(result, n_candidates, n):
    idx = []
    if isinstance(result, list):
        for v in result:
            if isinstance(v, int) and 0 <= v < n_candidates and v not in idx:
                idx.append(v)
    if len(idx) < n:
        for i in range(n_candidates):
            if len(idx) >= n:
                break
            if i not in idx:
                idx.append(i)
    return idx[:n]


def select_top10(candidates, context, n=TOP_N):
    if len(candidates) <= n:
        return candidates

    n_clusters = min(5, max(2, len(candidates) // 10))
    try:
        nlp.discover_topics(candidates, n_topics=n_clusters, method="kmeans")
    except Exception:
        for d in candidates:
            d.setdefault("topic", -1)

    listing = "\n".join(
        f"{i}. [cluster {d.get('topic', -1)}] {d['title']} — "
        f"{d['abstract'][:100]}"
        for i, d in enumerate(candidates)
    )
    user_prompt = (
        f"Contexto do utilizador: {context['summary']}\n\n"
        f"Candidatos (numerados, agrupados por cluster de sub-tópico):\n"
        f"{listing}\n\n"
        f"Escolhe os {n} mais relevantes E diversos (tenta cobrir vários "
        "clusters, não só o maior). Responde APENAS com JSON: lista dos "
        "índices escolhidos, ex: [0, 3, 7, ...]."
    )
    result = _chat(
        system=("És um editor de newsletter que seleciona os melhores "
               "artigos científicos para um utilizador, equilibrando "
               "relevância e diversidade de temas."),
        user=user_prompt, max_new_tokens=100, expect_json=True,
    )
    indices = _sanitize_indices(result, len(candidates), n)
    return [candidates[i] for i in indices]


# ==========================================================================
#  Step 4 — summaries + weekly preview (one batched LLM call)
# ==========================================================================
def _sanitize_summaries(result, selected):
    fallback_preview = ("A seleção desta semana cobre os temas que mais "
                        "tens pesquisado.")
    fallback_summaries = [d["abstract"][:180].rsplit(" ", 1)[0] + "…"
                          for d in selected]
    if not isinstance(result, dict):
        return fallback_preview, fallback_summaries
    preview = result.get("preview") or fallback_preview
    summaries = result.get("summaries")
    if not isinstance(summaries, list) or len(summaries) != len(selected):
        summaries = fallback_summaries
    else:
        summaries = [s if isinstance(s, str) and s.strip() else fb
                     for s, fb in zip(summaries, fallback_summaries)]
    return preview, summaries


def summarize_and_preview(selected, context):
    listing = "\n".join(
        f"{i}. {d['title']} — {d['abstract'][:180]}"
        for i, d in enumerate(selected)
    )
    user_prompt = (
        f"Contexto do utilizador: {context['summary']}\n\n"
        f"Artigos selecionados ({len(selected)} no total):\n{listing}\n\n"
        "Para cada artigo, escreve um resumo de UMA frase curta em "
        "português, claro e específico (não repitas o título). Escreve "
        "também um parágrafo de preview (1-2 frases, português) sobre os "
        "destaques desta semana.\n\n"
        "Responde APENAS com JSON válido. Exemplo do FORMATO (inventa "
        "sempre conteúdo novo — nunca copies este exemplo):\n"
        '{"preview": "Esta semana destacam-se novos métodos de recuperação '
        'híbrida e agentes para tarefas científicas.", "summaries": '
        '["Este artigo propõe uma técnica que reduz o custo de indexação '
        'em bases vetoriais grandes.", "Este trabalho mostra que agentes '
        'multi-passo superam abordagens de único passo em tarefas '
        'complexas."]}\n\n'
        f'A tua lista "summaries" tem de ter exatamente {len(selected)} '
        "elementos — um por artigo, pela mesma ordem da lista acima."
    )
    result = _chat(
        system=("És um editor de newsletter científica que escreve resumos "
               "curtos, claros e fiéis ao conteúdo original, em português "
               "europeu."),
        user=user_prompt, max_new_tokens=480, expect_json=True,
    )
    return _sanitize_summaries(result, selected)


# ==========================================================================
#  Step 5 — quality-control critique
# ==========================================================================
def critique(draft):
    items = draft["items"]
    issues = []
    if len(items) != TOP_N:
        issues.append(f"esperado {TOP_N} itens, encontrados {len(items)}")
    titles = [it["title"] for it in items]
    if len(set(titles)) != len(titles):
        issues.append("há títulos duplicados")
    for it in items:
        summ = (it.get("summary_pt") or "").strip()
        if len(summ) < 15 or summ.lower() in it["title"].lower():
            issues.append(f"resumo fraco/repetido para: {it['title'][:60]}")

    if issues:
        return {"ok": False, "issues": issues}

    listing = "\n".join(f"- {it['title']}: {it['summary_pt']}" for it in items)
    user_prompt = (
        f"Preview: {draft['preview']}\n\nItens:\n{listing}\n\n"
        "Avalia esta newsletter: os itens são relevantes e diversos entre "
        "si? Os resumos são específicos e úteis? Responde APENAS com JSON. "
        'Se estiver tudo bem: {"ok": true, "issues": []}. Se houver '
        'problemas, descreve-os: {"ok": false, "issues": ["resumo 3 é '
        'vago", "item 7 não parece relevante"]}.'
    )
    result = _chat(
        system="És um crítico exigente de newsletters científicas.",
        user=user_prompt, max_new_tokens=120, expect_json=True,
    )
    if isinstance(result, dict) and "ok" in result:
        return {"ok": bool(result["ok"]), "issues": result.get("issues") or []}
    # critic itself failed to produce usable JSON — don't block forever
    return {"ok": True, "issues": []}


# ==========================================================================
#  Orchestrator — cache-aware, with the bounded refine loop
# ==========================================================================
def _iso_week_key(d=None):
    d = d or datetime.date.today()
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def week_range_label(d=None):
    d = d or datetime.date.today()
    monday = d - datetime.timedelta(days=d.weekday())
    sunday = monday + datetime.timedelta(days=6)
    if monday.month == sunday.month:
        return f"{monday.day}–{sunday.day} de {MONTHS_PT[monday.month]}"
    return (f"{monday.day} de {MONTHS_PT[monday.month]} – "
            f"{sunday.day} de {MONTHS_PT[sunday.month]}")


def _build_draft(context, week_key, attempt, progress=None):
    def step(msg):
        if progress:
            progress(msg)

    step("A planear pesquisas no arXiv com base no teu contexto...")
    queries = plan_queries(context)

    step(f"A procurar candidatos no arXiv ({', '.join(q['category'] for q in queries)})...")
    candidates = fetch_candidates(queries)
    if not candidates:
        queries = [{"category": c, "keyword": ""}
                  for c in DEFAULT_PROFILE["categories"]]
        candidates = fetch_candidates(queries)

    step(f"A selecionar os {TOP_N} mais relevantes e diversos "
        f"({len(candidates)} candidatos)...")
    selected = select_top10(candidates, context)

    step("A escrever resumos e o preview da semana...")
    preview, summaries = summarize_and_preview(selected, context)
    items = [
        {"title": d["title"], "authors": d["authors"][:4], "summary_pt": s,
         "arxiv_id": d["link"].rsplit("/", 1)[-1] if d.get("link") else "",
         "link": d["link"], "pdf_url": d["pdf_url"]}
        for d, s in zip(selected, summaries)
    ]
    return {"week_key": week_key, "week_label": week_range_label(),
           "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "interest_summary": context["summary"], "preview": preview,
           "items": items, "queries_used": queries,
           "refine_iterations": attempt}


def _cache_dir_for(profile_slug):
    """Cache lives in a per-profile subdirectory (not a filename prefix) so
    similar slugs — e.g. 'ana' vs 'ana-silva' — can never collide."""
    return CACHE_DIR / (profile_slug or "default")


def generate_weekly_newsletter(force=False, progress=None, profile_slug=None):
    """progress, if given, is called with a short Portuguese status string
    before each pipeline stage — lets the UI show real step-by-step
    feedback instead of an opaque spinner during the (rare) 1-2 min a full
    generation takes."""
    week_key = _iso_week_key()
    cache_dir = _cache_dir_for(profile_slug)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{week_key}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if progress:
        progress("A compactar o teu contexto de pesquisas...")
    history = load_recent_history(profile_slug=profile_slug)
    context = compact_context(history, profile_slug=profile_slug)

    draft = None
    for attempt in range(MAX_REFINE_LOOPS + 1):
        draft = _build_draft(context, week_key, attempt, progress=progress)
        if progress:
            progress("A rever a qualidade da seleção...")
        result = critique(draft)
        if result["ok"]:
            draft["quality_flag"] = None
            break
        draft["quality_flag"] = "; ".join(result["issues"]) \
            or "qualidade não confirmada"
        if progress and attempt < MAX_REFINE_LOOPS:
            progress("Qualidade insuficiente — a refinar a seleção...")

    draft["profile_slug"] = profile_slug or ""
    cache_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return draft


def peek_current_issue(profile_slug=None):
    """Read the current week's cached issue without triggering generation —
    used by the sidebar so just viewing it never calls the LLM."""
    cache_path = _cache_dir_for(profile_slug) / f"{_iso_week_key()}.json"
    if not cache_path.exists():
        return None
    return json.loads(cache_path.read_text(encoding="utf-8"))


def list_past_issues(profile_slug=None, limit=4):
    cache_dir = _cache_dir_for(profile_slug)
    if not cache_dir.exists():
        return []
    current = _iso_week_key()
    files = sorted(cache_dir.glob("*.json"), reverse=True)
    out = []
    for fp in files:
        if fp.stem == current:
            continue
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
        if len(out) >= limit:
            break
    return out
