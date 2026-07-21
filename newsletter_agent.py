"""
newsletter_agent.py
--------------------
Agentic pipeline behind the "Newsletter pessoal" page: turns what the user
searches in "Explore trends" into a personalised weekly digest, using a
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
import time
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

MONTHS = ["", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Bootstrap profile used only when there is no search history / CV yet, so the
# newsletter is never empty on first use.
DEFAULT_PROFILE = {
    "categories": ["cs.CL", "cs.AI"],
    "keywords": ["retrieval augmented generation"],
}


def _dedup(seq):
    """Order-preserving de-duplication, dropping empties."""
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


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
        f"Your previous answer was not valid JSON:\n{text}\n\n"
        "Reply ONLY with valid JSON, no surrounding text.")})
    prompt2 = tok.apply_chat_template(messages, tokenize=False,
                                      add_generation_prompt=True)
    out2 = gen(prompt2, max_new_tokens=max_new_tokens, do_sample=False,
               return_full_text=False, pad_token_id=tok.eos_token_id)
    return _try_parse_json(out2[0]["generated_text"].strip())


# ==========================================================================
#  Search history — what feeds the compacted context
# ==========================================================================
def record_search(subject_label, category, keyword, mode):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
             "mode": mode, "subject_label": subject_label,
             "category": category, "keyword": keyword or ""}
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_recent_history(days=30):
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
            if ts >= cutoff:
                out.append(entry)
    return out


# ==========================================================================
#  CV profile — extract the user's name + research interests from a pasted CV
# ==========================================================================
CV_PROFILE_PATH = DATA_DIR / "cv_profile.json"

# Best-effort mapping from free-text interests to the arXiv category codes we
# support. First matching hint wins per category.
_INTEREST_CATEGORY_HINTS = [
    ("cs.CL", ("nlp", "natural language", "language", "linguistic",
               "translation", "text mining", "retrieval", "rag",
               "question answering", "summari")),
    ("cs.CV", ("vision", "image", "video", "segmentation", "object detection")),
    ("cs.LG", ("machine learning", "deep learning", "neural network",
               "representation learning", "self-supervised")),
    ("cs.AI", ("artificial intelligence", "reasoning", "agent", "planning",
               "knowledge graph")),
    ("cs.RO", ("robot", "manipulation", "autonomous", "control")),
    ("cs.CR", ("security", "cryptograph", "privacy", "adversarial")),
    ("cs.HC", ("human-computer", "human computer", "interaction", "usability")),
    ("cs.SE", ("software engineering", "software", "program analysis")),
    ("stat.ML", ("statistical learning", "bayesian", "statistics")),
    ("eess.SP", ("signal processing", "signal")),
    ("math.OC", ("optimization", "optimisation", "control theory")),
    ("quant-ph", ("quantum",)),
    ("astro-ph", ("astro", "cosmolog", "galax")),
    ("q-bio", ("biolog", "genom", "protein", "neuroscience")),
    ("q-fin.ST", ("finance", "financial", "trading", "market")),
    ("econ.EM", ("econometric", "econom")),
]


def _map_interests_to_categories(interests, limit=3):
    """Map free-text interests to supported arXiv category codes."""
    valid = set(arxiv_api.CATEGORIES.values())
    found = []
    for interest in interests:
        low = interest.lower()
        for cat, hints in _INTEREST_CATEGORY_HINTS:
            if cat in valid and cat not in found \
                    and any(h in low for h in hints):
                found.append(cat)
    return found[:limit] or list(DEFAULT_PROFILE["categories"])


def _shorten_term(term):
    """Reduce a free-text interest to a short, arXiv-searchable keyword: take
    the first segment before a separator (& / , : ' and ' dash) and keep at
    most 3 words. Bare hyphens are preserved (e.g. 'retrieval-augmented')."""
    term = (term or "").strip()
    if not term:
        return ""
    term = re.split(r"\s*(?:&|/|,|:|\band\b|–|—| - )\s*", term, maxsplit=1)[0]
    return " ".join(term.split()[:3]).strip().lower()


def _guess_name_fallback(cv_text):
    """Heuristic used when the LLM fails to return a name: a CV's name is
    usually the first short, capitalised, digit/@-free line at the top."""
    lines = [ln.strip() for ln in cv_text.splitlines()[:15] if ln.strip()]
    for ln in lines:
        words = ln.split()
        if 1 < len(words) <= 4 and "@" not in ln \
                and not any(ch.isdigit() for ch in ln) \
                and sum(w[:1].isupper() for w in words) >= 2:
            return ln
    return lines[0][:40] if lines else ""


def extract_cv_profile(cv_text):
    """
    Extract the person's name and research interests from pasted CV text, using
    the local LLM with a deterministic fallback. Interests are also mapped to
    arXiv categories, which is what drives the personalised newsletter.
    Returns {name, interests: [...], categories: [...]} or None for empty input.
    """
    cv_text = (cv_text or "").strip()
    if not cv_text:
        return None
    result = _chat(
        system=("You extract structured information from CVs / resumes. You "
                "reply ONLY with valid JSON, no surrounding text."),
        user=(f"Here is the text of a CV:\n\n{cv_text[:4000]}\n\n"
              "Extract: (1) the person's full name; (2) 3 to 6 research "
              "interests or technical topics they focus on. Reply ONLY with "
              'JSON: {"name": "...", "interests": ["...", "..."]}.'),
        max_new_tokens=200, expect_json=True,
    )
    name, interests = "", []
    if isinstance(result, dict):
        name = str(result.get("name") or "").strip()
        raw = result.get("interests")
        if isinstance(raw, list):
            interests = [str(x).strip() for x in raw if str(x).strip()]
    if not name:
        name = _guess_name_fallback(cv_text)
    interests = interests[:6]
    # short, searchable keywords derived from the (possibly long) interests
    keywords = _dedup(_shorten_term(i) for i in interests)
    return {"name": name or "Reader", "interests": interests,
            "keywords": keywords,
            "categories": _map_interests_to_categories(interests)}


def save_cv_profile(profile):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CV_PROFILE_PATH.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cv_profile():
    if not CV_PROFILE_PATH.exists():
        return None
    try:
        return json.loads(CV_PROFILE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clear_cv_profile():
    CV_PROFILE_PATH.unlink(missing_ok=True)


def invalidate_current_week():
    """Drop this week's cached issue so the next view regenerates it (used when
    the CV profile changes, since it feeds the newsletter)."""
    (CACHE_DIR / f"{_iso_week_key()}.json").unlink(missing_ok=True)


# ==========================================================================
#  Step 1 — compact context (search history + CV interests)
# ==========================================================================
def compact_context(history, cv_profile=None):
    cv_profile = cv_profile or {}
    cv_name = (cv_profile.get("name") or "").strip()
    cv_interests = [i for i in cv_profile.get("interests", []) if i]
    cv_categories = [c for c in cv_profile.get("categories", []) if c]
    # short search keywords (fall back to interests for older saved profiles)
    cv_keywords = [k for k in cv_profile.get("keywords", []) if k] \
        or [_shorten_term(i) for i in cv_interests]

    cat_counts = Counter(h["category"] for h in history if h.get("category"))
    kw_counts = Counter(h["keyword"] for h in history if h.get("keyword"))
    hist_categories = [c for c, _ in cat_counts.most_common(3)]
    hist_keywords = [k for k, _ in kw_counts.most_common(3)]

    # CV categories/keywords take priority, then recent search history.
    top_categories = _dedup(cv_categories + hist_categories)[:3] \
        or list(DEFAULT_PROFILE["categories"])
    top_keywords = _dedup(cv_keywords + hist_keywords)[:4] \
        or list(DEFAULT_PROFILE["keywords"])

    if not history and not cv_interests and not cv_name:
        return {
            "summary": ("No searches or CV yet — showing starter suggestions "
                        "aligned with applied AI and NLP."),
            "top_categories": DEFAULT_PROFILE["categories"],
            "top_keywords": DEFAULT_PROFILE["keywords"],
            "n_searches": 0, "name": "",
        }

    parts = []
    if cv_name:
        parts.append(f"User: {cv_name}.")
    if cv_interests:
        parts.append("Research interests (from CV): "
                     + ", ".join(cv_interests) + ".")
    if history:
        parts.append(f"{len(history)} recent searches. Top categories: "
                     + ", ".join(hist_categories) + "."
                     + (f" Keywords: {', '.join(hist_keywords)}."
                        if hist_keywords else ""))
    stats_text = " ".join(parts)

    summary = _chat(
        system=("You summarise a person's research interests in 1-2 short, "
                "natural English sentences, with no preamble like 'Here is'."),
        user=(f"A user's profile:\n{stats_text}\n\n"
              "Write a short (1-2 sentence) summary of what this person seems "
              "to be researching or interested in lately."),
        max_new_tokens=100,
    )
    return {"summary": (summary or "").strip() or stats_text,
            "top_categories": top_categories, "top_keywords": top_keywords,
            "n_searches": len(history), "name": cv_name}


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
        f"User context: {context['summary']}\n"
        f"Most-searched categories: "
        f"{', '.join(context['top_categories']) or 'none'}\n"
        f"Keywords: {', '.join(context['top_keywords']) or 'none'}\n"
        f"Valid arXiv categories: {', '.join(arxiv_api.CATEGORIES.values())}\n\n"
        "Suggest 1 to 3 arXiv searches to find the most relevant papers for "
        "this user this week. Reply ONLY with JSON: a list of objects "
        '{"category": "<valid arXiv code>", "keyword": "<keyword or empty>"}.'
    )
    result = _chat(
        system=("You are a research assistant that decides which arXiv "
                "searches to run to satisfy a user's interests."),
        user=user_prompt, max_new_tokens=120, expect_json=True,
    )
    return _sanitize_queries(result, context)


def fetch_candidates(queries, per_query=8):
    seen = set()
    candidates = []
    for i, q in enumerate(queries):
        if i:
            time.sleep(3)   # stay under arXiv's rate limit between requests
        try:
            docs = arxiv_api.collect_papers(q["category"],
                                            max_results=per_query,
                                            keyword=q["keyword"] or None)
        except Exception:
            # one failing query (e.g. a stubborn 429) shouldn't sink the whole
            # newsletter — skip it and build from whatever else we can fetch
            continue
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
        f"User context: {context['summary']}\n\n"
        f"Candidates (numbered, grouped by sub-topic cluster):\n"
        f"{listing}\n\n"
        f"Pick the {n} most relevant AND diverse ones (try to cover several "
        "clusters, not just the biggest). Reply ONLY with JSON: a list of the "
        "chosen indices, e.g. [0, 3, 7, ...]."
    )
    result = _chat(
        system=("You are a newsletter editor selecting the best scientific "
                "papers for a user, balancing relevance and topic diversity."),
        user=user_prompt, max_new_tokens=100, expect_json=True,
    )
    indices = _sanitize_indices(result, len(candidates), n)
    return [candidates[i] for i in indices]


# ==========================================================================
#  Step 4 — summaries + weekly preview (one batched LLM call)
# ==========================================================================
def _sanitize_summaries(result, selected):
    fallback_preview = ("This week's selection covers the topics you've been "
                        "exploring the most.")
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
        f"User context: {context['summary']}\n\n"
        f"Selected papers ({len(selected)} total):\n{listing}\n\n"
        "For each paper, write a single short, clear, specific one-sentence "
        "summary in English (do not repeat the title). Also write a short "
        "preview paragraph (1-2 sentences) about this week's highlights.\n\n"
        "Reply ONLY with valid JSON. FORMAT example (always invent new "
        "content — never copy this example):\n"
        '{"preview": "This week features new hybrid retrieval methods and '
        'agents for scientific tasks.", "summaries": '
        '["This paper proposes a technique that cuts indexing cost in large '
        'vector databases.", "This work shows that multi-step agents '
        'outperform single-step approaches on complex tasks."]}\n\n'
        f'Your "summaries" list must have exactly {len(selected)} elements — '
        "one per paper, in the same order as the list above."
    )
    result = _chat(
        system=("You are a science newsletter editor who writes short, clear "
                "summaries that stay faithful to the original content, in "
                "English."),
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
        issues.append(f"expected {TOP_N} items, found {len(items)}")
    titles = [it["title"] for it in items]
    if len(set(titles)) != len(titles):
        issues.append("duplicate titles present")
    for it in items:
        summ = (it.get("summary") or "").strip()
        if len(summ) < 15 or summ.lower() in it["title"].lower():
            issues.append(f"weak/repeated summary for: {it['title'][:60]}")

    if issues:
        return {"ok": False, "issues": issues}

    listing = "\n".join(f"- {it['title']}: {it['summary']}" for it in items)
    user_prompt = (
        f"Preview: {draft['preview']}\n\nItems:\n{listing}\n\n"
        "Evaluate this newsletter: are the items relevant and diverse from "
        "each other? Are the summaries specific and useful? Reply ONLY with "
        'JSON. If all good: {"ok": true, "issues": []}. If there are '
        'problems, describe them: {"ok": false, "issues": ["summary 3 is '
        'vague", "item 7 does not seem relevant"]}.'
    )
    result = _chat(
        system="You are a demanding critic of science newsletters.",
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
        return f"{MONTHS[monday.month]} {monday.day}–{sunday.day}"
    return (f"{MONTHS[monday.month]} {monday.day} – "
            f"{MONTHS[sunday.month]} {sunday.day}")


def _build_draft(context, week_key, attempt, progress=None):
    def step(msg):
        if progress:
            progress(msg)

    step("Planning arXiv searches from your context...")
    queries = plan_queries(context)

    step(f"Searching arXiv for candidates ({', '.join(q['category'] for q in queries)})...")
    candidates = fetch_candidates(queries)
    if not candidates:
        queries = [{"category": c, "keyword": ""}
                  for c in DEFAULT_PROFILE["categories"]]
        candidates = fetch_candidates(queries)

    step(f"Selecting the {TOP_N} most relevant and diverse "
        f"({len(candidates)} candidates)...")
    selected = select_top10(candidates, context)

    step("Writing summaries and this week's preview...")
    preview, summaries = summarize_and_preview(selected, context)
    items = [
        {"title": d["title"], "authors": d["authors"][:4], "summary": s,
         "arxiv_id": d["link"].rsplit("/", 1)[-1] if d.get("link") else "",
         "link": d["link"], "pdf_url": d["pdf_url"]}
        for d, s in zip(selected, summaries)
    ]
    return {"week_key": week_key, "week_label": week_range_label(),
           "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
           "interest_summary": context["summary"], "preview": preview,
           "items": items, "queries_used": queries,
           "refine_iterations": attempt}


def generate_weekly_newsletter(force=False, progress=None):
    """progress, if given, is called with a short Portuguese status string
    before each pipeline stage — lets the UI show real step-by-step
    feedback instead of an opaque spinner during the (rare) 1-2 min a full
    generation takes."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    week_key = _iso_week_key()
    cache_path = CACHE_DIR / f"{week_key}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if progress:
        progress("Compacting your context (searches + CV)...")
    context = compact_context(load_recent_history(), load_cv_profile())

    draft = None
    for attempt in range(MAX_REFINE_LOOPS + 1):
        draft = _build_draft(context, week_key, attempt, progress=progress)
        if progress:
            progress("Reviewing the quality of the selection...")
        result = critique(draft)
        if result["ok"]:
            draft["quality_flag"] = None
            break
        draft["quality_flag"] = "; ".join(result["issues"]) \
            or "quality not confirmed"
        if progress and attempt < MAX_REFINE_LOOPS:
            progress("Quality insufficient — refining the selection...")

    cache_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return draft


def peek_current_issue():
    """Read the current week's cached issue without triggering generation —
    used by the sidebar so just viewing it never calls the LLM."""
    cache_path = CACHE_DIR / f"{_iso_week_key()}.json"
    if not cache_path.exists():
        return None
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def list_past_issues(limit=4):
    if not CACHE_DIR.exists():
        return []
    current = _iso_week_key()
    files = sorted(CACHE_DIR.glob("*.json"), reverse=True)
    out = []
    for fp in files:
        if fp.stem == current:
            continue
        try:
            with open(fp, encoding="utf-8") as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
        if len(out) >= limit:
            break
    return out
