"""
newsletter_agent.py
--------------------
Agentic pipeline behind the "Newsletter pessoal" page: builds a personalised
weekly digest of arXiv papers with Claude (Anthropic API — needs an API key;
the "Explore trends" side of the app keeps running on local open models).

An issue can be built from two kinds of request:
  • "profile"     — a short brief of the CV interests + recent searches
  • "description" — whatever the user typed in "Describe what I want"

Pipeline (deterministic multi-step, with a bounded quality-control loop):

    plan_request -> fetch_candidates -> discover_topics (clusters)
        -> select_and_summarize -> basic checks
        -> (retry the selection, up to MAX_REFINE_LOOPS)

Every Claude call uses structured outputs: the reply is parsed straight into
a pydantic model, so there is no JSON scraping or format-retry logic.

Issues are cached per ISO week in data/newsletter_cache/, so Claude and the
arXiv API are only hit when the user asks for a new issue — not on every page
view.
"""

import datetime
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, ValidationError

import arxiv_api
import nlp_pipeline as nlp

# Claude Haiku 4.5 is cheap and fast for trying things out; production sets
# MINDFEED_CLAUDE_MODEL = "claude-sonnet-5" (read by app.py) to scale up.
DEFAULT_MODEL = "claude-haiku-4-5"
EFFORT = "medium"     # only sent to models that support it (Haiku 4.5 doesn't)
MAX_TOKENS = 16000
MAX_REFINE_LOOPS = 1
TOP_N = 10
MAX_QUERIES = 3
PER_QUERY = 12
MAX_CANDIDATES = 30   # papers shown to Claude when it picks the top 10

DATA_DIR = Path(__file__).parent / "data"
HISTORY_PATH = DATA_DIR / "search_history.jsonl"
CACHE_DIR = DATA_DIR / "newsletter_cache"

MONTHS = ["", "January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Starter interests used when there is no CV or search history yet, so the
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
#  Claude — structured-output calls
# ==========================================================================
class NewsletterError(Exception):
    """A failure worth showing to the user as-is (bad model reply, no papers
    found, empty request...)."""


# Restricting categories to our taxonomy in the schema means Claude can only
# ever answer with codes the arXiv layer supports.
Category = Literal[tuple(arxiv_api.CATEGORIES.values())]


class CvProfile(BaseModel):
    name: str
    interests: list[str]
    categories: list[Category]


class Query(BaseModel):
    category: Category
    keyword: str


class SearchPlan(BaseModel):
    focus: str
    queries: list[Query]
    avoid: list[str]
    interests: list[str]


class Pick(BaseModel):
    index: int
    summary: str


class Selection(BaseModel):
    preview: str
    picks: list[Pick]


def _ask(client, model, system, user, schema):
    """One Claude call whose reply is parsed into the pydantic `schema`."""
    extra = {}
    if not model.startswith("claude-haiku"):     # Haiku 4.5 rejects `effort`
        extra["output_config"] = {"effort": EFFORT}
    try:
        resp = client.messages.parse(
            model=model, max_tokens=MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema, **extra,
        )
    except ValidationError as e:
        # the SDK validates while parsing: a reply that doesn't fit the schema
        # (refused, or cut off at max_tokens) surfaces here
        raise NewsletterError("Claude's answer came back incomplete — please "
                              "try again.") from e
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise NewsletterError("Claude couldn't handle this request — try "
                              "rephrasing it.")
    return resp.parsed_output


def friendly_error(exc):
    """Short, user-facing explanation of a failed Claude/arXiv step."""
    if isinstance(exc, NewsletterError):
        return str(exc)
    if isinstance(exc, anthropic.AuthenticationError):
        return "The Anthropic API key was rejected — check ANTHROPIC_API_KEY."
    if isinstance(exc, (anthropic.PermissionDeniedError,
                        anthropic.NotFoundError)):
        return ("This API key can't use the configured model — check "
                "MINDFEED_CLAUDE_MODEL.")
    if isinstance(exc, anthropic.RateLimitError):
        return "Claude is rate-limiting requests — wait a minute and try again."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Couldn't reach the Claude API — check your internet connection."
    if isinstance(exc, anthropic.APIStatusError):
        return (f"The Claude API returned an error ({exc.status_code}) — try "
                "again shortly.")
    return f"Something went wrong: {exc}"


# ==========================================================================
#  Search history — one of the two sources of the profile brief
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
#  CV profile — the user's name + research interests, read from a CV by Claude
# ==========================================================================
CV_PROFILE_PATH = DATA_DIR / "cv_profile.json"

# Best-effort mapping from free-text interests to the arXiv category codes we
# support — a fallback for when Claude returns no categories for a CV.
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
    """Heuristic used when Claude returns no name: a CV's name is usually the
    first short, capitalised, digit/@-free line at the top."""
    lines = [ln.strip() for ln in cv_text.splitlines()[:15] if ln.strip()]
    for ln in lines:
        words = ln.split()
        if 1 < len(words) <= 4 and "@" not in ln \
                and not any(ch.isdigit() for ch in ln) \
                and sum(w[:1].isupper() for w in words) >= 2:
            return ln
    return lines[0][:40] if lines else ""


def extract_cv_profile(cv_text, *, client, model):
    """
    Extract the person's name, research interests and matching arXiv
    categories from CV text with Claude, with deterministic fallbacks.
    Returns {name, interests, keywords, categories} or None for empty input.
    """
    cv_text = (cv_text or "").strip()
    if not cv_text:
        return None
    result = _ask(
        client, model,
        system="You extract structured information from CVs / resumes.",
        user=(f"<cv>\n{cv_text}\n</cv>\n\n"
              "Extract: (1) the person's full name; (2) 3 to 6 research "
              "interests or technical topics they focus on, as short phrases; "
              "(3) the 1 to 3 arXiv categories that best match those "
              "interests."),
        schema=CvProfile,
    )
    name = result.name.strip() or _guess_name_fallback(cv_text)
    interests = [i.strip() for i in result.interests if i.strip()][:6]
    categories = _dedup(result.categories)[:3] \
        or _map_interests_to_categories(interests)
    # short, searchable keywords derived from the (possibly long) interests
    return {"name": name or "Reader", "interests": interests,
            "keywords": _dedup(_shorten_term(i) for i in interests),
            "categories": categories}


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


def merge_into_profile(interests, categories):
    """Add interests (plus their short keywords) and categories to the saved
    profile, creating one if needed — backs "Save as my preferences". The
    current issue is kept: it was built from these very preferences."""
    profile = load_cv_profile() or {"name": "", "interests": [],
                                    "keywords": [], "categories": []}
    interests = [i.strip() for i in interests if i and i.strip()]
    valid = set(arxiv_api.CATEGORIES.values())
    profile["interests"] = _dedup(profile.get("interests", []) + interests)
    profile["keywords"] = _dedup(profile.get("keywords", [])
                                 + [_shorten_term(i) for i in interests])
    profile["categories"] = _dedup(profile.get("categories", [])
                                   + [c for c in categories if c in valid])
    save_cv_profile(profile)
    return profile


def invalidate_current_week():
    """Drop this week's cached issue so the next view regenerates it (used when
    the CV profile changes, since it feeds the newsletter)."""
    (CACHE_DIR / f"{_iso_week_key()}.json").unlink(missing_ok=True)


# ==========================================================================
#  Step 0 — the request: a profile brief, or the user's own description
# ==========================================================================
def profile_brief(history, cv_profile=None):
    """Plain-text brief of the CV profile + recent searches: what Claude plans
    the issue from in "From my profile" mode."""
    cv_profile = cv_profile or {}
    cv_name = (cv_profile.get("name") or "").strip()
    cv_interests = [i for i in cv_profile.get("interests", []) if i]
    cv_categories = [c for c in cv_profile.get("categories", []) if c]

    if not history and not cv_interests and not cv_name:
        return ("No CV or searches yet — suggest starter picks in applied AI "
                "and NLP (categories "
                + ", ".join(DEFAULT_PROFILE["categories"]) + "; e.g. "
                + DEFAULT_PROFILE["keywords"][0] + ").")

    cat_counts = Counter(h["category"] for h in history if h.get("category"))
    kw_counts = Counter(h["keyword"] for h in history if h.get("keyword"))
    hist_categories = [c for c, _ in cat_counts.most_common(3)]
    hist_keywords = [k for k, _ in kw_counts.most_common(3)]

    parts = []
    if cv_name:
        parts.append(f"Reader: {cv_name}.")
    if cv_interests:
        parts.append("Research interests (from their CV): "
                     + ", ".join(cv_interests) + ".")
    if cv_categories:
        parts.append("CV categories: " + ", ".join(cv_categories) + ".")
    if history:
        parts.append(f"{len(history)} searches in the last 30 days. Most "
                     "searched categories: " + ", ".join(hist_categories) + "."
                     + (f" Keywords: {', '.join(hist_keywords)}."
                        if hist_keywords else ""))
    if cv_interests and history:
        parts.append("Weigh the CV interests above the search history.")
    return " ".join(parts)


def _request_block(request_text, source, adjustments=(), previous_titles=()):
    """The reader's request as Claude sees it in every step, including any
    "Refine" adjustments made after earlier issues."""
    kind = ("a summary of the reader's profile (CV interests and recent "
            "searches)" if source == "profile" else
            "the reader's own description of what they want to read")
    parts = [f"The request below is {kind}.",
             f"<request>\n{request_text}\n</request>"]
    if adjustments:
        parts.append("After seeing an earlier issue, the reader asked for "
                     "these changes (most recent last) — they take priority "
                     "over the request:\n"
                     + "\n".join(f"- {a}" for a in adjustments))
    if previous_titles:
        parts.append("Papers in the previous issue (only repeat one if it is "
                     "still clearly among the best fits):\n"
                     + "\n".join(f"- {t}" for t in previous_titles))
    return "\n\n".join(parts)


# ==========================================================================
#  Step 1 — plan the arXiv searches
# ==========================================================================
def plan_request(request, *, client, model):
    categories = "\n".join(f"- {label}" for label in arxiv_api.CATEGORIES)
    return _ask(
        client, model,
        system=("You plan the arXiv searches for a personalised weekly "
                "research newsletter."),
        user=(f"{request}\n\n"
              "Plan this week's issue:\n"
              "- focus: one or two sentences on what this reader wants to "
              "read about.\n"
              "- queries: 1 to 3 arXiv searches. Each has one category from "
              "the list below and a keyword of 1-3 words, or an empty "
              "keyword to take the newest papers of the whole category. "
              "Keywords are matched against the paper text, multi-word ones "
              "as an exact phrase, so keep them short and common (e.g. "
              "\"retrieval augmented\", not \"retrieval-augmented generation "
              "for clinical question answering\").\n"
              "- avoid: topics the reader said they don't want (empty if "
              "none).\n"
              "- interests: 3 to 6 short phrases describing the reader's "
              "interests, suitable for saving in their profile.\n\n"
              f"arXiv categories:\n{categories}"),
        schema=SearchPlan,
    )


def _clean_queries(plan):
    queries, seen = [], set()
    for q in plan.queries:
        keyword = " ".join(q.keyword.split()[:3])
        if (q.category, keyword) not in seen:
            seen.add((q.category, keyword))
            queries.append({"category": q.category, "keyword": keyword})
    return queries[:MAX_QUERIES] or [
        {"category": c, "keyword": ""} for c in DEFAULT_PROFILE["categories"]]


# ==========================================================================
#  Step 2 — fetch candidates from arXiv
# ==========================================================================
def fetch_candidates(queries, per_query=PER_QUERY):
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


def _merge_candidates(first, more):
    seen = {d.get("link") or d.get("title") for d in first}
    extra = [d for d in more if (d.get("link") or d.get("title")) not in seen]
    return (first + extra)[:MAX_CANDIDATES]


# ==========================================================================
#  Step 3 — pick the top 10 + write the summaries (one Claude call)
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


def _fallback_summary(doc):
    return doc["abstract"][:180].rsplit(" ", 1)[0] + "…"


def select_and_summarize(request, plan, candidates, cluster_labels, n, *,
                         client, model, issues=None):
    def excerpt(text, limit=500):
        return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"

    listing = "\n".join(
        f"{i}. [cluster: {cluster_labels.get(d.get('topic', -1), 'misc')}] "
        f"{d['title']} — {excerpt(d['abstract'])}"
        for i, d in enumerate(candidates)
    )
    retry_note = ("\n\nYour previous selection had these problems — fix "
                  "them:\n" + "\n".join(f"- {x}" for x in issues)
                  if issues else "")
    return _ask(
        client, model,
        system=("You are the editor of a personalised research newsletter. "
                "Pick papers faithfully to the reader's request and summarise "
                "them accurately — never claim anything the abstract doesn't "
                "say."),
        user=(f"{request}\n\n"
              f"What the reader wants: {plan.focus}\n"
              f"Topics to avoid: {', '.join(plan.avoid) or 'none'}\n\n"
              "Candidate papers (numbered; papers in the same cluster share "
              f"a sub-topic):\n<candidates>\n{listing}\n</candidates>\n\n"
              f"Choose the {n} papers that best fit the request: relevance "
              "first, then variety across clusters. Skip papers on the avoid "
              "list. For each pick give its index and a one-sentence summary "
              "in English of what the paper contributes — specific, faithful "
              "to the abstract, without repeating the title. Also write a 1-2 "
              "sentence preview of this issue's highlights." + retry_note),
        schema=Selection,
    )


# ==========================================================================
#  Step 4 — quality control (deterministic checks)
# ==========================================================================
def _basic_issues(items, expected):
    issues = []
    if len(items) != expected:
        issues.append(f"expected {expected} items, found {len(items)}")
    titles = [it["title"] for it in items]
    if len(set(titles)) != len(titles):
        issues.append("duplicate titles present")
    for it in items:
        summ = (it.get("summary") or "").strip()
        if len(summ) < 15 or summ.lower() in it["title"].lower():
            issues.append(f"weak/repeated summary for: {it['title'][:60]}")
    return issues


def _make_items(selected, summaries):
    return [
        {"title": d["title"], "authors": d["authors"][:4], "summary": s,
         "arxiv_id": d["link"].rsplit("/", 1)[-1] if d.get("link") else "",
         "link": d["link"], "pdf_url": d["pdf_url"]}
        for d, s in zip(selected, summaries)
    ]


# ==========================================================================
#  Orchestrator — build (or refine) this week's issue
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


def _save_issue(issue):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{issue['week_key']}.json").write_text(
        json.dumps(issue, ensure_ascii=False, indent=2), encoding="utf-8")


def build_issue(request_text, *, source, client, model, adjustments=(),
                previous_titles=(), progress=None):
    """
    Build this week's issue with Claude and cache it. `source` is "profile"
    (request_text = profile_brief(...)) or "description" (the user's own
    words). `progress`, if given, is called with a short status string before
    each stage so the UI can show step-by-step feedback.
    """
    def step(msg):
        if progress:
            progress(msg)

    request_text = (request_text or "").strip()
    if not request_text:
        raise NewsletterError("Describe what you'd like to read first.")
    adjustments = list(adjustments)
    request = _request_block(request_text, source, adjustments,
                             previous_titles)

    step("Understanding your request...")
    plan = plan_request(request, client=client, model=model)
    queries = _clean_queries(plan)

    step("Searching arXiv ("
         + ", ".join(f"{q['category']}" + (f" · {q['keyword']}"
                                            if q["keyword"] else "")
                     for q in queries) + ")...")
    candidates = fetch_candidates(queries)
    if len(candidates) < TOP_N:
        step("Few matches — broadening the search...")
        time.sleep(3)   # arXiv rate limit
        broader = [{"category": c, "keyword": ""}
                   for c in _dedup(q["category"] for q in queries)]
        candidates = _merge_candidates(candidates, fetch_candidates(broader))
    if not candidates:
        raise NewsletterError("arXiv returned no papers — it may be "
                              "rate-limiting requests. Wait a minute and try "
                              "again.")

    step(f"Grouping {len(candidates)} candidates by sub-topic...")
    try:
        topics = nlp.discover_topics(
            candidates, n_topics=min(5, max(2, len(candidates) // 6)),
            method="kmeans")
    except Exception:
        topics = []
    cluster_labels = {t["id"]: t["label"] for t in topics}

    n = min(TOP_N, len(candidates))
    issues = []
    for attempt in range(MAX_REFINE_LOOPS + 1):
        step("Picking the best papers and writing summaries..." if not attempt
             else "Quality check failed — refining the selection...")
        selection = select_and_summarize(
            request, plan, candidates, cluster_labels, n,
            client=client, model=model, issues=issues)
        indices = _sanitize_indices([p.index for p in selection.picks],
                                    len(candidates), n)
        by_index = {p.index: p.summary.strip() for p in selection.picks}
        selected = [candidates[i] for i in indices]
        items = _make_items(selected, [by_index.get(i)
                                       or _fallback_summary(candidates[i])
                                       for i in indices])
        issues = _basic_issues(items, n)
        if not issues:
            break

    issue = {
        "week_key": _iso_week_key(), "week_label": week_range_label(),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "interest_summary": plan.focus.strip() or request_text[:200],
        "preview": selection.preview.strip()
        or "This week's selection, picked for you.",
        "items": items, "queries_used": queries, "refine_iterations": attempt,
        "quality_flag": "; ".join(issues) or None,
        "source": source, "request": request_text, "adjustments": adjustments,
        "engine": model,
        "plan_interests": _dedup(i.strip() for i in plan.interests)[:6],
        "plan_categories": _dedup(q["category"] for q in queries),
    }
    _save_issue(issue)
    return issue


def refine_issue(issue, adjustment, *, client, model, progress=None):
    """Rebuild an issue with one more user adjustment ("more benchmarks, no
    surveys"), keeping its original request and previous picks in view."""
    adjustment = (adjustment or "").strip()
    if not adjustment:
        raise NewsletterError("Say what you'd like to change first.")
    return build_issue(
        issue["request"], source=issue.get("source", "description"),
        adjustments=list(issue.get("adjustments", [])) + [adjustment],
        previous_titles=[it["title"] for it in issue.get("items", [])],
        client=client, model=model, progress=progress)


def peek_current_issue():
    """Read the current week's cached issue without triggering generation —
    used by the page so just viewing it never calls Claude."""
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
