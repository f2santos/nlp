"""
arxiv_api.py
------------
Data acquisition layer: talks to the public arXiv API.

arXiv (https://arxiv.org) is an open repository of scientific pre-prints. Its
free Atom API lets us query the latest papers in a given subject (category),
with no API key required.

We fetch the most recently submitted papers for a chosen category (optionally
narrowed by a keyword) and return them as clean dictionaries the NLP pipeline
can consume.

Docs: https://info.arxiv.org/help/api/user-manual.html
Endpoint: http://export.arxiv.org/api/query
No API key required.
"""

import time
import requests
import feedparser

ENDPOINT = "http://export.arxiv.org/api/query"
# arXiv asks for a descriptive User-Agent and a few seconds between requests.
HEADERS = {"User-Agent": "NLP-Course-Project/1.0 (educational use; arxiv trends)"}

SEARCH_TIMEOUT = 60        # seconds
SEARCH_RETRIES = 3         # attempts on timeout / transient error
RETRY_BACKOFF = 3          # seconds between attempts (also arXiv's rate-limit hint)

# Curated subject taxonomy: friendly label -> arXiv category code. These are the
# most active categories across fields; the app shows the labels in a dropdown.
CATEGORIES = {
    "Artificial Intelligence (cs.AI)": "cs.AI",
    "Computation & Language / NLP (cs.CL)": "cs.CL",
    "Machine Learning (cs.LG)": "cs.LG",
    "Computer Vision (cs.CV)": "cs.CV",
    "Robotics (cs.RO)": "cs.RO",
    "Cryptography & Security (cs.CR)": "cs.CR",
    "Human-Computer Interaction (cs.HC)": "cs.HC",
    "Software Engineering (cs.SE)": "cs.SE",
    "Data Structures & Algorithms (cs.DS)": "cs.DS",
    "Machine Learning - Stats (stat.ML)": "stat.ML",
    "Signal Processing (eess.SP)": "eess.SP",
    "Optimization & Control (math.OC)": "math.OC",
    "Quantum Physics (quant-ph)": "quant-ph",
    "Astrophysics (astro-ph)": "astro-ph",
    "Quantitative Biology (q-bio)": "q-bio",
    "Econometrics (econ.EM)": "econ.EM",
    "Statistical Finance (q-fin.ST)": "q-fin.ST",
}


def _build_query(category, keyword=None):
    """Build the arXiv `search_query` string: a category, optionally AND a
    free-text keyword across all fields."""
    query = f"cat:{category}"
    if keyword and keyword.strip():
        # quote multi-word keywords so arXiv treats them as a phrase
        kw = keyword.strip()
        query += f' AND all:"{kw}"' if " " in kw else f" AND all:{kw}"
    return query


def _request_entries(params):
    """Do one arXiv API request and return its parsed entries, retrying a few
    times on transient network errors (the API occasionally rate-limits or
    times out)."""
    last_err = None
    for attempt in range(1, SEARCH_RETRIES + 1):
        try:
            resp = requests.get(ENDPOINT, params=params, headers=HEADERS,
                                timeout=SEARCH_TIMEOUT)
            resp.raise_for_status()
            return feedparser.parse(resp.content).entries
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as e:
            resp_obj = getattr(e, "response", None)
            status = getattr(resp_obj, "status_code", None)
            # Retry transient failures: network/timeouts, 5xx, and 429 (rate
            # limit). Re-raise genuine client errors (other 4xx).
            if isinstance(e, requests.exceptions.HTTPError) \
                    and status is not None and status < 500 and status != 429:
                raise
            last_err = e
            if attempt < SEARCH_RETRIES:
                wait = RETRY_BACKOFF
                if status == 429:
                    # honour Retry-After if present, else back off harder
                    retry_after = (resp_obj.headers.get("Retry-After")
                                   if resp_obj is not None else None)
                    wait = (int(retry_after)
                            if retry_after and str(retry_after).isdigit()
                            else RETRY_BACKOFF * 3 * attempt)
                time.sleep(wait)
    raise last_err


def search(category, max_results=50, keyword=None):
    """
    Fetch the latest papers in a category (most recently submitted first).
    Returns a list of parsed feed entries (feedparser objects).
    """
    params = {
        "search_query": _build_query(category, keyword),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": min(int(max_results), 200),  # keep requests reasonable
    }
    return _request_entries(params)


def _year_query(category, year, keyword=None):
    """Category query narrowed to a single calendar year via submittedDate."""
    base = _build_query(category, keyword)
    return f"{base} AND submittedDate:[{year}01010000 TO {year}12312359]"


def search_paged(query, total, page_size=100, on_page=None):
    """
    Fetch up to `total` entries for a query, paging through the API in chunks
    (arXiv caps a single response and asks us to page with `start`). We stay
    polite with a short pause between pages. `on_page(fetched, total)` is an
    optional progress callback. Returns the combined entry list.
    """
    total = min(int(total), 1000)
    entries = []
    start = 0
    while start < total:
        n = min(page_size, total - start)
        batch = _request_entries({
            "search_query": query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": start,
            "max_results": n,
        })
        if not batch:                    # no more results for this query
            break
        entries.extend(batch)
        start += len(batch)
        if on_page:
            on_page(len(entries), total)
        if len(batch) < n:               # reached the end of available results
            break
        if start < total:
            time.sleep(RETRY_BACKOFF)     # be polite to the archive
    return entries


def collect_papers_by_year(category, years, max_per_year=200, keyword=None,
                           progress=None):
    """
    Fetch up to `max_per_year` papers for a category in each requested year.
    Returns an ordered dict {year: [doc, ...]} (years ascending). Years with no
    papers (e.g. before the category existed) map to an empty list.
    """
    result = {}
    years = sorted(set(int(y) for y in years))
    for i, yr in enumerate(years):
        if progress:
            progress(i / len(years), f"Fetching {yr}...")
        entries = search_paged(_year_query(category, yr, keyword), max_per_year)
        result[yr] = [_entry_to_doc(e) for e in entries]
        if progress:
            progress((i + 1) / len(years),
                     f"{yr}: {len(result[yr])} papers")
    return result


def _clean(text):
    """Collapse the whitespace/newlines arXiv puts inside titles and abstracts."""
    if not text:
        return ""
    return " ".join(text.split())


def _pdf_link(entry):
    """Find the PDF URL among an entry's links (fall back to the abstract page)."""
    for link in entry.get("links", []):
        if link.get("type") == "application/pdf":
            return link.get("href", "")
    return entry.get("link", "")


def _entry_to_doc(entry):
    """Turn one feed entry into the doc dict the NLP pipeline expects."""
    published = entry.get("published", "")          # e.g. 2024-05-01T12:00:00Z
    date = published[:10] if len(published) >= 10 else ""
    year = published[:4] if len(published) >= 4 else "?"
    authors = [a.get("name", "") for a in entry.get("authors", [])]
    categories = [t.get("term", "") for t in entry.get("tags", [])]
    primary = entry.get("arxiv_primary_category", {}).get("term", "") \
        or (categories[0] if categories else "")

    title = _clean(entry.get("title", "")) or "(no title)"
    abstract = _clean(entry.get("summary", ""))

    return {
        "title": title,
        "abstract": abstract,
        # `text` is what the NLP pipeline reads: title + abstract carry the
        # topical signal we care about for trend analysis.
        "text": f"{title}. {abstract}",
        "authors": authors,
        "published": published,
        "date": date,
        "year": year,
        "categories": categories,
        "primary_category": primary,
        "pdf_url": _pdf_link(entry),
        "link": entry.get("link", ""),
    }


def collect_papers(category, max_results=50, keyword=None, progress=None):
    """
    High-level helper the app calls.

    Returns a list of doc dicts, one per paper:
        { title, abstract, text, authors, published, date, year,
          categories, primary_category, pdf_url, link }

    `progress` is an optional callback(fraction, label) for the Streamlit bar.
    Unlike a web archive, arXiv returns everything in a single response, so the
    slow part is just that one request; we report coarse progress around it.
    """
    if progress:
        progress(0.1, "Querying the arXiv API...")
    entries = search(category, max_results, keyword)
    n = len(entries)
    docs = []
    for i, entry in enumerate(entries):
        docs.append(_entry_to_doc(entry))
        if progress and n:
            progress(0.1 + 0.9 * (i + 1) / n, f"Processing paper {i + 1}/{n}")
    return docs
