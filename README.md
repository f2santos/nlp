# 🛰️ arXiv Trend Radar

End-to-end NLP application that discovers and interprets the **latest research
trend** in a scientific subject, using **real data** from the **arXiv API**.

## What it does

The app has **two modes** (selectable in the sidebar):

- **Latest trend** — analyse the most recent papers in a subject (below).
- **Compare years** — fetch several hundred papers per year for a subject and
  compare selected years (e.g. Machine Learning in 2010 vs 2020) on a
  **radar/spider chart** of NLP-derived metrics: volume, collaboration
  (authors/paper), abstract length, method density, cross-disciplinarity, and
  lexical diversity. Includes per-year keyword tables and a volume chart.

### Latest trend mode

The user picks a subject (an arXiv category, e.g. *Computation & Language /
NLP*), optionally refines it with a keyword, and the app:

1. **Data acquisition** — queries the arXiv API for the most recently submitted
   papers in that category (real data via API, no key required).
2. **Cleaning** — normalises titles and abstracts.
3. **Keywords** — TF-IDF (unigrams + bigrams) over the abstracts to surface the
   trending vocabulary.
4. **Topic discovery** — break the subject into sub-topics ("sub-trends"),
   each labelled by its key terms. Two selectable methods:
   - **KMeans** over TF-IDF — fast, deterministic, great on small batches.
   - **BERTopic** (sentence embeddings + UMAP + HDBSCAN + c-TF-IDF) — more
     semantic, finds its own topic count; best with 150+ papers.
5. **NER** — named-entity recognition with spaCy (organisations, people,
   places...).
6. **Summary (LLM, optional)** — automatic overview of the latest work with
   distilBART (Hugging Face, local, no API key).
7. **Trend report** — a short report combining all of the above into a readable
   summary of what the field is doing right now.

> Note: we replaced *sentiment analysis* (an example technique in the brief)
> with **topic discovery**, because sentiment is not meaningful for research
> abstracts, whereas discovering sub-topics is exactly what "trend analysis"
> needs. Topic discovery is an unsupervised form of text classification.

### Newsletter pessoal

A second area ("MindFeed" landing page → *Newsletter pessoal*) turns what you
search in *Explorar tendências* into a personalised weekly digest, generated
by a small **agentic pipeline** (`newsletter_agent.py`):

1. **Compact context** — aggregates your recent searches (categories/keywords)
   into a short interest profile.
2. **Plan queries** — the local LLM decides 1–3 arXiv searches to run for
   that profile.
3. **Fetch candidates** — via the same `arxiv_api.collect_papers` used by the
   search tool.
4. **Select top 10** — candidates are clustered by sub-topic (reusing
   `nlp_pipeline.discover_topics`), then the LLM picks a relevant *and*
   diverse top 10 across clusters.
5. **Summaries + preview** — one batched LLM call writes a short summary per
   article plus the week's preview blurb.
6. **Quality-control loop** — deterministic checks (10 unique items,
   non-empty summaries) plus one LLM "critic" pass; if either flags issues,
   the pipeline retries up to 2 times before accepting its best draft.

Results are cached per ISO week under `data/newsletter_cache/`, so opening
the page is instant except the first time each week (or after clicking
"Gerar novamente esta semana").

## How to run locally

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python **3.13** (spaCy has no wheels for 3.14 yet). The first run
> downloads the spaCy and transformers models (1–2 min), then caches them.

> The Newsletter's local LLM (`Qwen/Qwen2.5-1.5B-Instruct`, ~3 GB) downloads
> the first time you open *Newsletter pessoal*. Each weekly generation runs
> several sequential local calls (planning, selection, summaries, critique),
> so it can take **1–3 minutes on CPU** — it's cached per week afterwards, so
> this only happens once a week (or when you force a regeneration).

> On Windows PowerShell, `source .venv/bin/activate` doesn't exist — either run
> `.venv\Scripts\Activate.ps1` first, or skip activation and call the venv's
> Python directly: `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Free deploy on Streamlit Community Cloud

1. Push these files to a GitHub repo:
   `app.py`, `arxiv_api.py`, `nlp_pipeline.py`, `requirements.txt`.
2. Go to https://share.streamlit.io → **New app** → point it at the repo.
3. Main file path: `app.py`. Click **Deploy**.
4. First run downloads the models (1–2 min), then caches them.
5. Copy the public link into the slides and the video.

> Alternative: Hugging Face Spaces (SDK = Streamlit) works the same way.

## Structure

| File                  | Role                                                |
|-----------------------|-----------------------------------------------------|
| `app.py`              | Streamlit UI and orchestration                      |
| `arxiv_api.py`        | Data layer (arXiv API + category taxonomy)          |
| `nlp_pipeline.py`     | NLP pipeline (keywords, topics, NER, summary, report) |
| `newsletter_agent.py` | Agentic pipeline behind the personal newsletter (local LLM: context → query planning → selection → summaries → quality-control loop) |
| `requirements.txt`    | Dependencies                                         |
| `data/`               | Local runtime state: search history + weekly newsletter cache. **Not committed** (see `.gitignore`) — it's personal, regenerable, and shouldn't be shared between users. |

## Notes

- No paid API keys: everything runs on open models — including the
  Newsletter's LLM (`Qwen/Qwen2.5-1.5B-Instruct`, local via `transformers`).
  It's a different, larger model than the `distilBART` used for the
  trend-report summary: that one only needs short extractive-style
  summarisation, while the newsletter pipeline needs to follow instructions
  and reliably return structured JSON (query planning, selection, critique).
- The LLM components (trend-report summary, newsletter pipeline) are
  **optional/isolated** and **complement** the deterministic NLP pipeline —
  the app still works with the summary checkbox off, or before any
  newsletter has been generated.
- Simplicity over complexity, as the brief asks.
