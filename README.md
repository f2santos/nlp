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

A second area ("MindFeed" landing page → *Newsletter pessoal*) builds a
personalised weekly digest with **Claude** (Anthropic API), through a small
**agentic pipeline** (`newsletter_agent.py`). You choose how each issue is
built:

- **From my profile** — your CV interests plus your recent searches in
  *Explore trends*.
- **Describe what I want** — write what you'd like to read in your own words
  (e.g. *"RAG for medical question answering, evaluation papers, nothing on
  computer vision"*) and Claude does the rest.

Steps:

0. **Profile (optional)** — paste or upload a CV (PDF/TXT) on the left;
   Claude extracts your name, interests and matching arXiv categories
   (`data/cv_profile.json` — not committed). Only the derived profile is
   stored; the CV itself is not.
1. **Understand the request** — Claude turns the profile brief (or your
   description) into a focus, 1–3 arXiv searches and topics to avoid.
2. **Fetch candidates** — via the same `arxiv_api.collect_papers` used by the
   search tool (broadened automatically if too few papers match).
3. **Cluster** — candidates are grouped by sub-topic with
   `nlp_pipeline.discover_topics` (TF-IDF + KMeans).
4. **Select + summarise** — one Claude call picks a relevant *and* diverse
   top 10 across clusters and writes a one-sentence summary per paper plus
   the week's preview.
5. **Quality-control loop** — deterministic checks (10 unique items,
   non-empty summaries); if they fail, the selection is redone once with the
   problems listed.

Every Claude call uses **structured outputs** (the reply is parsed straight
into a pydantic model), so there is no fragile JSON scraping.

After an issue is built you can **↻ Refine** it (*"more on benchmarks, no
surveys"*) or, for a described issue, **💾 Save as my preferences** to add
the interests Claude understood to your profile.

Issues are cached per ISO week under `data/newsletter_cache/`, so opening the
page never calls Claude — only pressing *Generate* or *Refine* does.

**Model & cost.** Claude Haiku 4.5 (`claude-haiku-4-5`) by default — about
1 cent per issue; production uses Claude Sonnet 5 (`claude-sonnet-5`) —
about 3–6 cents per issue. An issue takes ~20–40 s, mostly arXiv's
rate-limit pauses.

**Privacy.** Using the newsletter sends your CV text (when you analyse it),
your profile brief or your description to the Claude API.

## How to run locally

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python **3.13** (spaCy has no wheels for 3.14 yet). The first run
> downloads the spaCy and transformers models (1–2 min), then caches them.

> The newsletter needs an **Anthropic API key**. Create
> `.streamlit/secrets.toml` (git-ignored) with:
>
> ```toml
> ANTHROPIC_API_KEY = "sk-ant-..."
> # MINDFEED_CLAUDE_MODEL = "claude-sonnet-5"   # optional; default claude-haiku-4-5
> ```
>
> Environment variables with the same names work too. Without a key,
> *Explore trends* works as usual and the newsletter page explains how to
> add one.

> On Windows PowerShell, `source .venv/bin/activate` doesn't exist — either run
> `.venv\Scripts\Activate.ps1` first, or skip activation and call the venv's
> Python directly: `.\.venv\Scripts\python.exe -m streamlit run app.py`.

## Free deploy on Streamlit Community Cloud

1. Push these files to a GitHub repo:
   `app.py`, `arxiv_api.py`, `nlp_pipeline.py`, `newsletter_agent.py`,
   `requirements.txt`.
2. Go to https://share.streamlit.io → **New app** → point it at the repo.
3. Main file path: `app.py`. Click **Deploy**.
4. In the app's **Settings → Secrets**, add `ANTHROPIC_API_KEY = "sk-ant-..."`
   and `MINDFEED_CLAUDE_MODEL = "claude-sonnet-5"` (production model).
5. First run downloads the models (1–2 min), then caches them.
6. Copy the public link into the slides and the video.

> Alternative: Hugging Face Spaces (SDK = Streamlit) works the same way.

## Structure

| File                  | Role                                                |
|-----------------------|-----------------------------------------------------|
| `app.py`              | Streamlit UI and orchestration                      |
| `arxiv_api.py`        | Data layer (arXiv API + category taxonomy)          |
| `nlp_pipeline.py`     | NLP pipeline (keywords, topics, NER, summary, report) |
| `newsletter_agent.py` | Agentic pipeline behind the personal newsletter (Claude: CV reading → request → query planning → fetch → clustering → selection + summaries → quality-control loop) |
| `requirements.txt`    | Dependencies                                         |
| `data/`               | Local runtime state: search history, CV-derived profile, weekly newsletter cache. **Not committed** (see `.gitignore`) — it's personal, regenerable, and shouldn't be shared between users. |

## Notes

- *Explore trends* runs entirely on local open models, with no API key
  (TF-IDF, KMeans/BERTopic, spaCy, and `distilBART` for the trend-report
  summary). The **newsletter** uses the Claude API instead: it has to follow
  instructions and return reliable structured output (query planning,
  selection, summaries), which small local models did poorly and slowly on
  CPU.
- The LLM components (trend-report summary, newsletter pipeline) are
  **optional/isolated** and **complement** the deterministic NLP pipeline —
  the app still works with the summary checkbox off, or without an Anthropic
  API key (only the newsletter needs it).
- Simplicity over complexity, as the brief asks.
