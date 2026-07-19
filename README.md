# 🛰️ arXiv Trend Radar

End-to-end NLP application that discovers and interprets the **latest research
trend** in a scientific subject, using **real data** from the **arXiv API**.

## What it does

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

## How to run locally

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

> Requires Python **3.13** (spaCy has no wheels for 3.14 yet). The first run
> downloads the spaCy and transformers models (1–2 min), then caches them.

## Free deploy on Streamlit Community Cloud

1. Push these files to a GitHub repo:
   `app.py`, `arxiv_api.py`, `nlp_pipeline.py`, `requirements.txt`.
2. Go to https://share.streamlit.io → **New app** → point it at the repo.
3. Main file path: `app.py`. Click **Deploy**.
4. First run downloads the models (1–2 min), then caches them.
5. Copy the public link into the slides and the video.

> Alternative: Hugging Face Spaces (SDK = Streamlit) works the same way.

## Structure

| File               | Role                                                |
|--------------------|-----------------------------------------------------|
| `app.py`           | Streamlit UI and orchestration                      |
| `arxiv_api.py`     | Data layer (arXiv API + category taxonomy)          |
| `nlp_pipeline.py`  | NLP pipeline (keywords, topics, NER, summary, report) |
| `requirements.txt` | Dependencies                                         |

## Notes

- No paid API keys: everything runs on open models.
- The LLM component (summary) is **optional** and **complements** the pipeline.
- Simplicity over complexity, as the brief asks.
