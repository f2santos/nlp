# 🖥️ Slide deck — arXiv Trend Radar (max. 10 slides)

Content for the presentation PDF. One `##` block per slide. Keep the actual
slides visual (a headline + 3–5 bullets + one screenshot/chart); the notes
under each are what you *say*, not what you put on the slide.

---

## Slide 1 — Title

**arXiv Trend Radar**
*Discovering and interpreting the latest research trends with NLP*

- Applied NLP — Group Project
- Group members: _[names]_
- 🔗 Live demo: _[Streamlit Cloud link]_ · 🎥 Video: _[link]_

> Notes: one line — "We built an app that reads the latest science and tells
> you what a field is working on right now."

---

## Slide 2 — Problem & context (business relevance)

**Staying on top of a fast-moving field is impossible by hand**

- arXiv publishes **thousands of papers per week**; a single subject (e.g. NLP)
  gets **hundreds of new pre-prints monthly**.
- Researchers, R&D teams, analysts and investors need to know *what's trending*
  — which methods, which sub-topics are heating up — **without reading it all**.
- Manual literature review is slow, subjective, and always out of date.

> Notes: frame the real-world/business angle — competitive intelligence, R&D
> prioritisation, keeping a team current.

---

## Slide 3 — Why NLP

**This is a text-understanding problem at scale**

- The signal lives in **unstructured text** (titles + abstracts).
- NLP lets us **extract, group and summarise** that text automatically:
  keywords, sub-topics, entities, and a readable summary.
- Turns *hundreds of abstracts* → *one interpretable trend report*.

---

## Slide 4 — Data source

**Real data from the arXiv API** (no API key)

- Public arXiv Atom API — `export.arxiv.org/api/query`.
- Query the **latest papers** in a chosen category, sorted by submission date.
- 17 subjects selectable (cs.AI, cs.CL, cs.LG, cs.CV, quant-ph, q-bio, …),
  optional keyword refinement.
- Per paper: title, abstract, authors, date, categories, links.

> Notes: emphasise it's real, live data pulled at run time.

---

## Slide 5 — NLP pipeline overview

**A simple, complete, interpretable pipeline**

`arXiv API → clean → keywords → topic discovery → NER → summary → report`

- Each stage is small and explainable.
- Heavy models are cached; everything runs on **free, open tools**.

> Notes: "simplicity over complexity — a well-executed simple pipeline."

---

## Slide 6 — Methods used

**Standard NLP techniques, correctly applied**

- **Keyword extraction** — TF-IDF (unigrams + bigrams) → trending vocabulary.
- **Topic discovery** — TF-IDF + **KMeans** clustering → sub-topics
  (unsupervised text classification).
- **NER** — spaCy → organisations, people, places, systems.
- **Preprocessing** — cleaning, domain stopwords.

> Notes: mention we *replaced sentiment with topic discovery* on purpose —
> sentiment is meaningless for research abstracts; sub-topics are what trend
> analysis actually needs. (Good interpretation earns marks.)

---

## Slide 7 — Optional LLM component

**Automatic summary with a local language model**

- **distilBART** (Hugging Face) summarises the latest abstracts into a short
  "what the field is working on" paragraph.
- Runs **locally, no API key, free to deploy**.
- **Complements** the pipeline — it does not replace it.

---

## Slide 8 — Key results & interpretation

**Example: Computation & Language (cs.CL), latest 60 papers**

- Top terms: *reasoning, language, ai, multimodal, agents* → the current focus.
- Sub-topics detected (KMeans): *"ai · benchmark · agent"* (largest),
  *"reasoning · distillation"*, *"multimodal"* …
- Summary paragraph gives a one-glance overview of the batch.
- **Interpretation:** the field is currently centred on LLM **reasoning**,
  **agents** and **multimodal** work — visible without reading a single paper.

> Notes: put a real screenshot of the Trend report + Sub-topics tabs here.

---

## Slide 9 — Limitations & improvements

**Honest about scope**

- Limitations: `en_core_web_sm` NER is general-purpose (misses scientific
  entities); abstracts only (no full text); KMeans needs a chosen *k*;
  summariser is extractive-ish and English-only.
- Improvements: scientific NER (SciSpaCy), topic modelling (BERTopic/LDA),
  trend-over-time across months, citation signals, an LLM-API report option.

---

## Slide 10 — Prototype & closing

**A working demo turning live arXiv data into insight**

- 🔗 **Live app:** _[Streamlit Cloud link]_
- 🎥 **Video:** _[link]_
- Stack: Python · Streamlit · scikit-learn · spaCy · Hugging Face
- *Simple, complete, and applied to real data.*

> Notes: close on the same line as the video — "hundreds of abstracts into one
> interpretable trend report." Thank the audience.
