# 🎥 Video script (max. 3 minutes)

Goal from the brief: explain the **problem**, the **solution**, and a
**demonstration** of the prototype — focus on clarity, not technical depth.

Suggested split: ~35s problem · ~40s solution · ~95s demo · ~10s close.

---

## [0:00–0:35] Problem & context

> "Every week, thousands of research papers are posted to arXiv. If you work
> in AI — or any fast-moving field — staying current is basically impossible:
> a single subject gets hundreds of new papers a month. Researchers, R&D
> teams, and analysts all face the same question: *what is this field actually
> working on right now?* Reading everything by hand doesn't scale. This is a
> text problem — and that's exactly where NLP helps."

## [0:35–1:15] Solution

> "So we built **arXiv Trend Radar**, a Streamlit app. You pick a subject —
> say, Natural Language Processing — and the app pulls the latest papers
> straight from the arXiv API, then runs a full NLP pipeline:
>
> — it cleans the text;
> — extracts the trending keywords with TF-IDF;
> — groups the papers into sub-topics using clustering — the sub-trends inside
>   the field;
> — identifies named entities with spaCy;
> — and generates an automatic summary with a Hugging Face language model.
>
> Everything runs on free, open tools — no paid API keys. The result is a
> single, readable **trend report**."

## [1:15–2:50] Demonstration (share screen)

Steps to show live, narrating what appears:

1. **Pick a subject** in the sidebar (e.g. *Computation & Language / NLP*),
   set ~60 papers, keep the LLM summary on. Click **Analyse trend**.
   > "I choose the subject, set how many recent papers to pull, and hit
   > Analyse. The app is now fetching the latest papers live from arXiv."

2. **Trend report tab** — show the assembled report first.
   > "Here's the headline output: an automatic report — how many papers, over
   > what dates, a summary of the latest work, the leading terms, and the main
   > sub-topics."

3. **Keywords tab** — show the TF-IDF table/chart.
   > "These are the most distinctive terms right now — the field's current
   > vocabulary."

4. **Sub-topics tab** — show the clusters and their sizes.
   > "The app clustered the papers into sub-trends. You can see which
   > directions are getting the most attention."

5. **Entities tab** — show organisations/people.
   > "NER pulls out the organisations and names mentioned across the abstracts."

6. **Papers tab** — expand one paper.
   > "And everything is backed by the real papers, with links to the abstract
   > and PDF."

## [2:50–3:00] Close

> "A simple but complete NLP pipeline, applied to real, live data — turning
> hundreds of abstracts into one interpretable trend report. Thanks for
> watching."

---

### Recording tips

- Run one analysis **before** recording so the spaCy and distilBART models are
  cached — the demo won't stall on downloads.
- Pick a subject with rich results (test 2–3 first: `cs.CL`, `cs.LG`, `cs.CV`).
- Keep the pace up: point at the key numbers, don't read charts line by line.
- Have the deployed link open in a second tab in case the local run is slow.
