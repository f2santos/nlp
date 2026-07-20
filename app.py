"""
app.py
------
Streamlit prototype: "arXiv Trend Radar"

An end-to-end NLP application over REAL data from arXiv. Two modes:

  • Latest trend  — pick a subject, fetch the latest papers, run the full NLP
    pipeline (keywords, topic discovery, NER, summary, trend report).
  • Compare years — fetch several hundred papers per year for a subject and
    compare the years on a radar/spider chart of NLP-derived metrics
    (e.g. how Machine Learning looked in 2010 vs 2020).

Run locally:   streamlit run app.py
"""

import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as pgo

import arxiv_api
import nlp_pipeline as nlp

st.set_page_config(page_title="arXiv Trend Radar",
                   page_icon="🛰️", layout="wide")

st.title("🛰️ arXiv Trend Radar")

CURRENT_YEAR = datetime.date.today().year


# ==========================================================================
#  Mode 1 — Latest trend
# ==========================================================================
def render_latest_trend(subject_label, keyword, max_items, n_topics,
                        topic_method, run_llm):
    category = arxiv_api.CATEGORIES[subject_label]

    with st.status(f"Fetching the latest **{subject_label}** papers from "
                   f"arXiv...", expanded=True) as status:
        bar = st.progress(0.0)
        try:
            docs = arxiv_api.collect_papers(
                category, max_results=max_items, keyword=keyword or None,
                progress=lambda f, l: bar.progress(f, text=l),
            )
        except Exception as e:
            st.error(f"Error contacting the arXiv API: {e}")
            st.stop()
        if not docs:
            status.update(label="No results", state="error")
            st.warning("No papers returned. Try another subject, remove the "
                       "keyword, or increase the number of papers.")
            st.stop()
        status.update(label=f"{len(docs)} papers fetched ✔", state="complete")

    st.success(f"Collected **{len(docs)}** of the latest papers in "
               f"*{subject_label}*"
               + (f" matching *{keyword}*." if keyword else "."))

    keywords = nlp.extract_keywords(docs, top_n=20)
    with st.spinner(f"Discovering sub-topics ({topic_method})..."):
        topics = nlp.discover_topics(docs, n_topics=n_topics,
                                     method=topic_method)

    tab_report, tab_over, tab_kw, tab_topics, tab_ner, tab_papers = st.tabs(
        ["📈 Trend report", "📊 Overview", "🔑 Keywords",
         "🧩 Sub-topics", "🏷️ Entities", "📄 Papers"]
    )

    with tab_over:
        st.subheader("How recent are these papers?")
        by_date = pd.Series([d["date"] for d in docs if d["date"]]) \
            .value_counts().sort_index()
        if not by_date.empty:
            st.bar_chart(by_date)
            st.caption("Number of submissions per day in the fetched batch.")
        st.subheader("Category mix")
        cats = pd.Series(
            [c for d in docs for c in d["categories"]]
        ).value_counts().head(12)
        st.bar_chart(cats)
        st.caption("Most common arXiv categories among these papers "
                   "(papers are often cross-listed).")

    with tab_kw:
        st.subheader("Trending terms (TF-IDF over the abstracts)")
        if keywords:
            kdf = pd.DataFrame(keywords, columns=["Term", "Score"])
            st.dataframe(kdf, use_container_width=True, hide_index=True)
            st.bar_chart(kdf.set_index("Term")["Score"])
        else:
            st.info("Not enough text to extract keywords.")

    with tab_topics:
        method_name = "BERTopic" if topic_method == "bertopic" else "KMeans"
        st.subheader(f"Sub-topics within this subject ({method_name})")
        if topics:
            tdf = pd.DataFrame([
                {"Sub-topic": t["label"], "Papers": t["size"],
                 "Share (%)": t["share"], "Key terms": ", ".join(t["terms"])}
                for t in topics
            ])
            st.dataframe(tdf, use_container_width=True, hide_index=True)
            st.bar_chart(tdf.set_index("Sub-topic")["Papers"])
            st.caption("Each sub-topic is a cluster of related papers — a "
                       "'sub-trend' inside the subject.")
        else:
            st.info("Not enough papers to detect distinct sub-topics. "
                    "Increase the number of papers.")

    with tab_ner:
        st.subheader("Methods, models & named concepts")
        with st.spinner("Extracting entities..."):
            ents = nlp.extract_entities(docs)
        labels = {"METHOD": "Methods / models / systems (acronyms)",
                  "CONCEPT": "Named techniques & concepts (spaCy NER)"}
        if not ents:
            st.info("No entities detected.")
        for label in ("METHOD", "CONCEPT"):
            if label in ents:
                st.markdown(f"**{labels.get(label, label)}**")
                edf = pd.DataFrame(ents[label], columns=["Entity", "Mentions"])
                st.dataframe(edf, use_container_width=True, hide_index=True)
        st.caption("Acronyms come from a rule-based matcher (works across all "
                   "arXiv fields); the spelled-out named concepts come from "
                   "spaCy NER. Together they show which techniques are trending.")

    with tab_report:
        st.subheader("Automatic trend report")
        if run_llm:
            with st.spinner("Summarising with distilBART (first run "
                            "downloads the model)..."):
                summary = nlp.summarize(docs)
        else:
            summary = ("LLM summary disabled. Enable it in the sidebar for an "
                       "automatic overview of the latest work.")
        report = nlp.build_report(docs, keywords, topics, summary,
                                  subject_label)
        st.markdown(report)
        st.caption("Report assembled from the NLP pipeline outputs "
                   "(summary via a local Hugging Face model — no API key).")

    with tab_papers:
        st.subheader("Latest papers")
        for d in docs:
            with st.expander(f"[{d['date']}] {d['title'][:100]}"):
                if d["authors"]:
                    st.markdown("*" + ", ".join(d["authors"][:8])
                                + ("..." if len(d["authors"]) > 8 else "") + "*")
                st.write(d["abstract"][:1200] or "(no abstract)")
                links = []
                if d["link"]:
                    links.append(f"[Abstract]({d['link']})")
                if d["pdf_url"]:
                    links.append(f"[PDF]({d['pdf_url']})")
                if links:
                    st.markdown(" · ".join(links))


# ==========================================================================
#  Mode 2 — Compare years (radar / spider chart)
# ==========================================================================
def _radar_figure(profiles):
    """Build a Plotly radar chart: one filled trace per year, each metric
    normalised to [0,1] by its maximum across the compared years so the
    shapes are comparable."""
    metrics = nlp.PROFILE_METRICS
    maxima = {m: max((p[m] for p in profiles.values()), default=0) or 1
              for m in metrics}
    fig = pgo.Figure()
    for year in sorted(profiles):
        p = profiles[year]
        r = [p[m] / maxima[m] for m in metrics]
        fig.add_trace(pgo.Scatterpolar(
            r=r + [r[0]],                       # close the loop
            theta=metrics + [metrics[0]],
            fill="toself", name=str(year),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=520,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    return fig


def render_compare_years(subject_label, keyword, years, max_per_year):
    category = arxiv_api.CATEGORIES[subject_label]
    years = sorted(set(years))

    with st.status(f"Fetching **{subject_label}** papers for "
                   f"{', '.join(map(str, years))}...", expanded=True) as status:
        bar = st.progress(0.0)
        try:
            by_year = arxiv_api.collect_papers_by_year(
                category, years, max_per_year=max_per_year,
                keyword=keyword or None,
                progress=lambda f, l: bar.progress(f, text=l),
            )
        except Exception as e:
            st.error(f"Error contacting the arXiv API: {e}")
            st.stop()
        status.update(label="Fetch complete ✔", state="complete")

    nonempty = {y: d for y, d in by_year.items() if d}
    empty = [y for y, d in by_year.items() if not d]
    if empty:
        st.warning("No papers found for: **"
                   + ", ".join(map(str, empty))
                   + "** — the category likely didn't exist yet that year.")
    if len(nonempty) < 2:
        st.error("Need at least **two** years with papers to compare. "
                 "Pick more recent years or a broader subject.")
        st.stop()

    counts = ", ".join(f"{y}: {len(d)}" for y, d in nonempty.items())
    st.success(f"Comparing *{subject_label}* across "
               f"**{len(nonempty)} years** — {counts}.")

    profiles = {y: nlp.compute_profile(d) for y, d in nonempty.items()}

    tab_radar, tab_table, tab_vol, tab_kw = st.tabs(
        ["🕸️ Radar", "📋 Metrics", "📊 Volume", "🔑 Keywords by year"]
    )

    with tab_radar:
        st.subheader("Multi-metric profile per year")
        st.plotly_chart(_radar_figure(profiles), use_container_width=True)
        st.caption("Each axis is normalised to the strongest year (=1.0), so "
                   "the shapes show *relative* differences. Most axes are "
                   "per-paper averages; 'Papers' is raw volume.")

    with tab_table:
        st.subheader("Raw metric values")
        mdf = pd.DataFrame(profiles).T[nlp.PROFILE_METRICS]
        mdf.index.name = "Year"
        st.dataframe(mdf, use_container_width=True)
        st.caption("The values behind the radar chart, before normalisation.")

    with tab_vol:
        st.subheader("Volume of papers per year")
        vol = pd.Series({y: len(d) for y, d in nonempty.items()}).sort_index()
        st.bar_chart(vol)
        st.caption("How much the subject was published in each selected year "
                   "(capped at the 'papers per year' setting).")

    with tab_kw:
        st.subheader("Top terms in each year")
        cols = st.columns(len(nonempty))
        for col, (y, d) in zip(cols, sorted(nonempty.items())):
            with col:
                st.markdown(f"**{y}**  ·  {len(d)} papers")
                kws = nlp.extract_keywords(d, top_n=12)
                if kws:
                    st.dataframe(
                        pd.DataFrame(kws, columns=["Term", "Score"]),
                        use_container_width=True, hide_index=True)
                else:
                    st.caption("Too few papers for keywords.")
        st.caption("Side-by-side vocabularies make the shift in focus visible "
                   "— e.g. which methods appear or disappear between years.")


# ==========================================================================
#  Sidebar + dispatch
# ==========================================================================
with st.sidebar:
    mode = st.radio("Mode", ["Latest trend", "Compare years"], index=0)
    st.divider()

    if mode == "Latest trend":
        st.header("Latest trend")
        subject_label = st.selectbox(
            "Subject (arXiv category)",
            options=list(arxiv_api.CATEGORIES.keys()), index=0,
            help="Pick the scientific area whose latest trend to analyse.",
        )
        keyword = st.text_input(
            "Refine by keyword (optional)", value="",
            help="Narrow the papers, e.g. 'transformer'. Empty = whole category.",
        )
        max_items = st.slider("Number of latest papers", 20, 200, 60, step=20)
        n_topics = st.slider("Number of sub-topics to detect", 3, 8, 5)
        topic_method_label = st.radio(
            "Topic discovery method",
            ["KMeans (fast, light)", "BERTopic (semantic, heavier)"], index=0,
            help="KMeans is fast and works on small batches. BERTopic uses "
                 "sentence embeddings and shines with 150+ papers.",
        )
        topic_method = "bertopic" if topic_method_label.startswith("BERTopic") \
            else "kmeans"
        run_llm = st.checkbox("Generate summary with LLM", value=True)
        go = st.button("🔍 Analyse trend", type="primary",
                       use_container_width=True)
    else:
        st.header("Compare years")
        subject_label = st.selectbox(
            "Subject (arXiv category)",
            options=list(arxiv_api.CATEGORIES.keys()), index=0,
            help="The area to compare across years.",
        )
        keyword = st.text_input(
            "Refine by keyword (optional)", value="",
            help="Optional keyword filter applied to every year.",
        )
        years = st.multiselect(
            "Years to compare (pick 2–5)",
            options=list(range(CURRENT_YEAR, 1990, -1)),
            default=[2010, 2020],
            help="Older years may be sparse or empty if the category didn't "
                 "exist yet (e.g. cs.LG before ~2007).",
        )
        max_per_year = st.slider("Papers per year", 50, 500, 200, step=50)
        go = st.button("🕸️ Compare", type="primary", use_container_width=True)


if mode == "Latest trend":
    st.caption("Discover and interpret the latest research trend in a subject, "
               "from real arXiv data — with a full NLP pipeline.")
    if go:
        render_latest_trend(subject_label, keyword, max_items, n_topics,
                            topic_method, run_llm)
    else:
        st.info("👈 Choose a subject and hit **Analyse trend**.")
else:
    st.caption("Compare how a subject evolved across years on a radar chart of "
               "NLP-derived metrics — e.g. Machine Learning in 2010 vs 2020.")
    if go:
        if len(years) < 2:
            st.error("Pick at least two years to compare.")
        elif len(years) > 5:
            st.error("Please pick at most five years (each year is a separate "
                     "paged fetch from arXiv).")
        else:
            render_compare_years(subject_label, keyword, years, max_per_year)
    else:
        st.info("👈 Choose a subject and 2–5 years, then hit **Compare**.")
