"""
app.py
------
Streamlit prototype: "arXiv Trend Radar"

An end-to-end NLP application over REAL data from arXiv. The user picks a
scientific subject (arXiv category); the app fetches the latest papers and runs
a full NLP pipeline to surface and explain the current research trend:

    data acquisition -> cleaning -> keywords -> topic discovery -> NER
                     -> summarization -> trend report

Run locally:   streamlit run app.py
"""

import pandas as pd
import streamlit as st

import arxiv_api
import nlp_pipeline as nlp

st.set_page_config(page_title="arXiv Trend Radar",
                   page_icon="🛰️", layout="wide")

st.title("🛰️ arXiv Trend Radar")
st.caption("Discover and interpret the latest research trends in a subject, "
           "from real arXiv data — with a full NLP pipeline.")

# ---- Sidebar controls -----------------------------------------------------
with st.sidebar:
    st.header("Search parameters")
    subject_label = st.selectbox(
        "Subject (arXiv category)",
        options=list(arxiv_api.CATEGORIES.keys()),
        index=0,
        help="Pick the scientific area whose latest trend you want to analyse.",
    )
    keyword = st.text_input(
        "Refine by keyword (optional)",
        value="",
        help="Narrow the papers within the subject, e.g. 'transformer'. "
             "Leave empty for the whole category.",
    )
    max_items = st.slider("Number of latest papers", 20, 200, 60, step=20)
    n_topics = st.slider("Number of sub-topics to detect", 3, 8, 5)
    topic_method_label = st.radio(
        "Topic discovery method",
        ["KMeans (fast, light)", "BERTopic (semantic, heavier)"],
        index=0,
        help="KMeans is fast and works well on small batches. BERTopic uses "
             "sentence embeddings and shines with more papers (150+); first "
             "run downloads an embedding model.",
    )
    topic_method = "bertopic" if topic_method_label.startswith("BERTopic") \
        else "kmeans"
    run_llm = st.checkbox("Generate summary with LLM", value=True)
    go = st.button("🔍 Analyse trend", type="primary", use_container_width=True)

st.markdown(
    "Pick a **subject**, optionally refine it with a keyword, and hit "
    "**Analyse trend**. The app fetches the latest papers from arXiv and runs "
    "an NLP pipeline to explain what the field is working on right now."
)

# ---- Main flow ------------------------------------------------------------
if go:
    category = arxiv_api.CATEGORIES[subject_label]

    # 1. Data acquisition
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

    # ---- Compute pipeline outputs (cached-model steps run lazily) ----------
    keywords = nlp.extract_keywords(docs, top_n=20)
    with st.spinner(f"Discovering sub-topics ({topic_method})..."):
        topics = nlp.discover_topics(docs, n_topics=n_topics,
                                     method=topic_method)

    # ---- Tabs for each pipeline output ------------------------------------
    tab_report, tab_over, tab_kw, tab_topics, tab_ner, tab_papers = st.tabs(
        ["📈 Trend report", "📊 Overview", "🔑 Keywords",
         "🧩 Sub-topics", "🏷️ Entities", "📄 Papers"]
    )

    # Overview: temporal distribution + category mix
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

    # Keywords
    with tab_kw:
        st.subheader("Trending terms (TF-IDF over the abstracts)")
        if keywords:
            kdf = pd.DataFrame(keywords, columns=["Term", "Score"])
            st.dataframe(kdf, use_container_width=True, hide_index=True)
            st.bar_chart(kdf.set_index("Term")["Score"])
        else:
            st.info("Not enough text to extract keywords.")

    # Sub-topics (clustering)
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

    # Entities (methods/models + named concepts)
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

    # Trend report (headline tab)
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

    # Raw papers
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
else:
    st.info("👈 Choose a subject in the sidebar and hit **Analyse trend**.")
