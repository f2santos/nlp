"""
app.py
------
Streamlit app: "MindFeed" — an end-to-end NLP application over REAL data
from arXiv.

The landing page routes to two areas:
  • Search      — the trend-analysis tool (latest trend + compare years):
                  keywords, topic discovery, NER, summary, trend report,
                  radar chart.
  • Newsletter  — placeholder for a personalised digest (coming soon).

Run locally:   streamlit run app.py
"""

import datetime
import textwrap
from collections import Counter

import pandas as pd
import streamlit as st
import plotly.graph_objects as pgo

import arxiv_api
import nlp_pipeline as nlp
import newsletter_agent as na

st.set_page_config(page_title="MindFeed", page_icon="🧠", layout="wide")

CURRENT_YEAR = datetime.date.today().year

if "page" not in st.session_state:
    st.session_state.page = "landing"


# ==========================================================================
#  Global styling & small nav helpers
# ==========================================================================
def inject_css():
    st.markdown(textwrap.dedent("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
      html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
      h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #241F1A; }
      footer { visibility: hidden; }
      div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        transition: transform .15s ease, box-shadow .15s ease !important;
      }
      div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-1px);
      }
      .mf-eyebrow {
        text-align: center; letter-spacing: .18em; text-transform: uppercase;
        font-size: .75rem; font-weight: 600; color: #C9673F; margin-bottom: .75rem;
      }
      .mf-hero-title {
        text-align: center; font-family: 'Fraunces', serif; font-weight: 600;
        font-size: 4.2rem; line-height: 1.05; color: #241F1A; margin: 0;
      }
      .mf-hero-title span { color: #C9673F; }
      .mf-hero-sub {
        text-align: center; font-size: 1.15rem; color: #6B6259; max-width: 620px;
        margin: 1.1rem auto 3rem auto; line-height: 1.6;
      }
      .mf-card {
        background: #FFFFFF; border-radius: 20px; padding: 2.1rem 1.9rem 1.4rem;
        box-shadow: 0 1px 3px rgba(36,31,26,.06), 0 12px 28px rgba(36,31,26,.05);
        border: 1px solid #F0EAE1; margin-bottom: .9rem;
        min-height: 300px; display: flex; flex-direction: column;
        justify-content: flex-start;
      }
      .mf-icon {
        width: 52px; height: 52px; border-radius: 14px; background: #FBEDE4;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem; margin-bottom: 1rem;
      }
      .mf-card h3 { margin: 0 0 .5rem 0; font-size: 1.35rem; }
      .mf-card p { color: #6B6259; font-size: .95rem; line-height: 1.55; min-height: 4.6em; }
      .mf-badge {
        display: inline-block; font-size: .7rem; font-weight: 600; letter-spacing: .06em;
        text-transform: uppercase; background: #F0EAE1; color: #8A7F70;
        padding: .2rem .55rem; border-radius: 999px; margin-bottom: .6rem;
      }
      .mf-badge-accent { background: #C9673F; color: #FFFFFF; }
      .mf-footer { text-align: center; color: #A79C8D; font-size: .82rem; margin-top: 3rem; }
      .mf-rail-card {
        background: #FFFFFF; border-radius: 18px; padding: 1.4rem 1.3rem;
        margin-bottom: 1rem; border: 1px solid #F0EAE1;
        box-shadow: 0 1px 3px rgba(36,31,26,.05);
      }
      .mf-avatar {
        width: 46px; height: 46px; border-radius: 50%; background: #C9673F;
        color: #FFFFFF; display: flex; align-items: center; justify-content: center;
        font-family: 'Fraunces', serif; font-size: 1.2rem; font-weight: 600;
        margin-bottom: .7rem;
      }
      .mf-rail-name { font-family: 'Fraunces', serif; font-size: 1.15rem; font-weight: 600; color: #241F1A; }
      .mf-rail-role { font-size: .82rem; color: #8A7F70; margin-top: .15rem; }
      .mf-rail-title { font-weight: 600; color: #241F1A; margin-top: .3rem; }
      .mf-rail-text { font-size: .85rem; color: #6B6259; line-height: 1.5; margin-top: .4rem; }
      .mf-feed-hero {
        background: linear-gradient(135deg, #FFF7F0, #FDECE3); border-radius: 20px;
        padding: 1.8rem 2rem; border: 1px solid #F3DFCF; margin-bottom: 1.6rem;
      }
      .mf-feed-hero h2 { margin: .5rem 0 .6rem 0; font-size: 1.6rem; }
      .mf-hero-preview { color: #6B6259; font-size: .98rem; line-height: 1.6; margin: 0; }
    </style>
    """), unsafe_allow_html=True)


def hide_sidebar():
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;}</style>",
        unsafe_allow_html=True,
    )


def top_nav():
    if st.button("← MindFeed", type="tertiary", key="top_nav_back"):
        st.session_state.page = "landing"
        st.rerun()


# ==========================================================================
#  Landing page
# ==========================================================================
def render_landing():
    hide_sidebar()
    _, mid, _ = st.columns([1, 3.4, 1])
    with mid:
        st.markdown('<div class="mf-eyebrow">Investigação em tempo real · '
                    'via arXiv</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="mf-hero-title">Mind<span>Feed</span></h1>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="mf-hero-sub">Descobre as tendências de investigação '
            'mais recentes — resumidas, organizadas por tópicos e à tua '
            'medida, direto do arXiv.</p>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown(
                '<div class="mf-card">'
                '<span class="mf-badge" style="visibility:hidden;">Em breve</span>'
                '<div class="mf-icon">🔍</div>'
                '<h3>Explorar tendências</h3>'
                '<p>Pesquisa em tempo real nas últimas publicações do '
                'arXiv, com deteção automática de tópicos, entidades e '
                'resumos gerados por IA.</p>'
                '</div>', unsafe_allow_html=True,
            )
            if st.button("Começar a explorar →", type="primary",
                         use_container_width=True, key="go_search"):
                st.session_state.page = "search"
                st.rerun()
        with col2:
            st.markdown(
                '<div class="mf-card">'
                '<span class="mf-badge">Em breve</span>'
                '<div class="mf-icon">✉️</div>'
                '<h3>Newsletter pessoal</h3>'
                '<p>Um resumo feito à tua medida, com os temas e categorias '
                'que mais te interessam, entregue quando quiseres.</p>'
                '</div>', unsafe_allow_html=True,
            )
            if st.button("Configurar preferências →",
                         use_container_width=True, key="go_newsletter"):
                st.session_state.page = "newsletter"
                st.rerun()

        st.markdown(
            '<div class="mf-footer">Dados reais via arXiv API · '
            'Modelos open-source, sem custos</div>',
            unsafe_allow_html=True,
        )


# ==========================================================================
#  Newsletter — personal dashboard, backed by the newsletter_agent pipeline
# ==========================================================================
def render_profile_rail():
    st.markdown(textwrap.dedent("""
    <div class="mf-rail-card">
      <div class="mf-avatar">P</div>
      <div class="mf-rail-name">Pedro</div>
      <div class="mf-rail-role">Meta AI Engineer · RAG</div>
    </div>
    """), unsafe_allow_html=True)

    history = na.load_recent_history()
    current = na.peek_current_issue()
    if history:
        top_cats = ", ".join(
            c for c, _ in Counter(
                h["category"] for h in history if h.get("category")
            ).most_common(3))
        stats_line = f"Baseado em {len(history)} pesquisas recentes: {top_cats}."
    else:
        stats_line = "Ainda sem pesquisas registadas nesta sessão local."
    summary_line = (current["interest_summary"] if current else
                    "Gera a newsletter para veres aqui o teu contexto "
                    "compactado.")

    st.markdown(textwrap.dedent(f"""
    <div class="mf-rail-card">
      <span class="mf-badge">{'Ativo' if history else 'Em construção'}</span>
      <div class="mf-rail-title">Contexto compactado</div>
      <p class="mf-rail-text">{stats_line}</p>
      <p class="mf-rail-text">{summary_line}</p>
    </div>
    """), unsafe_allow_html=True)


def _generate_with_status(force=False):
    with st.status("A preparar a tua newsletter (LLM local, pode demorar "
                   "um pouco)...", expanded=True) as status:
        issue = na.generate_weekly_newsletter(
            force=force, progress=lambda msg: status.write(msg))
        status.update(label="Newsletter pronta ✔", state="complete")
    return issue


def render_newsletter_feed():
    issue = _generate_with_status()

    st.markdown(textwrap.dedent(f"""
    <div class="mf-feed-hero">
      <span class="mf-badge mf-badge-accent">Esta semana · {issue['week_label']}</span>
      <h2>O que achámos importante para ti</h2>
      <p class="mf-hero-preview">{issue['preview']}</p>
    </div>
    """), unsafe_allow_html=True)

    if issue.get("quality_flag"):
        st.caption(f"⚠️ Nota de qualidade: {issue['quality_flag']}")

    if st.button("🔄 Gerar novamente esta semana"):
        _generate_with_status(force=True)
        st.rerun()

    st.markdown("**🔝 Top 10 para ti**")
    for i, item in enumerate(issue["items"], start=1):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        with c1:
            st.markdown(f"**{i}. {item['title']}**")
            if item.get("authors"):
                st.caption(", ".join(item["authors"]))
            st.caption(item["summary_pt"])
        with c2:
            if item.get("pdf_url"):
                st.link_button("📄 Abrir PDF", item["pdf_url"],
                               use_container_width=True)
            else:
                st.button("📄 Abrir PDF", key=f"pdf_{i}", disabled=True,
                          use_container_width=True)
        st.divider()

    st.markdown("**Edições anteriores**")
    past_issues = na.list_past_issues()
    if not past_issues:
        st.caption("Ainda não há edições anteriores — volta na próxima "
                   "semana.")
    for past in past_issues:
        n_items = len(past.get("items", []))
        with st.expander(f"{past['week_label']} · {n_items} artigos"):
            st.caption(past.get("preview", ""))
            for it in past.get("items", []):
                st.markdown(f"- **{it['title']}** — {it['summary_pt']}")


def render_newsletter():
    hide_sidebar()
    top_nav()
    st.title("📰 MindFeed — Newsletter pessoal")
    st.caption("A tua seleção semanal de investigação em RAG e IA aplicada — "
              "gerada a partir do que pesquisas em Explorar tendências.")

    main_col, side_col = st.columns([2.6, 1], gap="large")
    with side_col:
        render_profile_rail()
    with main_col:
        render_newsletter_feed()


# ==========================================================================
#  Search app — Mode 1: Latest trend
# ==========================================================================
def render_latest_trend(subject_label, keyword, max_items, n_topics,
                        topic_method, run_llm):
    category = arxiv_api.CATEGORIES[subject_label]
    na.record_search(subject_label, category, keyword, mode="latest_trend")

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
#  Search app — Mode 2: Compare years (radar / spider chart)
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
    na.record_search(subject_label, category, keyword, mode="compare_years")
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

    tab_radar, tab_table, tab_vol, tab_evo, tab_kw = st.tabs(
        ["🕸️ Radar", "📋 Metrics", "📊 Volume",
         "📈 Term evolution", "🔑 Keywords by year"]
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

    with tab_evo:
        st.subheader("How the top terms changed across years")
        terms, yrs, matrix = nlp.term_evolution(nonempty, top_n=8, max_terms=20)
        if terms and len(yrs) >= 2:
            # prevalence as % ; rows = terms (most-changing first), cols = years
            prev = pd.DataFrame(matrix, index=terms,
                                columns=[str(y) for y in yrs]) * 100

            heat = pgo.Figure(pgo.Heatmap(
                z=prev.values[::-1],                 # flip so top row is on top
                x=prev.columns.tolist(),
                y=prev.index.tolist()[::-1],
                colorscale="Blues",
                colorbar=dict(title="% of papers"),
                hovertemplate="%{y} — %{x}: %{z:.0f}% of papers<extra></extra>",
            ))
            heat.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10),
                               xaxis_title="Year", yaxis_title="Term")
            st.plotly_chart(heat, use_container_width=True)
            st.caption("Colour = share of that year's papers mentioning the "
                       "term. Terms are ordered by how much they changed "
                       "(biggest movers on top).")

            st.markdown("**Trajectories of the biggest movers**")
            line_df = prev.T.iloc[:, :10]            # years as index, top movers
            line_df.index.name = "Year"
            st.line_chart(line_df)
            st.caption("Each line is a term's prevalence (% of papers) over the "
                       "selected years — rising lines are emerging topics, "
                       "falling lines are fading ones.")
        else:
            st.info("Need at least two years with papers to show term "
                    "evolution.")

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
#  Search app — sidebar + dispatch
# ==========================================================================
def render_search_app():
    top_nav()
    st.title("🔍 MindFeed — Explorar tendências")

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


# ==========================================================================
#  Page dispatch
# ==========================================================================
inject_css()

if st.session_state.page == "landing":
    render_landing()
elif st.session_state.page == "newsletter":
    render_newsletter()
else:
    render_search_app()
