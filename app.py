"""
app.py
------
Streamlit app: "MindFeed" — an end-to-end NLP application over REAL data
from arXiv.

The landing page routes to two areas:
  • Search      — the trend-analysis tool (latest trend + compare years):
                  keywords, topic discovery, NER, summary, trend report,
                  radar chart.
  • Newsletter  — a personalised weekly digest built by Claude, either from
                  the user's profile (CV + searches) or from a description.

Run locally:   streamlit run app.py
"""

import datetime
import html
import io
import os
import textwrap
from collections import Counter

import anthropic
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
      :root {
        --ink: #1A1712; --muted: #6B6259; --faint: #A79C8D;
        --accent: #C9673F; --accent-dark: #A84E2C; --accent-soft: #FBEDE4;
        --line: #ECE3D6; --card: #FFFFFF;
      }
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--ink); }
      footer { visibility: hidden; }
      .stApp {
        background:
          radial-gradient(1200px 500px at 50% -8%, #FDF1E7 0%, rgba(253,241,231,0) 60%),
          linear-gradient(180deg, #FDFBF7 0%, #F7F2EA 100%);
      }
      div.stButton > button, div.stFormSubmitButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid var(--line) !important;
        transition: transform .15s ease, box-shadow .15s ease,
                    background .15s ease, border-color .15s ease !important;
      }
      div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(36,31,26,.10) !important;
      }
      div.stButton > button[kind="primary"] {
        background: var(--accent) !important; color: #fff !important;
        border-color: var(--accent) !important;
      }
      div.stButton > button[kind="primary"]:hover {
        background: var(--accent-dark) !important; border-color: var(--accent-dark) !important;
      }
      div.stButton > button[kind="secondary"] {
        background: #fff !important; color: var(--ink) !important;
      }
      div.stButton > button[kind="secondary"]:hover { border-color: var(--accent) !important; }
      .mf-nav {
        display: flex; align-items: center; justify-content: space-between;
        max-width: 1080px; margin: .2rem auto 2.6rem auto; padding: 0 .4rem;
      }
      .mf-nav-brand { font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.3rem; color: var(--ink); }
      .mf-nav-brand span { color: var(--accent); }
      .mf-nav-tag { font-size: .85rem; color: var(--muted); font-weight: 500; }
      .mf-eyebrow {
        display: inline-block; letter-spacing: .16em; text-transform: uppercase;
        font-size: .72rem; font-weight: 700; color: var(--accent);
        background: var(--accent-soft); border: 1px solid #F3DBC9;
        padding: .34rem .8rem; border-radius: 999px; margin: 0 auto .95rem auto;
      }
      .mf-hero-wrap { text-align: center; }
      .mf-hero-title {
        font-family: 'Fraunces', serif; font-weight: 600;
        font-size: 4.6rem; line-height: 1.03; color: var(--ink); margin: 0;
        letter-spacing: -.02em;
      }
      .mf-hero-title span { color: var(--accent); }
      .mf-hero-sub {
        font-size: 1.16rem; color: var(--muted); max-width: 600px;
        margin: 1.15rem auto 2.8rem auto; line-height: 1.65;
      }
      .mf-card {
        background: var(--card); border-radius: 18px; padding: 2rem 1.85rem 1.5rem;
        box-shadow: 0 1px 2px rgba(36,31,26,.05), 0 14px 34px rgba(36,31,26,.06);
        border: 1px solid var(--line); margin-bottom: .85rem;
        height: 300px; display: flex; flex-direction: column;
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
      }
      .mf-card:hover {
        transform: translateY(-4px); border-color: #E6D3C0;
        box-shadow: 0 6px 14px rgba(36,31,26,.07), 0 22px 48px rgba(36,31,26,.10);
      }
      .mf-icon {
        width: 54px; height: 54px; border-radius: 15px; background: var(--accent-soft);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.55rem; margin-bottom: 1.1rem;
      }
      .mf-card h3 { margin: 0 0 .55rem 0; font-size: 1.4rem; }
      .mf-card p { color: var(--muted); font-size: .96rem; line-height: 1.58; margin: 0; }
      .mf-badge {
        display: inline-block; font-size: .7rem; font-weight: 600; letter-spacing: .06em;
        text-transform: uppercase; background: #F0EAE1; color: #8A7F70;
        padding: .2rem .55rem; border-radius: 999px; margin-bottom: .6rem;
      }
      .mf-badge-accent { background: var(--accent); color: #fff; }
      .mf-strip {
        display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center;
        margin: 3rem auto 1rem auto; max-width: 900px;
      }
      .mf-chip {
        display: flex; align-items: center; gap: .55rem;
        background: rgba(255,255,255,.7); border: 1px solid var(--line);
        border-radius: 999px; padding: .55rem 1.05rem;
        font-size: .86rem; font-weight: 500; color: #4E463D;
      }
      .mf-chip span { font-size: 1.05rem; }
      .mf-footer { text-align: center; color: var(--faint); font-size: .82rem;
        margin-top: 2.6rem; padding-top: 1.4rem; border-top: 1px solid var(--line); }
      .mf-rail-card {
        background: var(--card); border-radius: 16px; padding: 1.4rem 1.3rem;
        margin-bottom: 1rem; border: 1px solid var(--line);
        box-shadow: 0 1px 3px rgba(36,31,26,.05);
      }
      .mf-avatar {
        width: 46px; height: 46px; border-radius: 50%; background: var(--accent);
        color: #fff; display: flex; align-items: center; justify-content: center;
        font-family: 'Fraunces', serif; font-size: 1.2rem; font-weight: 600;
        margin-bottom: .7rem;
      }
      .mf-rail-name { font-family: 'Fraunces', serif; font-size: 1.15rem; font-weight: 600; color: var(--ink); }
      .mf-rail-role { font-size: .82rem; color: #8A7F70; margin-top: .15rem; }
      .mf-rail-title { font-weight: 600; color: var(--ink); margin-top: .3rem; }
      .mf-rail-text { font-size: .85rem; color: var(--muted); line-height: 1.5; margin-top: .4rem; }
      .mf-feed-hero {
        background: linear-gradient(135deg, #FFF7F0, #FDECE3); border-radius: 20px;
        padding: 1.8rem 2rem; border: 1px solid #F3DFCF; margin-bottom: 1.6rem;
      }
      .mf-feed-hero h2 { margin: .5rem 0 .6rem 0; font-size: 1.6rem; }
      .mf-hero-preview { color: var(--muted); font-size: .98rem; line-height: 1.6; margin: 0; }
      .mf-hero-source { color: #8A7F70; font-size: .84rem; margin: .8rem 0 0 0; }
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

    st.markdown(textwrap.dedent("""
    <div class="mf-nav">
      <div class="mf-nav-brand">🧠 Mind<span>Feed</span></div>
      <div class="mf-nav-tag">Research trend intelligence · powered by arXiv</div>
    </div>
    """), unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 3.4, 1])
    with mid:
        st.markdown(textwrap.dedent("""
        <div class="mf-hero-wrap">
          <span class="mf-eyebrow">Real-time research · open models</span>
          <h1 class="mf-hero-title">Mind<span>Feed</span></h1>
          <p class="mf-hero-sub">Turn thousands of new papers into clear,
          personalised insight. MindFeed reads the latest arXiv research,
          detects the trends, and tailors them to you.</p>
        </div>
        """), unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown(textwrap.dedent("""
            <div class="mf-card">
              <div class="mf-icon">🔍</div>
              <h3>Explore trends</h3>
              <p>Real-time search over the latest arXiv papers, with automatic
              topic discovery, method &amp; concept extraction, and
              AI-generated summaries.</p>
            </div>
            """), unsafe_allow_html=True)
            if st.button("Start exploring →", type="primary",
                         use_container_width=True, key="go_search"):
                st.session_state.page = "search"
                st.rerun()
        with col2:
            st.markdown(textwrap.dedent("""
            <div class="mf-card">
              <div class="mf-icon">✉️</div>
              <h3>Personal newsletter</h3>
              <p>A weekly digest tailored to you — built from your CV and
              searches, or from whatever you describe — curated and summarised
              by Claude.</p>
            </div>
            """), unsafe_allow_html=True)
            if st.button("Set up my feed →", type="primary",
                         use_container_width=True, key="go_newsletter"):
                st.session_state.page = "newsletter"
                st.rerun()

        st.markdown(textwrap.dedent("""
        <div class="mf-strip">
          <div class="mf-chip"><span>📡</span> Real arXiv data</div>
          <div class="mf-chip"><span>🧩</span> Full NLP pipeline</div>
          <div class="mf-chip"><span>🔒</span> Local NLP models</div>
          <div class="mf-chip"><span>✨</span> Personalised by Claude</div>
        </div>
        <div class="mf-footer">Real data via the arXiv API · Open-source NLP
        models · Newsletter by Claude · Built for the Applied NLP project</div>
        """), unsafe_allow_html=True)


# ==========================================================================
#  Newsletter — personal dashboard, backed by the newsletter_agent pipeline
# ==========================================================================
MODEL_NAMES = {"claude-haiku-4-5": "Claude Haiku 4.5",
               "claude-sonnet-5": "Claude Sonnet 5"}


def _secret(name):
    """A setting from .streamlit/secrets.toml, falling back to the
    environment (st.secrets raises when there is no secrets file)."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.environ.get(name)


@st.cache_resource
def _claude_client(api_key):
    return anthropic.Anthropic(api_key=api_key, timeout=120.0)


def _claude():
    """{"client", "model"} for the newsletter's Claude calls, or None when no
    API key is configured (the newsletter then can't generate)."""
    api_key = _secret("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return {"client": _claude_client(api_key),
            "model": _secret("MINDFEED_CLAUDE_MODEL") or na.DEFAULT_MODEL}


def _model_name(model):
    return MODEL_NAMES.get(model, model)


def _read_cv_upload(uploaded):
    """Return plain text from an uploaded CV file (.txt / .md / .pdf)."""
    if uploaded is None:
        return ""
    data = uploaded.read()
    if uploaded.name.lower().endswith(".pdf"):
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return ""
    return data.decode("utf-8", errors="ignore")


def render_cv_form(profile, llm):
    """CV input area: paste text or upload a file; Claude extracts name +
    interests (which drive the personalised newsletter)."""
    with st.expander("📄 " + ("Update my CV" if profile else "Add my CV"),
                     expanded=not profile):
        cv_text = st.text_area(
            "Paste your CV text", height=150, key="cv_text",
            placeholder="Paste your CV / resume here...",
        )
        uploaded = st.file_uploader("…or upload a file (PDF / TXT)",
                                    type=["pdf", "txt", "md"], key="cv_file")
        c1, c2 = st.columns(2)
        analyse = c1.button("Analyse CV", type="primary", disabled=not llm,
                            use_container_width=True, key="cv_analyse")
        if profile and c2.button("Remove", use_container_width=True,
                                 key="cv_clear"):
            na.clear_cv_profile()
            na.invalidate_current_week()
            st.rerun()

        if analyse:
            text = cv_text.strip() or _read_cv_upload(uploaded)
            if not text.strip():
                st.warning("Paste some CV text or upload a file first.")
            else:
                with st.spinner("Reading your CV with Claude..."):
                    try:
                        prof = na.extract_cv_profile(text, **llm)
                    except Exception as e:
                        st.error(na.friendly_error(e))
                        return
                na.save_cv_profile(prof)
                na.invalidate_current_week()   # newsletter depends on interests
                st.success(f"Got it, {prof['name']}! Interests: "
                           + ", ".join(prof["interests"] or ["—"]))
                st.rerun()


def render_profile_rail(llm):
    profile = na.load_cv_profile()
    name = (profile or {}).get("name") or "Guest"
    initial = (name.strip()[:1] or "?").upper()
    interests = (profile or {}).get("interests", [])
    role = ", ".join(interests[:2]) if interests else "No CV added yet"

    st.markdown(textwrap.dedent(f"""
    <div class="mf-rail-card">
      <div class="mf-avatar">{html.escape(initial)}</div>
      <div class="mf-rail-name">{html.escape(name)}</div>
      <div class="mf-rail-role">{html.escape(role)}</div>
    </div>
    """), unsafe_allow_html=True)

    render_cv_form(profile, llm)

    history = na.load_recent_history()
    current = na.peek_current_issue()

    interests_line = ("Interests from your CV: " + ", ".join(interests) + "."
                      if interests else
                      "Add your CV to personalise the newsletter.")
    if history:
        top_cats = ", ".join(
            c for c, _ in Counter(
                h["category"] for h in history if h.get("category")
            ).most_common(3))
        stats_line = f"Based on {len(history)} recent searches: {top_cats}."
    else:
        stats_line = "No searches logged in this local session yet."
    summary_line = (current["interest_summary"] if current else
                    "Generate an issue to see what it focuses on here.")

    badge = "Active" if (interests or history) else "Getting started"
    st.markdown(textwrap.dedent(f"""
    <div class="mf-rail-card">
      <span class="mf-badge">{badge}</span>
      <div class="mf-rail-title">Your profile</div>
      <p class="mf-rail-text">{html.escape(interests_line)}</p>
      <p class="mf-rail-text">{html.escape(stats_line)}</p>
      <p class="mf-rail-text">{html.escape(summary_line)}</p>
    </div>
    """), unsafe_allow_html=True)


def _generate_with_status(llm, build):
    """Run one newsletter build — `build(progress)` — inside a status box that
    shows the pipeline's steps. Returns the issue, or None on failure."""
    with st.status(f"Preparing your newsletter with "
                   f"{_model_name(llm['model'])}...", expanded=True) as status:
        try:
            issue = build(lambda msg: status.write(msg))
        except Exception as e:
            status.update(label="Generation failed", state="error")
            st.error(f"Couldn't build the newsletter: {na.friendly_error(e)}")
            return None
        status.update(label="Newsletter ready ✔", state="complete")
    return issue


def render_generator(llm, profile, has_issue):
    """Build a new issue — from the saved profile, or from what the user
    describes. Both run the Claude pipeline in newsletter_agent."""
    box = (st.expander("✨ Make a new issue") if has_issue
           else st.container(border=True))
    with box:
        mode = st.radio("Build it from",
                        ["From my profile", "Describe what I want"],
                        horizontal=True, key="nl_mode",
                        label_visibility="collapsed")
        if mode == "From my profile":
            interests = (profile or {}).get("interests", [])
            st.caption("Uses the interests from your CV ("
                       + ", ".join(interests) + ") and your recent searches."
                       if interests else
                       "Uses your recent searches — add your CV on the left "
                       "for more personal picks.")
            if st.button("✨ Generate from my profile", type="primary",
                         key="nl_gen_profile"):
                brief = na.profile_brief(na.load_recent_history(), profile)
                if _generate_with_status(llm, lambda p: na.build_issue(
                        brief, source="profile", progress=p, **llm)):
                    st.rerun()
        else:
            request = st.text_area(
                "What do you want to read about?", key="nl_request",
                height=110,
                placeholder=("e.g. Retrieval-augmented generation for medical "
                             "question answering — especially evaluation and "
                             "benchmarks. Nothing on computer vision."))
            if st.button("✨ Generate from my description", type="primary",
                         key="nl_gen_description"):
                if not request.strip():
                    st.warning("Describe what you'd like to read first.")
                elif _generate_with_status(llm, lambda p: na.build_issue(
                        request, source="description", progress=p, **llm)):
                    st.rerun()
        st.caption(f"Built by {_model_name(llm['model'])} from the latest "
                   "arXiv papers — takes about 20–40 s.")


def _source_line(issue):
    """How an issue was made, shown under its preview ("" for issues made
    before the Claude pipeline existed)."""
    if issue.get("source") == "description":
        req = " ".join(issue.get("request", "").split())
        if len(req) > 140:
            req = req[:140].rsplit(" ", 1)[0] + "…"
        line = f"From your request: “{req}”"
    elif issue.get("source") == "profile":
        line = "From your profile"
    else:
        return ""
    if issue.get("adjustments"):
        line += f" · refined: “{issue['adjustments'][-1]}”"
    return line + f" · by {_model_name(issue.get('engine', ''))}"


def _render_issue(issue, llm, profile):
    """Display an already-generated newsletter issue (no generation here),
    plus the Save / Refine actions."""
    preview = html.escape(" ".join(issue["preview"].split()))
    source = _source_line(issue)
    source_html = (f'<p class="mf-hero-source">{html.escape(source)}</p>'
                   if source else "")
    st.markdown(textwrap.dedent(f"""
    <div class="mf-feed-hero">
      <span class="mf-badge mf-badge-accent">This week · {issue['week_label']}</span>
      <h2>What we picked for you</h2>
      <p class="mf-hero-preview">{preview}</p>{source_html}
    </div>
    """), unsafe_allow_html=True)

    if issue.get("quality_flag"):
        st.caption(f"⚠️ Quality note: {issue['quality_flag']}")

    if issue.get("source") == "description" and issue.get("plan_interests"):
        saved = set((profile or {}).get("interests", []))
        new = [i for i in issue["plan_interests"] if i not in saved]
        if not new:
            st.caption("✔ The interests behind this issue are saved in your "
                       "profile.")
        elif st.button("💾 Save as my preferences", key="nl_save_prefs",
                       help="Adds to your profile: " + ", ".join(new)):
            na.merge_into_profile(new, issue.get("plan_categories", []))
            st.session_state.nl_flash = ("Saved to your profile: "
                                         + ", ".join(new) + ".")
            st.rerun()

    if llm and issue.get("request"):
        with st.form("nl_refine", border=False):
            c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
            adjustment = c1.text_input(
                "Adjust this issue",
                placeholder="e.g. more on benchmarks, no survey papers")
            refine = c2.form_submit_button("↻ Refine",
                                           use_container_width=True)
        if refine:
            if not adjustment.strip():
                st.warning("Say what you'd like to change first.")
            elif _generate_with_status(llm, lambda p: na.refine_issue(
                    issue, adjustment, progress=p, **llm)):
                st.rerun()

    st.markdown(f"**🔝 Top {len(issue['items'])} for you**")
    for i, item in enumerate(issue["items"], start=1):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        with c1:
            st.markdown(f"**{i}. {item['title']}**")
            if item.get("authors"):
                st.caption(", ".join(item["authors"]))
            st.caption(item.get("summary", ""))
        with c2:
            if item.get("pdf_url"):
                st.link_button("📄 Open PDF", item["pdf_url"],
                               use_container_width=True)
            else:
                st.button("📄 Open PDF", key=f"pdf_{i}", disabled=True,
                          use_container_width=True)
        st.divider()


def _render_past_issues():
    st.markdown("**Past issues**")
    past_issues = na.list_past_issues()
    if not past_issues:
        st.caption("No past issues yet — check back next week.")
    for past in past_issues:
        n_items = len(past.get("items", []))
        with st.expander(f"{past['week_label']} · {n_items} papers"):
            st.caption(past.get("preview", ""))
            for it in past.get("items", []):
                st.markdown(f"- **{it['title']}** — {it.get('summary', '')}")


def render_newsletter_main(llm):
    """Right-hand area. Never generates on load — waits for the user to pick
    how to build the issue and press generate. A cached issue is shown."""
    profile = na.load_cv_profile()
    current = na.peek_current_issue()          # reads cache, no generation

    flash = st.session_state.pop("nl_flash", None)
    if flash:
        st.success(flash)

    if llm is None:
        st.info("✉️ The newsletter is written by Claude, so it needs an "
                "Anthropic API key. Add `ANTHROPIC_API_KEY = \"sk-ant-...\"` "
                "to `.streamlit/secrets.toml` (or to the app's Secrets on "
                "Streamlit Cloud) and reload the page.")

    if current is None:
        name = (profile or {}).get("name") or ""
        greeting = f"Ready when you are{', ' + name if name else ''}."
        st.markdown(textwrap.dedent(f"""
        <div class="mf-feed-hero">
          <span class="mf-badge mf-badge-accent">Your weekly digest</span>
          <h2>{html.escape(greeting)}</h2>
          <p class="mf-hero-preview">We'll scan the latest arXiv papers and
          curate a personalised top 10 — from your profile, or from whatever
          you describe. Nothing runs until you press generate.</p>
        </div>
        """), unsafe_allow_html=True)
        if llm:
            render_generator(llm, profile, has_issue=False)
    else:
        if llm:
            render_generator(llm, profile, has_issue=True)
        _render_issue(current, llm, profile)

    _render_past_issues()


def render_newsletter():
    hide_sidebar()
    top_nav()
    st.title("📰 MindFeed — Personal newsletter")
    st.caption("Your weekly research pick, built by Claude from your profile — "
               "or from whatever you describe.")

    llm = _claude()
    left_col, main_col = st.columns([1, 2.6], gap="large")
    with left_col:
        render_profile_rail(llm)
    with main_col:
        render_newsletter_main(llm)


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
    st.title("🔍 MindFeed — Explore trends")

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
