"""
nlp_pipeline.py
---------------
The NLP core of the project. Heavy models are lazy-loaded and cached so the
Streamlit app stays responsive and models are only downloaded once.

Pipeline stages (applied to the latest arXiv abstracts for a subject):
  1. Text preprocessing / cleaning          -> preprocess()
  2. Keyword extraction (TF-IDF)             -> extract_keywords()
  3. Topic discovery (TF-IDF + KMeans)       -> discover_topics()
  4. Named Entity Recognition (spaCy)        -> extract_entities()
  5. Summarization (HF transformers, LLM)    -> summarize()   [optional component]
  6. Trend report assembly                   -> build_report()

We deliberately keep each step small and interpretable, which is what the
assignment rewards ("a well-executed simple pipeline is better than a complex
but incomplete system"). For research abstracts we discover *topics* (an
unsupervised form of text classification) instead of sentiment, since sentiment
is not meaningful for scientific papers.
"""

import re
import functools
from collections import Counter

# Domain stopwords: generic academic-writing boilerplate that would otherwise
# dominate the keyword list without saying anything about the actual trend.
# Combined with scikit-learn's built-in English stopword list.
EXTRA_STOPWORDS = {
    "paper", "papers", "using", "used", "use", "based", "approach",
    "approaches", "method", "methods", "results", "result", "propose",
    "proposed", "proposes", "present", "presents", "show", "shows", "shown",
    "study", "studies", "work", "works", "novel", "new", "via", "e.g", "i.e",
    "et", "al", "however", "thus", "therefore", "also", "well", "achieve",
    "achieves", "state", "art", "performance", "task", "tasks", "problem",
    "problems", "model", "models", "framework", "data", "dataset", "datasets",
    "learning", "training", "train", "trained", "different", "existing",
    "recent", "various", "provide", "provides", "compared", "including",
}


@functools.lru_cache(maxsize=1)
def _stopwords():
    """The full English stopword set (sklearn's list + our domain extras)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return list(ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS))


def preprocess(text):
    """Lowercase, keep letters only, drop very short tokens. (Kept simple; the
    TF-IDF vectoriser handles stopword removal and n-grams downstream.)"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def _corpus(docs):
    """The list of texts (title + abstract) we run models over."""
    return [d.get("text", "") for d in docs]


# ---- 1. Keyword extraction (TF-IDF over the retrieved corpus) -------------

@functools.lru_cache(maxsize=1)
def _vectorizer_class():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer


def extract_keywords(docs, top_n=20):
    """
    TF-IDF over the latest abstracts to surface the most distinctive terms
    (the "trending" vocabulary). We include unigrams and bigrams so multi-word
    concepts like "large language" or "diffusion model" show up.

    Returns list of (term, score), highest first.
    """
    texts = [t for t in _corpus(docs) if t.strip()]
    if len(texts) < 2:
        return []
    TfidfVectorizer = _vectorizer_class()
    vec = TfidfVectorizer(
        stop_words=_stopwords(),
        ngram_range=(1, 2),
        min_df=2,                       # ignore terms in a single paper only
        max_features=2000,
        token_pattern=r"[a-zA-Z][a-zA-Z\-]+",
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []                        # empty vocab (too little text)
    scores = X.sum(axis=0).A1            # total TF-IDF mass per term
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
    return [(t, round(float(s), 3)) for t, s in ranked[:top_n]]


# ---- 2. Topic discovery (KMeans, or BERTopic) ---------------------------
#
# Two interchangeable methods, both returning the same shape so the app can
# offer them as a toggle:
#   - "kmeans"   : TF-IDF + KMeans. Fast, deterministic, light deps. Great on
#                  small batches (~60 papers); needs a chosen number of topics.
#   - "bertopic" : sentence-embeddings + UMAP + HDBSCAN + c-TF-IDF. More
#                  semantic, discovers the topic count itself, but heavier and
#                  best with more papers (150+).

def discover_topics(docs, n_topics=5, method="kmeans", terms_per_topic=6):
    """
    Discover sub-topics ("sub-trends") among the papers and tag each doc in
    place with a 'topic' id. Dispatches to the chosen method.

    Returns list of dicts: { id, label, terms, size, share }, largest first.
    """
    if method == "bertopic":
        return _discover_topics_bertopic(docs, n_topics, terms_per_topic)
    return _discover_topics_kmeans(docs, n_topics, terms_per_topic)


def _discover_topics_kmeans(docs, n_topics, terms_per_topic):
    """TF-IDF + KMeans clustering. Labels each cluster by its most
    characteristic terms (from the cluster centroid)."""
    texts = _corpus(docs)
    valid_idx = [i for i, t in enumerate(texts) if t.strip()]
    if len(valid_idx) < n_topics or len(valid_idx) < 4:
        return []                        # not enough papers to cluster sensibly

    from sklearn.cluster import KMeans
    TfidfVectorizer = _vectorizer_class()
    vec = TfidfVectorizer(
        stop_words=_stopwords(),
        ngram_range=(1, 2),
        min_df=2,
        max_features=2000,
        token_pattern=r"[a-zA-Z][a-zA-Z\-]+",
    )
    try:
        X = vec.fit_transform([texts[i] for i in valid_idx])
    except ValueError:
        return []

    k = min(n_topics, X.shape[0])
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # tag each doc with its topic id (docs without text get -1)
    for d in docs:
        d["topic"] = -1
    for pos, i in enumerate(valid_idx):
        docs[i]["topic"] = int(labels[pos])

    terms = vec.get_feature_names_out()
    centroids = km.cluster_centers_
    sizes = Counter(labels.tolist())
    total = len(valid_idx)

    topics = []
    for cid in range(k):
        top_term_idx = centroids[cid].argsort()[::-1][:terms_per_topic]
        top_terms = [terms[j] for j in top_term_idx]
        topics.append({
            "id": cid,
            "label": ", ".join(top_terms[:3]),
            "terms": top_terms,
            "size": sizes.get(cid, 0),
            "share": round(100 * sizes.get(cid, 0) / total, 1),
        })
    topics.sort(key=lambda t: t["size"], reverse=True)
    return topics


def _discover_topics_bertopic(docs, n_topics, terms_per_topic):
    """
    BERTopic: embeds each abstract (sentence-transformers), reduces dimensions
    with UMAP, clusters with HDBSCAN, and labels topics with class-based TF-IDF.
    We fix UMAP's random_state for reproducibility and reduce to n_topics.
    Papers HDBSCAN considers outliers get topic id -1 and are not shown.
    """
    texts = _corpus(docs)
    valid_idx = [i for i, t in enumerate(texts) if t.strip()]
    if len(valid_idx) < 4:
        return []
    valid_texts = [texts[i] for i in valid_idx]

    from bertopic import BERTopic
    from umap import UMAP
    from sklearn.feature_extraction.text import CountVectorizer

    # small-corpus-friendly settings + reproducibility
    umap_model = UMAP(
        n_neighbors=min(15, len(valid_texts) - 1),
        n_components=5, min_dist=0.0, metric="cosine", random_state=42,
    )
    # c-TF-IDF over unigrams+bigrams with our domain stopwords removed
    vectorizer_model = CountVectorizer(
        stop_words=_stopwords(), ngram_range=(1, 2))

    topic_model = BERTopic(
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=max(3, len(valid_texts) // 20),
        nr_topics=n_topics,              # reduce to the requested number
        calculate_probabilities=False,
        verbose=False,
    )
    try:
        labels, _ = topic_model.fit_transform(valid_texts)
    except Exception:
        return []                        # BERTopic can fail on tiny/degenerate corpora

    # tag each doc with its topic id
    for d in docs:
        d["topic"] = -1
    for pos, i in enumerate(valid_idx):
        docs[i]["topic"] = int(labels[pos])

    total = len(valid_idx)
    topics = []
    for row in topic_model.get_topic_info().itertuples(index=False):
        tid = int(row.Topic)
        if tid == -1:                    # outlier bucket - skip
            continue
        words = [w for w, _ in (topic_model.get_topic(tid) or [])
                 if w][:terms_per_topic]
        if not words:
            continue
        size = int(row.Count)
        topics.append({
            "id": tid,
            "label": ", ".join(words[:3]),
            "terms": words,
            "size": size,
            "share": round(100 * size / total, 1),
        })
    topics.sort(key=lambda t: t["size"], reverse=True)
    return topics


# ---- 3. Entity extraction: methods/models (rules) + institutions (spaCy) --

@functools.lru_cache(maxsize=1)
def _load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


# All-caps tokens that are common English words / boilerplate, not methods.
_ACRONYM_STOP = {
    "THE", "AND", "FOR", "WITH", "THIS", "THAT", "FROM", "WE", "OUR", "ARE",
    "CAN", "NOT", "USE", "USING", "PDF", "URL", "HTML", "III", "II", "IV",
    "USA", "UK", "EU", "OK",
}


def _scientific_terms(text):
    """
    Rule-based extraction of the entities that actually matter in research:
    methods, models, datasets and named systems. These show up as acronyms or
    CamelCase names (LLM, GAN, RLHF, BERT, ResNet, ImageNet, GPT-4), i.e. tokens
    with two or more capital letters. Cheap, model-free, and domain-agnostic
    across all arXiv fields (unlike a general-purpose newswire NER).
    """
    out = []
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,15}", text):
        caps = sum(1 for c in tok if c.isupper())
        if caps >= 2 and tok.upper() not in _ACRONYM_STOP:
            out.append(tok)
    return out


def extract_entities(docs, max_chars=100000):
    """
    Surface the entities that are meaningful for a scientific trend, in two
    buckets, returned as {label: [(entity, count), ...]}:

      - "METHOD"  : methods / models / datasets / named systems, via a
                    rule-based acronym/CamelCase matcher (LLM, GAN, BERT, ...).
                    The abbreviated form of "what techniques are trending".
      - "CONCEPT" : named techniques / concepts via spaCy NER - the spelled-out
                    form (Reinforcement Learning, Convolutional Neural Networks).

    We deliberately drop PERSON/GPE/NORP/LOC: for research abstracts those are
    mostly noise (author names, country mentions) rather than trend signal.
    """
    blob = " ".join(_corpus(docs))[:max_chars]   # original case preserved
    result = {}

    # 1. methods / models / acronyms (rule-based, no model needed)
    terms = Counter(_scientific_terms(blob))
    if terms:
        result["METHOD"] = terms.most_common(15)
    method_set = set(terms)

    # 2. named techniques / concepts via spaCy NER. Abstracts don't contain
    #    author affiliations, so spaCy's ORG/PRODUCT hits on scientific text are
    #    almost never real institutions - they are capitalised named techniques
    #    and concepts ("Reinforcement Learning", "Convolutional Neural
    #    Networks"). We keep those: multi-word Title-Case phrases (the richest
    #    signal) or mixed-case named tokens, de-duplicated against the acronyms.
    nlp = _load_spacy()
    doc = nlp(blob)
    concepts = Counter()
    for ent in doc.ents:
        if ent.label_ not in ("ORG", "PRODUCT", "WORK_OF_ART"):
            continue
        txt = ent.text.strip()
        multiword = " " in txt
        mixed_case = any(c.islower() for c in txt) and any(c.isupper() for c in txt)
        if (multiword or (mixed_case and len(txt) >= 4)) and txt not in method_set:
            concepts[txt] += 1
    if concepts:
        result["CONCEPT"] = concepts.most_common(12)

    return result


# ---- 4. Summarization (optional LLM component, HF transformers) ----------

@functools.lru_cache(maxsize=1)
def _load_summarizer():
    from transformers import pipeline
    # distilbart: a distilled BART, free, runs locally, no API key
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")


def summarize(docs, max_input_chars=3000):
    """
    Feed the concatenated abstracts into a distilled BART summariser to get an
    automatic overview of what the latest papers are about. This is the
    optional LLM component - it complements, not replaces, the rest of the
    pipeline.
    """
    summarizer = _load_summarizer()
    blob = " ".join(_corpus(docs))[:max_input_chars]
    if len(blob.split()) < 40:
        return "Not enough text to generate a reliable summary."
    out = summarizer(blob, max_length=140, min_length=45, do_sample=False)
    return out[0]["summary_text"]


# ---- 5. Trend report assembly --------------------------------------------

def build_report(docs, keywords, topics, summary, subject_label):
    """
    Assemble a short, human-readable trend report by combining the pipeline
    outputs: coverage stats, the auto-summary, the leading keywords, and the
    discovered sub-topics. This is the "report about the latest trend" the app
    delivers. Returns a markdown string.
    """
    n = len(docs)
    dates = sorted(d["date"] for d in docs if d.get("date"))
    span = f"{dates[0]} to {dates[-1]}" if dates else "recent submissions"
    kw_line = ", ".join(f"**{t}**" for t, _ in keywords[:8]) or "n/a"

    lines = [
        f"## Trend report — {subject_label}",
        "",
        f"Based on the **{n} most recent papers** on arXiv "
        f"({span}).",
        "",
        "### What the latest work is about",
        summary,
        "",
        "### Leading terms right now",
        kw_line,
        "",
        "### Main sub-topics detected",
    ]
    if topics:
        for t in topics:
            lines.append(
                f"- **{t['label']}** — {t['size']} papers ({t['share']}%) · "
                f"key terms: {', '.join(t['terms'])}"
            )
    else:
        lines.append("- Not enough papers to detect distinct sub-topics.")
    return "\n".join(lines)
