import streamlit as st
import torch
import joblib
import requests
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from nltk.sentiment import SentimentIntensityAnalyzer
from newspaper import Article
import numpy as np
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Analyzer",
    page_icon="📰",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
#  WARM EDITORIAL PALETTE  (hardcoded — cream / amber / espresso)
#
#  Page bg          #fdf6ec   (warm cream)
#  Surface bg       #fffdf7   (off-white cream)
#  Mid layer        #faefd8   (light amber wash)
#  Light accent     #f5e4c0   (amber tint)
#  Border           #e8d5b0   (warm tan)
#  Accent           #b45309   (amber-700)
#  Strong / heading #78350f   (espresso)
#  Muted text       #92400e   (amber-800)
#  Body text        #3b1f07   (dark espresso)
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=Source+Serif+4:wght@300;400;600&family=DM+Sans:wght@400;500;600&display=swap');

    /* ── Global background & font ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"],
    .main .block-container {
        background-color: #fdf6ec !important;
        font-family: 'DM Sans', sans-serif;
        color: #3b1f07;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #faefd8 !important;
        border-right: 1px solid #e8d5b0 !important;
    }
    [data-testid="stSidebar"] * { color: #3b1f07 !important; }

    .sidebar-brand {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #78350f !important;
        display: block;
        margin-bottom: 1.2rem;
    }
    .sidebar-info {
        font-size: 0.8rem;
        color: #92400e !important;
        line-height: 1.7;
        background: #f5e4c0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        border-left: 3px solid #b45309;
    }

    /* ── Tabs ── */
    div[data-testid="stTabs"] button {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        color: #92400e;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #78350f;
        border-bottom: 2px solid #b45309;
    }

    /* ── Inputs ── */
    .stTextArea textarea,
    .stTextInput input {
        background-color: #fffdf7 !important;
        border: 1px solid #e8d5b0 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #3b1f07 !important;
    }
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #b45309 !important;
        box-shadow: 0 0 0 2px #f5e4c0 !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background-color: #b45309 !important;
        color: #fffdf7 !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em;
    }
    .stButton > button[kind="primary"]:hover { background-color: #92400e !important; }
    .stButton > button[kind="primary"]:disabled {
        background-color: #e8d5b0 !important;
        color: #b45309 !important;
    }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div > div { background-color: #b45309 !important; }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #fffdf7 !important;
        border: 1px solid #e8d5b0 !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary { color: #78350f !important; font-weight: 500; }

    /* ── Hero banner ── */
    .hero {
        padding: 2.6rem 2.2rem 2rem;
        border-radius: 16px;
        background-color: #78350f;
        color: #fdf6ec;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        border: 1px solid #92400e;
    }
    .hero::after {
        content: "📰";
        position: absolute;
        right: 2rem; top: 50%;
        transform: translateY(-50%);
        font-size: 6rem;
        opacity: 0.07;
        pointer-events: none;
    }
    .hero-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #f5e4c0;
        margin: 0 0 0.5rem;
        opacity: 0.8;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0 0 0.4rem;
        letter-spacing: -0.01em;
        color: #fdf6ec;
        line-height: 1.15;
    }
    .hero-sub {
        font-size: 0.92rem;
        color: #f5e4c0;
        margin: 0;
        opacity: 0.75;
    }

    /* ── Pill badge ── */
    .pill {
        display: inline-block;
        padding: 0.42rem 1.1rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }

    /* ── Result cards ── */
    .result-card {
        background: #fffdf7;
        border: 1px solid #e8d5b0;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
    }
    .card-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #b45309;
        margin-bottom: 0.65rem;
    }
    .card-sub {
        font-size: 0.78rem;
        color: #92400e;
        margin-top: 0.35rem;
    }

    /* ── Stat blocks ── */
    .stat-block {
        text-align: center;
        padding: 1.1rem 1rem;
        background: #faefd8;
        border-radius: 10px;
        border: 1px solid #e8d5b0;
    }
    .stat-num {
        font-family: 'Playfair Display', serif;
        font-size: 1.9rem;
        font-weight: 600;
        color: #78350f;
        line-height: 1;
    }
    .stat-lbl {
        font-size: 0.68rem;
        color: #b45309;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    /* ── Summary box ── */
    .summary-box {
        background: #fffdf7;
        border-left: 4px solid #b45309;
        border-top: 1px solid #e8d5b0;
        border-bottom: 1px solid #e8d5b0;
        border-right: 1px solid #e8d5b0;
        border-radius: 0 10px 10px 0;
        padding: 1.3rem 1.6rem;
        font-family: 'Source Serif 4', serif;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #3b1f07;
    }

    /* ── Section titles ── */
    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #78350f;
        margin: 1.8rem 0 0.9rem;
        padding-bottom: 0.45rem;
        border-bottom: 2px solid #e8d5b0;
    }

    /* ── Bar chart rows ── */
    .bar-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.6rem;
    }
    .bar-label {
        width: 200px;
        font-size: 0.83rem;
        color: #78350f;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .bar-track {
        flex: 1;
        height: 8px;
        background: #f5e4c0;
        border-radius: 99px;
        overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 99px; }
    .bar-val {
        font-size: 0.78rem;
        color: #b45309;
        width: 50px;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <p class="hero-eyebrow">NLP · Summarization · Sentiment · Topic</p>
        <p class="hero-title">News Article Analyzer</p>
        <p class="hero-sub">DistilBART summary &nbsp;·&nbsp; RoBERTa sentiment &nbsp;·&nbsp; NMF topic detection</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── TOPIC LABELS ───────────────────────────────────────────────────────────────
TOPIC_LABELS = {
    0: {"label": "Religion & Law",               "bg": "#fef3c7", "fg": "#78350f"},
    1: {"label": "Court, Crime & Justice",       "bg": "#fee2e2", "fg": "#991b1b"},
    2: {"label": "Police, Assaults & Incidents", "bg": "#ffedd5", "fg": "#9a3412"},
    3: {"label": "Entertainment",                "bg": "#fce7f3", "fg": "#9d174d"},
    4: {"label": "International Politics",       "bg": "#fef3c7", "fg": "#92400e"},
    5: {"label": "Film & Lifestyle",             "bg": "#ede9fe", "fg": "#5b21b6"},
    6: {"label": "Government & Security",        "bg": "#dcfce7", "fg": "#166534"},
    7: {"label": "Business & Education",         "bg": "#faefd8", "fg": "#78350f"},
}

# ── MODEL LOADERS ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_distilbart():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    model = BartForConditionalGeneration.from_pretrained(
        "sshleifer/distilbart-cnn-12-6"
    ).to(device)
    return tokenizer, model, device


@st.cache_resource
def load_roberta_sentiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ).to(device)
    model.eval()
    return tokenizer, model, device


@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


@st.cache_resource
def load_nmf_pipeline():
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("nmf_model.pkl")
    return vectorizer, model


# ── NLP FUNCTIONS ──────────────────────────────────────────────────────────────
def distilbart_summary(text: str, max_length: int = 130, min_length: int = 30) -> str:
    tokenizer, model, device = load_distilbart()
    inputs = tokenizer(
        text[:3000], return_tensors="pt",
        truncation=True, max_length=1024, padding=True,
    )
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=max_length, min_length=min_length,
            num_beams=4, early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def roberta_sentiment(text: str):
    tokenizer, model, device = load_roberta_sentiment()
    inputs = tokenizer(
        text[:512], return_tensors="pt",
        truncation=True, max_length=512, padding=True,
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    pred = int(np.argmax(probs))
    color_map = {
        "Positive": ("#dcfce7", "#166534"),
        "Negative": ("#fee2e2", "#991b1b"),
        "Neutral":  ("#fef3c7", "#78350f"),   # warm neutral instead of cold gray
    }
    bg, fg = color_map[label_map[pred]]
    return {
        "label": label_map[pred],
        "confidence": round(float(probs[pred]), 3),
        "bg": bg, "fg": fg,
        "all_scores": {
            "Negative": round(float(probs[0]), 3),
            "Neutral":  round(float(probs[1]), 3),
            "Positive": round(float(probs[2]), 3),
        },
    }


def vader_sentiment(text: str):
    scores = load_vader().polarity_scores(text)
    c = scores["compound"]
    if c >= 0.05:
        label, bg, fg = "Positive", "#dcfce7", "#166534"
    elif c <= -0.05:
        label, bg, fg = "Negative", "#fee2e2", "#991b1b"
    else:
        label, bg, fg = "Neutral", "#fef3c7", "#78350f"
    return {
        "label": label, "confidence": round(abs(c), 3),
        "bg": bg, "fg": fg, "all_scores": scores,
    }


def predict_topic(text: str):
    vectorizer, model = load_nmf_pipeline()
    weights = model.transform(vectorizer.transform([text]))[0]
    top = int(np.argmax(weights))
    info = TOPIC_LABELS.get(top, {"label": f"Topic {top}", "bg": "#faefd8", "fg": "#78350f"})
    return {
        "label": info["label"], "topic_id": top,
        "confidence": round(float(weights[top] / (weights.sum() + 1e-9)), 3),
        "bg": info["bg"], "fg": info["fg"],
        "all_weights": {
            TOPIC_LABELS.get(i, {"label": f"Topic {i}"})["label"]: round(float(w), 4)
            for i, w in enumerate(weights)
        },
    }


# ── URL FETCHER ────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "DNT": "1",
}


def fetch_article_from_url(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    a = Article(url)
    a.set_html(r.text)
    a.parse()
    return a.text


# ── HTML HELPERS ───────────────────────────────────────────────────────────────
def pill(label: str, bg: str, fg: str, size: str = "0.92rem") -> str:
    return (
        f'<span class="pill" style="background:{bg}; color:{fg}; font-size:{size};">'
        f"{label}</span>"
    )


def bar_row(label: str, value: float, max_val: float = 1.0, color: str = "#b45309") -> str:
    pct = min(100, round(value / max(max_val, 1e-9) * 100))
    return (
        f'<div class="bar-row">'
        f'<span class="bar-label">{label}</span>'
        f'<div class="bar-track">'
        f'<div class="bar-fill" style="width:{pct}%; background:{color};"></div>'
        f'</div>'
        f'<span class="bar-val">{value:.4f}</span>'
        f'</div>'
    )


# ── WARM-UP ────────────────────────────────────────────────────────────────────
with st.spinner("Loading models — first run only…"):
    load_distilbart()
    load_roberta_sentiment()
    load_vader()
    load_nmf_pipeline()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-brand">⚙️ Settings</span>', unsafe_allow_html=True)

    sentiment_choice = st.radio(
        "Sentiment model",
        ["RoBERTa (most accurate)", "VADER (fastest)"],
        index=0,
        help="RoBERTa: Macro F1 0.382 · VADER: Macro F1 0.356 (7× faster)",
    )

    summary_length = st.slider(
        "Summary length (tokens)",
        min_value=50, max_value=200, value=130, step=10,
    )

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-info">'
        '<strong>Models used</strong><br>'
        '📝 DistilBART-CNN-12-6<br>'
        '🤖 Cardiff RoBERTa sentiment<br>'
        '📊 NMF topic model (8 classes)'
        '</div>',
        unsafe_allow_html=True,
    )

# ── INPUT TABS ─────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝  Paste Article", "🔗  Paste URL"])
article_text = ""

with tab1:
    raw = st.text_area(
        "article", height=220,
        placeholder="Paste a full news article here and click Analyze…",
        label_visibility="collapsed",
    )
    if raw.strip():
        article_text = raw.strip()

with tab2:
    url_input = st.text_input(
        "url", placeholder="https://www.bbc.com/news/…",
        label_visibility="collapsed",
    )
    if url_input.strip():
        with st.spinner("Fetching article…"):
            try:
                article_text = fetch_article_from_url(url_input.strip())
                st.success(f"✅  Fetched {len(article_text):,} characters")
                with st.expander("Preview fetched text"):
                    st.write(article_text[:1500] + ("…" if len(article_text) > 1500 else ""))
            except Exception as e:
                st.error(f"Could not fetch article: {e}")

analyze = st.button(
    "🚀  Analyze Article",
    type="primary",
    use_container_width=True,
    disabled=not bool(article_text),
)

# ── ANALYSIS & RESULTS ─────────────────────────────────────────────────────────
if analyze:
    if len(article_text.strip()) < 100:
        st.warning("Please provide at least ~100 characters of text.")
    else:
        prog = st.progress(0, text="Starting…")

        with st.spinner("Generating summary…"):
            summary = distilbart_summary(article_text, max_length=summary_length)
        prog.progress(40, text="Summary done · Analyzing sentiment…")

        with st.spinner("Analyzing sentiment…"):
            src = summary or article_text
            sentiment = (
                roberta_sentiment(src) if "RoBERTa" in sentiment_choice
                else vader_sentiment(src)
            )
        prog.progress(70, text="Sentiment done · Detecting topic…")

        with st.spinner("Detecting topic…"):
            topic = predict_topic(summary or article_text)
        prog.progress(100, text="Complete!")
        prog.empty()

        # Results header
        st.markdown('<p class="section-title">Results</p>', unsafe_allow_html=True)

        # Stat row
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f'<div class="stat-block">'
            f'<div class="stat-num">{len(article_text.split()):,}</div>'
            f'<div class="stat-lbl">Article words</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c2.markdown(
            f'<div class="stat-block">'
            f'<div class="stat-num">{len(summary.split()) if summary else 0}</div>'
            f'<div class="stat-lbl">Summary words</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        compression = (
            round(100 * len(summary.split()) / max(len(article_text.split()), 1))
            if summary else 0
        )
        c3.markdown(
            f'<div class="stat-block">'
            f'<div class="stat-num">{compression}%</div>'
            f'<div class="stat-lbl">Compression</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Topic + Sentiment cards
        col_t, col_s = st.columns(2)
        with col_t:
            st.markdown(
                f'<div class="result-card">'
                f'<div class="card-label">📌 Topic</div>'
                f'{pill(topic["label"], topic["bg"], topic["fg"])}'
                f'<div class="card-sub">Confidence {topic["confidence"]:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_s:
            model_name = "RoBERTa" if "RoBERTa" in sentiment_choice else "VADER"
            st.markdown(
                f'<div class="result-card">'
                f'<div class="card-label">😊 Sentiment &nbsp;·&nbsp; <em>{model_name}</em></div>'
                f'{pill(sentiment["label"], sentiment["bg"], sentiment["fg"])}'
                f'<div class="card-sub">Confidence {sentiment["confidence"]:.1%}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Summary section
        st.markdown('<p class="section-title">Summary</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="summary-box">{summary or "Could not generate a summary."}</div>',
            unsafe_allow_html=True,
        )

        # Expanders
        with st.expander("📊  Sentiment breakdown"):
            scores = sentiment["all_scores"]
            max_v = max(scores.values()) if scores else 1
            sent_colors = {
                "Positive": "#16a34a",   # green
                "Negative": "#dc2626",   # red
                "Neutral":  "#b45309",   # amber
                "pos":      "#16a34a",
                "neg":      "#dc2626",
                "neu":      "#b45309",
                "compound": "#92400e",   # espresso
            }
            for k, v in scores.items():
                st.markdown(
                    bar_row(k, v, max(max_v, 0.001), sent_colors.get(k, "#b45309")),
                    unsafe_allow_html=True,
                )

        with st.expander("🗂️  Topic weights across all 8 NMF topics"):
            sorted_topics = sorted(
                topic["all_weights"].items(), key=lambda x: x[1], reverse=True
            )
            max_w = sorted_topics[0][1] if sorted_topics else 1
            for lbl, w in sorted_topics:
                is_top = lbl == topic["label"]
                color = "#78350f" if is_top else "#d97706"   # espresso vs amber
                st.markdown(
                    bar_row(("★ " if is_top else "  ") + lbl, w, max(max_w, 1e-6), color),
                    unsafe_allow_html=True,
                )
