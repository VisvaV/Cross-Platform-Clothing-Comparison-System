"""Streamlit UI — Clothing Comparison System (premium dark fashion theme)."""

import os
import sys
import tempfile

# Ensure project root is on sys.path regardless of working directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from ranking.ranking_engine import rank_by_price, rank_by_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAISS_INDEX_PATH = os.path.join(_ROOT, "data/faiss_index.bin")
FAISS_MAP_PATH   = os.path.join(_ROOT, "data/faiss_id_map.pkl")
TFIDF_PATH       = os.path.join(_ROOT, "data/tfidf_vectorizer.pkl")
TEXT_EMB_PATH    = os.path.join(_ROOT, "data/text_embeddings.npy")
DB_PATH          = os.path.join(_ROOT, "data/products.db")

PIPELINE_MSG = (
    "No product index found. Run the offline pipeline first:\n\n"
    "```\npython -m embeddings.generate_embeddings --categories tops dresses --limit 50\n```"
)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clothing Comparison System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0a0f; color: #e8e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }
@keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.hero-bar { background: linear-gradient(135deg,#6c3de8,#c850c0,#ff6b6b,#ffd93d,#6c3de8); background-size:300% 300%; animation:gradientShift 8s ease infinite; border-radius:16px; padding:3rem 2.5rem 2.5rem; margin-bottom:2.5rem; position:relative; overflow:hidden; }
.hero-bar::before { content:''; position:absolute; inset:0; background:rgba(10,10,15,0.45); border-radius:16px; }
.hero-title { font-family:'Playfair Display',serif; font-size:3rem; font-weight:700; color:#fff; position:relative; z-index:1; margin:0 0 0.4rem; }
.hero-sub { font-size:1rem; color:rgba(255,255,255,0.75); position:relative; z-index:1; margin:0; font-weight:300; }
.section-label { font-size:0.7rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:#9b8ec4; margin-bottom:0.6rem; }
div[data-testid="stTextInput"] input { background:rgba(255,255,255,0.05)!important; border:1.5px solid rgba(255,255,255,0.1)!important; border-radius:10px!important; color:#e8e8f0!important; }
div[data-testid="stTextInput"] input:focus { border-color:#6c3de8!important; box-shadow:0 0 0 3px rgba(108,61,232,0.2)!important; }
@keyframes pulseGlow { 0%,100%{box-shadow:0 0 18px rgba(108,61,232,0.5)} 50%{box-shadow:0 0 32px rgba(200,80,192,0.7)} }
div[data-testid="stButton"] > button[kind="primary"] { background:linear-gradient(135deg,#6c3de8,#c850c0)!important; border:none!important; border-radius:12px!important; color:#fff!important; font-weight:600!important; animation:pulseGlow 3s ease-in-out infinite; width:100%; }
div[data-testid="stButton"] > button[kind="primary"]:hover { transform:translateY(-2px)!important; opacity:0.92!important; }
hr { border:none!important; border-top:1px solid rgba(255,255,255,0.07)!important; margin:1.5rem 0!important; }
.results-header { font-size:0.72rem; font-weight:600; letter-spacing:2.5px; text-transform:uppercase; color:#9b8ec4; margin-bottom:1.2rem; }
@keyframes fadeSlideUp { from{opacity:0;transform:translateY(22px)} to{opacity:1;transform:translateY(0)} }
.product-card { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:16px; overflow:hidden; transition:transform 0.25s,border-color 0.25s,box-shadow 0.25s; animation:fadeSlideUp 0.45s ease both; margin-bottom:1.2rem; }
.product-card:hover { transform:translateY(-5px); border-color:rgba(108,61,232,0.5); box-shadow:0 12px 40px rgba(108,61,232,0.18); }
.card-body { padding:1rem 1.1rem 1.2rem; }
.card-title { font-size:0.9rem; font-weight:600; color:#e8e8f0; margin:0 0 0.3rem; line-height:1.35; }
.card-meta { font-size:0.75rem; color:#7c6fa0; margin:0 0 0.5rem; text-transform:uppercase; letter-spacing:0.8px; }
.card-price { font-size:1.1rem; font-weight:700; color:#a78bfa; margin:0 0 0.35rem; }
.card-score { font-size:0.72rem; color:#5a5270; margin:0 0 0.8rem; }
.score-bar-bg { background:rgba(255,255,255,0.07); border-radius:99px; height:4px; margin-bottom:0.9rem; overflow:hidden; }
.score-bar-fill { height:4px; border-radius:99px; background:linear-gradient(90deg,#6c3de8,#c850c0); }
div[data-testid="stLinkButton"] a { background:rgba(108,61,232,0.15)!important; border:1px solid rgba(108,61,232,0.4)!important; border-radius:8px!important; color:#a78bfa!important; font-size:0.8rem!important; text-decoration:none!important; display:inline-block; width:100%; text-align:center; }
div[data-testid="stLinkButton"] a:hover { background:rgba(108,61,232,0.3)!important; color:#fff!important; }
.card-delay-0{animation-delay:0s} .card-delay-1{animation-delay:.06s} .card-delay-2{animation-delay:.12s}
.card-delay-3{animation-delay:.18s} .card-delay-4{animation-delay:.24s} .card-delay-5{animation-delay:.30s}
.card-delay-6{animation-delay:.36s} .card-delay-7{animation-delay:.42s} .card-delay-8{animation-delay:.48s}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-bar">
    <p class="hero-title">Clothing Comparison</p>
    <p class="hero-sub">Find similar styles across every major fashion platform instantly</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def index_exists() -> bool:
    return os.path.exists(FAISS_INDEX_PATH)

@st.cache_resource(show_spinner=False)
def load_searcher():
    from vector_search.similarity_search import SimilaritySearch
    return SimilaritySearch(
        db_path=DB_PATH,
        faiss_index_path=FAISS_INDEX_PATH,
        faiss_map_path=FAISS_MAP_PATH,
        tfidf_path=TFIDF_PATH,
        text_embeddings_path=TEXT_EMB_PATH,
    )

def save_upload(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("results", []),
    ("search_type", "text"),
    ("search_error", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Input UI
# ---------------------------------------------------------------------------
col_img, col_sep, col_txt = st.columns([3, 0.1, 3])

with col_img:
    st.markdown('<p class="section-label">Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "upload", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed"
    )

with col_sep:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:center;height:100%;'
        'padding-top:2rem;color:rgba(255,255,255,0.15);font-size:1.4rem">|</div>',
        unsafe_allow_html=True,
    )

with col_txt:
    st.markdown('<p class="section-label">Describe the Item</p>', unsafe_allow_html=True)
    text_query = st.text_input(
        "query", placeholder="e.g. red floral summer dress", label_visibility="collapsed"
    )

search_clicked = st.button("🔍  Search", type="primary", use_container_width=True)

# Sort + divider
sort_col, _ = st.columns([2, 5])
with sort_col:
    sort_mode = st.radio(
        "Sort", ["Best Match", "Lowest Price"], horizontal=True, label_visibility="collapsed"
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Run search on click — persist to session_state
# ---------------------------------------------------------------------------
if search_clicked:
    has_image = uploaded_file is not None
    has_text  = bool(text_query.strip())

    if not has_image and not has_text:
        st.warning("Please upload an image or enter a description.")
    elif not index_exists():
        st.warning(PIPELINE_MSG)
    else:
        st.session_state.search_error = None
        tmp_path = None
        with st.spinner("Searching across platforms..."):
            try:
                searcher = load_searcher()
                if has_image:
                    tmp_path = save_upload(uploaded_file)

                if has_image and has_text:
                    st.session_state.results = searcher.search_hybrid(tmp_path, text_query.strip())
                    st.session_state.search_type = "hybrid"
                elif has_image:
                    st.session_state.results = searcher.search_by_image(tmp_path)
                    st.session_state.search_type = "image"
                else:
                    st.session_state.results = searcher.search_by_text(text_query.strip())
                    st.session_state.search_type = "text"

            except Exception as exc:
                st.session_state.search_error = str(exc)
                st.session_state.results = []
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------
if st.session_state.search_error:
    st.error(f"Search failed: {st.session_state.search_error}")

if st.session_state.results:
    results = list(st.session_state.results)
    search_type = st.session_state.search_type

    if sort_mode == "Lowest Price":
        results = rank_by_price(results)
    else:
        score_key = "hybrid_score" if search_type == "hybrid" else "similarity_score"
        results = sorted(results, key=lambda r: r.get(score_key, 0.0), reverse=True)

    st.markdown(
        f'<p class="results-header">{len(results)} results found</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="medium")
    for i, product in enumerate(results):
        delay_cls  = f"card-delay-{min(i, 8)}"
        score      = product.get("hybrid_score") or product.get("similarity_score") or 0.0
        score_pct  = min(max(score * 100, 0), 100)
        price      = product.get("price")
        price_str  = f"₹{price:,.0f}" if price else "Price unavailable"
        title      = product.get("product_title") or "Unknown"
        platform   = (product.get("platform") or "").upper()
        brand      = product.get("brand") or ""
        meta       = " / ".join(filter(None, [platform, brand]))
        product_url = product.get("product_url") or ""
        image_path  = product.get("image_path") or ""

        with cols[i % 3]:
            if image_path and os.path.exists(image_path):
                st.image(image_path, use_container_width=True)
            else:
                st.markdown(
                    '<div style="background:rgba(255,255,255,0.04);border-radius:12px 12px 0 0;'
                    'height:200px;display:flex;align-items:center;justify-content:center;'
                    'color:rgba(255,255,255,0.15);font-size:0.8rem;">NO IMAGE</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"""
<div class="product-card {delay_cls}" style="border-radius:0 0 16px 16px;border-top:none;">
  <div class="card-body">
    <p class="card-title">{title}</p>
    <p class="card-meta">{meta}</p>
    <p class="card-price">{price_str}</p>
    <p class="card-score">Match score &nbsp; {score:.3f}</p>
    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{score_pct:.1f}%"></div></div>
  </div>
</div>""", unsafe_allow_html=True)

            if product_url:
                st.link_button("View Product", product_url, use_container_width=True)
            st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)
