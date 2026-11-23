import streamlit as st
import os
import uuid
import time
import logging
from datetime import timezone
from typing import Iterator

# sqlite3 swap for Linux (Chroma requires SQLite >= 3.35; Ubuntu 20.04 ships 3.31)
# Must run before any 'import chromadb' anywhere in this process.
import sys as _sys_init, platform as _platform_init
if _platform_init.system() == "Linux":
    try:
        import pysqlite3 as _pysqlite3
        _sys_init.modules["sqlite3"] = _pysqlite3
    except ImportError:
        pass
del _sys_init, _platform_init
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from db import init_indexes, register_user, verify_user, save_chat, list_chats, load_chat
from i18n import LANGUAGES, DB_MSG_KEYS, t
from intent_config import (
    ENABLE_INTENT_DETECTION,
    MAX_FOLLOWUP_ROUNDS,
    SKIP_FOLLOWUP_TRIGGERS,
    SKIP_MAX_LEN,
    MAX_CLARIFY_QUESTIONS,
    INTENT_ANALYSIS_PROMPT,
)

st.set_page_config(
    page_title="勞資爭議智慧法務 AI 顧問",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="auto",
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

def _clean_secret(name: str) -> str | None:
    """Env secret with stray whitespace / quotes removed.

    Pasting a key into a hosting dashboard often adds a trailing newline or
    the quotes from .env; httpx then rejects the header and the SDK reports
    it as APIConnectionError, which looks like a network problem.
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    val = raw.strip().strip('"').strip("'").strip()
    # Warn on a suspicious shape (never log the value). Not on every rerun:
    # Streamlit re-executes this module on each interaction.
    if val != raw or not val.startswith("sk-ant-"):
        print(f"[CONFIG] {name}: len={len(val)} prefix_ok={val.startswith('sk-ant-')} "
              f"cleaned={val != raw}", flush=True)
    return val or None


ANTHROPIC_API_KEY = _clean_secret("ANTHROPIC_API_KEY")

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chroma").lower()

# BM25 (jieba) + vector fused by RRF; set HYBRID_SEARCH=0 for vector-only.
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "1") != "0"

EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3").lower()
if "text2vec" in EMBED_MODEL:
    _MODEL_NAME    = "shibing624/text2vec-base-chinese"
    _MODEL_SUFFIX  = ""
else:
    _MODEL_NAME    = "BAAI/bge-m3"
    _MODEL_SUFFIX  = "_bgem3"
_LAW_COL_NAME   = f"law_chunks{_MODEL_SUFFIX}"
_CASES_COL_NAME = f"cases_chunks{_MODEL_SUFFIX}"


def _resolve_chroma_path() -> str:
    """Return writable path for ChromaDB; falls back to tempdir if preferred path fails.

    Chroma's native HNSW writer can't handle non-ASCII paths on Windows (e.g. a
    project under "新增資料夾"): it silently writes no index files, so every
    restart hits "Error loading hnsw index" and rebuilds. Such paths are
    redirected to an ASCII directory under LOCALAPPDATA.
    """
    import tempfile
    preferred = os.getenv(
        "CHROMA_DB_PATH",
        os.path.join(base_dir, "backend", "chroma_db"),
    )
    if not preferred.isascii():
        fallback = os.path.join(
            os.getenv("LOCALAPPDATA") or os.path.expanduser("~"), "law_rag", "chroma_db"
        )
        print(f"[CHROMA] 路徑含非 ASCII 字元，改用：{fallback}", flush=True)
        preferred = fallback
    try:
        os.makedirs(preferred, exist_ok=True)
        probe = os.path.join(preferred, ".write_probe")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return preferred
    except (OSError, PermissionError):
        tmp = os.path.join(tempfile.gettempdir(), "chroma_db")
        os.makedirs(tmp, exist_ok=True)
        return tmp


_CHROMA_DB_PATH = _resolve_chroma_path()


# ==========================================
# 1. 系統初始化
# ==========================================
init_indexes()

defaults = {
    "session_id": str(uuid.uuid4()),
    "logged_in": False,
    "username": None,
    "page": "Login",       # Login → Landing → Chat
    "user_role": None,
    "messages": [],
    "current_chat_id": None,
    "btn_input": None,
    "ui_lang": "繁體中文",
    # intent-detection state (reset when a complete RAG answer is given)
    "intent_ctx": None,    # pending clarification: query/rounds/questions/analysis
    "_pending_cases": None,  # list[(text, meta, score)] from last RAG query; shown on button click
    "_show_cases": False,    # whether the case panel is expanded
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=Inter:wght@400;500;600&display=swap');

    /* ── 色票（墨綠 + 燕麥） ── */
    :root {
        --green-900: #0f2318;
        --green-800: #1a3a28;
        --green-700: #24512f;  /* sidebar hover */
        --green-600: #2d6a3f;
        --green-500: #3d8a55;
        --oat-100:   #faf8f4;  /* 最淺燕麥 */
        --oat-200:   #f3ede0;  /* 主背景 */
        --oat-300:   #e6dece;  /* 分隔 / border */
        --oat-400:   #c9bfa8;  /* 次要 border */
        --oat-600:   #7a6e58;  /* 次要文字 */
        --oat-900:   #1e1a14;  /* 主文字 */
        --accent:    #8b6c2a;  /* 金銅色 accent */
        --accent-lt: #c9a84c;  /* 金色高亮 */
        --white:     #ffffff;
    }

    html, body, .stApp {
        font-family: 'Noto Sans TC', 'Inter', -apple-system, sans-serif !important;
    }

    /* 隱藏 Streamlit 預設元素 */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none !important; }

    /* ── 主背景（燕麥色） ── */
    .stApp { background: var(--oat-200) !important; }

    /* ── 側邊欄（墨綠漸層） ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--green-900) 0%, var(--green-800) 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #d4e8da !important;
    }
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        background-color: rgba(255,255,255,0.07) !important;
        background: rgba(255,255,255,0.07) !important;
        color: #d4e8da !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 8px !important;
        font-size: 0.875rem;
        text-align: left;
        transition: all 0.18s ease;
    }
    section[data-testid="stSidebar"] button:hover,
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(201,168,76,0.2) !important;
        background: rgba(201,168,76,0.2) !important;
        border-color: rgba(201,168,76,0.5) !important;
        color: #f0d98a !important;
    }
    section[data-testid="stSidebar"] button:focus,
    section[data-testid="stSidebar"] button:active,
    section[data-testid="stSidebar"] .stButton > button:focus,
    section[data-testid="stSidebar"] .stButton > button:active {
        background-color: rgba(255,255,255,0.07) !important;
        background: rgba(255,255,255,0.07) !important;
        color: #d4e8da !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* selectbox in sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.09) !important;
        border-color: rgba(255,255,255,0.18) !important;
        border-radius: 8px !important;
        color: #d4e8da !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #d4e8da !important; }

    /* ── 標題 ── */
    h1 { color: var(--green-800) !important; font-weight: 700 !important; letter-spacing: -0.3px; }
    h2, h3 { color: var(--green-700) !important; font-weight: 600 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid var(--oat-300);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.55rem 1.4rem;
        font-weight: 500;
        color: var(--oat-600);
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--green-800) !important;
        background: var(--oat-100) !important;
        border-top: 3px solid var(--accent-lt) !important;
        font-weight: 600;
    }

    /* ── 輸入框 ── */
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1.5px solid var(--oat-300) !important;
        padding: 0.55rem 1rem !important;
        font-size: 0.95rem;
        background: var(--oat-100) !important;
        color: var(--oat-900) !important;
        transition: border-color 0.18s;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--green-600) !important;
        box-shadow: 0 0 0 3px rgba(45,106,63,0.10) !important;
        background: var(--white) !important;
    }

    /* ── 一般按鈕（主內容區） ── */
    .main .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        border: 1.5px solid var(--oat-300) !important;
        color: var(--green-800) !important;
        background: var(--oat-100) !important;
        transition: all 0.18s ease !important;
    }
    .main .stButton > button:hover {
        background: var(--green-800) !important;
        color: var(--oat-100) !important;
        border-color: var(--green-800) !important;
        box-shadow: 0 4px 14px rgba(15,35,24,0.20) !important;
        transform: translateY(-1px);
    }

    /* ── 表單送出按鈕 ── */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, var(--green-900), var(--green-700)) !important;
        color: #d4e8da !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem !important;
        letter-spacing: 0.2px;
        transition: all 0.18s ease !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, var(--green-800), var(--green-600)) !important;
        box-shadow: 0 6px 20px rgba(15,35,24,0.35) !important;
        transform: translateY(-1px);
    }

    /* ── Alert 卡片 ── */
    .stAlert { border-radius: 12px !important; }

    /* ── 聊天訊息 ── */
    [data-testid="stChatMessage"] {
        border-radius: 12px !important;
        background: var(--oat-100) !important;
        border: 1px solid var(--oat-300) !important;
    }

    /* ── 分隔線 ── */
    hr { border-color: var(--oat-300) !important; margin: 0.8rem 0 !important; }

    /* ── 登入頁置中 ── */
    .login-wrapper {
        max-width: 440px;
        margin: 0 auto;
        padding: 2.5rem 2rem;
        background: var(--oat-100);
        border-radius: 16px;
        border: 1px solid var(--oat-300);
        box-shadow: 0 4px 24px rgba(15,35,24,0.10);
    }

    /* ── Landing 身份卡片 ── */
    .role-card {
        background: var(--oat-100);
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 12px rgba(15,35,24,0.07);
        border: 2px solid transparent;
        transition: border-color 0.18s, box-shadow 0.18s;
        height: 100%;
    }
    .role-card:hover {
        border-color: var(--accent-lt);
        box-shadow: 0 6px 24px rgba(15,35,24,0.13);
    }
    .role-card h3 { margin-top: 0; font-size: 1.15rem; color: var(--green-800) !important; }
    .role-card p { color: var(--oat-600); font-size: 0.92rem; line-height: 1.6; margin-bottom: 1.2rem; }

    /* ── 使用者 / AI 訊息泡泡區別 ── */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: #eef7f0 !important;
        border-left: 3px solid var(--green-600) !important;
    }
    [data-testid="stChatMessage"][data-testid*="assistant"] {
        background: var(--oat-100) !important;
        border-left: 3px solid var(--accent-lt) !important;
    }

    /* ── 聊天輸入框 ── */
    [data-testid="stChatInput"] textarea {
        background: var(--oat-100) !important;
        border: 1.5px solid var(--oat-300) !important;
        border-radius: 10px !important;
        color: var(--oat-900) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--green-600) !important;
        box-shadow: 0 0 0 3px rgba(45,106,63,0.10) !important;
    }

    /* ── expander（案例卡片）── */
    [data-testid="stExpander"] {
        background: var(--oat-100) !important;
        border: 1px solid var(--oat-300) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--green-800) !important;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)


def render_language_switcher():
    with st.sidebar:
        st.selectbox("🌐 Language / 語言", list(LANGUAGES.keys()), key="ui_lang")


# ==========================================
# 3. Numpy 向量索引（@st.cache_resource 只載入一次）
# ==========================================
@st.cache_resource(show_spinner="正在載入模型與索引…")
def _load_artifacts():
    """Model + precomputed vectors from backend/build_index.py (loaded once).

    Nothing is encoded at startup: vectors are built offline, so a cold
    start only loads the model and two .npy files.
    Returns dict(model, law/cases → (emb float32, texts, metas, version)).
    """
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer

    emb_dir = os.path.join(base_dir, "backend", "embeddings")

    def _load(prefix):
        npy_path = os.path.join(emb_dir, f"{prefix}_embeddings{_MODEL_SUFFIX}.npy")
        doc_path = os.path.join(emb_dir, f"{prefix}_docs{_MODEL_SUFFIX}.json")
        if not (os.path.exists(npy_path) and os.path.exists(doc_path)):
            raise FileNotFoundError(f"索引檔不存在：{npy_path}\n請先執行：python backend/build_index.py")
        emb = np.load(npy_path).astype(np.float32)   # stored as float16
        with open(doc_path, encoding="utf-8") as f:
            data = json.load(f)
        if len(emb) != len(data["texts"]):
            raise ValueError(f"{prefix}: 向量數 {len(emb)} 與文字數 {len(data['texts'])} 不符，請重跑 build_index.py")
        return emb, data["texts"], data["metas"], data.get("corpus_version", str(len(emb)))

    t0 = time.perf_counter()
    model = SentenceTransformer(_MODEL_NAME)
    load_time = time.perf_counter() - t0
    out = {"model": model, "law": _load("law"), "cases": _load("cases")}
    print(f"[MODEL] {_MODEL_NAME} load={load_time:.1f}s | "
          f"[INDEX] law={len(out['law'][1])} cases={len(out['cases'][1])}", flush=True)
    return out


@st.cache_resource(show_spinner="正在載入索引...")
def _load_index():
    a = _load_artifacts()
    (law_emb, law_texts, law_metas, _), (case_emb, case_texts, case_metas, _) = a["law"], a["cases"]
    return a["model"], law_emb, law_texts, law_metas, case_emb, case_texts, case_metas


# ── ChromaDB helpers ───────────────────────────────────────────────────────

def _sanitize_meta(embed_text: str, meta: dict) -> dict:
    """Chroma metadata accepts only str/int/float/bool: drop None, join lists.
    display_text is stored as the Chroma document instead."""
    out = {"embed_text": embed_text}
    for k, v in meta.items():
        if k == "display_text" or v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = "、".join(map(str, v))
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return out


def _populate_collection(client, name, emb, texts, metas, version):
    """(Re)create a collection from precomputed vectors — no encoding."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine", "corpus_version": version})
    documents = [m.get("display_text", t) for t, m in zip(texts, metas)]
    metadatas = [_sanitize_meta(t, m) for t, m in zip(texts, metas)]
    for s in range(0, len(texts), 1000):
        col.add(ids=[f"{name}_{i}" for i in range(s, min(s + 1000, len(texts)))],
                embeddings=emb[s:s + 1000].tolist(),
                documents=documents[s:s + 1000], metadatas=metadatas[s:s + 1000])
    return col


@st.cache_resource(show_spinner="正在準備索引…")
def _load_chroma_index():
    """Chroma collections filled from the precomputed vectors.

    Rebuilt only when missing or when corpus_version (content hash written by
    build_index.py) changed; that takes seconds since nothing is re-encoded.
    Returns (model, law_col, cases_col, law_texts, law_metas, case_texts,
    case_metas), or None to signal the NumPy fallback.
    """
    try:
        import chromadb
        a = _load_artifacts()
        client = chromadb.PersistentClient(path=_CHROMA_DB_PATH)
        cols = {}
        for kind, name in (("law", _LAW_COL_NAME), ("cases", _CASES_COL_NAME)):
            emb, texts, metas, version = a[kind]
            try:
                col = client.get_collection(name)
                fresh = (col.metadata or {}).get("corpus_version") == version and col.count() == len(texts)
            except Exception:
                col, fresh = None, False
            if not fresh:
                t0 = time.perf_counter()
                col = _populate_collection(client, name, emb, texts, metas, version)
                print(f"[CHROMA] {name}: 由預算向量重建 {col.count()} 筆，{time.perf_counter() - t0:.1f}s", flush=True)
            cols[kind] = col
        law, cases = a["law"], a["cases"]
        print(f"[CHROMA] 索引就緒：law={cols['law'].count()}, cases={cols['cases'].count()}", flush=True)
        return a["model"], cols["law"], cols["cases"], law[1], law[2], cases[1], cases[2]
    except Exception as e:
        print(f"[CHROMA] 索引失敗，改用 NumPy：{e}", flush=True)
        return None


def _search(model, query, embeddings, texts, metas, top_k=5, _top_k_chunks=20):
    import numpy as np
    q_emb = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:_top_k_chunks]

    # 以 parent_id 合併：同一份文件只保留分數最高的那塊
    seen = {}
    for i in top_idx:
        pid = metas[i].get("parent_id", str(i))
        if pid not in seen or scores[i] > seen[pid][1]:
            seen[pid] = (i, float(scores[i]))

    merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(texts[i], metas[i], score) for i, score in merged]


def _chroma_search(model, collection, query, top_k=5, _top_k_chunks=20):
    """ChromaDB equivalent of _search(); returns [(text, meta, score)] in the same format."""
    q_emb = model.encode([query], normalize_embeddings=True)
    results = collection.query(
        query_embeddings=[q_emb[0].tolist()],
        n_results=_top_k_chunks,
        include=["documents", "metadatas", "distances"],
    )
    docs      = results["documents"][0]
    metas_raw = results["metadatas"][0]
    distances = results["distances"][0]

    # cosine distance ∈ [0, 2]; score = 1 - dist (0→identical, 1→orthogonal)
    seen = {}  # parent_id → (text, meta, score)
    for doc, raw_meta, dist in zip(docs, metas_raw, distances):
        score = 1.0 - dist
        pid   = raw_meta.get("parent_id", "")
        if pid not in seen or score > seen[pid][2]:
            meta = dict(raw_meta)
            meta["display_text"] = doc       # restore display_text stored in documents
            text = meta.pop("embed_text", doc)
            seen[pid] = (text, meta, score)

    ranked = sorted(seen.values(), key=lambda x: x[2], reverse=True)
    return ranked[:top_k]


@st.cache_resource(show_spinner="正在建立關鍵字索引…")
def _load_bm25(kind: str, backend: str, _texts: list, _metas: list):
    """BM25 index per corpus. kind/backend form the cache key; the
    underscore-prefixed lists are skipped by Streamlit's hasher."""
    from hybrid_search import BM25Index
    t0 = time.perf_counter()
    idx = BM25Index(_texts, _metas)
    print(f"[BM25] {kind}/{backend}: {len(_texts)} chunks in {time.perf_counter() - t0:.1f}s", flush=True)
    return idx


@st.cache_resource
def _penalty_index(backend: str, _texts: list, _metas: list):
    """(article → penalty articles map, (law, article) → (text, meta) of chunk 0)."""
    pmap = build_penalty_map(_texts, _metas)
    first = {}
    for t, m in zip(_texts, _metas):
        first.setdefault((m.get("source"), m.get("article_no")), (t, m))
    print(f"[PENALTY] {backend}: {len(pmap)} articles mapped to penalty provisions", flush=True)
    return pmap, first


_PENALTY_KW = ("罰", "違法", "違反", "檢舉", "申訴")


def _wants_penalty(query: str, violation: bool | None) -> bool:
    """Employers always get the penalty exposure (it's their legal risk);
    workers when the question is about a violation or asks about penalties."""
    if st.session_state.get("user_role") == "Employer":
        return True
    return bool(violation) or any(k in query for k in _PENALTY_KW)


def _retrieve(vector_fn, kind: str, backend: str, query: str, texts: list, metas: list,
              top_k: int = 5, allow=None, cite_query: str | None = None):
    """Vector search, optionally fused with BM25 via RRF.

    vector_fn(top_k, top_k_chunks) -> [(text, meta, score)]
    allow(meta) -> bool restricts results (e.g. to the laws the intent names).
    cite_query: text scanned for explicit article citations (default *query*).
        Pass the user's own question when *query* carries LLM search terms,
        so a model-written article number can't trigger the citation boost.
    """
    pool = 30

    def _vector_only():
        if allow is None:
            return vector_fn(top_k, 20)
        return [r for r in vector_fn(pool, pool) if allow(r[1])][:top_k]

    if not HYBRID_SEARCH:
        return _vector_only()
    try:
        from hybrid_search import rrf_merge
        vec = vector_fn(pool, pool)
        bm25_idx = _load_bm25(kind, backend, texts, metas).search(query, top_n=pool)
        return rrf_merge(vec, bm25_idx, texts, metas, top_k=top_k,
                         query=cite_query or query, allow=allow)
    except Exception as e:  # keyword layer must never break answering
        print(f"[BM25] fallback to vector-only: {e}", flush=True)
        return _vector_only()


# Force-include rules, constants, and helper functions live in force_rules.py
# so eval/test_force_rules.py can import the real implementation without
# pulling in Streamlit / Gemini / numpy.
from force_rules import (
    FORCE_ARTICLE_RULES,
    MAX_FORCED_LAW_RESULTS,
    RESTDAY_OVERTIME_FORCE_TITLES,
    _force_include_articles,
    _force_include_cases,
)
from referral_text import REFERRAL_BLOCK
from prompt_helpers import (
    OUT_OF_SCOPE_TEXT,
    build_system_prompt,
    build_lang_rule,
    build_missing_note,
    build_user_prompt,
    build_case_ctx,
    stream_reply,
)
from intent_router import make_plan, heuristic_intent, detect_laws, INTENTS
from eval_holdout import case_filter
from penalties import build_penalty_map, penalties_for, label_paragraphs

PENALTY_POOL = 10   # law results searched for penalty provisions when asked about 罰則

# None in the live app; with EVAL_HOLDOUT=1 the test set's own FAQ answers
# are excluded from case retrieval (see eval_holdout.py).
_CASE_ALLOW = case_filter()
if _CASE_ALLOW is not None:
    print("[EVAL] EVAL_HOLDOUT on: 台北 FAQ test questions excluded from case retrieval", flush=True)


# ==========================================
# 3b. 意圖判斷 + 情境追問輔助函式
# ==========================================
import json as _json
import re as _re

# Retry only when the stream fails before any text was shown to the user
CLAUDE_MAX_ATTEMPTS = 2
CLAUDE_MODEL = "claude-haiku-5-5"


@st.cache_resource
def _get_anthropic_client(api_key: str):
    """Shared Anthropic client — cached to avoid repeated TLS setup."""
    import anthropic as _ant
    return _ant.Anthropic(api_key=api_key)


# Haiku 5.5 thinks by default, so content[0] may be a ThinkingBlock — only
# text blocks carry the answer.
def _response_text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text").strip()


# Short utility calls (intent JSON, translation) don't benefit from thinking;
# turning it off keeps them fast. Allowed on Haiku 5.5 at effort high or below.
_NO_THINKING = {"type": "disabled"}


def _claude_json_call(prompt_text: str, _label: str = "intent", _attempts: int = 2) -> dict | None:
    """Fire a Claude call and return parsed JSON dict, or None on any failure.

    Retries once on an unparsable reply (rarely the model returns an empty
    ```json``` fence); callers fall back to keyword heuristics on None.
    """
    for _attempt in range(_attempts):
        result = _claude_json_call_once(prompt_text, _label)
        if result is not None:
            return result
    return None


def _claude_json_call_once(prompt_text: str, _label: str) -> dict | None:
    try:
        api_key = ANTHROPIC_API_KEY
        if not api_key:
            print(f"[INTENT] [{_label}] API key 未設定，略過")
            return None
        client = _get_anthropic_client(api_key)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            thinking=_NO_THINKING,
            messages=[{"role": "user", "content": prompt_text}],
        )
        text = _response_text(resp)
        print(f"[INTENT] [{_label}] Claude 原始回傳: {text!r}")
        m = _re.search(r'\{.*\}', text, _re.DOTALL)
        if m:
            text = m.group()
        result = _json.loads(text)
        print(f"[INTENT] [{_label}] 解析成功")
        return result
    except Exception as _e:
        print(f"[INTENT] [{_label}] 解析失敗: {_e} (cause: {_e.__cause__!r})")
        return None


# ── Conversation history ─────────────────────────────────────────────────────
# Follow-ups like 「那休息日呢？」 only make sense with the earlier turns: the
# intent call rewrites them into a standalone query (translated_query) for
# retrieval, and the answer call sees the recent turns so it stays consistent
# with numbers the user already gave.
HISTORY_TURNS = 3          # user+assistant pairs kept
_HIST_INTENT_CHARS = 300   # per message, in the intent prompt
_HIST_ANSWER_CHARS = 800   # per assistant message, in the answer call


def _recent(history: list | None) -> list[dict]:
    msgs = [m for m in (history or []) if m.get("role") in ("user", "assistant")]
    return msgs[-2 * HISTORY_TURNS:]


def _clip(text, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + "…（略）"


def _history_text(history: list | None) -> str:
    """Recent turns as plain text for the intent prompt."""
    lines = [f"{'使用者' if m['role'] == 'user' else '助理'}：{_clip(m['content'], _HIST_INTENT_CHARS)}"
             for m in _recent(history)]
    return "\n".join(lines) or "（無）"


def _history_messages(history: list | None) -> list[dict]:
    """Recent turns as API messages: must start with a user turn and alternate."""
    out: list[dict] = []
    for m in _recent(history):
        if not out and m["role"] != "user":
            continue
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n" + str(m["content"])   # merge same-role runs
            continue
        out.append({"role": m["role"], "content": str(m["content"])})
    if out and out[-1]["role"] == "user":                    # next message is ours
        out.pop()
    for m in out:
        if m["role"] == "assistant":
            m["content"] = _clip(m["content"], _HIST_ANSWER_CHARS)
    return out


def _analyze_intent(user_message: str, ui_lang: str, history: list | None = None,
                    clarify_allowed: bool = True) -> dict | None:
    """One Claude call: standalone query, intent, laws, search terms, violation,
    and whether to ask the user for missing facts first (clarify).

    *history* (earlier chat messages) lets the model rewrite a follow-up — or
    the user's answers to a clarifying question — into a standalone query.
    Returns None on failure (callers fall back to keyword heuristics).
    """
    print(f"[INTENT] _analyze_intent 呼叫 | 問句: {user_message!r}")
    prompt = INTENT_ANALYSIS_PROMPT.format(
        history=_history_text(history),
        user_message=user_message,
        ui_lang=ui_lang,
        clarify_allowed="可以" if clarify_allowed else "不可以（已問過，直接依現有資訊回答）",
    )
    result = _claude_json_call(prompt, _label="analyze_intent")
    if result:
        print(f"[INTENT]   query        = {result.get('translated_query')!r}")
        print(f"[INTENT]   followup={result.get('is_followup')} intent={result.get('intent')!r} "
              f"violation={result.get('violation')} clarify={result.get('clarify')}")
    else:
        print("[INTENT] _analyze_intent: 結果為 None（解析失敗）")
    return result


class ClarifyReply(str):
    """An assistant reply that asks for missing facts before answering.

    A str, so callers that display or save replies keep working; it also
    carries the structured questions, which the chat UI renders as option
    buttons under the message.
    """

    def __new__(cls, text: str, questions: list[dict]):
        obj = super().__new__(cls, text)
        obj.questions = questions
        return obj


def _clean_questions(raw) -> list[dict]:
    """Validate the model's clarify.questions: [{"q": str, "options": [str]}]."""
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not str(item.get("q") or "").strip():
            continue
        opts = [str(o).strip() for o in (item.get("options") or []) if str(o).strip()]
        out.append({"q": str(item["q"]).strip(), "options": opts[:4]})
    return out[:MAX_CLARIFY_QUESTIONS]


def _clarify_text(questions: list[dict], lang: str) -> str:
    lines = [t(lang, "clarify_lead"), ""]
    lines += [f"{n}. {q['q']}" for n, q in enumerate(questions, 1)]
    lines += ["", t(lang, "clarify_hint")]
    return "\n".join(lines)


def _is_skip(text: str) -> bool:
    return len(text) <= SKIP_MAX_LEN and any(kw in text for kw in SKIP_FOLLOWUP_TRIGGERS)


def _handle_prompt(prompt: str, persona: str, lang: str) -> str | Iterator[str]:
    """Route a user message: ask for missing facts first, or answer.

    Flow (design doc step 3, generalised from the old overtime-only follow-up):
      - The intent call decides whether facts the user hasn't given would
        change the answer. If so, reply with up to 3 clarifying questions
        (ClarifyReply, shown with option buttons) instead of answering.
      - The user's reply goes through the same call; with the conversation
        history it is rewritten into one complete question (original question
        + answers). At most MAX_FOLLOWUP_ROUNDS rounds.
      - Skipping (「直接回答」) or running out of rounds gives a general answer
        that says which facts would change it.
    """
    msgs = st.session_state.messages
    history = msgs[:-1] if msgs and msgs[-1].get("role") == "user" else list(msgs)
    ctx = st.session_state.intent_ctx   # set while we're waiting for clarifying answers
    print(f"[INTENT] ── _handle_prompt ── {prompt!r} | 補問中={'是' if ctx else '否'}")

    if not ENABLE_INTENT_DETECTION:
        st.session_state.intent_ctx = None
        intent, laws = heuristic_intent(prompt), detect_laws(prompt)
        print(f"[INTENT] 功能關閉 → 關鍵字分類 intent={intent!r} laws={laws}")
        return query_rag_system(prompt, persona, lang, intent=intent, laws=laws, history=history)

    if ctx and _is_skip(prompt):
        print("[INTENT] 使用者跳過補問 → 一般說明")
        st.session_state.intent_ctx = None
        return _answer(ctx["query"], ctx["analysis"], persona, lang, history,
                       unanswered=ctx["questions"])

    rounds = ctx["rounds"] if ctx else 0
    analysis = _analyze_intent(prompt, lang, history,
                               clarify_allowed=rounds < MAX_FOLLOWUP_ROUNDS)
    if analysis is None:                  # LLM call / JSON failed → keyword fallback
        st.session_state.intent_ctx = None
        intent, laws = heuristic_intent(prompt), detect_laws(prompt)
        return query_rag_system(prompt, persona, lang, intent=intent, laws=laws, history=history)

    if ctx and not analysis.get("is_followup"):
        rounds = 0                        # the user moved on to a new question
    query = analysis.get("translated_query") or prompt
    cl = analysis.get("clarify") or {}
    questions = _clean_questions(cl.get("questions"))
    needs = bool(cl.get("needed")) and bool(questions) and analysis.get("intent") == "solve_problem"

    if needs and rounds < MAX_FOLLOWUP_ROUNDS:
        print(f"[INTENT] 分支: 補問 (round={rounds + 1}) | {[q['q'] for q in questions]}")
        st.session_state.intent_ctx = {"query": query, "rounds": rounds + 1,
                                       "questions": questions, "analysis": analysis}
        st.session_state["_pending_cases"] = None
        return ClarifyReply(_clarify_text(questions, lang), questions)

    st.session_state.intent_ctx = None
    return _answer(query, analysis, persona, lang, history,
                   unanswered=questions if needs else [])


def _answer(query: str, analysis: dict, persona: str, lang: str, history: list,
            unanswered: list[dict] | None = None):
    """Retrieve + generate for a standalone *query*. *unanswered* clarifying
    questions switch the answer to the general format and are listed as the
    information still needed."""
    intent = analysis.get("intent") if analysis.get("intent") in INTENTS else heuristic_intent(query)
    laws = analysis.get("laws") or detect_laws(query)
    terms = [x for x in (analysis.get("search_terms") or []) if isinstance(x, str)]
    missing = [q["q"] for q in (unanswered or [])]
    print(f"[INTENT] 分支: 回答 | intent={intent!r} laws={laws} general={bool(missing)}")
    return query_rag_system(query, persona, lang, search_query_override=query,
                            intent=intent, laws=laws, search_terms=terms,
                            violation=analysis.get("violation"), history=history,
                            missing_fields=missing or None, general=bool(missing))


# ==========================================
# 4. 後端 RAG 引擎 (Gemini + Numpy 搜尋)
# ==========================================
_rag_logger = logging.getLogger("rag_timing")
if not _rag_logger.handlers:
    _rag_log_path = os.path.join(base_dir, "rag_timing.log")
    _rag_handler = logging.FileHandler(_rag_log_path, encoding="utf-8")
    _rag_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    _rag_logger.addHandler(_rag_handler)
    _rag_logger.setLevel(logging.INFO)


def query_rag_system(
    user_prompt: str,
    system_prompt: str,
    ui_lang: str,
    *,
    search_query_override: str | None = None,
    missing_fields: list[str] | None = None,
    intent: str = "solve_problem",
    laws: list[str] | None = None,
    search_terms: list[str] | None = None,
    history: list | None = None,
    violation: bool | None = None,
    general: bool = False,
) -> str | Iterator[str]:
    """Run retrieval eagerly, then return a generator that streams the answer.

    Returns a plain string for early errors (missing API key, retrieval failure)
    and for out-of-scope questions (fixed referral text, no retrieval/LLM).

    intent / laws: from _analyze_intent (or keyword fallback); they pick the
        corpora, result counts, whether cases enter the prompt, and the answer
        format — see intent_router.make_plan.

    search_terms: statutory wording from _analyze_intent, appended to the law
        search query (vector + BM25). Bridges colloquial questions to the
        article text: Hit@5 73% → 77% mean (74-80% over 10 sampled runs) on the 台北 FAQ set (eval_retrieval.py).
    history: earlier chat messages; the last few turns go into the answer
        call so follow-ups stay consistent (retrieval uses the rewritten
        standalone query from _analyze_intent instead).
    violation: intent call's judgement that the question involves a possible
        violation; decides (with the user role) whether penalty articles for
        the retrieved articles are added — see penalties.py.
    search_query_override: pre-translated Chinese query from _analyze_intent;
        if provided, the internal translation call is skipped.
    missing_fields: facts the user didn't give (the unanswered clarifying
        questions); listed in the prompt so the model doesn't assume them.
    general: the user skipped the clarifying questions; a consult answer uses
        the 「一般說明」 format (no 結論 the facts can't support).
    """
    plan = make_plan(intent, laws)
    if plan is None:
        _rag_logger.info(f"q={user_prompt[:25]!r} intent=out_of_scope → fixed referral, no retrieval")
        st.session_state["_pending_cases"] = None
        return OUT_OF_SCOPE_TEXT

    try:
        api_key = ANTHROPIC_API_KEY
        if not api_key:
            return t(ui_lang, "err_no_api_key")

        _t0 = time.perf_counter()
        client = _get_anthropic_client(api_key)

        _use_chroma = VECTOR_BACKEND == "chroma"
        if _use_chroma:
            _chroma_res = _load_chroma_index()
            if _chroma_res is None:
                _use_chroma = False  # fallback: chroma failed both times
            else:
                model, law_col, cases_col, law_texts, law_metas, case_texts, case_metas = _chroma_res
        if not _use_chroma:
            model, law_emb, law_texts, law_metas, case_emb, case_texts, case_metas = _load_index()
        _t_load = time.perf_counter()

        def _is_chinese(text: str) -> bool:
            chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
            return chinese_chars / max(len(text), 1) >= 0.3

        if search_query_override is not None:
            # Translation already done by _analyze_intent — skip the extra call
            search_query = search_query_override
        else:
            search_query = user_prompt
            if not _is_chinese(user_prompt):
                try:
                    search_query = _response_text(client.messages.create(
                        model=CLAUDE_MODEL,
                        max_tokens=512,
                        thinking=_NO_THINKING,
                        messages=[{"role": "user", "content": f"請將以下使用者問題翻譯成繁體中文，只需要輸出翻譯結果，不要加任何說明：\n\n{user_prompt}"}],
                    )) or user_prompt
                except Exception:
                    search_query = user_prompt
        _t_trans = time.perf_counter()

        _backend = "chroma" if _use_chroma else "numpy"
        law_query = " ".join([search_query, *(search_terms or [])])
        if _use_chroma:
            _law_vec = lambda k, kc: _chroma_search(model, law_col, law_query, top_k=k, _top_k_chunks=kc)
        else:
            _law_vec = lambda k, kc: _search(model, law_query, law_emb, law_texts, law_metas, top_k=k, _top_k_chunks=kc)
        # A few extra results are kept only for the penalty lookup below.
        l_pool = _retrieve(_law_vec, "law", _backend, law_query, law_texts, law_metas,
                           top_k=max(plan.law_k, PENALTY_POOL), cite_query=search_query)
        l_results = l_pool[:plan.law_k]
        _t_law_search = time.perf_counter()

        fired_mutex: set = set()
        for rule in (FORCE_ARTICLE_RULES if plan.force_rules else []):
            if rule["mutex"] and rule["mutex"] in fired_mutex:
                continue
            if any(k in search_query for k in rule["keywords"]):
                l_results = _force_include_articles(
                    search_query, l_results, law_texts, law_metas,
                    rule["keywords"], rule["articles"]
                )
                if rule["mutex"]:
                    fired_mutex.add(rule["mutex"])
        l_results = l_results[:MAX_FORCED_LAW_RESULTS]

        # Penalty provisions cite articles only by number, so retrieval rarely
        # finds them; look them up from the retrieved articles instead.
        pen_results = []
        if _wants_penalty(search_query, violation):
            pmap, first = _penalty_index(_backend, law_texts, law_metas)
            # Asked about penalties outright: the violated article may sit just
            # below the top results (e.g. 勞基法 §70 for 「不訂工作規則有罰則嗎」).
            asks = any(k in search_query for k in _PENALTY_KW)
            src = l_results + (l_pool[plan.law_k:] if asks else [])
            pen_results = [first[k] for k in penalties_for(src, pmap) if k in first]
        _t_force_l = time.perf_counter()

        if _use_chroma:
            _case_vec = lambda k, kc: _chroma_search(model, cases_col, search_query, top_k=k, _top_k_chunks=kc)
        else:
            _case_vec = lambda k, kc: _search(model, search_query, case_emb, case_texts, case_metas, top_k=k, _top_k_chunks=kc)
        c_results = _retrieve(_case_vec, "cases", _backend, search_query, case_texts, case_metas,
                              top_k=plan.case_k, allow=_CASE_ALLOW)
        _t_case_search = time.perf_counter()

        _overtime_kw = next(r["keywords"] for r in FORCE_ARTICLE_RULES if r["name"] == "加班延長工時")
        c_results = _force_include_cases(
            search_query, c_results, case_texts, case_metas, _overtime_kw, RESTDAY_OVERTIME_FORCE_TITLES
        )
        _t_force_c = time.perf_counter()

        law_ctx = ""
        for d, m, _ in l_results:
            display = m.get("display_text", d)
            law_ctx += f"【法規內容】：{display}\n【條號】：{m.get('source', '')} {m.get('article_no', '')}\n\n"
        for d, m in pen_results:
            display = label_paragraphs(m.get("display_text", d))
            law_ctx += (f"【罰則條文】（違反上列條文時的罰鍰）：{display}\n"
                        f"【條號】：{m.get('source', '')} {m.get('article_no', '')}\n\n")

        # Cases go into the prompt per the plan (find_cases: all 5, consult: top 2,
        # law lookup: none); the full list is also kept for the "查看相似案例" button.
        case_ctx = build_case_ctx(c_results[:plan.cases_in_prompt])
        st.session_state["_pending_cases"] = c_results
        st.session_state["_show_cases"]    = False

        # 回答語言邏輯：永遠跟著 UI 語言選項走。
        lang_rule = build_lang_rule(ui_lang, LANGUAGES.get(ui_lang, "Traditional Chinese"))

        # Facts still missing after the clarifying round(s), so the model
        # won't invent numbers or pick one scenario.
        missing_note = build_missing_note("；".join(missing_fields or []))
        mode = "general" if general and plan.mode == "consult" else plan.mode

        final_prompt = build_user_prompt(law_ctx, user_prompt, lang_rule, missing_note,
                                         case_ctx=case_ctx, mode=mode)
        _t_prompt = time.perf_counter()

    except Exception as e:
        return f"🚨 異常：{str(e)}"

    _stats = {"t_first": None, "usage": None, "retries": 0}

    def _raw() -> Iterator[str]:
        """Yield raw Claude text chunks, recording first-token time and usage.

        Retries a failed/timed-out request only if no text was yielded yet;
        once the user has seen part of the answer, the error is re-raised.
        """
        for attempt in range(1, CLAUDE_MAX_ATTEMPTS + 1):
            emitted = False
            try:
                with client.messages.stream(
                    model=CLAUDE_MODEL,
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[*_history_messages(history),
                              {"role": "user", "content": final_prompt}],
                ) as stream:
                    for text_chunk in stream.text_stream:
                        if text_chunk:
                            if _stats["t_first"] is None:
                                _stats["t_first"] = time.perf_counter()
                            emitted = True
                            yield text_chunk
                    final_msg = stream.get_final_message()
                    if final_msg:
                        _stats["usage"] = final_msg.usage
                return
            except Exception as e:
                if emitted or attempt == CLAUDE_MAX_ATTEMPTS:
                    raise
                _stats["retries"] += 1
                print(f"[CLAUDE] 第 {attempt} 次呼叫失敗，重試：{e}", flush=True)

    def _stream() -> Iterator[str]:
        """Stream the <answer> body; OUT scope gets the fixed referral text instead."""
        is_out = False
        try:
            for piece in stream_reply(_raw()):
                # stream_reply yields the OUT_OF_SCOPE_TEXT object itself for OUT scope
                is_out = is_out or piece is OUT_OF_SCOPE_TEXT
                yield piece
        except Exception as e:
            _rag_logger.info(
                f"q={user_prompt[:25]!r} ERROR after {time.perf_counter()-_t_prompt:.3f}s "
                f"retries={_stats['retries']} shown={_stats['t_first'] is not None} err={e!r}"
            )
            yield f"\n\n🚨 回答中斷（{type(e).__name__}），請再問一次。"
            return

        _t_claude = time.perf_counter()
        _t_first  = _stats["t_first"]
        _usage    = _stats["usage"]

        # Token usage (Anthropic SDK: usage.input_tokens / output_tokens)
        _input_tok  = int(getattr(_usage, "input_tokens",  0) or 0)
        _output_tok = int(getattr(_usage, "output_tokens", 0) or 0)

        _rag_logger.info(
            f"q={user_prompt[:25]!r} zh={_is_chinese(user_prompt)} "
            f"load={_t_load-_t0:.3f}s "
            f"trans={_t_trans-_t_load:.3f}s "
            f"law_search={_t_law_search-_t_trans:.3f}s "
            f"force_l={_t_force_l-_t_law_search:.3f}s "
            f"case_search={_t_case_search-_t_force_l:.3f}s "
            f"force_c={_t_force_c-_t_case_search:.3f}s "
            f"prompt={_t_prompt-_t_force_c:.3f}s "
            f"ttft={(_t_first or _t_claude)-_t_prompt:.3f}s "
            f"claude={_t_claude-_t_prompt:.3f}s "
            f"total={_t_claude-_t0:.3f}s "
            f"prompt_chars={len(final_prompt)} "
            f"input_tok={_input_tok} output_tok={_output_tok} "
            f"scope={'OUT' if is_out else 'IN'} retries={_stats['retries']} "
            f"intent={plan.intent} laws={','.join(plan.laws)}"
        )

        # OUT_OF_SCOPE_TEXT already lists referral channels
        if not is_out:
            yield REFERRAL_BLOCK

    return _stream()


# ==========================================
# 4. 頁面一：登入 / 註冊
# ==========================================
def show_login_page():
    render_language_switcher()
    lang = st.session_state.ui_lang

    st.markdown("<br><br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown(
            "<div style='"
            "background:#faf8f4;"
            "border:1px solid #e6dece;"
            "border-radius:16px;"
            "padding:2.4rem 2.2rem 2rem;"
            "box-shadow:0 4px 24px rgba(15,35,24,0.09);"
            "'>"
            "<div style='text-align:center; margin-bottom:0.8rem;'>"
            "<span style='font-size:2.6rem; line-height:1;'>⚖️</span>"
            "</div>"
            f"<h2 style='text-align:center; color:#1a3a28; font-size:1.4rem;"
            f" font-weight:700; margin-bottom:0.25rem; letter-spacing:-0.2px;'>"
            f"{t(lang, 'app_title').replace('⚖️ ', '')}</h2>"
            f"<p style='text-align:center; color:#7a6e58; margin-bottom:1.6rem;"
            f" font-size:0.9rem;'>{t(lang, 'login_subheader')}</p>",
            unsafe_allow_html=True,
        )

        tab_login, tab_register = st.tabs([t(lang, "tab_login"), t(lang, "tab_register")])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input(t(lang, "field_username"))
                password = st.text_input(t(lang, "field_password"), type="password")
                if st.form_submit_button(t(lang, "btn_login"), use_container_width=True):
                    if not username or not password:
                        st.error(t(lang, "err_empty_credentials"))
                    else:
                        ok, result = verify_user(username.strip(), password)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.username  = result
                            st.session_state.page      = "Landing"
                            st.rerun()
                        else:
                            st.error(t(lang, DB_MSG_KEYS.get(result, result)))

        with tab_register:
            with st.form("register_form"):
                new_user  = st.text_input(t(lang, "field_username"), key="reg_user")
                new_pass  = st.text_input(t(lang, "field_password_hint"), type="password")
                new_pass2 = st.text_input(t(lang, "field_password_confirm"), type="password")
                if st.form_submit_button(t(lang, "btn_register"), use_container_width=True):
                    if not new_user or not new_pass:
                        st.error(t(lang, "err_empty_credentials"))
                    elif len(new_pass) < 6:
                        st.error(t(lang, "err_password_too_short"))
                    elif new_pass != new_pass2:
                        st.error(t(lang, "err_password_mismatch"))
                    else:
                        ok, msg = register_user(new_user.strip(), new_pass)
                        localized_msg = t(lang, DB_MSG_KEYS.get(msg, msg))
                        if ok:
                            st.success(localized_msg + t(lang, "msg_register_success_suffix"))
                        else:
                            st.error(localized_msg)

        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# 5. 頁面二：身份選擇 (Landing Page)
# ==========================================
def show_landing_page():
    render_language_switcher()
    lang = st.session_state.ui_lang

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center; padding:0.5rem 0 1.8rem;'>"
        "<div style='display:inline-block; background:#1a3a28; border-radius:50%;"
        " width:64px; height:64px; line-height:64px; font-size:2rem; margin-bottom:1rem;'>⚖️</div>"
        "<h1 style='color:#1a3a28; font-size:1.9rem; font-weight:700; margin-bottom:0.35rem;'>"
        + t(lang, "app_title").replace("⚖️ ", "") +
        "</h1>"
        f"<p style='color:#2d6a3f; font-size:1.05rem; font-weight:500; margin-bottom:0.3rem;'>"
        f"{t(lang, 'landing_subheader')}</p>"
        f"<p style='color:#7a6e58; font-size:0.9rem;'>{t(lang, 'landing_desc')}</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 角色卡 ────────────────────────────────────────────────────────────────
    _, c1, _, c2, _ = st.columns([0.8, 3, 0.4, 3, 0.8])

    with c1:
        st.markdown(
            "<div class='role-card'>"
            "<div style='font-size:2rem; margin-bottom:0.7rem;'>🙋</div>"
            "<h3 style='color:#1a3a28; font-size:1.15rem; margin-bottom:0.5rem;'>"
            + t(lang, "role_label_employee") +
            "</h3>"
            f"<p style='color:#7a6e58; font-size:0.9rem; line-height:1.65; margin-bottom:1.2rem;'>"
            f"{t(lang, 'role_employee_desc')}</p>"
            "<div style='height:3px; background:linear-gradient(90deg,#2d6a3f,#c9a84c);"
            " border-radius:2px; margin-bottom:0.8rem;'></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button(t(lang, "role_employee_btn"), use_container_width=True, key="emp_btn"):
            st.session_state.user_role        = "Employee"
            st.session_state.page             = "Chat"
            st.session_state.messages         = []
            st.session_state.intent_ctx = None   # drop a pending clarification
            st.session_state.current_chat_id  = None
            st.rerun()

    with c2:
        st.markdown(
            "<div class='role-card'>"
            "<div style='font-size:2rem; margin-bottom:0.7rem;'>🏢</div>"
            "<h3 style='color:#1a3a28; font-size:1.15rem; margin-bottom:0.5rem;'>"
            + t(lang, "role_label_employer") +
            "</h3>"
            f"<p style='color:#7a6e58; font-size:0.9rem; line-height:1.65; margin-bottom:1.2rem;'>"
            f"{t(lang, 'role_employer_desc')}</p>"
            "<div style='height:3px; background:linear-gradient(90deg,#2d6a3f,#c9a84c);"
            " border-radius:2px; margin-bottom:0.8rem;'></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button(t(lang, "role_employer_btn"), use_container_width=True, key="er_btn"):
            st.session_state.user_role        = "Employer"
            st.session_state.page             = "Chat"
            st.session_state.messages         = []
            st.session_state.intent_ctx = None   # drop a pending clarification
            st.session_state.current_chat_id  = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center; color:#7a6e58; font-size:0.85rem;'>"
        f"{t(lang, 'landing_caption').format(username=st.session_state.username)}</p>",
        unsafe_allow_html=True,
    )


CLARIFY_SKIP_TEXT = "直接回答"   # matches SKIP_FOLLOWUP_TRIGGERS in every UI language


def _render_clarify_form(questions: list[dict], lang: str, n: str) -> None:
    """Option buttons for the clarifying questions. Submitting sends the picks
    as the user's next message (「問題：選項；…」), which _handle_prompt merges
    with the original question; the user can also just type in the chat box.

    *n* makes the widget keys unique per clarification round: reused keys
    would pre-fill a new round (or a new chat) with the previous picks."""
    with st.form(f"clarify_{n}"):
        picks = []
        for i, q in enumerate(questions):
            if q["options"]:
                v = st.radio(q["q"], q["options"], index=None, horizontal=True, key=f"cq_{n}_{i}")
            else:
                v = st.text_input(q["q"], key=f"cq_{n}_{i}")
            picks.append((q["q"], v))
        extra = st.text_input(t(lang, "clarify_other"), key=f"cq_{n}_x")
        c1, c2 = st.columns(2)
        sent = c1.form_submit_button(t(lang, "clarify_submit"), type="primary")
        skip = c2.form_submit_button(t(lang, "clarify_skip"))
    if skip:
        st.session_state.btn_input = CLARIFY_SKIP_TEXT
        st.rerun()
    if sent:
        parts = [f"{q}：{v}" for q, v in picks if v]
        if extra and extra.strip():
            parts.append(extra.strip())
        if parts:
            st.session_state.btn_input = "；".join(parts)
            st.rerun()


def _render_case_results(c_results: list, lang: str) -> None:
    """Render similar-case cards below the answer. Called only when user clicks the button."""
    st.markdown("---")
    st.markdown("**🔍 相似案例** *(案例僅供參考，每件情況不同)*")
    if not c_results:
        st.info("未找到相似案例。")
        return
    for text, meta, score in c_results:
        title    = meta.get("title") or ""
        source   = meta.get("source") or ""
        category = meta.get("category") or ""
        display  = meta.get("display_text", text)
        summary  = display[:150] + ("…" if len(display) > 150 else "")
        url      = meta.get("url") or ""
        header   = title or f"{source} — {category}"
        with st.expander(f"📄 {header}", expanded=False):
            if source or category:
                st.markdown(f"**類別**：{source}{'  —  ' + category if category else ''}")
            st.markdown(summary)
            if url:
                st.markdown(f"[🔗 查看原文]({url})")


# ==========================================
# 6. 頁面三：AI 聊天室 (Chat Room)
# ==========================================
def show_chat_room():
    render_language_switcher()
    lang = st.session_state.ui_lang

    persona = build_system_prompt(st.session_state.user_role)
    if st.session_state.user_role == "Employee":
        title   = t(lang, "chat_title_employee")
        btns = [
            (t(lang, "qe_btn1"), "我被資遣了！我的權益是什麼？"),
            (t(lang, "qe_btn2"), "加班費怎麼算？"),
            (t(lang, "qe_btn3"), "如何申請勞資調解？"),
            (t(lang, "qe_btn4"), "發生職災我可以申請哪些補償？"),
        ]
    else:
        title   = t(lang, "chat_title_employer")
        btns = [
            (t(lang, "qr_btn1"), "合法資遣流程？"),
            (t(lang, "qr_btn2"), "員工連續曠職怎麼辦？"),
            (t(lang, "qr_btn3"), "如何合法設定工時與薪資？"),
            (t(lang, "qr_btn4"), "如何避免被員工索賠？"),
        ]

    role_label = t(lang, "role_label_employee" if st.session_state.user_role == "Employee" else "role_label_employer")

    # --- 側邊欄 ---
    with st.sidebar:
        # Brand header
        st.markdown(
            "<div style='padding:0.6rem 0 0.2rem; text-align:center;'>"
            "<span style='font-size:1.6rem;'>⚖️</span>"
            f"<p style='color:#d4e8da; font-size:0.78rem; margin:0.3rem 0 0;"
            f" letter-spacing:0.5px; opacity:0.75;'>{t(lang, 'sidebar_title')}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # Account info
        st.markdown(
            f"<p style='color:#a8c8b0; font-size:0.8rem; margin-bottom:0.15rem;'>"
            f"👤 {st.session_state.username}</p>"
            f"<p style='color:#c9a84c; font-size:0.78rem; margin-bottom:0.8rem;"
            f" font-weight:500;'>[ {role_label} ]</p>",
            unsafe_allow_html=True,
        )

        if st.button(t(lang, "btn_new_chat"), use_container_width=True):
            st.session_state.messages        = []
            st.session_state.intent_ctx = None   # drop a pending clarification
            st.session_state.current_chat_id = None
            st.session_state["_pending_cases"] = None
            st.session_state["_show_cases"]    = False
            st.rerun()

        if st.button(t(lang, "btn_switch_role"), use_container_width=True):
            st.session_state.page            = "Landing"
            st.session_state.messages        = []
            st.session_state.intent_ctx = None   # drop a pending clarification
            st.session_state.current_chat_id = None
            st.session_state["_pending_cases"] = None
            st.session_state["_show_cases"]    = False
            st.rerun()

        if st.button(t(lang, "btn_logout"), use_container_width=True):
            for k in ["logged_in", "username", "user_role", "messages", "current_chat_id"]:
                st.session_state[k] = defaults[k]
            st.session_state["_pending_cases"] = None
            st.session_state["_show_cases"]    = False
            st.session_state.page = "Login"
            st.rerun()

        st.divider()

        # Quick questions
        st.markdown(
            f"<p style='color:#a8c8b0; font-size:0.78rem; font-weight:600;"
            f" letter-spacing:0.4px; text-transform:uppercase; margin-bottom:0.5rem;'>"
            f"{t(lang, 'quick_questions_title')}</p>",
            unsafe_allow_html=True,
        )
        for label, value in btns:
            if st.button(label, use_container_width=True):
                st.session_state.btn_input = value

        st.divider()

        # Chat history
        st.markdown(
            f"<p style='color:#a8c8b0; font-size:0.78rem; font-weight:600;"
            f" letter-spacing:0.4px; text-transform:uppercase; margin-bottom:0.5rem;'>"
            f"{t(lang, 'history_title')}</p>",
            unsafe_allow_html=True,
        )
        saved_chats = list_chats(st.session_state.username)
        if not saved_chats:
            st.info(t(lang, "history_empty"))
        else:
            for chat in saved_chats:
                cid = chat["chat_id"]
                updated_at = chat["updated_at"].replace(tzinfo=timezone.utc)
                local_time = updated_at.astimezone(ZoneInfo("Asia/Taipei"))
                time_str = local_time.strftime("%m/%d %H:%M")
                label = chat.get("title") or time_str
                if st.button(label, key=cid, help=time_str, use_container_width=True):
                    st.session_state.messages        = load_chat(st.session_state.username, cid)
                    st.session_state.intent_ctx = None   # drop a pending clarification
                    st.session_state.current_chat_id = cid
                    st.rerun()

    # --- 主聊天區 ---
    st.markdown(
        f"<div style='border-bottom:2px solid #e6dece; padding-bottom:0.6rem; margin-bottom:0.5rem;'>"
        f"<h2 style='color:#1a3a28; font-size:1.35rem; font-weight:700; margin:0;'>"
        f"⚖️ {title}</h2>"
        f"<span style='color:#7a6e58; font-size:0.85rem;'>勞動基準法 AI 顧問 · 勞動部法規資料庫</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 「查看相似案例」按鈕（出現在最後一則 AI 回答之後）──
    _has_cases = bool(st.session_state.get("_pending_cases"))
    _last_is_assistant = (
        bool(st.session_state.messages)
        and st.session_state.messages[-1]["role"] == "assistant"
    )
    if _has_cases and _last_is_assistant:
        if not st.session_state.get("_show_cases"):
            if st.button("🔍 查看相似案例"):
                st.session_state["_show_cases"] = True
                st.rerun()
        else:
            _render_case_results(st.session_state["_pending_cases"], lang)

    # ── Clarifying questions: option buttons under the latest assistant message ──
    _last = st.session_state.messages[-1] if st.session_state.messages else None
    if _last and _last.get("role") == "assistant" and _last.get("clarify"):
        _render_clarify_form(_last["clarify"], lang,
                             _last.get("clarify_id") or str(len(st.session_state.messages)))

    prompt = st.chat_input(t(lang, "chat_input_placeholder"))
    if st.session_state.btn_input:
        prompt = st.session_state.btn_input
        st.session_state.btn_input = None

    if prompt:
        # Clear previous case results when a new question is submitted
        st.session_state["_pending_cases"] = None
        st.session_state["_show_cases"]    = False

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Spinner covers retrieval only; the answer then streams in
            with st.spinner(t(lang, "thinking_spinner")):
                resp = _handle_prompt(prompt, persona, lang)
            if isinstance(resp, str):
                st.markdown(resp)
            else:
                resp = st.write_stream(resp)
        reply = {"role": "assistant", "content": str(resp)}
        if isinstance(resp, ClarifyReply):
            reply["clarify"] = resp.questions      # rendered as option buttons
            reply["clarify_id"] = uuid.uuid4().hex  # unique widget keys per round
        st.session_state.messages.append(reply)

        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = str(uuid.uuid4())

        save_chat(st.session_state.username, st.session_state.current_chat_id, st.session_state.messages)

        st.rerun()


# ==========================================
# 7. 主程式 — 頁面路由
# ==========================================

# Pre-load index at startup so the spinner shows immediately instead of on first query.
# @st.cache_resource ensures this runs only once per process lifetime.
# The BM25 indexes are warmed here too: jieba-tokenising the case corpus
# takes ~20 s, which would otherwise land on the first user's first question.
_startup_backend = "numpy"
_startup_res = _load_chroma_index() if VECTOR_BACKEND == "chroma" else None
if _startup_res is not None:
    _startup_backend = "chroma"
    _, _, _, _lt, _lm, _ct, _cm = _startup_res
else:
    if VECTOR_BACKEND == "chroma":
        st.warning("⚠️ Chroma 索引載入失敗（已嘗試重建），已自動退回 NumPy 路徑。詳情請查看終端機日誌。")
    _, _, _lt, _lm, _, _ct, _cm = _load_index()
if HYBRID_SEARCH:
    _load_bm25("law", _startup_backend, _lt, _lm)
    _load_bm25("cases", _startup_backend, _ct, _cm)
_penalty_index(_startup_backend, _lt, _lm)

inject_css()

if not st.session_state.logged_in:
    show_login_page()
elif st.session_state.page == "Landing":
    show_landing_page()
else:
    show_chat_room()
