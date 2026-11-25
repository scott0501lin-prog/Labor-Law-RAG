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
    MISSING_FIELDS_LABELS,
    INTENT_ANALYSIS_PROMPT,
    OVERTIME_EXTRACTION_PROMPT,
)

st.set_page_config(
    page_title="勞資爭議智慧法務 AI 顧問",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="auto",
)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "numpy").lower()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text2vec").lower()
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
@st.cache_resource(show_spinner=False)
def _init_db_once():
    init_indexes()
    return True

try:
    _init_db_once()
except Exception:
    st.error("資料庫連線失敗，請確認 MONGO_URI、MongoDB Atlas 網路存取設定及帳號權限。")
    st.stop()

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
    "intent_ctx": None,    # dict with original_query/translated_query/scenario/known_info
    "followup_count": 0,   # number of follow-up rounds completed in current dialog
    "_case_query": None,
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
@st.cache_resource(show_spinner="正在載入索引...")
def _load_index():
    import json
    import time
    import numpy as np
    from sentence_transformers import SentenceTransformer

    emb_dir = os.path.join(base_dir, "backend", "embeddings")

    t0 = time.perf_counter()
    model = SentenceTransformer(_MODEL_NAME)
    load_time = time.perf_counter() - t0

    def _load(prefix):
        npy_path = os.path.join(emb_dir, f"{prefix}_embeddings{_MODEL_SUFFIX}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"索引檔不存在：{npy_path}\n"
                f"請先執行：python backend/build_index.py\n"
                f"或設定 EMBED_MODEL=text2vec 使用現有索引"
            )
        emb  = np.load(npy_path)
        with open(os.path.join(emb_dir, f"{prefix}_docs{_MODEL_SUFFIX}.json"), encoding="utf-8") as f:
            data = json.load(f)
        return emb, data["texts"], data["metas"]

    law_emb,  law_texts,  law_metas  = _load("law")
    case_emb, case_texts, case_metas = _load("cases")

    dim = law_emb.shape[1]
    try:
        import psutil as _ps
        rss = _ps.Process().memory_info().rss / 1024 ** 2
        t_enc = time.perf_counter()
        model.encode(["test"], normalize_embeddings=True)
        enc_ms = (time.perf_counter() - t_enc) * 1000
        print(f"[MODEL] {_MODEL_NAME} | dim={dim} | load={load_time:.1f}s | RSS={rss:.0f}MB | encode={enc_ms:.0f}ms", flush=True)
    except ImportError:
        print(f"[MODEL] {_MODEL_NAME} | dim={dim} | load={load_time:.1f}s", flush=True)
    print(f"[INDEX] law={len(law_texts)} cases={len(case_texts)} total={len(law_texts)+len(case_texts)}", flush=True)

    return model, law_emb, law_texts, law_metas, case_emb, case_texts, case_metas


# ── ChromaDB helpers ───────────────────────────────────────────────────────

def _sanitize_meta(embed_text: str, meta: dict) -> dict:
    """Return a Chroma-compatible metadata dict (no None values)."""
    return {
        "embed_text": embed_text,
        "article_no": meta.get("article_no") or "",
        "source":     meta.get("source") or "",
        "parent_id":  meta.get("parent_id") or "",
        "chunk_idx":  int(meta.get("chunk_idx") or 0),
        "category":   meta.get("category") or "",
        "title":      meta.get("title") or "",
        "url":        meta.get("url") or "",
    }


def _build_chroma_collections(client, emb_dir, law_texts, law_metas, case_texts, case_metas):
    """Populate law_chunks + cases_chunks from existing .npy / .json. No re-encoding."""
    import numpy as np

    def _populate(prefix, col_name, texts, metas):
        npy_path = os.path.join(emb_dir, f"{prefix}_embeddings{_MODEL_SUFFIX}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"索引檔不存在：{npy_path}\n"
                f"請先執行：python backend/build_index.py"
            )
        emb = np.load(npy_path)
        try:
            client.delete_collection(col_name)
        except Exception:
            pass
        col = client.create_collection(col_name, metadata={"hnsw:space": "cosine"})
        ids       = [f"{prefix}_{i}" for i in range(len(texts))]
        documents = [m.get("display_text", t) for t, m in zip(texts, metas)]
        metadatas = [_sanitize_meta(t, m) for t, m in zip(texts, metas)]
        BATCH = 1000
        for s in range(0, len(ids), BATCH):
            col.add(
                ids=ids[s:s + BATCH],
                embeddings=emb[s:s + BATCH].tolist(),
                documents=documents[s:s + BATCH],
                metadatas=metadatas[s:s + BATCH],
            )

    _populate("law",   _LAW_COL_NAME,   law_texts,  law_metas)
    _populate("cases", _CASES_COL_NAME, case_texts, case_metas)


@st.cache_resource(show_spinner="正在準備索引，首次約需 2-3 分鐘…")
def _load_chroma_index():
    """Load or build ChromaDB using local bge-m3 embeddings from JSON data files.

    Returns (model, law_col, cases_col, law_texts, law_metas,
             case_texts, case_metas) on success, or None to signal fallback.
    """
    import chromadb, json
    from sentence_transformers import SentenceTransformer

    _BGE_LAW_COL   = "law_chunks_bgem3"
    _BGE_CASES_COL = "cases_chunks_bgem3"

    def _smart_split(text, max_chars, overlap):
        if len(text) <= max_chars:
            return [text]
        chunks, start = [], 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                for b in ["。", "；", "！", "？", "\n"]:
                    pos = text.rfind(b, start + max_chars // 3, end)
                    if pos != -1:
                        end = pos + 1
                        break
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _load_data():
        data_dir = os.path.join(base_dir, "backend", "data")
        def _jload(fn):
            path = os.path.join(data_dir, fn)
            return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []

        law_texts, law_metas = [], []
        for fn in ["labor_law_cleaned.json", "labor_law_rules_cleaned.json", "gender_equality_law.json"]:
            for item in _jload(fn):
                art_no    = item.get("article_no", "")
                parent_id = f"law_{art_no}"
                for idx, chunk in enumerate(_smart_split(item.get("text", ""), 500, 50)[:50]):
                    law_texts.append(chunk)
                    law_metas.append({"data_type": "law", "source": item.get("source", ""),
                                      "article_no": art_no, "parent_id": parent_id,
                                      "chunk_idx": idx, "category": "", "title": "", "url": ""})

        case_texts, case_metas = [], []
        for fn in ["taipei_cases.json", "mol_qa_cases.json", "mol_fint_cases.json",
                   "judicial_cases.json", "ntpc_cases.json", "derived_interpretations.json"]:
            for doc_idx, item in enumerate(_jload(fn)):
                content = item.get("content", "").strip()
                if not content:
                    continue
                source    = item.get("source", "")
                category  = item.get("category", "")
                label     = f"【{source} {category}】"
                parent_id = f"{fn.replace('.json', '')}_{doc_idx}"
                for idx, chunk in enumerate(_smart_split(content, 400, 50)[:50]):
                    case_texts.append(f"{label}\n{chunk}")
                    case_metas.append({"data_type": "case", "source": source, "category": category,
                                       "title": item.get("title", "").strip(), "url": item.get("url", ""),
                                       "parent_id": parent_id, "chunk_idx": idx, "article_no": ""})
        return law_texts, law_metas, case_texts, case_metas

    def _populate(client, name, texts, metas, model):
        try:
            client.delete_collection(name)
        except Exception:
            pass
        col  = client.create_collection(name, metadata={"hnsw:space": "cosine"})
        embs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        ids  = [f"{name}_{i}" for i in range(len(texts))]
        for s in range(0, len(ids), 256):
            col.add(ids=ids[s:s+256], embeddings=embs[s:s+256].tolist(),
                    documents=texts[s:s+256], metadatas=metas[s:s+256])
        return col

    try:
        law_texts, law_metas, case_texts, case_metas = _load_data()

        chroma_client = chromadb.PersistentClient(path=_CHROMA_DB_PATH)
        existing = {c.name for c in chroma_client.list_collections()}

        if _BGE_LAW_COL in existing and _BGE_CASES_COL in existing:
            law_col   = chroma_client.get_collection(_BGE_LAW_COL)
            cases_col = chroma_client.get_collection(_BGE_CASES_COL)
            if law_col.count() == len(law_texts) and cases_col.count() == len(case_texts):
                model = SentenceTransformer("BAAI/bge-m3")
                print(f"[CHROMA] bge-m3 索引就緒：law={law_col.count()}, cases={cases_col.count()}", flush=True)
                return model, law_col, cases_col, law_texts, law_metas, case_texts, case_metas

        print(f"[CHROMA] 建立 bge-m3 索引（首次或資料更新後）…", flush=True)
        model     = SentenceTransformer("BAAI/bge-m3")
        law_col   = _populate(chroma_client, _BGE_LAW_COL,   law_texts,  law_metas,  model)
        cases_col = _populate(chroma_client, _BGE_CASES_COL, case_texts, case_metas, model)
        print(f"[CHROMA] 完成：law={law_col.count()}, cases={cases_col.count()}", flush=True)
        return model, law_col, cases_col, law_texts, law_metas, case_texts, case_metas

    except Exception as e:
        print(f"[CHROMA] bge-m3 索引失敗：{e}", flush=True)
        return None


def _search(model, query, embeddings, texts, metas, top_k=5, _top_k_chunks=20):
    import numpy as np
    q_emb = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:_top_k_chunks]

    # 以 parent_id 合併：同一份文件只保留分數最高的那塊
    seen = {}
    for i in top_idx:
        pid = (metas[i].get("source"), metas[i].get("article_no")) if metas[i].get("article_no") else metas[i].get("parent_id", str(i))
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
        pid = (raw_meta.get("source"), raw_meta.get("article_no")) if raw_meta.get("article_no") else raw_meta.get("parent_id", doc)
        if pid not in seen or score > seen[pid][2]:
            meta = dict(raw_meta)
            meta["display_text"] = doc       # restore display_text stored in documents
            text = meta.pop("embed_text", doc)
            seen[pid] = (text, meta, score)

    ranked = sorted(seen.values(), key=lambda x: x[2], reverse=True)
    return ranked[:top_k]


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
from labor_calculator import calculate_overtime
from verified_rag import answer as verified_answer, direct_ids, search_cases, MODEL as VERIFIED_MODEL, REFUSAL
from prompt_helpers import (
    OUT_OF_SCOPE_TEXT,
    build_system_prompt,
    build_lang_rule,
    build_missing_note,
    build_user_prompt,
    stream_reply,
)


# ==========================================
# 3b. 意圖判斷 + 情境追問輔助函式
# ==========================================
import json as _json
import re as _re

# Retry only when the stream fails before any text was shown to the user
CLAUDE_MAX_ATTEMPTS = 2
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", VERIFIED_MODEL)


@st.cache_resource
def _get_anthropic_client(api_key: str):
    """Shared Anthropic client — cached to avoid repeated TLS setup."""
    import anthropic as _ant
    return _ant.Anthropic(api_key=api_key, timeout=15.0, max_retries=0)


def _claude_json_call(prompt_text: str, _label: str = "intent") -> dict | None:
    """Fire a Claude call and return parsed JSON dict, or None on any failure."""
    try:
        api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"[INTENT] [{_label}] API key 未設定，略過")
            return None
        client = _get_anthropic_client(api_key)
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt_text}],
        )
        text = resp.content[0].text.strip()
        print(f"[INTENT] [{_label}] Claude 原始回傳: {text!r}")
        m = _re.search(r'\{.*\}', text, _re.DOTALL)
        if m:
            text = m.group()
        result = _json.loads(text)
        print(f"[INTENT] [{_label}] 解析成功")
        return result
    except Exception as _e:
        print(f"[INTENT] [{_label}] 解析失敗: {_e}")
        return None


def _analyze_intent(user_message: str, known_info: dict, ui_lang: str) -> dict | None:
    """First-pass: classify intent, detect scenario, extract known fields.

    Returns a dict with keys: translated_query, intent, scenario, known_info,
    missing_fields, follow_up. Returns None on JSON parse failure (caller falls
    back to the original RAG flow).
    """
    print(f"[INTENT] _analyze_intent 呼叫 | 問句: {user_message!r}")
    prompt = INTENT_ANALYSIS_PROMPT.format(
        user_message=user_message,
        known_info_json=_json.dumps(known_info, ensure_ascii=False),
        ui_lang=ui_lang,
    )
    result = _claude_json_call(prompt, _label="analyze_intent")
    if result:
        print(f"[INTENT]   intent       = {result.get('intent')!r}")
        print(f"[INTENT]   scenario     = {result.get('scenario')!r}")
        print(f"[INTENT]   known_info   = {result.get('known_info')}")
        print(f"[INTENT]   missing_fields = {result.get('missing_fields')}")
        print(f"[INTENT]   follow_up    = {result.get('follow_up')!r}")
    else:
        print("[INTENT] _analyze_intent: 結果為 None（解析失敗）")
    return result


def _extract_overtime_fields(user_message: str, known_info: dict, ui_lang: str) -> dict | None:
    """Mid-dialog extraction: parse field values from a follow-up answer.

    Does NOT re-classify intent. Returns a dict with is_new_question, known_info,
    missing_fields, follow_up. Returns None on failure (caller falls back).
    """
    print(f"[INTENT] _extract_overtime_fields 呼叫 | 回覆: {user_message!r}")
    prompt = OVERTIME_EXTRACTION_PROMPT.format(
        user_message=user_message,
        known_info_json=_json.dumps(known_info, ensure_ascii=False),
        ui_lang=ui_lang,
    )
    result = _claude_json_call(prompt, _label="extract_fields")
    if result:
        print(f"[INTENT]   is_new_question = {result.get('is_new_question')}")
        print(f"[INTENT]   known_info      = {result.get('known_info')}")
        print(f"[INTENT]   missing_fields  = {result.get('missing_fields')}")
        print(f"[INTENT]   follow_up       = {result.get('follow_up')!r}")
    else:
        print("[INTENT] _extract_overtime_fields: 結果為 None（解析失敗）")
    return result


def _build_enriched_query(translated_query: str, known_info: dict) -> str:
    """Append collected field values to translated_query for a richer RAG search."""
    parts = [translated_query]
    if known_info.get("day_type"):
        parts.append(known_info["day_type"])
    if known_info.get("hours") is not None:
        parts.append(f"加班 {known_info['hours']} 小時")
    if known_info.get("wage_type") and known_info.get("wage_amount"):
        parts.append(f"{known_info['wage_type']} {known_info['wage_amount']} 元")
    elif known_info.get("wage_type"):
        parts.append(known_info["wage_type"])
    elif known_info.get("wage_amount"):
        parts.append(f"薪資 {known_info['wage_amount']} 元")
    if known_info.get("compensation_type"):
        parts.append(known_info["compensation_type"])
    return "，".join(parts)


def _handle_prompt(prompt: str, persona: str, lang: str) -> str:
    """Strict evidence-only route. Never stream unverified AI legal prose."""
    return query_rag_system(prompt, persona, lang)


def query_rag_system(user_prompt: str, system_prompt: str, ui_lang: str, **_unused) -> str:
    started = time.perf_counter()
    st.session_state["_pending_cases"] = None
    st.session_state["_case_query"] = None
    st.session_state["_show_cases"] = False
    api_key = ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    client = None
    semantic_keys = []
    if direct_ids(user_prompt) is None:
        if api_key:
            try:
                client = _get_anthropic_client(api_key)
            except Exception:
                pass
        # Optional semantic retrieval, disabled by default to avoid heavy cold start.
        if os.getenv("RAG_SEMANTIC", "false").lower() == "true":
            try:
                from verified_rag import number, FILES
                model, law_emb, law_texts, law_metas, *_ = _load_index()
                for _, meta, _ in _search(model, user_prompt, law_emb, law_texts, law_metas, top_k=12, _top_k_chunks=60):
                    if meta.get("source") in FILES:
                        semantic_keys.append(FILES[meta["source"]][0]+":"+number(meta["article_no"]))
            except Exception as exc:
                logging.getLogger("verified_rag").warning("optional semantic retrieval unavailable: %s", type(exc).__name__)
    result = verified_answer(user_prompt, ui_lang, client, CLAUDE_MODEL,
                             semantic_keys=semantic_keys,
                             online=os.getenv("VERIFY_OFFICIAL", "true").lower() == "true")
    if "### 法規查詢結果" in result:
        st.session_state["_case_query"] = user_prompt
    logging.getLogger("verified_rag").info("answer elapsed=%.3fs", time.perf_counter()-started)
    return result


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
            st.session_state.current_chat_id  = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center; color:#7a6e58; font-size:0.85rem;'>"
        f"{t(lang, 'landing_caption').format(username=st.session_state.username)}</p>",
        unsafe_allow_html=True,
    )


def _render_case_results(c_results: list, lang: str) -> None:
    """Render similar-case cards below the answer. Called only when user clicks the button."""
    st.markdown("---")
    st.markdown("**🔍 相關案例與參考資料** *(本地資料未即時核對；包括函釋、問答及裁判書，不一定是裁罰案例)*")
    if not c_results:
        st.info("未找到相似案例。")
        return
    for text, meta, score in c_results:
        title    = meta.get("title") or ""
        source   = meta.get("source") or ""
        category = meta.get("category") or ""
        display  = meta.get("display_text", text)
        summary  = display[:450] + ("…" if len(display) > 450 else "")
        url      = meta.get("url") or ""
        header   = title or f"{source} — {category}"
        with st.expander(f"📄 {header}", expanded=False):
            if source or category:
                st.markdown(f"**資料來源／類別**：{source}{'  —  ' + category if category else ''}")
            # 僅顯示索引資料實際存在的欄位，不推測裁罰資訊。
            metadata_labels = [
                ("地區", ("city", "county", "location")),
                ("違規法條", ("article_no", "law_article", "violation_article")),
                ("裁罰日期", ("penalty_date",)),
                ("資料日期", ("date",)),
                ("裁罰金額", ("fine", "fine_amount", "penalty_amount")),
            ]
            for label, keys in metadata_labels:
                value = next((meta.get(key) for key in keys if meta.get(key) not in (None, "")), None)
                if value is not None:
                    st.markdown(f"**{label}**：{value}")
            st.text(summary)
            if url and url.startswith("https://"):
                st.markdown(f"[🔗 查看原文]({url})")
            st.caption("本地案例資料未即時核對；BM25 排序分數不是正確率或違法認定。")


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
            st.session_state.current_chat_id = None
            st.session_state["_pending_cases"] = None
            st.session_state["_case_query"] = None
            st.session_state["_show_cases"]    = False
            st.rerun()

        if st.button(t(lang, "btn_switch_role"), use_container_width=True):
            st.session_state.page            = "Landing"
            st.session_state.messages        = []
            st.session_state.current_chat_id = None
            st.session_state["_pending_cases"] = None
            st.session_state["_case_query"] = None
            st.session_state["_show_cases"]    = False
            st.rerun()

        if st.button(t(lang, "btn_logout"), use_container_width=True):
            for k in ["logged_in", "username", "user_role", "messages", "current_chat_id"]:
                st.session_state[k] = defaults[k]
            st.session_state["_pending_cases"] = None
            st.session_state["_case_query"] = None
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
                    st.session_state.current_chat_id = cid
                    st.session_state["_case_query"] = None
                    st.session_state["_pending_cases"] = None
                    st.session_state["_show_cases"] = False
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

    # 加班費概算工具：由程式依級距計算，不讓語言模型自行心算。
    with st.expander("🧮 加班費快速概算（一般月薪制）", expanded=False):
        st.caption(
            "目前支援平日及休息日的一般月薪制概算，依月薪 ÷ 240 換算每小時工資額。"
            "例假日、國定假日、變形工時、部分工時及特殊工資項目請另行核對。"
        )
        calc_col1, calc_col2, calc_col3 = st.columns(3)
        with calc_col1:
            calc_wage = st.number_input(
                "月薪（元）", min_value=1.0, value=36000.0, step=1000.0,
                key="overtime_calc_monthly_wage"
            )
        with calc_col2:
            calc_hours = st.number_input(
                "加班時數（平日最多 4 小時）", min_value=0.5, max_value=12.0, value=2.0, step=0.5,
                key="overtime_calc_hours"
            )
        with calc_col3:
            calc_day_type = st.selectbox(
                "加班日類型", ["平日", "休息日"], key="overtime_calc_day_type"
            )
        if st.button("計算加班費", key="overtime_calc_submit"):
            try:
                calc_result = calculate_overtime(calc_wage, calc_hours, calc_day_type)
                st.success(f"估算加班費：NT$ {calc_result['total']:,.2f}")
                st.write(f"每小時工資額：NT$ {calc_result['hourly_wage']:,.2f}")
                for item in calc_result["breakdown"]:
                    st.write(
                        f"- {item['label']}：{item['hours']:g} 小時 × "
                        f"{item['fraction']} 倍 = NT$ {item['amount']:,.2f}"
                    )
                st.markdown("[勞動部月薪制加班費試算](https://calcr2.mol.gov.tw/Monthly)")
                st.warning(
                    "公式未即時核對現行法規。這是一般情境的估算，不是正式工資核算或法律認定。"
                    "請確認工資是否包含應計入的經常性給與、實際出勤時數及適用工時制度。"
                )
            except ValueError as exc:
                st.error(str(exc))

    st.caption("嚴格證據模式：法規原文經本地比對，另標示官方核對狀態；AI 負責挑選證據，不直接判定個案。其他語言仍保留中文法條原文。歷史回答可能來自舊版，請重新查詢。")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 「查看相似案例」按鈕（出現在最後一則 AI 回答之後）──
    _has_cases = bool(st.session_state.get("_case_query"))
    _last_is_assistant = (
        bool(st.session_state.messages)
        and st.session_state.messages[-1]["role"] == "assistant"
    )
    if _has_cases and _last_is_assistant:
        if not st.session_state.get("_show_cases"):
            if st.button("🔍 查看相似案例"):
                with st.spinner("搜尋本地案例資料…"):
                    try:
                        st.session_state["_pending_cases"] = search_cases(st.session_state["_case_query"])
                    except Exception:
                        st.session_state["_pending_cases"] = []
                st.session_state["_show_cases"] = True
                st.rerun()
        else:
            _render_case_results(st.session_state["_pending_cases"], lang)

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
            # Display only after the complete response has been validated.
            with st.spinner(t(lang, "thinking_spinner")):
                resp = _handle_prompt(prompt, persona, lang)
            st.markdown(resp if isinstance(resp, str) else REFUSAL)
            if not isinstance(resp, str):
                resp = REFUSAL
        st.session_state.messages.append({"role": "assistant", "content": resp})

        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = str(uuid.uuid4())

        save_chat(st.session_state.username, st.session_state.current_chat_id, st.session_state.messages)

        st.rerun()


# ==========================================
# 7. 主程式 — 頁面路由
# ==========================================

# Lazy retrieval: no embedding or case index is loaded at startup.

inject_css()

if not st.session_state.logged_in:
    show_login_page()
elif st.session_state.page == "Landing":
    show_landing_page()
else:
    show_chat_room()
