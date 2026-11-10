import streamlit as st
import os
import uuid
import time
import logging
from datetime import timezone
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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
    "intent_ctx": None,    # dict with original_query/translated_query/scenario/known_info
    "followup_count": 0,   # number of follow-up rounds completed in current dialog
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, .stApp {
        font-family: 'Noto Sans TC', 'Inter', -apple-system, sans-serif !important;
    }

    /* 隱藏 Streamlit 預設元素 */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none !important; }

    /* 主背景 */
    .stApp { background: #f0f4f8; }

    /* ── 側邊欄 ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2744 0%, #243257 100%) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #dde3f0 !important;
    }
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
        background-color: #1e3060 !important;
        background: #1e3060 !important;
        color: #dde3f0 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        font-size: 0.875rem;
        text-align: left;
        transition: all 0.18s ease;
    }
    section[data-testid="stSidebar"] button:hover,
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(201,168,76,0.25) !important;
        background: rgba(201,168,76,0.25) !important;
        border-color: rgba(201,168,76,0.55) !important;
        color: #f5e090 !important;
    }
    section[data-testid="stSidebar"] button:focus,
    section[data-testid="stSidebar"] button:active,
    section[data-testid="stSidebar"] .stButton > button:focus,
    section[data-testid="stSidebar"] .stButton > button:active {
        background-color: #1e3060 !important;
        background: #1e3060 !important;
        color: #dde3f0 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* selectbox in sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: rgba(255,255,255,0.1) !important;
        border-color: rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: #dde3f0 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #dde3f0 !important; }

    /* ── 標題 ── */
    h1 { color: #1a2744 !important; font-weight: 700 !important; letter-spacing: -0.3px; }
    h2, h3 { color: #243257 !important; font-weight: 600 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #d1dae8;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.55rem 1.4rem;
        font-weight: 500;
        color: #64748b;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1a2744 !important;
        background: white !important;
        border-top: 3px solid #c9a84c !important;
        font-weight: 600;
    }

    /* ── 輸入框 ── */
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        border: 1.5px solid #c8d4e3 !important;
        padding: 0.55rem 1rem !important;
        font-size: 0.95rem;
        background: white !important;
        transition: border-color 0.18s;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1a2744 !important;
        box-shadow: 0 0 0 3px rgba(26,39,68,0.08) !important;
    }

    /* ── 一般按鈕（主內容區） ── */
    .main .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        border: 1.5px solid #c0ccdd !important;
        color: #1a2744 !important;
        background: white !important;
        transition: all 0.18s ease !important;
    }
    .main .stButton > button:hover {
        background: #1a2744 !important;
        color: white !important;
        border-color: #1a2744 !important;
        box-shadow: 0 4px 14px rgba(26,39,68,0.18) !important;
        transform: translateY(-1px);
    }

    /* ── 表單送出按鈕 ── */
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #1a2744, #2a4080) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem !important;
        letter-spacing: 0.2px;
        transition: all 0.18s ease !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(135deg, #243257, #1a2744) !important;
        box-shadow: 0 6px 20px rgba(26,39,68,0.35) !important;
        transform: translateY(-1px);
    }

    /* ── Alert 卡片 ── */
    .stAlert { border-radius: 12px !important; }

    /* ── 聊天訊息 ── */
    [data-testid="stChatMessage"] { border-radius: 12px !important; }

    /* ── 分隔線 ── */
    hr { border-color: #dde3f0 !important; margin: 0.8rem 0 !important; }

    /* ── 登入頁置中 ── */
    .login-wrapper {
        max-width: 440px;
        margin: 0 auto;
        padding: 2.5rem 2rem;
        background: white;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(26,39,68,0.10);
    }

    /* ── Landing 身份卡片 ── */
    .role-card {
        background: white;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 12px rgba(26,39,68,0.08);
        border: 2px solid transparent;
        transition: border-color 0.18s, box-shadow 0.18s;
        height: 100%;
    }
    .role-card:hover { border-color: #c9a84c; box-shadow: 0 6px 24px rgba(26,39,68,0.14); }
    .role-card h3 { margin-top: 0; font-size: 1.15rem; }
    .role-card p { color: #4a5568; font-size: 0.92rem; line-height: 1.6; margin-bottom: 1.2rem; }
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
    import numpy as np
    from sentence_transformers import SentenceTransformer

    emb_dir = os.path.join(base_dir, "backend", "embeddings")
    model   = SentenceTransformer("shibing624/text2vec-base-chinese")

    def _load(prefix):
        emb  = np.load(os.path.join(emb_dir, f"{prefix}_embeddings.npy"))
        with open(os.path.join(emb_dir, f"{prefix}_docs.json"), encoding="utf-8") as f:
            data = json.load(f)
        return emb, data["texts"], data["metas"]

    law_emb,  law_texts,  law_metas  = _load("law")
    case_emb, case_texts, case_metas = _load("cases")
    return model, law_emb, law_texts, law_metas, case_emb, case_texts, case_metas


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


# ==========================================
# 3b. 意圖判斷 + 情境追問輔助函式
# ==========================================
import json as _json
import re as _re


def _gemini_json_call(prompt_text: str, _label: str = "intent") -> dict | None:
    """Fire a Gemini call and return parsed JSON dict, or None on any failure."""
    try:
        from google import genai as _genai
        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print(f"[INTENT] [{_label}] API key 未設定，略過")
            return None
        client = _genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text,
        )
        text = resp.text.strip()
        print(f"[INTENT] [{_label}] Gemini 原始回傳: {text!r}")
        # Strip optional ```json ... ``` wrapper Gemini sometimes adds
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
    result = _gemini_json_call(prompt, _label="analyze_intent")
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
    result = _gemini_json_call(prompt, _label="extract_fields")
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
    """Route a user prompt through intent detection + optional follow-up + RAG.

    Falls back to query_rag_system() directly whenever intent detection is
    disabled, JSON parsing fails, or the scenario is not overtime_pay.
    """
    print(f"[INTENT] ── _handle_prompt ──────────────────────────────")
    print(f"[INTENT] ENABLE_INTENT_DETECTION = {ENABLE_INTENT_DETECTION}")
    print(f"[INTENT] 使用者問句: {prompt!r}")

    def _is_skip(text: str) -> bool:
        return (
            len(text) <= SKIP_MAX_LEN
            and any(kw in text for kw in SKIP_FOLLOWUP_TRIGGERS)
        )

    def _new_question(qprompt: str) -> str:
        """Handle a fresh (non-mid-dialog) question."""
        analysis = _analyze_intent(qprompt, {}, lang)
        if (
            analysis
            and analysis.get("intent") == "solve_problem"
            and analysis.get("scenario") == "overtime_pay"
            and analysis.get("missing_fields")
            and not _is_skip(qprompt)
        ):
            print(f"[INTENT] 分支: 追問 | follow_up={analysis.get('follow_up')!r}")
            st.session_state.intent_ctx = {
                "original_query":   qprompt,
                "translated_query": analysis.get("translated_query") or qprompt,
                "scenario":         "overtime_pay",
                "known_info":       analysis.get("known_info") or {},
            }
            st.session_state.followup_count = 0
            return analysis.get("follow_up") or ""
        else:
            reason = "解析失敗" if not analysis else f"intent={analysis.get('intent')!r} scenario={analysis.get('scenario')!r}"
            print(f"[INTENT] 分支: 直接 RAG | {reason}")
            sq = analysis.get("translated_query") if analysis else None
            return query_rag_system(qprompt, persona, lang, search_query_override=sq)

    if not ENABLE_INTENT_DETECTION:
        print("[INTENT] 未執行: 功能關閉 (ENABLE_INTENT_DETECTION=False) → 直接 RAG")
        return query_rag_system(prompt, persona, lang)

    ctx = st.session_state.intent_ctx
    print(f"[INTENT] intent_ctx = {'有進行中對話' if ctx else '無 (新問題)'}")

    if ctx is not None:
        # ── Mid-dialog: processing a follow-up answer ─────────────────────
        if _is_skip(prompt):
            print(f"[INTENT] 分支: mid-dialog 跳過 → 直接 RAG (原問句)")
            st.session_state.intent_ctx = None
            st.session_state.followup_count = 0
            return query_rag_system(
                ctx["original_query"], persona, lang,
                search_query_override=ctx["translated_query"],
            )

        result = _extract_overtime_fields(prompt, ctx["known_info"], lang)
        if result is None:
            # JSON failure → fallback with what we have
            print("[INTENT] 分支: extraction 解析失敗 → fallback RAG (原問句)")
            st.session_state.intent_ctx = None
            st.session_state.followup_count = 0
            return query_rag_system(
                ctx["original_query"], persona, lang,
                search_query_override=ctx["translated_query"],
            )

        if result.get("is_new_question"):
            print("[INTENT] 分支: is_new_question=True → 清空 ctx，重新分析新問題")
            st.session_state.intent_ctx = None
            st.session_state.followup_count = 0
            return _new_question(prompt)

        # Merge new non-null values into known_info
        for k, v in (result.get("known_info") or {}).items():
            if v is not None:
                ctx["known_info"][k] = v
        st.session_state.followup_count += 1
        missing = result.get("missing_fields") or []

        if missing and st.session_state.followup_count < MAX_FOLLOWUP_ROUNDS:
            print(f"[INTENT] 分支: 繼續追問 (round={st.session_state.followup_count}) | missing={missing}")
            return result.get("follow_up") or ""

        # Reached limit or all fields collected → proceed to RAG
        enriched = _build_enriched_query(ctx["translated_query"], ctx["known_info"])
        print(f"[INTENT] 分支: 追問完成 → RAG | enriched={enriched!r} | missing={missing or None}")
        st.session_state.intent_ctx = None
        st.session_state.followup_count = 0
        return query_rag_system(
            enriched, persona, lang,
            search_query_override=enriched,
            missing_fields=missing or None,
        )

    else:
        return _new_question(prompt)


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
) -> str:
    """Run the full RAG pipeline and return a response string.

    search_query_override: pre-translated Chinese query from _analyze_intent;
        if provided, the internal translation call is skipped.
    missing_fields: list of required field names that were not collected;
        if non-empty, a disclaimer is appended to final_prompt asking Gemini
        NOT to assume values.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return t(ui_lang, "err_no_api_key")

        client = genai.Client(api_key=api_key)
        _t0 = time.perf_counter()

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
                    search_query = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"請將以下使用者問題翻譯成繁體中文，只需要輸出翻譯結果，不要加任何說明：\n\n{user_prompt}"
                    ).text.strip()
                except Exception:
                    search_query = user_prompt
        _t_trans = time.perf_counter()

        l_results = _search(model, search_query, law_emb,  law_texts,  law_metas,  top_k=5)
        _t_law_search = time.perf_counter()

        fired_mutex: set = set()
        for rule in FORCE_ARTICLE_RULES:
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
        _t_force_l = time.perf_counter()

        c_results = _search(model, search_query, case_emb, case_texts, case_metas, top_k=5)
        _t_case_search = time.perf_counter()

        _overtime_kw = next(r["keywords"] for r in FORCE_ARTICLE_RULES if r["name"] == "加班延長工時")
        c_results = _force_include_cases(
            search_query, c_results, case_texts, case_metas, _overtime_kw, RESTDAY_OVERTIME_FORCE_TITLES
        )
        _t_force_c = time.perf_counter()

        law_ctx = ""
        for d, m, _ in l_results:
            display = m.get("display_text", d)   # 完整條文；舊索引 fallback 用 embed chunk
            law_ctx += f"【法規內容】：{display}\n【條號】：{m.get('source', '')} {m.get('article_no', '')}\n\n"

        case_ctx = ""
        for d, m, _ in c_results:
            display  = m.get("display_text", d)  # 主文+鄰近chunk；舊索引 fallback 用 embed chunk
            url_line = f"\n【網址】：{m['url']}" if m.get("url") else ""
            case_ctx += f"【{m.get('source', '')} — {m.get('category', '')}】\n{display}{url_line}\n\n"

        # 回答語言邏輯：永遠跟著 UI 語言選項走。
        # 使用者在語言下拉選 English → AI 用英文回答，以此類推。
        if ui_lang != "繁體中文":
            lang_name = LANGUAGES.get(ui_lang, "Traditional Chinese")
            lang_rule = (
                f"\n8. 除「📖 法條依據」區塊中的法條原文須保留繁體中文（避免翻譯造成法律歧義）外，"
                f"其餘所有文字（結論、📂 參考案例、💡 說明）請使用「{lang_name}」撰寫；"
                f"法條原文後方請附上一句{lang_name}白話翻譯。"
                f"加班費倍率的數字（如 4/3、5/3、1.33、1.66）必須維持原本數值與格式，不可換算。"
            )
        else:
            lang_rule = ""

        # Missing-fields disclaimer — only appended when fields were still absent
        # after the follow-up dialog, so Gemini won't invent numbers.
        if missing_fields:
            labels = "、".join(
                MISSING_FIELDS_LABELS.get(f, f) for f in missing_fields
            )
            missing_note = (
                f"\n\n【重要提示：本問題因缺少以下資訊，僅能提供一般情況說明，"
                f"請勿自行假設任何數值，如需精確計算請補充後重新詢問。】\n"
                f"缺少的資訊：{labels}"
            )
        else:
            missing_note = ""

        final_prompt = f"""
【相關法規條文】（資料庫檢索）：
{law_ctx}

【相關案例 / Q&A / 裁判書】（資料庫檢索）：
{case_ctx}

使用者問題：{user_prompt}

【回答格式要求】：
1. 先給出簡明的直接結論。
2. 在「📖 法條依據」區塊中，逐條列出引用的法條號碼與原文：
   > **勞動基準法 第 X 條**
   > （條文原文）
3. 在「📂 參考案例」區塊中，若有相關案例或 Q&A，請摘要說明並附上來源網址（可點擊連結）。
4. 在「💡 說明」區塊中，依據以上資料說明理由與注意事項。
5. 若資料庫無相關條文或案例，須明確說明，不可捏造。
6. 若回答涉及加班費倍率計算（例如平日延長工時前 2 小時為 1 又 1/3 倍〈4/3，約 1.33 倍〉、
   超過 2 小時至 4 小時為 1 又 2/3 倍〈5/3，約 1.66 倍〉），必須同時列出精確分數與約略小數
   （例如「4/3 倍（約 1.33 倍）」），不可只寫約略小數，也不可寫成「多 33%」等容易誤解為
   百分比加成的說法；並清楚說明此倍率是以平日每小時工資額為計算基準。
7. 若問題涉及「依年資對應天數／期間／金額」的法條級距（例如特別休假天數、資遣預告期間、
   資遣費／退休金計算等），作答前必須先明確判斷使用者的年資落在哪一款／哪一個級距，
   全文只能使用該款對應的數字；「結論」與「📖 法條依據」引用的級距與數字必須完全一致，
   不可一邊寫某一級距的天數、另一邊卻引用其他級距的法條。若年資剛好落在級距交界（例如
   「剛滿」某年），須以該年資對應的較低級距（未滿下一級距）計算，並可在「💡 說明」中
   註明何時會晉升到下一級距。{lang_rule}{missing_note}
"""
        _t_prompt = time.perf_counter()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=final_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        _t_gemini = time.perf_counter()

        _rag_logger.info(
            f"q={user_prompt[:25]!r} zh={_is_chinese(user_prompt)} "
            f"load={_t_load-_t0:.3f}s "
            f"trans={_t_trans-_t_load:.3f}s "
            f"law_search={_t_law_search-_t_trans:.3f}s "
            f"force_l={_t_force_l-_t_law_search:.3f}s "
            f"case_search={_t_case_search-_t_force_l:.3f}s "
            f"force_c={_t_force_c-_t_case_search:.3f}s "
            f"prompt={_t_prompt-_t_force_c:.3f}s "
            f"gemini={_t_gemini-_t_prompt:.3f}s "
            f"total={_t_gemini-_t0:.3f}s"
        )

        return response.text

    except Exception as e:
        return f"🚨 異常：{str(e)}"


# ==========================================
# 4. 頁面一：登入 / 註冊
# ==========================================
def show_login_page():
    render_language_switcher()
    lang = st.session_state.ui_lang

    st.markdown("<br>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            "<div style='text-align:center; margin-bottom:0.5rem;'>"
            "<span style='font-size:3rem;'>⚖️</span>"
            "</div>"
            f"<h2 style='text-align:center; color:#1a2744; font-size:1.45rem; margin-bottom:0.2rem;'>{t(lang, 'app_title').replace('⚖️ ','')}</h2>"
            f"<p style='text-align:center; color:#64748b; margin-bottom:1.5rem; font-size:0.95rem;'>{t(lang, 'login_subheader')}</p>",
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


# ==========================================
# 5. 頁面二：身份選擇 (Landing Page)
# ==========================================
def show_landing_page():
    render_language_switcher()
    lang = st.session_state.ui_lang

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align:center; color:#1a2744;'>⚖️ " + t(lang, "app_title").replace("⚖️ ", "") + "</h1>"
        f"<p style='text-align:center; color:#4a5568; font-size:1.05rem; margin-bottom:0.3rem;'>{t(lang, 'landing_subheader')}</p>"
        f"<p style='text-align:center; color:#64748b; font-size:0.9rem; margin-bottom:2rem;'>{t(lang, 'landing_desc')}</p>",
        unsafe_allow_html=True,
    )

    _, c1, _, c2, _ = st.columns([0.5, 3, 0.3, 3, 0.5])

    with c1:
        st.markdown(
            "<div class='role-card'>"
            "<h3>🙋‍♂️ " + t(lang, "role_label_employee") + "</h3>"
            f"<p>{t(lang, 'role_employee_desc')}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
        if st.button(t(lang, "role_employee_btn"), use_container_width=True, key="emp_btn"):
            st.session_state.user_role        = "Employee"
            st.session_state.page             = "Chat"
            st.session_state.messages         = []
            st.session_state.current_chat_id  = None
            st.rerun()

    with c2:
        st.markdown(
            "<div class='role-card'>"
            "<h3>🏢 " + t(lang, "role_label_employer") + "</h3>"
            f"<p>{t(lang, 'role_employer_desc')}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
        if st.button(t(lang, "role_employer_btn"), use_container_width=True, key="er_btn"):
            st.session_state.user_role        = "Employer"
            st.session_state.page             = "Chat"
            st.session_state.messages         = []
            st.session_state.current_chat_id  = None
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption(t(lang, "landing_caption").format(username=st.session_state.username))


# ==========================================
# 6. 頁面三：AI 聊天室 (Chat Room)
# ==========================================
def show_chat_room():
    render_language_switcher()
    lang = st.session_state.ui_lang

    if st.session_state.user_role == "Employee":
        title   = t(lang, "chat_title_employee")
        persona = (
            "你是一位專業勞工法律顧問，專門協助勞工了解並保護自身的勞動權益。"
            "你具備深厚的台灣勞動法律知識，包括勞動基準法、勞工保險條例、職業安全衛生法等。"
            "請以友善、清晰、易懂的語氣回答問題。"
            "回答時必須：①先給結論，②在「📖 法條依據」區塊逐條引用法條號碼與原文，"
            "③在「💡 說明」區塊解釋理由。若資料庫無相關條文，須明確說明，不可捏造條號。"
        )
        btns = [
            (t(lang, "qe_btn1"), "我被資遣了！我的權益是什麼？"),
            (t(lang, "qe_btn2"), "加班費怎麼算？"),
            (t(lang, "qe_btn3"), "如何申請勞資調解？"),
            (t(lang, "qe_btn4"), "發生職災我可以申請哪些補償？"),
        ]
    else:
        title   = t(lang, "chat_title_employer")
        persona = (
            "你是一位專業企業勞動法律顧問，專門協助雇主和HR在合法合規的前提下管理勞資關係。"
            "你具備深厚的台灣勞動法律知識，並能從企業管理角度提供實用建議。"
            "請以專業、嚴謹的語氣回答。"
            "回答時必須：①先給結論，②在「📖 法條依據」區塊逐條引用法條號碼與原文，"
            "③在「💡 說明」區塊解釋理由與風險評估。若資料庫無相關條文，須明確說明，不可捏造條號。"
        )
        btns = [
            (t(lang, "qr_btn1"), "合法資遣流程？"),
            (t(lang, "qr_btn2"), "員工連續曠職怎麼辦？"),
            (t(lang, "qr_btn3"), "如何合法設定工時與薪資？"),
            (t(lang, "qr_btn4"), "如何避免被員工索賠？"),
        ]

    role_label = t(lang, "role_label_employee" if st.session_state.user_role == "Employee" else "role_label_employer")

    # --- 側邊欄 ---
    with st.sidebar:
        st.title(t(lang, "sidebar_title"))
        st.write(t(lang, "sidebar_account").format(username=st.session_state.username))
        st.write(t(lang, "sidebar_role").format(role=role_label))

        if st.button(t(lang, "btn_new_chat"), use_container_width=True):
            st.session_state.messages        = []
            st.session_state.current_chat_id = None
            st.rerun()

        if st.button(t(lang, "btn_switch_role"), use_container_width=True):
            st.session_state.page            = "Landing"
            st.session_state.messages        = []
            st.session_state.current_chat_id = None
            st.rerun()

        if st.button(t(lang, "btn_logout"), use_container_width=True):
            for k in ["logged_in", "username", "user_role", "messages", "current_chat_id"]:
                st.session_state[k] = defaults[k]
            st.session_state.page = "Login"
            st.rerun()

        st.divider()
        st.markdown(f"### {t(lang, 'quick_questions_title')}")
        for label, value in btns:
            if st.button(label, use_container_width=True):
                st.session_state.btn_input = value

        st.divider()
        st.markdown(f"### {t(lang, 'history_title')}")
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
                    st.rerun()

    # --- 主聊天區 ---
    st.title(f"⚖️ {title}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(t(lang, "chat_input_placeholder"))
    if st.session_state.btn_input:
        prompt = st.session_state.btn_input
        st.session_state.btn_input = None

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner(t(lang, "thinking_spinner")):
                resp = _handle_prompt(prompt, persona, lang)
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})

        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = str(uuid.uuid4())

        save_chat(st.session_state.username, st.session_state.current_chat_id, st.session_state.messages)

        st.rerun()


# ==========================================
# 7. 主程式 — 頁面路由
# ==========================================
inject_css()

if not st.session_state.logged_in:
    show_login_page()
elif st.session_state.page == "Landing":
    show_landing_page()
else:
    show_chat_room()
