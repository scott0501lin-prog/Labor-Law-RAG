import streamlit as st
import os
import uuid
from datetime import timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from db import init_indexes, register_user, verify_user, save_chat, list_chats, load_chat
from i18n import LANGUAGES, DB_MSG_KEYS, t

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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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


def _search(model, query, embeddings, texts, metas, top_k=3):
    import numpy as np
    q_emb = model.encode([query], normalize_embeddings=True)
    scores = (embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(texts[i], metas[i], float(scores[i])) for i in top_idx]


# 加班費是最常被問、也最容易被 embedding 檢索漏掉的主題（問句越長、細節越多，
# 排名越不穩定）。偵測到這些關鍵字時，強制把核心條文塞進結果，不依賴 embedding 排名。
#
# 「換補休」類提問實際對應的法條是第 32-1 條（補休轉換規則），跟第 24/32/36 條
# （加班費倍率、工時上限、休息日）關聯度較低；若混在同一組關鍵字裡，會對換補休
# 問題硬塞不對題的加班費條文。因此拆成兩組、依優先序互斥處理：換補休類優先比對，
# 命中就只保底第 32-1 條；否則才落回一般加班費/工時類的保底。
COMP_LEAVE_KEYWORDS = ["補休", "換休", "以休代薪", "代替休假"]
COMP_LEAVE_FORCE_ARTICLES = {"第 32-1 條"}

OVERTIME_KEYWORDS = ["加班", "延長工作時間", "延長工時", "休息日工作", "例假工作", "假日加班"]
OVERTIME_FORCE_ARTICLES = {"第 24 條", "第 32 條", "第 36 條"}

TERMINATION_KEYWORDS = ["資遣", "預告", "解僱", "解雇", "終止契約", "預告期間", "預告工資",
                        "合法解僱", "解僱事由", "終止勞動契約"]
TERMINATION_FORCE_ARTICLES = {"第 11 條", "第 12 條", "第 13 條", "第 14 條", "第 16 條", "第 17 條"}

LEAVE_KEYWORDS = ["特別休假", "年假", "休假天數", "特休"]
LEAVE_FORCE_ARTICLES = {"第 38 條"}

HOLIDAY_KEYWORDS = ["國定假日", "例假", "休息日", "國假", "補休", "假日出勤", "假日上班", "輪班假日"]
HOLIDAY_FORCE_ARTICLES = {"第 36 條", "第 37 條", "第 39 條"}

# 「休息日出勤超過 8 小時，第 9 小時起計給 2 又 2/3 倍」出自勞動部勞動條 2 字第
# 1050030466 號書函，但這筆函釋在 embedding 檢索排名不穩定、常常掉出前 5 名
# （見 backend/derived_interpretations.json），所以跟法條一樣用關鍵字強制帶入。
# 只在「休息日」與加班關鍵字同時出現時才觸發，避免非休息日的一般加班問題也被硬塞。
RESTDAY_OVERTIME_FORCE_TITLES = {"勞動部勞動條 2 字第 1050030466 號書函"}


def _force_include_articles(query, results, texts, metas, keywords, force_article_nos, source="勞動基準法"):
    # 法條索引現在同時混了勞基法本法與施行細則，兩者的 article_no 可能撞號
    # （例如兩邊都各自有自己的「第 24 條」），所以強制帶入時一定要限定
    # source，否則可能誤把施行細則裡不相關的同號條文塞進來。
    if not any(k in query for k in keywords):
        return results
    existing = {m.get("article_no") for _, m, _ in results}
    forced = [
        (texts[i], m, None)
        for i, m in enumerate(metas)
        if m.get("article_no") in force_article_nos
        and m.get("article_no") not in existing
        and m.get("source") == source
    ]
    return forced + results


def _force_include_cases(query, results, texts, metas, keywords, force_titles):
    if not any(k in query for k in keywords) or "休息日" not in query:
        return results
    existing = {m.get("title") for _, m, _ in results}
    forced = [
        (texts[i], m, None)
        for i, m in enumerate(metas)
        if m.get("title") in force_titles and m.get("title") not in existing
    ]
    return forced + results


# ==========================================
# 4. 後端 RAG 引擎 (Gemini + Numpy 搜尋)
# ==========================================
def query_rag_system(user_prompt: str, system_prompt: str, ui_lang: str) -> str:
    try:
        import google.generativeai as genai

        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return t(ui_lang, "err_no_api_key")

        genai.configure(api_key=api_key)
        model, law_emb, law_texts, law_metas, case_emb, case_texts, case_metas = _load_index()

        # 檢索用的 embedding 模型是中文專用。
        # 判斷方式：中文字（含標點）佔比 < 30% 就視為非中文，無論 UI 語言設定為何都先翻譯。
        # 這樣即使使用者 UI 設定為「繁體中文」但直接打越南文/英文，仍能正確檢索。
        def _is_chinese(text: str) -> bool:
            chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
            return chinese_chars / max(len(text), 1) >= 0.3

        search_query = user_prompt
        if not _is_chinese(user_prompt):
            try:
                translator = genai.GenerativeModel(model_name="gemini-2.5-flash")
                search_query = translator.generate_content(
                    f"請將以下使用者問題翻譯成繁體中文，只需要輸出翻譯結果，不要加任何說明：\n\n{user_prompt}"
                ).text.strip()
            except Exception:
                search_query = user_prompt

        l_results = _search(model, search_query, law_emb,  law_texts,  law_metas,  top_k=5)
        if any(k in search_query for k in COMP_LEAVE_KEYWORDS):
            l_results = _force_include_articles(
                search_query, l_results, law_texts, law_metas, COMP_LEAVE_KEYWORDS, COMP_LEAVE_FORCE_ARTICLES
            )
        else:
            l_results = _force_include_articles(
                search_query, l_results, law_texts, law_metas, OVERTIME_KEYWORDS, OVERTIME_FORCE_ARTICLES
            )
        l_results = _force_include_articles(
            search_query, l_results, law_texts, law_metas, TERMINATION_KEYWORDS, TERMINATION_FORCE_ARTICLES
        )
        l_results = _force_include_articles(
            search_query, l_results, law_texts, law_metas, LEAVE_KEYWORDS, LEAVE_FORCE_ARTICLES
        )
        l_results = _force_include_articles(
            search_query, l_results, law_texts, law_metas, HOLIDAY_KEYWORDS, HOLIDAY_FORCE_ARTICLES
        )
        c_results = _search(model, search_query, case_emb, case_texts, case_metas, top_k=5)
        c_results = _force_include_cases(
            search_query, c_results, case_texts, case_metas, OVERTIME_KEYWORDS, RESTDAY_OVERTIME_FORCE_TITLES
        )

        law_ctx = ""
        for d, m, _ in l_results:
            law_ctx += f"【法規內容】：{d}\n【條號】：{m.get('source', '')} {m.get('article_no', '')}\n\n"

        case_ctx = ""
        for d, m, _ in c_results:
            url_line = f"\n【網址】：{m['url']}" if m.get("url") else ""
            case_ctx += f"【{m.get('source', '')} — {m.get('category', '')}】\n{d}{url_line}\n\n"

        # 回答語言邏輯：
        # 1. 若輸入為非中文 → 偵測輸入語言，用輸入語言回答（優先）
        # 2. 若輸入為中文但 UI 語言設為其他 → 用 UI 語言回答
        # 3. 其餘 → 繁體中文
        input_is_chinese = _is_chinese(user_prompt)
        if not input_is_chinese:
            lang_name = "the same language as the user's question (auto-detect)"
            lang_rule = (
                f"\n8. 除「📖 法條依據」區塊中的法條原文須保留繁體中文（避免翻譯造成法律歧義）外，"
                f"其餘所有文字（結論、📂 參考案例、💡 說明）請使用與使用者提問相同的語言撰寫"
                f"（例如使用者用越南文提問就用越南文回答，英文提問就用英文回答）；"
                f"法條原文後方請附上一句該語言的白話翻譯。"
                f"加班費倍率的數字（如 4/3、5/3、1.33、1.66）必須維持原本數值與格式，不可換算。"
            )
        elif ui_lang != "繁體中文":
            lang_name = LANGUAGES.get(ui_lang, "Traditional Chinese")
            lang_rule = (
                f"\n8. 除「📖 法條依據」區塊中的法條原文須保留繁體中文（避免翻譯造成法律歧義）外，"
                f"其餘所有文字（結論、📂 參考案例、💡 說明）請使用「{lang_name}」撰寫；"
                f"法條原文後方請附上一句{lang_name}白話翻譯。"
                f"加班費倍率的數字（如 4/3、5/3、1.33、1.66）必須維持原本數值與格式，不可換算。"
            )
        else:
            lang_name = "Traditional Chinese"
            lang_rule = ""

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
   註明何時會晉升到下一級距。{lang_rule}
"""
        model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=system_prompt)
        return model.generate_content(final_prompt).text

    except Exception as e:
        return f"🚨 異常：{str(e)}"


# ==========================================
# 4. 頁面一：登入 / 註冊
# ==========================================
def show_login_page():
    render_language_switcher()
    lang = st.session_state.ui_lang

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title(t(lang, "app_title"))
    st.subheader(t(lang, "login_subheader"))

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

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title(t(lang, "app_title"))
    st.subheader(t(lang, "landing_subheader"))
    st.write(t(lang, "landing_desc"))
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.info(t(lang, "role_employee_title"))
        st.write(t(lang, "role_employee_desc"))
        if st.button(t(lang, "role_employee_btn"), use_container_width=True):
            st.session_state.user_role        = "Employee"
            st.session_state.page             = "Chat"
            st.session_state.messages         = []
            st.session_state.current_chat_id  = None
            st.rerun()

    with col2:
        st.warning(t(lang, "role_employer_title"))
        st.write(t(lang, "role_employer_desc"))
        if st.button(t(lang, "role_employer_btn"), use_container_width=True):
            st.session_state.user_role        = "Employer"
            st.session_state.page             = "Chat"
            st.session_state.messages         = []
            st.session_state.current_chat_id  = None
            st.rerun()

    st.markdown("<br><br><br>", unsafe_allow_html=True)
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
                resp = query_rag_system(prompt, persona, lang)
                st.markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})

        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = str(uuid.uuid4())

        save_chat(st.session_state.username, st.session_state.current_chat_id, st.session_state.messages)

        st.rerun()


# ==========================================
# 7. 主程式 — 頁面路由
# ==========================================
if not st.session_state.logged_in:
    show_login_page()
elif st.session_state.page == "Landing":
    show_landing_page()
else:
    show_chat_room()
