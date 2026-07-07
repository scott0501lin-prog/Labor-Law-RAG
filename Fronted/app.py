import streamlit as st
import os
import uuid
from dotenv import load_dotenv

from db import init_indexes, register_user, verify_user, save_chat, list_chats, load_chat
from translate import translate, LANGUAGES

# ← 把你的 Gemini API Key 貼在這裡的引號內
GEMINI_API_KEY = "AIzaSyBTnT9W_N9gTuFIud9BObUKmbo3GCLUUsA"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, ".env"))


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
    "translation": None,
    "translation_lang": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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


# ==========================================
# 4. 後端 RAG 引擎 (Gemini + Numpy 搜尋)
# ==========================================
def query_rag_system(user_prompt: str, system_prompt: str) -> str:
    try:
        import google.generativeai as genai

        api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "🚨 找不到 GEMINI_API_KEY，請在 app.py 第 9 行填入金鑰。"

        genai.configure(api_key=api_key)
        model, law_emb, law_texts, law_metas, case_emb, case_texts, case_metas = _load_index()

        l_results = _search(model, user_prompt, law_emb,  law_texts,  law_metas,  top_k=3)
        c_results = _search(model, user_prompt, case_emb, case_texts, case_metas, top_k=3)

        law_ctx = ""
        for d, m, _ in l_results:
            law_ctx += f"【法規內容】：{d}\n【條號】：{m.get('article_no', '')}\n\n"

        case_ctx = ""
        for d, m, _ in c_results:
            url_line = f"\n【網址】：{m['url']}" if m.get("url") else ""
            case_ctx += f"【{m.get('source', '')} — {m.get('category', '')}】\n{d}{url_line}\n\n"

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
"""
        model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=system_prompt)
        return model.generate_content(final_prompt).text

    except Exception as e:
        return f"🚨 異常：{str(e)}"


# ==========================================
# 4. 頁面一：登入 / 註冊
# ==========================================
def show_login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("⚖️ 勞資爭議智慧法務 AI 顧問系統")
    st.subheader("請先登入或建立帳號")

    tab_login, tab_register = st.tabs(["🔐 登入", "📝 註冊"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("帳號")
            password = st.text_input("密碼", type="password")
            if st.form_submit_button("登入", use_container_width=True):
                if not username or not password:
                    st.error("帳號和密碼不能為空")
                else:
                    ok, result = verify_user(username.strip(), password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username  = result
                        st.session_state.page      = "Landing"
                        st.rerun()
                    else:
                        st.error(result)

    with tab_register:
        with st.form("register_form"):
            new_user  = st.text_input("帳號")
            new_pass  = st.text_input("密碼（至少 6 個字元）", type="password")
            new_pass2 = st.text_input("確認密碼", type="password")
            if st.form_submit_button("註冊", use_container_width=True):
                if not new_user or not new_pass:
                    st.error("帳號和密碼不能為空")
                elif len(new_pass) < 6:
                    st.error("密碼至少需要 6 個字元")
                elif new_pass != new_pass2:
                    st.error("兩次密碼輸入不一致")
                else:
                    ok, msg = register_user(new_user.strip(), new_pass)
                    if ok:
                        st.success(msg + " 請切換至「登入」頁面。")
                    else:
                        st.error(msg)


# ==========================================
# 5. 頁面二：身份選擇 (Landing Page)
# ==========================================
def show_landing_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("⚖️ 勞資爭議智慧法務 AI 顧問系統")
    st.subheader("在開始對話前，請選擇您的身份入口：")
    st.write("系統將根據您的身份提供不同的法律檢索邏輯與諮詢語氣。")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.info("### 🙋‍♂️ 我是勞方（員工）")
        st.write("針對薪資費用計算、加班費補償、職災補償、職場調解等個人權益進行諮詢。")
        if st.button("進入勞方諮詢入口", use_container_width=True):
            st.session_state.user_role        = "Employee"
            st.session_state.page             = "Chat"
            st.session_state.messages         = []
            st.session_state.current_chat_id  = None
            st.rerun()

    with col2:
        st.warning("### 🏢 我是資方（雇主 / HR）")
        st.write("針對企業合規制度、解僱流程管控、工時薪資設定、防範被資遣等進行諮詢。")
        if st.button("進入資方法務入口", use_container_width=True):
            st.session_state.user_role        = "Employer"
            st.session_state.page             = "Chat"
            st.session_state.messages         = []
            st.session_state.current_chat_id  = None
            st.rerun()

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.caption(f"👤 已登入：{st.session_state.username} ｜ 🔒 對話紀錄綁定至您的帳號，請放心使用。")


# ==========================================
# 6. 頁面三：AI 聊天室 (Chat Room)
# ==========================================
def show_chat_room():
    if st.session_state.user_role == "Employee":
        title   = "勞方專屬 AI 顧問"
        persona = (
            "你是一位專業勞工法律顧問，專門協助勞工了解並保護自身的勞動權益。"
            "你具備深厚的台灣勞動法律知識，包括勞動基準法、勞工保險條例、職業安全衛生法等。"
            "請以友善、清晰、易懂的語氣回答問題。"
            "回答時必須：①先給結論，②在「📖 法條依據」區塊逐條引用法條號碼與原文，"
            "③在「💡 說明」區塊解釋理由。若資料庫無相關條文，須明確說明，不可捏造條號。"
        )
        btns = [
            ("被惡意資遣？",    "我被資遣了！我的權益是什麼？"),
            ("加班費算法",      "加班費怎麼算？"),
            ("申請勞資調解",    "如何申請勞資調解？"),
            ("職災補償",        "發生職災我可以申請哪些補償？"),
        ]
    else:
        title   = "資方法務 AI 顧問"
        persona = (
            "你是一位專業企業勞動法律顧問，專門協助雇主和HR在合法合規的前提下管理勞資關係。"
            "你具備深厚的台灣勞動法律知識，並能從企業管理角度提供實用建議。"
            "請以專業、嚴謹的語氣回答。"
            "回答時必須：①先給結論，②在「📖 法條依據」區塊逐條引用法條號碼與原文，"
            "③在「💡 說明」區塊解釋理由與風險評估。若資料庫無相關條文，須明確說明，不可捏造條號。"
        )
        btns = [
            ("合法資遣流程",    "合法資遣流程？"),
            ("連續曠職處理",    "員工連續曠職怎麼辦？"),
            ("工時薪資設定",    "如何合法設定工時與薪資？"),
            ("防範被員工索賠",  "如何避免被員工索賠？"),
        ]

    # --- 側邊欄 ---
    with st.sidebar:
        st.title("⚙️ 控制台")
        st.write(f"帳號：**{st.session_state.username}**")
        st.write(f"身份：**{st.session_state.user_role}**")

        if st.button("🆕 新對話", use_container_width=True):
            st.session_state.messages        = []
            st.session_state.current_chat_id = None
            st.rerun()

        if st.button("🔄 切換身份（回首頁）", use_container_width=True):
            st.session_state.page            = "Landing"
            st.session_state.messages        = []
            st.session_state.current_chat_id = None
            st.rerun()

        if st.button("🚪 登出", use_container_width=True):
            for k in ["logged_in", "username", "user_role", "messages", "current_chat_id"]:
                st.session_state[k] = defaults[k]
            st.session_state.page = "Login"
            st.rerun()

        st.divider()
        st.markdown("### 🌐 翻譯最後一則回答")
        lang_choice = st.selectbox("目標語言", list(LANGUAGES.keys()), label_visibility="collapsed")
        if st.button("翻譯", use_container_width=True):
            last_ai = next(
                (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
                None,
            )
            if last_ai:
                with st.spinner("翻譯中..."):
                    st.session_state.translation = translate(last_ai, lang_choice, GEMINI_API_KEY)
                    st.session_state.translation_lang = lang_choice
            else:
                st.warning("尚無 AI 回答可翻譯")

        st.divider()
        st.markdown("### ⚡ 快速發問")
        for label, value in btns:
            if st.button(label, use_container_width=True):
                st.session_state.btn_input = value

        st.divider()
        st.markdown("### 🗂️ 您的歷史紀錄")
        saved_chats = list_chats(st.session_state.username)
        if not saved_chats:
            st.info("無紀錄")
        else:
            for chat in saved_chats:
                cid   = chat["chat_id"]
                label = chat["updated_at"].strftime("%m/%d %H:%M")
                if st.button(label, key=cid, use_container_width=True):
                    st.session_state.messages        = load_chat(st.session_state.username, cid)
                    st.session_state.current_chat_id = cid
                    st.rerun()

    # --- 主聊天區 ---
    st.title(f"⚖️ {title}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 翻譯結果顯示在最後一則 AI 回答下方
    if st.session_state.translation and st.session_state.messages and \
            st.session_state.messages[-1]["role"] == "assistant":
        with st.expander(f"🌐 翻譯結果（{st.session_state.translation_lang}）", expanded=True):
            st.markdown(st.session_state.translation)
            if st.button("關閉翻譯", key="close_translation"):
                st.session_state.translation = None
                st.session_state.translation_lang = None
                st.rerun()

    prompt = st.chat_input("輸入問題...")
    if st.session_state.btn_input:
        prompt = st.session_state.btn_input
        st.session_state.btn_input = None

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                resp = query_rag_system(prompt, persona)
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
