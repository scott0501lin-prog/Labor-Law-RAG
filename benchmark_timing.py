"""
分段計時 benchmark — 不依賴 Streamlit
執行：python benchmark_timing.py
"""

import os
import sys
import time
import json
import numpy as np
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMB_DIR = os.path.join(BASE_DIR, "backend", "embeddings")

# ── 測試題目（來自題庫 題號 1-5 中文、6-10 非中文）──────────────────────
QUESTIONS = [
    ("中文", "加班費要怎麼算？"),
    ("中文", "我做滿一年被資遣，可以領多少資遣費？"),
    ("中文", "公司要資遣我，需要提前多久通知？"),
    ("中文", "我在工作時受傷，公司要怎麼補償我？"),
    ("中文", "特別休假是怎麼計算天數的？"),
    ("英文",   "How much is overtime pay if I work on a rest day?"),
    ("越南文",  "Nếu công ty sa thải tôi trái luật, tôi phải làm gì?"),
    ("印尼文",  "Berapa uang pesangon yang saya dapatkan jika di-PHK?"),
    ("菲律賓語","Ano ang gagawin ko kung nasaktan ako sa trabaho?"),
    ("韓文",   "특별휴가는 며칠 받을 수 있나요？"),
]

SYSTEM_PROMPT = (
    "你是一位專業勞工法律顧問，專門協助勞工了解並保護自身的勞動權益。"
    "請以友善、清晰、易懂的語氣回答問題。"
)

# ── 關鍵字常數（同 app.py）──────────────────────────────────────────────
COMP_LEAVE_KEYWORDS     = ["補休", "換休", "以休代薪", "代替休假"]
COMP_LEAVE_FORCE_ARTICLES = {"第 32-1 條"}
OVERTIME_KEYWORDS       = ["加班", "延長工作時間", "延長工時", "休息日工作", "例假工作", "假日加班"]
OVERTIME_FORCE_ARTICLES = {"第 24 條", "第 32 條", "第 36 條"}
TERMINATION_KEYWORDS    = ["資遣", "預告", "解僱", "解雇", "終止契約", "預告期間", "預告工資",
                           "合法解僱", "解僱事由", "終止勞動契約"]
TERMINATION_FORCE_ARTICLES = {"第 11 條", "第 12 條", "第 13 條", "第 14 條", "第 16 條", "第 17 條"}
LEAVE_KEYWORDS          = ["特別休假", "年假", "休假天數", "特休"]
LEAVE_FORCE_ARTICLES    = {"第 38 條"}
HOLIDAY_KEYWORDS        = ["國定假日", "例假", "休息日", "國假", "補休", "假日出勤", "假日上班", "輪班假日"]
HOLIDAY_FORCE_ARTICLES  = {"第 36 條", "第 37 條", "第 39 條"}
RESTDAY_OVERTIME_FORCE_TITLES = {"勞動部勞動條 2 字第 1050030466 號書函"}


def _is_chinese(text: str) -> bool:
    chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
    return chinese_chars / max(len(text), 1) >= 0.3


def _search_split(model, query, embeddings, texts, metas, top_k=5):
    """回傳 (results, encode_sec, compare_sec)"""
    t_enc0 = time.perf_counter()
    q_emb = model.encode([query], normalize_embeddings=True)
    t_enc1 = time.perf_counter()
    scores = (embeddings @ q_emb.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    t_cmp1 = time.perf_counter()
    results = [(texts[i], metas[i], float(scores[i])) for i in top_idx]
    return results, (t_enc1 - t_enc0), (t_cmp1 - t_enc1)


def _force_include_articles(query, results, texts, metas, keywords, force_article_nos, source="勞動基準法"):
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


def load_index():
    from sentence_transformers import SentenceTransformer
    print("載入嵌入模型...", end=" ", flush=True)
    t0 = time.perf_counter()
    model = SentenceTransformer("shibing624/text2vec-base-chinese")
    print(f"{time.perf_counter()-t0:.2f}s")

    def _load(prefix):
        emb  = np.load(os.path.join(EMB_DIR, f"{prefix}_embeddings.npy"))
        with open(os.path.join(EMB_DIR, f"{prefix}_docs.json"), encoding="utf-8") as f:
            data = json.load(f)
        return emb, data["texts"], data["metas"]

    print("載入向量索引...", end=" ", flush=True)
    t1 = time.perf_counter()
    law_emb,  law_texts,  law_metas  = _load("law")
    case_emb, case_texts, case_metas = _load("cases")
    print(f"{time.perf_counter()-t1:.2f}s")
    return model, law_emb, law_texts, law_metas, case_emb, case_texts, case_metas


def run_one(client, model, law_emb, law_texts, law_metas,
            case_emb, case_texts, case_metas, user_prompt):
    t0 = time.perf_counter()
    timings = {}

    # ── 翻譯 ──────────────────────────────────────────────────────────
    search_query = user_prompt
    if not _is_chinese(user_prompt):
        try:
            search_query = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"請將以下使用者問題翻譯成繁體中文，只需要輸出翻譯結果，不要加任何說明：\n\n{user_prompt}"
            ).text.strip()
        except Exception:
            search_query = user_prompt
    timings["翻譯"] = time.perf_counter() - t0

    # ── 法條向量搜尋 ───────────────────────────────────────────────────
    t1 = time.perf_counter()
    l_results, law_enc, law_cmp = _search_split(model, search_query, law_emb, law_texts, law_metas)
    timings["法條encode"] = law_enc
    timings["法條compare"] = law_cmp

    # ── 強制帶入（法條）───────────────────────────────────────────────
    t2 = time.perf_counter()
    if any(k in search_query for k in COMP_LEAVE_KEYWORDS):
        l_results = _force_include_articles(search_query, l_results, law_texts, law_metas, COMP_LEAVE_KEYWORDS, COMP_LEAVE_FORCE_ARTICLES)
    else:
        l_results = _force_include_articles(search_query, l_results, law_texts, law_metas, OVERTIME_KEYWORDS, OVERTIME_FORCE_ARTICLES)
    l_results = _force_include_articles(search_query, l_results, law_texts, law_metas, TERMINATION_KEYWORDS, TERMINATION_FORCE_ARTICLES)
    l_results = _force_include_articles(search_query, l_results, law_texts, law_metas, LEAVE_KEYWORDS, LEAVE_FORCE_ARTICLES)
    l_results = _force_include_articles(search_query, l_results, law_texts, law_metas, HOLIDAY_KEYWORDS, HOLIDAY_FORCE_ARTICLES)
    timings["強制帶入(法)"] = time.perf_counter() - t2

    # ── 案例向量搜尋 ───────────────────────────────────────────────────
    c_results, case_enc, case_cmp = _search_split(model, search_query, case_emb, case_texts, case_metas)
    timings["案例encode"] = case_enc
    timings["案例compare"] = case_cmp

    # ── 強制帶入（案例）───────────────────────────────────────────────
    t3 = time.perf_counter()
    c_results = _force_include_cases(search_query, c_results, case_texts, case_metas, OVERTIME_KEYWORDS, RESTDAY_OVERTIME_FORCE_TITLES)
    timings["強制帶入(案)"] = time.perf_counter() - t3

    # ── 組 prompt ─────────────────────────────────────────────────────
    t4 = time.perf_counter()
    law_ctx = ""
    for d, m, _ in l_results:
        law_ctx += f"【法規內容】：{d}\n【條號】：{m.get('source','')} {m.get('article_no','')}\n\n"
    case_ctx = ""
    for d, m, _ in c_results:
        url_line = f"\n【網址】：{m['url']}" if m.get("url") else ""
        case_ctx += f"【{m.get('source','')} — {m.get('category','')}】\n{d}{url_line}\n\n"
    final_prompt = f"【相關法規條文】：\n{law_ctx}\n【相關案例】：\n{case_ctx}\n使用者問題：{user_prompt}"
    timings["組prompt"] = time.perf_counter() - t4

    # ── Gemini 生成 ───────────────────────────────────────────────────
    from google.genai import types
    t5 = time.perf_counter()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    timings["Gemini生成"] = time.perf_counter() - t5
    timings["總計"] = time.perf_counter() - t0

    return timings, response.text[:60]


def main():
    from google import genai

    if not GEMINI_API_KEY:
        print("❌ 找不到 GEMINI_API_KEY，請確認 .env 檔案")
        sys.exit(1)

    client = genai.Client(api_key=GEMINI_API_KEY)
    model, law_emb, law_texts, law_metas, case_emb, case_texts, case_metas = load_index()

    COLS = ["翻譯", "法條encode", "法條compare", "強制帶入(法)", "案例encode", "案例compare", "強制帶入(案)", "組prompt", "Gemini生成", "總計"]
    rows = []

    print(f"\n開始跑 {len(QUESTIONS)} 題...\n")
    for i, (lang, q) in enumerate(QUESTIONS, 1):
        print(f"  [{i:02d}] {lang}: {q[:35]}...", end=" ", flush=True)
        try:
            timings, snippet = run_one(client, model, law_emb, law_texts, law_metas,
                                       case_emb, case_texts, case_metas, q)
            rows.append((i, lang, q[:20], timings))
            print(f"總計 {timings['總計']:.2f}s ✓")
        except Exception as e:
            print(f"❌ {e}")
            rows.append((i, lang, q[:20], {}))

    # ── 列印表格 ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'#':>2}  {'語言':<6}  {'問題(前20字)':<22}", end="")
    for c in COLS:
        print(f"  {c:>10}", end="")
    print()
    print("-" * 100)

    totals = {c: [] for c in COLS}
    for no, lang, qshort, t in rows:
        print(f"{no:>2}  {lang:<6}  {qshort:<22}", end="")
        for c in COLS:
            val = t.get(c)
            if val is not None:
                print(f"  {val:>9.3f}s", end="")
                totals[c].append(val)
            else:
                print(f"  {'—':>10}", end="")
        print()

    print("-" * 100)
    print(f"{'平均':>32}", end="")
    for c in COLS:
        if totals[c]:
            print(f"  {sum(totals[c])/len(totals[c]):>9.3f}s", end="")
        else:
            print(f"  {'—':>10}", end="")
    print()
    print("=" * 100)
    print("\n同步寫入 rag_timing.log（透過 Streamlit 執行時會自動記錄）\n")


if __name__ == "__main__":
    main()
