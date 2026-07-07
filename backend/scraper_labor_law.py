"""
勞動基準法爬蟲
資料來源：全國法規資料庫 (law.moj.gov.tw)
輸出：
  - backend/labor_law.json   （原始條文 JSON）
  - backend/law_db/           （ChromaDB，供 RAG 使用）
"""

import json
import os
import re
import time
import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCODE    = "N0030001"   # 勞動基準法
LAW_NAME = "勞動基準法"
LAW_URL  = f"https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={PCODE}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}


# ==========================================
# 1. 取得頁面 HTML
# ==========================================
def fetch_html(url: str) -> BeautifulSoup:
    print(f"[Fetch] {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return BeautifulSoup(resp.text, "html.parser")


# ==========================================
# 2a. 策略一：MOJ 新版 div 結構
#     .col-no / .col-data
# ==========================================
def parse_div_structure(soup: BeautifulSoup) -> list[dict]:
    articles = []
    rows = soup.select("div.row.no-gutters, div.law-reg-content div.row")
    for row in rows:
        no_cell   = row.select_one(".col-no")
        data_cell = row.select_one(".col-data")
        if not no_cell or not data_cell:
            continue
        no_text = no_cell.get_text(strip=True)
        no_match = re.search(r"\d+", no_text)
        if not no_match:
            continue
        no      = no_match.group()
        content = data_cell.get_text("\n", strip=True)
        if content:
            articles.append(_build(no, content))
    return articles


# ==========================================
# 2b. 策略二：MOJ 舊版 table 結構
#     table > tr > td.no-dot / td.law-con
# ==========================================
def parse_table_structure(soup: BeautifulSoup) -> list[dict]:
    articles = []
    current_no, current_lines = None, []

    for row in soup.select("tr"):
        no_td  = row.select_one("td.no-dot")
        con_td = row.select_one("td.law-con")
        if no_td:
            if current_no and current_lines:
                articles.append(_build(current_no, "\n".join(current_lines)))
            raw    = no_td.get_text(strip=True)
            m      = re.search(r"\d+", raw)
            current_no    = m.group() if m else None
            current_lines = []
        if con_td:
            current_lines.append(con_td.get_text("\n", strip=True))

    if current_no and current_lines:
        articles.append(_build(current_no, "\n".join(current_lines)))
    return articles


# ==========================================
# 2c. 策略三：純文字 + regex（最後防線）
#     抓「第 X 條」後的所有文字直到下一條
# ==========================================
def parse_text_fallback(soup: BeautifulSoup) -> list[dict]:
    # 移除不必要的區塊
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")

    # 分割：以「第 數字 條」為分界（支援「之1」等附條）
    parts = re.split(r"(第\s*\d+\s*條(?:之\s*\d+)?)", text)

    articles = []
    i = 1                           # parts[0] 是條文前的頁首雜訊
    while i < len(parts) - 1:
        header  = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""

        no_m = re.search(r"\d+", header)
        if not no_m:
            i += 2
            continue

        no      = no_m.group()
        # 清理：去除連續空白、頁碼雜訊
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        # 只保留有實質內容的條文（>15字）
        if len(content) > 15:
            articles.append(_build(no, content))
        i += 2

    return articles


# ==========================================
# 3. 統一格式
# ==========================================
def _build(no: str, content: str) -> dict:
    return {
        "article_no": no,
        "title"     : f"第 {no} 條",
        "content"   : content,
        "source"    : f"https://law.moj.gov.tw/LawClass/LawSingle.aspx?pcode={PCODE}&flno={no}",
        "law_name"  : LAW_NAME,
    }


# ==========================================
# 4. 主爬蟲：依序嘗試三種策略
# ==========================================
def fetch_articles() -> list[dict]:
    soup = fetch_html(LAW_URL)

    for name, fn in [
        ("div 結構", parse_div_structure),
        ("table 結構", parse_table_structure),
        ("文字 regex", parse_text_fallback),
    ]:
        print(f"[Parser] 嘗試：{name}")
        articles = fn(soup)
        if articles:
            print(f"[Parser] ✅ 成功，共 {len(articles)} 條")
            return articles
        print(f"[Parser] ❌ 無結果，換下一個")

    return []


# ==========================================
# 5. 儲存 JSON
# ==========================================
def save_json(articles: list[dict]) -> str:
    out = os.path.join(BASE_DIR, "labor_law.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[JSON] 已儲存：{out}")
    return out


# ==========================================
# 6. 匯入 ChromaDB
# ==========================================
def load_to_chromadb(articles: list[dict]):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        print("[ChromaDB] 未安裝，跳過。pip install chromadb sentence-transformers")
        return

    db_path = os.path.join(BASE_DIR, "law_db")
    client  = chromadb.PersistentClient(path=db_path)
    emb_fn  = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="shibing624/text2vec-base-chinese"
    )

    try:
        client.delete_collection("labor_law_collection")
        print("[ChromaDB] 已清除舊 collection")
    except Exception:
        pass

    col = client.create_collection(name="labor_law_collection", embedding_function=emb_fn)

    batch = 50
    for i in range(0, len(articles), batch):
        chunk = articles[i: i + batch]
        col.add(
            ids       =[f"art_{a['article_no']}" for a in chunk],
            documents =[f"{a['title']}\n{a['content']}" for a in chunk],
            metadatas =[{"source": a["source"], "law": a["law_name"], "no": a["article_no"]} for a in chunk],
        )
        print(f"[ChromaDB] 匯入 {min(i+batch, len(articles))}/{len(articles)} 條")
        time.sleep(0.05)

    print(f"[ChromaDB] 完成！共 {col.count()} 筆")


# ==========================================
# 7. 主程式
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print(f"  {LAW_NAME} 爬蟲啟動")
    print("=" * 50)

    articles = fetch_articles()

    if not articles:
        print("❌ 三種解析策略均失敗，請截圖給開發者。")
        exit(1)

    save_json(articles)
    load_to_chromadb(articles)

    print("\n✅ 全部完成！")
    print(f"   條文數：{len(articles)}")
    print(f"   JSON  ：{os.path.join(BASE_DIR, 'labor_law.json')}")
    print(f"   DB    ：{os.path.join(BASE_DIR, 'law_db')}")

    # 印出前 3 條預覽
    print("\n── 前 3 條預覽 ──")
    for a in articles[:3]:
        print(f"  [{a['title']}] {a['content'][:60]}...")
