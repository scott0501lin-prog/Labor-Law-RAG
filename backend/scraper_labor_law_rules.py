"""
勞動基準法施行細則爬蟲
資料來源：全國法規資料庫 (law.moj.gov.tw)
輸出（獨立檔案，不會動到既有的 labor_law_cleaned.json）：
  - backend/labor_law_rules.json          （原始條文 JSON，格式同 labor_law.json）
  - backend/labor_law_rules_cleaned.json  （embedding 用格式，同 labor_law_cleaned.json）
"""

import json
import os
import re
import ssl
import sys
import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCODE    = "N0030002"   # 勞動基準法施行細則
LAW_NAME = "勞動基準法施行細則"
LAW_URL  = f"https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={PCODE}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}


class _RelaxedX509Adapter(HTTPAdapter):
    """law.moj.gov.tw 的憑證鏈缺少 Subject Key Identifier 擴充欄位，
    新版 OpenSSL（3.2+）預設的嚴格模式會因此擋下連線。這裡只關閉
    VERIFY_X509_STRICT 這一項嚴格檢查，憑證鏈驗證與網域比對仍正常進行，
    不是整個關閉 SSL 驗證。"""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def _make_session() -> requests.Session:
    session = requests.Session()
    session.mount("https://", _RelaxedX509Adapter())
    return session

# 條號可能帶「之X」附條，MOJ 網站本身以連字號呈現（例如「第 7-1 條」），
# 舊版 scraper_labor_law.py 用 re.search(r"\d+", ...) 只抓得到「7」，
# 會跟正牌第7條的 article_no 撞號，這裡改用完整比對含連字號的條號。
ARTICLE_NO_RE = re.compile(r"第\s*(\d+(?:-\d+)?)\s*條")
FLNO_RE       = re.compile(r"[?&]flno=([^&]+)")


# ==========================================
# 1. 取得頁面 HTML
# ==========================================
def fetch_html(url: str) -> BeautifulSoup:
    print(f"[Fetch] {url}")
    resp = _make_session().get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    return BeautifulSoup(resp.text, "html.parser")


def _extract_no(no_cell) -> str | None:
    """優先從連結網址的 flno 參數取條號（MOJ 官方編號，含連字號最準），
    抓不到再退回比對條號文字。"""
    link = no_cell.select_one("a[href*='flno=']")
    if link:
        m = FLNO_RE.search(link.get("href", ""))
        if m:
            return m.group(1)
    m = ARTICLE_NO_RE.search(no_cell.get_text(strip=True))
    return m.group(1) if m else None


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
        no = _extract_no(no_cell)
        if not no:
            continue
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
            current_no    = _extract_no(no_td)
            current_lines = []
        if con_td:
            current_lines.append(con_td.get_text("\n", strip=True))

    if current_no and current_lines:
        articles.append(_build(current_no, "\n".join(current_lines)))
    return articles


# ==========================================
# 2c. 策略三：純文字 + regex（最後防線）
#     抓「第 X 條」或「第 X-Y 條」後的所有文字直到下一條
# ==========================================
def parse_text_fallback(soup: BeautifulSoup) -> list[dict]:
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    parts = re.split(r"(第\s*\d+(?:-\d+)?\s*條)", text)

    articles = []
    i = 1
    while i < len(parts) - 1:
        header  = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""

        no_m = ARTICLE_NO_RE.search(header)
        if not no_m:
            i += 2
            continue

        no      = no_m.group(1)
        content = re.sub(r"[ \t]+", " ", content)
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        if len(content) > 10:
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
# 5. 儲存 JSON（獨立檔案，原始格式 + embedding 用清洗格式）
# ==========================================
def save_json(articles: list[dict]) -> str:
    out = os.path.join(BASE_DIR, "labor_law_rules.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"[JSON] 已儲存：{out}")
    return out


def save_cleaned_json(articles: list[dict]) -> str:
    cleaned = [
        {
            "source"    : LAW_NAME,
            "article_no": a["title"],
            "text"      : f"【{LAW_NAME} {a['title']}】\n{a['content']}",
        }
        for a in articles
    ]
    out = os.path.join(BASE_DIR, "labor_law_rules_cleaned.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"[JSON] 已儲存：{out}")
    return out


# ==========================================
# 6. 主程式
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
    save_cleaned_json(articles)

    print("\n✅ 全部完成！")
    print(f"   條文數：{len(articles)}")
    print(f"   JSON    ：{os.path.join(BASE_DIR, 'labor_law_rules.json')}")
    print(f"   Cleaned：{os.path.join(BASE_DIR, 'labor_law_rules_cleaned.json')}")

    print("\n── 前 3 條預覽 ──")
    for a in articles[:3]:
        print(f"  [{a['title']}] {a['content'][:60]}...")
