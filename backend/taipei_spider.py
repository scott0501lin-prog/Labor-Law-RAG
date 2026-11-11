"""
台北市勞動局爬蟲  (requests + BeautifulSoup)
網址：https://bola.gov.taipei
翻頁：ASP.NET __doPostBack POST 同一 URL
SSL ：verify=False（憑證缺陷）

執行：python taipei_spider.py
輸出：taipei_cases.json
"""

import re, json, time, random, logging
from urllib.parse import urljoin

import requests, urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 常數 ──────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}
SKIP_KEYWORDS = ["法規全文","法條","條文","立法院","法律條文","勞動基準法第","勞基法第","施行細則全文"]
DELAY_MIN   = 2.5
DELAY_MAX   = 5.0
MAX_PAGES   = 10
MAX_PER_CAT = 80
BASE        = "https://bola.gov.taipei"

TARGETS = [
    {"category": "常見問答", "url": f"{BASE}/News.aspx?n=FDEDF5DCB0A26A46&sms=87415A8B9CE81B16"},
    {"category": "新聞稿",   "url": f"{BASE}/News.aspx?n=098B457D83590D7F&sms=72544237BBE4C5F6"},
    {"category": "最新消息", "url": f"{BASE}/News.aspx?n=4A519A52B9D376CD&sms=78D644F2755ACCAA"},
]

# ── 民國 → 西元 ───────────────────────────────────
ROC_PATTERNS = [
    re.compile(r"(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})"),
    re.compile(r"民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"),
    re.compile(r"^(\d{3})(\d{2})(\d{2})$"),
]
def roc_to_iso(raw: str) -> str:
    raw = raw.strip()
    if not raw: return ""
    if re.match(r"\d{4}[-/]\d{2}[-/]\d{2}", raw):
        return raw.replace("/", "-")
    for pat in ROC_PATTERNS:
        m = pat.search(raw)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 200: y += 1911
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return raw

# ── 工具 ──────────────────────────────────────────
def delay(): time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def make_session():
    """Session 自動維護 Cookie，解決 ASP.NET VIEWSTATE MAC 驗證問題"""
    s = requests.Session()
    s.headers.update(HEADERS)
    s.verify = False   # 台北市政府 SSL 憑證缺陷
    return s

def fetch(session, url, retries=3, **kwargs):
    method = "POST" if "data" in kwargs else "GET"
    for i in range(retries):
        try:
            r = session.request(method, url, timeout=30, **kwargs)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            logger.warning(f"[第{i+1}次失敗] {url}：{e}")
            if i < retries - 1: time.sleep(random.uniform(5, 10))
    return None

def clean(t): return re.sub(r"\s+", " ", t).strip()

def skip(title, url): return any(k in title+url for k in SKIP_KEYWORDS)

def keywords(title):
    terms = ["加班","資遣","解僱","薪資","工資","休假","特休","勞保","勞退",
             "職災","懷孕","育嬰","產假","陪產","工時","契約","試用期",
             "競業","退休","罷工","工會","最低工資","例假","國定假日",
             "排班","夜班","調職","霸凌","性騷擾"]
    return [t for t in terms if t in title]

# ── ASP.NET 翻頁 ───────────────────────────────────
def aspnet_form(soup, page_num):
    def v(name):
        el = soup.find("input", {"name": name})
        return el["value"] if el and el.get("value") else ""
    return {
        "__EVENTTARGET":        "ctl00$ContentPlaceHolder1$GridView1",
        "__EVENTARGUMENT":      f"Page${page_num}",
        "__VIEWSTATE":          v("__VIEWSTATE"),
        "__VIEWSTATEGENERATOR": v("__VIEWSTATEGENERATOR"),
        "__EVENTVALIDATION":    v("__EVENTVALIDATION"),
    }

def has_next(soup, cur):
    for a in soup.select("tr td[colspan] a, .pagination a, table a"):
        t = a.get_text(strip=True)
        if t.isdigit() and int(t) > cur: return True
        if t in (">", "下一頁", "Next"): return True
    return False

# ── 解析 ──────────────────────────────────────────
def parse_list(soup):
    """
    列表頁只取 title + url。
    ★ 最後一欄是「發布科室」不是日期，日期在內文頁抓。
    """
    items = []
    for row in soup.select("table tr:not(:first-child)"):
        a = row.select_one("a[href]")
        if not a: continue
        title = clean(a.get_text())
        href  = a.get("href", "")
        if not href or href.startswith("javascript") or len(title) < 2: continue
        items.append({"title": title, "url": urljoin(BASE, href)})
    return items

def parse_content(soup) -> tuple[str, str]:
    """
    回傳 (content, date_iso)。
    日期從內文頁文字中找「資料更新：YYY-MM-DD」等格式。
    """
    content = ""
    for sel in [".area-essay", ".Content",
                "#ctl00_ContentPlaceHolder1_NewsContent",
                "article", ".news-content", ".main-content"]:
        el = soup.select_one(sel)
        if el:
            content = clean(el.get_text())
            break
    if not content:
        m = soup.select_one("main") or soup.select_one("body")
        content = clean(m.get_text()) if m else ""

    date_iso = ""
    full = soup.get_text()
    for pat in [r"資料更新[：:]\s*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})",
                r"更新日期[：:]\s*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})",
                r"發布日期[：:]\s*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})",
                r"公告日期[：:]\s*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})"]:
        m = re.search(pat, full)
        if m:
            date_iso = roc_to_iso(m.group(1))
            break
    return content, date_iso

# ── 主爬蟲 ────────────────────────────────────────
def crawl():
    results = []
    logger.info("▶ 台北市勞動局")

    for target in TARGETS:
        cat, base_url = target["category"], target["url"]
        logger.info(f"  分類：{cat}")
        session  = make_session()
        collected = []
        page = 1
        soup = fetch(session, base_url)

        while soup and page <= MAX_PAGES:
            items = parse_list(soup)
            logger.info(f"    第 {page} 頁，{len(items)} 筆")
            collected.extend(items)
            if len(collected) >= MAX_PER_CAT: break
            if not has_next(soup, page): break
            page += 1
            delay()
            soup = fetch(session, base_url, data=aspnet_form(soup, page))

        for item in collected[:MAX_PER_CAT]:
            if skip(item["title"], item["url"]):
                logger.info(f"    [跳過-法條] {item['title'][:30]}")
                continue
            cs = fetch(session, item["url"])
            if not cs: delay(); continue
            content, date_iso = parse_content(cs)
            if len(content) < 30:
                logger.warning(f"    [內容過短] {item['title'][:30]}")
                delay(); continue
            results.append({
                "source":      "台北市勞動局",
                "source_type": "gov",
                "category":    cat,
                "title":       item["title"],
                "url":         item["url"],
                "date":        date_iso,
                "content":     content,
                "keywords":    keywords(item["title"]),
            })
            logger.info(f"    ✓ [{date_iso}] {item['title'][:40]}")
            delay()

    logger.info(f"完成，共 {len(results)} 筆")
    return results

def main():
    data = crawl()
    with open("taipei_cases.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 已存至 taipei_cases.json")

if __name__ == "__main__":
    main()