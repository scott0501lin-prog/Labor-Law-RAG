"""
勞動部 常見問答爬蟲（只爬與勞基法／勞退條例／災保法相關的 16 個次分類）
網址：https://www.mol.gov.tw/1607/28690/2282/nodeListSearch

與舊版 scrapers/mol_qa_spider.py 的差異：
  - 舊版寫死 9 個列表網址，其中數個標籤錯置（如「預告期間」實為退休準備金監督委員會）、
    「工時加班」實為友善列印按鈕、基本工資網址已 410
  - 新版直接依官方「主分類／次分類」節點 id 翻頁（/1607/28690/2282/<L1>/<L2>/?Page=&PageSize=），
    分類名稱即網站上的名稱
  - 以文章節點 id 去重；同一題出現在多個次分類時，categories 會列出全部

執行：python backend/scraper2/spiders/mol_faq_spider.py
輸出：backend/scraper2/data/mol_faq_cases.json
"""

import io, re, sys, json, time, random, logging, os
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # backend/scraper2/data
OUT_FILE = os.path.join(DATA_DIR, "mol_faq_cases.json")

BASE     = "https://www.mol.gov.tw"
LIST_URL = BASE + "/1607/28690/2282/{l1}/{l2}/?Page={page}&PageSize={size}"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}
DELAY_MIN, DELAY_MAX = 1.5, 3.0
PAGE_SIZE = 40          # 網站下拉選單最大值
MAX_PAGES = 10

# ── 要爬的次分類：(主分類 id, 次分類 id, 主分類名, 次分類名) ────────
CATEGORIES = [
    ("2284", "2286",  "勞動條件、就業平等", "勞動基準法適用"),
    ("2284", "2290",  "勞動條件、就業平等", "工資"),
    ("2284", "2292",  "勞動條件、就業平等", "工時(休息、休假、請假)"),
    ("2284", "2294",  "勞動條件、就業平等", "童工、母性保護、技術生"),
    ("2284", "2296",  "勞動條件、就業平等", "職業災害補償"),
    ("2284", "2298",  "勞動條件、就業平等", "工作規則"),
    ("2326", "2288",  "勞動關係",           "勞動契約"),
    ("2340", "2346",  "勞動保險",           "勞工職業災害保險"),
    ("2302", "2304",  "勞動福祉、退休",     "勞退(舊制)適用對象及工作年資"),
    ("2302", "2306",  "勞動福祉、退休",     "勞動基準法退休要件"),
    ("2302", "2308",  "勞動福祉、退休",     "勞動基準法退休金給付"),
    ("2302", "2310",  "勞動福祉、退休",     "資遣費給付"),
    ("2302", "2314",  "勞動福祉、退休",     "勞工退休準備金監督委員會"),
    ("2302", "2316",  "勞動福祉、退休",     "新制勞工個人退休金專戶之提繳"),
    ("2302", "86592", "勞動福祉、退休",     "勞工退休準備金提撥對象"),
    ("2302", "86595", "勞動福祉、退休",     "勞工退休準備金按月、足額提撥及動支"),
]
# 不爬：性別平等工作／性騷擾防治／就業歧視／身心障礙者職場合理調整（其他法規）、
# 勞資會議／團體協約／裁決（集體勞動法）、勞工保險／就業保險（勞保條例、就保法）

# ── 排除的題目（標題包含下列字串即略過）─────────────────────
EXCLUDE_TITLES = [
    # 整份問答集塞成一題（5,000～15,000 字），主題冷門、會稀釋檢索
    "大廈管理委員會適用勞基法疑義常見問答專區",
    "指定私立各級學校編制外之工作者",
    "政府機關(構)運用勞務承攬勞動權益保障問答",
    "外國專業人才延攬及僱用法自115年1月1日施行後",
    # 涉及性別平等工作法（不在資料範圍內）
    "依性別平等工作法規定請家庭照顧假、生理假、產假期間之工資",
    "天然災害（如颱風）發生時或過後，勞工因照顧家人、整理家園",
    # 依據為就業服務法（資遣通報），不在資料範圍內
    "雇主資遣勞工時，應如何辦理資遣通報",
]

def excluded(title: str) -> bool:
    return any(k in title for k in EXCLUDE_TITLES)

# ── 工具 ──────────────────────────────────────────
def delay(): time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def fetch(session, url, retries=3):
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            logger.warning(f"  請求失敗（{i + 1}/{retries}）：{e}")
            time.sleep(3 * (i + 1))
    return None

def find_date(soup, label):
    m = re.search(label + r"\s*[:：]\s*(\d{4}-\d{2}-\d{2})", soup.get_text(" "))
    return m.group(1) if m else ""

# ── 解析 ──────────────────────────────────────────
def parse_list(soup, l1, l2):
    """回傳 [(節點 id, 標題, 網址)]；只取本次分類底下的文章連結（排除側欄、導覽）。"""
    pat = re.compile(rf"/1607/28690/2282/{l1}/{l2}/(\d+)/post")
    items, seen = [], set()
    for a in soup.find_all("a", href=True):
        m = pat.search(a["href"])
        if not m or m.group(1) in seen: continue
        seen.add(m.group(1))
        items.append((m.group(1), clean(a.get_text()), urljoin(BASE, a["href"])))
    return items

def list_total(soup):
    m = re.search(r"共\s*(\d+)\s*筆", soup.get_text(" "))
    return int(m.group(1)) if m else None

def parse_content(soup):
    """回傳 (question, answer, 發布單位, 發布日期, 更新日期)。"""
    question = answer = ""
    for tr in soup.select("section.cp table.scroltable tr"):
        th, td = tr.find("th"), tr.find("td")
        if not th or not td: continue
        label = clean(th.get_text())
        if label == "問題":
            question = clean(td.get_text(" "))
        elif label == "答案":
            # 網站編輯殘留字樣（如「(將新增超連結)」）
            answer = clean(td.get_text(" ")).replace("(將新增超連結)", "")
    unit = re.search(r"發布單位\s*[:：]\s*(\S+)", soup.get_text(" "))
    return (question, answer, unit.group(1) if unit else "",
            find_date(soup, "發布日期"), find_date(soup, "更新日期"))

# ── 主爬蟲 ────────────────────────────────────────
def crawl_lists(session):
    found = {}
    for l1, l2, main_cat, sub_cat in CATEGORIES:
        logger.info(f"▶ {main_cat} / {sub_cat}")
        cat_ids, total = set(), None
        for page in range(1, MAX_PAGES + 1):
            soup = fetch(session, LIST_URL.format(l1=l1, l2=l2, page=page, size=PAGE_SIZE))
            delay()
            if not soup: break
            if total is None: total = list_total(soup)
            new = [it for it in parse_list(soup, l1, l2) if it[0] not in cat_ids]
            if not new: break
            for nid, title, url in new:
                cat_ids.add(nid)
                if excluded(title):
                    logger.info(f"    排除：{title[:40]}")
                    continue
                rec = found.setdefault(nid, {"title": title, "url": url,
                                             "main_category": main_cat, "categories": []})
                rec["categories"].append(sub_cat)
            if total is not None and len(cat_ids) >= total: break
        flag = "" if total is None or len(cat_ids) == total else f"  ⚠ 網站顯示 {total} 筆"
        logger.info(f"    抓到 {len(cat_ids)} 題{flag}")
    return found

def crawl():
    session = requests.Session()
    found = crawl_lists(session)
    logger.info(f"列表完成：不重複 {len(found)} 題，開始抓內文")

    results = []
    for n, (nid, it) in enumerate(found.items(), 1):
        soup = fetch(session, it["url"])
        delay()
        if not soup:
            logger.warning(f"  [{n}] 內文抓取失敗：{it['url']}")
            continue
        question, answer, unit, published, updated = parse_content(soup)
        if len(answer) < 10:
            logger.warning(f"  [{n}] 答案過短或抓不到，略過：{it['title'][:30]}（{it['url']}）")
            continue
        results.append({
            "source":        "勞動部",
            "source_type":   "gov",
            "category":      f"常見問答-{it['categories'][0]}",
            "main_category": it["main_category"],
            "categories":    it["categories"],
            "title":         question or it["title"],
            "url":           it["url"],
            "date":          published,
            "updated":       updated,
            "unit":          unit,
            "content":       answer,
            "keywords":      [],
        })
        if n % 20 == 0: logger.info(f"  內文進度 {n}/{len(found)}")
    return results

def main():
    data = crawl()
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 共 {len(data)} 筆，已存至 {OUT_FILE}")

if __name__ == "__main__":
    main()
