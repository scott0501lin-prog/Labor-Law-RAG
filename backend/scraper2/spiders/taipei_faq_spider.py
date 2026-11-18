"""
臺北市勞動局 常見問答爬蟲（只爬與勞動基準法相關的分類）
網址：https://bola.gov.taipei/News.aspx?n=FDEDF5DCB0A26A46&sms=87415A8B9CE81B16

與舊版 scrapers/taipei_spider.py 的差異：
  - 舊版用 ASP.NET postback 翻頁，網站一直回傳第 1 頁 → 同一批資料重複 4 次
  - 新版依分類（_CSN 參數）+ GET 翻頁（page / PageSize），並以文章 id（s 參數）去重
  - 同一題出現在多個分類時，categories 會列出全部分類

執行：python backend/scraper2/spiders/taipei_faq_spider.py
輸出：backend/scraper2/data/taipei_faq_cases.json
"""

import io, re, sys, json, time, random, logging
import os
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)

OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # backend/scraper2/data
OUT_FILE = os.path.join(OUT_DIR, "taipei_faq_cases.json")

BASE     = "https://bola.gov.taipei"
LIST_URL = f"{BASE}/News.aspx?n=FDEDF5DCB0A26A46&sms=87415A8B9CE81B16"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}
DELAY_MIN, DELAY_MAX = 1.5, 3.0
PAGE_SIZE = 200
MAX_PAGES = 20

# ── 要爬的分類（名稱 → _CSN）─────────────────────────────────
# 只保留與勞動基準法直接相關者
CATEGORIES = {
    "工作規則":                 "98D817529BB6D8B1",
    "勞工退休準備金":           "71F09BDD5807B858",
    "勞動基準法第84條之1約定書": "90A5AC398DE095C9",
    "勞動契約、調動、留職停薪":   "026199D6B5AC5A6A",
    "勞工休假、請假等相關規定":   "0FAF18998E49EA5F",
    "職業災害":                 "ACA764CB415B8B1B",
    "減少工時":                 "8D7CC77DC2122374",
    "天然災害":                 "FD3873E6B806811C",
    "技術生":                   "71213669AD2C885B",
    "資遣費":                   "0845928A6469CE08",
    "適用勞動基準法":           "3C942065A1C7CDF1",
    "工時、工資、獎金等相關規定": "784C593755F82002",
    "居家工作":                 "BC81A94E33B1EFC5",
}
# 不爬（屬其他法規或非法令問答）：工會、團體協約／職工福利／失業勞工子女就學費用補助／
# 勞資爭議調解、仲裁／勞工權益基金／勞工法律諮詢／職業安全衛生／資遣通報／失業認定／
# 私立就業服務機構／就業歧視／職場性騷擾／勞動教育補助／照護補助／大量解僱／
# 勞工住宅貸款／狂犬病／性別平等工作／其他

# ── 排除的題目：答覆只轉介網站／其他機關，沒有實質回答（標題包含下列字串即略過）──
EXCLUDE_TITLES = [
    # 純轉介
    "勞動局是否有提供相關範例供參",                 # 工作規則範例 → 網站下載
    "公司如欲補繳勞工退休準備金，但並無存款單",       # → 臺灣銀行
    "繳款存款單應如何取得",                         # → 臺灣銀行
    "有關勞工退休準備金稽催之相關Q＆A可在何處查詢",   # → 網站
    "有關適用勞動基準法第84條之1工作者，其工時限制為何", # → 一覽表
    "在事業場所外從事工作之勞工，其工作時間認定及出勤紀錄", # → 指導原則
    "對加班費或假日工資計算如有疑義，有無相關管道",     # → 線上試算
    "線上試算或勞動精算師APP之計算依據",              # → APP 說明
    # 大部分為轉介，僅一句實質內容
    "如何變更勞工退休準備金監督委員會",
    "如已經提足員工退休金，是否還要補繳",
    "如勞工被迫實施減少工時，造成權益受損",
    "資遣費要扣稅嗎",
    # 與勞動部常見問答同題同答（保留勞動部版本：中央主管機關、答案通常較完整）
    "勞工每月工資為最低工資29,500元（含全勤獎金），如勞工請事假如何扣薪",
    "事業單位欲降低工作規則中之勞動條件，如何處理",
    "如例假及休息日未出勤，雇主是否應另外加給休息日工資",
    "事業單位全部或局部遷至他市，其原核備之工作規則是否需重新報備",
    "休息日出勤是否需符合勞動基準法第32條第1項所定程序",
    "事業單位停工，其停工期間工資應如何發給",
    "例假與休息日之差別為何",
    "之部分工時勞工，是否有休息日規定的適用",
    "勞動基準法第36條所定之例假、休息日，如何與一般正常工時或彈性工時配套執行",
    "休息日當天，雇主是否應該要照給薪水",
    # 臺北市自治條例（非勞基法），其他縣市不適用
    "臺北市工作者職業災害慰問金發放額度",
    "臺北市勞工職業災害慰問金如何申請",
    "申請臺北市勞工職業災害慰問金應備文件",
    "申請臺北市勞工職業災害慰問金之資格",
    "臺北市外送平台業者管理自治條例",
    # 向臺北市勞動局辦理之行政手續（非勞資權利義務）
    "事業單位如何報核工作規則",
    "事業單位主體所在地與分散地所屬分支機構工作規則之報備疑義",
    "事業單位招收技術生之報備疑義",
    "如何設立勞工退休準備金",
    "事業單位申請退還勞工退休準備金之程序",
    "事業單位勞工退休準備金資料變更時，是否需報請主管機關核備",
    "若收到勞動局逾6個月未按月提撥之催繳公文",
    "公司通報減少工時後，何時會收到回覆",
    "員工是否可查詢公司有無通報減少工時",
    "公司原本實施減少工時並通報勞動局，後因景氣回升",
]
# 答覆結尾固定附加的聯絡資訊（對檢索是雜訊，且各題相同）
FOOTER_RE = re.compile(r"\s*聯絡科室[：:].*$", re.S)

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

def roc_to_iso(raw: str) -> str:
    m = re.match(r"(\d{2,3})-(\d{1,2})-(\d{1,2})", raw or "")
    if not m: return raw or ""
    y, mo, d = int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))
    return f"{y:04d}-{mo:02d}-{d:02d}"

def item_id(url: str) -> str:
    return parse_qs(urlparse(url).query).get("s", [url])[0]

# ── 解析 ──────────────────────────────────────────
def parse_list(soup):
    items = []
    # 限定常見問答（n=FDEDF5DCB0A26A46），排除每頁側欄都有的其他單元連結
    for a in soup.select('a[href*="News_Content.aspx"][href*="n=FDEDF5DCB0A26A46"]'):
        title = clean(a.get_text())
        if len(title) < 2: continue
        items.append({"title": title, "url": urljoin(BASE + "/", a["href"])})
    return items

def parse_content(soup):
    """回傳 (question, answer, date_iso)。"""
    h3 = soup.select_one("h3")
    essay = soup.select_one(".area-essay")
    question = clean(h3.get_text()) if h3 else ""
    answer = FOOTER_RE.sub("", clean(essay.get_text(" "))) if essay else ""
    m = re.search(r"資料更新[：:]\s*(\d{2,3}-\d{1,2}-\d{1,2})", soup.get_text())
    return question, answer, roc_to_iso(m.group(1)) if m else ""

# ── 主爬蟲 ────────────────────────────────────────
def crawl_lists(session):
    """逐分類翻頁，回傳 {文章 id: {"title", "url", "categories"}}。"""
    found = {}
    for cat, csn in CATEGORIES.items():
        logger.info(f"▶ 分類：{cat}")
        before = len(found)
        cat_ids = set()
        for page in range(1, MAX_PAGES + 1):
            soup = fetch(session, f"{LIST_URL}&_CSN={csn}&page={page}&PageSize={PAGE_SIZE}")
            delay()
            if not soup: break
            items = parse_list(soup)
            new = [it for it in items if item_id(it["url"]) not in cat_ids]
            logger.info(f"    第 {page} 頁：{len(items)} 筆（新 {len(new)}）")
            if not new: break   # 超過最後一頁時網站會回傳空列表或重複頁
            for it in new:
                sid = item_id(it["url"])
                cat_ids.add(sid)
                found.setdefault(sid, {**it, "categories": []})["categories"].append(cat)
            if len(items) < PAGE_SIZE: break
        logger.info(f"    {cat}：{len(cat_ids)} 題（新增不重複 {len(found) - before}）")
    return found

def crawl():
    session = requests.Session()
    found = crawl_lists(session)
    logger.info(f"列表完成：不重複 {len(found)} 題，開始抓內文")

    results = []
    for n, (sid, it) in enumerate(found.items(), 1):
        soup = fetch(session, it["url"])
        delay()
        if not soup:
            logger.warning(f"  [{n}] 內文抓取失敗：{it['url']}")
            continue
        question, answer, date_iso = parse_content(soup)
        if excluded(question or it["title"]):
            logger.info(f"  [{n}] 純轉介，略過：{(question or it['title'])[:30]}")
            continue
        if len(answer) < 10:
            logger.warning(f"  [{n}] 內文過短，略過：{it['title'][:30]}")
            continue
        results.append({
            "source":      "台北市勞動局",
            "source_type": "gov",
            "category":    f"常見問答-{it['categories'][0]}",
            "categories":  it["categories"],
            "title":       question or it["title"],
            "url":         it["url"],
            "date":        date_iso,
            "content":     answer,
        })
        if n % 20 == 0: logger.info(f"  內文進度 {n}/{len(found)}")
    return results

def _bigrams(s):
    s = re.sub(r"[\s\W_]+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}

def merge_8401(data, threshold=0.95):
    """「84條之1」分類中有多題只換職業、答案相同（如 9 題皆為每月上限 288 小時），
    合併為一筆：職業清單＋共同工時限制。答案不同（240／260 小時、空勤組員等）者保留。"""
    suffix = re.compile(r"[。，,]?\s*以上(8|八)?適用勞動基準法第\s*84\s*條之\s*1工作者，?其工時限制為何[?？]?\s*$")
    groups, others = [], []
    for x in data:
        if "84條之1" not in x["category"] or not suffix.search(x["title"]):
            others.append(x); continue
        bg = _bigrams(x["content"])
        for g in groups:
            inter = len(bg & g["bg"]) / max(1, len(bg | g["bg"]))
            if inter >= threshold:
                g["items"].append(x); break
        else:
            groups.append({"bg": bg, "items": [x]})
    merged = []
    for g in groups:
        items = g["items"]
        if len(items) == 1:
            merged.append(items[0]); continue
        jobs = [suffix.sub("", x["title"]).strip() for x in items]
        base = max(items, key=lambda x: len(x["content"]))
        merged.append({
            **base,
            "title":   "下列工作者適用勞動基準法第84條之1，其工時限制為何？" + "；".join(jobs),
            "url":     items[0]["url"],
            "urls":    [x["url"] for x in items],
            "date":    max(x["date"] for x in items),
            "content": "適用對象：" + "；".join(jobs) + "\n工時限制：" + base["content"],
            "merged_from": len(items),
        })
    return others + merged

def main():
    data = merge_8401(crawl())
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ 共 {len(data)} 筆，已存至 {OUT_FILE}")

if __name__ == "__main__":
    main()
