"""
勞動部常見問答爬蟲  mol.gov.tw  v12

【v12 修正】parse_post：
  - #aC 是空的（只是錨點）
  - 真正內容在 .wrapper 裡
  - 頁面用「問題」「答案」文字標記問答
  - 改用正規表達式從整頁文字萃取問答段落
"""

import re, json, time, random, logging, urllib3
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MOL_BASE = "https://www.mol.gov.tw"
DELAY_MIN, DELAY_MAX = 1.5, 3.0
MAX_PAGES, MAX_TOTAL = 20, 600

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

LAW_TITLE_RE = re.compile(r"勞動基準法第\s*\d+|勞基法第\s*\d+|^第\s*\d+\s*條$")
SKIP_PATH_KW = ["lpsimplelist","nodelist","normalnodelist","nodeListSearch",
                "CountAndRedirect","umbraco","sitemap","cloud/",
                "English","RSS",".pdf",".doc",".xls"]

QA_TARGETS = [
    ("資遣費",       f"{MOL_BASE}/1607/28690/2282/2302/2310/lpsimplelist"),
    ("勞動契約",     f"{MOL_BASE}/1607/28690/2282/2326/2288/lpsimplelist"),
    ("預告期間",     f"{MOL_BASE}/1607/28690/2282/2302/2314/lpsimplelist"),
    ("職業災害補償", f"{MOL_BASE}/1607/28690/2282/2302/2316/lpsimplelist"),
    ("工時休假請假", f"{MOL_BASE}/1607/28162/28166/28218/lpsimplelist"),
    ("工時加班",     f"{MOL_BASE}/1607/28162/28166/28218/28226/30382/lpsimplelist"),
    ("天然災害出勤", f"{MOL_BASE}/1607/28162/28166/28218/28230/lpsimplelist"),
    ("病假權益",     f"{MOL_BASE}/1607/28162/28166/28218/86988/lpsimplelist"),
    ("基本工資",     f"{MOL_BASE}/1607/28162/28166/28180/28182/28190/lpsimplelist"),
]

ROC_RE = [
    re.compile(r"(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})"),
    re.compile(r"民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"),
]

def roc_to_iso(raw):
    raw = raw.strip()
    if not raw: return ""
    if re.match(r"\d{4}[-/]\d{2}[-/]\d{2}", raw):
        return raw.replace("/", "-")
    for pat in ROC_RE:
        m = pat.search(raw)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 200: y += 1911
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return raw

def delay(): time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
def clean(t): return re.sub(r"\s+", " ", t).strip()
def is_pure_law(t): return bool(LAW_TITLE_RE.search(t))

def extract_kw(text):
    terms = ["加班","資遣","解僱","薪資","工資","休假","特休","勞保","勞退",
             "職災","懷孕","育嬰","產假","陪產","工時","契約","試用期","競業",
             "退休","罷工","工會","最低工資","例假","國定假日","排班","夜班",
             "調職","霸凌","性騷擾","資遣費"]
    return [t for t in terms if t in text]

def load_fps(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            laws = json.load(f)
        fps = {item["text"][:20].strip() for item in laws if "text" in item}
        logger.info(f"載入 {len(fps)} 筆法條指紋")
        return fps
    except Exception as e:
        logger.warning(f"無法載入法條 JSON：{e}")
        return set()

def is_dup(content, fps):
    if not fps: return False
    return sum(1 for fp in fps if fp in content) >= 3

def pw_wait(page, timeout=8000):
    try: page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception: pass
    time.sleep(random.uniform(0.8, 1.5))

def pw_get_soup(page, url, timeout=25000):
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        pw_wait(page)
        return BeautifulSoup(page.content(), "html.parser")
    except Exception as e:
        logger.warning(f"  取頁失敗：{url} → {e}")
        return None

def _lps_prefix(lps_url):
    return [s for s in urlparse(lps_url).path.split("/") if s.isdigit()]

def is_article_href(href, lps_url, prefix=None):
    if not href or href.startswith(("javascript","mailto","#")): return False
    if any(kw in href for kw in SKIP_PATH_KW): return False
    full = urljoin(lps_url, href)
    if "mol.gov.tw" not in full: return False
    path = urlparse(full).path
    segs = [s for s in path.split("/") if s]
    ns   = [s for s in segs if s.isdigit()]
    if len(ns) < 4: return False
    last = segs[-1] if segs else ""
    if not (last.isdigit() or last == "post"): return False
    if prefix is None: prefix = _lps_prefix(lps_url)
    ptr = 0
    for ln in prefix:
        while ptr < len(segs) and segs[ptr] != ln: ptr += 1
        if ptr >= len(segs): return False
        ptr += 1
    if path.rstrip("/") == urlparse(lps_url).path.rstrip("/"): return False
    return True

def get_article_links(soup, lps_url):
    prefix = _lps_prefix(lps_url)
    items, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not is_article_href(href, lps_url, prefix): continue
        full = urljoin(lps_url, href)
        if full in seen: continue
        seen.add(full)
        title = clean(a.get_text())
        if len(title) < 4:
            parent = a.find_parent(["li","div","td","article","h2","h3"])
            if parent: title = clean(parent.get_text())[:80]
        if title: items.append((title, full))
    return items

def find_next_page(soup, lps_url, cur):
    for a in soup.find_all("a", href=True):
        t = clean(a.get_text())
        h = a["href"]
        if t in ("下一頁","Next","›",">","»"):
            return urljoin(lps_url, h)
        if t == str(cur+1) and any(k in h.lower() for k in ["page","p=","&p"]):
            return urljoin(lps_url, h)
    if soup.find("a", rel="next"):
        return urljoin(lps_url, soup.find("a", rel="next")["href"])
    return None


def parse_post(soup: BeautifulSoup) -> tuple[str, str]:
    """
    【v12】mol.gov.tw 文章頁結構：
      - #aC 是空錨點（只有 :::）
      - 內容在 .wrapper 裡
      - 問答用「問題」「答案」兩個 span/label 標記
    
    策略：
    1. 找頁面上帶「問題」「答案」文字的元素，取其後的文字
    2. fallback：從整頁文字用正規表達式取問答段落
    """
    content = ""

    # 策略1：找 span/label/p 含「問題」或「答案」的標記元素
    # mol.gov.tw 結構通常是：
    # <span class="...">問題</span> 文字...
    # <span class="...">答案</span> 文字...
    q_text, a_text = "", ""

    for el in soup.find_all(["span","label","strong","b","dt","th","div"]):
        t = clean(el.get_text())
        if t == "問題":
            # 取同層下一個兄弟或父元素的文字
            parent = el.parent
            if parent:
                full = clean(parent.get_text())
                # 去掉「問題」標籤本身
                q_text = full.replace("問題", "", 1).strip()
        elif t == "答案":
            parent = el.parent
            if parent:
                full = clean(parent.get_text())
                a_text = full.replace("答案", "", 1).strip()

    if q_text or a_text:
        content = f"問題 {q_text} 答案 {a_text}".strip()

    # 策略2：從整頁文字用正規表達式
    if not content or len(content) < 20:
        page_text = clean(soup.get_text())
        # 取「問題」到「答案」到結尾（或到下一個頁面元素）
        m = re.search(
            r"問題\s+(.+?)\s+答案\s+(.+?)(?=回上一頁|友善列印|更新日期|發布日期|$)",
            page_text, re.S
        )
        if m:
            q = clean(m.group(1))
            a = clean(m.group(2))
            content = f"問題 {q} 答案 {a}"

    # 策略3：找 .wrapper，移除導覽，取剩餘
    if not content or len(content) < 20:
        wrapper = soup.select_one(".wrapper")
        if wrapper:
            work = BeautifulSoup(str(wrapper), "html.parser")
            # 移除導覽和功能區塊
            for sel in ["header","footer","nav",".sidebar","aside",
                        "script","style","noscript",
                        ".customer_service_block",".m_customer",
                        ".m_search",".scrollToTop",
                        ".fn-bar",".function-bar",".social",
                        ".breadcrumb",".pagerbar",
                        # 移除功能按鈕列（回上一頁、友善列印、轉寄友人）
                        "button",".btn"]:
                for el in work.select(sel):
                    el.decompose()
            # 移除短連結（導覽連結）
            for a in work.find_all("a"):
                if len(clean(a.get_text())) <= 15:
                    a.decompose()
            content = clean(work.get_text())

    # 日期
    date_iso = ""
    meta = soup.find("meta", attrs={"name": "DC.Date"})
    if meta and meta.get("content"):
        date_iso = roc_to_iso(meta["content"][:10])
    if not date_iso:
        for pat in [
            r"最後異動日期[：:]\s*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})",
            r"更新日期[：:]\s*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})",
            r"發布日期[：:]\s*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})",
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{3}[/-]\d{2}[/-]\d{2})",
        ]:
            m = re.search(pat, soup.get_text())
            if m:
                date_iso = roc_to_iso(m.group(1))
                if date_iso: break

    return content, date_iso


def crawl_qa(pw_page, fps, targets):
    results, seen_urls = [], set()
    logger.info(f"▶ 勞動部 常見問答（{len(targets)} 個主題）")

    for name, lps_url in targets:
        logger.info(f"\n  主題：{name}")
        page, cur_url = 1, lps_url

        while page <= MAX_PAGES:
            soup = pw_get_soup(pw_page, cur_url)
            if not soup:
                logger.warning(f"    第 {page} 頁取得失敗")
                break

            articles = get_article_links(soup, lps_url)
            logger.info(f"    第 {page} 頁：{len(articles)} 篇")

            if not articles:
                prefix = _lps_prefix(lps_url)
                logger.warning(f"    lps 前綴：{prefix}，無文章連結")
                for a in soup.find_all("a", href=True):
                    h = a["href"]
                    if h.startswith(("#","javascript","mailto")): continue
                    full = urljoin(lps_url, h)
                    if "mol.gov.tw" not in full: continue
                    segs = [s for s in urlparse(full).path.split("/") if s]
                    ns = [s for s in segs if s.isdigit()]
                    last = segs[-1] if segs else ""
                    ok = is_article_href(h, lps_url, prefix)
                    if ok or (len(ns) >= 3 and (last.isdigit() or last == "post")):
                        logger.warning(f"      {'✓' if ok else '✗'} last={last!r} ns={len(ns)} {h[:80]}")
                break

            for title, post_url in articles:
                if post_url in seen_urls: continue
                seen_urls.add(post_url)
                if is_pure_law(title): continue

                post_soup = pw_get_soup(pw_page, post_url)
                if not post_soup: delay(); continue

                content, date_iso = parse_post(post_soup)

                if not content or len(content) < 20:
                    logger.warning(f"    [正文未命中] {title[:30]}")
                    delay(); continue
                if is_dup(content, fps):
                    logger.info(f"    [跳過-重複] {title[:30]}")
                    delay(); continue

                results.append({
                    "source":      "勞動部",
                    "source_type": "gov",
                    "category":    f"常見問答-{name}",
                    "title":       title,
                    "url":         post_url,
                    "date":        date_iso,
                    "content":     content,
                    "keywords":    extract_kw(title + " " + content[:300]),
                })
                logger.info(f"    ✓ [{date_iso}] {title[:40]}")
                delay()
                if len(results) >= MAX_TOTAL: return results

            next_url = find_next_page(soup, lps_url, page)
            if not next_url: break
            cur_url = next_url
            page += 1
            delay()

    logger.info(f"\n常見問答 完成，共 {len(results)} 筆")
    return results


def main():
    from playwright.sync_api import sync_playwright
    fps = load_fps("labor_law_cleaned.json")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox","--disable-dev-shm-usage",
                  "--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(user_agent=UA,
                                   viewport={"width":1280,"height":800},
                                   locale="zh-TW")
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
        page = ctx.new_page()

        try:
            page.goto(MOL_BASE, wait_until="domcontentloaded", timeout=20000)
            pw_wait(page, 6000)
            logger.info("首頁暖機完成")
        except Exception as e:
            logger.warning(f"首頁暖機失敗：{e}")

        results = crawl_qa(page, fps, QA_TARGETS)
        browser.close()

    output = "mol_qa_cases.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ 完成！共 {len(results)} 筆 → {output}")
    if results:
        print("\n輸出範例（第一筆）：")
        print(json.dumps(results[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()