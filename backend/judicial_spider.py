"""
司法院裁判書爬蟲  judgment.judicial.gov.tw  v14

【v14 修正】
判決列表在 iframe 裡，page.content() 取不到。
改為用 page.frames 找到正確的 frame，從 frame.content() 解析。

執行：python judicial_spider.py（會開啟瀏覽器視窗）
輸出：judicial_cases.json
"""

import re, json, time, random, logging
from urllib.parse import urljoin
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE_URL   = "https://judgment.judicial.gov.tw"
FJUD_BASE  = f"{BASE_URL}/FJUD/"   # 判決連結的 base（相對路徑從 FJUD/ 開始）
SEARCH_URL = f"{BASE_URL}/FJUD/Default_AD.aspx"

DELAY_MIN, DELAY_MAX = 2.0, 4.0
MAX_PAGES  = 3
MAX_TOTAL  = 500

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

SEARCH_QUERIES = [
    "給付工資",
    "給付資遣費",
    "確認僱傭關係",
    "給付退休金",
    "職業災害",
    "給付加班費",
    "勞動契約",
]

def delay(): time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
def clean(t): return re.sub(r"\s+", " ", t).strip()

def extract_kw(text):
    terms = ["加班","資遣","解僱","薪資","工資","休假","特休","勞保","勞退",
             "職災","懷孕","育嬰","產假","陪產","工時","契約","試用期","競業",
             "退休","罷工","工會","最低工資","例假","國定假日","排班","資遣費"]
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

def pw_wait(page, timeout=10000):
    try: page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception: pass
    time.sleep(random.uniform(1.0, 2.0))


def get_result_frame(page):
    """
    找包含判決列表的 frame。
    診斷：印出所有 frame 的名稱和 URL。
    """
    frames = page.frames
    logger.info(f"  共 {len(frames)} 個 frame：")
    result_frame = None
    for f in frames:
        logger.info(f"    name={f.name!r} url={f.url[:70]}")
        # 判決列表的 frame 通常 URL 含 qryresultlst 或 FJUD
        if "qryresultlst" in f.url or (
            "FJUD" in f.url and f.url != page.url
        ):
            result_frame = f
    return result_frame


def wait_for_result_frame(page, timeout_sec=20):
    """等待含判決列表的 frame 出現"""
    for _ in range(timeout_sec * 2):
        time.sleep(0.5)
        try: page.wait_for_load_state("networkidle", timeout=2000)
        except Exception: pass
        for f in page.frames:
            if "qryresultlst" in f.url:
                return f
    return None


def do_search(page, cause: str) -> bool:
    try:
        page.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=25000)
        pw_wait(page, 8000)
    except Exception as e:
        logger.warning(f"  搜尋頁失敗：{e}")
        return False

    ok = page.evaluate(f"""() => {{
        document.querySelectorAll('input[type=checkbox]').forEach(cb => {{
            const label = document.querySelector('label[for="'+cb.id+'"]');
            const txt = (label?.innerText || '').trim();
            if (txt === '民事') cb.checked = true;
            else if (['憲法','刑事','行政','懲戒'].includes(txt)) cb.checked = false;
        }});
        const dy1 = document.getElementById('dy1');
        const dy2 = document.getElementById('dy2');
        if (dy1) dy1.value = '108';
        if (dy2) dy2.value = '115';
        ['jud_jmain','jud_kw'].forEach(id => {{
            const el = document.getElementById(id);
            if (el) el.value = '';
        }});
        const jtitle = document.getElementById('jud_title');
        if (!jtitle) return false;
        jtitle.value = '{cause}';
        return true;
    }}""")
    if not ok: return False

    page.click("#btnQry")
    time.sleep(5)
    pw_wait(page, 8000)

    text = clean(BeautifulSoup(page.content(), "html.parser").get_text())
    if "查詢結果" not in text: return False
    m = re.search(r"查詢結果\s*([\d,]+)", text)
    logger.info(f"  查詢結果 {m.group(1) if m else '?'} 筆")

    # 點「查詢結果XXXXX」
    page.evaluate("""() => {
        for (const a of document.querySelectorAll('a[href*="qryresultlst"]')) {
            if (/查詢結果/.test(a.innerText)) { a.click(); return; }
        }
    }""")
    time.sleep(4)
    pw_wait(page, 5000)

    # 點法院
    clicked = page.evaluate(r"""() => {
        const preferred = ['臺灣臺北地方法院','臺灣新北地方法院','臺灣高等法院'];
        for (const court of preferred) {
            for (const a of document.querySelectorAll('a[href*="qryresultlst"]')) {
                if (a.innerText.includes(court)) {
                    a.click();
                    return a.innerText.trim().replace(/\s+/g,' ').substring(0,25);
                }
            }
        }
        return null;
    }""")
    if clicked:
        logger.info(f"  點擊：{clicked}")
        time.sleep(5)
        pw_wait(page, 8000)

    # 診斷：印出所有 frame
    get_result_frame(page)
    return True


def parse_judgment_list(soup: BeautifulSoup, cause: str) -> list[dict]:
    items = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "qryresultlst" in href: continue
        if "opendata" in href: continue
        if href.startswith("#"): continue

        title = clean(a.get_text())
        if not title or len(title) < 10: continue
        if not (("判決" in title or "裁定" in title) and
                "號" in title and "法院" in title):
            continue

        tr = a.find_parent("tr")
        date_raw, cause_text = "", ""
        if tr:
            cells = [clean(td.get_text()) for td in tr.find_all("td")]
            date_raw = next(
                (c for c in cells if re.match(r"\d{3}\.\d{2}\.\d{2}", c)), ""
            )
            cause_text = cells[-1] if cells else ""

        items.append({
            "title": title, "url": urljoin(FJUD_BASE, href),
            "date_raw": date_raw, "cause": cause_text or cause,
        })

    if not items:
        all_a = [(clean(a.get_text())[:40], a["href"][:70])
                 for a in soup.find_all("a", href=True)
                 if len(clean(a.get_text())) > 5]
        logger.warning(f"    0筆，所有連結（{len(all_a)}個）：")
        for t, h in all_a[:10]:
            logger.warning(f"      {t!r} → {h}")
    return items


def get_total_pages(soup: BeautifulSoup) -> int:
    text = soup.get_text()
    m = re.search(r"\d+\s*/\s*(\d+)\s*頁", text)
    if m: return int(m.group(1))
    m = re.search(r"共\s*(\d+)\s*頁", text)
    if m: return int(m.group(1))
    return 1


def click_next_in_frame(result_frame) -> bool:
    try:
        clicked = result_frame.evaluate("""() => {
            for (const a of document.querySelectorAll('a')) {
                if (a.innerText.trim() === '下一頁') { a.click(); return true; }
            }
            return false;
        }""")
        if clicked:
            time.sleep(3)
            return True
    except Exception: pass
    return False


def parse_judgment(soup: BeautifulSoup) -> dict:
    text = soup.get_text(separator="\n")
    m = re.search(r"主\s*文\s*\n(.+?)(?=事\s*實|理\s*由|附\s*表|$)", text, re.S)
    main_text = clean(m.group(1))[:500] if m else ""
    m = re.search(
        r"(?:事實及)?理\s*由\s*\n(.+?)(?=以上正本|本件裁判費|此\s*致|中\s*華\s*民\s*國\s*\d|$)",
        text, re.S
    )
    reason = clean(m.group(1))[:2000] if m else ""
    m = re.search(r"中\s*華\s*民\s*國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日", text)
    date_iso = ""
    if m:
        y = int(m.group(1)) + 1911
        date_iso = f"{y:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    if main_text and reason:
        content = f"主文：{main_text} 理由：{reason}"
    elif reason:
        content = reason
    elif main_text:
        content = main_text
    else:
        content = ""
    return {"date_iso": date_iso, "content": content}


def crawl_judicial(page, fps):
    results, seen_urls = [], set()
    logger.info(f"▶ 司法院 裁判書（每案由前 {MAX_PAGES} 頁）")

    for cause in SEARCH_QUERIES:
        logger.info(f"\n  案由：{cause}")
        if not do_search(page, cause):
            logger.warning("  搜尋失敗，跳過")
            continue

        # 嘗試從 frame 取結果
        result_frame = wait_for_result_frame(page, 10)
        if result_frame:
            logger.info(f"  找到結果 frame：{result_frame.url[:70]}")
            soup = BeautifulSoup(result_frame.content(), "html.parser")
        else:
            logger.warning("  無 qryresultlst frame，用主頁面")
            soup = BeautifulSoup(page.content(), "html.parser")

        total_pages = get_total_pages(soup)
        logger.info(f"  共 {total_pages} 頁（只取前 {MAX_PAGES} 頁）")

        page_num, cause_count = 1, 0

        while page_num <= min(total_pages, MAX_PAGES):
            items = parse_judgment_list(soup, cause)
            logger.info(f"    第 {page_num}/{min(total_pages,MAX_PAGES)} 頁："
                        f"{len(items)} 筆")

            if not items:
                break

            for item in items:
                if item["url"] in seen_urls: continue
                seen_urls.add(item["url"])
                try:
                    page.goto(item["url"],
                              wait_until="domcontentloaded", timeout=25000)
                    pw_wait(page, 6000)
                    jsoup = BeautifulSoup(page.content(), "html.parser")
                except Exception as e:
                    logger.warning(f"    取判決失敗：{e}")
                    delay(); continue

                jdata = parse_judgment(jsoup)
                if not jdata["content"] or len(jdata["content"]) < 50:
                    delay(); continue

                date_str = jdata["date_iso"]
                if not date_str and item["date_raw"]:
                    m2 = re.match(r"(\d{3})\.(\d{2})\.(\d{2})", item["date_raw"])
                    if m2:
                        date_str = (f"{int(m2.group(1))+1911}"
                                    f"-{m2.group(2)}-{m2.group(3)}")

                results.append({
                    "source":      "司法院",
                    "source_type": "judicial",
                    "category":    f"裁判書-{cause}",
                    "title":       item["title"],
                    "url":         item["url"],
                    "date":        date_str,
                    "content":     jdata["content"],
                    "keywords":    extract_kw(jdata["content"][:500]),
                })
                cause_count += 1
                logger.info(f"    ✓ [{date_str}] {item['title'][:45]}")
                delay()
                if len(results) >= MAX_TOTAL: break

            if len(results) >= MAX_TOTAL: break
            if page_num >= min(total_pages, MAX_PAGES): break

            # 翻頁：在 frame 或主頁面點下一頁
            if result_frame:
                if not click_next_in_frame(result_frame):
                    break
                soup = BeautifulSoup(result_frame.content(), "html.parser")
            else:
                page.go_back()
                pw_wait(page, 5000)
                clicked = page.evaluate("""() => {
                    for (const a of document.querySelectorAll('a'))
                        if (a.innerText.trim()==='下一頁') { a.click(); return true; }
                    return false;
                }""")
                if not clicked: break
                time.sleep(3)
                pw_wait(page, 5000)
                soup = BeautifulSoup(page.content(), "html.parser")

            page_num += 1
            delay()

        logger.info(f"  「{cause}」新增 {cause_count} 筆（累計 {len(results)} 筆）")
        if len(results) >= MAX_TOTAL: break
        time.sleep(3)

    logger.info(f"\n裁判書 完成，共 {len(results)} 筆")
    return results


def main():
    from playwright.sync_api import sync_playwright
    fps = load_fps("labor_law_cleaned.json")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--no-sandbox",
                  "--disable-blink-features=AutomationControlled"],
        )
        ctx = browser.new_context(
            user_agent=UA,
            viewport={"width":1280,"height":900},
            locale="zh-TW",
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        pw_page = ctx.new_page()
        results = crawl_judicial(pw_page, fps)
        browser.close()

    output = "judicial_cases.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ 完成！共 {len(results)} 筆 → {output}")
    if results:
        print("\n輸出範例（第一筆）：")
        print(json.dumps(results[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()