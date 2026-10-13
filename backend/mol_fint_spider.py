"""
勞動部函釋爬蟲  laws.mol.gov.tw  v18

【v18 修正】
從 log 確認正確的搜尋框 id：
  關鍵字 input：id='cph_content_txtKeyword'
  送出按鈕：    id='cph_content_btnSend'
  （不是全站搜尋框 txtGlobalSearchOut）

安裝：pip install playwright beautifulsoup4
      python -m playwright install chromium
執行：python mol_fint_spider.py
輸出：mol_fint_cases.json
"""

import re, json, time, random, logging, urllib3
from urllib.parse import urljoin
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LAWS_BASE  = "https://laws.mol.gov.tw"
FINT_INDEX = f"{LAWS_BASE}/FINT/index-1.aspx"  # 從 log 確認的函釋搜尋頁

# 從 log 確認的正確 selector
KW_INPUT_SEL  = "#cph_content_txtKeyword"
SUBMIT_BTN_SEL = "#cph_content_btnSend"

DELAY_MIN, DELAY_MAX = 2.0, 4.0
MAX_PAGES  = 400
MAX_TOTAL  = 3000
KW_DELAY   = 5.0

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

LAW_TITLE_RE = re.compile(r"勞動基準法第\s*\d+|勞基法第\s*\d+|^第\s*\d+\s*條$")

FINT_KEYWORDS = [
    "勞動基準法", "工資", "工時", "休假", "資遣", "退休金", "職業災害",
]

ROC_RE = [
    re.compile(r"(\d{2,3})[/-](\d{1,2})[/-](\d{1,2})"),
    re.compile(r"民國\s*(\d{2,3})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?"),
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

def pw_wait(page, timeout=10000):
    try: page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception: pass
    time.sleep(random.uniform(1.0, 2.0))


def do_search(page, keyword: str) -> bool:
    """
    前往 FINT/index-1.aspx，用正確的搜尋框送出搜尋。
    """
    try:
        page.goto(FINT_INDEX, wait_until="domcontentloaded", timeout=25000)
        pw_wait(page, 8000)
        logger.info(f"  搜尋頁：{page.url}")
    except Exception as e:
        logger.warning(f"  無法訪問搜尋頁：{e}")
        return False

    # 填入關鍵字（直接用確認的 id）
    try:
        page.fill(KW_INPUT_SEL, "")
        page.fill(KW_INPUT_SEL, keyword)
        logger.info(f"  填入：{keyword!r} → {KW_INPUT_SEL}")
        time.sleep(0.5)
    except Exception as e:
        logger.warning(f"  填入失敗：{e}")
        return False

    # 點送出按鈕
    try:
        page.click(SUBMIT_BTN_SEL)
        pw_wait(page, 12000)
        logger.info(f"  送出完成 → 結果頁：{page.url[:80]}")
        return True
    except Exception as e:
        logger.warning(f"  送出失敗（{SUBMIT_BTN_SEL}）：{e}")
        # fallback：Enter 鍵
        try:
            page.press(KW_INPUT_SEL, "Enter")
            pw_wait(page, 12000)
            logger.info(f"  Enter 送出 → 結果頁：{page.url[:80]}")
            return True
        except Exception as e2:
            logger.warning(f"  Enter 送出也失敗：{e2}")
            return False


def parse_results(soup: BeautifulSoup) -> list[dict]:
    """解析 FINT/results.aspx 頁面，取發文字號/日期/要旨"""
    items = []
    text  = soup.get_text(separator="\n")

    if "發文字號" not in text:
        return []

    # 以「數字. 換行」或「換行 發文字號」分割每筆
    blocks = re.split(r"\n\s*\d+\.\s*\n|(?=\n發文字號)", text)

    for block in blocks:
        if "發文字號" not in block:
            continue

        m = re.search(r"發文字號[：:]\s*(.+?)(?=\n|發文日期|$)", block)
        doc_no = clean(m.group(1)) if m else ""

        m = re.search(r"發文日期[：:]\s*(.+?)(?=\n|要\s*旨|相關法條|$)", block)
        date_raw = clean(m.group(1)) if m else ""

        m = re.search(
            r"要\s*旨[：:]\s*(.+?)(?=\n\s*\d+\.\s*\n|\n發文字號|$)",
            block, re.S
        )
        gist = clean(m.group(1)) if m else ""

        url = ""
        if doc_no:
            for a in soup.find_all("a", href=True):
                a_txt = clean(a.get_text())
                if doc_no[:8] in a_txt or (a_txt and a_txt in doc_no):
                    url = urljoin(LAWS_BASE, a["href"])
                    break

        if gist and len(gist) > 15:
            items.append({
                "doc_no":   doc_no,
                "date_iso": roc_to_iso(date_raw),
                "gist":     gist,
                "url":      url,
            })

    return items


def get_total_pages(soup: BeautifulSoup) -> int:
    m = re.search(r"第\s*\d+\s*/\s*(\d+)\s*頁", soup.get_text())
    return int(m.group(1)) if m else 1


def click_next(page) -> bool:
    for sel in ["a:has-text('下一頁')","a:has-text('下頁')","a:has-text('»')"]:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click()
                pw_wait(page, 8000)
                return True
        except Exception: continue
    return False


def crawl_fint(page, fps):
    results, seen = [], set()
    logger.info("▶ 勞動部 函釋")

    for kw_str in FINT_KEYWORDS:
        logger.info(f"\n  關鍵字：{kw_str}")

        if not do_search(page, kw_str):
            logger.warning(f"  搜尋失敗，跳過")
            continue

        soup = BeautifulSoup(page.content(), "html.parser")
        total_pages = get_total_pages(soup)
        logger.info(f"  共 {total_pages} 頁")

        page_num, kw_count = 1, 0

        while page_num <= min(total_pages, MAX_PAGES):
            if page_num > 1:
                soup = BeautifulSoup(page.content(), "html.parser")

            items = parse_results(soup)
            logger.info(f"    第 {page_num}/{total_pages} 頁：{len(items)} 筆")

            if not items:
                preview = clean(soup.get_text())[:200]
                logger.warning(f"    無結果，預覽：{preview}")
                break

            for item in items:
                key = item["doc_no"] or f"{item['date_iso']}-{item['gist'][:20]}"
                if key in seen: continue
                seen.add(key)

                gist = item["gist"]
                if not gist or len(gist) < 15: continue
                if is_pure_law(gist[:50]) or is_dup(gist, fps): continue

                results.append({
                    "source":      "勞動部",
                    "source_type": "gov",
                    "category":    "函釋",
                    "title":       item["doc_no"] or f"函釋-{item['date_iso']}",
                    "url":         item["url"],
                    "date":        item["date_iso"],
                    "content":     gist,
                    "related_law": "",
                    "keywords":    extract_kw(gist),
                })
                kw_count += 1

            if len(results) >= MAX_TOTAL: break
            if page_num >= total_pages: break

            if not click_next(page):
                logger.warning(f"  無法翻到第 {page_num+1} 頁")
                break
            page_num += 1
            delay()

        logger.info(f"  「{kw_str}」新增 {kw_count} 筆（累計 {len(results)} 筆）")
        if len(results) >= MAX_TOTAL: break

        logger.info(f"  等待 {KW_DELAY}s…")
        time.sleep(KW_DELAY)

    logger.info(f"\n函釋 完成，共 {len(results)} 筆")
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
        ctx = browser.new_context(
            user_agent=UA,
            viewport={"width":1280,"height":800},
            locale="zh-TW",
        )
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        pw_page = ctx.new_page()
        results = crawl_fint(pw_page, fps)
        browser.close()

    output = "mol_fint_cases.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"\n✅ 完成！共 {len(results)} 筆 → {output}")
    if results:
        print("\n輸出範例（前2筆）：")
        for r in results[:2]:
            print(json.dumps(r, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()