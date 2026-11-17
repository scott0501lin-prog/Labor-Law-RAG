"""
勞動部 行政函釋爬蟲 v2（依「相關法條」逐條抓取，全文 + 固定網址 + 排除廢止）
來源：勞動部勞動法令查詢系統 https://laws.mol.gov.tw

與舊版 scrapers/mol_fint_spider.py 的差異：
  - 舊版：用關鍵字（勞動基準法、工資、工時、職業災害…）全文搜尋 → 混入勞保、職安、就服、
    工會等法規；只存「要旨」一句（313 筆只有問題沒有答案）；網址是「搜尋結果第 N 筆」，
    會隨新函釋上架而指到別篇；混入網頁頁尾；未排除已廢止／停止適用的函釋
  - 新版：
    1. 從法規每一條的「行政函釋」清單抓（＝網站上以該條為相關法條的函釋），只限下列法規
    2. 存要旨＋全文內容（主旨、說明），並把相關法條結構化（可對應法規資料與判決 cited_laws）
    3. 網址改用函釋固定網址（發文字號＋日期）
    4. 排除：已廢止／停止適用（網站「廢」標記或編註）、公告、只是宣告其他函釋停止適用的函
    5. 同一函釋出現在多條條文下只抓一次，from_articles 記錄出現在哪些條文

執行：python backend/scraper2/spiders/mol_fint_spider.py      （可中斷續跑）
輸出：backend/scraper2/data/mol_fint_cases.json、mol_fint_skipped.json（被排除者與原因）
"""

import io, os, re, sys, json, time, random, logging
from collections import Counter
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)

HERE      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # backend/scraper2/data
OUT_FILE  = os.path.join(HERE, "mol_fint_cases.json")
SKIP_FILE = os.path.join(HERE, "mol_fint_skipped.json")
INDEX_FILE = os.path.join(HERE, "mol_fint_index.json")     # 階段 1 結果（續跑用）

BASE    = "https://laws.mol.gov.tw/FLAW/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}
DELAY_MIN, DELAY_MAX = 1.0, 2.0

# 網站內部法規 id
LAWS = {
    "FL014930": "勞動基準法",
    "FL014931": "勞動基準法施行細則",
    "FL030634": "勞工退休金條例",
    "FL096423": "勞工職業災害保險及保護法",
}

STOP_RE = re.compile(r"停止適用|廢止|不再援用|不予援用|失其效力")


# ── 工具 ──────────────────────────────────────────
def delay(): time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def clean(t):
    return re.sub(r"\s+", " ", (t or "").replace("　", " ")).strip()

def get(session, url, retries=3):
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=40)
            r.raise_for_status()
            return BeautifulSoup(r.content.decode("utf-8", "replace"), "html.parser")
        except requests.RequestException as e:
            logger.warning(f"  請求失敗（{i + 1}/{retries}）：{e}")
            time.sleep(5 * (i + 1))
    return None

def roc_to_iso(s):
    m = re.search(r"民國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日", s or "")
    return f"{int(m.group(1)) + 1911:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else ""

def norm_doc_no(s):
    return re.sub(r"\s+", "", s or "")


def join_pre(text):
    """函釋內文為固定寬度斷行（含縮排）→ 接回段落；遇編號、標題或句末才換段。"""
    lines = [l.strip() for l in (text or "").replace("　", " ").split("\n")]
    lines = [l for l in lines if l]
    start = re.compile(r"^([一二三四五六七八九十]+、|[（(][一二三四五六七八九十\d]+[）)]|\d+[.、．]"
                       r"|主\s*旨|說\s*明|全文內容|正\s*本|副\s*本|附件)")
    out = []
    for l in lines:
        if re.match(r"^(正\s*本|副\s*本)\s*[：:]", l):
            break                                    # 之後是受文者清單，對內容無用
        if out and (l[0] in "」』）)，、。；" or
                    (not start.match(l) and not out[-1].endswith(("。", "：", ":")))):
            out[-1] += l
        else:
            out.append(l)
    return "\n".join(re.sub(r"\s{2,}", " ", l) for l in out)


def split_laws(text):
    """「勞動基準法 第 24、39 條」→ ["勞動基準法 第 24 條", "勞動基準法 第 39 條"]。"""
    m = re.match(r"(.+?)\s*第\s*([\d\-、,， ]+)\s*條", text)
    if not m: return [clean(text)] if text.strip() else []
    law = clean(m.group(1))
    return [f"{law} 第 {n.strip()} 條" for n in re.split(r"[、,，]", m.group(2)) if n.strip()]


# ══════════════════════════════════════════════════
# 階段 1：逐條列出相關函釋
# ══════════════════════════════════════════════════
def article_nos(session, law_id):
    soup = get(session, f"{BASE}FLAWDAT0201.aspx?id={law_id}")
    delay()
    nos = []
    for a in soup.select('a[href*="FLAWDOC01.aspx"]') if soup else []:
        m = re.search(r"flno=([^&]+)", a["href"])
        if m and m.group(1) not in nos: nos.append(m.group(1))
    return nos


def article_list(session, law_id, law_name, flno):
    """回傳該條文下所有行政函釋：[(發文字號, 單篇網址)]。"""
    rtype = quote("E/0/行政函釋")
    items, page, total = [], 1, None
    while True:
        soup = get(session, f"{BASE}FLAWDOC02.aspx?rtype={rtype}&id={law_id}&flno={flno}&page={page}")
        delay()
        if not soup: break
        if total is None:
            m = re.search(r"共\s*(\d+)\s*筆", soup.get_text(" "))
            total = int(m.group(1)) if m else 0
            if total == 0: break
        new = 0
        for a in soup.select('a[href*="FLAWDOC03.aspx"]'):
            if "號" in a.get_text():
                items.append((norm_doc_no(a.get_text()), urljoin(BASE, a["href"])))
                new += 1
        if new == 0 or len(items) >= total: break
        page += 1
    return items


def stage_index(session):
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, encoding="utf-8") as f:
            idx = json.load(f)
        logger.info(f"沿用已建立的清單：{len(idx)} 則函釋")
        return idx
    idx = {}          # 發文字號 → {"url":..., "from_articles":[...]}
    for law_id, law_name in LAWS.items():
        nos = article_nos(session, law_id)
        logger.info(f"▶ {law_name}：{len(nos)} 條")
        law_total = 0
        for flno in nos:
            items = article_list(session, law_id, law_name, flno)
            law_total += len(items)
            for doc_no, url in items:
                rec = idx.setdefault(doc_no, {"url": url, "from_articles": []})
                tag = f"{law_name} 第 {flno} 條"
                if tag not in rec["from_articles"]: rec["from_articles"].append(tag)
        logger.info(f"    {law_name} 各條函釋合計 {law_total}，累計不重複 {len(idx)}")
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)
    return idx


# ══════════════════════════════════════════════════
# 階段 2：抓單篇全文
# ══════════════════════════════════════════════════
def parse_doc(soup):
    box = soup.select_one("#cph_content_pletype")
    if not box: return None
    f = {"status": "", "related": [], "body": [], "note": ""}
    fei = box.select_one(".fei")
    if fei: f["status"] = clean(fei.get_text())
    for tr in box.select("tr"):
        th = tr.select_one("td.th")
        if th:
            label = re.sub(r"[\s：:]", "", th.get_text())
            td = th.find_next_sibling("td")
            if not td: continue
            if label == "發文單位":   f["issuer"] = clean(td.get_text())
            elif label == "發文字號":
                if fei: fei.extract()
                f["doc_no"] = clean(td.get_text())
            elif label == "發文日期": f["date"] = roc_to_iso(td.get_text())
            elif label == "相關法條":
                for li in td.select("li"):
                    a = li.find("a")
                    for x in split_laws(a.get_text() if a else li.get_text()):
                        if x not in f["related"]: f["related"].append(x)
            elif label == "要旨":     f["summary"] = join_pre(td.get_text("\n"))
            elif label == "編註":     f["note"] = join_pre(td.get_text("\n"))
        else:
            pre = tr.select_one("td.fint-pre")
            if pre: f["body"].append(join_pre(pre.get_text("\n")))
    fd = soup.select_one("#cph_content_FDLink")
    f["fdlink"] = fd.get("value", "") if fd else ""
    return f


def permalink(fdlink):
    """FDLink = 'E,N00000,勞動福 3,1150153773,20260908' → 固定網址。"""
    p = [x.strip() for x in fdlink.split(",")]
    if len(p) < 5: return ""
    return (f"{BASE}FLAWDOC03.aspx?datatype=etype&ecode={p[1]}&ecase={quote(p[2])}"
            f"&eno={p[3]}&edate={p[4]}")


def skip_reason(f):
    if f["status"]:                                   return f"網站標記「{f['status']}」"
    if STOP_RE.search(f.get("note", "")):            return "編註：已廢止／停止適用"
    if f.get("doc_no", "").endswith("公告"):          return "公告（非函釋）"
    body = "\n".join(f["body"])
    if STOP_RE.search(f.get("summary", "")) and len(body) < 300:
        return "僅宣告其他函釋停止適用"
    if len(body) < 20 and len(f.get("summary", "")) < 20: return "內容過短"
    # 歷年基本工資調整公告：網站未標廢止，但金額已過時（現行依最低工資法每年公告）
    if f.get("date", "") < "2000" and re.search(r"基本工資調整為|調整後之基本工資", f.get("summary", "")):
        return "過時的基本工資調整公告"
    return ""


def main():
    session = requests.Session()
    idx = stage_index(session)

    done, skipped = {}, {}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as fh: done = {x["doc_no_key"]: x for x in json.load(fh)}
    if os.path.exists(SKIP_FILE):
        with open(SKIP_FILE, encoding="utf-8") as fh: skipped = {x["doc_no_key"]: x for x in json.load(fh)}
    todo = [(k, v) for k, v in idx.items() if k not in done and k not in skipped]
    logger.info(f"共 {len(idx)} 則，已處理 {len(done) + len(skipped)}，待抓 {len(todo)}")

    for n, (key, meta) in enumerate(todo, 1):
        soup = get(session, meta["url"])
        delay()
        f = parse_doc(soup) if soup else None
        if not f or not f.get("doc_no"):
            logger.warning(f"  [{n}] 解析失敗：{key}")
            continue
        if norm_doc_no(f["doc_no"]) != key and key not in norm_doc_no(f["doc_no"]):
            # 清單順序若在爬取期間變動（有新函釋上架），recordno 會錯位 → 記錄後略過，下次重建清單
            logger.warning(f"  [{n}] 錯位：清單 {key} ≠ 頁面 {f['doc_no']}")
            continue
        reason = skip_reason(f)
        if reason:
            skipped[key] = {"doc_no_key": key, "doc_no": f["doc_no"], "date": f.get("date", ""),
                            "reason": reason, "summary": f.get("summary", "")[:120],
                            "note": f.get("note", "")[:200]}
        else:
            body = "\n".join(f["body"])
            summary = f.get("summary", "")
            done[key] = {
                "source":        "勞動部",
                "source_type":   "gov",
                "category":      "函釋",
                "title":         f["doc_no"],
                "doc_no_key":    key,
                "issuer":        f.get("issuer", ""),
                "url":           permalink(f["fdlink"]) or meta["url"],
                "date":          f.get("date", ""),
                "related_laws":  f["related"],
                "from_articles": meta["from_articles"],
                "summary":       summary,                       # 要旨（一句話）
                "content":       f"要旨：{summary}\n{body}" if summary else body,
                "note":          f.get("note", ""),
                "keywords":      [],
            }
        if n % 50 == 0:
            logger.info(f"  進度 {n}/{len(todo)}：保留 {len(done)}、排除 {len(skipped)}")
            save(done, skipped)
    save(done, skipped)
    logger.info(f"✅ 保留 {len(done)} 則 → {OUT_FILE}")
    logger.info(f"   排除 {len(skipped)} 則：" +
                "、".join(f"{k} {v}" for k, v in Counter(x['reason'] for x in skipped.values()).most_common()))


def save(done, skipped):
    for path, data in ((OUT_FILE, done), (SKIP_FILE, skipped)):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(sorted(data.values(), key=lambda x: x.get("date", ""), reverse=True),
                      fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


if __name__ == "__main__":
    main()
