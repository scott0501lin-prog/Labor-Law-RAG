"""
司法院裁判書爬蟲 v2（勞動事件判決，全文 + 段落解析）
來源：https://judgment.judicial.gov.tw/FJUD/Default_AD.aspx（不需瀏覽器，直接送出查詢表單）

與舊版 scrapers/judicial_spider.py 的差異：
  - 舊版：Playwright、只爬臺北地院、主文截 500 字＋理由截 2000 字（多半只截到兩造主張，
    法院判斷被截掉）、未排除裁定
  - 新版：
    1. 依「勞動專庭字別」查詢（勞動事件法 109 年施行後設立）：地院 勞訴／勞簡，高院 勞上／勞上易，
       最高法院 台上（以案由篩選勞動案件）
    2. 只收「判決」，排除裁定
    3. 依案由分類後平均抽樣，避免集中在少數爭點
    4. 存全文，並把段落標上所屬章節（主文／原告主張／被告抗辯／不爭執事項／爭點／本院之判斷／結論…），
       供建索引時「依段落切塊並保存所屬段落」（覆核書 3(2)）
    5. 額外擷取：引用法條、判決結果、主文金額、規則式摘要（供案例查詢卡片顯示，覆核書 C6）

執行：
  python backend/scraper2/spiders/judicial_spider.py list    # 階段 1：只抓列表 → judicial_candidates.json
  python backend/scraper2/spiders/judicial_spider.py fetch   # 階段 2：抽樣並抓全文 → judicial_cases.json
  python backend/scraper2/spiders/judicial_spider.py         # 兩階段一起跑
  python backend/scraper2/spiders/judicial_spider.py reparse # 不連線，用已存段落重新解析（調整規則後用）
階段 2 可中斷續跑（已抓過的判決會略過）。
"""

import io, os, re, sys, json, time, random, logging
from collections import defaultdict
from urllib.parse import urljoin, unquote

import requests
from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)

HERE            = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # backend/scraper2/data
CANDIDATES_FILE = os.path.join(HERE, "judicial_candidates.json")
OUT_FILE        = os.path.join(HERE, "judicial_cases.json")

BASE       = "https://judgment.judicial.gov.tw/FJUD/"
SEARCH_URL = BASE + "Default_AD.aspx"
HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}
DELAY_MIN, DELAY_MAX = 2.0, 4.0

# ── 查詢範圍 ───────────────────────────────────────
YEAR_FROM, YEAR_TO = 112, 115          # 民國裁判年度（2023–2026）
MAX_LIST_PAGES     = 15                # 每組查詢最多翻幾頁（每頁 20 筆）

# (法院代碼, 法院名, 字別, 案由關鍵字或 None)
QUERIES = [
    ("TPH", "臺灣高等法院",        "勞上",   None),
    ("TPH", "臺灣高等法院",        "勞上易", None),
    ("TCH", "臺灣高等法院臺中分院", "勞上",   None),
    ("KSH", "臺灣高等法院高雄分院", "勞上",   None),
    ("TPD", "臺灣臺北地方法院",    "勞訴",   None),
    ("PCD", "臺灣新北地方法院",    "勞訴",   None),
    ("TYD", "臺灣桃園地方法院",    "勞訴",   None),
    ("TCD", "臺灣臺中地方法院",    "勞訴",   None),
    ("KSD", "臺灣高雄地方法院",    "勞訴",   None),
    ("TPD", "臺灣臺北地方法院",    "勞簡",   None),
]
# 最高法院沒有勞動專用字別（一律「台上」），改用案由篩選
SUPREME_TITLES = ["給付工資", "給付資遣費", "確認僱傭關係", "給付加班費",
                  "給付退休金", "職業災害補償"]

# ── 依案由分類（用於平均抽樣）─────────────────────────
TOPIC_RULES = [
    ("職業災害",     r"職業災害|職災"),
    ("退休金",       r"退休金|勞工退休|提繳"),
    ("資遣費",       r"資遣"),
    ("加班費",       r"加班|延長工時"),
    ("確認僱傭關係", r"僱傭關係|聘任關係|聘僱關係|勞動契約存在|回復原職|恢復原職"),
    ("特休",         r"特別休假|特休"),
    ("競業禁止",     r"競業|違約金|最低服務"),
    ("工資",         r"工資|薪資|獎金|報酬"),
]
PER_TOPIC = 70        # 每類最多抽幾筆
OTHER_CAP = 40        # 「其他」類最多幾筆


# 排除的案由（不適用勞基法者，如教師聘任適用教師法）
EXCLUDE_TITLE_RE = re.compile(r"教師|解聘|聘任")


def topic_of(title: str) -> str:
    for name, pat in TOPIC_RULES:
        if re.search(pat, title or ""):
            return name
    return "其他"


# ── 工具 ──────────────────────────────────────────
def delay(): time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("　", " ")).strip()

def compact(t: str) -> str:
    return re.sub(r"[\s　\xa0]+", "", t or "")

def get(session, url, retries=3):
    for i in range(retries):
        try:
            r = session.get(url, headers=HEADERS, timeout=40)
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except requests.RequestException as e:
            logger.warning(f"  請求失敗（{i + 1}/{retries}）：{e}")
            time.sleep(5 * (i + 1))
    return None


# ══════════════════════════════════════════════════
# 階段 1：列表
# ══════════════════════════════════════════════════
def search(session, court, case_word="", title=""):
    """送出進階查詢，回傳結果列表第 1 頁網址。"""
    html = get(session, SEARCH_URL)
    if not html: return None
    form = BeautifulSoup(html, "html.parser")
    data = {i["name"]: i.get("value", "")
            for i in form.select("input[type=hidden]") if i.get("name")}
    data.update({
        "jud_court": court, "jud_sys": "V", "jud_year": "", "sel_judword": "",
        "jud_case": case_word, "jud_no": "", "jud_no_end": "",
        "dy1": str(YEAR_FROM), "dm1": "1", "dd1": "1",
        "dy2": str(YEAR_TO), "dm2": "12", "dd2": "31",
        "jud_title": title, "jud_jmain": "", "jud_kw": "",
        "KbStart": "", "KbEnd": "", "judtype": "JUDBOOK", "whosub": "0",
        "ctl00$cp_content$btnQry": "送出查詢",
    })
    try:
        r = session.post(SEARCH_URL, data=data, headers=HEADERS, timeout=40)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"  查詢失敗：{e}")
        return None
    m = re.search(r"qryresultlst\.aspx\?[^\"']+", r.text)
    return urljoin(BASE, m.group(0).replace("&amp;", "&")) if m else None


def parse_result_page(html):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    for tr in soup.select("table#jud tr"):
        a = tr.select_one("a#hlTitle")
        tds = tr.find_all("td")
        if not a or len(tds) < 4: continue
        items.append({
            "case_no":  clean(a.get_text()),
            "url":      urljoin(BASE, a["href"]),
            "jdate":    clean(tds[2].get_text()),
            "title":    clean(tds[3].get_text()),        # 裁判案由
            "size_k":   int(re.search(r"(\d+)K", tds[1].get_text()).group(1))
                        if re.search(r"(\d+)K", tds[1].get_text()) else 0,
        })
    nxt = next((a["href"] for a in soup.find_all("a", href=True)
                if a.get_text(strip=True) == "下一頁"), None)
    total = re.search(r"共\s*(\d+)\s*筆", soup.get_text(" "))
    return items, (urljoin(BASE, nxt) if nxt else None), (int(total.group(1)) if total else None)


def crawl_list(session, court, court_name, case_word="", title=""):
    url = search(session, court, case_word, title)
    delay()
    label = f"{court_name} {case_word or ''}{('案由:' + title) if title else ''}"
    if not url:
        logger.warning(f"▶ {label}：查無結果或查詢失敗")
        return []
    out, page = [], 0
    while url and page < MAX_LIST_PAGES:
        html = get(session, url)
        delay()
        if not html: break
        items, url, total = parse_result_page(html)
        if page == 0: logger.info(f"▶ {label}：共 {total} 筆")
        out.extend(items)
        page += 1
    judgments = [x for x in out if x["case_no"].endswith("判決")
                 and not EXCLUDE_TITLE_RE.search(x["title"])]
    for x in judgments:
        x.update({"court": court_name, "case_word": case_word or "台上",
                  "level": "最高法院" if court == "TPS" else
                           ("高等法院" if court.endswith("H") else "地方法院"),
                  "topic": topic_of(x["title"])})
    logger.info(f"    抓到 {len(out)} 筆，其中判決 {len(judgments)} 筆")
    return judgments


def stage_list():
    session = requests.Session()
    found = {}
    for court, name, word, title in QUERIES:
        for it in crawl_list(session, court, name, word, title or ""):
            found.setdefault(it["url"], it)
    for t in SUPREME_TITLES:
        for it in crawl_list(session, "TPS", "最高法院", "台上", t):
            found.setdefault(it["url"], it)
    cands = list(found.values())
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=2)
    cnt = defaultdict(int)
    for c in cands: cnt[c["topic"]] += 1
    logger.info(f"✅ 候選判決 {len(cands)} 筆 → {CANDIDATES_FILE}")
    logger.info("   各類：" + "、".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])))
    return cands


# ══════════════════════════════════════════════════
# 階段 2：抽樣 + 全文 + 解析
# ══════════════════════════════════════════════════
def sample(cands):
    """每類依「審級 → 日期新→舊」輪流挑，讓各法院層級都有代表。"""
    rnd = random.Random(42)
    by_topic = defaultdict(list)
    for c in cands:
        if c["size_k"] and c["size_k"] < 4:      # 太短的多半是一造辯論或程序性判決
            continue
        by_topic[c["topic"]].append(c)
    picked = []
    for topic, items in by_topic.items():
        cap = OTHER_CAP if topic == "其他" else PER_TOPIC
        by_level = defaultdict(list)
        for it in items: by_level[it["level"]].append(it)
        for lv in by_level.values():
            rnd.shuffle(lv)
        order = ["最高法院", "高等法院", "地方法院"]
        while len([p for p in picked if p["topic"] == topic]) < cap and any(by_level.values()):
            for lv in order:
                if by_level[lv] and len([p for p in picked if p["topic"] == topic]) < cap:
                    picked.append(by_level[lv].pop())
    return picked


SECTION_RULES = [   # (章節名, 標題內容的 regex)；只比對段落開頭的短標題
    ("程序",       r"^程序(方面|部分|上理由)"),
    ("實體",       r"^實體(方面|部分|上理由)"),
    # 兩造標題不限定稱謂（最高法院、高院的「上訴人」常是雇主）：主張者＝原告主張，則以／抗辯者＝被告抗辯
    ("原告主張",   r"^(本件)?(原告|被告|上訴人|被上訴人)(即(原告|被告))?\S{0,25}?(主張|起訴意旨|請求意旨|意旨)[：:]"),
    ("被告抗辯",   r"^(本件)?(原告|被告|上訴人|被上訴人)(即(原告|被告))?\S{0,25}?(則以|抗辯|答辯|辯稱|辯以|陳述|則謂)[：:]"),
    ("原審認定",   r"^原審(審理結果|就|以|認|斟酌|判決)"),
    ("不爭執事項", r"不爭執"),
    ("爭點",       r"爭點|爭執事項|爭執要旨"),
    ("本院之判斷", r"本院之判斷|本院判斷|得心證之理由|本院得心證|法院之判斷"),
]
ENUM_RE   = re.compile(r"^(?:[壹貳參肆伍陸柒捌玖拾]+|[一二三四五六七八九十]+|[（(][一二三四五六七八九十]+[）)])[、．.]?")
MAIN_RE   = re.compile(r"^主文$")
REASON_RE = re.compile(r"^(事實及理由|事實與理由|理由|事實|理由要領)$")
CONCL_RE  = re.compile(r"^(綜上所述|綜上|據上論結|據上論斷|從而)")
SIGN_RE   = re.compile(r"^中華民國\d+年\d+月\d+日")
PARTY_SECS = ("原告主張", "被告抗辯", "不爭執事項", "爭點", "原審認定")
COURT_START_RE = re.compile(r"^(按|經查|查|惟查|惟按|次按|復按|再按|又按|經核|本院|惟)")
PARA_START_RE  = re.compile(r"^([壹貳參肆伍陸柒捌玖拾一二三四五六七八九十]+、|[（(][一二三四五六七八九十]+[）)]"
                            r"|[㈠-㈩⒈-⒛①-⑳]|主文$|事實及理由$|事實與理由$|理由$|中華民國\d|附表)")


def enum_style(c):
    """段落開頭編號的層級樣式：壹、→A；一、→B；（一）→C；無→None。"""
    if re.match(r"^[壹貳參肆伍陸柒捌玖拾]+、", c): return "A"
    if re.match(r"^[一二三四五六七八九十]+、", c): return "B"
    if re.match(r"^[（(][一二三四五六七八九十]+[）)]", c): return "C"
    return None


def paragraphs(soup):
    box = soup.select_one(".htmlcontent")
    if box and box.find("div"):
        ps = [d.get_text("") for d in box.find_all("div", recursive=False)]
    else:                                         # 舊式純文字判決
        pre = soup.select_one(".text-pre") or soup.select_one("#jud")
        ps = (pre.get_text("\n") if pre else "").split("\n")
    return merge_wrapped([p.replace("\xa0", " ").strip() for p in ps if p and p.strip()])


def merge_wrapped(ps):
    """舊式判決每行固定寬度斷行（約 28 字）→ 接回段落：遇到編號／章節標題，或上一行以句號、冒號結尾才換段。"""
    if len(ps) < 20: return ps
    lens = sorted(len(compact(p)) for p in ps)
    if lens[len(lens) // 2] > 40: return ps
    out = []
    for p in ps:
        c = compact(p)
        if (out and not PARA_START_RE.match(c) and not out[-1].endswith(("。", "：", ":"))
                and not MAIN_RE.match(out[-1]) and not REASON_RE.match(out[-1])):
            out[-1] += c
        else:
            out.append(c)
    return out


def is_argument(sec, style, party_style, c):
    """地院常見寫法：「三、本件爭點」之下直接以（一）（二）逐點論述 → 即法院判斷。"""
    return (sec == "爭點" and style != party_style and len(c) > 100
            and not c.endswith(("？", "?")))


def label_sections(paras):
    """回傳 [(章節, 段落文字)]。"""
    out, sec = [], "前言"
    party_style = None        # 「原告主張」那一層的編號樣式（判斷同層級的下一個大項）
    for p in paras:
        c = compact(p)
        if MAIN_RE.match(c):
            sec = "主文"; continue
        if REASON_RE.match(c):
            sec = "事實及理由"; continue
        if sec in ("前言", "主文", "署名"):
            if SIGN_RE.match(c): sec = "署名"
            out.append((sec, p)); continue
        if SIGN_RE.match(c):
            sec = "署名"; out.append((sec, p)); continue

        style = enum_style(c)
        body = ENUM_RE.sub("", c)
        top_level = style is not None and (party_style is None or style <= party_style)
        if CONCL_RE.match(body) and (top_level or style is None and sec != "本院之判斷"):
            sec = "結論"
        elif (style in ("A", "B") or (style == "C" and sec not in ("本院之判斷", "結論")) or len(c) <= 40
              or (style is None and sec in ("事實及理由",) + PARTY_SECS)):
            matched = None
            for name, pat in SECTION_RULES:
                if re.search(pat, body[:40]):
                    matched = name
                    break
            if matched == "實體":
                sec, party_style = "事實及理由", None
            elif matched in PARTY_SECS[:2] and style == "C":
                pass                      # 下層小標（如「(二)原告主張其…」）不切回兩造主張
            elif matched and not (sec in ("本院之判斷", "結論") and matched in PARTY_SECS):
                if matched in ("原告主張", "被告抗辯") and party_style is None:
                    party_style = style
                sec = matched
            elif not matched and sec in PARTY_SECS and style and style == party_style:
                # 高院常見寫法：兩造主張／不爭執事項之後，下一個同層大項即法院判斷（無標題）
                sec = "本院之判斷"
            elif (not matched and sec in PARTY_SECS and style == party_style
                  and COURT_START_RE.match(body)):
                # 無編號／無標題寫法：兩造主張之後以「按、經查、本院…」開頭的同層段落 → 法院判斷
                sec = "本院之判斷"
            elif not matched and is_argument(sec, style, party_style, c):
                sec = "本院之判斷"
        elif is_argument(sec, style, party_style, c):
            sec = "本院之判斷"
        out.append((sec, p))
    return out


CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
def cn2int(s):
    if s.isdigit(): return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch in CN_NUM: cur = CN_NUM[ch]
        elif ch == "十": total += (cur or 1) * 10; cur = 0
        elif ch == "百": total += (cur or 1) * 100; cur = 0
    return total + cur

LAW_ALIASES = {"勞動基準法": "勞動基準法", "勞基法": "勞動基準法",
               "勞工退休金條例": "勞工退休金條例", "勞退條例": "勞工退休金條例",
               "勞工職業災害保險及保護法": "勞工職業災害保險及保護法", "災保法": "勞工職業災害保險及保護法",
               "勞動基準法施行細則": "勞動基準法施行細則", "民法": "民法",
               "勞動事件法": "勞動事件法", "職業災害勞工保護法": "職業災害勞工保護法"}
LAW_RE = re.compile(
    r"(勞動基準法施行細則|勞動基準法|勞基法|勞工退休金條例|勞退條例|勞工職業災害保險及保護法|災保法|"
    r"職業災害勞工保護法|勞動事件法|民法)第\s*([0-9一二三四五六七八九十百〇零]+)\s*條(?:之\s*([0-9一二三四五六七八九十]+))?")

def cited_laws(text):
    seen = []
    for m in LAW_RE.finditer(text):
        law = LAW_ALIASES[m.group(1)]
        no = str(cn2int(m.group(2))) + (f"-{cn2int(m.group(3))}" if m.group(3) else "")
        key = f"{law} 第 {no} 條"
        if key not in seen: seen.append(key)
    return seen


def judge_result(main_text, level):
    m = compact(main_text)
    if level == "地方法院":
        if re.search(r"原告之訴(及假執行之聲請)?(均)?駁回", m): return "原告敗訴"
        if re.search(r"其餘之訴(均)?駁回", m): return "原告部分勝訴"
        if re.search(r"應給付|應提繳|確認.*存在|應(自|將)|應回復|應恢復", m): return "原告勝訴"
        return "其他"
    if re.search(r"原判決(關於)?.{0,30}(部分)?廢棄", m):
        return "原判決部分廢棄" if re.search(r"其餘上訴駁回|部分廢棄|關於.{0,30}廢棄", m) else "原判決廢棄"
    if re.search(r"上訴(均)?駁回", m): return "上訴駁回（維持原判）"
    return "其他"


def amounts(main_text):
    vals = []
    for m in re.finditer(r"新臺幣\s*([0-9,，]+)\s*元", main_text):
        v = int(re.sub(r"[,，]", "", m.group(1)) or 0)
        if v: vals.append(v)
    return vals


def parse_judgment(html, meta):
    y, mo, d = (int(x) for x in re.findall(r"\d+", meta["jdate"])[:3])
    return build_record(paragraphs(BeautifulSoup(html, "html.parser")), meta,
                        f"{y + 1911:04d}-{mo:02d}-{d:02d}")


# 司法院網站用私有區字元（Private Use Area）顯示部分編號與運算符號，依上下文推得對應：
#   U+F6B0 起遞減 = 一、二、…十三、（例：主文「十、」之後接 U+F6A6、U+F6A5 = 十一、十二）
#   U+F674 起遞減 = 下一層編號 ⑴⑵⑶⑷⑸（同一判決內依序出現）
#   U+F4E9 = ×、U+F4E8 = ÷（計算式：15萬元×6%＝9,000）
# 其餘少數（附表欄位符號、罕用字）無法確定，保留原字元。
CN_ENUM = "一二三四五六七八九十"
PUA_MAP = {chr(0xF6B0 - i): (CN_ENUM[i] if i < 10 else "十" + CN_ENUM[i - 10]) + "、"
           for i in range(13)}
PUA_MAP.update({chr(0xF674 - i): "⑴⑵⑶⑷⑸"[i] for i in range(5)})
PUA_MAP.update({"": "×", "": "÷"})
PUA_TABLE = str.maketrans(PUA_MAP)


def normalize(p):
    """私有區編號字元還原；全形空白、不換行空白 → 一般空白，並合併連續空白。"""
    p = p.translate(PUA_TABLE).replace("　", " ").replace("\xa0", " ")
    return re.sub(r" {2,}", " ", p).strip()


def build_record(paras, meta, date_iso):
    labeled = label_sections([normalize(p) for p in paras])
    if not labeled: return None
    sec_text = defaultdict(list)
    for sec, p in labeled: sec_text[sec].append(p)
    main_text = "\n".join(sec_text["主文"])
    court_part = "\n".join(sec_text["本院之判斷"] + sec_text["結論"])
    body = "\n".join(p for s, p in labeled if s not in ("前言", "署名"))
    result = judge_result(main_text, meta["level"])
    amts = amounts(main_text)
    brief = f"【{meta['topic']}】{meta['title']}｜{meta['court']}｜{result}"
    if amts and result not in ("原告敗訴", "上訴駁回（維持原判）"):
        brief += f"｜主文金額最高 {max(amts):,} 元"
    return {
        "source":          "司法院",
        "source_type":     "judicial",
        "category":        f"裁判書-{meta['topic']}",
        "topic":           meta["topic"],
        "title":           meta["case_no"],
        "case_title":      meta["title"],          # 裁判案由
        "court":           meta["court"],
        "level":           meta["level"],
        "case_word":       meta["case_word"],
        "url":             meta["url"],
        "date":            date_iso,
        "result":          result,
        "main_amounts":    amts,
        "cited_laws":      cited_laws(body),
        "brief":           brief,                  # 規則式摘要，供卡片顯示；之後可換成 LLM 摘要
        "main_text":       main_text,
        "court_reasoning": court_part,             # 本院之判斷＋結論（RAG 主要檢索內容）
        # False＝規則沒切出法院判斷（約 4%，多為不爭執事項與判斷混寫）→ 建索引時改用全文 full_text(rec)
        "reasoning_ok":    len(court_part) >= 0.1 * max(1, len(body)),
        # 不存「前言」（當事人、律師姓名）與「署名」（法官、書記官）：對檢索無用，也少存個人姓名
        "sections":        [{"section": s, "text": p} for s, p in labeled if s not in ("前言", "署名")],
    }
    # 不另存全文 content：與 sections 重複，需要時用 full_text(rec) 由 sections 接回


def full_text(rec):
    """判決全文（不含當事人表頭與署名）＝ sections 依序接回。"""
    return "\n".join(x["text"] for x in rec["sections"])


def stage_fetch(cands=None):
    if cands is None:
        with open(CANDIDATES_FILE, encoding="utf-8") as f:
            cands = json.load(f)
    picked = sample(cands)
    done = {}
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f:
            done = {x["url"]: x for x in json.load(f)}
    logger.info(f"抽樣 {len(picked)} 筆（已完成 {len(done)} 筆會略過）")
    session = requests.Session()
    for n, meta in enumerate(picked, 1):
        if meta["url"] in done: continue
        html = get(session, meta["url"])
        delay()
        if not html:
            logger.warning(f"  [{n}] 抓取失敗：{meta['case_no']}")
            continue
        rec = parse_judgment(html, meta)
        if not rec or len(full_text(rec)) < 300:
            logger.warning(f"  [{n}] 內容過短或無法解析，略過：{meta['case_no']}")
            continue
        done[meta["url"]] = rec
        if n % 20 == 0:
            logger.info(f"  進度 {n}/{len(picked)}，已存 {len(done)} 筆")
            save(done)
    save(done)
    logger.info(f"✅ 共 {len(done)} 筆 → {OUT_FILE}")


def stage_reparse():
    """不連線：用已存的段落重新標章節、重算欄位（調整解析規則後使用）。"""
    with open(OUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    done = {}
    for r in data:
        secs = r["sections"]
        paras = [x["text"] for x in secs if x["section"] == "前言"] + ["主文"]
        paras += [x["text"] for x in secs if x["section"] == "主文"] + ["事實及理由"]
        paras += [x["text"] for x in secs if x["section"] not in ("前言", "主文")]
        meta = {"case_no": r["title"], "title": r["case_title"], "court": r["court"],
                "level": r["level"], "case_word": r["case_word"], "topic": r["topic"],
                "url": r["url"]}
        done[r["url"]] = build_record(merge_wrapped(paras), meta, r["date"])
    save(done)
    logger.info(f"✅ 重新解析 {len(done)} 筆 → {OUT_FILE}")


def save(done):
    tmp = OUT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(done.values()), f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_FILE)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "list":
        stage_list()
    elif mode == "reparse":
        stage_reparse()
    elif mode == "fetch":
        stage_fetch()
    else:
        stage_fetch(stage_list())
