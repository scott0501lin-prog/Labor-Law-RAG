"""
勞動法規條文爬蟲（勞基法以外、與勞基法互相銜接的法規）
資料來源：全國法規資料庫 (law.moj.gov.tw)

  - 勞工退休金條例（N0030020）：退休新制（6% 個人專戶）、新制年資資遣費（第 12 條）
  - 勞工職業災害保險及保護法（N0050031）：勞保局職災保險給付，可依勞基法第 59 條抵充雇主補償
  - 最低工資法（N0030028）：113 年起取代勞基法第 21 條「基本工資」
  - 勞工請假規則（N0030006）：婚喪病事假、公傷病假等假別與日數（勞基法第 43 條授權）

輸出格式與 backend/data/labor_law_cleaned.json 相同（source / article_no / text），
另加 chapter（所屬章節）、law_date（最新修正日期，未曾修正者為公布日期），並略過「（刪除）」條文。

執行：python backend/scraper2/spiders/law_spider.py
輸出：backend/scraper2/data/<法規名>.json
"""

import io, os, re, ssl, sys, json, time

import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")  # backend/scraper2/data

LAWS = {
    "N0030001": ("勞動基準法", "labor_standards_act.json"),
    "N0030002": ("勞動基準法施行細則", "labor_standards_rules.json"),
    "N0030020": ("勞工退休金條例", "labor_pension_act.json"),
    "N0050031": ("勞工職業災害保險及保護法", "occupational_accident_act.json"),
    "N0030028": ("最低工資法", "minimum_wage_act.json"),
    "N0030006": ("勞工請假規則", "leave_rules.json"),
}
LAW_URL = "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9",
}

ARTICLE_NO_RE = re.compile(r"第\s*(\d+(?:-\d+)?)\s*條")
FLNO_RE       = re.compile(r"[?&]flno=([^&]+)")
DELETED_RE    = re.compile(r"^[（(]\s*刪除\s*[）)]$")


class _RelaxedX509Adapter(HTTPAdapter):
    """law.moj.gov.tw 憑證缺 Subject Key Identifier，OpenSSL 3.2+ 嚴格模式會擋；
    只關閉 VERIFY_X509_STRICT，憑證鏈與網域驗證仍照常進行。"""
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def fetch(url: str) -> BeautifulSoup:
    s = requests.Session()
    s.mount("https://", _RelaxedX509Adapter())
    for i in range(3):
        try:
            r = s.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            print(f"  請求失敗（{i + 1}/3）：{e}")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"無法取得 {url}")


def extract_no(no_cell) -> str | None:
    link = no_cell.select_one("a[href*='flno=']")
    if link:
        m = FLNO_RE.search(link.get("href", ""))
        if m: return m.group(1)
    m = ARTICLE_NO_RE.search(no_cell.get_text(strip=True))
    return m.group(1) if m else None


def parse(soup: BeautifulSoup, law_name: str) -> list[dict]:
    """依序走 law-reg-content 底下的章節標題（h3）與條文列（div.row）。"""
    articles, heads = [], {"章": "", "節": "", "款": ""}
    for el in soup.select("div.law-reg-content > div"):
        cls = el.get("class") or []
        if "h3" in cls:
            title = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            m = re.match(r"第\s*\S+?\s*([章節款])", title)
            level = m.group(1) if m else "款"
            heads[level] = title
            for lower in ("章", "節", "款")[("章", "節", "款").index(level) + 1:]:
                heads[lower] = ""          # 進入新章（節）時清掉下層標題
            continue
        no_cell, data_cell = el.select_one(".col-no"), el.select_one(".col-data")
        if not no_cell or not data_cell: continue
        no = extract_no(no_cell)
        body = data_cell.get_text("\n", strip=True)
        if not no or not body: continue
        if DELETED_RE.match(body):
            print(f"  略過已刪除：第 {no} 條")
            continue
        article_no = f"第 {no} 條"
        articles.append({
            "source":     law_name,
            "article_no": article_no,
            "chapter":    " ".join(v for v in heads.values() if v),
            "text":       f"【{law_name} {article_no}】\n{body}",
        })
    return articles


def law_date(soup: BeautifulSoup) -> str:
    m = re.search(r"(?:修正|公布)日期[：:]\s*民國\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日", soup.get_text())
    if not m: return ""
    return f"{int(m.group(1)) + 1911:04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def main():
    for pcode, (law_name, fname) in LAWS.items():
        soup = fetch(LAW_URL.format(pcode))
        page_name = soup.select_one("#hlLawName")
        # 網站名稱可能附註制定日期，如「最低工資法（112.12.27制定）」→ 以開頭比對
        if page_name and not page_name.get_text(strip=True).startswith(law_name):
            raise RuntimeError(f"pcode {pcode} 對應到「{page_name.get_text(strip=True)}」，不是 {law_name}")
        arts = parse(soup, law_name)
        date = law_date(soup)
        for a in arts: a["law_date"] = date
        nos = [a["article_no"] for a in arts]
        if len(nos) != len(set(nos)):
            raise RuntimeError(f"{law_name} 條號重複，解析有誤")
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(arts, f, ensure_ascii=False, indent=2)
        print(f"✅ {law_name}（{date}）：{len(arts)} 條 → {path}")
        time.sleep(2)


if __name__ == "__main__":
    main()
