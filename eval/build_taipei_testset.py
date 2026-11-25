"""
Build a retrieval test set from 台北市勞動局常見問答 (scraper2/data/taipei_faq_cases.json).

Each FAQ answer cites statutes ("依勞動基準法第70條及其施行細則第37條…").
Those citations become the expected law articles for the FAQ question, so the
set measures *law* retrieval: query = question title, hit = an expected
article appears in the top-k law results.

Note: backend/build_index.py leaves these questions out of the case index, so
the system never retrieves a test question's own answer. After regenerating
this set, rebuild the case index (python backend/build_index.py --only cases).

Usage:  python eval/build_taipei_testset.py
Output: eval/taipei_faq_testset.json
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Fronted"))
from hybrid_search import cn_to_int  # noqa: E402

S2 = os.path.join(ROOT, "backend", "scraper2", "data")
SRC = os.path.join(S2, "taipei_faq_cases.json")
OUT = os.path.join(ROOT, "eval", "taipei_faq_testset.json")

LAB, RUL = "勞動基準法", "勞動基準法施行細則"
PEN, OCC = "勞工退休金條例", "勞工職業災害保險及保護法"
MIN, LEAVE = "最低工資法", "勞工請假規則"
LAW_FILES = ["labor_standards_act", "labor_standards_rules", "labor_pension_act",
             "occupational_accident_act", "minimum_wage_act", "leave_rules"]

# Name → law in our corpus. "施行細則" forms refer to 勞基法's rules in these
# FAQs (all categories are 勞基法 topics), as does a bare "本法".
_KNOWN = [
    ("勞動基準法施行細則", RUL), ("勞基法施行細則", RUL), ("本法施行細則", RUL),
    ("同法施行細則", RUL), ("其施行細則", RUL), ("施行細則", RUL),
    ("勞動基準法", LAB), ("勞基法", LAB), ("本法", LAB),
    ("勞工退休金條例", PEN), ("勞退條例", PEN),
    ("勞工職業災害保險及保護法", OCC), ("災保法", OCC),
    ("最低工資法", MIN), ("勞工請假規則", LEAVE), ("請假規則", LEAVE),
]
# Laws outside the corpus: articles cited after them are ignored.
_OTHER = ["性別平等工作法", "性別工作平等法", "性平法", "就業服務法", "職業安全衛生法",
          "勞工保險條例", "勞資爭議處理法", "勞動事件法", "民法", "刑法", "公司法",
          "所得稅法", "就業保險法", "大量解僱勞工保護法", "工會法", "團體協約法",
          "勞動檢查法", "職工福利金條例", "身心障礙者權益保障法",
          "中高齡者及高齡者就業促進法", "勞工保險", "行政程序法", "職業災害勞工保護法"]

_NUM = r"[0-9０-９一二三四五六七八九十百零〇]+"
_names = sorted([n for n, _ in _KNOWN] + _OTHER, key=len, reverse=True)
_TOKEN = re.compile(
    "(?P<law>" + "|".join(map(re.escape, _names)) + ")"
    + rf"|(?P<art>第\s*(?P<n>{_NUM})\s*(?:[-－]\s*(?P<s1>{_NUM})\s*)?條(?:\s*之\s*(?P<s2>{_NUM}))?)"
)
_LAW_OF = dict(_KNOWN)


def extract_citations(text: str) -> list[tuple[str, str]]:
    """[(law, 'article_no')] in order of appearance, deduplicated."""
    cur = None
    out = []
    for m in _TOKEN.finditer(text):
        if m.group("law"):
            cur = _LAW_OF.get(m.group("law"), "OTHER")
            continue
        if cur in (None, "OTHER"):
            continue
        n = cn_to_int(m.group("n"))
        sub = m.group("s1") or m.group("s2")
        if n is None:
            continue
        art = f"第 {n}-{cn_to_int(sub)} 條" if sub else f"第 {n} 條"
        if (cur, art) not in out:
            out.append((cur, art))
    return out


def main():
    with open(SRC, encoding="utf-8-sig") as f:
        faqs = json.load(f)
    corpus = set()
    for fn in LAW_FILES:
        with open(os.path.join(S2, fn + ".json"), encoding="utf-8-sig") as f:
            corpus |= {(x["source"], x["article_no"]) for x in json.load(f)}

    items, missing, no_cite, self_cite = [], [], 0, 0
    for i, x in enumerate(faqs):
        # A question that names an article (「勞基法第56條…」) hands the answer to
        # law retrieval: an explicit citation gets a rank-1 vote in hybrid_search.
        if extract_citations(x["title"]):
            self_cite += 1
            continue
        cites = extract_citations(x["content"])
        valid = [c for c in cites if c in corpus]
        missing += [(x["title"], c) for c in cites if c not in corpus]
        if not valid:
            no_cite += 1
            continue
        items.append({
            "id": f"taipei_faq_{i}",
            "question": x["title"].strip(),
            "category": x.get("category", ""),
            "expected": [{"law": l, "article_no": a} for l, a in valid],
            "answer": x["content"].strip(),
            "url": x.get("url", ""),
        })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)

    sys.stdout.reconfigure(encoding="utf-8")
    print(f"FAQs: {len(faqs)}  →  test items: {len(items)}  "
          f"(question cites an article: {self_cite}, no usable citation: {no_cite})")
    print(f"expected articles: {sum(len(x['expected']) for x in items)}")
    by_law = {}
    for x in items:
        for e in x["expected"]:
            by_law[e["law"]] = by_law.get(e["law"], 0) + 1
    print("by law:", by_law)
    if missing:
        print(f"citations not in corpus ({len(missing)}):")
        for t, c in missing:
            print(f"  {c} ← {t[:30]}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
