"""
Article → penalty-article lookup, parsed from the statutes themselves.

Penalty provisions cite other articles by number only («違反第二十二條至第二
十五條……處新臺幣二萬元以上一百萬元以下罰鍰»), so retrieval almost never
finds them for a question about wages or overtime (1 of 93 台北 FAQ questions
had one in the top 5). Instead, after retrieval each retrieved article is
looked up here and the penalty article that cites it is added to the prompt.
Amounts come from the statute text, not from the model's memory.

Pure Python (no Streamlit) so tests can import it.
"""
from __future__ import annotations

import re

from hybrid_search import cn_to_int

_NUM = r"[0-9零〇一二兩三四五六七八九十百千]+"
# 第二十二條 / 第十七條之一 (paragraph refs like 第一項 are ignored: we map
# whole articles). Optional range: 第二十二條至第二十五條.
_REF = re.compile(
    rf"第({_NUM})條(?:之({_NUM}))?(?:第{_NUM}項(?:至第{_NUM}項)?)?"
    rf"(?:至第({_NUM})條(?:之({_NUM}))?)?"
)
# A citation of a different statute («勞動基準法第…», «不適用勞動基準法之罰鍰規定»)
_OTHER_LAW = re.compile(r"(?:法|條例|規則)$")

# Statutes whose penalty clause names the violation instead of citing an
# article: every article of the law maps to that clause.
_WHOLE_LAW = {"最低工資法": "第 17 條"}

# Generic provisions (publication of offenders, liability of legal persons,
# collection) — they apply to every violation, so they'd only add noise.
_SKIP = {
    ("勞動基準法", "第 80-1 條"), ("勞動基準法", "第 81 條"), ("勞動基準法", "第 82 條"),
    ("勞工退休金條例", "第 5 條"), ("勞工退休金條例", "第 53-1 條"),
    ("勞工退休金條例", "第 54 條"), ("勞工退休金條例", "第 55 條"),
    ("勞工職業災害保險及保護法", "第 59 條"), ("勞工職業災害保險及保護法", "第 100 條"),
}

PENALTY_WORDS = ("罰鍰", "罰金")


def _art_no(main: int, sub: int | None = None) -> str:
    return f"第 {main}-{sub} 條" if sub else f"第 {main} 條"


def cited_articles(text: str) -> set[str]:
    """Article numbers (index format, e.g. '第 22 條', '第 17-1 條') cited in *text*."""
    out: set[str] = set()
    for m in _REF.finditer(text):
        if _OTHER_LAW.search(text[max(0, m.start() - 6):m.start()]):
            continue
        a, a_sub, b, b_sub = (cn_to_int(g) if g else None for g in m.groups())
        if a is None:
            continue
        out.add(_art_no(a, a_sub))
        if b is not None and b > a:            # 第二十二條至第二十五條
            out.update(_art_no(n) for n in range(a + 1, b + 1))
            if b_sub:
                out.add(_art_no(b, b_sub))
    return out


def build_penalty_map(texts: list[str], metas: list[dict]) -> dict[tuple, list[tuple]]:
    """{(law, article_no): [(law, penalty_article_no), ...]} from the law index."""
    full: dict[tuple, str] = {}
    for t, m in zip(texts, metas):
        key = (m.get("source"), m.get("article_no"))
        full.setdefault(key, m.get("display_text") or t)

    pmap: dict[tuple, list[tuple]] = {}

    def add(target, pen):
        lst = pmap.setdefault(target, [])
        if pen not in lst and pen != target:
            lst.append(pen)

    for (law, art), text in full.items():
        if (law, art) in _SKIP or not any(w in text for w in PENALTY_WORDS):
            continue
        body = re.sub(r"^【[^】]*】\s*", "", text)  # drop the 【law 第 N 條】 header
        for cited in cited_articles(body):
            add((law, cited), (law, art))

    for law, pen_art in _WHOLE_LAW.items():
        for (l, art) in full:
            if l == law:
                add((l, art), (law, pen_art))
    return pmap


def penalties_for(results: list[tuple], pmap: dict, limit: int = 2) -> list[tuple]:
    """Penalty articles for retrieved [(text, meta, score)] not already in it."""
    have = {(m.get("source"), m.get("article_no")) for _, m, _ in results}
    out: list[tuple] = []
    for _, m, _ in results:
        for pen in pmap.get((m.get("source"), m.get("article_no")), []):
            if pen not in have and pen not in out:
                out.append(pen)
    return out[:limit]


_ITEM = re.compile(r"^[一二三四五六七八九十]+、")


def label_paragraphs(text: str) -> str:
    """Prefix each 項 with 「第 N 項：」 so the model cites the right paragraph.

    Statute text puts one 項 per line, with 款 (一、二、…) as further lines
    under the 項 they belong to; models tend to count the 款 as 項 (an
    employer answer cited 第 79 條「第 4 項」 for what is 第 2 項).
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and lines[0].startswith("【"):
        header, lines = lines[0], lines[1:]
    else:
        header = ""
    out, n = [], 0
    for line in lines:
        if _ITEM.match(line.strip()):
            out.append("　" + line.strip())          # 款: stays under its 項
        else:
            n += 1
            out.append(f"第 {n} 項：{line.strip()}")
    return "\n".join(([header] if header else []) + out)
