"""
BM25 keyword retrieval (jieba + custom legal dictionary) fused with vector
results via Reciprocal Rank Fusion (RRF).

Purpose: vector search is weak on exact article numbers ("第 24 條") and rare
legal terms; BM25 catches those. Pure Python — no Streamlit deps, so eval
scripts can import it directly.
"""
from __future__ import annotations

import os
import re

import jieba
from rank_bm25 import BM25Okapi

_DICT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal_dict.txt")

RRF_K = int(os.getenv("RRF_K", "60"))

# ── Law-name aliases (applied to both documents and queries) ─────────────────
# Longest first so "勞基法施行細則" is replaced before "勞基法".
_LAW_ALIASES = [
    ("勞基法施行細則", "勞動基準法施行細則"),
    ("勞基細則", "勞動基準法施行細則"),
    ("性別工作平等法", "性別平等工作法"),
    ("性工法", "性別平等工作法"),
    ("性平法", "性別平等工作法"),
    ("勞基法", "勞動基準法"),
]

# ── Article-number normalisation ─────────────────────────────────────────────
_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUM = r"[0-9０-９零〇一二兩三四五六七八九十百千]+"

# 第 24 條 / 第二十四條 / 第 32-1 條 / 第三十二條之一 / 第32條-1
_ART_RE = re.compile(
    rf"第\s*({_NUM})\s*(?:[-－之]\s*({_NUM})\s*)?條(?:\s*之\s*({_NUM}))?"
)
# Bare "24條" / "32-1條" typed by users without 第
_BARE_ART_RE = re.compile(r"(?<![第0-9０-９])([0-9０-９]+)\s*(?:[-－]\s*([0-9０-９]+)\s*)?條")


def cn_to_int(s: str) -> int | None:
    s = s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            cur = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (cur or 1) * _CN_UNITS[ch]   # "十二" → 1*10 + 2
            cur = 0
        else:
            return None
    return total + cur


def _art_token(main: str, sub: str | None) -> str | None:
    n = cn_to_int(main)
    if n is None:
        return None
    if sub:
        m = cn_to_int(sub)
        if m is not None:
            return f"第{n}-{m}條"
    return f"第{n}條"


def normalize(text: str) -> tuple[str, list[str]]:
    """Expand law aliases and pull article references out as canonical tokens.

    Returns (remaining_text, article_tokens). Article spans are blanked out of
    the text so jieba doesn't shred them into "第", "24", "條".
    """
    for short, full in _LAW_ALIASES:
        text = text.replace(short, full)
    arts: list[str] = []

    def _sub(m: re.Match) -> str:
        g = m.groups() + (None,)
        tok = _art_token(g[0], g[1] or g[2])
        if tok:
            arts.append(tok)
        return " "

    text = _ART_RE.sub(_sub, text)
    text = _BARE_ART_RE.sub(_sub, text)
    return text, arts


# ── Tokeniser ────────────────────────────────────────────────────────────────
_STOPWORDS = set(
    "的 了 是 我 你 他 她 它 在 有 和 與 及 或 也 都 就 而 但 被 把 嗎 呢 吧 啊 喔 "
    "這 那 個 之 其 於 為 以 對 等 者 所 如 若 則 應 得 可 不 請 問 會 要 怎麼 什麼 如何 "
    "嗎 可以 是否 一個 我們 他們 公司 老闆".split()
)
_TOKEN_OK = re.compile(r"[\w]", re.UNICODE)

# Colloquial term → statutory wording, so user phrasing hits the law text.
_SYNONYMS = {
    "特休": ["特別休假"], "未休特休": ["特別休假"], "年假": ["特別休假"],
    "加班": ["延長工作時間"], "加班費": ["延長工作時間", "工資"],
    "延長工時": ["延長工作時間"], "工時": ["工作時間"],
    "職災": ["職業災害"], "工傷": ["職業災害"],
    "解雇": ["解僱"], "開除": ["解僱", "終止契約"], "被裁員": ["資遣"], "裁員": ["資遣"],
    "欠薪": ["積欠工資"], "薪水": ["工資"], "薪資": ["工資"], "月薪": ["工資"],
    "國假": ["國定假日"], "產檢假": ["陪產檢假"], "勞退": ["勞工退休金"],
    "試用": ["試用期"], "例假日": ["例假"],
    # statute uses older wording
    "女性": ["女工"], "女生": ["女工"], "夜間工作": ["午後十時"], "夜班": ["午後十時"],
}

_LAW_NAMES = ["勞動基準法施行細則", "勞動基準法", "性別平等工作法"]


def query_articles(query: str) -> tuple[set[str], str | None]:
    """Article tokens in the query and the law it names (longest match)."""
    rest, arts = normalize(query)
    law = next((n for n in _LAW_NAMES if n in rest), None)
    return set(arts), law
_initialised = False


def _init_jieba() -> None:
    global _initialised
    if _initialised:
        return
    jieba.setLogLevel(60)  # silence "Building prefix dict…"
    if os.path.exists(_DICT_PATH):
        jieba.load_userdict(_DICT_PATH)
    # Keywords from force rules double as domain vocabulary.
    try:
        from force_rules import FORCE_ARTICLE_RULES
        for rule in FORCE_ARTICLE_RULES:
            for kw in rule["keywords"]:
                jieba.add_word(kw, freq=20000)
    except ImportError:
        pass
    _initialised = True


def tokenize(text: str) -> list[str]:
    _init_jieba()
    rest, arts = normalize(text)
    toks = [
        w for w in jieba.lcut(rest)
        if w.strip() and _TOKEN_OK.search(w) and w not in _STOPWORDS
    ]
    expanded = [s for w in toks for s in _SYNONYMS.get(w, ())]
    return arts + toks + expanded


# ── BM25 index ───────────────────────────────────────────────────────────────
class BM25Index:
    def __init__(self, texts: list[str], metas: list[dict]):
        self.texts = texts
        self.metas = metas
        corpus = []
        for t, m in zip(texts, metas):
            # Prefix law name + article no. so every chunk of an article (not
            # just the first, which carries the 【…】 header) matches it.
            prefix = f"{m.get('source', '')} {m.get('article_no', '')} {m.get('title', '')} "
            corpus.append(tokenize(prefix + t))
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_n: int = 30) -> list[int]:
        """Return chunk indices ranked by BM25 (zero-score hits dropped)."""
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [i for i in order if scores[i] > 0]


# ── Reciprocal Rank Fusion ───────────────────────────────────────────────────
def doc_key(meta: dict, fallback: str = "") -> str:
    # source is included because the law index mixes 勞基法 and 施行細則,
    # whose parent_id ("law_第 1 條") collides.
    return f"{meta.get('source', '')}|{meta.get('parent_id') or fallback}"


def rrf_merge(
    vec_results: list[tuple],
    bm25_idx: list[int],
    texts: list[str],
    metas: list[dict],
    top_k: int = 5,
    k: int = RRF_K,
    query: str | None = None,
) -> list[tuple]:
    """Fuse vector results [(text, meta, score)] and BM25 chunk indices.

    Both lists are collapsed to document level (first hit per doc_key = its
    best rank) before scoring: score = Σ 1 / (k + rank). If *query* cites an
    article number, docs with that exact article_no get a third, rank-1
    vote (two votes when the cited law also matches) — an explicit citation
    is the strongest signal we have. Returns [(text, meta, rrf_score)] in
    the same shape as the vector searchers.
    """
    fused: dict[str, float] = {}
    payload: dict[str, tuple] = {}

    if query:
        arts, law = query_articles(query)
        if arts:
            for i, m in enumerate(metas):
                tok = _art_token(*(re.findall(r"\d+", m.get("article_no", "")) + [None])[:2]) \
                    if m.get("article_no") else None
                if tok not in arts:
                    continue
                if law and m.get("source") != law:
                    continue
                key = doc_key(m, str(i))
                if key in payload:
                    continue
                fused[key] = (2 if law else 1) / (k + 1)
                payload[key] = (texts[i], m)

    seen = set()
    rank = 0
    for text, meta, _ in vec_results:
        key = doc_key(meta)
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
        payload[key] = (text, meta)

    seen = set()
    rank = 0
    for i in bm25_idx:
        key = doc_key(metas[i], str(i))
        if key in seen:
            continue
        seen.add(key)
        rank += 1
        fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
        payload.setdefault(key, (texts[i], metas[i]))  # prefer vector's chunk

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(payload[key][0], payload[key][1], score) for key, score in ranked]
