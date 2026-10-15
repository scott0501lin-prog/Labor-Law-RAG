"""
Force-include rules and dispatch logic.

Imported by app.py (runtime) and eval/test_force_rules.py (testing).
No Streamlit / Gemini / numpy deps — pure Python.
"""

# ── Rule table ────────────────────────────────────────────────────────────────
# Each entry: name (debug label), keywords (any match triggers), articles (set
# of article_no strings to guarantee), mutex (only first match in group fires).
FORCE_ARTICLE_RULES = [
    # mutex="加班補休": 補休優先，命中補休就不再觸發加班組
    {
        "name": "補休換休",
        "keywords": ["補休", "換休", "以休代薪", "代替休假"],
        "articles": {"第 32-1 條"},
        "mutex": "加班補休",
    },
    {
        "name": "加班延長工時",
        "keywords": ["加班", "延長工作時間", "延長工時", "休息日工作", "例假工作", "假日加班"],
        "articles": {"第 24 條", "第 32 條", "第 36 條"},
        "mutex": "加班補休",
    },
    # Independent groups (mutex=None)
    {
        "name": "資遣解僱",
        "keywords": ["資遣", "預告", "解僱", "解雇", "終止契約", "預告期間", "預告工資",
                     "合法解僱", "解僱事由", "終止勞動契約"],
        "articles": {"第 11 條", "第 12 條", "第 13 條", "第 14 條", "第 16 條", "第 17 條",
                     "第 18 條", "第 19 條"},
        "mutex": None,
    },
    {
        "name": "特別休假",
        "keywords": ["特別休假", "年假", "休假天數", "特休"],
        "articles": {"第 38 條"},
        "mutex": None,
    },
    {
        "name": "假日出勤",
        "keywords": ["國定假日", "例假", "休息日", "國假", "補休", "假日出勤", "假日上班", "輪班假日"],
        "articles": {"第 36 條", "第 37 條", "第 39 條", "第 40 條"},
        "mutex": None,
    },
    {
        "name": "欠薪工資",
        "keywords": ["欠薪", "積欠工資", "扣薪", "預扣", "基本工資", "工資清冊", "發薪"],
        "articles": {"第 21 條", "第 22 條", "第 23 條", "第 26 條", "第 27 條", "第 28 條"},
        "mutex": None,
    },
    {
        "name": "工時休息",
        "keywords": ["工時", "工作時間", "輪班", "換班", "休息時間", "彈性工時", "變形工時"],
        "articles": {"第 30 條", "第 30-1 條", "第 32 條", "第 34 條", "第 35 條"},
        "mutex": None,
    },
    {
        "name": "請假",
        "keywords": ["婚假", "喪假", "病假", "事假"],
        "articles": {"第 43 條"},
        "mutex": None,
    },
    {
        "name": "職災",
        "keywords": ["職災", "職業災害", "工傷", "職業病"],
        "articles": {"第 59 條", "第 60 條", "第 61 條", "第 62 條"},
        "mutex": None,
    },
    {
        "name": "退休",
        "keywords": ["退休", "退休金", "強制退休"],
        "articles": {"第 53 條", "第 54 條", "第 55 條", "第 56 條", "第 58 條"},
        "mutex": None,
    },
    {
        "name": "契約調動",
        "keywords": ["定期契約", "不定期契約", "競業禁止", "調動", "調職", "調到", "被調", "最低服務年限"],
        "articles": {"第 9 條", "第 9-1 條", "第 10-1 條", "第 15-1 條"},
        "mutex": None,
    },
    {
        "name": "母性保護",
        "keywords": ["產假", "懷孕", "哺乳", "安胎", "妊娠"],
        "articles": {"第 13 條", "第 50 條", "第 51 條", "第 52 條"},
        "mutex": None,
    },
    {
        "name": "童工女工",
        "keywords": ["童工", "未滿十八歲", "未成年", "夜間工作"],
        "articles": {"第 44 條", "第 45 條", "第 46 條", "第 47 條", "第 48 條", "第 49 條"},
        "mutex": None,
    },
    {
        "name": "派遣",
        "keywords": ["派遣", "要派單位"],
        "articles": {"第 17-1 條", "第 22-1 條", "第 63-1 條"},
        "mutex": None,
    },
    {
        "name": "申訴罰則",
        "keywords": ["申訴", "檢舉", "報復", "罰鍰"],
        "articles": {"第 74 條", "第 79 條"},
        "mutex": None,
    },
]

# Cap total forced law results to avoid bloating the Gemini prompt
MAX_FORCED_LAW_RESULTS = 15

# Ministerial letter forced into case results (only on rest-day overtime queries)
RESTDAY_OVERTIME_FORCE_TITLES = {"勞動部勞動條 2 字第 1050030466 號書函"}


# ── Core functions ────────────────────────────────────────────────────────────

def _force_include_articles(query, results, texts, metas, keywords, force_article_nos,
                             source="勞動基準法"):
    # The law index mixes the main act and its enforcement rules; both may share
    # the same article_no (e.g. both have their own "第 24 條"), so source must
    # be checked to avoid injecting unrelated enforcement-rule articles.
    if not any(k in query for k in keywords):
        return results
    existing = {m.get("article_no") for _, m, _ in results}
    forced = [
        (texts[i], m, None)
        for i, m in enumerate(metas)
        if m.get("article_no") in force_article_nos
        and m.get("article_no") not in existing
        and m.get("source") == source
    ]
    return forced + results


def _force_include_cases(query, results, texts, metas, keywords, force_titles):
    if not any(k in query for k in keywords) or "休息日" not in query:
        return results
    existing = {m.get("title") for _, m, _ in results}
    forced = [
        (texts[i], m, None)
        for i, m in enumerate(metas)
        if m.get("title") in force_titles and m.get("title") not in existing
    ]
    return forced + results


def run_force_include(query, law_texts, law_metas):
    """Run the full law force-include dispatch on *query*.

    Mirrors the dispatch loop in query_rag_system() (app.py).
    Starts from an empty result set so the return value contains
    only force-included articles — suitable for isolated testing.

    Returns list of (text, meta, None) tuples, capped at MAX_FORCED_LAW_RESULTS.
    """
    results = []
    fired_mutex: set = set()
    for rule in FORCE_ARTICLE_RULES:
        if rule["mutex"] and rule["mutex"] in fired_mutex:
            continue
        if any(k in query for k in rule["keywords"]):
            results = _force_include_articles(
                query, results, law_texts, law_metas,
                rule["keywords"], rule["articles"],
            )
            if rule["mutex"]:
                fired_mutex.add(rule["mutex"])
    return results[:MAX_FORCED_LAW_RESULTS]
