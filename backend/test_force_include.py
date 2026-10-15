"""
驗證 FORCE_ARTICLE_RULES 的關鍵字觸發與強制帶入條號。
不需要啟動 embedding model 或 Streamlit，純字串比對。

執行：
    python backend/test_force_include.py
"""

import io
import json
import os
import sys

# 強制 UTF-8 輸出，避免 Windows CP950 UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 規則定義（與 Fronted/app.py 同步）────────────────────────────────────
FORCE_ARTICLE_RULES = [
    {"name": "補休換休",
     "keywords": ["補休", "換休", "以休代薪", "代替休假"],
     "articles": {"第 32-1 條"}, "mutex": "加班補休"},
    {"name": "加班延長工時",
     "keywords": ["加班", "延長工作時間", "延長工時", "休息日工作", "例假工作", "假日加班"],
     "articles": {"第 24 條", "第 32 條", "第 36 條"}, "mutex": "加班補休"},
    {"name": "資遣解僱",
     "keywords": ["資遣", "預告", "解僱", "解雇", "終止契約", "預告期間", "預告工資",
                  "合法解僱", "解僱事由", "終止勞動契約"],
     "articles": {"第 11 條", "第 12 條", "第 13 條", "第 14 條", "第 16 條",
                  "第 17 條", "第 18 條", "第 19 條"}, "mutex": None},
    {"name": "特別休假",
     "keywords": ["特別休假", "年假", "休假天數", "特休"],
     "articles": {"第 38 條"}, "mutex": None},
    {"name": "假日出勤",
     "keywords": ["國定假日", "例假", "休息日", "國假", "補休", "假日出勤", "假日上班", "輪班假日"],
     "articles": {"第 36 條", "第 37 條", "第 39 條", "第 40 條"}, "mutex": None},
    {"name": "欠薪工資",
     "keywords": ["欠薪", "積欠工資", "扣薪", "預扣", "基本工資", "工資清冊", "發薪"],
     "articles": {"第 21 條", "第 22 條", "第 23 條", "第 26 條", "第 27 條", "第 28 條"},
     "mutex": None},
    {"name": "工時休息",
     "keywords": ["工時", "工作時間", "輪班", "換班", "休息時間", "彈性工時", "變形工時"],
     "articles": {"第 30 條", "第 30-1 條", "第 32 條", "第 34 條", "第 35 條"}, "mutex": None},
    {"name": "請假",
     "keywords": ["婚假", "喪假", "病假", "事假"],
     "articles": {"第 43 條"}, "mutex": None},
    {"name": "職災",
     "keywords": ["職災", "職業災害", "工傷", "職業病"],
     "articles": {"第 59 條", "第 60 條", "第 61 條", "第 62 條"}, "mutex": None},
    {"name": "退休",
     "keywords": ["退休", "退休金", "強制退休"],
     "articles": {"第 53 條", "第 54 條", "第 55 條", "第 56 條", "第 58 條"}, "mutex": None},
    {"name": "契約調動",
     "keywords": ["定期契約", "不定期契約", "競業禁止", "調動", "調職", "調到", "被調", "最低服務年限"],
     "articles": {"第 9 條", "第 9-1 條", "第 10-1 條", "第 15-1 條"}, "mutex": None},
    {"name": "母性保護",
     "keywords": ["產假", "懷孕", "哺乳", "安胎", "妊娠"],
     "articles": {"第 13 條", "第 50 條", "第 51 條", "第 52 條"}, "mutex": None},
    {"name": "童工女工",
     "keywords": ["童工", "未滿十八歲", "未成年", "夜間工作"],
     "articles": {"第 44 條", "第 45 條", "第 46 條", "第 47 條", "第 48 條", "第 49 條"},
     "mutex": None},
    {"name": "派遣",
     "keywords": ["派遣", "要派單位"],
     "articles": {"第 17-1 條", "第 22-1 條", "第 63-1 條"}, "mutex": None},
    {"name": "申訴罰則",
     "keywords": ["申訴", "檢舉", "報復", "罰鍰"],
     "articles": {"第 74 條", "第 79 條"}, "mutex": None},
]

MAX_FORCED_LAW_RESULTS = 15

# ── 讀取資料庫中實際存在的條號 ──────────────────────────────────────────
docs_path = os.path.join(os.path.dirname(__file__), "embeddings", "law_docs.json")
with open(docs_path, encoding="utf-8") as f:
    law_docs = json.load(f)
# law_docs.json 格式：{"texts": [...], "metas": [...]}
metas_list = law_docs.get("metas", law_docs) if isinstance(law_docs, dict) else law_docs
valid_articles = {m.get("article_no") for m in metas_list if m.get("source") == "勞動基準法"}

# ── 驗證每個規則的條號確實在資料庫裡 ─────────────────────────────────────
print("=" * 60)
print("【條號存在性驗證】")
all_ok = True
for rule in FORCE_ARTICLE_RULES:
    missing = rule["articles"] - valid_articles
    if missing:
        print(f"  ✗ {rule['name']}: 條號不在 law_docs.json → {missing}")
        all_ok = False
if all_ok:
    print("  ✓ 全部條號均存在於 law_docs.json")

# ── 模擬 force-include 邏輯 ──────────────────────────────────────────────
def simulate_force_include(query: str):
    triggered = {}
    fired_mutex: set = set()
    forced_articles: list = []
    seen: set = set()

    for rule in FORCE_ARTICLE_RULES:
        if rule["mutex"] and rule["mutex"] in fired_mutex:
            continue
        if any(k in query for k in rule["keywords"]):
            new = sorted(rule["articles"] - seen)
            triggered[rule["name"]] = sorted(rule["articles"])
            forced_articles.extend(new)
            seen.update(new)
            if rule["mutex"]:
                fired_mutex.add(rule["mutex"])

    capped = len(forced_articles) > MAX_FORCED_LAW_RESULTS
    return triggered, forced_articles[:MAX_FORCED_LAW_RESULTS], capped


# ── 測試問句 ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    # 既有組
    ("補休換休",   "公司叫我加班可以換補休嗎"),
    ("補休換休",   "補休到期沒用完算薪水嗎"),
    ("加班延長工時", "休息日加班費怎麼計算"),
    ("加班延長工時", "延長工時每月上限是幾小時"),
    ("資遣解僱",   "公司以虧損為由資遣我需要給多少資遣費"),
    ("資遣解僱",   "被解僱後雇主要給預告工資嗎"),
    ("特別休假",   "年資兩年可以請幾天特休"),
    ("假日出勤",   "國定假日上班工資要怎麼算"),
    # 新增組
    ("欠薪工資",   "老闆積欠工資三個月怎麼辦"),
    ("欠薪工資",   "雇主預扣薪水合法嗎"),
    ("工時休息",   "公司要求彈性工時我可以拒絕嗎"),
    ("工時休息",   "輪班換班之間的休息時間規定"),
    ("請假",       "父母過世可以請幾天喪假"),
    ("請假",       "員工病假期間工資怎麼算"),
    ("職災",       "上班途中發生車禍算職業災害嗎"),
    ("職災",       "職業病的醫療費用由誰負擔"),
    ("退休",       "我工作15年可以自請退休嗎"),
    ("退休",       "退休金舊制的計算方式"),
    ("契約調動",   "公司把我從台北調到高雄合法嗎"),
    ("契約調動",   "簽了競業禁止條款還可以跳槽嗎"),
    ("母性保護",   "懷孕員工被解僱有什麼保護"),
    ("母性保護",   "產假多少天工資怎麼發"),
    ("童工女工",   "未滿十八歲可以在夜間工作嗎"),
    ("童工女工",   "童工每天最多工作幾小時"),
    ("派遣",       "派遣公司積欠工資可以向要派單位請求嗎"),
    ("派遣",       "派遣勞工發生職業災害由誰賠"),
    ("申訴罰則",   "向勞工局申訴後被公司報復怎麼辦"),
    ("申訴罰則",   "違反工資規定的罰鍰是多少"),
    # 跨組測試
    ("跨組-懷孕資遣", "懷孕後被公司以業務緊縮為由資遣"),
    ("跨組-輪班工時", "輪班工作時間和換班休息怎麼算"),
]

print()
print("=" * 60)
print("【Force-Include 觸發驗證】")
pass_count = fail_count = 0
for expected_group, query in TEST_QUERIES:
    triggered, forced, capped = simulate_force_include(query)
    group_names = ", ".join(triggered.keys()) if triggered else "（無觸發）"
    articles_str = ", ".join(forced) if forced else "（無）"
    cap_note = f"  ⚠ 超過 {MAX_FORCED_LAW_RESULTS} 上限，已裁切" if capped else ""

    # 跨組測試只檢查有觸發，不檢查特定組名
    ok = bool(triggered) if expected_group.startswith("跨組") else (expected_group in triggered)
    status = "✓" if ok else "✗"
    if ok:
        pass_count += 1
    else:
        fail_count += 1

    print(f"\n  {status} [{expected_group}] {query}")
    print(f"       觸發組: {group_names}")
    print(f"       強制帶入({len(forced)}條): {articles_str}{cap_note}")

print()
print("=" * 60)
print(f"結果：{pass_count} 通過 / {fail_count} 失敗 / 共 {len(TEST_QUERIES)} 句")
