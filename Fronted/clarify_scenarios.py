"""
Clarifying-question catalogue (team document 「情境」, 2026-11).

The intent call only picks a scenario and the question ids whose answers are
missing and would change the answer; the wording and options come from here,
so the same situation is always asked the same way. Options are fixed where
the answer is a choice; free-form answers (amounts, dates, descriptions) get
an empty list and are shown as text inputs.
Pure Python (no Streamlit).
"""
from __future__ import annotations

YES_NO = ["有", "沒有"]

# key: (name, example question, [(id, question, options), ...])
SCENARIOS: dict[str, tuple[str, str, list[tuple[str, str, list[str]]]]] = {
    "wage": ("工資與薪資問題", "公司沒有發薪水是否合法？", [
        ("w1", "請問公司目前積欠您多少薪資？", []),
        ("w2", "公司原本約定每月幾號發放薪水？", []),
        ("w3", "公司目前已經延遲多久尚未發薪？", ["未滿 1 個月", "1 到 3 個月", "超過 3 個月"]),
        ("w4", "您是正職、兼職、工讀生還是約聘人員？", ["正職", "兼職", "工讀生", "約聘人員"]),
        ("w5", "公司是否有告知延遲發薪的原因？", ["有告知", "沒有告知"]),
    ]),
    "hours": ("工時與加班問題", "公司要求加班是否合法？", [
        ("h1", "您每天正常工作時間是幾點到幾點？", []),
        ("h2", "您平均每天加班幾小時？", []),
        ("h3", "您每週實際工作幾天？", ["5 天", "6 天", "7 天"]),
        ("h4", "公司是否有事先要求或安排加班？", ["公司要求或安排", "沒有，是自己留下", "不確定"]),
        ("h5", "公司是否有提供加班費或補休？", ["有加班費", "有補休", "都沒有"]),
        ("h6", "公司是否有記錄您的實際出勤時間？", ["有打卡或紀錄", "沒有紀錄", "不確定"]),
    ]),
    "overtime_pay": ("加班費問題", "公司沒有給加班費是否合法？", [
        ("o1", "您的每月薪資是多少？", []),
        ("o2", "您是在平日、休息日還是國定假日加班？", ["平日", "休息日", "國定假日", "例假日"]),
        ("o3", "您總共加班幾小時？", []),
        ("o4", "公司是否有計算加班時數？", ["有", "沒有", "不確定"]),
        ("o5", "公司是否曾以責任制為理由拒絕支付加班費？", YES_NO),
    ]),
    "rest_day": ("休假與例假問題", "公司不讓我休假是否合法？", [
        ("r1", "您目前一週休息幾天？", ["2 天", "1 天", "不到 1 天"]),
        ("r2", "公司是否要求您連續多日工作？", ["是", "否"]),
        ("r3", "公司要求您在哪一天出勤？", []),
        ("r4", "當天屬於例假、休息日還是國定假日？", ["例假", "休息日", "國定假日", "不確定"]),
        ("r5", "公司是否有提供加班費或補休？", ["有加班費", "有補休", "都沒有"]),
    ]),
    "annual_leave": ("特別休假問題", "公司不讓我請特休是否合法？", [
        ("a1", "您在公司工作多久了？", []),
        ("a2", "您目前還剩下幾天特別休假？", []),
        ("a3", "您是否已經向主管提出特休申請？", ["已經申請", "還沒申請"]),
        ("a4", "公司拒絕您請假的理由是什麼？", []),
        ("a5", "您是否已經離職？如果已離職，離職日期是哪一天？", []),
    ]),
    "dismissal": ("資遣與解僱問題", "公司可以無故開除我嗎？", [
        ("d1", "公司是否已經正式通知您解僱或資遣？", ["已經書面通知", "口頭告知", "還沒有"]),
        ("d2", "公司通知解僱的理由是什麼？", []),
        ("d3", "您在公司工作多久了？", []),
        ("d4", "公司是否有提前通知您？", YES_NO),
        ("d5", "公司是否有支付資遣費？", ["有", "沒有", "還不知道"]),
        ("d6", "公司是否要求您簽署自願離職書？", YES_NO),
    ]),
    "contract": ("勞動契約問題", "公司可以任意更改我的工作內容嗎？", [
        ("c1", "您原本的工作職務是什麼？", []),
        ("c2", "公司希望將您的工作調整成什麼內容？", []),
        ("c3", "公司是否有告知調動原因？", YES_NO),
        ("c4", "工作地點是否有所變更？", ["有變更", "沒有變更"]),
        ("c5", "調動後的薪資及工作條件是否受到影響？", ["沒有影響", "變差"]),
    ]),
    "injury": ("職業災害問題", "工作受傷，公司需要負責嗎？", [
        ("i1", "您是在工作中還是上下班途中受傷？", ["工作中", "上下班途中"]),
        ("i2", "受傷的原因及經過是什麼？", []),
        ("i3", "是否有就醫及取得診斷證明？", ["有", "還沒有"]),
        ("i4", "公司是否知道您發生職業災害？", ["知道", "不知道"]),
        ("i5", "您目前是否因傷勢而無法工作？", ["無法工作", "可以工作"]),
        ("i6", "公司是否有提供相關補償？", ["有", "沒有", "還沒討論"]),
    ]),
    "pension": ("勞工退休金問題", "公司沒有提繳 6% 退休金是否合法？", [
        ("p1", "您目前任職的公司是哪一類型的事業單位？", []),
        ("p2", "您在公司任職多久了？", []),
        ("p3", "您是否有查詢過勞退個人專戶？", ["查過", "沒查過"]),
        ("p4", "公司是否曾告知退休金提繳方式？", YES_NO),
        ("p5", "您是否知道公司每月是否有提繳退休金？", ["有提繳", "沒有提繳", "不知道"]),
    ]),
    "penalty": ("違法裁罰與申訴問題", "公司違反勞基法會被罰多少錢？", [
        ("v1", "公司具體違反哪一項勞動法令？", []),
        ("v2", "發生違規的時間是什麼時候？", []),
        ("v3", "公司是否曾經發生相同違規情形？", ["有", "沒有", "不知道"]),
        ("v4", "是否有相關證據，例如薪資單、出勤紀錄或勞動契約？", YES_NO),
        ("v5", "您是否已經向地方勞工主管機關提出申訴？", ["已經申訴", "還沒有"]),
    ]),
}

# 「建議加入的通用反問問題」: usable in any scenario, and the only ones for "other".
GENERIC: list[tuple[str, str, list[str]]] = [
    ("g1", "請問事情發生在什麼時候？", []),
    ("g2", "您在這家公司工作多久了？", []),
    ("g3", "您目前是正職、兼職還是工讀生？", ["正職", "兼職", "工讀生"]),
    ("g4", "公司具體做了什麼事情？", []),
    ("g5", "您是否有相關紀錄或證明？", YES_NO),
    ("g6", "您是否曾向公司反映？公司如何回覆？", []),
    ("g7", "您希望了解法律規定、檢舉方式，還是賠償權益？", ["法律規定", "檢舉方式", "賠償權益"]),
]

_BY_ID = {qid: (q, opts) for _, _, qs in SCENARIOS.values() for qid, q, opts in qs}
_BY_ID.update({qid: (q, opts) for qid, q, opts in GENERIC})


def catalog_text() -> str:
    """The catalogue as it appears in the intent prompt."""
    lines = []
    for key, (name, example, qs) in SCENARIOS.items():
        lines.append(f"[{key}] {name}（例：「{example}」）")
        lines += [f"  {qid} {q}" for qid, q, _ in qs]
    lines.append("[通用，任何情境都可用；other 情境只能用這些]")
    lines += [f"  {qid} {q}" for qid, q, _ in GENERIC]
    return "\n".join(lines)


def build_questions(ids, limit: int = 3) -> list[dict]:
    """Question ids from the model → [{"id", "q", "options"}] in the given
    order; unknown ids and duplicates are dropped."""
    out, seen = [], set()
    for qid in ids if isinstance(ids, list) else []:
        qid = str(qid).strip()
        if qid in _BY_ID and qid not in seen:
            seen.add(qid)
            q, opts = _BY_ID[qid]
            out.append({"id": qid, "q": q, "options": list(opts)})
    return out[:limit]
