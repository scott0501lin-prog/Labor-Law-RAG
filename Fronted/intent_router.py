"""
Intent → retrieval plan.

Each intent decides how many laws / cases are retrieved, whether case
excerpts go into the prompt, and which answer format the model must use.
Pure Python (no Streamlit) so eval scripts can import it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LAB = "勞動基準法"
RUL = "勞動基準法施行細則"
PEN = "勞工退休金條例"
OCC = "勞工職業災害保險及保護法"
MIN = "最低工資法"
LEAVE = "勞工請假規則"
ALL_LAWS = [LAB, RUL, PEN, OCC, MIN, LEAVE]

INTENTS = ("solve_problem", "find_cases", "law_lookup", "out_of_scope")


@dataclass(frozen=True)
class RetrievalPlan:
    intent: str
    mode: str               # answer format key: consult / cases / law
    law_k: int              # law results kept
    case_k: int             # case results retrieved (shown via button)
    cases_in_prompt: int    # how many case excerpts go into the prompt
    laws: tuple[str, ...]   # laws the question is about, priority order (logged)
    force_rules: bool       # apply 勞基法 keyword force-include rules


def make_plan(intent: str, laws: list[str] | None) -> RetrievalPlan | None:
    """None means out of scope: skip retrieval and generation entirely.

    All laws in the corpus are labor law and often apply together (請假 →
    勞基法 §43 + 勞工請假規則), so retrieval is not restricted by *laws*; it
    only decides whether the 勞基法 force-include rules apply.
    """
    if intent == "out_of_scope":
        return None
    laws_t = tuple(l for l in (laws or []) if l in ALL_LAWS) or (LAB, RUL)
    force = LAB in laws_t
    if intent == "find_cases":
        return RetrievalPlan(intent, "cases", law_k=3, case_k=5, cases_in_prompt=5,
                             laws=laws_t, force_rules=False)
    if intent == "law_lookup":
        return RetrievalPlan(intent, "law", law_k=5, case_k=5, cases_in_prompt=0,
                             laws=laws_t, force_rules=force)
    return RetrievalPlan("solve_problem", "consult", law_k=5, case_k=5, cases_in_prompt=2,
                         laws=laws_t, force_rules=force)


# ── Keyword fallback (intent detection off, or the LLM call failed) ─────────
_LAW_KW = [
    (PEN, ["勞退", "退休金條例", "提繳", "個人專戶", "新制退休"]),
    (OCC, ["災保", "職業災害保險", "職災保險", "失能給付", "職災給付"]),
    (MIN, ["最低工資"]),
    (LEAVE, ["請假", "病假", "事假", "婚假", "喪假", "公假"]),
    (RUL, ["施行細則", "細則"]),
]
_CASE_KW = ["案例", "判決", "判例", "案件", "實例", "函釋", "解釋令", "裁罰", "處分", "法院"]
_ART_RE = re.compile(r"第\s*[0-9一二三四五六七八九十百]+\s*條|\d+\s*條")
_LOOKUP_KW = ["條文", "規定是什麼", "怎麼規定", "內容", "全文"]


def detect_laws(query: str) -> list[str]:
    hits = [law for law, kws in _LAW_KW if any(k in query for k in kws)]
    return hits + [l for l in (LAB, RUL) if l not in hits]


def heuristic_intent(query: str) -> str:
    if any(k in query for k in _CASE_KW):
        return "find_cases"
    if _ART_RE.search(query) and (len(query) <= 20 or any(k in query for k in _LOOKUP_KW)):
        return "law_lookup"
    return "solve_problem"
