"""
Intent → retrieval plan.

Each intent decides which corpora to search, how many results, whether case
excerpts go into the prompt, and which answer format the model must use.
Pure Python (no Streamlit) so eval scripts can import it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LAB = "勞動基準法"
RUL = "勞動基準法施行細則"
GEN = "性別平等工作法"
ALL_LAWS = [LAB, RUL, GEN]

INTENTS = ("solve_problem", "find_cases", "law_lookup", "out_of_scope")


@dataclass(frozen=True)
class RetrievalPlan:
    intent: str
    mode: str               # answer format key: consult / cases / law
    law_k: int              # law results kept
    case_k: int             # case results retrieved (shown via button)
    cases_in_prompt: int    # how many case excerpts go into the prompt
    laws: tuple[str, ...]   # allowed law sources, priority order
    force_rules: bool       # apply 勞基法 keyword force-include rules


def make_plan(intent: str, laws: list[str] | None) -> RetrievalPlan | None:
    """None means out of scope: skip retrieval and generation entirely."""
    laws_t = tuple(l for l in (laws or []) if l in ALL_LAWS) or (LAB, RUL)
    # 勞基法 keyword rules (e.g. 申訴 → 第 74/79 條) are noise when the
    # question is primarily about 性平法.
    force = laws_t[0] != GEN
    if intent == "out_of_scope":
        return None
    if intent == "find_cases":
        return RetrievalPlan(intent, "cases", law_k=3, case_k=5, cases_in_prompt=5,
                             laws=laws_t, force_rules=False)
    if intent == "law_lookup":
        return RetrievalPlan(intent, "law", law_k=5, case_k=5, cases_in_prompt=0,
                             laws=laws_t, force_rules=force)
    return RetrievalPlan("solve_problem", "consult", law_k=5, case_k=5, cases_in_prompt=2,
                         laws=laws_t, force_rules=force)


# ── Keyword fallback (intent detection off, or the LLM call failed) ─────────
_GENDER_KW = ["性騷擾", "性別", "性平", "陪產", "育嬰", "生理假", "家庭照顧假",
              "哺乳", "托兒", "產檢", "安胎", "懷孕歧視", "婚育"]
_CASE_KW = ["案例", "判決", "判例", "案件", "實例", "函釋", "解釋令", "裁罰", "處分", "法院"]
_ART_RE = re.compile(r"第\s*[0-9一二三四五六七八九十百]+\s*條|\d+\s*條")
_LOOKUP_KW = ["條文", "規定是什麼", "怎麼規定", "內容", "全文"]


def detect_laws(query: str) -> list[str]:
    if any(k in query for k in _GENDER_KW):
        return [GEN, LAB]
    if "施行細則" in query or "細則" in query:
        return [RUL, LAB]
    return [LAB, RUL]


def heuristic_intent(query: str) -> str:
    if any(k in query for k in _CASE_KW):
        return "find_cases"
    if _ART_RE.search(query) and (len(query) <= 20 or any(k in query for k in _LOOKUP_KW)):
        return "law_lookup"
    return "solve_problem"
