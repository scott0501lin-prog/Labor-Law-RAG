"""
Hold the test set's own answers out of case retrieval during evaluation.

The 台北市勞動局 FAQ is both a case source in the index and the eval test set
(eval/taipei_faq_testset.json). Asked one of those questions, the app would
retrieve that FAQ's official answer as case #1, so end-to-end answers would
be graded with the answer key in the prompt.

Set EVAL_HOLDOUT=1 (or a path to another test-set JSON) and those entries are
dropped from case retrieval. Off by default: the live app keeps the full
FAQ, which is a legitimate source for real users. Law retrieval is not
affected (law Hit@5 never used the case index).
"""
from __future__ import annotations

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT = os.path.join(_ROOT, "eval", "taipei_faq_testset.json")
HOLDOUT_SOURCE = "台北市勞動局"


def _holdout_path() -> str | None:
    v = os.getenv("EVAL_HOLDOUT", "").strip()
    if not v or v.lower() in ("0", "false", "no"):
        return None
    return _DEFAULT if v.lower() in ("1", "true", "yes") else v


def load_holdout_titles(path: str | None = None) -> set[str]:
    path = path or _holdout_path()
    if not path:
        return set()
    with open(path, encoding="utf-8") as f:
        return {x["question"].strip() for x in json.load(f)}


def case_filter(titles: set[str] | None = None):
    """allow(meta) -> bool for case retrieval, or None when holdout is off.

    Index titles of FAQ entries are the question text (all 93 test questions
    match a title exactly).
    """
    titles = load_holdout_titles() if titles is None else titles
    if not titles:
        return None
    return lambda m: not (m.get("source") == HOLDOUT_SOURCE
                          and (m.get("title") or "").strip() in titles)
