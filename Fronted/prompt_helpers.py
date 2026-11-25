"""
與 system_prompt.txt / user_prompt.txt 搭配的程式端輔助。
放在 Fronted/，app.py 匯入即可。
"""
import os
import re
from typing import Iterable, Iterator

_DIR = os.path.dirname(os.path.abspath(__file__))


def _read(name: str) -> str:
    with open(os.path.join(_DIR, name), encoding="utf-8") as f:
        return f.read()


SYSTEM_TEMPLATE = _read("system_prompt.txt")
USER_TEMPLATE = _read("user_prompt.txt")

ROLE_DESC = {
    "Employee": "你的服務對象是勞工，協助他們了解並保護自身的勞動權益。語氣友善、清晰、易懂。",
    "Employer": "你的服務對象是雇主與人資，協助他們在合法合規的前提下管理勞資關係。語氣專業、嚴謹，並在「結論與做法」中標示法律風險。",
}

# 超出範圍時給使用者看的固定文字（不由模型生成）
OUT_OF_SCOPE_TEXT = (
    "此問題不在本系統目前的資料範圍內（勞動基準法與施行細則）。\n\n"
    "建議洽詢：\n"
    "- 律師\n"
    "- 法律扶助基金會\n"
    "- 1955 勞工諮詢專線"
)


def build_system_prompt(user_role: str) -> str:
    return SYSTEM_TEMPLATE.format(role_desc=ROLE_DESC.get(user_role, ROLE_DESC["Employee"]))


def build_lang_rule(ui_lang: str, lang_name: str) -> str:
    if ui_lang == "繁體中文":
        return ""
    return (
        f"\n回答語言：<answer> 內的文字請使用「{lang_name}」撰寫，小節標題一併翻譯；"
        f"法條重點保留繁體中文原有用語，並在其後附一句{lang_name}白話說明。"
    )


def build_missing_note(labels: str) -> str:
    if not labels:
        return ""
    return (
        f"\n【資料不足提示】使用者尚未提供：{labels}。"
        "請勿假設任何數值，只作一般性說明，並明說缺少哪些資訊。"
    )


def build_user_prompt(law_ctx: str, user_prompt: str, lang_rule: str = "", missing_note: str = "") -> str:
    return USER_TEMPLATE.format(
        law_ctx=law_ctx,
        user_prompt=user_prompt,
        lang_rule=lang_rule,
        missing_note=missing_note,
    )


_SCOPE = re.compile(r"<scope>\s*(IN|OUT)\s*</scope>", re.I)
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def parse_reply(text: str):
    """Strictly require a complete, single response envelope."""
    m = re.fullmatch(r"\s*<scope>\s*(IN|OUT)\s*</scope>\s*<answer>(.*?)</answer>\s*", text, re.S | re.I)
    if not m or re.search(r"</?(?:scope|answer)\b", m.group(2), re.I):
        return "OUT", ""
    return m.group(1).upper(), m.group(2).strip()


def finalize_reply(raw_text: str) -> str:
    scope, body = parse_reply(raw_text)
    return OUT_OF_SCOPE_TEXT if scope == "OUT" or not body else body


def stream_reply(pieces: Iterable[str]) -> Iterator[str]:
    # Legacy research utility only. Never stream an incomplete envelope.
    try:
        yield finalize_reply("".join(pieces))
    except Exception:
        yield OUT_OF_SCOPE_TEXT
