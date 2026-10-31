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
    """回傳 (scope, body)。解析失敗時一律視為 IN，並盡量只回傳可讀內容，避免誤殺。"""
    m = _SCOPE.search(text)
    scope = m.group(1).upper() if m else "IN"
    a = _ANSWER.search(text)
    if a:
        body = a.group(1).strip()
    else:
        body = _SCOPE.sub("", text)
        body = re.sub(r"</?answer>", "", body, flags=re.I).strip()
    return scope, body


def finalize_reply(raw_text: str) -> str:
    """app.py 在取得 Gemini 回應後呼叫：OUT 換成固定轉介文字，IN 回傳乾淨的回答內容。"""
    scope, body = parse_reply(raw_text)
    return OUT_OF_SCOPE_TEXT if scope == "OUT" else body


# 串流版：等 <scope> 出現後才開始輸出，超過這個長度還沒看到就視為 IN
_SCOPE_WAIT_CHARS = 300
_OPEN, _CLOSE = "<answer>", "</answer>"


def stream_reply(pieces: Iterable[str]) -> Iterator[str]:
    """finalize_reply 的串流版：邊收邊輸出 <answer> 內文。

    OUT 時只輸出 OUT_OF_SCOPE_TEXT，但仍會把 pieces 讀完（讓上游能記錄 token 用量）。
    解析失敗一律視為 IN，與 parse_reply 相同。
    """
    it = iter(pieces)
    buf = ""
    for piece in it:
        buf += piece
        if _SCOPE.search(buf) or len(buf) > _SCOPE_WAIT_CHARS:
            break

    m = _SCOPE.search(buf)
    if m and m.group(1).upper() == "OUT":
        yield OUT_OF_SCOPE_TEXT
        for _ in it:
            pass
        return

    pending = _SCOPE.sub("", buf)
    state = "pre"  # pre → body → post
    started = False

    def _emit(text: str):
        nonlocal started
        if not started:
            text = text.lstrip()
            started = bool(text)
        return text

    def _process() -> Iterator[str]:
        nonlocal pending, state
        if state == "pre":
            idx = pending.lower().find(_OPEN)
            if idx != -1:
                pending = pending[idx + len(_OPEN):]
                state = "body"
            else:
                head = pending.lstrip()
                if head and not _OPEN.startswith(head[:len(_OPEN)].lower()):
                    state = "body"  # 沒有 <answer> 標籤：直接當內文輸出
        if state == "body":
            idx = pending.lower().find(_CLOSE)
            if idx != -1:
                out, pending, state = pending[:idx], "", "post"
            else:
                # 保留尾端可能是被切開的 </answer>
                k = pending.rfind("<")
                if k != -1 and _CLOSE.startswith(pending[k:].lower()):
                    out, pending = pending[:k], pending[k:]
                else:
                    out, pending = pending, ""
            out = _emit(out)
            if out:
                yield out
        if state == "post":
            pending = ""

    yield from _process()
    for piece in it:
        pending += piece
        yield from _process()

    if pending and state != "post":
        tail = _emit(re.sub(r"</?answer>", "", pending, flags=re.I))
        if tail:
            yield tail
