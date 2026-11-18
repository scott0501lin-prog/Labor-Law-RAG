"""
Configurable parameters and prompts for intent detection + follow-up system.
Edit this file to tune triggers, required fields, and Gemini prompts
without touching app.py.
"""
import os

# ── Feature toggle ────────────────────────────────────────────────────────────
# On by default: the intent decides which corpora are searched and which
# answer format is used (see intent_router.py). The same call also does the
# translation, so non-Chinese questions need no extra round-trip.
# Set INTENT_DETECTION=false to fall back to keyword heuristics.
ENABLE_INTENT_DETECTION = os.getenv("INTENT_DETECTION", "true").lower() == "true"

# ── Follow-up limits ──────────────────────────────────────────────────────────
MAX_FOLLOWUP_ROUNDS = 2

# Skip condition: reply length <= SKIP_MAX_LEN AND contains a trigger keyword.
# Replies longer than SKIP_MAX_LEN are never treated as a skip request even if
# they happen to contain a trigger word (e.g. "我不知道休息日加班費怎麼算").
SKIP_FOLLOWUP_TRIGGERS = ["不知道", "不清楚", "直接回答", "算了", "隨便", "先不說", "跳過", "skip"]
SKIP_MAX_LEN = 10

# ── Required fields for overtime_pay scenario ─────────────────────────────────
# Missing any of these triggers a follow-up question.
# compensation_type is OPTIONAL — never prompt for it.
OVERTIME_PAY_REQUIRED_FIELDS = ["day_type", "hours", "wage_amount"]

# ── Human-readable labels for missing-field disclaimer ────────────────────────
MISSING_FIELDS_LABELS = {
    "day_type":          "加班日類型（平日／休息日／例假日／國定假日）",
    "hours":             "加班時數",
    "wage_type":         "薪資類型（月薪／時薪）",
    "wage_amount":       "薪資金額",
    "compensation_type": "給付方式（加班費／補休）",
}

# ── Prompts ───────────────────────────────────────────────────────────────────
# Placeholders (filled at runtime):
#   {user_message}     — the raw user input
#   {known_info_json}  — JSON of fields collected so far (empty dict on first call)
#   {ui_lang}          — interface language (e.g. "繁體中文", "English")

INTENT_ANALYSIS_PROMPT = """\
你是一個勞動法諮詢助理的前置分析模組。分析使用者訊息並以 JSON 格式回傳結果。
除 JSON 本身外，請勿輸出任何其他文字（包括 ```json 標記、說明文字等）。

## 目前已知資訊（空 {{}} 表示首次提問）
{known_info_json}

## 使用者訊息
{user_message}

## 介面語言（follow_up 欄位須使用此語言）
{ui_lang}

## 回傳格式（嚴格 JSON）
{{
  "translated_query": "若原文非繁體中文請翻譯，否則原樣輸出",
  "intent": "solve_problem 或 find_cases 或 law_lookup 或 out_of_scope",
  "laws": ["相關法規，依關聯度排序，只能從：勞動基準法、勞動基準法施行細則、勞工退休金條例、勞工職業災害保險及保護法、最低工資法、勞工請假規則"],
  "scenario": "overtime_pay 或 other",
  "known_info": {{
    "day_type": "平日 或 休息日 或 例假日 或 國定假日 或 null",
    "hours": 數字或 null,
    "wage_type": "月薪 或 時薪 或 null",
    "wage_amount": 數字或 null,
    "compensation_type": "加班費 或 補休 或 null"
  }},
  "missing_fields": ["缺少的必填欄位名稱（僅在 overtime_pay + solve_problem 時填寫）"],
  "follow_up": "追問句（使用 {ui_lang}，若無需追問則為空字串）"
}}

## 說明
- intent 定義：
  - solve_problem：描述自己的處境，想知道權益或怎麼處理（勞資爭議諮詢）
  - find_cases：想看案例、判決、函釋、問答或實際處理情形（例：「資遣的判決」「加班費的函釋」）
  - law_lookup：想知道某條法條或某項制度的規定內容（例：「勞基法第 30 條」「特休怎麼規定」）
  - out_of_scope：與勞動、職場無關，或明確只涉及資料庫以外的法規：性別平等工作法事項（性騷擾、性別歧視、
    育嬰留職停薪、陪產檢及陪產假、生理假）、勞工保險條例、就業服務法、民法、刑法、稅法、公司法
  - 不確定時選 solve_problem，不要選 out_of_scope
- laws：退休金提繳與個人專戶把「勞工退休金條例」排第一；職災保險給付把「勞工職業災害保險及保護法」排第一；
  最低工資把「最低工資法」排第一；婚喪病事公假把「勞工請假規則」排第一；其餘勞動條件把「勞動基準法」排第一；
  問施行細則才把施行細則排第一。out_of_scope 時給空陣列
- scenario overtime_pay：問題涉及加班費計算或補休換算
- known_info：從訊息中提取已知值，未知設 null
- missing_fields 必填欄位：day_type、hours、wage_amount（compensation_type 不追問）
- follow_up：一次最多問兩個問題，語氣親切；資訊已足夠或非 overtime_pay+solve_problem 則輸出空字串
"""

OVERTIME_EXTRACTION_PROMPT = """\
你是一個欄位提取模組。使用者剛剛回覆了一個關於加班費的追問。
請從使用者的回覆中提取欄位值，並判斷這是否是一個全新的問題（與加班費無關）。
除 JSON 本身外，請勿輸出任何其他文字（包括 ```json 標記、說明文字等）。

## 目前已知資訊
{known_info_json}

## 使用者回覆
{user_message}

## 介面語言（follow_up 欄位須使用此語言）
{ui_lang}

## 回傳格式（嚴格 JSON）
{{
  "is_new_question": true 或 false,
  "known_info": {{
    "day_type": "平日 或 休息日 或 例假日 或 國定假日 或 null",
    "hours": 數字或 null,
    "wage_type": "月薪 或 時薪 或 null",
    "wage_amount": 數字或 null,
    "compensation_type": "加班費 或 補休 或 null"
  }},
  "missing_fields": ["缺少的必填欄位（day_type、hours、wage_amount）"],
  "follow_up": "追問句（使用 {ui_lang}，若無需追問則為空字串）"
}}

## 說明
- is_new_question：若回覆明顯是一個全新問題（與加班費計算無關）設為 true，否則 false
- known_info：保留已知資訊，並從此次回覆補充新的欄位值（null 表示仍未知）
- missing_fields：必填欄位 day_type, hours, wage_amount 缺任一即列入
- follow_up：一次最多問兩個問題，語氣親切；資訊已足夠則輸出空字串
"""
