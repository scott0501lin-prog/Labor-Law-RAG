"""
Configurable parameters and prompts for intent detection + follow-up system.
Edit this file to tune triggers, required fields, and Gemini prompts
without touching app.py.
"""
import os

from clarify_scenarios import catalog_text

# ── Feature toggle ────────────────────────────────────────────────────────────
# On by default: the intent decides which corpora are searched and which
# answer format is used (see intent_router.py). The same call also does the
# translation, so non-Chinese questions need no extra round-trip.
# Set INTENT_DETECTION=false to fall back to keyword heuristics.
ENABLE_INTENT_DETECTION = os.getenv("INTENT_DETECTION", "true").lower() == "true"

# ── Clarifying questions ──────────────────────────────────────────────────────
# When the answer depends on facts the user hasn't given (still employed? how
# many hours? company closed?), the app asks first instead of listing every
# possibility. At most MAX_FOLLOWUP_ROUNDS rounds; the user can always skip
# and get a general answer.
MAX_FOLLOWUP_ROUNDS = 2
MAX_CLARIFY_QUESTIONS = 3

# Skip condition: reply length <= SKIP_MAX_LEN AND contains a trigger keyword.
# Replies longer than SKIP_MAX_LEN are never treated as a skip request even if
# they happen to contain a trigger word (e.g. "我不知道休息日加班費怎麼算").
SKIP_FOLLOWUP_TRIGGERS = ["不知道", "不清楚", "直接回答", "算了", "隨便", "先不說", "跳過", "skip"]
SKIP_MAX_LEN = 10

# ── Prompts ───────────────────────────────────────────────────────────────────
# Placeholders (filled at runtime):
#   {history}          — recent turns as text, 「（無）」 on the first question
#   {user_message}     — the raw user input
#   {ui_lang}          — interface language (e.g. "繁體中文", "English")
#   {clarify_allowed}  — 「可以」/「不可以」: whether another clarifying round is allowed

INTENT_ANALYSIS_PROMPT = """\
你是一個勞動法諮詢助理的前置分析模組。分析使用者訊息並以 JSON 格式回傳結果。
除 JSON 本身外，請勿輸出任何其他文字（包括 ```json 標記、說明文字等）。

## 先前對話（最近幾輪，舊→新；「（無）」表示這是第一個問題）
{history}

## 使用者訊息
{user_message}

## 介面語言
{ui_lang}

## 這次是否還可以反問使用者
{clarify_allowed}

## 回傳格式（嚴格 JSON）
{{
  "translated_query": "這次問題的完整繁體中文版本（規則見說明）",
  "is_followup": true 或 false,
  "intent": "solve_problem 或 find_cases 或 law_lookup 或 out_of_scope",
  "laws": ["相關法規，依關聯度排序，只能從：勞動基準法、勞動基準法施行細則、勞工退休金條例、勞工職業災害保險及保護法、最低工資法、勞工請假規則"],
  "violation": true 或 false,
  "overtime": {{"day_type": "平日／休息日／國定假日／例假日 或 null", "hours": 數字或 null,
               "monthly_wage": 數字或 null, "hourly_wage": 數字或 null}} 或 null,
  "clarify": {{
    "needed": true 或 false,
    "scenario": "反問清單中的情境代號，或 other",
    "ask": ["要問的問題代號，例如 o2、o3"],
    "translated": [{{"q": "問題", "options": ["選項"]}}]
  }}
}}

## 說明
- translated_query：用來檢索法條與案例，必須不看先前對話也能理解。
  - 先判斷使用者訊息能不能單獨看懂：
    - 能單獨看懂（有自己的主題，例如「預告工資以何薪資為基準？」）→ 原樣輸出，
      即使和先前對話的主題相近，也不得加入先前對話的任何內容。
    - 不能單獨看懂（用「那」「這個」「這樣」「呢」「如果…呢」指涉前文、只補充條件，
      或是在回答助理的反問，例如「那休息日呢？」「如果年資三年呢？」「已離職；公司還在營運」）
      → 把先前對話的主題與使用者提供過的所有條件補回來，改寫成完整問題。
    - 訊息看得懂但缺少情境（例如前文在談女性勞工夜間工作，這次問「公司沒經過工會同意可以嗎？」）
      → 只補上前文的情境（「女性勞工夜間工作」），問句其餘部分保持原樣。
  - 原文非繁體中文時一併翻譯成繁體中文。
- is_followup：translated_query 有用到先前對話的內容時為 true（包括回答助理的反問），否則 false
- intent、laws、violation、overtime、clarify 都以改寫後的完整問題判斷。
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
- violation：問題涉及雇主（或投保單位）可能違反規定（例如不給加班費、未提繳退休金、
  未依法預告就解僱），或直接詢問罰則、罰鍰、檢舉時為 true；單純詢問制度內容或計算方式為 false
- overtime：只有在使用者想知道自己某次加班可以拿多少加班費時填寫（含先前對話提供的事實），其他問題給 null。
  數字一律換成阿拉伯數字（「月薪五萬」→ monthly_wage 50000；「三小時」→ hours 3），不知道的欄位填 null，不要猜。
- clarify：本系統要像顧問一樣先弄清楚狀況再給建議，不要把各種可能情況全部列給使用者。
  反問只能從下方「反問清單」挑選，用問題代號表示，不要自己寫新的問題。
  - needed 為 true 的條件（全部成立）：intent 為 solve_problem；使用者（含先前對話）沒有提供某些事實，
    而這些事實會改變適用的規定、處理方式或計算結果；「這次是否還可以反問使用者」為「可以」。
  - 問題已經具體、事實足以回答、或只是問制度規定（law_lookup）、找案例（find_cases）時 needed 為 false。
  - scenario：選最符合的情境；都不符合時填 other。
  - ask：從該情境（以及通用問題）挑最多 3 題，依重要性排序：先挑會改變適用規定或計算結果的事實，
    只挑使用者還沒回答過的；不要挑與答案無關的題目。needed 為 false 時給空陣列。
  - translated：介面語言是繁體中文時給空陣列；否則依 ask 的順序，把每題問題與選項翻譯成介面語言。

## 反問清單
""" + catalog_text() + """
"""
