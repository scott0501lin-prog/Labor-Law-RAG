---
title: 勞資爭議智慧法務 AI 顧問
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.45.0
app_file: Fronted/app.py
pinned: false
---

# 勞動法規專家知識庫 — 正確性優先 / Fast Verified RAG

以使用者提供的 `Labor-Law-RAG-main 6(1).zip` 為基礎，保留原本 Streamlit UI、勞資雙角色、MongoDB 帳號及歷史紀錄、原始資料與舊向量索引。未修改遠端 GitHub。正式聊天改用 `Fronted/verified_rag.py`，不再把模型撰寫的法律結論直接顯示給使用者。

## 快速啟動（Mac / VS Code）

解壓縮後，使用 VS Code 開啟 `Labor-Law-RAG-Verified` 資料夾，在終端機執行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

將原本的 `ANTHROPIC_API_KEY`、`MONGO_URI` 填入 `.env`，沿用原本的 MongoDB 資料庫名稱。不要提交 `.env`。沒有 Claude 金鑰仍可做精準法條及支援的 FAQ 查詢，但登入需要 MongoDB。

```bash
python -m unittest discover -s tests -v
streamlit run Fronted/app.py
```

### Hugging Face Space

維持原本的 `Fronted/app.py` 為啟動檔，將新檔案同步至你的 repository，再部署。原有 Secrets `ANTHROPIC_API_KEY`、`MONGO_URI` 不需放入 ZIP；確認 Space 仍設定這些 Secrets。`CLAUDE_MODEL` 可依帳號可用模型調整。不要覆蓋遠端資料庫。若使用的 Space 不支援 Streamlit SDK，需沿用原本 Space 的 Docker 或部署設定；本 ZIP 未取得其遠端設定。

## 核心流程

1. 讀取原始及清理法規 JSON，逐條比較法規名稱、條號、網址與內容；排除刪除條文。原始與清理資料有差異就拒答。
2. 精準法條或有限的 FAQ 使用程式直接查詢，不呼叫 LLM。
3. 其餘問題透過 BM25 中文字元檢索、精準條號及主題規則選擇候選條文。可選用既有向量索引進行 RRF 混合排序。
4. Claude 只回傳證據 ID 與段落編號，單次呼叫，限制輸出長度。結構錯誤、虛構 ID、截斷或 API 失敗一律拒答。
5. 預設核對官方全國法規資料庫網頁原文，成功快取 15 分鐘；同時需要兩部法律時並行抓取。若有異動，停止輸出該次舊條文；若網站無法連線，只能明確標示為未核對的本地歷史參考資料。這不是自動更新或歷史有效版本驗證。
6. 由程式組裝完整原文、相關段落與官方網址，整份驗證後才顯示。案例等按下按鈕才查詢，且不作為已驗證的法律依據。

## 效能設定

| 環境變數 | 預設 | 說明 |
|---|---|---|
| `VERIFY_OFFICIAL` | `true` | 官方文字核對。關閉時必須標示離線未核對。 |
| `RAG_SEMANTIC` | `false` | 預設 BM25＋規則，免下載大型模型；true 開啟向量融合。 |
| `EMBED_MODEL` | `text2vec` | 只有啟用向量融合才需使用；隨附 text2vec 索引。 |
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | 可設定帳號目前有權使用的 Claude 模型。 |

若要進行向量研究或比較原始系統，另安裝 `pip install -r requirements-research.txt`，設定 `RAG_SEMANTIC=true` 並確認對應索引已存在。舊版 benchmark_timing.py 及 eval/benchmark_models.py 用於歷史架構，不是新版端到端效能測試。使用 `python eval/benchmark_verified.py` 測量新版離線檢索。部署若未上傳被 .gitignore 排除的 embeddings，預設模式仍可使用。

## 重要限制

- 不是 100% 法律正確的專家系統。引用存在與文字一致不等於檢索完整、解釋正確或個案適用。正式模式刻意限制自由生成白話法律結論；多語系 UI 保留，但來源原文是繁體中文，翻譯及完整對話追問暫不啟用。請將完整事件事實放在同一問題中。
- 法律資料目前僅支援勞動基準法與施行細則。其他法規、歷史修法版本、法律函釋適用性及法院裁判效力尚未全面核對。不能將本地資料當作現行完整法規。
- 內建加班費計算器只支援一般月薪制平日（最多 4 小時）及休息日（最多 12 小時），公式使用有理數；不支援特殊制度。公式參考[勞動部月薪制試算](https://calcr2.mol.gov.tw/Monthly)及[勞動部加班補休規定](https://www.mol.gov.tw/1607/28162/28166/28180/28198/28202/47031/post)，尚未自動同步公式修法。
- 本地案例原文與資料欄位未改寫，不保證內容最新；衍生整理資料不進入案例查詢。舊聊天紀錄保留，不代表重新驗證過。
- 原版 MongoDB 帳密及權限機制未全面安全審計。正式公開上線前應額外檢查資安、個資與法律專家驗收。
- 測試使用模擬 Claude 與官方網頁資料。沒有你的金鑰、MongoDB 和實際 Space 運行環境，未做完整登入、雲端 API、官方網站連線及真實法律專家準確率測試。無法保證真實端到端時間。

詳見 `VERIFICATION_REPORT.md`。
