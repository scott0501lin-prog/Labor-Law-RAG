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

# ⚖️ 勞資爭議智慧法務 AI 顧問系統

一套以 RAG（檢索增強生成）為核心的台灣勞動法律諮詢平台，協助勞方與資方快速查詢勞動基準法相關規定、案例與權益說明。

---

## 功能特色

- **雙角色入口**：勞方（員工）與資方（雇主 / HR）分別提供不同的諮詢語氣與法律視角
- **RAG 智慧檢索**：先由 Claude 判斷問題意圖並補上法條用語，再以 bge-m3 向量檢索與 jieba BM25 關鍵字檢索並行，用 RRF 合併結果，最後由 Claude Haiku 依檢索到的法條與案例作答
- **意圖分流**：問題分為諮詢、找案例、查法條、範圍外四類，各自使用不同的檢索數量與回答格式；範圍外問題直接轉介
- **資料範圍**：勞動基準法及施行細則、勞工退休金條例、勞工職業災害保險及保護法、最低工資法、勞工請假規則，以及勞動部函釋與問答、臺北市勞動局常見問答、法院勞動判決
- **法條強制帶入**：針對加班費、資遣、特休、國定假日等高頻主題，確保關鍵條文不被遺漏
- **多語系支援**：繁體中文、English、日本語、한국어、Bahasa Indonesia、Tiếng Việt、Filipino
- **歷史對話紀錄**：登入後自動儲存每次對話，隨時可切換查看
- **帳號系統**：SHA-256 加鹽雜湊密碼保護，MongoDB Atlas 雲端儲存

---

## 技術架構

| 層級 | 技術 |
|------|------|
| 前端介面 | Streamlit |
| 語言生成模型 | Claude Haiku 5.5（意圖分析、回答生成）|
| 語意嵌入模型 | BAAI/bge-m3（Sentence-Transformers，離線預先編碼）|
| 向量檢索 | ChromaDB（失敗時退回 NumPy）|
| 關鍵字檢索 | jieba + 自訂法律詞典 + BM25，與向量結果以 RRF 合併 |
| 帳號 / 對話儲存 | MongoDB Atlas |
| 資料蒐集 | Requests + BeautifulSoup |
| 版本控制 | Git / GitHub（向量索引以 Git LFS 儲存）|
| 雲端部署 | Hugging Face Spaces |

---

## 安裝與執行

### 1. 複製專案

向量索引檔以 Git LFS 儲存，複製前請先安裝 [Git LFS](https://git-lfs.com)：

```bash
git lfs install
git clone https://github.com/scott0501lin-prog/Labor-Law-RAG.git
cd Labor-Law-RAG
```

### 2. 安裝套件

```bash
pip install -r requirements.txt
```

### 3. 設定環境變數

在根目錄建立 `.env` 檔案：

```
ANTHROPIC_API_KEY=你的_Anthropic_API_Key
MONGO_URI=你的_MongoDB_連線字串
```

### 4. 向量索引

`backend/embeddings/` 已附上預先建好的 bge-m3 索引，可以直接啟動。
修改 `backend/scraper2/data/` 的資料後才需要重建（CPU 約需 2～3 小時）：

```bash
python backend/build_index.py              # 法條＋案例
python backend/build_index.py --only law   # 只重建法條（數分鐘）
```

### 5. 啟動應用程式

```bash
streamlit run Fronted/app.py
```

---

## 專案結構

```
├── Fronted/
│   ├── app.py          # 主應用程式（Streamlit UI + RAG 引擎）
│   ├── db.py           # MongoDB 資料層（帳號、對話紀錄）
│   ├── i18n.py         # 多語系介面文字
│   ├── force_rules.py  # 關鍵字強制帶入條文規則
│   ├── intent_config.py # 意圖判斷與追問設定
│   ├── intent_router.py # 依意圖決定檢索範圍與回答格式
│   └── hybrid_search.py # jieba BM25 + 向量檢索，以 RRF 合併
├── backend/
│   ├── build_index.py  # 離線切塊並以 bge-m3 編碼，輸出至 embeddings/
│   ├── embeddings/     # 預計算的向量索引（.npy + .json）
│   ├── scraper2/       # 爬蟲（spiders/）與法規、問答、函釋、判決資料（data/）
│   └── data/           # 手動整理的補充資料
├── eval/               # 測試與評估腳本（檢索命中率、計時、強制帶入規則測試）
├── requirements.txt
└── .env                # 機密設定（不上傳 GitHub）
```

---

## 雲端部署

本系統部署於 **Hugging Face Spaces**（Streamlit SDK，設定見本檔開頭的 YAML）。

1. Space 的 **Settings → Variables and secrets** 設定 `ANTHROPIC_API_KEY` 與 `MONGO_URI`
2. 將程式推送到 Space 的 git 儲存庫（向量索引透過 Git LFS 一併上傳）：

```bash
git remote add space https://huggingface.co/spaces/<帳號>/<Space 名稱>
git push space feature/scraper2-data:main
```

啟動時會載入 bge-m3 模型與預先算好的向量，寫入 ChromaDB 並建立 BM25 索引（約 1～2 分鐘），之後每題不需重新編碼文件。

檢索評估：`python eval/eval_retrieval.py`（臺北市勞動局常見問答 93 題，Hit@5 80%）。

---

## 注意事項

- `.env` 檔案含有 API 金鑰與資料庫連線字串，已加入 `.gitignore`，**請勿上傳至 GitHub**
- 本系統提供的法律資訊僅供參考，不構成正式法律建議

---

## 開發團隊

淡江大學 資訊管理學系  
畢業專題 — 2026 年
