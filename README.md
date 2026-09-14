# ⚖️ 勞資爭議智慧法務 AI 顧問系統

一套以 RAG（檢索增強生成）為核心的台灣勞動法律諮詢平台，協助勞方與資方快速查詢勞動基準法相關規定、案例與權益說明。

---

## 功能特色

- **雙角色入口**：勞方（員工）與資方（雇主 / HR）分別提供不同的諮詢語氣與法律視角
- **RAG 智慧檢索**：結合 NumPy 向量搜尋與 Gemini 2.5 Flash，依據問題自動檢索最相關的法條與案例
- **法條強制帶入**：針對加班費、資遣、特休、國定假日等高頻主題，確保關鍵條文不被遺漏
- **多語系支援**：繁體中文、English、日本語、한국어、Bahasa Indonesia、Tiếng Việt、Filipino
- **歷史對話紀錄**：登入後自動儲存每次對話，隨時可切換查看
- **帳號系統**：SHA-256 加鹽雜湊密碼保護，MongoDB Atlas 雲端儲存

---

## 技術架構

| 層級 | 技術 |
|------|------|
| 前端介面 | Streamlit |
| 語言生成模型 | Google Gemini 2.5 Flash |
| 語意嵌入模型 | shibing624/text2vec-base-chinese（Sentence-Transformers）|
| 向量檢索 | NumPy 餘弦相似度 |
| 帳號 / 對話儲存 | MongoDB Atlas |
| 資料蒐集 | Requests + BeautifulSoup + Playwright/Selenium |
| 版本控制 | Git / GitHub |
| 雲端部署 | Streamlit Community Cloud |

---

## 安裝與執行

### 1. 複製專案

```bash
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
GEMINI_API_KEY=你的_Gemini_API_Key
MONGO_URI=你的_MongoDB_連線字串
```

### 4. 建立向量索引（首次執行）

```bash
python backend/build_index.py
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
│   └── i18n.py         # 多語系介面文字
├── backend/
│   ├── build_index.py  # 預計算 embedding 並儲存為 .npy
│   ├── embeddings/     # 預計算的向量索引（.npy + .json）
│   ├── labor_law_cleaned.json
│   ├── labor_law_rules_cleaned.json
│   └── *_cases.json    # 各類案例資料
├── requirements.txt
└── .env                # 機密設定（不上傳 GitHub）
```

---

## 雲端部署

本系統已部署至 **Streamlit Community Cloud**，無需本地安裝即可直接使用。

---

## 注意事項

- `.env` 檔案含有 API 金鑰與資料庫連線字串，已加入 `.gitignore`，**請勿上傳至 GitHub**
- 本系統提供的法律資訊僅供參考，不構成正式法律建議

---

## 開發團隊

淡江大學 資訊管理學系  
畢業專題 — 2026 年
