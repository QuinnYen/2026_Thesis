# 跨領域情感分析數據處理工具 [開發中]

此專案開發用於處理跨領域情感分析研究所需的數據集，提供資料載入、統計分析、視覺化和預處理功能，作為碩士論文研究的基礎工具。

## 功能特色

- **多領域資料支援**：支援 Amazon、Yelp 和 IMDB 三種不同領域的數據集
- **數據統計與分析**：
  - 情感分布分析
  - 評論長度分析
  - 詞頻分析
  - 詞彙重疊分析（使用 Jaccard 相似度）
  - 情感詞分布分析
- **數據預處理**：
  - 清除 HTML 標籤
  - 移除標點符號
  - 轉換為小寫
  - 移除停用詞
  - 詞幹提取
  - 詞形還原
- **批次處理**：支援大型資料集的批次預處理

## 環境需求

- Python 3.6+
- 依賴套件：
  ```
  pandas
  numpy
  matplotlib
  nltk
  jieba
  tkinter
  ```

## 安裝說明

1. 克隆此專案到本地：
   ```
   git clone https://github.com/yourusername/cross-domain-sentiment-analysis.git
   cd cross-domain-sentiment-analysis
   ```

2. 安裝所需套件：
   ```
   pip install -r requirements.txt
   ```

3. 下載 NLTK 資源：
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('vader_lexicon')
   ```

## 使用方法

1. 啟動應用程式：
   ```
   python P1_main.py
   ```

2. 資料載入：
   - 在「Amazon數據」、「Yelp數據」和「IMDB數據」標籤頁中載入對應的數據檔案
   - 支援的資料格式：CSV（Amazon、IMDB）、JSON（Yelp）

3. 資料分析：
   - 在「數據統計」標籤頁中選擇分析類型
   - 點擊「執行分析」生成視覺化結果

4. 資料預處理：
   - 在「數據預處理」標籤頁設定預處理選項
   - 點擊「執行預處理」處理載入的數據
   - 點擊「批次預處理（全量資料）」處理大型資料集
   - 點擊「儲存預處理結果」保存處理後的資料

## 資料集

此工具設計用於處理以下公開資料集：

1. **Amazon 評論數據集**：[Amazon Product Reviews](https://jmcauley.ucsd.edu/data/amazon/)
   - 提供不同產品類別的評論，適合跨領域研究
   - 建議使用 Electronics、Books、Movies_and_TV 等類別

2. **Yelp 評論數據集**：[Yelp Dataset Challenge](https://www.yelp.com/dataset)
   - 在 `ReviewsDatabase` 資料夾中使用 `yelp_academic_dataset_review.json` 檔案

3. **IMDB 電影評論數據集**：[Stanford's IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
   - 提供電影評論及其情感標籤（正面/負面）

## 注意事項

- 大型數據集處理可能需要較長時間，建議使用批次處理功能
- `ReviewsDatabase` 資料夾不包含在版本控制中，需自行下載相關資料集
- 對於跨領域分析，建議選擇至少兩個不同領域的數據集進行比較

## 專案結構

```
.
├── Part01_/               # 主專案資料夾
│   ├── P1_main.py         # 主程式
│   └── preprocessing_scripts/  # 預處理腳本
├── ReviewsDatabase/       # 數據集資料夾 (不包含在Git中)
├── venv/                  # Python虛擬環境 (不包含在Git中)
├── nltk_Download.py       # NLTK資源下載腳本
├── requirements.txt       # 依賴套件清單
├── .gitignore             # Git忽略檔案清單
└── README.md              # 說明文件
```

## 未來擴展

- 添加更多數據集和領域支援
- 集成注意力機制的可視化分析
- 支援中文數據處理
- 實現跨領域模型訓練與評估
