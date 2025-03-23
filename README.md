# 跨領域情感分析系統 [開發中]

此專案開發用於跨領域情感分析研究，提供資料處理、特徵提取、面向切割和跨領域模型建構的完整流程，作為碩士論文研究的核心工具。

## 功能特色

- **多領域資料處理**：
  - 支援 Amazon (產品評論)、Yelp (餐廳評論) 和 IMDB (電影評論) 三種不同領域的數據集
  - 自動識別並處理不同格式的數據 (CSV、JSON、TXT)
  - 高效批次處理大型資料集

- **文本預處理**：
  - 移除 HTML 標籤、URL、特殊符號
  - 文本正規化與標準化
  - 分詞、移除停用詞、詞形還原

- **特徵表示與面向切割**：
  - **BERT 語義表示提取**：使用預訓練 BERT 模型生成高質量文本向量
  - LDA 主題建模：識別不同領域間的共享面向
  - 面向表示構建：計算面向相關句子的平均向量

- **跨領域模型構建**：
  - 領域適應/遷移學習
  - 注意力機制整合 (面向級注意力)
  - 整合領域間共性和差異特徵

- **評估與分析**：
  - 領域內評估 (內部測試) 和跨領域評估 (外部測試)
  - 面向遷移效果分析
  - 領域特性比較與可視化

## 系統需求

- **Python 3.7+**
- **基礎套件**：
  ```
  pandas
  numpy
  nltk
  beautifulsoup4
  matplotlib
  tkinter
  ```

- **深度學習套件**：
  ```
  torch
  transformers
  scikit-learn
  gensim
  tqdm
  ```

## 安裝說明

1. 克隆此專案到本地：
   ```
   git clone https://github.com/yourusername/cross-domain-sentiment-analysis.git
   cd cross-domain-sentiment-analysis
   ```

2. 建立虛擬環境 (建議使用)：
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. 安裝必要套件：
   ```
   pip install -r requirements.txt
   ```

4. 下載 NLTK 資源：
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

5. (選用) 預下載 BERT 模型 (適用於離線環境)：
   ```
   python bert_offline_downloader.py --model bert-base-uncased --output ./bert_models
   ```

## 使用方法

1. 啟動應用程式：
   ```
   python Main.py
   ```

2. **資料處理流程**：

   a. **導入數據**：
   - 點擊「選擇文件」選擇要分析的評論資料文件
   - 點擊「開始導入數據」執行資料清理與預處理
   - 系統會自動識別文本列，並進行清洗、分詞和詞形還原

   b. **BERT 語義提取**：
   - 點擊「執行 BERT 語義提取」將文本轉換為向量表示
   - 系統會使用預訓練的 BERT 模型提取語義特徵
   - 結果將保存為 NPZ (向量) 和 CSV (元數據) 檔案

   c. **LDA 面向切割**：
   - 設定希望識別的主題數量
   - 點擊「執行 LDA 面向切割」進行主題建模
   - 系統會識別出跨領域的共享面向

   d. **計算面向向量**：
   - 點擊「執行計算」計算每個面向的平均向量表示
   - 點擊「匯出平均向量」保存結果以供後續使用

3. **模型訓練與評估** (開發中)：
   - 在「模型訓練」和「評估分析」標籤頁中操作
   - 設定源領域與目標領域
   - 訓練跨領域模型並評估效果

## 資料集

此系統設計用於處理以下公開資料集：

1. **Amazon 評論數據集**：[Amazon Product Reviews](https://jmcauley.ucsd.edu/data/amazon/)
   - 包含不同產品類別的評論，適合跨領域研究
   - 推薦使用 Electronics、Books、Movies_and_TV 等類別

2. **Yelp 評論數據集**：[Yelp Dataset Challenge](https://www.yelp.com/dataset)
   - 包含餐廳及其他商家評論，可作為獨立領域

3. **IMDB 電影評論數據集**：[Stanford's IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
   - 提供電影評論及其情感標籤

## 專案結構

```
.
├── Main.py                # 主程式 (GUI)
├── data_importer.py       # 資料導入與預處理模組
├── bert_embedder.py       # BERT 語義提取模組
├── lda_aspect_extractor.py # LDA 面向切割模組 (開發中)
├── aspect_vector_calculator.py # 面向向量計算模組 (開發中)
├── console_output.py      # 控制台輸出管理工具
├── bert_offline_downloader.py # BERT 模型離線下載工具
├── data/                  # 處理後數據目錄
├── results/               # 結果目錄
├── logs/                  # 日誌目錄
├── models/                # 模型目錄
├── bert_models/           # 預下載的 BERT 模型 (可選)
├── requirements.txt       # 依賴套件清單
└── README.md              # 說明文件
```

## 注意事項

- 處理大型資料集時，BERT 提取可能需要較長時間，建議使用支援 CUDA 的 GPU 加速
- 第一次運行時，系統會自動下載 NLTK 資源和 BERT 模型，需要網絡連接
- 對於跨領域分析，建議選擇至少兩個不同領域的數據集進行比較
- 目前系統的「模型訓練」和「評估分析」模塊仍在開發中

## 未來擴展

- 完成模型訓練和評估分析模塊
- 添加更多數據集和領域支援
- 實現面向對齊和映射功能
- 集成更多預訓練模型選項 (如 RoBERTa、XLNet)
- 改進可視化分析功能
- 支援中文資料處理和分析
