# 跨領域情感分析系統 [開發中]

這是一個完整的跨領域情感分析系統，專為處理不同領域（如電影、產品、餐廳等）的評論資料而設計。系統採用現代NLP技術，支援BERT語義提取和LDA主題建模，能夠有效識別跨領域的評論面向和情感。

## 功能特色

- **多源資料處理**：支援IMDB電影評論、Amazon產品評論和Yelp餐廳評論等多種資料來源
- **BERT語義提取**：使用預訓練的BERT模型將文字轉換為高維向量表示
- **LDA面向切割**：採用LDA主題建模技術自動識別評論中的不同面向
- **面向向量計算**：為每個面向生成代表性向量，用於跨領域分析
- **視覺化呈現**：生成多種視覺化結果，幫助理解資料和模型
- **中文界面**：完整的中文使用者界面，操作簡單直觀

## 系統要求

- Python 3.7+
- 建議使用獨立的虛擬環境
- NVIDIA GPU (推薦，但非必需)
- 僅在 Windows 上運行

## 安裝指南

1. 克隆此專案到本地：
```
git clone https://github.com/QuinnYen/2026_Thesis.git
cd cross-domain-sentiment-analysis
```

2. 安裝所需的相依套件：
```
pip install -r requirements.txt
```

3. 安裝必要的NLTK資源（系統會在首次運行時自動下載，但也可以手動安裝）：
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 使用方法

### 啟動應用程式

執行主程式以啟動圖形界面：
```
python Part02_/Main.py
```

### 處理流程

系統設計為分步驟操作，處理流程如下：

1. **資料導入與預處理**：
   - 選擇資料來源（IMDB、Amazon、Yelp或其他）
   - 上傳評論資料檔案（支援CSV、JSON、TXT格式）
   - 系統會自動清理文字、分詞並移除停用詞

2. **BERT語義提取**：
   - 將清理後的文字轉換為BERT嵌入向量
   - 生成語義表示，捕捉文字的深層語意

3. **LDA面向切割**：
   - 自動識別評論中的不同主題/面向
   - 生成主題-詞語分佈和文檔-主題分佈
   - 為每條評論標記主要面向

4. **面向向量計算**：
   - 計算每個面向的代表性向量
   - 生成t-SNE降維可視化結果
   - 支援多種格式導出（CSV、JSON、Pickle）

### 結果瀏覽

系統提供完整的結果瀏覽功能：
- 查看處理資料和各種可視化結果
- 生成處理報告和概覽報告
- 輕鬆匯出分析結果

## 檔案結構

```
Part02_/
├── data/                  # 存放原始資料
├── results/               # 存放處理結果
│   ├── 01_processed_data/ # 處理後的原始資料
│   ├── 02_bert_embeddings/ # BERT嵌入向量
│   ├── 03_lda_topics/     # LDA主題結果
│   ├── 04_aspect_vectors/ # 面向向量結果
│   ├── models/            # 保存的模型
│   ├── visualizations/    # 可視化結果
│   └── exports/           # 匯出的資料
├── logs/                  # 日誌文件
└── src/                   # 源碼
    ├── bert_embedder.py          # BERT嵌入提取器
    ├── bert_offline_downloader.py # BERT模型離線下載工具
    ├── data_importer.py          # 資料導入器
    ├── lda_aspect_extractor.py   # LDA主題提取器
    ├── aspect_vector_calculator.py # 面向向量計算器
    ├── console_output.py         # 控制台輸出管理
    ├── result_manager.py         # 結果管理工具
    └── settings/                 # 設定檔案
        ├── topic_labels.py       # 主題標籤定義
        └── visualization_config.py # 可視化配置
```

## 資料來源支援

系統預設支援以下資料來源的處理和分析：

1. **IMDB電影評論**：
   - 電影評論資料，包含10個主題：觀影體驗、英雄故事元素、女性角色與喜劇、整體評價等
   - 適合分析觀眾對電影的不同面向評價

2. **Amazon產品評論**：
   - 產品評論資料，包含10個主題：產品質量、價格價值、使用體驗、客戶服務等
   - 適合分析消費者對產品的多方面評價

3. **Yelp餐廳評論**：
   - 提供餐廳評論的處理功能，包含合併business和review文件
   - 自動抽樣，方便處理大型資料集

## Yelp資料處理特別說明

系統提供Yelp資料的特殊處理功能，用於處理Yelp Academic Dataset：

1. 需要分別選擇business.json和review.json文件
2. 可設置抽樣數量，處理大型資料集
3. 系統會自動合併資料，只保留餐廳相關的評論

## 疑難排解

- **記憶體不足錯誤**：對於大型資料集，建議使用較小的批處理大小，或減小抽樣數量
- **中文顯示問題**：系統會自動配置中文字體，若有問題可檢查日誌中的字體配置訊息
- **BERT模型下載失敗**：可使用bert_offline_downloader.py手動下載並配置模型

## 關於BERT離線使用

若需在無網路環境使用：

1. 使用bert_offline_downloader.py下載BERT模型：
```
python Part02_/src/bert_offline_downloader.py --model bert-base-uncased --output ./bert_models
```

2. 系統會在需要時自動使用本地模型

## 注意事項

- 處理大型資料集時可能需要較長時間，系統會顯示進度並記錄日誌
- 建議使用具有CUDA支援的GPU以加速BERT處理
- 所有處理結果會自動保存，可隨時瀏覽和匯出

## 授權資訊

此系統為學術研究和教育用途開發，使用請遵循相關資料集的授權規定。

## 參考文獻

- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Latent Dirichlet Allocation
- Cross-Domain Sentiment Analysis: A Survey
