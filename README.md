# 跨領域情感分析系統 (Part04)

這是一個跨領域情感分析系統的第四部分，專為處理不同領域（如電影、產品、餐廳等）的評論資料而設計。系統採用現代自然語言處理 (NLP) 技術，支援 BERT 語義提取和 LDA 主題建模，能夠有效識別跨領域的評論面向和情感，並提供視覺化分析結果。

## 功能特色

- **多源資料處理**：支援匯入 CSV、JSON、TXT 等格式的評論資料，並特別針對 Yelp 資料集提供合併處理功能。
- **BERT 語義提取**：使用預訓練的 BERT 模型將文字轉換為高維向量表示，捕捉文字的深層語意。
- **LDA 主題建模與面向切割**：採用 LDA (Latent Dirichlet Allocation) 主題建模技術自動識別評論中的不同主題或面向，並生成主題-詞語分佈。
- **面向向量計算**：基於多種注意力機制（包括無注意力、相似度、關鍵詞、自注意力及組合注意力）為每個面向生成代表性向量。
- **模型評估**：計算主題一致性、主題分離度等指標來評估模型的表現。
- **視覺化呈現**：
    - 以表格形式顯示原始資料和預處理後資料。
    - 展示 LDA 主題、面向向量及評估結果。
    - 提供多種圖表類型，如主題分佈圖、面向向量質量圖、情感分析性能圖、注意力機制評估圖及綜合比較圖等。
- **使用者友善介面**：基於 PyQt5 建構圖形化使用者介面，包含分析處理、結果視覺化及設定等分頁。
- **參數化設定**：允許使用者透過設定頁面調整 BERT、LDA、注意力機制、路徑、視覺化及系統等相關參數。
- **NLTK 資源管理**：啟動時自動檢查並下載必要的 NLTK 資源 (punkt, stopwords, wordnet)。

## 系統要求

- Python 3.7+
- 建議使用獨立的虛擬環境
- NVIDIA GPU (推薦，用於加速 BERT 處理，但非必需)
- 作業系統：主要於 Windows 環境開發與測試

## 安裝指南

1.  **克隆專案**：
    ```bash
    git clone https://github.com/QuinnYen/2026_Thesis.git
    cd 2026_Thesis
    ```

2.  **安裝相依套件**：
    建議先建立虛擬環境，然後安裝 `requirements.txt` (需自行根據 `import` 語句建立此檔案，主要套件可能包含 `torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `nltk`, `beautifulsoup4`, `pyqt5`, `matplotlib`, `seaborn`, `wordcloud`, `tqdm`, `plotly` 等)。
    ```bash
    pip install -r requirements.txt
    ```

3.  **NLTK 資源下載**：
    系統會在首次運行 `Part04_/main.py` 或 `Part04_/download_nltk_data.py` 時嘗試自動下載必要的 NLTK 資源 (punkt, stopwords, wordnet) 至 `Part04_/nltk_data` 目錄。若自動下載失敗，可以手動執行：
    ```bash
    python Part04_/download_nltk_data.py
    ```
    或在 Python 環境中執行：
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

## 使用方法

### 啟動應用程式

執行主程式以啟動圖形化使用者介面：
```bash
python Part04_/main.py
```

### 處理流程

系統提供一個分頁式的操作介面，主要處理流程如下：

1.  **分析處理 (Analysis Tab)**
    * **資料匯入**：點擊「導入CSV/JSON...」或「Yelp合併導入」按鈕載入評論資料。可設定處理的數據數量。
    * **參數設定**：設定 LDA 的主題數量、迭代次數，以及選擇要使用的注意力機制。
    * **執行分析**：點擊「運行全部」開始分析。系統將依序執行：
        1.  資料預處理 (文字清洗、分詞、詞形還原)
        2.  BERT 嵌入計算
        3.  LDA 主題建模
        4.  面向向量計算 (使用選定的注意力機制)
        5.  模型評估
    * **結果查看**：
        * 「原始數據」和「預處理數據」分頁會以表格展示資料。
        * 「LDA主題」、「面向向量」和「評估結果」分頁會顯示相應的分析結果。
    * **結果儲存**：點擊「保存結果」按鈕可將面向向量等結果儲存為 JSON 檔案。

2.  **結果視覺化 (Visualization Tab)**
    * **載入結果**：點擊「瀏覽...」按鈕選擇先前分析儲存的結果檔案 (通常是 JSON 格式) 或整個 `run_xxxx` 資料夾。
    * **選擇視覺化類型**：系統提供多種視覺化選項，分布在不同的標籤頁下，例如：
        * 面向向量質量 (內聚度、分離度、綜合得分、輪廓係數、困惑度)
        * 情感分析性能 (準確率、精確率、召回率、F1 分數)
        * 注意力機制評估 (注意力分布、權重比較)
        * 主題模型評估 (主題連貫性、主題分布)
        * 綜合比較 (降維視覺化、多指標綜合視圖)
    * **產生圖表**：設定相應圖表的參數後，點擊「生成並保存圖片」。圖片會儲存於 `Part04_/1_output/visualizations/` 下的對應子目錄。

3.  **設定 (Settings Tab)**
    * 在此頁面可以詳細配置系統的各項參數，包括：
        * **BERT 設定**：模型名稱、最大長度、批次大小、是否使用 GPU、快取目錄等。
        * **LDA 設定**：主題數量、Alpha/Beta 參數、最大迭代次數、演算法、批次大小等。
        * **注意力機制設定**：啟用哪些機制、各機制權重、窗口大小、閾值等。
        * **路徑設定**：設定資料、輸出、模型、日誌等目錄的路徑。
        * **視覺化設定**：圖表 DPI、格式、顏色方案、字體等。
        * **系統設定**：日誌等級、是否啟用多處理、工作執行緒數、快取設定等。
    * 修改設定後需點擊「應用」或「儲存」使設定生效。部分設定可能需要重啟應用程式。

## 檔案結構 (Part04)

```
Part04_/
├── 1_output/                       # 所有輸出的根目錄 (動態生成 run_{timestamp} 子目錄)
│   └── run_{timestamp}/
│       ├── 01_processed_data/      # 預處理後的資料
│       ├── 02_bert_embeddings/     # BERT 嵌入向量
│       ├── 03_lda_topics/          # LDA 主題結果 (模型、主題詞、文檔主題分佈)
│       ├── 04_aspect_vectors/      # 面向向量結果
│       ├── 05_evaluation/          # 模型評估結果
│       ├── logs/                   # 日誌檔案
│       ├── models/                 # 儲存的訓練模型 (如 LDA 模型)
│       ├── visualizations/         # 視覺化圖表
│       └── exports/                # 匯出的報告或資料
├── data/                           # (建議) 存放原始資料集 (需自行建立或透過設定頁面配置)
├── gui/                            # 圖形化使用者介面相關模組
│   ├── analysis_tab.py           # 分析處理分頁
│   ├── main_window.py            # 主視窗
│   ├── settings_tab.py           # 設定分頁
│   └── visualization_tab.py      # 結果視覺化分頁
├── modules/                        # 核心功能模組
│   ├── aspect_calculator.py      # 面向向量計算
│   ├── attention_mechanism.py    # 各種注意力機制實現
│   ├── bert_embedder.py          # BERT 嵌入提取
│   ├── data_processor.py         # 資料預處理
│   ├── evaluator.py              # 模型評估
│   ├── lda_modeler.py            # LDA 主題建模
│   └── visualizer.py             # 結果視覺化
├── utils/                          # 工具模組
│   ├── config.py                 # 配置管理
│   ├── file_manager.py           # 檔案操作管理
│   ├── logger.py                 # 日誌記錄
│   └── settings/                 # 設定檔相關
│       ├── config.json           # (預期) 系統設定檔
│       └── topic_labels.py       # 主題標籤定義
├── resources/                      # 存放應用程式資源 (如圖示、Logo)
├── download_nltk_data.py           # NLTK 資源下載腳本
├── main.py                         # 應用程式主入口點
└── nltk_data/                      # (自動生成) NLTK 下載資源存放目錄
```

## 資料來源支援

系統設計支援處理多種來源的評論資料，特別針對以下資料集進行了適配或提供了處理建議：

1.  **IMDB 電影評論**：可分析電影的劇情、演技、視覺效果等多個面向。
2.  **Amazon 產品評論**：可分析產品的品質、功能、性價比、客戶服務等多個面向。
3.  **Yelp 餐廳評論**：
    * `analysis_tab.py` 中包含 `import_yelp_data` 方法，用於合併 Yelp 的 business 和 review 資料檔案。
    * `data_processor.py` 中的 `YelpProcessor` 類別提供專門處理 Yelp 資料的邏輯，包括抽樣和合併。

## 疑難排解

- **記憶體不足錯誤**：對於大型資料集，建議在「分析處理」分頁設定較小的「處理數據數量」，或在「設定」分頁調整 BERT 和 LDA 的批次大小。
- **中文顯示問題**：系統會嘗試自動配置 Matplotlib 的中文字體。若視覺化圖表中的中文顯示為方框，請確保系統中已安裝合適的中文字體 (如 Microsoft JhengHei, Microsoft YaHei, SimHei 等)，或在 `Part04_/modules/visualizer.py` 中的 `set_chinese_font()` 函數中指定正確的字體路徑。
- **BERT 模型下載失敗**：若系統無法自動下載 BERT 模型，可參考 `transformers` 函式庫文件，手動下載模型至本地，並在「設定」分頁的 BERT 設定中指定模型路徑或名稱。
- **NLTK 資源問題**：若 `punkt_tab` 相關錯誤持續出現，請確保 `download_nltk_data.py` 腳本已正確執行，並在 `Part04_/nltk_data/tokenizers/` 路徑下生成了 `punkt_tab` 目錄及其內容。

## 注意事項

- 處理大型資料集時可能需要較長時間，系統介面會顯示進度，詳細執行過程會記錄在日誌檔案中 (`Part04_/1_output/run_{timestamp}/logs/`)。
- 建議使用具有 CUDA 支援的 NVIDIA GPU 以加速 BERT 模型的嵌入計算。可在「設定」分頁中啟用 GPU。
- 所有分析過程中的中間結果和最終輸出都會儲存於 `Part04_/1_output/run_{timestamp}/` 下的對應子目錄中。

## 授權資訊

此系統主要為學術研究和教育用途開發。若使用公開資料集，請遵循其原始授權規定。

## 參考概念與技術

-   **BERT**: Bidirectional Encoder Representations from Transformers
-   **LDA**: Latent Dirichlet Allocation
-   **Attention Mechanisms** in Natural Language Processing
-   **Cross-Domain Sentiment Analysis**
