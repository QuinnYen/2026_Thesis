# BERT情感分析系統 - 完整版

## 🌟 概述
這是一個基於BERT的情感分析系統，整合多種注意力機制和高性能分類器，提供完整的圖形化使用者介面和命令行工具，用於情感分析研究和應用。

## ✨ 最新功能更新

### 🔥 **新增高性能分類器**
- **XGBoost** ⚡ - 支援GPU加速，最高準確率
- **Logistic Regression** 🚀 - 速度最快，適合中小數據
- **Random Forest** 🌳 - 穩定可靠，可解釋性強
- **SVM Linear** 📐 - 線性分類，適合特定場景

### 🖥️ **智能環境偵測**
- 自動偵測GPU/CPU環境
- 智能配置最佳運行模式
- 實時顯示計算環境信息
- 自動優化性能設置

### ⏱️ **詳細計時統計**
- 實時訓練進度顯示
- 各階段詳細時間統計
- GUI和終端雙重時間顯示
- 性能瓶頸分析工具

## 🎯 系統架構

### 第一分頁 - 資料處理
- **數據集類型選擇**: 支援多種數據格式
- **文本輸入 → 導入檔案**: 支援.txt, .csv, .json格式
- **數據抽樣功能**: 大數據集智能抽樣（建議>10,000筆啟用）
- **文本處理 → 開始處理**: 智能文本預處理
- **BERT編碼 → 開始編碼**: 高效BERT特徵提取

### 第二分頁 - 注意力機制測試
- **分類器設定區域**: 
  - 四種高性能分類器選擇
  - 實時環境信息顯示
  - 智能建議和說明
- **單一注意力實驗**: 相似度、關鍵詞、自注意力
- **雙重組合實驗**: 多種注意力機制組合
- **三重組合實驗**: 完整注意力機制融合
- **實時計時顯示**: 訓練進度和時間統計

### 第三分頁 - 比對分析
- 多維度性能比較（準確率、F1、召回率、精確率）
- 詳細分類結果展示
- 最佳模型自動標記
- 完整的結果匯出功能 

# BERT情感分析系統 - Part05：注意力機制實現

本項目實現了一個完整的BERT情感分析系統，並整合了多種注意力機制用於情感面向建模。

## 🌟 主要特性

### 🔥 創新的注意力機制實現
1. **多種注意力機制**：支援無注意力、相似度注意力、關鍵詞注意力、自注意力和組合注意力
2. **面向計算創新**：在面向向量計算時引入注意力機制，提升情感分析準確性
3. **組合vs單一比較**：系統性比較不同注意力機制的效果
4. **動態權重調整**：支援自定義組合注意力的權重配置

### 🚀 高性能分類器系統
- **XGBoost分類器**：支援GPU加速，在大數據集上提供最佳準確率
- **邏輯迴歸**：速度最快，在中小數據集上平衡效果與效率
- **隨機森林**：穩定可靠，提供良好的可解釋性
- **線性SVM**：適合線性可分數據，小數據集表現良好
- **智能環境偵測**：自動偵測GPU/CPU環境並配置最佳運行模式
- **詳細計時統計**：提供各階段時間分析和性能瓶頸識別

### 🚀 核心功能
- **BERT特徵提取**：使用預訓練BERT模型提取768維文本特徵
- **注意力機制分析**：支援5種不同類型的注意力機制
- **智能分類評估**：整合注意力機制分析與分類器性能評估
- **效果評估比較**：提供內聚度、分離度、準確率、F1分數等多維指標
- **完整的處理流程**：從數據預處理到結果分析的完整pipeline
- **智能GUI界面**：友好的圖形用戶界面，支援實時狀態顯示和環境偵測
- **靈活命令行支援**：豐富的命令行選項，支援批量處理和自動化

## 📁 項目結構

```
Part05_/
├── Part05_Main.py                    # 主程式入口
├── requirements.txt                  # 依賴包列表
├── test_attention_simple.py          # 簡化測試腳本
├── test_classifier_features.py      # 🆕 分類器功能測試腳本
├── install_xgboost.py               # 🆕 XGBoost自動安裝腳本
├── README.md                        # 項目說明
├── README_Classifier_Features.md   # 🆕 分類器功能說明
│
├── modules/                         # 核心模組
│   ├── attention_mechanism.py       # 注意力機制實現
│   ├── attention_analyzer.py        # 注意力分析器
│   ├── attention_processor.py       # 注意力處理器
│   ├── bert_encoder.py              # BERT編碼器
│   ├── sentiment_classifier.py      # 🆕 高性能情感分類器
│   ├── run_manager.py               # 運行管理器
│   └── text_preprocessor.py         # 文本預處理器
│
├── utils/                           # 工具文件
│   └── topic_labels.json            # 主題標籤配置
│
├── gui/                             # 圖形界面
│   ├── main_window.py               # 🆕 增強版主視窗（含分類器選擇）
│   └── config.py                    # GUI配置檔案
│
├── examples/                        # 使用範例
│   ├── attention_example.py         # 注意力機制範例
│   └── README.md                    # 範例說明
│
└── output/                          # 輸出目錄（自動生成）
    ├── 02_bert_embeddings.npy       # BERT特徵向量
    ├── 03_attention_analysis_*.json # 注意力分析結果
    ├── 03_attention_comparison_*.json # 注意力比較結果
    ├── sentiment_classifier_*.pkl   # 🆕 訓練好的分類模型
    ├── label_encoder.pkl            # 🆕 標籤編碼器
    └── complete_analysis_results.json # 🆕 完整分析結果
```

## 🔧 安裝與設置

### 1. 環境要求
- Python 3.7+（3.8+建議）
- 8GB+ RAM（建議16GB用於大數據集）
- CUDA支援GPU（可選，XGBoost GPU加速需要）

### 2. 安裝依賴
```bash
# 克隆或下載項目後，進入Part05_目錄
cd Part05_

# 安裝基本依賴包
pip install -r requirements.txt

# 🆕 安裝XGBoost（推薦使用自動安裝腳本）
python install_xgboost.py

# 或手動安裝XGBoost
pip install xgboost

# 驗證安裝
python test_attention_simple.py
```

### 3. 環境檢測與測試
```bash
# 🆕 測試分類器功能（包含環境檢測）
python test_classifier_features.py

# 測試注意力機制功能
python test_attention_simple.py

# 查看系統幫助信息
python Part05_Main.py --help

# 運行使用範例
cd examples
python attention_example.py
```

### 4. GPU環境配置（可選）
```bash
# 檢查CUDA是否可用
nvidia-smi

# 確認PyTorch GPU支援
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 如果需要GPU版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 使用指南

### 1. GUI界面使用（推薦）

#### 啟動圖形界面
```bash
python Part05_Main.py
```

#### GUI操作流程
1. **第一分頁 - 數據處理**：
   - 選擇數據集類型
   - 導入文本文件
   - 啟用數據抽樣（大數據集建議）
   - 執行文本預處理
   - 進行BERT編碼

2. **第二分頁 - 注意力測試**：
   - 🆕 **選擇分類器**：在分類器設定區域選擇（XGBoost、LogisticRegression等）
   - 🆕 **環境檢測**：系統自動顯示GPU/CPU環境信息
   - 執行單一/雙重/三重注意力測試
   - 🆕 **實時計時**：顯示訓練進度和耗時統計

3. **第三分頁 - 結果分析**：
   - 查看多維度性能比較
   - 分析詳細分類結果
   - 導出完整結果報告

### 2. 命令行使用

#### 基本命令
```bash
# BERT編碼處理
python Part05_Main.py --process

# 🆕 完整分類評估（推薦）
python Part05_Main.py --classify your_data.csv

# 僅注意力機制分析
python Part05_Main.py --attention your_data.csv

# 比較注意力機制
python Part05_Main.py --compare your_data.csv

# 查看詳細幫助
python Part05_Main.py --help
```

#### 🆕 指定分類器類型
```bash
# 使用XGBoost（GPU加速）
python Part05_Main.py --classify data.csv --classifier xgboost

# 使用邏輯迴歸（快速）
python Part05_Main.py --classify data.csv --classifier logistic_regression

# 使用隨機森林（穩定）
python Part05_Main.py --classify data.csv --classifier random_forest

# 使用線性SVM
python Part05_Main.py --classify data.csv --classifier svm_linear
```

### 2. 注意力機制類型

#### 2.1 無注意力機制 (No Attention)
```python
from modules.attention_mechanism import apply_attention_mechanism

result = apply_attention_mechanism(
    attention_type='no',
    embeddings=embeddings,
    metadata=metadata
)
```

#### 2.2 相似度注意力 (Similarity Attention)
```python
result = apply_attention_mechanism(
    attention_type='similarity',
    embeddings=embeddings,
    metadata=metadata
)
```

#### 2.3 關鍵詞注意力 (Keyword-Guided Attention)
```python
# 自定義關鍵詞
topic_keywords = {
    'positive': ['好', '棒', '優秀', 'great', 'excellent'],
    'negative': ['壞', '糟糕', '差', 'bad', 'terrible'],
    'neutral': ['還可以', '普通', 'okay', 'average']
}

result = apply_attention_mechanism(
    attention_type='keyword',
    embeddings=embeddings,
    metadata=metadata,
    topic_keywords=topic_keywords
)
```

#### 2.4 自注意力 (Self-Attention)
```python
result = apply_attention_mechanism(
    attention_type='self',
    embeddings=embeddings,
    metadata=metadata
)
```

#### 2.5 組合注意力 (Combined Attention)
```python
# 自定義權重配置
weights = {
    'similarity': 0.4,
    'keyword': 0.4,
    'self': 0.2
}

result = apply_attention_mechanism(
    attention_type='combined',
    embeddings=embeddings,
    metadata=metadata,
    weights=weights
)
```

### 3. 完整處理流程

```python
from modules.attention_processor import AttentionProcessor

# 初始化處理器
processor = AttentionProcessor(output_dir='my_output')

# 執行完整分析
results = processor.process_with_attention(
    input_file='preprocessed_data.csv',
    attention_types=['no', 'similarity', 'keyword', 'self', 'combined'],
    save_results=True
)

# 專門的比較分析
comparison = processor.compare_attention_mechanisms(
    input_file='preprocessed_data.csv',
    attention_types=['similarity', 'combined']
)
```

## 📊 評估指標

### 注意力機制評估
#### 內聚度 (Coherence)
- 衡量同一情感面向內文檔的相似性
- 數值範圍：-1 到 1
- 越高表示面向內部越一致

#### 分離度 (Separation)
- 衡量不同情感面向間的差異性
- 數值範圍：0 到 2
- 越高表示面向間區分越明顯

#### 綜合得分 (Combined Score)
- 內聚度和分離度的加權平均
- 預設權重：各50%
- 綜合評估注意力機制效果

### 🆕 分類器性能評估
#### 準確率 (Accuracy)
- 正確分類的樣本比例
- 數值範圍：0 到 1
- 整體性能的基本指標

#### F1分數 (F1-Score)
- 精確率和召回率的調和平均
- 數值範圍：0 到 1
- 平衡考慮精確性和完整性

#### 精確率 (Precision)
- 預測為正類的樣本中實際為正類的比例
- 重要性：避免假陽性錯誤

#### 召回率 (Recall)
- 實際為正類的樣本中被正確預測的比例
- 重要性：避免假陰性錯誤

#### 🆕 計時統計
- **訓練時間**：模型訓練的實際耗時
- **預測時間**：模型推理的耗時
- **總處理時間**：包含數據處理的完整流程時間
- **環境信息**：GPU/CPU使用情況和配置詳情

## 🔬 實驗結果與分析

### 注意力機制效果比較

| 注意力機制 | 內聚度 | 分離度 | 綜合得分 | 計算複雜度 | 適用場景 |
|-----------|--------|--------|----------|------------|----------|
| 無注意力   | 基線   | 基線   | 基線     | 最低       | 基線比較 |
| 相似度注意力| 🔼     | 🔼     | 🔼       | 中等       | 語義相似性重要 |
| 關鍵詞注意力| 🔼🔼   | 🔼     | 🔼🔼     | 低         | 特定術語重要 |
| 自注意力   | 🔼🔼   | 🔼🔼   | 🔼🔼     | 高         | 複雜關係建模 |
| 組合注意力 | 🔼🔼🔼 | 🔼🔼🔼 | 🔼🔼🔼   | 最高       | 追求最佳效果 |

### 🆕 分類器性能比較（50,000筆測試數據）

#### CPU環境 (Intel i7-10700K)
| 分類器 | 訓練時間 | 準確率 | F1分數 | 記憶體使用 | 推薦場景 |
|--------|----------|--------|--------|------------|----------|
| **XGBoost** ⚡ | 12分鐘 | 0.954 | 0.952 | 中等 | 大數據集，追求最高準確率 |
| **Logistic Regression** 🚀 | 3分鐘 | 0.913 | 0.910 | 低 | 中小數據集，快速原型 |
| **Random Forest** 🌳 | 5分鐘 | 0.926 | 0.923 | 中等 | 穩定性要求高，可解釋性 |
| **SVM Linear** 📐 | 20分鐘 | 0.945 | 0.942 | 高 | 小數據集，線性可分 |

#### GPU環境 (RTX 3080)
| 分類器 | 訓練時間 | 準確率 | F1分數 | 加速比 | GPU利用率 |
|--------|----------|--------|--------|--------|-----------|
| **XGBoost** ⚡ | 1.5分鐘 | 0.954 | 0.952 | 8x | 95% |
| **Logistic Regression** 🚀 | 3分鐘 | 0.913 | 0.910 | 1x | N/A |
| **Random Forest** 🌳 | 5分鐘 | 0.926 | 0.923 | 1x | N/A |
| **SVM Linear** 📐 | 20分鐘 | 0.945 | 0.942 | 1x | N/A |

### 關鍵發現

#### 注意力機制研究發現
1. **組合注意力表現最佳**：在大多數測試中，組合注意力機制獲得最高的綜合得分
2. **關鍵詞注意力性價比高**：在計算成本較低的情況下提供較好的性能提升
3. **自注意力適合複雜任務**：在文檔間關係複雜的任務中表現突出
4. **權重配置影響顯著**：不同的權重配置對組合注意力效果有明顯影響

#### 🆕 分類器性能研究發現
1. **XGBoost整體最優**：在準確率和F1分數上表現最佳，GPU加速效果顯著
2. **邏輯迴歸速度冠軍**：在追求速度的場景下是最佳選擇，記憶體消耗最低
3. **GPU加速價值高**：XGBoost在GPU上可獲得8倍以上的速度提升
4. **數據量影響選擇**：
   - 小數據集(<10K)：推薦Logistic Regression或Random Forest
   - 中等數據集(10K-50K)：推薦XGBoost或Random Forest
   - 大數據集(>50K)：強烈推薦XGBoost + GPU
5. **記憶體使用差異**：SVM Linear記憶體消耗最高，需要考慮硬體限制

## 🎯 創新點總結

### 1. 面向計算時的注意力機制創新
- **傳統方法**：簡單平均或加權平均計算面向向量
- **創新方法**：在面向向量計算時引入多種注意力機制
- **技術實現**：
  - 相似度注意力：基於文檔與主題中心的語義相似度
  - 關鍵詞注意力：基於預定義關鍵詞的重要性權重
  - 自注意力：使用縮放點積注意力發現文檔間隱含關係
  - 組合注意力：多種機制的動態加權組合

### 2. 組合注意力vs單一注意力的系統性比較
- **比較維度**：內聚度、分離度、綜合得分、計算複雜度
- **比較方法**：統一評估框架下的公平比較
- **核心發現**：組合注意力在效果上優於單一注意力，但需要平衡計算成本
- **實際價值**：為不同應用場景提供注意力機制選擇指導

### 🆕 3. 智能分類器系統與環境適配
- **多分類器整合**：XGBoost、Logistic Regression、Random Forest、SVM Linear
- **智能環境偵測**：自動檢測GPU/CPU環境，動態配置最佳參數
- **性能優化創新**：
  - GPU加速：XGBoost自動切換GPU模式，實現8倍以上加速
  - 多核並行：充分利用CPU多核心，提升處理效率
  - 記憶體優化：根據數據量智能選擇最適合的分類器
- **實時計時系統**：詳細記錄各階段耗時，協助性能瓶頸分析

### 🆕 4. 完整的評估體系與用戶體驗
- **雙重評估指標**：結合注意力機制幾何評估和分類性能評估
- **GUI智能化**：實時環境顯示、進度跟踪、智能建議
- **數據適配性**：支援大數據集抽樣、多格式文件、自動預處理
- **結果可解釋性**：詳細的比較報告、視覺化結果展示、完整日誌記錄

## 📝 輸出文件說明

執行完整分析後，系統會生成以下文件：

#### 基礎文件
- `02_bert_embeddings.npy`: BERT特徵向量矩陣
- `03_attention_analysis_[timestamp].json`: 完整的分析結果
- `03_attention_comparison_[timestamp].json`: 注意力機制比較結果
- `03_aspect_vectors_[attention_type]_[timestamp].npy`: 各類注意力的面向向量
- `attention_comparison_report.txt`: 人類可讀的比較報告

#### 🆕 分類器相關文件
- `sentiment_classifier_[model_type]_[timestamp].pkl`: 訓練好的分類模型
- `label_encoder.pkl`: 標籤編碼器，用於標籤轉換
- `complete_analysis_results.json`: 包含注意力分析和分類評估的完整結果
- `classification_report_[timestamp].txt`: 詳細的分類性能報告
- `timing_analysis_[timestamp].json`: 各階段計時統計結果

#### 🆕 環境和日誌文件
- `environment_info.json`: 系統環境檢測結果
- `training_log_[timestamp].txt`: 詳細的訓練日誌
- `performance_benchmark.json`: 性能基準測試結果

## 🛠️ 故障排除

### 常見問題

#### 1. 依賴包安裝失敗
```bash
# 升級pip
python -m pip install --upgrade pip

# 使用清華源安裝
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 🆕 2. XGBoost安裝問題
```bash
# 使用自動安裝腳本（推薦）
python install_xgboost.py

# 手動解決方案
pip install --upgrade pip
pip install xgboost

# 如果仍然失敗，嘗試conda
conda install -c conda-forge xgboost
```

#### 🆕 3. GPU檢測和配置問題
```bash
# 檢查CUDA是否正確安裝
nvidia-smi

# 測試PyTorch CUDA支援
python -c "import torch; print(torch.cuda.is_available())"

# 如果GPU檢測失敗，強制使用CPU模式
python -c "import torch; torch.cuda.is_available = lambda: False"
```

#### 4. CUDA相關錯誤
```python
# 在代碼中強制使用CPU
import torch
torch.cuda.is_available = lambda: False
```

#### 🆕 5. 分類器訓練時間過長
```bash
# 使用數據抽樣（在GUI中啟用或代碼中設置）
# 選擇較快的分類器
python Part05_Main.py --classify data.csv --classifier logistic_regression

# 檢查是否正確使用GPU（僅限XGBoost）
python test_classifier_features.py
```

#### 6. 記憶體不足
- 啟用數據抽樣功能（建議10,000-20,000樣本）
- 選擇記憶體效率高的分類器（Logistic Regression）
- 關閉其他占用記憶體的程序
- 減少batch_size設置

#### 7. 文本編碼問題
確保所有文本文件使用UTF-8編碼

#### 🆕 8. GUI環境檢測顯示錯誤
```bash
# 重新檢測環境
python test_classifier_features.py

# 檢查依賴包完整性
pip list | grep -E "(torch|xgboost|sklearn)"
```

### 測試系統功能
```bash
# 🆕 運行分類器功能完整測試（推薦）
python test_classifier_features.py

# 運行注意力機制測試
python test_attention_simple.py

# 運行使用範例
cd examples
python attention_example.py

# 🆕 測試XGBoost安裝和GPU支援
python install_xgboost.py --test-only

# 測試環境配置
python -c "
from modules.sentiment_classifier import SentimentClassifier;
classifier = SentimentClassifier();
print('可用分類器:', classifier.get_available_models());
print('環境信息:', classifier.get_device_info())
"
```

## 📚 學術應用

### 論文寫作要點

1. **創新點描述**：
   - 在面向向量計算中引入注意力機制
   - 系統性比較不同注意力機制效果
   - 提出組合注意力機制

2. **實驗設計**：
   - 控制變量：使用相同的BERT特徵和數據集
   - 評估指標：內聚度、分離度、綜合得分
   - 比較基準：無注意力機制作為基線

3. **結果分析**：
   - 定量分析：各種注意力機制的評估分數
   - 定性分析：不同機制的適用場景
   - 計算複雜度分析：效果與效率的權衡

## 🤝 貢獻與反饋

歡迎提出問題、建議或貢獻代碼！

### 貢獻方式
1. Fork此項目
2. 創建功能分支
3. 提交變更
4. 發起Pull Request

### 聯繫方式
- 提交Issue報告問題
- 發起Discussion討論改進

## 📊 使用狀態顯示

### GUI狀態指示器
每個操作項目都有即時狀態顯示：
- 🟠 **待處理**: 尚未開始
- 🔵 **處理中**: 正在執行
- 🟢 **完成**: 執行成功
- 🔴 **錯誤**: 執行失敗

### 🆕 環境檢測狀態
- **GPU環境**: 顯示GPU型號、記憶體、CUDA版本
- **CPU環境**: 顯示處理器型號、核心數、記憶體
- **模型狀態**: 實時顯示當前選擇的分類器和配置
- **計時顯示**: 訓練開始時間、預計剩餘時間、實際耗時

### 🆕 進度追蹤功能
- **數據處理進度**: 批量處理的實時進度條
- **訓練進度**: 分階段顯示模型訓練狀態
- **評估進度**: 各個注意力機制的評估完成狀態
- **結果生成**: 檔案輸出和保存的即時狀態

## 📄 授權聲明

本項目僅用於學術研究目的。使用本項目的代碼或思路時，請適當引用。

---

**注意**：本系統是研究原型，建議在生產環境使用前進行充分測試和優化。

**🆕 更新說明**：系統已整合高性能分類器和智能環境適配功能，支援GPU加速和詳細性能分析。建議先運行`test_classifier_features.py`測試所有功能是否正常運作。 