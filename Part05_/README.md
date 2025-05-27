# BERT情感分析系統 - GUI介面

## 概述
這是一個基於BERT的情感分析系統，提供圖形化使用者介面來進行文本處理、注意力機制測試和比對分析。

## 系統架構
系統分為三個主要分頁：

### 第一分頁 - 資料處理
- **文本輸入 → 導入檔案**: 選擇並預覽文本檔案
- **文本處理 → 開始處理**: 對文本進行預處理
- **Bert編碼 → 開始編碼**: 使用BERT模型進行文本編碼

### 第二分頁 - 注意力機制測試
- **基準（單一）**: 測試單一注意力機制
  - 相似度
  - 關鍵詞
  - 目標
- **組合（雙頭）**: 測試雙頭注意力機制
  - 相似度 + 關鍵詞
  - 相似度 + 目標
  - 關鍵詞 + 目標
- **組合（三頭）**: 測試三頭注意力機制
  - 相似度 + 關鍵詞 + 目標

### 第三分頁 - 比對分析
- 比較不同注意力機制的性能
- 顯示準確率、F1分數、召回率、精確率
- 提供詳細的分析報告

## 使用方法

### 啟動系統
```bash
cd Part05_
python run_gui.py
```

### 操作流程
1. **資料準備**: 在第一分頁選擇文本檔案並進行處理
2. **模型測試**: 在第二分頁執行不同的注意力機制測試
3. **結果分析**: 在第三分頁查看比對分析結果

## 狀態顯示
每個操作項目都有狀態顯示：
- 🟠 **待處理**: 尚未開始
- 🔵 **處理中**: 正在執行
- 🟢 **完成**: 執行成功
- 🔴 **錯誤**: 執行失敗

## 檔案結構
```
Part05_/
├── gui/
│   ├── main_window.py      # 主要GUI程式
│   └── config.py          # 配置檔案
├── utils/                  # 工具模組（待開發）
├── output/                 # 輸出結果
├── run_gui.py             # 啟動腳本
├── sample_data.txt        # 範例數據
└── README.md              # 說明文件
```

## 特色功能
- 視窗自動置中於螢幕（900x650像素）
- 簡潔的三分頁設計，緊湊佈局
- 即時狀態顯示
- 支援多種檔案格式：.txt, .csv, .json
- 響應式設計，最小視窗尺寸800x600

## 注意事項
- 目前為UI框架，實際的BERT處理功能需要後續開發
- 建議使用UTF-8編碼的文本檔案

## 後續開發
- [ ] 整合實際的BERT模型
- [ ] 實現注意力機制算法
- [ ] 添加更多的評估指標
- [ ] 支援更多檔案格式
- [ ] 添加結果匯出功能 

# BERT情感分析系統 - Part05：注意力機制實現

本項目實現了一個完整的BERT情感分析系統，並整合了多種注意力機制用於情感面向建模。

## 🌟 主要特性

### 🔥 創新的注意力機制實現
1. **多種注意力機制**：支援無注意力、相似度注意力、關鍵詞注意力、自注意力和組合注意力
2. **面向計算創新**：在面向向量計算時引入注意力機制，提升情感分析準確性
3. **組合vs單一比較**：系統性比較不同注意力機制的效果
4. **動態權重調整**：支援自定義組合注意力的權重配置

### 🚀 核心功能
- **BERT特徵提取**：使用預訓練BERT模型提取768維文本特徵
- **注意力機制分析**：支援5種不同類型的注意力機制
- **效果評估比較**：提供內聚度、分離度和綜合得分評估
- **完整的處理流程**：從數據預處理到結果分析的完整pipeline
- **GUI界面**：友好的圖形用戶界面
- **命令行支援**：靈活的命令行操作

## 📁 項目結構

```
Part05_/
├── Part05_Main.py              # 主程式入口
├── requirements.txt            # 依賴包列表
├── test_attention_simple.py    # 簡化測試腳本
├── README.md                   # 項目說明
│
├── modules/                    # 核心模組
│   ├── attention_mechanism.py  # 注意力機制實現
│   ├── attention_analyzer.py   # 注意力分析器
│   ├── attention_processor.py  # 注意力處理器
│   ├── bert_encoder.py         # BERT編碼器
│   ├── run_manager.py          # 運行管理器
│   └── text_preprocessor.py    # 文本預處理器
│
├── utils/                      # 工具文件
│   └── topic_labels.json       # 主題標籤配置
│
├── gui/                        # 圖形界面
│   └── main_window.py          # 主視窗
│
├── examples/                   # 使用範例
│   ├── attention_example.py    # 注意力機制範例
│   └── README.md               # 範例說明
│
└── output/                     # 輸出目錄（自動生成）
    ├── 02_bert_embeddings.npy
    ├── 03_attention_analysis_*.json
    └── 03_attention_comparison_*.json
```

## 🔧 安裝與設置

### 1. 環境要求
- Python 3.8+
- 8GB+ RAM（建議）
- CUDA支援GPU（可選，用於加速）

### 2. 安裝依賴
```bash
# 克隆或下載項目後，進入Part05_目錄
cd Part05_

# 安裝依賴包
pip install -r requirements.txt

# 驗證安裝
python test_attention_simple.py
```

### 3. 快速測試
```bash
# 測試系統功能
python test_attention_simple.py

# 查看幫助信息
python Part05_Main.py --help

# 運行範例
cd examples
python attention_example.py
```

## 🚀 使用指南

### 1. 基本使用

#### 啟動GUI界面
```bash
python Part05_Main.py
```

#### 命令行使用
```bash
# BERT編碼處理
python Part05_Main.py --process

# 注意力機制分析
python Part05_Main.py --attention your_data.csv

# 比較注意力機制
python Part05_Main.py --compare your_data.csv

# 查看幫助
python Part05_Main.py --help
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

### 內聚度 (Coherence)
- 衡量同一情感面向內文檔的相似性
- 數值範圍：-1 到 1
- 越高表示面向內部越一致

### 分離度 (Separation)
- 衡量不同情感面向間的差異性
- 數值範圍：0 到 2
- 越高表示面向間區分越明顯

### 綜合得分 (Combined Score)
- 內聚度和分離度的加權平均
- 預設權重：各50%
- 綜合評估注意力機制效果

## 🔬 實驗結果與分析

### 注意力機制效果比較

| 注意力機制 | 內聚度 | 分離度 | 綜合得分 | 計算複雜度 | 適用場景 |
|-----------|--------|--------|----------|------------|----------|
| 無注意力   | 基線   | 基線   | 基線     | 最低       | 基線比較 |
| 相似度注意力| 🔼     | 🔼     | 🔼       | 中等       | 語義相似性重要 |
| 關鍵詞注意力| 🔼🔼   | 🔼     | 🔼🔼     | 低         | 特定術語重要 |
| 自注意力   | 🔼🔼   | 🔼🔼   | 🔼🔼     | 高         | 複雜關係建模 |
| 組合注意力 | 🔼🔼🔼 | 🔼🔼🔼 | 🔼🔼🔼   | 最高       | 追求最佳效果 |

### 關鍵發現

1. **組合注意力表現最佳**：在大多數測試中，組合注意力機制獲得最高的綜合得分
2. **關鍵詞注意力性價比高**：在計算成本較低的情況下提供較好的性能提升
3. **自注意力適合複雜任務**：在文檔間關係複雜的任務中表現突出
4. **權重配置影響顯著**：不同的權重配置對組合注意力效果有明顯影響

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

## 📝 輸出文件說明

執行注意力分析後，系統會生成以下文件：

- `02_bert_embeddings.npy`: BERT特徵向量矩陣
- `03_attention_analysis_[timestamp].json`: 完整的分析結果
- `03_attention_comparison_[timestamp].json`: 注意力機制比較結果
- `03_aspect_vectors_[attention_type]_[timestamp].npy`: 各類注意力的面向向量
- `attention_comparison_report.txt`: 人類可讀的比較報告

## 🛠️ 故障排除

### 常見問題

#### 1. 依賴包安裝失敗
```bash
# 升級pip
python -m pip install --upgrade pip

# 使用清華源安裝
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 2. CUDA相關錯誤
```python
# 在代碼中強制使用CPU
import torch
torch.cuda.is_available = lambda: False
```

#### 3. 記憶體不足
- 減少batch_size
- 使用較小的模型
- 分批處理數據

#### 4. 文本編碼問題
確保所有文本文件使用UTF-8編碼

### 測試系統功能
```bash
# 運行完整測試
python test_attention_simple.py

# 運行範例
cd examples
python attention_example.py
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

## 📄 授權聲明

本項目僅用於學術研究目的。使用本項目的代碼或思路時，請適當引用。

---

**注意**：本系統是研究原型，建議在生產環境使用前進行充分測試和優化。 