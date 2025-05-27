# 注意力機制範例與使用指南

本目錄包含了BERT情感分析系統中注意力機制的使用範例和說明。

## 文件說明

### `attention_example.py`
完整的注意力機制使用範例，展示了以下功能：
- 單個注意力機制的使用
- 組合注意力機制的配置
- 關鍵詞注意力機制的應用
- 完整的注意力處理器功能

## 快速開始

### 1. 運行完整範例
```bash
cd Part05_/examples
python attention_example.py
```

### 2. 使用主程式進行注意力分析
```bash
cd Part05_
python Part05_Main.py --attention your_data.csv
```

### 3. 比較不同注意力機制
```bash
cd Part05_
python Part05_Main.py --compare your_data.csv
```

## 注意力機制類型

### 1. 無注意力機制 (No Attention)
- 類型: `'no'` 或 `'none'`
- 描述: 使用平均權重，相當於傳統的平均池化
- 適用場景: 基線比較、計算資源有限的情況

### 2. 相似度注意力 (Similarity Attention)
- 類型: `'similarity'`
- 描述: 基於文檔與主題中心向量的相似度計算權重
- 適用場景: 語義相似性重要的任務

### 3. 關鍵詞注意力 (Keyword-Guided Attention)
- 類型: `'keyword'`
- 描述: 基於預定義關鍵詞出現頻率計算權重
- 適用場景: 特定領域術語重要的任務

### 4. 自注意力 (Self-Attention)
- 類型: `'self'`
- 描述: 使用縮放點積注意力機制
- 適用場景: 文檔間關係複雜的任務

### 5. 組合注意力 (Combined Attention)
- 類型: `'combined'`
- 描述: 多種注意力機制的加權組合
- 適用場景: 追求最佳性能的生產環境

## 使用範例

### 基本使用
```python
from modules.attention_mechanism import apply_attention_mechanism
import pandas as pd
import numpy as np

# 準備數據
df = pd.DataFrame({
    'text': ['positive text', 'negative text'],
    'sentiment': ['positive', 'negative']
})
embeddings = np.random.randn(2, 768)  # BERT特徵向量

# 應用注意力機制
result = apply_attention_mechanism(
    attention_type='similarity',
    embeddings=embeddings,
    metadata=df
)

print(f"內聚度: {result['metrics']['coherence']:.4f}")
print(f"分離度: {result['metrics']['separation']:.4f}")
```

### 使用處理器進行完整分析
```python
from modules.attention_processor import AttentionProcessor

processor = AttentionProcessor(output_dir='output')
results = processor.process_with_attention(
    input_file='data.csv',
    attention_types=['similarity', 'keyword', 'combined'],
    save_results=True
)
```

### 組合注意力配置
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
    metadata=df,
    weights=weights
)
```

### 關鍵詞注意力配置
```python
# 定義關鍵詞
topic_keywords = {
    'positive': ['好', '棒', '優秀', 'great', 'excellent'],
    'negative': ['壞', '糟糕', '差', 'bad', 'terrible'],
    'neutral': ['還可以', '普通', 'okay', 'average']
}

result = apply_attention_mechanism(
    attention_type='keyword',
    embeddings=embeddings,
    metadata=df,
    topic_keywords=topic_keywords
)
```

## 評估指標

### 內聚度 (Coherence)
- 衡量同一面向內文檔的相似性
- 數值越高表示面向內部越一致

### 分離度 (Separation)
- 衡量不同面向間的差異性
- 數值越高表示面向間區分越明顯

### 綜合得分 (Combined Score)
- 內聚度和分離度的加權平均
- 綜合評估注意力機制的整體效果

## 輸出文件說明

執行注意力分析後，會在輸出目錄生成以下文件：

- `02_bert_embeddings.npy`: BERT特徵向量
- `03_attention_analysis_[timestamp].json`: 完整分析結果
- `03_attention_comparison_[timestamp].json`: 比較結果
- `03_aspect_vectors_[attention_type]_[timestamp].npy`: 各類注意力的面向向量
- `attention_comparison_report.txt`: 可讀的比較報告

## 配置選項

### 注意力機制配置
```python
config = {
    'coherence_weight': 0.5,      # 內聚度權重
    'separation_weight': 0.5,     # 分離度權重
}
```

### 組合注意力權重
```python
weights = {
    'similarity': 0.33,           # 相似度注意力權重
    'keyword': 0.33,              # 關鍵詞注意力權重
    'self': 0.34                  # 自注意力權重
}
```

## 常見問題

### Q: 如何選擇合適的注意力機制？
A: 建議先運行比較分析，根據您的數據特點和任務需求選擇：
- 語義相似性重要：選擇相似度注意力
- 特定術語重要：選擇關鍵詞注意力
- 追求最佳效果：選擇組合注意力

### Q: 組合注意力的權重如何設定？
A: 可以嘗試不同的權重配置，觀察評估指標的變化：
- 均衡配置：各機制權重相等
- 基於數據特點調整：例如關鍵詞豐富的數據可增加關鍵詞權重

### Q: 如何提供自定義關鍵詞？
A: 創建包含關鍵詞的JSON文件或直接在代碼中定義字典，格式如下：
```json
{
  "positive": ["好", "棒", "優秀"],
  "negative": ["壞", "糟糕", "差"],
  "neutral": ["還可以", "普通"]
}
```

## 性能考慮

- **無注意力機制**: 最快，適合大規模數據
- **相似度注意力**: 中等速度，平衡效果與效率
- **關鍵詞注意力**: 依賴關鍵詞數量，通常較快
- **自注意力**: 較慢，但效果通常較好
- **組合注意力**: 最慢，但通常效果最佳

建議在開發階段使用組合注意力找到最佳配置，在生產環境中根據性能需求選擇合適的單一機制。 