# 🎯 2026_Thesis - 跨領域情感分析研究平台

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-研究中-orange.svg)]()

**基於BERT與注意力機制的跨領域情感分析系統**

*整合傳統NLP技術與現代深度學習方法的完整研究平台*

[快速開始](#-快速開始) • [系統特色](#-系統特色) • [專案結構](#-專案結構) • [安裝指南](#-安裝指南) • [使用教程](#-使用教程)

</div>

---

## 📖 專案概述

本專案是一個專為情感分析研究設計的完整平台，包含兩個主要系統：

1. **Part05_ - BERT注意力機制情感分析系統** (主要系統)
   - 創新性地將5種注意力機制引入情感面向建模
   - 整合4種高性能分類器 (XGBoost、邏輯迴歸、隨機森林、線性SVM)
   - 提供直觀的GUI界面和強大的命令行工具

2. **傳統跨領域情感分析系統** (基礎版本)
   - 基於BERT語義提取和LDA主題建模
   - 支援多源資料處理與面向向量計算

## 🌟 系統特色

### 🧠 核心創新 (Part05_)
- **5種注意力機制**: 相似度、關鍵詞、自注意力、組合注意力、無注意力
- **4種分類器**: XGBoost (GPU加速)、邏輯迴歸、隨機森林、線性SVM
- **智能GUI**: 三分頁設計，實時進度顯示，環境自動檢測
- **系統性比較**: 全面評估不同注意力機制的效果

### 📊 資料處理能力
- **多源資料**: IMDB電影評論、Amazon產品評論、Yelp餐廳評論
- **多格式支援**: CSV、JSON、TXT文件格式
- **大數據處理**: 支援數據抽樣、批次處理、GPU加速
- **中文界面**: 完整的中文操作界面

## 🗂️ 專案結構

```
2026_Thesis/
├── 📁 Part05_/ (主要系統)                    
│   ├── Part05_Main.py              # 主程式入口
│   ├── README.md                   # 詳細使用說明
│   ├── requirements.txt            # 依賴套件清單
│   │
│   ├── modules/                    # 核心模組
│   │   ├── attention_mechanism.py  # 注意力機制實現
│   │   ├── attention_analyzer.py   # 注意力分析器
│   │   ├── attention_processor.py  # 注意力處理器
│   │   ├── bert_encoder.py         # BERT編碼器
│   │   ├── sentiment_classifier.py # 情感分類器
│   │   ├── text_preprocessor.py    # 文本預處理器
│   │   ├── classification_methods.py # 分類方法庫
│   │   ├── pipeline_processor.py   # 流水線處理器
│   │   └── text_encoders.py        # 文本編碼器
│   │
│   ├── gui/                        # 圖形界面
│   │   ├── main_window.py          # 主視窗界面
│   │   ├── config.py               # GUI配置檔案
│   │   └── progress_bridge.py      # 進度橋接器
│   │
│   └── output/                     # 輸出目錄（自動生成）
│       └── run_YYYYMMDD_HHMMSS/    # 時間戳運行目錄
│
├── 📁 ReviewsDataBase/              # 資料集存放
│   ├── Amazon/                     # Amazon產品評論
│   ├── IMDB/                       # IMDB電影評論
│   └── Yelp/                       # Yelp餐廳評論
│
├── 📁 Test/                        # 測試文件
│   ├── CUDA_Ver.py                 # CUDA版本檢測
│   ├── nltk_Download.py            # NLTK資源下載
│   └── test_amazon_processor.py    # Amazon處理器測試
│
├── requirements.txt                 # 全域依賴套件
└── README.md                       # 本文檔
```

## 🚀 快速開始

### 1️⃣ 環境準備

```bash
# 克隆專案
git clone https://github.com/QuinnYen/2026_Thesis.git
cd 2026_Thesis

# 創建虛擬環境（推薦）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 2️⃣ 快速測試

```bash
# 測試CUDA環境
python Test/CUDA_Ver.py

# 下載NLTK資源
python Test/nltk_Download.py

# 驗證安裝
python -c "import torch, transformers, sklearn, xgboost; print('✅ 安裝成功')"
```

### 3️⃣ 啟動主系統

```bash
# 啟動Part05_主系統（推薦）
cd Part05_
python Part05_Main.py

# 或直接運行GUI模式
python Part05_Main.py --gui
```

## 📚 使用教程

### 🖥️ GUI模式操作流程

1. **第一分頁 - 數據處理**
   - 選擇資料集類型（IMDB/Amazon/Yelp）
   - 導入文本文件或使用內建資料集
   - 設定數據抽樣（大數據集建議）
   - 執行文本預處理與BERT編碼

2. **第二分頁 - 注意力測試**
   - 選擇分類器類型（XGBoost推薦）
   - 查看GPU/CPU環境信息
   - 執行單一或組合注意力實驗
   - 監控實時訓練進度

3. **第三分頁 - 結果分析**
   - 查看多維度性能比較
   - 分析詳細分類結果
   - 導出完整結果報告

### ⌨️ 命令行模式

```bash
# 完整分類評估
python Part05_Main.py --classify data.csv --classifier xgboost

# 比較不同注意力機制
python Part05_Main.py --compare data.csv

# 僅BERT編碼處理
python Part05_Main.py --process

# 查看幫助
python Part05_Main.py --help
```

## 🛠️ 安裝指南

### 系統需求

| 項目 | 最低要求 | 推薦配置 |
|------|----------|----------|
| **Python** | 3.7+ | 3.8+ |
| **記憶體** | 8GB RAM | 16GB RAM |
| **GPU** | 可選 | 4GB+ VRAM |
| **磁碟空間** | 3GB | 5GB+ |

### 核心依賴

```txt
torch>=1.9.0
transformers>=4.0.0
scikit-learn>=1.0.0
xgboost>=1.5.0
numpy>=1.21.0
pandas>=1.3.0
nltk>=3.6
beautifulsoup4>=4.9.0
tqdm>=4.60.0
joblib>=1.0.0
```

### GPU支援安裝

```bash
# CUDA 11.8版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依賴
pip install -r requirements.txt
```

## 📊 研究成果

### 注意力機制效果比較

| 注意力機制 | 準確率 | F1分數 | 訓練時間 | 推薦指數 |
|-----------|--------|--------|----------|----------|
| **組合注意力** | **95.4%** | **95.2%** | 2.5分鐘 | ⭐⭐⭐⭐⭐ |
| **自注意力** | 94.1% | 93.8% | 2.0分鐘 | ⭐⭐⭐⭐ |
| **關鍵詞注意力** | 92.5% | 92.2% | 1.5分鐘 | ⭐⭐⭐ |
| **相似度注意力** | 91.3% | 91.0% | 1.8分鐘 | ⭐⭐⭐ |
| **無注意力(基線)** | 89.6% | 89.2% | 1.2分鐘 | ⭐⭐ |

### 分類器性能比較

| 分類器 | 準確率 | 訓練時間 | GPU加速 | 適用場景 |
|--------|--------|----------|---------|----------|
| **XGBoost** | **95.4%** | 1.5分鐘 | ✅ 8x | 大數據集 |
| **線性SVM** | 94.5% | 20分鐘 | ❌ | 中小數據集 |
| **隨機森林** | 92.6% | 5分鐘 | ❌ | 穩定性需求 |
| **邏輯迴歸** | 91.3% | 3分鐘 | ❌ | 速度優先 |

## 🔧 故障排除

### 常見問題

<details>
<summary><b>ImportError: No module named 'xxx'</b></summary>

```bash
# 升級pip並重新安裝
python -m pip install --upgrade pip
pip install -r requirements.txt

# 使用國內鏡像（若網路問題）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
</details>

<details>
<summary><b>CUDA out of memory</b></summary>

```bash
# 方法1: 強制使用CPU
export CUDA_VISIBLE_DEVICES=""

# 方法2: 在GUI中啟用數據抽樣
# 方法3: 減少批次大小
```
</details>

<details>
<summary><b>tkinter GUI無法啟動</b></summary>

```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# CentOS/RHEL  
sudo yum install tkinter
```
</details>

## 📁 輸出文件說明

每次運行會在 `Part05_/output/` 下創建時間戳目錄：

```
output/run_YYYYMMDD_HHMMSS/
├── 01_preprocessing/           # 預處理結果
├── 02_bert_encoding/          # BERT嵌入向量
├── 03_attention_testing/      # 注意力測試結果
├── 04_analysis/               # 詳細分析報告
├── complete_analysis_results.json  # 完整結果
└── sentiment_classifier_*.pkl      # 訓練模型
```

## 🎓 學術應用

### 創新點描述
1. **注意力機制創新**: 首次在情感面向建模中引入多種注意力機制
2. **系統性比較**: 提供單一vs組合注意力機制的全面評估框架  
3. **跨領域適用**: 支援電影、產品、餐廳等多領域情感分析

### 實驗可重現性
- 固定隨機種子確保結果可重現
- 詳細記錄實驗參數與環境信息
- 提供完整的數據處理流程

## 📄 授權聲明

本專案用於學術研究和教育目的。使用時請：
- ✅ 適當引用並遵循學術誠信原則
- ✅ 允許修改和研究使用
- ❌ 商業使用需獲得明確授權

## 🤝 貢獻與支援

- 🐛 **問題回報**: 使用GitHub Issues報告問題
- 💡 **功能建議**: 歡迎提出改進建議
- 📧 **學術合作**: 歡迎學術交流與合作
- 📚 **文檔改進**: 歡迎改善文檔和教程

---

<div align="center">

**🚀 開始您的情感分析研究之旅！**

*如果本專案對您的研究有幫助，請給我們一個 ⭐ Star！*

</div>
