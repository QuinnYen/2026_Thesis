# -*- coding: utf-8 -*-
"""
GUI配置檔案
"""

# 視窗設定
WINDOW_TITLE = "情感分析系統"
WINDOW_SIZE = "900x680"
WINDOW_MIN_SIZE = (800, 620)

# 顏色設定
COLORS = {
    'pending': 'orange',    # 待處理
    'processing': 'blue',   # 處理中
    'success': 'green',     # 成功
    'error': 'red',         # 錯誤
    'info': 'black',        # 一般資訊
    'warning': '#FF8C00'    # 警告
}

# 狀態文字
STATUS_TEXT = {
    'pending': '狀態: 待處理',
    'processing': '狀態: 處理中...',
    'success': '狀態: 完成',
    'error': '狀態: 錯誤',
    'file_selected': '狀態: 檔案已選擇',
    'encoding_processing': '狀態: 編碼中...',
    'encoding_complete': '狀態: 編碼完成',
    'analysis_pending': '狀態: 待分析',
    'analysis_processing': '狀態: 分析中...',
    'analysis_complete': '狀態: 分析完成',
    # 預處理狀態
    'cleaning': '狀態: 文本清理中...',
    'tokenizing': '狀態: 分詞處理中...',
    'normalizing': '狀態: 標準化處理中...',
    'removing_tags': '狀態: 移除標籤中...',
    'preprocessing_complete': '狀態: 預處理完成'
}

# 數據集設定
DATASETS = {
    'IMDB': {
        'name': 'IMDB影評數據集',
        'file_type': 'csv',
        'description': 'IMDB電影評論數據集'
    },
    'Yelp': {
        'name': 'Yelp評論數據集',
        'file_type': 'csv',
        'description': 'Yelp商家評論數據集'
    },
    'Amazon': {
        'name': 'Amazon評論數據集',
        'file_type': 'json',
        'description': 'Amazon商品評論數據集'
    }
}

# 支援的檔案類型
SUPPORTED_FILE_TYPES = [
    ("CSV檔案 (IMDB/Yelp)", "*.csv"),
    ("JSON檔案 (Amazon)", "*.json"),
    ("所有檔案", "*.*")
]

# 字體設定
FONTS = {
    'title': ('Arial', 16, 'bold'),
    'subtitle': ('Arial', 12, 'bold'),
    'normal': ('Arial', 10),
    'small': ('Arial', 8)
}

# 間距設定
PADDING = {
    'large': 20,
    'medium': 15,
    'small': 10,
    'tiny': 5
}

# 模擬處理時間（毫秒）
SIMULATION_DELAYS = {
    'file_processing': 2000,
    'bert_encoding': 3000,
    'attention_test': 2000,
    'analysis': 3000
}

# 預處理步驟設定
PREPROCESSING_STEPS = {
    'text_cleaning': {
        'name': '文本清理',
        'description': '移除特殊字符、多餘空格等',
        'delay': 500  # 模擬處理時間（毫秒）
    },
    'tokenization': {
        'name': '分詞處理',
        'description': '將文本分割為單詞/字元',
        'delay': 800
    },
    'normalization': {
        'name': '標準化處理',
        'description': '統一大小寫、標點符號等',
        'delay': 600
    },
    'tag_removal': {
        'name': '移除標籤',
        'description': '移除HTML標籤等標記',
        'delay': 400
    }
} 