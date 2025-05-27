# -*- coding: utf-8 -*-
"""
GUI配置檔案
"""

# 視窗設定
WINDOW_TITLE = "BERT情感分析系統"
WINDOW_SIZE = "900x650"
WINDOW_MIN_SIZE = (800, 600)

# 顏色設定
COLORS = {
    'pending': 'orange',    # 待處理
    'processing': 'blue',   # 處理中
    'success': 'green',     # 成功
    'error': 'red',         # 錯誤
    'info': 'black'         # 一般資訊
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
    'analysis_complete': '狀態: 分析完成'
}

# 支援的檔案類型
SUPPORTED_FILE_TYPES = [
    ("Text files", "*.txt"),
    ("CSV files", "*.csv"),
    ("JSON files", "*.json"),
    ("All files", "*.*")
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