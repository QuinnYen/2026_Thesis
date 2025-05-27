#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT情感分析系統 - GUI啟動腳本
"""

import sys
import os

# 添加當前目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from gui.main_window import main
    
    if __name__ == "__main__":
        print("正在啟動BERT情感分析系統...")
        main()
        
except ImportError as e:
    print(f"導入錯誤: {e}")
    print("請確保所有必要的模組都已安裝")
except Exception as e:
    print(f"啟動錯誤: {e}")
    input("按Enter鍵退出...") 