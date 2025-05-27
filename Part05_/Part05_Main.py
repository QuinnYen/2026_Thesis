#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT情感分析系統 - 主程式
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional

# 添加當前目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from modules.run_manager import RunManager

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化RunManager
run_manager = RunManager(current_dir)

def process_bert_encoding(input_file: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """
    使用BERT模型處理文本並提取特徵向量
    
    Args:
        input_file: 輸入文件路徑，如果為None則使用預設路徑
        output_dir: 輸出目錄路徑，如果為None則自動生成
        
    Returns:
        str: 輸出目錄路徑
    """
    from modules.bert_encoder import BertEncoder
    
    try:
        # 初始化BERT編碼器，傳入輸出目錄
        encoder = BertEncoder(output_dir=output_dir)
        
        # 檢查輸入文件
        if input_file is None or not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到輸入文件：{input_file}")
        
        # 讀取預處理後的數據
        logger.info(f"讀取數據: {input_file}")
        df = pd.read_csv(input_file)
        
        # 檢查必要的欄位，按優先順序排列
        required_columns = ['processed_text', 'clean_text', 'text', 'review']
        text_column = None
        for col in required_columns:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            available_columns = ', '.join(df.columns)
            raise ValueError(f"在輸入文件中找不到文本欄位（優先順序：processed_text > clean_text > text > review）。可用的欄位有：{available_columns}")
        
        # 對文本進行編碼
        logger.info(f"開始BERT編碼...使用欄位：{text_column}")
        embeddings = encoder.encode(df[text_column])
        
        # 保存特徵向量
        logger.info("保存特徵向量...")
        encoder.save_embeddings(embeddings, "02_bert_embeddings.npy")
        
        logger.info(f"處理完成！結果保存在: {encoder.output_dir}")
        return encoder.output_dir
        
    except Exception as e:
        logger.error(f"BERT編碼過程中發生錯誤: {str(e)}")
        raise

def main():
    """
    主程式入口點
    """
    try:
        # 檢查是否有命令行參數
        if len(sys.argv) > 1:
            if sys.argv[1] == '--process':
                # 執行BERT編碼處理
                process_bert_encoding()
            elif sys.argv[1] == '--new-run':
                # 清除上次執行的記錄，強制創建新的run目錄
                pass  # 已移除clear_last_run，保留佔位
        else:
            # 啟動GUI
            from gui.main_window import main as gui_main
            print("正在啟動BERT情感分析系統...")
            gui_main()
            
    except ImportError as e:
        print(f"導入錯誤: {e}")
        print("請確保所有必要的模組都已安裝")
    except Exception as e:
        print(f"執行錯誤: {e}")
        input("按Enter鍵退出...")

if __name__ == "__main__":
    main() 