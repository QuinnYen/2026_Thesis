#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT情感分析系統 - 主程式
整合注意力機制分析功能
"""

import sys
import os
import logging
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict
import numpy as np

# 添加當前目錄到Python路徑，並定義為全域變數
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from modules.run_manager import RunManager
from modules.attention_processor import AttentionProcessor
from modules.sentiment_classifier import SentimentClassifier

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化RunManager
run_manager = RunManager(CURRENT_DIR)

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
        # 如果沒有指定輸出目錄，使用預設目錄
        if output_dir is None:
            output_dir = os.path.join(CURRENT_DIR, "output")
            
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

def process_attention_analysis(input_file: Optional[str] = None, 
                             output_dir: Optional[str] = None,
                             attention_types: Optional[List[str]] = None,
                             topics_path: Optional[str] = None,
                             attention_weights: Optional[Dict] = None) -> Dict:
    """
    執行注意力機制分析（原有功能，保持向後兼容性）
    
    Args:
        input_file: 輸入文件路徑
        output_dir: 輸出目錄路徑
        attention_types: 要測試的注意力機制類型
        topics_path: 關鍵詞文件路徑
        attention_weights: 組合注意力權重配置
        
    Returns:
        Dict: 分析結果
    """
    try:
        # 初始化注意力處理器
        processor = AttentionProcessor(output_dir=output_dir)
        
        # 檢查輸入文件
        if input_file is None or not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到輸入文件：{input_file}")
        
        # 設定預設的注意力機制類型
        if attention_types is None:
            attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
        
        logger.info(f"開始注意力機制分析...")
        logger.info(f"測試的注意力機制: {', '.join(attention_types)}")
        
        # 執行注意力分析
        results = processor.process_with_attention(
            input_file=input_file,
            attention_types=attention_types,
            topics_path=topics_path,
            attention_weights=attention_weights,
            save_results=True
        )
        
        # 輸出結果摘要
        if 'comparison' in results and 'summary' in results['comparison']:
            summary = results['comparison']['summary']
            logger.info(f"分析完成！最佳注意力機制: {summary.get('best_mechanism', 'N/A')}")
            logger.info(f"最佳綜合得分: {summary.get('best_score', 0):.4f}")
        
        logger.info(f"結果保存在: {output_dir}")
        return results
        
    except Exception as e:
        logger.error(f"注意力機制分析過程中發生錯誤: {str(e)}")
        raise

def process_attention_analysis_with_classification(input_file: Optional[str] = None, 
                                                 output_dir: Optional[str] = None,
                                                 attention_types: Optional[List[str]] = None,
                                                 topics_path: Optional[str] = None,
                                                 attention_weights: Optional[Dict] = None) -> Dict:
    """
    執行完整的注意力機制分析和分類評估
    
    Args:
        input_file: 輸入文件路徑
        output_dir: 輸出目錄路徑
        attention_types: 要測試的注意力機制類型
        topics_path: 關鍵詞文件路徑
        attention_weights: 組合注意力權重配置
        
    Returns:
        Dict: 完整的分析和分類結果
    """
    try:
        # 初始化注意力處理器
        processor = AttentionProcessor(output_dir=output_dir)
        
        # 檢查輸入文件
        if input_file is None or not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到輸入文件：{input_file}")
        
        # 設定預設的注意力機制類型
        if attention_types is None:
            attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
        
        logger.info(f"開始完整的注意力機制分析和分類評估...")
        logger.info(f"測試的注意力機制: {', '.join(attention_types)}")
        
        # 讀取元數據
        df = pd.read_csv(input_file)
        
        # 第一步：執行注意力分析
        attention_results = processor.process_with_attention(
            input_file=input_file,
            attention_types=attention_types,
            topics_path=topics_path,
            attention_weights=attention_weights,
            save_results=False  # 暫不保存，等分類評估完成後一起保存
        )
        
        # 第二步：執行分類評估
        logger.info("開始執行分類評估...")
        classifier = SentimentClassifier(output_dir=output_dir)
        
        # 評估不同注意力機制的分類性能
        classification_results = classifier.evaluate_attention_mechanisms(
            attention_results, df
        )
        
        # 第三步：整合結果
        final_results = {
            'attention_analysis': attention_results,
            'classification_evaluation': classification_results,
            'processing_info': attention_results.get('processing_info', {}),
            'summary': {}
        }
        
        # 生成綜合摘要
        if 'comparison' in classification_results:
            class_comparison = classification_results['comparison']
            final_results['summary'] = {
                'best_attention_mechanism': class_comparison.get('best_mechanism', 'N/A'),
                'best_classification_accuracy': class_comparison['summary'].get('best_accuracy', 0),
                'best_f1_score': class_comparison['summary'].get('best_f1', 0),
                'mechanisms_tested': len(attention_types),
                'evaluation_completed': True
            }
            
            logger.info(f"評估完成！最佳注意力機制: {final_results['summary']['best_attention_mechanism']}")
            logger.info(f"最佳分類準確率: {final_results['summary']['best_classification_accuracy']:.4f}")
            logger.info(f"最佳F1分數: {final_results['summary']['best_f1_score']:.4f}")
        
        # 保存完整結果
        if output_dir:
            results_file = os.path.join(output_dir, "complete_analysis_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                import json
                # 處理不可序列化的對象
                serializable_results = _make_serializable(final_results)
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            logger.info(f"完整結果已保存至: {results_file}")
        
        logger.info(f"完整分析結果保存在: {output_dir}")
        return final_results
        
    except Exception as e:
        logger.error(f"完整分析過程中發生錯誤: {str(e)}")
        raise

def _make_serializable(obj):
    """將物件轉換為可序列化的格式"""
    if isinstance(obj, dict):
        return {key: _make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    else:
        return obj

def compare_attention_mechanisms(input_file: Optional[str] = None,
                               output_dir: Optional[str] = None,
                               attention_types: Optional[List[str]] = None) -> Dict:
    """
    專門用於比較不同注意力機制效果
    
    Args:
        input_file: 輸入文件路徑
        output_dir: 輸出目錄路徑
        attention_types: 要比較的注意力機制類型
        
    Returns:
        Dict: 比較結果
    """
    try:
        # 初始化注意力處理器
        processor = AttentionProcessor(output_dir=output_dir)
        
        # 執行比較
        results = processor.compare_attention_mechanisms(
            input_file=input_file,
            attention_types=attention_types
        )
        
        # 生成可讀的比較報告
        if output_dir:
            report_file = os.path.join(output_dir, "attention_comparison_report.txt")
            processor.export_comparison_report(results, report_file)
        
        return results
        
    except Exception as e:
        logger.error(f"注意力機制比較過程中發生錯誤: {str(e)}")
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
            elif sys.argv[1] == '--attention':
                # 執行注意力機制分析
                if len(sys.argv) > 2:
                    input_file = sys.argv[2]
                else:
                    input_file = None
                process_attention_analysis(input_file=input_file)
            elif sys.argv[1] == '--classify':
                # 執行完整的注意力機制分析和分類評估
                if len(sys.argv) > 2:
                    input_file = sys.argv[2]
                else:
                    input_file = None
                process_attention_analysis_with_classification(input_file=input_file)
            elif sys.argv[1] == '--compare':
                # 比較注意力機制
                if len(sys.argv) > 2:
                    input_file = sys.argv[2]
                else:
                    input_file = None
                compare_attention_mechanisms(input_file=input_file)
            elif sys.argv[1] == '--help':
                print_help()
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

def print_help():
    """輸出幫助信息"""
    help_text = """
BERT情感分析系統 - 使用說明

基本用法:
    python Part05_Main.py                    # 啟動GUI介面
    python Part05_Main.py --help            # 顯示此幫助信息

命令行選項:
    --process                               # 執行BERT編碼處理
    --attention [input_file]               # 執行注意力機制分析（僅幾何評估）
    --classify [input_file]                # 執行完整的注意力機制分析和分類評估
    --compare [input_file]                 # 比較不同注意力機制效果
    --new-run                              # 創建新的執行目錄

功能說明:
    --attention: 執行注意力機制分析，計算面向向量的內聚度和分離度
    --classify:  執行完整流程，包括注意力分析和情感分類評估
                包含訓練分類器、預測情感標籤、計算準確率等指標

注意力機制分析:
    系統支援以下注意力機制類型:
    - no/none: 無注意力機制（平均）
    - similarity: 基於相似度的注意力機制
    - keyword: 基於關鍵詞的注意力機制  
    - self: 自注意力機制
    - combined: 組合型注意力機制

範例:
    python Part05_Main.py --attention data.csv        # 僅分析注意力機制
    python Part05_Main.py --classify data.csv         # 完整分類評估
    python Part05_Main.py --compare processed_data.csv

注意：
    - input_file 應該是經過預處理的CSV文件
    - 系統會自動檢測文本欄位（processed_text, clean_text, text, review）
    - --classify選項會執行完整的機器學習流程，包括分類器訓練和評估
    - 結果會保存在自動生成的輸出目錄中
    """
    print(help_text)

if __name__ == "__main__":
    main() 