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
from modules.pipeline_processor import AnalysisPipeline, MultiPipelineComparison, create_simple_pipeline, run_quick_analysis
from modules.text_encoders import TextEncoderFactory
from modules.classification_methods import ClassificationMethodFactory

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 注意：移除了全域RunManager，每個處理器會創建自己的RunManager實例

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
        
        # 注意：embed.encode() 方法已經自動保存了特徵向量，無需再次保存
        
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
                                                 attention_weights: Optional[Dict] = None,
                                                 classifier_type: Optional[str] = None) -> Dict:
    """
    執行完整的注意力機制分析和分類評估
    
    Args:
        input_file: 輸入文件路徑
        output_dir: 輸出目錄路徑
        attention_types: 要測試的注意力機制類型
        topics_path: 關鍵詞文件路徑
        attention_weights: 組合注意力權重配置
        classifier_type: 分類器類型 (xgboost, logistic_regression, random_forest, svm_linear)
        
    Returns:
        Dict: 完整的分析和分類結果
    """
    try:
        # GPU環境檢測和預處理
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"檢測到GPU環境: {torch.cuda.get_device_name()}")
                # 設置GPU記憶體管理
                torch.cuda.empty_cache()
                # 防止GPU記憶體碎片化
                import gc
                gc.collect()
            else:
                logger.info("運行在CPU環境")
        except ImportError:
            logger.info("PyTorch未安裝，運行在CPU環境")
        except Exception as gpu_error:
            logger.warning(f"GPU環境檢測失敗，繼續使用CPU: {str(gpu_error)}")
        
        print("\n" + "="*80)
        print("🚀 開始執行完整的注意力機制分析和分類評估")
        print("="*80)
        
        # 初始化注意力處理器
        processor = AttentionProcessor(output_dir=output_dir)
        
        # 檢查輸入文件
        if input_file is None or not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到輸入文件：{input_file}")
        
        # 設定預設的注意力機制類型
        if attention_types is None:
            attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
        
        print(f"\n📋 分析配置:")
        print(f"   • 輸入文件: {input_file}")
        print(f"   • 輸出目錄: {output_dir}")
        print(f"   • 注意力機制: {', '.join(attention_types)}")
        
        logger.info(f"開始完整的注意力機制分析和分類評估...")
        logger.info(f"測試的注意力機制: {', '.join(attention_types)}")
        
        # 讀取元數據
        df = pd.read_csv(input_file)
        
        # 第一階段：執行注意力分析
        print(f"\n🔬 階段 1/3: 注意力機制分析")
        print("-" * 50)
        attention_results = processor.process_with_attention(
            input_file=input_file,
            attention_types=attention_types,
            topics_path=topics_path,
            attention_weights=attention_weights,
            save_results=False  # 暫不保存，等分類評估完成後一起保存
        )
        
        # 第二階段：執行分類評估
        print(f"\n🎯 階段 2/3: 分類性能評估")
        print("-" * 50)
        logger.info("開始執行分類評估...")
        classifier = SentimentClassifier(output_dir=output_dir)
        
        # 修正：獲取或載入原始BERT嵌入向量
        print(f"   🔍 載入原始BERT嵌入向量用於分類評估...")
        original_embeddings = None
        # 確保從run目錄根目錄讀取
        if any(subdir in output_dir for subdir in ["01_preprocessing", "02_bert_encoding", "03_attention_testing", "04_analysis"]):
            run_dir = os.path.dirname(output_dir)
        else:
            run_dir = output_dir
        embeddings_file = os.path.join(run_dir, "02_bert_embeddings.npy")
        
        if os.path.exists(embeddings_file):
            # 載入已存在的BERT嵌入向量
            original_embeddings = np.load(embeddings_file)
            print(f"   ✅ 已載入原始BERT嵌入向量，形狀: {original_embeddings.shape}")
            logger.info(f"載入原始BERT嵌入向量: {original_embeddings.shape}")
        else:
            # 如果沒有找到，重新生成
            print(f"   🔄 未找到BERT嵌入向量文件，開始重新生成...")
            logger.info("未找到BERT嵌入向量，開始重新生成...")
            
            from modules.bert_encoder import BertEncoder
            bert_encoder = BertEncoder(output_dir=output_dir)
            
            # 找到文本欄位
            text_column = None
            for col in ['processed_text', 'clean_text', 'text', 'review']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column:
                original_embeddings = bert_encoder.encode(df[text_column])
                print(f"   ✅ BERT嵌入向量生成完成，形狀: {original_embeddings.shape}")
                logger.info(f"生成的BERT嵌入向量形狀: {original_embeddings.shape}")
            else:
                raise ValueError("無法找到文本欄位來生成BERT嵌入向量")
        
        # 評估不同注意力機制的分類性能（修正：傳遞原始嵌入向量）
        classification_results = classifier.evaluate_attention_mechanisms(
            attention_results, df, original_embeddings, model_type=classifier_type
        )
        
        # 第三階段：整合結果
        print(f"\n📊 階段 3/3: 整合結果和生成報告")
        print("-" * 50)
        final_results = {
            'attention_analysis': attention_results,
            'classification_evaluation': classification_results,
            'processing_info': attention_results.get('processing_info', {}),
            'summary': {}
        }
        
        # 生成綜合摘要
        if 'comparison' in classification_results:
            class_comparison = classification_results['comparison']
            # 添加安全檢查
            summary = class_comparison.get('summary', {})
            if summary:  # 確保summary不為空
                final_results['summary'] = {
                    'best_attention_mechanism': class_comparison.get('best_mechanism', 'N/A'),
                    'best_classification_accuracy': summary.get('best_accuracy', 0),
                    'best_f1_score': summary.get('best_f1', 0),
                    'mechanisms_tested': len(attention_types),
                    'evaluation_completed': True
                }
            else:
                # 如果沒有summary，創建基本摘要
                final_results['summary'] = {
                    'best_attention_mechanism': class_comparison.get('best_mechanism', 'N/A'),
                    'best_classification_accuracy': 0,
                    'best_f1_score': 0,
                    'mechanisms_tested': len(attention_types),
                    'evaluation_completed': False,
                    'error': 'Summary not available'
                }
            
            print(f"\n🏆 最終評估結果:")
            print(f"   • 最佳注意力機制: {final_results['summary']['best_attention_mechanism']}")
            print(f"   • 最佳分類準確率: {final_results['summary']['best_classification_accuracy']:.4f}")
            print(f"   • 最佳F1分數: {final_results['summary']['best_f1_score']:.4f}")
            
            logger.info(f"評估完成！最佳注意力機制: {final_results['summary']['best_attention_mechanism']}")
            logger.info(f"最佳分類準確率: {final_results['summary']['best_classification_accuracy']:.4f}")
            logger.info(f"最佳F1分數: {final_results['summary']['best_f1_score']:.4f}")
        else:
            # 如果沒有comparison結果，創建錯誤摘要
            final_results['summary'] = {
                'best_attention_mechanism': 'N/A',
                'best_classification_accuracy': 0,
                'best_f1_score': 0,
                'mechanisms_tested': len(attention_types),
                'evaluation_completed': False,
                'error': 'Classification comparison not available'
            }
        
        # 保存完整結果到run目錄根目錄
        if output_dir:
            print(f"\n💾 保存完整分析結果...")
            # 確保保存到run目錄的根目錄
            if any(subdir in output_dir for subdir in ["01_preprocessing", "02_bert_encoding", "03_attention_testing", "04_analysis"]):
                # 如果輸出目錄是子目錄，改為父目錄（run目錄根目錄）
                run_dir = os.path.dirname(output_dir)
            else:
                run_dir = output_dir
            
            results_file = os.path.join(run_dir, "complete_analysis_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                import json
                # 處理不可序列化的對象
                serializable_results = _make_serializable(final_results)
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"✅ 完整結果已保存至: {results_file}")
            logger.info(f"完整結果已保存至: {results_file}")
        
        print(f"\n🎉 完整分析評估完成！")
        print(f"📁 所有結果保存在: {output_dir}")
        print("="*80)
        
        logger.info(f"完整分析結果保存在: {output_dir}")
        return final_results
        
    except Exception as e:
        # 詳細的錯誤追蹤
        import traceback
        error_details = traceback.format_exc()
        
        logger.error(f"完整分析過程中發生錯誤: {str(e)}")
        logger.error(f"詳細錯誤追蹤:\n{error_details}")
        
        # 嘗試GPU記憶體清理
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info("已清理GPU記憶體")
        except:
            pass
        
        # 重新拋出錯誤，但添加更多上下文信息
        raise RuntimeError(f"完整分析失敗: {str(e)}。請檢查日誌文件以獲取詳細的錯誤追蹤信息。") from e

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

def process_attention_analysis_with_multiple_combinations(input_file: Optional[str] = None, 
                                                       output_dir: Optional[str] = None,
                                                       attention_types: Optional[List[str]] = None,
                                                       attention_combinations: Optional[List[Dict]] = None,
                                                       classifier_type: Optional[str] = None) -> Dict:
    """
    執行多種注意力機制組合分析
    
    Args:
        input_file: 輸入文件路徑
        output_dir: 輸出目錄路徑
        attention_types: 要測試的基本注意力機制類型
        attention_combinations: 多個組合注意力的權重配置列表
        classifier_type: 分類器類型
        
    Returns:
        Dict: 完整的分析和分類結果
    """
    try:
        # GPU環境檢測和預處理
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"檢測到GPU環境: {torch.cuda.get_device_name()}")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            else:
                logger.info("運行在CPU環境")
        except ImportError:
            logger.info("PyTorch未安裝，運行在CPU環境")
        except Exception as gpu_error:
            logger.warning(f"GPU環境檢測失敗，繼續使用CPU: {str(gpu_error)}")
        
        print("\n" + "="*80)
        print("🚀 開始執行多重注意力機制組合分析")
        print("="*80)
        
        # 初始化注意力處理器
        processor = AttentionProcessor(output_dir=output_dir)
        
        # 檢查輸入文件
        if input_file is None or not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到輸入文件：{input_file}")
        
        # 設定預設的注意力機制類型
        if attention_types is None:
            attention_types = ['no', 'similarity', 'keyword', 'self']
        
        # 設定預設的組合配置
        if attention_combinations is None:
            attention_combinations = []
        
        # 組合所有要測試的注意力機制
        all_attention_types = attention_types.copy()
        
        print(f"\n📋 分析配置:")
        print(f"   • 輸入文件: {input_file}")
        print(f"   • 輸出目錄: {output_dir}")
        print(f"   • 基本注意力機制: {', '.join(attention_types)}")
        print(f"   • 組合配置數量: {len(attention_combinations)}")
        
        # 讀取元數據
        df = pd.read_csv(input_file)
        
        # 第一階段：執行基本注意力分析
        print(f"\n🔬 階段 1/3: 基本注意力機制分析")
        print("-" * 50)
        
        # 先執行基本注意力機制分析
        basic_results = processor.process_with_attention(
            input_file=input_file,
            attention_types=attention_types,
            save_results=False
        )
        
        # 第二階段：執行組合注意力分析
        if attention_combinations:
            print(f"\n🔗 階段 2/3: 組合注意力機制分析")
            print("-" * 50)
            
            combination_results = {}
            for i, combination in enumerate(attention_combinations, 1):
                combination_name = f"combination_{i}"
                print(f"   🔄 處理組合 {i}/{len(attention_combinations)}: {combination}")
                
                # 執行單個組合分析
                combo_result = processor.process_with_attention(
                    input_file=input_file,
                    attention_types=['combined'],
                    attention_weights=combination,
                    save_results=False
                )
                
                # 將組合結果添加到基本結果中
                combination_results[combination_name] = combo_result['combined']
                # 為了統一格式，也添加到all_attention_types中
                all_attention_types.append(combination_name)
            
            # 合併基本結果和組合結果
            final_attention_results = basic_results.copy()
            final_attention_results.update(combination_results)
        else:
            final_attention_results = basic_results
            print(f"\n⏭️ 跳過組合分析階段（無組合配置）")
        
        # 第三階段：執行分類評估
        print(f"\n🎯 階段 3/3: 分類性能評估")
        print("-" * 50)
        logger.info("開始執行分類評估...")
        classifier = SentimentClassifier(output_dir=output_dir)
        
        # 獲取或載入原始BERT嵌入向量
        print(f"   🔍 載入原始BERT嵌入向量用於分類評估...")
        original_embeddings = None
        # 確保從run目錄根目錄讀取
        if any(subdir in output_dir for subdir in ["01_preprocessing", "02_bert_encoding", "03_attention_testing", "04_analysis"]):
            run_dir = os.path.dirname(output_dir)
        else:
            run_dir = output_dir
        embeddings_file = os.path.join(run_dir, "02_bert_embeddings.npy")
        
        if os.path.exists(embeddings_file):
            original_embeddings = np.load(embeddings_file)
            print(f"   ✅ 已載入原始BERT嵌入向量，形狀: {original_embeddings.shape}")
            logger.info(f"載入原始BERT嵌入向量: {original_embeddings.shape}")
        else:
            print(f"   🔄 未找到BERT嵌入向量文件，開始重新生成...")
            logger.info("未找到BERT嵌入向量，開始重新生成...")
            
            from modules.bert_encoder import BertEncoder
            bert_encoder = BertEncoder(output_dir=output_dir)
            
            # 找到文本欄位
            text_column = None
            for col in ['processed_text', 'clean_text', 'text', 'review']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column:
                original_embeddings = bert_encoder.encode(df[text_column])
                print(f"   ✅ BERT嵌入向量生成完成，形狀: {original_embeddings.shape}")
                logger.info(f"生成的BERT嵌入向量形狀: {original_embeddings.shape}")
            else:
                raise ValueError("無法找到文本欄位來生成BERT嵌入向量")
        
        # 評估所有注意力機制的分類性能
        classification_results = classifier.evaluate_attention_mechanisms(
            final_attention_results, df, original_embeddings, model_type=classifier_type
        )
        
        # 整合結果
        print(f"\n📊 整合分析結果...")
        final_results = {
            'attention_analysis': final_attention_results,
            'classification_evaluation': classification_results,
            'processing_info': basic_results.get('processing_info', {}),
            'summary': {}
        }
        
        # 生成綜合摘要
        if 'comparison' in classification_results:
            class_comparison = classification_results['comparison']
            summary = class_comparison.get('summary', {})
            if summary:
                final_results['summary'] = {
                    'best_attention_mechanism': class_comparison.get('best_mechanism', 'N/A'),
                    'best_classification_accuracy': summary.get('best_accuracy', 0),
                    'best_f1_score': summary.get('best_f1', 0),
                    'mechanisms_tested': len(all_attention_types),
                    'combinations_tested': len(attention_combinations),
                    'evaluation_completed': True
                }
            else:
                final_results['summary'] = {
                    'best_attention_mechanism': class_comparison.get('best_mechanism', 'N/A'),
                    'best_classification_accuracy': 0,
                    'best_f1_score': 0,
                    'mechanisms_tested': len(all_attention_types),
                    'combinations_tested': len(attention_combinations),
                    'evaluation_completed': False,
                    'error': 'Summary not available'
                }
            
            print(f"\n🏆 最終評估結果:")
            print(f"   • 最佳注意力機制: {final_results['summary']['best_attention_mechanism']}")
            print(f"   • 最佳分類準確率: {final_results['summary']['best_classification_accuracy']:.4f}")
            print(f"   • 最佳F1分數: {final_results['summary']['best_f1_score']:.4f}")
            print(f"   • 測試機制總數: {final_results['summary']['mechanisms_tested']}")
            print(f"   • 組合配置數量: {final_results['summary']['combinations_tested']}")
            
            logger.info(f"評估完成！最佳注意力機制: {final_results['summary']['best_attention_mechanism']}")
            logger.info(f"最佳分類準確率: {final_results['summary']['best_classification_accuracy']:.4f}")
            logger.info(f"最佳F1分數: {final_results['summary']['best_f1_score']:.4f}")
        else:
            final_results['summary'] = {
                'best_attention_mechanism': 'N/A',
                'best_classification_accuracy': 0,
                'best_f1_score': 0,
                'mechanisms_tested': len(all_attention_types),
                'combinations_tested': len(attention_combinations),
                'evaluation_completed': False,
                'error': 'Classification comparison not available'
            }
        
        # 保存完整結果到run目錄根目錄
        if output_dir:
            print(f"\n💾 保存完整分析結果...")
            # 確保保存到run目錄的根目錄
            if any(subdir in output_dir for subdir in ["01_preprocessing", "02_bert_encoding", "03_attention_testing", "04_analysis"]):
                # 如果輸出目錄是子目錄，改為父目錄（run目錄根目錄）
                run_dir = os.path.dirname(output_dir)
            else:
                run_dir = output_dir
            
            results_file = os.path.join(run_dir, "multiple_combinations_analysis_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                import json
                serializable_results = _make_serializable(final_results)
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"✅ 完整結果已保存至: {results_file}")
            logger.info(f"完整結果已保存至: {results_file}")
        
        print(f"\n🎉 多重組合分析完成！")
        print(f"📁 所有結果保存在: {output_dir}")
        print("="*80)
        
        logger.info(f"多重組合分析結果保存在: {output_dir}")
        return final_results
        
    except Exception as e:
        # 詳細的錯誤追蹤
        import traceback
        error_details = traceback.format_exc()
        
        logger.error(f"多重組合分析過程中發生錯誤: {str(e)}")
        logger.error(f"詳細錯誤追蹤:\n{error_details}")
        
        # 嘗試GPU記憶體清理
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info("已清理GPU記憶體")
        except:
            pass
        
        raise RuntimeError(f"多重組合分析失敗: {str(e)}。請檢查日誌文件以獲取詳細的錯誤追蹤信息。") from e

def run_new_pipeline_analysis(input_file: Optional[str] = None, 
                             output_dir: Optional[str] = None,
                             encoder_type: str = 'bert',
                             classifier_type: str = 'sentiment',
                             encoder_config: Optional[Dict] = None,
                             classifier_config: Optional[Dict] = None) -> Dict:
    """
    執行新的流程分析架構
    
    Args:
        input_file: 輸入文件路徑
        output_dir: 輸出目錄路徑
        encoder_type: 編碼器類型 ('bert', 'gpt', 't5', 'cnn', 'elmo')
        classifier_type: 分類器類型 ('sentiment', 'lda', 'bertopic', 'nmf', 'clustering')
        encoder_config: 編碼器配置參數
        classifier_config: 分類器配置參數
        
    Returns:
        Dict: 完整的分析結果
    """
    try:
        print("\n" + "="*80)
        print("🚀 開始執行新架構流程分析")
        print("="*80)
        
        # 檢查輸入文件
        if input_file is None or not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到輸入文件：{input_file}")
        
        # 讀取數據
        logger.info(f"讀取數據: {input_file}")
        df = pd.read_csv(input_file)
        
        # 檢查必要的欄位
        text_column = None
        for col in ['processed_text', 'clean_text', 'text', 'review']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            available_columns = ', '.join(df.columns)
            raise ValueError(f"在輸入文件中找不到文本欄位。可用的欄位有：{available_columns}")
        
        texts = df[text_column]
        
        # 檢查標籤欄位
        labels = None
        label_column = None
        for col in ['label', 'sentiment', 'category', 'class']:
            if col in df.columns:
                label_column = col
                labels = df[col]
                break
        
        print(f"\n📋 分析配置:")
        print(f"   • 輸入文件: {input_file}")
        print(f"   • 輸出目錄: {output_dir}")
        print(f"   • 編碼器: {encoder_type}")
        print(f"   • 分類器: {classifier_type}")
        print(f"   • 文本欄位: {text_column}")
        print(f"   • 標籤欄位: {label_column if label_column else '無'}")
        print(f"   • 數據量: {len(texts)}")
        
        # 創建流程
        pipeline = AnalysisPipeline(output_dir=output_dir)
        
        # 配置流程
        config_info = pipeline.configure_pipeline(
            encoder_type=encoder_type,
            classifier_type=classifier_type,
            encoder_config=encoder_config or {},
            classifier_config=classifier_config or {}
        )
        
        # 顯示配置信息
        print(f"\n🔧 流程配置:")
        print(f"   • 編碼器: {config_info['encoder']['name']}")
        print(f"   • 嵌入維度: {config_info['encoder']['embedding_dim']}")
        print(f"   • 分類器: {config_info['classifier']['method_name']}")
        
        # 顯示兼容性信息
        compatibility = config_info['compatibility']
        if compatibility['warnings']:
            print(f"   ⚠️  警告: {'; '.join(compatibility['warnings'])}")
        if compatibility['recommendations']:
            print(f"   💡 建議: {'; '.join(compatibility['recommendations'])}")
        
        # 運行流程
        results = pipeline.run_pipeline(texts=texts, labels=labels)
        
        # 顯示結果摘要
        if 'summary' in results:
            summary = results['summary']
            print(f"\n📊 結果摘要:")
            print(f"   • 分析類型: {summary.get('analysis_type', 'N/A')}")
            
            if summary['analysis_type'] == '情感分析':
                print(f"   • 準確率: {summary.get('accuracy', 0):.4f}")
                print(f"   • F1分數: {summary.get('f1_score', 0):.4f}")
                print(f"   • 類別數: {summary.get('n_classes', 0)}")
            elif summary['analysis_type'] == '主題建模':
                print(f"   • 方法: {summary.get('method', 'N/A')}")
                print(f"   • 主題數: {summary.get('n_topics', 0)}")
                if 'coherence_score' in summary:
                    print(f"   • 一致性分數: {summary['coherence_score']:.4f}")
            elif summary['analysis_type'] == '聚類分析':
                print(f"   • 聚類方法: {summary.get('clustering_method', 'N/A')}")
                print(f"   • 聚類數: {summary.get('n_clusters', 0)}")
                if summary.get('silhouette_score'):
                    print(f"   • 輪廓分數: {summary['silhouette_score']:.4f}")
        
        print(f"\n🎉 新架構流程分析完成！")
        print(f"📁 結果保存在: {output_dir}")
        print("="*80)
        
        logger.info(f"新架構流程分析結果保存在: {output_dir}")
        return results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        logger.error(f"新架構流程分析過程中發生錯誤: {str(e)}")
        logger.error(f"詳細錯誤追蹤:\n{error_details}")
        
        # 嘗試GPU記憶體清理
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info("已清理GPU記憶體")
        except:
            pass
        
        raise RuntimeError(f"新架構流程分析失敗: {str(e)}。請檢查日誌文件以獲取詳細的錯誤追蹤信息。") from e

def run_multi_pipeline_comparison(input_file: Optional[str] = None, 
                                output_dir: Optional[str] = None,
                                pipeline_configs: Optional[List[Dict]] = None) -> Dict:
    """
    執行多流程比較分析
    
    Args:
        input_file: 輸入文件路徑
        output_dir: 輸出目錄路徑
        pipeline_configs: 流程配置列表
        
    Returns:
        Dict: 比較分析結果
    """
    try:
        print("\n" + "="*80)
        print("🔬 開始執行多流程比較分析")
        print("="*80)
        
        # 檢查輸入文件
        if input_file is None or not os.path.exists(input_file):
            raise FileNotFoundError(f"找不到輸入文件：{input_file}")
        
        # 讀取數據
        logger.info(f"讀取數據: {input_file}")
        df = pd.read_csv(input_file)
        
        # 檢查文本和標籤欄位
        text_column = None
        for col in ['processed_text', 'clean_text', 'text', 'review']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            raise ValueError("找不到文本欄位")
        
        texts = df[text_column]
        
        labels = None
        for col in ['label', 'sentiment', 'category', 'class']:
            if col in df.columns:
                labels = df[col]
                break
        
        # 設置預設的流程配置
        if pipeline_configs is None:
            pipeline_configs = [
                {
                    'name': 'BERT+情感分析',
                    'encoder_type': 'bert',
                    'classifier_type': 'sentiment'
                },
                {
                    'name': 'BERT+LDA主題建模',
                    'encoder_type': 'bert',
                    'classifier_type': 'lda',
                    'classifier_config': {'n_topics': 5}
                },
                {
                    'name': 'CNN+聚類分析',
                    'encoder_type': 'cnn',
                    'classifier_type': 'clustering',
                    'classifier_config': {'method': 'kmeans', 'n_clusters': 5}
                }
            ]
        
        print(f"\n📋 比較配置:")
        print(f"   • 輸入文件: {input_file}")
        print(f"   • 輸出目錄: {output_dir}")
        print(f"   • 流程數量: {len(pipeline_configs)}")
        print(f"   • 數據量: {len(texts)}")
        
        # 創建多流程比較器
        comparator = MultiPipelineComparison(output_dir=output_dir)
        
        # 添加流程配置
        for config in pipeline_configs:
            comparator.add_pipeline_config(
                name=config['name'],
                encoder_type=config['encoder_type'],
                classifier_type=config['classifier_type'],
                encoder_config=config.get('encoder_config', {}),
                classifier_config=config.get('classifier_config', {})
            )
        
        # 運行比較
        comparison_results = comparator.run_comparison(texts=texts, labels=labels)
        
        # 顯示比較結果
        metrics = comparison_results.get('comparison_metrics', {})
        if 'error' not in metrics:
            print(f"\n📊 比較結果:")
            print(f"   • 成功流程: {metrics['successful_pipelines']}")
            print(f"   • 失敗流程: {metrics['failed_pipelines']}")
            print(f"   • 平均處理時間: {metrics['performance_stats']['avg_processing_time']:.2f}秒")
            print(f"   • 最快流程: {metrics['fastest_pipeline']}")
            print(f"   • 最省記憶體流程: {metrics['most_memory_efficient']}")
            
            recommendations = comparison_results.get('recommendations', {})
            if 'best_overall' in recommendations:
                print(f"   • 綜合推薦: {recommendations['best_overall']}")
        
        print(f"\n🎉 多流程比較分析完成！")
        print(f"📁 結果保存在: {output_dir}")
        print("="*80)
        
        logger.info(f"多流程比較分析結果保存在: {output_dir}")
        return comparison_results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        
        logger.error(f"多流程比較分析過程中發生錯誤: {str(e)}")
        logger.error(f"詳細錯誤追蹤:\n{error_details}")
        
        raise RuntimeError(f"多流程比較分析失敗: {str(e)}。請檢查日誌文件以獲取詳細的錯誤追蹤信息。") from e

def show_available_options():
    """顯示可用的編碼器和分類器選項"""
    print("\n" + "="*80)
    print("📋 可用的分析選項")
    print("="*80)
    
    # 顯示編碼器選項
    print("\n🔤 文本編碼器:")
    encoder_info = TextEncoderFactory.get_encoder_info()
    for encoder_type, info in encoder_info.items():
        status = "✅" if info.get('available', True) else "❌"
        print(f"   {status} {encoder_type.upper()}: {info['name']}")
        print(f"      - 描述: {info['description']}")
        print(f"      - 嵌入維度: {info['embedding_dim']}")
        print(f"      - 優勢: {info['advantages']}")
        if not info.get('available', True) and 'note' in info:
            print(f"      - 注意: {info['note']}")
        print()
    
    # 顯示分類器選項
    print("🎯 分類/建模方法:")
    classifier_info = ClassificationMethodFactory.get_method_info()
    for classifier_type, info in classifier_info.items():
        status = "✅" if info.get('available', True) else "❌"
        print(f"   {status} {classifier_type.upper()}: {info['name']}")
        print(f"      - 描述: {info['description']}")
        print(f"      - 類型: {info['type']}")
        print(f"      - 需要標籤: {'是' if info['needs_labels'] else '否'}")
        print(f"      - 優勢: {info['advantages']}")
        if not info.get('available', True) and 'note' in info:
            print(f"      - 注意: {info['note']}")
        print()
    
    print("="*80)

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
            elif sys.argv[1] == '--new-run':
                # 清除上次執行的記錄，強制創建新的run目錄
                pass  # 已移除clear_last_run，保留佔位
            elif sys.argv[1] == '--new-pipeline':
                # 執行新架構流程分析
                input_file = sys.argv[2] if len(sys.argv) > 2 else None
                encoder_type = sys.argv[3] if len(sys.argv) > 3 else 'bert'
                classifier_type = sys.argv[4] if len(sys.argv) > 4 else 'sentiment'
                run_new_pipeline_analysis(input_file=input_file, 
                                         encoder_type=encoder_type,
                                         classifier_type=classifier_type)
            elif sys.argv[1] == '--multi-compare':
                # 執行多流程比較分析
                input_file = sys.argv[2] if len(sys.argv) > 2 else None
                run_multi_pipeline_comparison(input_file=input_file)
            elif sys.argv[1] == '--show-options':
                # 顯示可用選項
                show_available_options()
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
    
新架構選項:
    --new-pipeline [input_file] [encoder] [classifier]  # 執行新架構流程分析
    --multi-compare [input_file]           # 執行多流程比較分析
    --show-options                         # 顯示可用的編碼器和分類器選項

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

進度顯示功能:
    ✨ 新增功能：系統現在會顯示詳細的執行進度信息
    📊 BERT編碼進度條：顯示批量處理的進度
    🔬 注意力分析進度：顯示各個階段的完成狀態
    🎯 分類評估進度：顯示每個注意力機制的評估進度
    📈 實時結果顯示：即時顯示各項指標的計算結果

範例:
    python Part05_Main.py --attention data.csv        # 僅分析注意力機制
    python Part05_Main.py --classify data.csv         # 完整分類評估
    python Part05_Main.py --compare processed_data.csv
    
新架構範例:
    python Part05_Main.py --show-options              # 查看所有可用選項
    python Part05_Main.py --new-pipeline data.csv bert sentiment  # BERT+情感分析
    python Part05_Main.py --new-pipeline data.csv bert lda        # BERT+LDA主題建模
    python Part05_Main.py --new-pipeline data.csv cnn clustering  # CNN+聚類分析
    python Part05_Main.py --new-pipeline data.csv t5 bertopic     # T5+BERTopic主題建模
    python Part05_Main.py --multi-compare data.csv               # 多流程自動比較

測試進度功能:
    python test_progress.py                           # 測試進度顯示功能

注意：
    - input_file 應該是經過預處理的CSV文件
    - 系統會自動檢測文本欄位（processed_text, clean_text, text, review）
    - --classify選項會執行完整的機器學習流程，包括分類器訓練和評估
    - 結果會保存在自動生成的輸出目錄中
    - 進度信息會同時顯示在終端機和日誌文件中
    """
    print(help_text)

if __name__ == "__main__":
    main() 