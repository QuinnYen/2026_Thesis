"""
實驗流程模組
此模組負責面向分析的完整實驗流程
"""

import os
import sys
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 導入相關模組
from Part03_.utils.config_manager import ConfigManager
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.result_manager import ResultManager
from Part03_.core.data_importer import DataImporter
from Part03_.core.bert_embedder import BertEmbedder
from Part03_.core.topic_extractor import TopicExtractor
from Part03_.core.aspect_calculator import AspectCalculator
from Part03_.experiments.evaluator import Evaluator
from Part03_.experiments.report_generator import ReportGenerator

class Pipeline:
    """
    面向分析流程類
    用於自動化執行完整的面向分析流程
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化分析流程
        
        Args:
            config: 配置管理器，如果沒有提供則創建新的
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 設置日誌
        log_dir = self.config.get('logging.log_dir', './Part03_/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'pipeline.log')
        
        self.logger = logging.getLogger('pipeline')
        self.logger.setLevel(logging.INFO)
        
        # 移除所有處理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 添加文件處理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # 添加控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        # 初始化組件
        results_dir = self.config.get('data_settings.output_directory', './Part03_/results')
        self.result_manager = ResultManager(base_dir=results_dir)
        self.data_importer = DataImporter(config=self.config)
        self.bert_embedder = BertEmbedder(config=self.config)
        self.topic_extractor = TopicExtractor(config=self.config)
        self.aspect_calculator = AspectCalculator(config=self.config)
        self.evaluator = Evaluator(config=self.config)
        self.report_generator = ReportGenerator(config=self.config)
        
        # 儲存流程結果
        self.results = {}
        
    def run_full_pipeline(self, dataset_type: str, dataset_params: Dict[str, Any], 
                        sample_size: Optional[int] = None, 
                        description: str = "", 
                        console_output: bool = True) -> Dict[str, Any]:
        """
        執行完整的面向分析流程
        
        Args:
            dataset_type: 數據集類型，可為 'imdb', 'amazon', 'yelp' 或 'custom'
            dataset_params: 數據集參數，例如文件路徑等
            sample_size: 樣本大小，如果指定則隨機抽取
            description: 實驗描述
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            流程結果字典
        """
        start_time = time.time()
        self.logger.info(f"開始執行完整面向分析流程：{dataset_type}")
        
        # 清理CUDA記憶體（如果有）
        try:
            import torch
            if torch.cuda.is_available():
                self.logger.info("清理CUDA記憶體以確保穩定運行")
                torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass
        
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("面向分析流程")
            logger = ConsoleOutputManager.setup_console_logger("pipeline", log_file)
        else:
            logger = self.logger
        
        try:
            # 1. 導入數據
            logger.info("步驟 1: 導入並預處理數據")
            df = self._import_data(dataset_type, dataset_params, sample_size, console_output=False)
            
            if df.empty:
                logger.error("未能成功導入數據")
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                return {"success": False, "error": "數據導入失敗"}
            
            logger.info(f"成功導入 {len(df)} 條數據記錄")
            
            # 創建結果集
            result_id = self.result_manager.create_result_set(
                dataset_name=dataset_type, 
                description=description or f"{dataset_type} 面向分析 ({len(df)} 條記錄)"
            )
            
            # 保存處理後的數據
            data_file = self.result_manager.save_data(result_id, df, 'processed_data')
            logger.info(f"預處理數據已保存，結果ID: {result_id}")
            
            # 2. 生成BERT嵌入
            logger.info("步驟 2: 生成BERT嵌入向量")
            try:
                bert_embeddings = self.bert_embedder.process_dataset(
                    df, text_column='processed_text', id_column='id', console_output=False
                )
                
                # 保存BERT嵌入
                bert_file = self.result_manager.save_data(result_id, bert_embeddings, 'bert_embeddings')
                logger.info(f"BERT嵌入已生成，共 {len(bert_embeddings)} 條向量")
            except Exception as e:
                logger.error(f"生成BERT嵌入時發生錯誤: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 嘗試清理CUDA記憶體
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                    
                # 更新結果集狀態
                self.result_manager.update_result_set(
                    result_id, 
                    status="failed",
                    error_message=f"BERT處理失敗: {str(e)}"
                )
                
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                    
                return {
                    "success": False, 
                    "error": f"生成BERT嵌入時發生錯誤: {str(e)}",
                    "result_id": result_id
                }
            
            # 3. 提取主題
            logger.info("步驟 3: 提取LDA主題模型")
            try:
                lda_results = self.topic_extractor.extract_topics(
                    df, text_column='processed_text', console_output=False
                )
                
                # 保存LDA結果
                lda_file = self.result_manager.save_data(result_id, lda_results, 'lda_topics')
                
                topics = lda_results['topics']
                topic_summary = "\n".join([f"主題 {t['id']+1}: {', '.join(t['top_words'][:5])}" for t in topics])
                logger.info(f"已提取 {len(topics)} 個主題，概覽：\n{topic_summary}")
            except Exception as e:
                logger.error(f"提取LDA主題時發生錯誤: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 更新結果集狀態
                self.result_manager.update_result_set(
                    result_id, 
                    status="failed",
                    error_message=f"LDA主題提取失敗: {str(e)}"
                )
                
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                    
                return {
                    "success": False, 
                    "error": f"提取LDA主題時發生錯誤: {str(e)}",
                    "result_id": result_id
                }
            
            # 4. 計算面向向量
            logger.info("步驟 4: 計算面向向量")
            try:
                aspect_results = self.aspect_calculator.calculate_aspect_vectors(
                    df, text_column='processed_text', id_column='id',
                    bert_embeddings=bert_embeddings, lda_results=lda_results,
                    console_output=False
                )
                
                # 保存面向結果
                aspect_file = self.result_manager.save_data(result_id, aspect_results, 'aspect_vectors')
                
                # 顯示面向結果概覽
                aspect_labels = aspect_results['aspect_labels']
                logger.info(f"已計算 {len(aspect_labels)} 個面向向量")
            except Exception as e:
                logger.error(f"計算面向向量時發生錯誤: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 更新結果集狀態
                self.result_manager.update_result_set(
                    result_id, 
                    status="failed",
                    error_message=f"面向向量計算失敗: {str(e)}"
                )
                
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                    
                return {
                    "success": False, 
                    "error": f"計算面向向量時發生錯誤: {str(e)}",
                    "result_id": result_id
                }
            
            # 5. 評估結果
            logger.info("步驟 5: 評估結果")
            try:
                evaluation_results = self.evaluator.evaluate_aspects(
                    df, aspect_results, console_output=False
                )
                
                # 保存評估結果
                eval_file = self.result_manager.save_data(result_id, evaluation_results, 'evaluation')
                
                # 更新指標
                self.result_manager.save_metrics(result_id, evaluation_results['metrics'])
            except Exception as e:
                logger.error(f"評估結果時發生錯誤: {str(e)}")
                logger.error(traceback.format_exc())
                
                # 繼續執行報告生成，這個錯誤不是致命的
                evaluation_results = {
                    "metrics": {"error": "評估失敗"},
                    "evaluation_details": {}
                }
            
            # 6. 生成報告
            logger.info("步驟 6: 生成報告")
            try:
                report_data = {
                    'dataset': dataset_type,
                    'description': description,
                    'sample_size': len(df),
                    'processed_data': df,
                    'aspect_results': aspect_results,
                    'evaluation': evaluation_results,
                    'lda_results': lda_results
                }
                
                report_file = self.report_generator.generate_report(
                    result_id, report_data, console_output=False
                )
            except Exception as e:
                logger.error(f"生成報告時發生錯誤: {str(e)}")
                logger.error(traceback.format_exc())
                report_file = None
            
            # 7. 完成流程
            end_time = time.time()
            duration = end_time - start_time
            
            # 更新結果集狀態
            self.result_manager.update_result_set(
                result_id, 
                status="completed",
                processing_time=f"{duration:.2f} 秒"
            )
            
            # 準備結果
            result = {
                "success": True,
                "result_id": result_id,
                "dataset_type": dataset_type,
                "sample_size": len(df),
                "num_aspects": len(aspect_labels),
                "processing_time": duration,
                "files": {
                    "processed_data": data_file,
                    "bert_embeddings": bert_file,
                    "lda_topics": lda_file,
                    "aspect_vectors": aspect_file,
                    "evaluation": eval_file,
                    "report": report_file
                }
            }
            
            # 儲存結果
            self.results[result_id] = result
            
            logger.info(f"流程完成！耗時: {duration:.2f} 秒")
            
            # 清理CUDA記憶體（如果有）
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
                
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return result
            
        except Exception as e:
            error_msg = f"面向分析流程執行出錯: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # 嘗試更新結果集狀態（如果已創建）
            try:
                if 'result_id' in locals():
                    self.result_manager.update_result_set(
                        result_id, 
                        status="failed",
                        error_message=str(e)
                    )
            except:
                pass
            
            # 清理CUDA記憶體（如果有）
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
                
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return {"success": False, "error": error_msg}
    
    def _import_data(self, dataset_type: str, dataset_params: Dict[str, Any], 
                   sample_size: Optional[int] = None, console_output: bool = False) -> pd.DataFrame:
        """
        導入數據
        
        Args:
            dataset_type: 數據集類型
            dataset_params: 數據集參數
            sample_size: 樣本大小
            console_output: 是否顯示處理進度
            
        Returns:
            處理後的DataFrame
        """
        if dataset_type.lower() == 'imdb':
            return self.data_importer.import_imdb_data(
                file_path=dataset_params.get('file_path'),
                sample_size=sample_size,
                min_length=dataset_params.get('min_length', 50),
                console_output=console_output
            )
        
        elif dataset_type.lower() == 'amazon':
            return self.data_importer.import_amazon_data(
                file_path=dataset_params.get('file_path'),
                product_category=dataset_params.get('product_category'),
                min_rating=dataset_params.get('min_rating', 1.0),
                max_rating=dataset_params.get('max_rating', 5.0),
                sample_size=sample_size,
                console_output=console_output
            )
        
        elif dataset_type.lower() == 'yelp':
            return self.data_importer.import_yelp_data(
                review_file=dataset_params.get('review_file'),
                business_file=dataset_params.get('business_file'),
                category_filter=dataset_params.get('category_filter', 'Restaurants'),
                min_stars=dataset_params.get('min_stars', 1),
                max_stars=dataset_params.get('max_stars', 5),
                sample_size=sample_size,
                console_output=console_output
            )
        
        elif dataset_type.lower() == 'custom' and 'dataframe' in dataset_params:
            # 使用自定義DataFrame
            df = dataset_params['dataframe']
            
            # 確保必要的列存在
            required_cols = ['id', 'text', 'processed_text']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'id':
                        df['id'] = [f"custom_{i}" for i in range(len(df))]
                    elif col == 'processed_text' and 'text' in df.columns:
                        df['processed_text'] = df['text'].apply(self.data_importer._preprocess_text)
                    else:
                        self.logger.error(f"自定義數據缺少必要列: {col}")
                        return pd.DataFrame()
            
            # 如果指定了樣本大小，隨機抽取
            if sample_size and sample_size < len(df):
                df = df.sample(sample_size, random_state=42)
            
            return df
        
        else:
            self.logger.error(f"不支持的數據集類型: {dataset_type}")
            return pd.DataFrame()
    
    def load_result(self, result_id: str) -> Dict[str, Any]:
        """
        載入流程結果
        
        Args:
            result_id: 結果ID
            
        Returns:
            結果字典
        """
        if result_id in self.results:
            return self.results[result_id]
        
        # 嘗試從結果管理器載入
        result_set = self.result_manager.get_result_set(result_id)
        if not result_set:
            return {"success": False, "error": f"未找到結果ID: {result_id}"}
        
        # 構造結果字典
        result = {
            "success": True,
            "result_id": result_id,
            "dataset_type": result_set.get('dataset_name', '未知數據集'),
            "status": result_set.get('status', 'unknown'),
            "description": result_set.get('description', '')
        }
        
        self.results[result_id] = result
        return result
    
    def compare_results(self, result_ids: List[str]) -> Dict[str, Any]:
        """
        比較多個實驗結果
        
        Args:
            result_ids: 結果ID列表
            
        Returns:
            比較結果
        """
        if not result_ids or len(result_ids) < 2:
            return {"success": False, "error": "至少需要兩個結果ID進行比較"}
        
        try:
            # 載入所有結果
            results = []
            result_sets = []
            
            for result_id in result_ids:
                result_set = self.result_manager.get_result_set(result_id)
                if not result_set:
                    return {"success": False, "error": f"未找到結果ID: {result_id}"}
                
                result_sets.append(result_set)
                
                # 載入評估結果
                evaluation = None
                if 'files' in result_set and 'evaluation' in result_set['files']:
                    eval_files = result_set['files']['evaluation']
                    if eval_files:
                        evaluation = self.result_manager.load_data(result_id, 'evaluation')
                
                results.append({
                    "result_id": result_id,
                    "dataset_name": result_set.get('dataset_name', '未知'),
                    "metrics": result_set.get('metrics', {}),
                    "evaluation": evaluation
                })
            
            # 比較指標
            comparison = {
                "success": True,
                "results": results,
                "metrics_comparison": {}
            }
            
            # 合併所有指標
            all_metrics = set()
            for result in results:
                all_metrics.update(result.get('metrics', {}).keys())
            
            # 構建比較表
            for metric in all_metrics:
                comparison["metrics_comparison"][metric] = [
                    result.get('metrics', {}).get(metric, None) for result in results
                ]
            
            # 生成比較報告
            report_file = self.report_generator.generate_comparison_report(
                result_ids, comparison, console_output=False
            )
            
            comparison["report_file"] = report_file
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"比較結果時出錯: {str(e)}")
            return {"success": False, "error": f"比較結果時出錯: {str(e)}"}

# 使用範例
if __name__ == "__main__":
    # 初始化分析流程
    pipeline = Pipeline()
    
    # 準備IMDB數據集參數
    imdb_params = {
        "file_path": None,  # 使用默認路徑
        "min_length": 50
    }
    
    # 運行面向分析流程
    result = pipeline.run_full_pipeline(
        dataset_type='imdb',
        dataset_params=imdb_params,
        sample_size=100,
        description="IMDB測試分析流程",
        console_output=True
    )
    
    # 顯示結果概覽
    if result["success"]:
        print(f"分析流程成功完成！結果ID: {result['result_id']}")
        print(f"處理時間: {result['processing_time']:.2f} 秒")
        print(f"樣本大小: {result['sample_size']}")
        print(f"識別的面向數量: {result['num_aspects']}")
    else:
        print(f"分析流程出錯: {result.get('error', '未知錯誤')}")
