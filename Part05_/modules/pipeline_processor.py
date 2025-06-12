#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流程處理器模組 - 整合多種文本編碼器和分類方法
支援靈活的文本分析流程配置
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import time

from .text_encoders import TextEncoderFactory, BaseTextEncoder
from .classification_methods import ClassificationMethodFactory, BaseClassificationMethod

logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """分析流程管道類"""
    
    def __init__(self, output_dir: Optional[str] = None, progress_callback=None):
        """
        初始化分析流程
        
        Args:
            output_dir: 輸出目錄
            progress_callback: 進度回調函數
        """
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        
        # 創建輸出目錄
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # 流程組件
        self.encoder = None
        self.classifier = None
        
        # 結果存儲
        self.encoding_results = {}
        self.classification_results = {}
        self.pipeline_config = {}
        
        logger.info("分析流程已初始化")
    
    def configure_pipeline(self, 
                          encoder_type: str,
                          classifier_type: str,
                          encoder_config: Optional[Dict] = None,
                          classifier_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        配置分析流程
        
        Args:
            encoder_type: 編碼器類型 ('bert', 'gpt', 't5', 'cnn', 'elmo')
            classifier_type: 分類器類型 ('sentiment', 'lda', 'bertopic', 'nmf', 'clustering')
            encoder_config: 編碼器配置參數
            classifier_config: 分類器配置參數
            
        Returns:
            Dict: 配置信息
        """
        if encoder_config is None:
            encoder_config = {}
        if classifier_config is None:
            classifier_config = {}
        
        # 添加共同參數
        encoder_config.update({
            'output_dir': self.output_dir,
            'progress_callback': self.progress_callback
        })
        classifier_config.update({
            'output_dir': self.output_dir,
            'progress_callback': self.progress_callback
        })
        
        # 創建編碼器
        try:
            self.encoder = TextEncoderFactory.create_encoder(encoder_type, **encoder_config)
            logger.info(f"已配置編碼器: {encoder_type}")
        except Exception as e:
            raise ValueError(f"編碼器配置失敗: {str(e)}")
        
        # 創建分類器
        try:
            self.classifier = ClassificationMethodFactory.create_method(classifier_type, **classifier_config)
            logger.info(f"已配置分類器: {classifier_type}")
        except Exception as e:
            raise ValueError(f"分類器配置失敗: {str(e)}")
        
        # 保存配置
        self.pipeline_config = {
            'encoder_type': encoder_type,
            'classifier_type': classifier_type,
            'encoder_config': encoder_config,
            'classifier_config': classifier_config,
            'configured_at': datetime.now().isoformat()
        }
        
        # 檢查兼容性
        compatibility_info = self._check_compatibility()
        
        config_info = {
            'pipeline_configured': True,
            'encoder': {
                'type': encoder_type,
                'name': self.encoder.__class__.__name__,
                'embedding_dim': self.encoder.get_embedding_dim()
            },
            'classifier': {
                'type': classifier_type,
                'name': self.classifier.__class__.__name__,
                'method_name': self.classifier.get_method_name()
            },
            'compatibility': compatibility_info,
            'config': self.pipeline_config
        }
        
        self._save_config(config_info)
        
        logger.info("流程配置完成")
        return config_info
    
    def _check_compatibility(self) -> Dict[str, Any]:
        """檢查組件兼容性"""
        compatibility = {
            'status': 'compatible',
            'warnings': [],
            'recommendations': []
        }
        
        encoder_type = self.pipeline_config['encoder_type']
        classifier_type = self.pipeline_config['classifier_type']
        
        # 檢查是否需要原始文本
        text_dependent_classifiers = ['lda', 'bertopic', 'nmf']
        if classifier_type in text_dependent_classifiers:
            compatibility['requires_text'] = True
            compatibility['recommendations'].append(
                f"{classifier_type}需要原始文本，請確保在運行時提供文本數據"
            )
        else:
            compatibility['requires_text'] = False
        
        # 檢查是否需要標籤
        if classifier_type == 'sentiment':
            compatibility['requires_labels'] = True
            compatibility['recommendations'].append(
                "情感分析需要標籤數據，請確保提供正確的標籤"
            )
        else:
            compatibility['requires_labels'] = False
        
        # 性能建議
        if encoder_type in ['gpt', 't5', 'elmo'] and classifier_type == 'bertopic':
            compatibility['warnings'].append(
                "大型編碼器與BERTopic組合可能需要大量計算資源"
            )
        
        if encoder_type == 'cnn' and classifier_type in ['lda', 'bertopic', 'nmf']:
            compatibility['recommendations'].append(
                "CNN編碼器與主題建模組合，建議調整CNN的向量維度以獲得更好效果"
            )
        
        return compatibility
    
    def run_pipeline(self, 
                    texts: pd.Series,
                    labels: Optional[pd.Series] = None,
                    save_intermediates: bool = True) -> Dict[str, Any]:
        """
        運行完整的分析流程
        
        Args:
            texts: 輸入文本
            labels: 標籤（如果需要）
            save_intermediates: 是否保存中間結果
            
        Returns:
            Dict: 完整的分析結果
        """
        if self.encoder is None or self.classifier is None:
            raise ValueError("流程未配置，請先調用configure_pipeline方法")
        
        start_time = time.time()
        
        # 通知流程開始
        if self.progress_callback:
            self.progress_callback('phase', {
                'phase_name': '開始分析流程',
                'current_phase': 1,
                'total_phases': 3
            })
        
        print(f"\n🚀 開始執行分析流程")
        print(f"📋 配置信息:")
        print(f"   • 編碼器: {self.pipeline_config['encoder_type']}")
        print(f"   • 分類器: {self.pipeline_config['classifier_type']}")
        print(f"   • 文本數量: {len(texts)}")
        print("="*60)
        
        try:
            # 階段1: 文本編碼
            print(f"\n🔤 階段 1/3: 文本向量化")
            print("-" * 40)
            
            if self.progress_callback:
                self.progress_callback('phase', {
                    'phase_name': f'文本編碼 ({self.pipeline_config["encoder_type"]})',
                    'current_phase': 1,
                    'total_phases': 3
                })
            
            embeddings = self.encoder.encode(texts)
            
            self.encoding_results = {
                'encoder_type': self.pipeline_config['encoder_type'],
                'embedding_shape': embeddings.shape,
                'embedding_dim': self.encoder.get_embedding_dim(),
                'n_samples': len(texts),
                'encoding_completed_at': datetime.now().isoformat()
            }
            
            print(f"✅ 編碼完成 - 向量形狀: {embeddings.shape}")
            
            # 階段2: 分類/分析
            print(f"\n🎯 階段 2/3: {self.classifier.get_method_name()}")
            print("-" * 40)
            
            if self.progress_callback:
                self.progress_callback('phase', {
                    'phase_name': f'分類分析 ({self.pipeline_config["classifier_type"]})',
                    'current_phase': 2,
                    'total_phases': 3
                })
            
            # 檢查是否需要文本和標籤
            fit_kwargs = {'features': embeddings}
            if self._check_compatibility()['requires_text']:
                fit_kwargs['texts'] = texts
            if self._check_compatibility()['requires_labels']:
                if labels is None:
                    raise ValueError(f"{self.classifier.get_method_name()}需要標籤數據")
                fit_kwargs['labels'] = labels.values if hasattr(labels, 'values') else labels
            
            classification_result = self.classifier.fit(**fit_kwargs)
            
            self.classification_results = classification_result
            
            print(f"✅ 分析完成")
            
            # 階段3: 結果整合
            print(f"\n📊 階段 3/3: 結果整合")
            print("-" * 40)
            
            if self.progress_callback:
                self.progress_callback('phase', {
                    'phase_name': '結果整合',
                    'current_phase': 3,
                    'total_phases': 3
                })
            
            # 整合結果
            pipeline_results = self._integrate_results(embeddings, start_time)
            
            # 保存結果
            if save_intermediates:
                self._save_pipeline_results(pipeline_results)
            
            print(f"✅ 流程完成")
            print(f"⏱️  總用時: {pipeline_results['processing_time']:.2f}秒")
            print(f"📁 結果保存在: {self.output_dir}")
            print("="*60)
            
            return pipeline_results
            
        except Exception as e:
            error_msg = f"流程執行失敗: {str(e)}"
            logger.error(error_msg)
            
            # 嘗試GPU記憶體清理
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            except:
                pass
            
            raise RuntimeError(error_msg) from e
    
    def _integrate_results(self, embeddings: np.ndarray, start_time: float) -> Dict[str, Any]:
        """整合分析結果"""
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 基本信息
        integrated_results = {
            'pipeline_info': {
                'encoder_type': self.pipeline_config['encoder_type'],
                'classifier_type': self.pipeline_config['classifier_type'],
                'pipeline_config': self.pipeline_config,
                'processing_time': processing_time,
                'completed_at': datetime.now().isoformat()
            },
            'encoding_results': self.encoding_results,
            'classification_results': self.classification_results,
            'embeddings_info': {
                'shape': embeddings.shape,
                'dtype': str(embeddings.dtype),
                'memory_usage_mb': embeddings.nbytes / (1024 * 1024)
            }
        }
        
        # 根據分類器類型添加特定信息
        classifier_type = self.pipeline_config['classifier_type']
        
        if classifier_type == 'sentiment':
            # 情感分析特定信息
            if 'test_accuracy' in self.classification_results:
                integrated_results['summary'] = {
                    'analysis_type': '情感分析',
                    'accuracy': self.classification_results['test_accuracy'],
                    'f1_score': self.classification_results.get('f1_score', 0),
                    'n_classes': self.classification_results.get('n_classes', 0)
                }
        
        elif classifier_type in ['lda', 'bertopic', 'nmf']:
            # 主題建模特定信息
            n_topics = self.classification_results.get('n_topics', 0)
            integrated_results['summary'] = {
                'analysis_type': '主題建模',
                'method': classifier_type.upper(),
                'n_topics': n_topics,
                'corpus_size': self.classification_results.get('corpus_size', 0)
            }
            
            # 添加主題質量指標
            if classifier_type == 'lda' and 'coherence_score' in self.classification_results:
                integrated_results['summary']['coherence_score'] = self.classification_results['coherence_score']
            elif classifier_type == 'nmf' and 'reconstruction_error' in self.classification_results:
                integrated_results['summary']['reconstruction_error'] = self.classification_results['reconstruction_error']
        
        elif classifier_type == 'clustering':
            # 聚類分析特定信息
            integrated_results['summary'] = {
                'analysis_type': '聚類分析',
                'clustering_method': self.classification_results.get('clustering_method', ''),
                'n_clusters': self.classification_results.get('n_clusters', 0),
                'silhouette_score': self.classification_results.get('silhouette_score', None)
            }
        
        return integrated_results
    
    def predict_new_data(self, 
                        new_texts: pd.Series,
                        save_results: bool = True) -> Dict[str, Any]:
        """
        對新數據進行預測
        
        Args:
            new_texts: 新的文本數據
            save_results: 是否保存預測結果
            
        Returns:
            Dict: 預測結果
        """
        if self.encoder is None or self.classifier is None:
            raise ValueError("流程未配置或未訓練，請先運行完整流程")
        
        print(f"\n🔮 開始預測新數據")
        print(f"📊 新數據量: {len(new_texts)}")
        
        # 編碼新文本
        print("🔤 編碼新文本...")
        new_embeddings = self.encoder.encode(new_texts)
        
        # 預測
        print(f"🎯 執行{self.classifier.get_method_name()}預測...")
        predict_kwargs = {'features': new_embeddings}
        if self._check_compatibility()['requires_text']:
            predict_kwargs['texts'] = new_texts
        
        predictions = self.classifier.predict(**predict_kwargs)
        
        # 整合預測結果
        prediction_results = {
            'prediction_info': {
                'encoder_type': self.pipeline_config['encoder_type'],
                'classifier_type': self.pipeline_config['classifier_type'],
                'n_samples': len(new_texts),
                'predicted_at': datetime.now().isoformat()
            },
            'predictions': predictions,
            'embedding_info': {
                'shape': new_embeddings.shape,
                'dtype': str(new_embeddings.dtype)
            }
        }
        
        if save_results:
            self._save_prediction_results(prediction_results)
        
        print(f"✅ 預測完成")
        return prediction_results
    
    def get_available_encoders(self) -> Dict[str, Any]:
        """獲取可用的編碼器信息"""
        return TextEncoderFactory.get_encoder_info()
    
    def get_available_classifiers(self) -> Dict[str, Any]:
        """獲取可用的分類器信息"""
        return ClassificationMethodFactory.get_method_info()
    
    def get_pipeline_recommendations(self, data_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        根據數據特點推薦流程配置
        
        Args:
            data_info: 數據信息 {'n_samples': int, 'has_labels': bool, 'avg_text_length': int, 'language': str}
            
        Returns:
            List[Dict]: 推薦的配置列表
        """
        recommendations = []
        
        n_samples = data_info.get('n_samples', 0)
        has_labels = data_info.get('has_labels', False)
        avg_length = data_info.get('avg_text_length', 100)
        
        # 基於數據量的推薦
        if n_samples < 1000:
            # 小數據集
            if has_labels:
                recommendations.append({
                    'name': '小數據集情感分析',
                    'encoder': 'bert',
                    'classifier': 'sentiment',
                    'reason': '小數據集適合使用預訓練BERT配合簡單分類器',
                    'priority': 'high'
                })
            
            recommendations.append({
                'name': '小數據集主題發現',
                'encoder': 'bert',
                'classifier': 'lda',
                'reason': '小數據集適合LDA進行主題建模',
                'priority': 'medium'
            })
        
        elif n_samples < 10000:
            # 中等數據集
            if has_labels:
                recommendations.append({
                    'name': '中等數據集情感分析',
                    'encoder': 'bert',
                    'classifier': 'sentiment',
                    'classifier_config': {'model_type': 'xgboost'},
                    'reason': '中等數據集可以使用XGBoost獲得更好性能',
                    'priority': 'high'
                })
            
            recommendations.append({
                'name': '中等數據集主題建模',
                'encoder': 'bert',
                'classifier': 'bertopic',
                'reason': 'BERTopic在中等數據集上表現優異',
                'priority': 'high'
            })
            
        else:
            # 大數據集
            recommendations.append({
                'name': '大數據集快速分析',
                'encoder': 'cnn',
                'classifier': 'clustering',
                'reason': 'CNN編碼器配合聚類分析，適合大數據集快速探索',
                'priority': 'medium'
            })
            
            if has_labels:
                recommendations.append({
                    'name': '大數據集精確分析',
                    'encoder': 't5',
                    'classifier': 'sentiment',
                    'reason': 'T5編碼器在大數據集上可能獲得更好的泛化性能',
                    'priority': 'high'
                })
        
        # 基於文本長度的推薦
        if avg_length > 500:
            recommendations.append({
                'name': '長文本分析',
                'encoder': 'bert',
                'encoder_config': {'model_name': 'bert-base-uncased'},
                'classifier': 'nmf',
                'reason': '長文本適合使用NMF進行主題分解',
                'priority': 'medium'
            })
        
        # 無監督學習推薦
        if not has_labels:
            recommendations.append({
                'name': '無標籤探索性分析',
                'encoder': 'bert',
                'classifier': 'clustering',
                'classifier_config': {'method': 'kmeans', 'n_clusters': 5},
                'reason': '無標籤數據適合聚類分析發現數據結構',
                'priority': 'high'
            })
        
        # 按優先級排序
        recommendations.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']])
        
        return recommendations
    
    def _save_config(self, config_info: Dict[str, Any]):
        """保存配置信息"""
        if self.output_dir:
            config_file = os.path.join(self.output_dir, "pipeline_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_info, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"配置已保存到: {config_file}")
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """保存流程結果"""
        if self.output_dir:
            results_file = os.path.join(self.output_dir, "pipeline_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"流程結果已保存到: {results_file}")
    
    def _save_prediction_results(self, results: Dict[str, Any]):
        """保存預測結果"""
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(self.output_dir, f"prediction_results_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"預測結果已保存到: {results_file}")

class MultiPipelineComparison:
    """多流程比較類"""
    
    def __init__(self, output_dir: Optional[str] = None, progress_callback=None):
        """
        初始化多流程比較
        
        Args:
            output_dir: 輸出目錄
            progress_callback: 進度回調函數
        """
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.pipelines = {}
        self.results = {}
        
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
    
    def add_pipeline_config(self, 
                           name: str,
                           encoder_type: str,
                           classifier_type: str,
                           encoder_config: Optional[Dict] = None,
                           classifier_config: Optional[Dict] = None):
        """添加流程配置"""
        self.pipelines[name] = {
            'encoder_type': encoder_type,
            'classifier_type': classifier_type,
            'encoder_config': encoder_config or {},
            'classifier_config': classifier_config or {}
        }
        logger.info(f"已添加流程配置: {name}")
    
    def run_comparison(self, 
                      texts: pd.Series,
                      labels: Optional[pd.Series] = None,
                      test_split: float = 0.2) -> Dict[str, Any]:
        """
        運行多流程比較
        
        Args:
            texts: 輸入文本
            labels: 標籤（如果有）
            test_split: 測試集比例（用於性能評估）
            
        Returns:
            Dict: 比較結果
        """
        if not self.pipelines:
            raise ValueError("沒有添加任何流程配置")
        
        print(f"\n🔬 開始多流程比較")
        print(f"📊 流程數量: {len(self.pipelines)}")
        print(f"📊 數據量: {len(texts)}")
        print("="*60)
        
        comparison_results = {
            'comparison_info': {
                'n_pipelines': len(self.pipelines),
                'n_samples': len(texts),
                'test_split': test_split,
                'started_at': datetime.now().isoformat()
            },
            'pipeline_results': {},
            'comparison_metrics': {},
            'recommendations': {}
        }
        
        # 分割數據（如果需要評估）
        if labels is not None and test_split > 0:
            from sklearn.model_selection import train_test_split
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=test_split, random_state=42, stratify=labels
            )
            use_train_data = True
        else:
            train_texts, test_texts = texts, None
            train_labels, test_labels = labels, None
            use_train_data = False
        
        # 運行每個流程
        for i, (name, config) in enumerate(self.pipelines.items(), 1):
            print(f"\n🔄 運行流程 {i}/{len(self.pipelines)}: {name}")
            print(f"   編碼器: {config['encoder_type']}")
            print(f"   分類器: {config['classifier_type']}")
            
            try:
                # 創建流程
                pipeline_output_dir = os.path.join(self.output_dir, f"pipeline_{name}")
                pipeline = AnalysisPipeline(
                    output_dir=pipeline_output_dir,
                    progress_callback=self.progress_callback
                )
                
                # 配置流程
                pipeline.configure_pipeline(
                    encoder_type=config['encoder_type'],
                    classifier_type=config['classifier_type'],
                    encoder_config=config['encoder_config'],
                    classifier_config=config['classifier_config']
                )
                
                # 運行流程
                start_time = time.time()
                pipeline_result = pipeline.run_pipeline(
                    texts=train_texts,
                    labels=train_labels,
                    save_intermediates=True
                )
                
                processing_time = time.time() - start_time
                
                # 評估（如果有測試數據）
                evaluation_results = {}
                if use_train_data and test_texts is not None:
                    try:
                        prediction_results = pipeline.predict_new_data(test_texts, save_results=True)
                        # 這裡可以添加評估邏輯
                        evaluation_results = {
                            'test_completed': True,
                            'test_samples': len(test_texts)
                        }
                    except Exception as e:
                        evaluation_results = {
                            'test_completed': False,
                            'test_error': str(e)
                        }
                
                # 記錄結果
                comparison_results['pipeline_results'][name] = {
                    'config': config,
                    'pipeline_result': pipeline_result,
                    'evaluation': evaluation_results,
                    'processing_time': processing_time,
                    'memory_usage': pipeline_result.get('embeddings_info', {}).get('memory_usage_mb', 0),
                    'status': 'success'
                }
                
                print(f"   ✅ 完成 (用時: {processing_time:.2f}秒)")
                
            except Exception as e:
                error_msg = str(e)
                comparison_results['pipeline_results'][name] = {
                    'config': config,
                    'status': 'failed',
                    'error': error_msg,
                    'processing_time': 0
                }
                print(f"   ❌ 失敗: {error_msg}")
                logger.error(f"流程 {name} 執行失敗: {error_msg}")
        
        # 生成比較分析
        comparison_results['comparison_metrics'] = self._analyze_comparison(comparison_results)
        comparison_results['recommendations'] = self._generate_recommendations(comparison_results)
        comparison_results['comparison_info']['completed_at'] = datetime.now().isoformat()
        
        # 保存比較結果
        self._save_comparison_results(comparison_results)
        
        print(f"\n📊 比較分析完成")
        print(f"📁 結果保存在: {self.output_dir}")
        print("="*60)
        
        return comparison_results
    
    def _analyze_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析比較結果"""
        pipeline_results = results['pipeline_results']
        successful_pipelines = {k: v for k, v in pipeline_results.items() if v['status'] == 'success'}
        
        if not successful_pipelines:
            return {'error': '沒有成功的流程可供比較'}
        
        # 性能統計
        processing_times = [v['processing_time'] for v in successful_pipelines.values()]
        memory_usages = [v['memory_usage'] for v in successful_pipelines.values()]
        
        comparison_metrics = {
            'successful_pipelines': len(successful_pipelines),
            'failed_pipelines': len(pipeline_results) - len(successful_pipelines),
            'performance_stats': {
                'avg_processing_time': np.mean(processing_times),
                'min_processing_time': np.min(processing_times),
                'max_processing_time': np.max(processing_times),
                'avg_memory_usage_mb': np.mean(memory_usages),
                'total_memory_usage_mb': np.sum(memory_usages)
            },
            'fastest_pipeline': min(successful_pipelines.items(), key=lambda x: x[1]['processing_time'])[0],
            'most_memory_efficient': min(successful_pipelines.items(), key=lambda x: x[1]['memory_usage'])[0]
        }
        
        return comparison_metrics
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成推薦建議"""
        metrics = results.get('comparison_metrics', {})
        
        if 'error' in metrics:
            return {'error': '無法生成推薦，沒有成功的流程'}
        
        recommendations = {
            'best_overall': None,
            'fastest': metrics.get('fastest_pipeline'),
            'most_efficient': metrics.get('most_memory_efficient'),
            'suggestions': []
        }
        
        # 綜合推薦邏輯（這裡可以根據具體需求調整）
        pipeline_results = results['pipeline_results']
        successful_pipelines = {k: v for k, v in pipeline_results.items() if v['status'] == 'success'}
        
        if successful_pipelines:
            # 簡單的評分機制（速度 + 記憶體效率）
            scores = {}
            for name, result in successful_pipelines.items():
                time_score = 1.0 / (result['processing_time'] + 1)  # 時間越短分數越高
                memory_score = 1.0 / (result['memory_usage'] + 1)   # 記憶體越少分數越高
                scores[name] = time_score + memory_score
            
            best_pipeline = max(scores.items(), key=lambda x: x[1])[0]
            recommendations['best_overall'] = best_pipeline
            
            # 生成建議
            if metrics['performance_stats']['avg_processing_time'] > 300:  # 5分鐘
                recommendations['suggestions'].append("考慮使用CNN編碼器以提高處理速度")
            
            if metrics['performance_stats']['avg_memory_usage_mb'] > 1000:  # 1GB
                recommendations['suggestions'].append("考慮使用較小的模型或批量處理以減少記憶體使用")
        
        return recommendations
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """保存比較結果"""
        if self.output_dir:
            results_file = os.path.join(self.output_dir, "multi_pipeline_comparison.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"比較結果已保存到: {results_file}")

# 便利函數
def create_simple_pipeline(encoder_type: str, 
                          classifier_type: str,
                          output_dir: Optional[str] = None,
                          **kwargs) -> AnalysisPipeline:
    """
    快速創建簡單的分析流程
    
    Args:
        encoder_type: 編碼器類型
        classifier_type: 分類器類型
        output_dir: 輸出目錄
        **kwargs: 其他配置參數
        
    Returns:
        AnalysisPipeline: 配置好的分析流程
    """
    pipeline = AnalysisPipeline(output_dir=output_dir)
    pipeline.configure_pipeline(
        encoder_type=encoder_type,
        classifier_type=classifier_type,
        **kwargs
    )
    return pipeline

def run_quick_analysis(texts: pd.Series,
                      labels: Optional[pd.Series] = None,
                      encoder_type: str = 'bert',
                      classifier_type: str = 'sentiment',
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    快速執行分析
    
    Args:
        texts: 輸入文本
        labels: 標籤（可選）
        encoder_type: 編碼器類型
        classifier_type: 分類器類型
        output_dir: 輸出目錄
        
    Returns:
        Dict: 分析結果
    """
    pipeline = create_simple_pipeline(encoder_type, classifier_type, output_dir)
    return pipeline.run_pipeline(texts, labels)