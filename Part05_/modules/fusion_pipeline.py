"""
融合流程管線 - 整合新的情感分析架構
包含文字預處理、三種注意力機制並行計算、門控融合和分類預測
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

# 導入必要的模組
from .text_preprocessor import TextPreprocessor
from .attention_fusion_network import AttentionFusionProcessor
from .sentiment_classifier import SentimentClassifier
from .run_manager import RunManager

# 匯入錯誤處理工具
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.error_handler import handle_error, handle_warning, handle_info
from utils.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class FusionPipeline:
    """融合流程管線 - 完整的新架構實現"""
    
    def __init__(self, output_dir: Optional[str] = None, encoder_type: str = 'bert', config: Optional[Dict] = None):
        """
        初始化融合流程管線
        
        Args:
            output_dir: 輸出目錄
            encoder_type: 編碼器類型
            config: 配置參數
        """
        self.output_dir = output_dir
        self.encoder_type = encoder_type
        self.config = config or {}
        
        # 初始化組件
        self.text_preprocessor = TextPreprocessor(output_dir)
        self.attention_fusion = AttentionFusionProcessor()
        self.classifier = SentimentClassifier(output_dir, encoder_type)
        
        # 初始化管理組件
        self.run_manager = RunManager(output_dir) if output_dir else None
        self.storage_manager = StorageManager(output_dir) if output_dir else None
        
        logger.info("融合流程管線已初始化")
        logger.info(f"編碼器類型: {encoder_type.upper()}")
        logger.info(f"輸出目錄: {output_dir}")
    
    def run_complete_pipeline(self, 
                            input_data: pd.DataFrame,
                            text_column: str,
                            test_size: float = 0.2,
                            save_results: bool = True) -> Dict[str, Any]:
        """
        運行完整的新架構流程
        
        Args:
            input_data: 輸入數據
            text_column: 文本欄位名稱
            test_size: 測試集比例
            save_results: 是否保存結果
            
        Returns:
            完整的流程結果
        """
        pipeline_start = datetime.now()
        logger.info("="*60)
        logger.info("開始新架構融合流程")
        logger.info("="*60)
        
        results = {
            'pipeline_info': {
                'start_time': pipeline_start.isoformat(),
                'architecture': 'attention_fusion_network',
                'encoder_type': self.encoder_type,
                'stages': []
            }
        }
        
        try:
            # 階段 1: 文字預處理和標籤轉換
            stage_start = datetime.now()
            print("📝 階段 1: 文字預處理...")
            
            processed_data = self.text_preprocessor.preprocess(input_data, text_column)
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            results['pipeline_info']['stages'].append({
                'stage': 1,
                'name': '文字預處理和標籤轉換',
                'duration_seconds': stage_time,
                'output_shape': processed_data.shape,
                'sentiment_distribution': processed_data['sentiment_numeric'].value_counts().to_dict() if 'sentiment_numeric' in processed_data.columns else {}
            })
            
            print(f"   ✅ 完成 - {processed_data.shape[0]} 條記錄 ({stage_time:.1f}s)")
            
            # 階段 2: 獲取文本嵌入
            stage_start = datetime.now()
            print(f"🤖 階段 2: 生成{self.encoder_type.upper()}嵌入向量...")
            
            embeddings = self._get_text_embeddings(processed_data)
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            results['pipeline_info']['stages'].append({
                'stage': 2,
                'name': f'{self.encoder_type.upper()}文本嵌入',
                'duration_seconds': stage_time,
                'embeddings_shape': embeddings.shape
            })
            
            print(f"   ✅ 完成 - 形狀{embeddings.shape} ({stage_time:.1f}s)")
            
            # 階段 3: 並行計算三種注意力機制特徵
            stage_start = datetime.now()
            print("⚡ 階段 3: 並行計算三種注意力機制...")
            
            similarity_features, keyword_features, self_attention_features, attention_info = \
                self.attention_fusion.compute_parallel_attention_features(embeddings, processed_data)
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            results['pipeline_info']['stages'].append({
                'stage': 3,
                'name': '並行注意力機制計算',
                'duration_seconds': stage_time,
                'attention_info': attention_info,
                'feature_shapes': {
                    'similarity': similarity_features.shape,
                    'keyword': keyword_features.shape,
                    'self_attention': self_attention_features.shape
                }
            })
            
            print(f"   ✅ 完成 - 三種注意力特徵 ({stage_time:.1f}s)")
            
            # 階段 4: 門控融合網路
            stage_start = datetime.now()
            print("🔀 階段 4: 門控融合網路...")
            
            fused_features, fusion_info = self.attention_fusion.fuse_attention_features(
                similarity_features, keyword_features, self_attention_features
            )
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            results['pipeline_info']['stages'].append({
                'stage': 4,
                'name': '門控融合網路',
                'duration_seconds': stage_time,
                'fusion_info': fusion_info,
                'fused_features_shape': fused_features.shape
            })
            
            # 顯示重要的權重信息
            avg_weights = fusion_info.get('average_weights', {})
            weights_str = f"相似度:{avg_weights.get('similarity', 0):.3f} | 關鍵詞:{avg_weights.get('keyword', 0):.3f} | 自注意力:{avg_weights.get('self_attention', 0):.3f}"
            print(f"   ✅ 完成 - 權重分配: {weights_str} ({stage_time:.1f}s)")
            
            # 階段 5: 分類器訓練和預測
            stage_start = datetime.now()
            print("🎯 階段 5: 訓練分類器...")
            
            # 準備分類特徵（使用融合特徵）
            features, labels = self.classifier.prepare_features(
                aspect_vectors={},  # 不需要aspect_vectors，因為使用融合特徵
                metadata=processed_data,
                fused_features=fused_features
            )
            
            # 訓練分類器
            classification_results = self.classifier.train(
                features, labels, 
                test_size=test_size,
                original_data=processed_data
            )
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            results['pipeline_info']['stages'].append({
                'stage': 5,
                'name': '分類器訓練和預測',
                'duration_seconds': stage_time,
                'classification_results': {
                    'test_accuracy': classification_results['test_accuracy'],
                    'test_f1': classification_results['test_f1'],
                    'test_precision': classification_results['test_precision'],
                    'test_recall': classification_results['test_recall']
                }
            })
            
            print(f"   ✅ 完成 - 準確率:{classification_results['test_accuracy']:.4f} ({stage_time:.1f}s)")
            
            # 階段 6: 與原始標籤比較準確率
            stage_start = datetime.now()
            print("📊 階段 6: 結果分析...")
            
            accuracy_analysis = self._analyze_accuracy(classification_results, processed_data)
            
            stage_time = (datetime.now() - stage_start).total_seconds()
            results['pipeline_info']['stages'].append({
                'stage': 6,
                'name': '準確率分析',
                'duration_seconds': stage_time,
                'accuracy_analysis': accuracy_analysis
            })
            
            print(f"   ✅ 完成 - 分析報告生成 ({stage_time:.1f}s)")
            
            # 整合所有結果
            pipeline_end = datetime.now()
            total_time = (pipeline_end - pipeline_start).total_seconds()
            
            results.update({
                'preprocessing_results': {
                    'original_shape': input_data.shape,
                    'processed_shape': processed_data.shape,
                    'sentiment_encoding': {
                        0: '負面',
                        1: '中性', 
                        2: '正面'
                    }
                },
                'attention_results': {
                    'similarity_features': similarity_features,
                    'keyword_features': keyword_features,
                    'self_attention_features': self_attention_features,
                    'attention_info': attention_info
                },
                'fusion_results': {
                    'fused_features': fused_features,
                    'fusion_info': fusion_info,
                    'gate_weights': fusion_info.get('average_weights', {})
                },
                'classification_results': classification_results,
                'accuracy_analysis': accuracy_analysis,
                'pipeline_info': {
                    **results['pipeline_info'],
                    'end_time': pipeline_end.isoformat(),
                    'total_duration_seconds': total_time,
                    'success': True
                }
            })
            
            # 顯示簡潔的最終結果
            print("\n" + "="*50)
            print("🎉 新融合架構完成！")
            print("="*50)
            print(f"⏱️  總耗時: {total_time:.1f} 秒")
            print(f"🎯 最終準確率: {classification_results['test_accuracy']:.4f}")
            print(f"📈 F1分數: {classification_results['test_f1']:.4f}")
            
            # 顯示門控權重分配
            avg_weights = fusion_info.get('average_weights', {})
            print(f"⚖️  權重分配:")
            print(f"   相似度注意力: {avg_weights.get('similarity', 0):.3f}")
            print(f"   關鍵詞注意力: {avg_weights.get('keyword', 0):.3f}")
            print(f"   自注意力機制: {avg_weights.get('self_attention', 0):.3f}")
            
            # 顯示各類別性能
            per_class = accuracy_analysis.get('per_class_metrics', {})
            if per_class:
                print(f"📊 各類別F1分數:")
                for class_name, metrics in per_class.items():
                    print(f"   {class_name}: {metrics.get('f1_score', 0):.3f}")
            
            print("="*50)
            
            # 保存結果
            if save_results and self.output_dir:
                self._save_pipeline_results(results)
            
            return results
            
        except Exception as e:
            error_time = datetime.now()
            error_info = {
                'error': str(e),
                'error_time': error_time.isoformat(),
                'error_stage': len(results['pipeline_info']['stages']) + 1
            }
            results['pipeline_info'].update(error_info)
            
            logger.error(f"流程在階段 {error_info['error_stage']} 發生錯誤: {str(e)}")
            handle_error(e, "融合流程管線")
            raise
    
    def _get_text_embeddings(self, processed_data: pd.DataFrame) -> np.ndarray:
        """獲取文本嵌入向量"""
        # 嘗試從存儲管理器獲取已存在的嵌入
        if self.storage_manager:
            existing_path = self.storage_manager.check_existing_embeddings(self.encoder_type)
            if existing_path:
                logger.info(f"載入已存在的{self.encoder_type.upper()}嵌入向量: {existing_path}")
                return np.load(existing_path)
        
        # 如果沒有找到，生成新的嵌入
        logger.info(f"生成新的{self.encoder_type.upper()}嵌入向量...")
        
        if self.encoder_type == 'bert':
            from .bert_encoder import BertEncoder
            encoder = BertEncoder(output_dir=self.output_dir)
        else:
            # 使用模組化編碼器
            try:
                from .text_encoders import TextEncoderFactory
                encoder = TextEncoderFactory.create_encoder(
                    self.encoder_type, 
                    output_dir=self.output_dir
                )
            except Exception as e:
                logger.warning(f"無法創建{self.encoder_type}編碼器，回退到BERT: {e}")
                from .bert_encoder import BertEncoder
                encoder = BertEncoder(output_dir=self.output_dir)
                self.encoder_type = 'bert'
        
        embeddings = encoder.encode(processed_data['processed_text'])
        
        # 保存嵌入向量
        if self.storage_manager:
            self.storage_manager.save_embeddings(embeddings, self.encoder_type)
        
        return embeddings
    
    def _analyze_accuracy(self, classification_results: Dict, processed_data: pd.DataFrame) -> Dict[str, Any]:
        """分析分類準確率"""
        analysis = {
            'overall_metrics': {
                'test_accuracy': classification_results['test_accuracy'],
                'test_precision': classification_results['test_precision'],
                'test_recall': classification_results['test_recall'],
                'test_f1': classification_results['test_f1']
            },
            'confusion_matrix': classification_results['confusion_matrix'],
            'classification_report': classification_results['classification_report'],
            'label_mapping': {
                0: '負面',
                1: '中性',
                2: '正面'
            }
        }
        
        # 分析每個類別的性能
        per_class_analysis = {}
        class_names = ['負面', '中性', '正面']
        
        for i, class_name in enumerate(class_names):
            if i < len(classification_results.get('per_class_precision', [])):
                per_class_analysis[class_name] = {
                    'precision': classification_results['per_class_precision'][i],
                    'recall': classification_results['per_class_recall'][i],
                    'f1_score': classification_results['per_class_f1'][i],
                    'support': classification_results['per_class_support'][i]
                }
        
        analysis['per_class_metrics'] = per_class_analysis
        
        # 預測詳情分析
        if 'prediction_details' in classification_results:
            pred_details = classification_results['prediction_details']
            
            # 計算預測分佈
            pred_counts = {}
            true_counts = {}
            
            for true_label, pred_label in zip(pred_details['true_labels'], pred_details['predicted_labels']):
                true_name = class_names[true_label] if true_label < len(class_names) else f'類別_{true_label}'
                pred_name = class_names[pred_label] if pred_label < len(class_names) else f'類別_{pred_label}'
                
                true_counts[true_name] = true_counts.get(true_name, 0) + 1
                pred_counts[pred_name] = pred_counts.get(pred_name, 0) + 1
            
            analysis['prediction_distribution'] = {
                'true_labels': true_counts,
                'predicted_labels': pred_counts
            }
        
        return analysis
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """保存流程結果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.storage_manager:
                # 使用存儲管理器保存結果
                self.storage_manager.save_analysis_results(
                    results,
                    "fusion_pipeline",
                    f"fusion_pipeline_results_{timestamp}.json"
                )
                
                # 單獨保存融合特徵
                if 'fusion_results' in results and 'fused_features' in results['fusion_results']:
                    fused_features = results['fusion_results']['fused_features']
                    self.storage_manager.save_features(
                        fused_features,
                        f"fused_features_{timestamp}.npy"
                    )
                
                logger.info("流程結果已通過存儲管理器保存")
            else:
                # 直接保存到輸出目錄
                results_file = os.path.join(self.output_dir, f"fusion_pipeline_results_{timestamp}.json")
                
                # 準備可序列化的結果
                serializable_results = self._make_json_serializable(results)
                
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_results, f, ensure_ascii=False, indent=2)
                
                logger.info(f"流程結果已保存到: {results_file}")
                
        except Exception as e:
            logger.error(f"保存流程結果時發生錯誤: {str(e)}")
    
    def _make_json_serializable(self, obj):
        """將對象轉換為JSON可序列化格式"""
        if isinstance(obj, np.ndarray):
            return {
                '__type__': 'numpy_array',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'data': obj.tolist()
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def compare_with_baseline(self, 
                            input_data: pd.DataFrame,
                            text_column: str,
                            baseline_methods: List[str] = None) -> Dict[str, Any]:
        """
        與基準方法比較性能
        
        Args:
            input_data: 輸入數據
            text_column: 文本欄位
            baseline_methods: 基準方法列表
            
        Returns:
            比較結果
        """
        if baseline_methods is None:
            baseline_methods = ['no', 'similarity', 'keyword', 'self_attention', 'combined']
        
        logger.info("開始與基準方法比較...")
        
        # 運行新架構
        logger.info("運行新融合架構...")
        fusion_results = self.run_complete_pipeline(input_data, text_column, save_results=False)
        
        # 運行基準方法
        baseline_results = {}
        
        # 這裡可以集成現有的注意力分析方法
        from .attention_processor import AttentionProcessor
        processor = AttentionProcessor(self.output_dir, encoder_type=self.encoder_type)
        
        # 準備輸入文件
        temp_file = os.path.join(self.output_dir, "temp_comparison_data.csv")
        processed_data = self.text_preprocessor.preprocess(input_data, text_column)
        processed_data.to_csv(temp_file, index=False)
        
        try:
            # 運行傳統注意力分析
            attention_results = processor.process_with_attention(
                input_file=temp_file,
                attention_types=baseline_methods,
                save_results=False
            )
            
            # 評估基準方法的分類性能
            classifier_baseline = SentimentClassifier(self.output_dir, self.encoder_type)
            baseline_classification = classifier_baseline.evaluate_attention_mechanisms(
                attention_results, processed_data
            )
            
            baseline_results = baseline_classification
            
        finally:
            # 清理臨時文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # 比較結果
        comparison = {
            'fusion_architecture': {
                'accuracy': fusion_results['classification_results']['test_accuracy'],
                'f1_score': fusion_results['classification_results']['test_f1'],
                'precision': fusion_results['classification_results']['test_precision'],
                'recall': fusion_results['classification_results']['test_recall'],
                'training_time': fusion_results['classification_results']['training_time']
            },
            'baseline_methods': {},
            'performance_improvement': {}
        }
        
        # 提取基準方法結果
        for method, results in baseline_results.items():
            if method != 'comparison' and isinstance(results, dict) and 'test_accuracy' in results:
                comparison['baseline_methods'][method] = {
                    'accuracy': results['test_accuracy'],
                    'f1_score': results['test_f1'],
                    'precision': results['test_precision'],
                    'recall': results['test_recall'],
                    'training_time': results.get('training_time', 0)
                }
        
        # 計算改善幅度
        fusion_acc = comparison['fusion_architecture']['accuracy']
        for method, metrics in comparison['baseline_methods'].items():
            baseline_acc = metrics['accuracy']
            improvement = ((fusion_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
            comparison['performance_improvement'][method] = {
                'accuracy_improvement_percent': improvement,
                'absolute_improvement': fusion_acc - baseline_acc
            }
        
        logger.info("性能比較完成")
        logger.info(f"融合架構準確率: {fusion_acc:.4f}")
        
        # 找出最佳基準方法
        if comparison['baseline_methods']:
            best_baseline = max(comparison['baseline_methods'].items(), key=lambda x: x[1]['accuracy'])
            best_method, best_metrics = best_baseline
            logger.info(f"最佳基準方法: {best_method} (準確率: {best_metrics['accuracy']:.4f})")
            logger.info(f"相對改善: {comparison['performance_improvement'][best_method]['accuracy_improvement_percent']:.2f}%")
        
        return {
            'comparison_results': comparison,
            'fusion_results': fusion_results,
            'baseline_results': baseline_results
        }


# 測試代碼
if __name__ == "__main__":
    # 配置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 創建測試數據
    test_data = pd.DataFrame({
        'text': [
            'This product is absolutely amazing! I love it so much.',
            'The quality is terrible and I hate this purchase.',
            'It\'s okay, nothing special but not bad either.',
            'Outstanding quality and excellent customer service!',
            'Worst experience ever, very disappointed.',
            'Average product, meets basic expectations.'
        ],
        'review_stars': [5, 1, 3, 5, 1, 3]
    })
    
    # 測試融合流程
    pipeline = FusionPipeline(encoder_type='bert')
    
    try:
        results = pipeline.run_complete_pipeline(
            input_data=test_data,
            text_column='text',
            test_size=0.3,
            save_results=False
        )
        
        print("融合流程測試完成！")
        print(f"準確率: {results['classification_results']['test_accuracy']:.4f}")
        print(f"門控權重: {results['fusion_results']['gate_weights']}")
        
    except Exception as e:
        print(f"測試失敗: {str(e)}")
        import traceback
        traceback.print_exc()