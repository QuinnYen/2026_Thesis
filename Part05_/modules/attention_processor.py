"""
注意力處理器 - 負責執行注意力機制分析和比較
"""

import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .attention_analyzer import AttentionAnalyzer
from .bert_encoder import BertEncoder
from .run_manager import RunManager

logger = logging.getLogger(__name__)

class AttentionProcessor:
    """注意力機制處理器，用於執行完整的注意力分析流程"""
    
    def __init__(self, output_dir: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化注意力處理器
        
        Args:
            output_dir: 輸出目錄
            config: 配置參數
        """
        self.output_dir = output_dir
        self.config = config or {}
        self.run_manager = RunManager(output_dir) if output_dir else None
        
        # 初始化組件
        self.bert_encoder = None
        self.attention_analyzer = None
        
        logger.info("注意力處理器已初始化")
    
    def process_with_attention(self, 
                             input_file: str,
                             attention_types: List[str] = None,
                             topics_path: Optional[str] = None,
                             attention_weights: Optional[Dict] = None,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        執行完整的注意力機制分析流程
        
        Args:
            input_file: 輸入的預處理數據文件
            attention_types: 要測試的注意力機制類型
            topics_path: 關鍵詞文件路徑
            attention_weights: 組合注意力權重配置
            save_results: 是否保存結果
            
        Returns:
            Dict: 完整的分析結果
        """
        try:
            start_time = datetime.now()
            logger.info("開始注意力機制分析流程")
            
            # 1. 讀取預處理數據
            logger.info(f"讀取數據: {input_file}")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"找不到輸入文件: {input_file}")
            
            df = pd.read_csv(input_file)
            logger.info(f"成功讀取 {len(df)} 條數據")
            
            # 2. 檢查必要欄位
            text_column = self._find_text_column(df)
            if text_column is None:
                raise ValueError("找不到有效的文本欄位")
            
            # 3. 初始化BERT編碼器和獲取特徵向量
            embeddings = self._get_embeddings(df, text_column)
            
            # 4. 準備元數據
            metadata = self._prepare_metadata(df)
            
            # 5. 初始化注意力分析器
            if self.attention_analyzer is None:
                topic_labels_path = self._find_topic_labels_path()
                self.attention_analyzer = AttentionAnalyzer(
                    topic_labels_path=topic_labels_path,
                    config=self.config
                )
            
            # 6. 執行注意力分析
            results = self.attention_analyzer.analyze_with_attention(
                embeddings=embeddings,
                metadata=metadata,
                attention_types=attention_types,
                topics_path=topics_path,
                attention_weights=attention_weights
            )
            
            # 7. 添加處理信息
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            results['processing_info'] = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'processing_time_seconds': processing_time,
                'input_file': input_file,
                'data_samples': len(df),
                'text_column': text_column,
                'embeddings_shape': embeddings.shape,
                'attention_types_tested': attention_types or ['no', 'similarity', 'keyword', 'self', 'combined']
            }
            
            # 8. 保存結果
            if save_results and self.output_dir:
                self._save_analysis_results(results)
            
            logger.info(f"注意力機制分析完成，耗時 {processing_time:.2f} 秒")
            return results
            
        except Exception as e:
            logger.error(f"注意力機制分析過程中發生錯誤: {str(e)}")
            raise
    
    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """尋找有效的文本欄位"""
        text_columns = ['processed_text', 'clean_text', 'text', 'review', 'content']
        
        for col in text_columns:
            if col in df.columns and not df[col].isna().all():
                logger.info(f"使用文本欄位: {col}")
                return col
        
        return None
    
    def _get_embeddings(self, df: pd.DataFrame, text_column: str) -> np.ndarray:
        """獲取或生成BERT特徵向量"""
        # 檢查是否已存在特徵向量文件
        embeddings_file = None
        if self.output_dir:
            embeddings_file = os.path.join(self.output_dir, "02_bert_embeddings.npy")
            
        if embeddings_file and os.path.exists(embeddings_file):
            logger.info(f"載入已存在的特徵向量: {embeddings_file}")
            embeddings = np.load(embeddings_file)
            logger.info(f"特徵向量形狀: {embeddings.shape}")
        else:
            # 生成新的特徵向量
            logger.info("生成BERT特徵向量...")
            if self.bert_encoder is None:
                self.bert_encoder = BertEncoder(output_dir=self.output_dir)
            
            embeddings = self.bert_encoder.encode(df[text_column])
            logger.info(f"生成的特徵向量形狀: {embeddings.shape}")
        
        return embeddings
    
    def _prepare_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """準備用於注意力分析的元數據"""
        metadata = df.copy()
        
        # 檢查是否有情感標籤
        if 'sentiment' not in metadata.columns:
            # 如果沒有情感標籤，可以嘗試從其他欄位推斷或創建假設標籤
            if 'label' in metadata.columns:
                metadata['sentiment'] = metadata['label']
            elif 'rating' in metadata.columns:
                # 基於評分創建情感標籤
                metadata['sentiment'] = metadata['rating'].apply(self._rating_to_sentiment)
            else:
                # 創建隨機情感標籤用於測試
                sentiments = ['positive', 'negative', 'neutral']
                metadata['sentiment'] = np.random.choice(sentiments, size=len(metadata))
                logger.warning("未找到情感標籤，已創建隨機標籤用於測試")
        
        logger.info(f"情感標籤分布: {metadata['sentiment'].value_counts().to_dict()}")
        return metadata
    
    def _rating_to_sentiment(self, rating):
        """將評分轉換為情感標籤"""
        if pd.isna(rating):
            return 'neutral'
        elif rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'negative'
        else:
            return 'neutral'
    
    def _find_topic_labels_path(self) -> Optional[str]:
        """尋找主題標籤文件"""
        possible_paths = [
            os.path.join(self.output_dir, "topic_labels.json") if self.output_dir else None,
            "utils/topic_labels.json",
            "Part05_/utils/topic_labels.json"
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                logger.info(f"找到主題標籤文件: {path}")
                return path
        
        logger.warning("未找到主題標籤文件")
        return None
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """保存分析結果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存完整結果
            results_file = os.path.join(self.output_dir, f"03_attention_analysis_{timestamp}.json")
            self.attention_analyzer.save_results(results, results_file)
            
            # 保存簡化的比較報告
            if 'comparison' in results:
                report_file = os.path.join(self.output_dir, f"03_attention_comparison_{timestamp}.json")
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(results['comparison'], f, ensure_ascii=False, indent=2)
                
                logger.info(f"比較報告已保存: {report_file}")
            
            # 保存面向向量（如果存在）
            for attention_type, result in results.items():
                if attention_type in ['processing_info', 'comparison']:
                    continue
                    
                if 'aspect_vectors' in result:
                    vectors_file = os.path.join(self.output_dir, f"03_aspect_vectors_{attention_type}_{timestamp}.npy")
                    
                    # 轉換為數組格式保存
                    aspect_vectors = result['aspect_vectors']
                    if aspect_vectors:
                        vector_array = np.array(list(aspect_vectors.values()))
                        np.save(vectors_file, vector_array)
                        logger.info(f"面向向量已保存: {vectors_file}")
            
        except Exception as e:
            logger.error(f"保存結果時發生錯誤: {str(e)}")
    
    def compare_attention_mechanisms(self, 
                                   attention_types: List[str] = None,
                                   input_file: str = None,
                                   topics_path: Optional[str] = None) -> Dict[str, Any]:
        """
        專門用於比較不同注意力機制效果的方法
        
        Args:
            attention_types: 要比較的注意力機制類型
            input_file: 輸入數據文件
            topics_path: 關鍵詞文件路徑
            
        Returns:
            Dict: 比較結果
        """
        if attention_types is None:
            attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
        
        logger.info(f"開始比較 {len(attention_types)} 種注意力機制")
        
        # 執行分析
        results = self.process_with_attention(
            input_file=input_file,
            attention_types=attention_types,
            topics_path=topics_path,
            save_results=True
        )
        
        # 提取比較結果
        comparison = results.get('comparison', {})
        
        # 生成詳細比較報告
        report = self._generate_comparison_report(results, attention_types)
        
        return {
            'comparison': comparison,
            'detailed_report': report,
            'processing_info': results.get('processing_info', {})
        }
    
    def _generate_comparison_report(self, results: Dict[str, Any], attention_types: List[str]) -> Dict[str, Any]:
        """生成詳細的比較報告"""
        report = {
            'summary': {},
            'detailed_metrics': {},
            'recommendations': []
        }
        
        # 收集指標
        metrics_data = []
        for attention_type in attention_types:
            if attention_type in results:
                metrics = results[attention_type].get('metrics', {})
                metrics_data.append({
                    'type': attention_type,
                    'coherence': metrics.get('coherence', 0),
                    'separation': metrics.get('separation', 0),
                    'combined_score': metrics.get('combined_score', 0)
                })
        
        if metrics_data:
            # 找出最佳機制
            best_combined = max(metrics_data, key=lambda x: x['combined_score'])
            best_coherence = max(metrics_data, key=lambda x: x['coherence'])
            best_separation = max(metrics_data, key=lambda x: x['separation'])
            
            report['summary'] = {
                'best_overall': best_combined['type'],
                'best_coherence': best_coherence['type'],
                'best_separation': best_separation['type'],
                'total_mechanisms_tested': len(metrics_data)
            }
            
            report['detailed_metrics'] = metrics_data
            
            # 生成建議
            if best_combined['type'] == 'combined':
                report['recommendations'].append("組合注意力機制表現最佳，建議在生產環境中使用")
            elif best_combined['type'] == 'similarity':
                report['recommendations'].append("相似度注意力機制表現最佳，適合語義相似性重要的任務")
            elif best_combined['type'] == 'keyword':
                report['recommendations'].append("關鍵詞注意力機制表現最佳，適合特定術語重要的任務")
            elif best_combined['type'] == 'self':
                report['recommendations'].append("自注意力機制表現最佳，適合文檔間關係複雜的任務")
            
            # 性能差異分析
            scores = [item['combined_score'] for item in metrics_data]
            score_std = np.std(scores)
            if score_std < 0.05:
                report['recommendations'].append("各機制性能差異較小，可考慮計算成本選擇簡單機制")
            else:
                report['recommendations'].append("各機制性能差異明顯，建議選擇最佳性能的機制")
        
        return report
    
    def load_previous_results(self, results_file: str) -> Dict[str, Any]:
        """載入之前的分析結果"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.info(f"成功載入結果: {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"載入結果失敗: {str(e)}")
            return {}
    
    def export_comparison_report(self, results: Dict[str, Any], output_file: str):
        """匯出比較報告為可讀格式"""
        try:
            comparison = results.get('comparison', {})
            
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("注意力機制比較報告")
            report_lines.append("=" * 60)
            report_lines.append("")
            
            # 總體摘要
            if 'summary' in comparison:
                summary = comparison['summary']
                report_lines.append("總體摘要:")
                report_lines.append(f"  最佳機制: {summary.get('best_mechanism', 'N/A')}")
                report_lines.append(f"  最佳得分: {summary.get('best_score', 0):.4f}")
                report_lines.append(f"  測試機制數: {summary.get('total_mechanisms', 0)}")
                report_lines.append("")
            
            # 詳細排名
            rankings = ['coherence_ranking', 'separation_ranking', 'combined_ranking']
            ranking_names = ['內聚度排名', '分離度排名', '綜合得分排名']
            
            for ranking, name in zip(rankings, ranking_names):
                if ranking in comparison:
                    report_lines.append(f"{name}:")
                    for i, (mechanism, score) in enumerate(comparison[ranking], 1):
                        report_lines.append(f"  {i}. {mechanism}: {score:.4f}")
                    report_lines.append("")
            
            # 寫入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"比較報告已匯出: {output_file}")
            
        except Exception as e:
            logger.error(f"匯出報告失敗: {str(e)}")


# 測試代碼
if __name__ == "__main__":
    # 配置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 測試注意力處理器
    processor = AttentionProcessor()
    
    # 創建測試數據
    test_data = pd.DataFrame({
        'text': [
            'This is a great product!',
            'I love this item very much.',
            'Not bad, could be better.',
            'Terrible quality, very disappointed.',
            'Worst purchase ever made.'
        ],
        'sentiment': ['positive', 'positive', 'neutral', 'negative', 'negative']
    })
    
    test_file = 'test_data.csv'
    test_data.to_csv(test_file, index=False)
    
    try:
        # 執行比較
        results = processor.compare_attention_mechanisms(
            input_file=test_file,
            attention_types=['no', 'similarity', 'combined']
        )
        
        print("比較完成！")
        print(f"最佳機制: {results['comparison']['summary']['best_mechanism']}")
        
        # 清理測試文件
        os.remove(test_file)
        
    except Exception as e:
        print(f"測試失敗: {str(e)}")
        if os.path.exists(test_file):
            os.remove(test_file) 