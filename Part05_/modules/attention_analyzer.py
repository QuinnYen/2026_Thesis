import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm

from .attention_mechanism import (
    create_attention_mechanism, 
    apply_attention_mechanism,
    AttentionMechanism
)

logger = logging.getLogger(__name__)

class AttentionAnalyzer:
    """增強的注意力分析器，整合多種注意力機制"""
    
    def __init__(self, topic_labels_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化注意力分析器
        
        Args:
            topic_labels_path (str): 主題標籤的json檔案路徑
            config (Dict): 配置參數
        """
        self.config = config or {}
        self.topic_labels = None
        
        # 載入主題標籤（如果提供）
        if topic_labels_path and Path(topic_labels_path).exists():
            try:
                with open(topic_labels_path, 'r', encoding='utf-8') as f:
                    self.topic_labels = json.load(f)
                logger.info(f"成功載入主題標籤: {topic_labels_path}")
            except Exception as e:
                logger.warning(f"載入主題標籤失敗: {str(e)}")
        
        # 初始化BERT注意力層（如果需要）
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def analyze_with_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, 
                             attention_types: List[str] = None, 
                             topics_path: Optional[str] = None,
                             attention_weights: Optional[Dict] = None) -> Dict[str, Any]:
        """
        使用不同注意力機制分析文本特徵
        
        Args:
            embeddings: BERT特徵向量
            metadata: 包含文本信息和標籤的DataFrame
            attention_types: 要測試的注意力機制列表
            topics_path: 關鍵詞文件路徑
            attention_weights: 組合注意力的權重配置
            
        Returns:
            Dict: 包含各種注意力機制結果的字典
        """
        if attention_types is None:
            attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
        
        results = {}
        
        logger.info(f"開始分析 {len(attention_types)} 種注意力機制")
        
        for attention_type in tqdm(attention_types, desc="分析注意力機制"):
            logger.info(f"分析注意力機制: {attention_type}")
            
            try:
                # 應用注意力機制
                result = apply_attention_mechanism(
                    attention_type=attention_type,
                    embeddings=embeddings,
                    metadata=metadata,
                    topics_path=topics_path,
                    weights=attention_weights
                )
                
                results[attention_type] = result
                
                # 輸出基本評估結果
                metrics = result['metrics']
                logger.info(f"{attention_type} 注意力 - 內聚度: {metrics['coherence']:.4f}, "
                          f"分離度: {metrics['separation']:.4f}, "
                          f"綜合得分: {metrics['combined_score']:.4f}")
                
            except Exception as e:
                logger.error(f"處理 {attention_type} 注意力機制時發生錯誤: {str(e)}")
                continue
        
        # 比較不同注意力機制的效果
        comparison = self._compare_attention_mechanisms(results)
        results['comparison'] = comparison
        
        return results
    
    def _compare_attention_mechanisms(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        比較不同注意力機制的效果
        
        Args:
            results: 各種注意力機制的結果
            
        Returns:
            Dict: 比較結果
        """
        if not results:
            return {}
        
        comparison = {
            'coherence_ranking': [],
            'separation_ranking': [],
            'combined_ranking': [],
            'summary': {}
        }
        
        # 提取各項指標
        mechanisms = []
        coherence_scores = []
        separation_scores = []
        combined_scores = []
        
        for mechanism, result in results.items():
            if mechanism == 'comparison':
                continue
                
            metrics = result.get('metrics', {})
            mechanisms.append(mechanism)
            coherence_scores.append(metrics.get('coherence', 0))
            separation_scores.append(metrics.get('separation', 0))
            combined_scores.append(metrics.get('combined_score', 0))
        
        # 排序（降序）
        coherence_ranking = sorted(zip(mechanisms, coherence_scores), 
                                 key=lambda x: x[1], reverse=True)
        separation_ranking = sorted(zip(mechanisms, separation_scores), 
                                  key=lambda x: x[1], reverse=True)
        combined_ranking = sorted(zip(mechanisms, combined_scores), 
                                key=lambda x: x[1], reverse=True)
        
        comparison['coherence_ranking'] = coherence_ranking
        comparison['separation_ranking'] = separation_ranking
        comparison['combined_ranking'] = combined_ranking
        
        # 生成摘要
        if combined_ranking:
            best_mechanism = combined_ranking[0][0]
            best_score = combined_ranking[0][1]
            
            comparison['summary'] = {
                'best_mechanism': best_mechanism,
                'best_score': best_score,
                'total_mechanisms': len(mechanisms),
                'ranking_details': {
                    'coherence': [{'mechanism': m, 'score': s} for m, s in coherence_ranking],
                    'separation': [{'mechanism': m, 'score': s} for m, s in separation_ranking],
                    'combined': [{'mechanism': m, 'score': s} for m, s in combined_ranking]
                }
            }
        
        # 安全訪問 summary 鍵
        summary = comparison.get('summary', {})
        best_mechanism = summary.get('best_mechanism', 'N/A') if summary else 'N/A'
        logger.info(f"注意力機制比較完成。最佳機制: {best_mechanism}")
        
        return comparison
    
    def analyze_topics(self, text_features: torch.Tensor, 
                      dataset_type: str = "IMDB", 
                      language: str = "zh") -> Dict[str, float]:
        """
        分析文本的主題分布（保持向後兼容）
        
        Args:
            text_features: 文本特徵張量
            dataset_type: 資料集類型 (IMDB/AMAZON/YELP)
            language: 語言選擇 (zh/en)
            
        Returns:
            dict: 主題分布結果
        """
        if self.topic_labels is None:
            logger.warning("未載入主題標籤，返回空結果")
            return {}
        
        try:
            # 使用多頭注意力機制
            with torch.no_grad():
                # 確保輸入是正確的形狀
                if text_features.dim() == 2:
                    # 添加序列長度維度
                    text_features = text_features.unsqueeze(1)
                
                # 應用多頭注意力
                attended_features, attention_weights = self.attention(
                    text_features, text_features, text_features
                )
                
                # 計算主題分數
                topic_scores = F.softmax(torch.mean(attended_features, dim=1), dim=-1)
                
                # 將分數對應到主題標籤
                results = {}
                labels = self.topic_labels.get(dataset_type, {}).get(language, {})
                
                for i in range(min(10, topic_scores.size(-1))):
                    topic_label = labels.get(str(i), f"Topic_{i}")
                    if i < topic_scores.size(-1):
                        results[topic_label] = float(topic_scores[0][i])
                    else:
                        results[topic_label] = 0.0
                        
                return results
                
        except Exception as e:
            logger.error(f"主題分析時發生錯誤: {str(e)}")
            return {}
    
    def get_topic_attention_weights(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        獲取每個主題的注意力權重（保持向後兼容）
        
        Args:
            text_features: 文本特徵張量
            
        Returns:
            torch.Tensor: 主題的注意力權重
        """
        try:
            with torch.no_grad():
                if text_features.dim() == 2:
                    text_features = text_features.unsqueeze(1)
                
                _, attention_weights = self.attention(
                    text_features, text_features, text_features
                )
                
                # 平均各個頭的注意力權重
                topic_attention = torch.mean(attention_weights, dim=1)
                return topic_attention
                
        except Exception as e:
            logger.error(f"獲取注意力權重時發生錯誤: {str(e)}")
            return torch.zeros_like(text_features[:, 0, :])
    
    def process_attention_weights(self, attention_weights: np.ndarray, layer_idx: int = -1) -> torch.Tensor:
        """
        處理BERT的注意力權重（保持向後兼容）
        
        Args:
            attention_weights: BERT注意力權重數組
            layer_idx: 要使用的BERT層索引，預設使用最後一層
        
        Returns:
            torch.Tensor: 處理後的注意力權重
        """
        try:
            # 獲取指定層的注意力權重
            if len(attention_weights.shape) > 2:
                layer_weights = torch.from_numpy(attention_weights[layer_idx])
            else:
                layer_weights = torch.from_numpy(attention_weights)
            
            return layer_weights
            
        except Exception as e:
            logger.error(f"處理注意力權重時發生錯誤: {str(e)}")
            return torch.zeros(1, 1)
    
    def apply_attention_mechanism(self, features: torch.Tensor, 
                                attention_weights: torch.Tensor) -> torch.Tensor:
        """
        應用注意力機制（保持向後兼容）
        
        Args:
            features: 輸入特徵
            attention_weights: 注意力權重
            
        Returns:
            torch.Tensor: 加權後的特徵
        """
        try:
            # 應用注意力權重
            attended_features = torch.matmul(attention_weights, features)
            return attended_features
            
        except Exception as e:
            logger.error(f"應用注意力機制時發生錯誤: {str(e)}")
            return features
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        保存注意力分析結果
        
        Args:
            results: 分析結果
            output_path: 輸出路徑
        """
        try:
            # 轉換numpy數組為列表以便JSON序列化
            serializable_results = self._make_serializable(results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"結果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存結果時發生錯誤: {str(e)}")
    
    def _make_serializable(self, obj):
        """
        將對象轉換為JSON可序列化格式
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            # 確保字典的鍵值也是可序列化的
            return {
                str(key) if isinstance(key, (np.integer, np.floating)) else key: 
                self._make_serializable(value) 
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj 