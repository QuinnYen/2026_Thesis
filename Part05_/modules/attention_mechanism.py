"""
注意力機制模組 - 提供各種注意力機制實現
包含相似度注意力、關鍵詞注意力、自注意力和組合注意力
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import logging
import time
import random
import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any

# 固定所有隨機種子，確保結果可重現
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 配置日誌
logger = logging.getLogger(__name__)

# 固定PyTorch隨機種子
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AttentionMechanism:
    """注意力機制基類"""
    
    def __init__(self, config=None):
        """初始化注意力機制
        
        Args:
            config: 配置參數字典
        """
        self.config = config or {}
        self.logger = logger
    
    def compute_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """計算注意力權重
        
        Args:
            embeddings: 文檔嵌入向量，形狀為 [n_docs, embed_dim]
            metadata: 文檔元數據，包含面向標籤
            **kwargs: 其他參數
            
        Returns:
            weights: 注意力權重，形狀為 [n_topics, n_docs]
            topic_indices: 每個主題包含的文檔索引字典
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def compute_aspect_vectors(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[Dict, Dict]:
        """計算面向向量
        
        Args:
            embeddings: 文檔嵌入向量
            metadata: 文檔元數據
            **kwargs: 其他參數
            
        Returns:
            aspect_vectors: 面向向量字典
            attention_data: 注意力權重和相關數據
        """
        # 計算注意力權重
        weights, topic_indices = self.compute_attention(embeddings, metadata, **kwargs)
        
        # 計算面向向量
        topics = list(topic_indices.keys())
        aspect_vectors = {}
        
        for topic_idx, topic in enumerate(topics):
            # 獲取該主題的文檔索引
            doc_indices = topic_indices[topic]
            
            if not doc_indices:
                self.logger.warning(f"主題 {topic} 沒有文檔，將使用零向量")
                aspect_vectors[topic] = np.zeros(embeddings.shape[1])
                continue
            
            # 獲取該主題的嵌入向量
            topic_embeddings = embeddings[doc_indices]
            
            # 獲取該主題的文檔權重
            topic_weights = weights[topic_idx][doc_indices]
            
            # 正規化權重確保總和為1
            if np.sum(topic_weights) > 0:
                topic_weights = topic_weights / np.sum(topic_weights)
            else:
                topic_weights = np.ones(len(doc_indices)) / len(doc_indices)
            
            # 計算加權平均
            weighted_vector = np.zeros(embeddings.shape[1])
            for i, embed in enumerate(topic_embeddings):
                weighted_vector += topic_weights[i] * embed
                
            aspect_vectors[topic] = weighted_vector
        
        # 返回面向向量和注意力數據
        attention_data = {
            "weights": weights,
            "topic_indices": topic_indices,
            "topics": topics
        }
        
        return aspect_vectors, attention_data
    
    def evaluate(self, aspect_vectors: Dict, embeddings: np.ndarray, metadata: pd.DataFrame) -> Dict:
        """評估注意力機制的效能
        
        Args:
            aspect_vectors: 面向向量字典
            embeddings: 文檔嵌入向量
            metadata: 文檔元數據
            
        Returns:
            metrics: 評估指標字典
        """
        topics = list(aspect_vectors.keys())
        self.logger.info(f"開始評估，共有 {len(topics)} 個面向")
        
        # 計算面向內聚度
        coherence = 0.0
        doc_count = 0
        topic_coherence_dict = {}
        
        self.logger.info("\n開始計算面向內聚度...")
        
        for topic in topics:
            # 獲取該面向的文檔
            topic_docs = metadata[metadata['sentiment'] == topic] if 'sentiment' in metadata.columns else metadata[metadata.index.isin([topic])]
            
            if len(topic_docs) == 0:
                self.logger.warning(f"面向 '{topic}' 沒有對應的文檔")
                topic_coherence_dict[topic] = 0.0
                continue
            
            # 獲取該面向的嵌入向量索引
            doc_indices = topic_docs.index.tolist()
            topic_embeddings = embeddings[doc_indices]
            aspect_vector = aspect_vectors[topic]
            
            # 計算該面向的內聚度
            topic_coherence = 0.0
            for embed in topic_embeddings:
                similarity = np.dot(aspect_vector, embed) / (np.linalg.norm(aspect_vector) * np.linalg.norm(embed))
                topic_coherence += similarity
            
            if len(topic_embeddings) > 0:
                topic_coherence /= len(topic_embeddings)
                topic_coherence_dict[topic] = float(topic_coherence)
                coherence += topic_coherence * len(topic_embeddings)
                doc_count += len(topic_embeddings)
                
                self.logger.info(f"面向 '{topic}' 的內聚度: {topic_coherence:.4f} (文檔數: {len(topic_embeddings)})")
            else:
                topic_coherence_dict[topic] = 0.0
        
        if doc_count > 0:
            coherence /= doc_count
            
        self.logger.info(f"\n總體內聚度: {coherence:.4f}")
        
        # 計算面向分離度
        separation = 0.0
        pair_count = 0
        topic_separation_dict = {}
        aspect_separation_values = {topic: [] for topic in topics}
        
        self.logger.info("\n開始計算面向間的分離度...")
        
        for i in range(len(topics)):
            for j in range(i+1, len(topics)):
                topic_i = topics[i]
                topic_j = topics[j]
                vec_i = aspect_vectors[topic_i]
                vec_j = aspect_vectors[topic_j]
                
                # 計算餘弦距離 (1 - 相似度)
                similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                distance = 1.0 - similarity
                
                separation += distance
                pair_count += 1
                
                # 存儲該對面向的分離度
                pair_key = f"{topic_i}_{topic_j}"
                topic_separation_dict[pair_key] = float(distance)
                
                # 為兩個面向都添加這個分離度值
                aspect_separation_values[topic_i].append(distance)
                aspect_separation_values[topic_j].append(distance)
                
                self.logger.info(f"面向 '{topic_i}' 和 '{topic_j}' 的分離度: {distance:.4f}")
        
        if pair_count > 0:
            separation /= pair_count
            
        self.logger.info(f"\n總體分離度: {separation:.4f}")
            
        # 計算每個面向的平均分離度
        aspect_avg_separation = {}
        for topic in topics:
            if aspect_separation_values[topic]:
                aspect_avg_separation[topic] = float(np.mean(aspect_separation_values[topic]))
                self.logger.info(f"面向 '{topic}' 的平均分離度: {aspect_avg_separation[topic]:.4f}")
            else:
                aspect_avg_separation[topic] = 0.0
            
        # 計算綜合得分
        coherence_weight = self.config.get('coherence_weight', 0.5)
        separation_weight = self.config.get('separation_weight', 0.5)
        combined_score = coherence_weight * coherence + separation_weight * separation
        
        self.logger.info(f"\n綜合得分: {combined_score:.4f} (內聚度權重: {coherence_weight}, 分離度權重: {separation_weight})")
            
        # 返回評估指標
        metrics = {
            # 總體指標
            "coherence": float(coherence),
            "separation": float(separation),
            "combined_score": float(combined_score),
            
            # 面向特定指標
            "topic_coherence": topic_coherence_dict,
            "topic_separation": topic_separation_dict,
            "aspect_separation": aspect_avg_separation,
            
            # 詳細指標
            "metrics_details": {
                "weights": {
                    "coherence_weight": coherence_weight,
                    "separation_weight": separation_weight
                },
                "per_aspect": {
                    "coherence": topic_coherence_dict,
                    "separation": aspect_avg_separation,
                    "pairwise_separation": topic_separation_dict
                }
            }
        }
        
        return metrics


class NoAttention(AttentionMechanism):
    """無注意力機制（平均）"""
    
    def compute_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """計算均等權重（相當於平均）"""
        # 獲取所有主題
        if 'sentiment' in metadata.columns:
            topics = metadata['sentiment'].unique()
            topic_column = 'sentiment'
        elif 'main_topic' in metadata.columns:
            topics = metadata['main_topic'].unique()
            topic_column = 'main_topic'
        else:
            # 創建假設的情感標籤
            topics = ['positive', 'negative', 'neutral']
            topic_column = 'sentiment'
        
        # 創建權重矩陣（所有權重相等）
        weights = np.zeros((len(topics), len(embeddings)))
        topic_indices = {}
        
        for topic_idx, topic in enumerate(topics):
            if topic_column in metadata.columns:
                # 獲取該主題的文檔索引
                doc_indices = metadata.index[metadata[topic_column] == topic].tolist()
            else:
                # 如果沒有標籤，平均分配
                doc_indices = list(range(topic_idx * len(embeddings) // len(topics),
                                       (topic_idx + 1) * len(embeddings) // len(topics)))
            
            topic_indices[topic] = doc_indices
            
            # 設置均等權重
            if doc_indices:
                weights[topic_idx][doc_indices] = 1.0 / len(doc_indices)
        
        return weights, topic_indices


class SimilarityAttention(AttentionMechanism):
    """基於相似度的注意力機制"""
    
    def compute_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """計算基於相似度的注意力權重"""
        # 獲取所有主題
        if 'sentiment' in metadata.columns:
            topics = metadata['sentiment'].unique()
            topic_column = 'sentiment'
        elif 'main_topic' in metadata.columns:
            topics = metadata['main_topic'].unique()
            topic_column = 'main_topic'
        else:
            # 創建假設的情感標籤
            topics = ['positive', 'negative', 'neutral']
            topic_column = 'sentiment'
        
        # 創建權重矩陣
        weights = np.zeros((len(topics), len(embeddings)))
        topic_indices = {}
        
        for topic_idx, topic in enumerate(topics):
            if topic_column in metadata.columns:
                # 獲取該主題的文檔索引
                doc_indices = metadata.index[metadata[topic_column] == topic].tolist()
            else:
                # 如果沒有標籤，平均分配
                doc_indices = list(range(topic_idx * len(embeddings) // len(topics),
                                       (topic_idx + 1) * len(embeddings) // len(topics)))
            
            topic_indices[topic] = doc_indices
            
            if not doc_indices:
                continue
                
            # 獲取該主題的嵌入向量
            topic_embeddings = embeddings[doc_indices]
            
            # 計算中心向量
            center = np.mean(topic_embeddings, axis=0)
            
            # 計算每個文檔與中心的相似度
            similarities = []
            for embed in topic_embeddings:
                similarity = np.dot(center, embed) / (np.linalg.norm(center) * np.linalg.norm(embed))
                similarities.append(similarity)
            
            # 轉換為 softmax 權重
            similarities = np.array(similarities)
            # 避免數值不穩定
            similarities = similarities - np.max(similarities)
            exp_sim = np.exp(similarities)
            topic_weights = exp_sim / np.sum(exp_sim)
            
            # 填充權重矩陣
            weights[topic_idx][doc_indices] = topic_weights
        
        return weights, topic_indices


class KeywordGuidedAttention(AttentionMechanism):
    """基於關鍵詞的注意力機制"""
    
    def compute_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """計算基於關鍵詞的注意力權重"""
        # 檢查是否提供了關鍵詞
        topic_keywords = kwargs.get('topic_keywords')
        if topic_keywords is None:
            # 嘗試從topics_path加載
            topics_path = kwargs.get('topics_path')
            if topics_path and os.path.exists(topics_path):
                try:
                    with open(topics_path, 'r', encoding='utf-8') as f:
                        topic_keywords = json.load(f)
                except Exception as e:
                    self.logger.warning(f"無法從 {topics_path} 加載主題關鍵詞: {str(e)}")
        
        if topic_keywords is None:
            # 使用預設關鍵詞
            topic_keywords = {
                'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', '好', '棒', '優秀'],
                'negative': ['bad', 'terrible', 'awful', 'horrible', 'worst', '壞', '糟糕', '可怕'],
                'neutral': ['okay', 'average', 'normal', 'fine', '還可以', '普通', '一般']
            }
            self.logger.info("使用預設關鍵詞進行注意力計算")
        
        # 獲取所有主題
        if 'sentiment' in metadata.columns:
            topics = metadata['sentiment'].unique()
            topic_column = 'sentiment'
        elif 'main_topic' in metadata.columns:
            topics = metadata['main_topic'].unique()
            topic_column = 'main_topic'
        else:
            topics = list(topic_keywords.keys())
            topic_column = 'sentiment'
        
        # 創建權重矩陣
        weights = np.zeros((len(topics), len(embeddings)))
        topic_indices = {}
        
        # 獲取原始文本
        text_column = None
        for col in ['processed_text', 'clean_text', 'text', 'review']:
            if col in metadata.columns:
                text_column = col
                break
        
        if text_column is None:
            self.logger.warning("未找到文本欄位，將使用相似度注意力")
            similarity_attn = SimilarityAttention(self.config)
            return similarity_attn.compute_attention(embeddings, metadata, **kwargs)
        
        texts = metadata[text_column].fillna('').tolist()
        
        for topic_idx, topic in enumerate(topics):
            if topic_column in metadata.columns:
                doc_indices = metadata.index[metadata[topic_column] == topic].tolist()
            else:
                doc_indices = list(range(topic_idx * len(embeddings) // len(topics),
                                       (topic_idx + 1) * len(embeddings) // len(topics)))
            
            topic_indices[topic] = doc_indices
            
            if not doc_indices:
                continue
            
            # 獲取該主題的關鍵詞
            keywords = topic_keywords.get(topic, [])
            if not keywords:
                # 如果沒有關鍵詞，使用均等權重
                weights[topic_idx][doc_indices] = 1.0 / len(doc_indices)
                continue
            
            # 計算關鍵詞分數
            keyword_scores = []
            
            for idx in doc_indices:
                text = texts[idx].lower()
                score = 0
                
                # 計算關鍵詞出現次數的加權和
                for i, keyword in enumerate(keywords):
                    # 關鍵詞權重隨排名遞減
                    keyword_weight = 1.0 / (i + 1)
                    count = text.count(keyword.lower())
                    score += count * keyword_weight
                
                keyword_scores.append(score)
            
            # 如果所有分數為零，使用平均權重
            if sum(keyword_scores) == 0:
                weights[topic_idx][doc_indices] = 1.0 / len(doc_indices)
            else:
                # 正規化分數
                keyword_scores = np.array(keyword_scores)
                topic_weights = keyword_scores / np.sum(keyword_scores)
                weights[topic_idx][doc_indices] = topic_weights
        
        return weights, topic_indices


class SelfAttention(AttentionMechanism):
    """自注意力機制"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """計算自注意力權重"""
        # 獲取所有主題
        if 'sentiment' in metadata.columns:
            topics = metadata['sentiment'].unique()
            topic_column = 'sentiment'
        elif 'main_topic' in metadata.columns:
            topics = metadata['main_topic'].unique()
            topic_column = 'main_topic'
        else:
            topics = ['positive', 'negative', 'neutral']
            topic_column = 'sentiment'
        
        # 創建權重矩陣
        weights = np.zeros((len(topics), len(embeddings)))
        topic_indices = {}
        
        for topic_idx, topic in enumerate(topics):
            if topic_column in metadata.columns:
                doc_indices = metadata.index[metadata[topic_column] == topic].tolist()
            else:
                doc_indices = list(range(topic_idx * len(embeddings) // len(topics),
                                       (topic_idx + 1) * len(embeddings) // len(topics)))
            
            topic_indices[topic] = doc_indices
            
            if not doc_indices or len(doc_indices) < 2:
                # 如果文檔數少於2，使用均等權重
                if doc_indices:
                    weights[topic_idx][doc_indices] = 1.0 / len(doc_indices)
                continue
                
            # 獲取該主題的嵌入向量
            topic_embeddings = embeddings[doc_indices]
            
            # 轉換為張量
            topic_embeddings_tensor = torch.tensor(topic_embeddings, dtype=torch.float32).to(self.device)
            
            # 計算注意力分數
            # 使用縮放點積注意力
            d_k = topic_embeddings_tensor.size(-1)
            scores = torch.matmul(topic_embeddings_tensor, topic_embeddings_tensor.transpose(-2, -1)) / math.sqrt(d_k)
            
            # 使用softmax轉換為權重
            attn_weights = F.softmax(scores, dim=-1)
            
            # 計算每個文檔的平均注意力得分
            avg_weights = torch.mean(attn_weights, dim=-1).cpu().numpy()
            
            # 填充權重矩陣
            weights[topic_idx][doc_indices] = avg_weights
        
        return weights, topic_indices


class CombinedAttention(AttentionMechanism):
    """組合型注意力機制"""
    
    def compute_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """計算組合型注意力權重"""
        # 獲取組合權重
        weights_dict = kwargs.get('weights', {})
        similarity_weight = weights_dict.get('similarity', 0.33)
        keyword_weight = weights_dict.get('keyword', 0.33)
        self_weight = weights_dict.get('self', 0.34)
        
        # 確保權重總和為1
        total = similarity_weight + keyword_weight + self_weight
        if abs(total - 1.0) > 1e-5:
            self.logger.warning(f"注意力權重總和 {total} 不為1，將進行歸一化")
            similarity_weight /= total
            keyword_weight /= total
            self_weight /= total
        
        # 初始化各種注意力機制
        similarity_attn = SimilarityAttention(self.config)
        keyword_attn = KeywordGuidedAttention(self.config)
        self_attn = SelfAttention(self.config)
        
        # 計算各類注意力權重
        similarity_weights, topic_indices = similarity_attn.compute_attention(embeddings, metadata, **kwargs)
        keyword_weights, _ = keyword_attn.compute_attention(embeddings, metadata, **kwargs)
        self_weights, _ = self_attn.compute_attention(embeddings, metadata, **kwargs)
        
        # 計算加權組合
        weights = (
            similarity_weight * similarity_weights +
            keyword_weight * keyword_weights +
            self_weight * self_weights
        )
        
        return weights, topic_indices


def create_attention_mechanism(attention_type: str, config=None) -> AttentionMechanism:
    """創建指定類型的注意力機制
    
    Args:
        attention_type: 注意力機制類型，可以是'no', 'similarity', 'keyword', 'self'或'combined'
        config: 配置參數
        
    Returns:
        AttentionMechanism: 注意力機制實例
    """
    attention_type = attention_type.lower()
    
    if attention_type == 'no' or attention_type == 'none':
        return NoAttention(config)
    elif attention_type == 'similarity':
        return SimilarityAttention(config)
    elif attention_type == 'keyword':
        return KeywordGuidedAttention(config)
    elif attention_type == 'self':
        return SelfAttention(config)
    elif attention_type == 'combined':
        return CombinedAttention(config)
    else:
        logger.warning(f"未知的注意力類型: {attention_type}，使用無注意力機制")
        return NoAttention(config)


def apply_attention_mechanism(attention_type: str, embeddings: np.ndarray, metadata: pd.DataFrame, 
                            topics_path: Optional[str] = None, weights: Optional[Dict] = None) -> Dict[str, Any]:
    """應用注意力機制計算面向向量
    
    Args:
        attention_type: 注意力機制類型
        embeddings: 嵌入向量
        metadata: 元數據
        topics_path: 主題詞文件路徑，用於關鍵詞注意力
        weights: 組合注意力的權重字典
        
    Returns:
        dict: 包含面向向量和評估指標的字典
    """
    # 創建配置
    config = {}
    
    # 創建注意力機制
    attention_mech = create_attention_mechanism(attention_type, config)
    
    # 準備關鍵字參數
    kwargs = {}
    if topics_path:
        kwargs['topics_path'] = topics_path
    if weights:
        kwargs['weights'] = weights
    
    # 計算面向向量
    aspect_vectors, attention_data = attention_mech.compute_aspect_vectors(
        embeddings, metadata, **kwargs
    )
    
    # 評估面向向量
    metrics = attention_mech.evaluate(aspect_vectors, embeddings, metadata)
    
    return {
        'aspect_vectors': aspect_vectors,
        'attention_data': attention_data,
        'metrics': metrics,
        'attention_type': attention_type
    }


# 測試代碼
if __name__ == "__main__":
    # 配置日誌級別
    logging.basicConfig(level=logging.INFO)
    
    # 測試注意力機制
    # 創建模擬數據
    np.random.seed(42)
    n_docs = 100
    embed_dim = 768  # 使用BERT的標準維度
    n_topics = 3
    
    # 模擬嵌入向量
    embeddings = np.random.randn(n_docs, embed_dim)
    
    # 模擬元數據
    sentiments = ['positive', 'negative', 'neutral']
    doc_sentiments = np.random.choice(sentiments, size=n_docs)
    metadata = pd.DataFrame({
        'sentiment': doc_sentiments,
        'text': ['This is document ' + str(i) for i in range(n_docs)]
    })
    
    # 測試所有注意力機制
    attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
    
    for attn_type in attention_types:
        logger.info(f"測試 {attn_type} 注意力機制")
        
        # 創建注意力機制
        attention_mech = create_attention_mechanism(attn_type)
        
        # 測試計算注意力權重
        weights, topic_indices = attention_mech.compute_attention(embeddings, metadata)
        
        # 測試計算面向向量
        aspect_vectors, _ = attention_mech.compute_aspect_vectors(embeddings, metadata)
        
        # 測試評估指標
        metrics = attention_mech.evaluate(aspect_vectors, embeddings, metadata)
        
        logger.info(f"內聚度: {metrics['coherence']:.4f}")
        logger.info(f"分離度: {metrics['separation']:.4f}")
        logger.info(f"綜合得分: {metrics['combined_score']:.4f}")
        logger.info("---")
    
    logger.info("注意力機制測試完成") 