"""
注意力機制模組 - 提供多種注意力機制的實現和組合
"""

import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from sklearn.metrics.pairwise import cosine_similarity
import math

class AttentionMechanism:
    """注意力機制基類"""
    
    def __init__(self, logger=None):
        """初始化注意力機制"""
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_attention(self, embeddings, metadata, **kwargs):
        """
        計算注意力權重
        
        Args:
            embeddings: 文檔嵌入向量，形狀為 [n_docs, embed_dim]
            metadata: 文檔元數據，包含面向標籤
            **kwargs: 其他參數
            
        Returns:
            weights: 注意力權重，形狀為 [n_topics, n_docs]
            topic_indices: 每個主題包含的文檔索引字典
        """
        raise NotImplementedError("子類必須實現此方法")
    
    def compute_aspect_vectors(self, embeddings, metadata, **kwargs):
        """
        計算面向向量
        
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
    
    def evaluate(self, aspect_vectors, embeddings, metadata):
        """
        評估注意力機制的效能
        
        Args:
            aspect_vectors: 面向向量字典
            embeddings: 文檔嵌入向量
            metadata: 文檔元數據
            
        Returns:
            metrics: 評估指標字典
        """
        topics = list(aspect_vectors.keys())
        
        # 計算面向內聚度（面向內文檔與面向向量的平均相似度）
        coherence = 0.0
        doc_count = 0
        
        for topic in topics:
            aspect_vec = aspect_vectors[topic]
            
            # 獲取該主題的文檔索引
            doc_indices = metadata.index[metadata['main_topic'] == topic].tolist()
            
            if not doc_indices:
                continue
                
            # 獲取該主題的嵌入向量
            topic_embeddings = embeddings[doc_indices]
            
            # 計算與面向向量的相似度
            topic_coherence = 0.0
            for embed in topic_embeddings:
                similarity = np.dot(aspect_vec, embed) / (np.linalg.norm(aspect_vec) * np.linalg.norm(embed))
                topic_coherence += similarity
            
            if len(topic_embeddings) > 0:
                topic_coherence /= len(topic_embeddings)
                coherence += topic_coherence * len(topic_embeddings)
                doc_count += len(topic_embeddings)
        
        if doc_count > 0:
            coherence /= doc_count
        
        # 計算面向分離度（不同面向間向量的平均餘弦距離）
        separation = 0.0
        pair_count = 0
        
        for i in range(len(topics)):
            for j in range(i+1, len(topics)):
                vec_i = aspect_vectors[topics[i]]
                vec_j = aspect_vectors[topics[j]]
                
                # 計算餘弦距離 (1 - 相似度)
                similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                distance = 1.0 - similarity
                
                separation += distance
                pair_count += 1
        
        if pair_count > 0:
            separation /= pair_count
            
        # 返回評估指標
        metrics = {
            "coherence": coherence,   # 越高越好，表示面向內部一致性
            "separation": separation,  # 越高越好，表示面向之間差異性
            "combined_score": coherence + separation  # 綜合分數
        }
        
        return metrics
class NoAttention(AttentionMechanism):
    """無注意力機制（平均）"""
    
    def compute_attention(self, embeddings, metadata, **kwargs):
        """計算均等權重（相當於平均）"""
        # 獲取所有主題
        topics = metadata['main_topic'].unique()
        
        # 創建權重矩陣（所有權重相等）
        weights = np.zeros((len(topics), len(embeddings)))
        topic_indices = {}
        
        for topic_idx, topic in enumerate(topics):
            # 獲取該主題的文檔索引
            doc_indices = metadata.index[metadata['main_topic'] == topic].tolist()
            topic_indices[topic] = doc_indices
            
            # 設置均等權重
            if doc_indices:
                weights[topic_idx][doc_indices] = 1.0 / len(doc_indices)
        
        return weights, topic_indices

class SimilarityAttention(AttentionMechanism):
    """基於相似度的注意力機制"""
    
    def compute_attention(self, embeddings, metadata, **kwargs):
        """計算基於相似度的注意力權重"""
        # 獲取所有主題
        topics = metadata['main_topic'].unique()
        
        # 創建權重矩陣
        weights = np.zeros((len(topics), len(embeddings)))
        topic_indices = {}
        
        for topic_idx, topic in enumerate(topics):
            # 獲取該主題的文檔索引
            doc_indices = metadata.index[metadata['main_topic'] == topic].tolist()
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
            topic_weights = np.exp(similarities) / np.sum(np.exp(similarities))
            
            # 填充權重矩陣
            weights[topic_idx][doc_indices] = topic_weights
        
        return weights, topic_indices

class KeywordGuidedAttention(AttentionMechanism):
    """基於關鍵詞的注意力機制"""
    
    def compute_attention(self, embeddings, metadata, **kwargs):
        """計算基於關鍵詞的注意力權重"""
        # 檢查是否提供了關鍵詞
        topic_keywords = kwargs.get('topic_keywords')
        if topic_keywords is None:
            self.logger.warning("未提供關鍵詞，將使用相似度注意力")
            similarity_attn = SimilarityAttention(self.logger)
            return similarity_attn.compute_attention(embeddings, metadata, **kwargs)
        
        # 獲取所有主題
        topics = metadata['main_topic'].unique()
        
        # 創建權重矩陣
        weights = np.zeros((len(topics), len(embeddings)))
        topic_indices = {}
        
        # 獲取原始文本
        texts = metadata['text'].fillna('').tolist() if 'text' in metadata.columns else metadata['clean_text'].fillna('').tolist()
        
        for topic_idx, topic in enumerate(topics):
            # 獲取該主題的文檔索引
            doc_indices = metadata.index[metadata['main_topic'] == topic].tolist()
            topic_indices[topic] = doc_indices
            
            if not doc_indices:
                continue
                
            # 獲取該主題的嵌入向量
            topic_embeddings = embeddings[doc_indices]
            
            # 獲取該主題的關鍵詞
            # 從topic中提取索引號（假設格式為'Topic_1_XXX'）
            topic_num = int(topic.split('_')[1]) - 1
            
            # 獲取關鍵詞列表
            keywords = topic_keywords.get(topic, [])
            if not keywords:
                # 嘗試使用索引獲取
                for key in topic_keywords.keys():
                    if str(topic_num) in key:
                        keywords = topic_keywords[key]
                        break
            
            if not keywords:
                self.logger.warning(f"主題 {topic} 沒有找到關鍵詞")
                # 將權重設為平均值
                weights[topic_idx][doc_indices] = 1.0 / len(doc_indices)
                continue
            
            # 計算每個文檔包含關鍵詞的程度
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
    
    def __init__(self, logger=None):
        super().__init__(logger)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compute_attention(self, embeddings, metadata, **kwargs):
        """計算自注意力權重"""
        # 獲取所有主題
        topics = metadata['main_topic'].unique()
        
        # 創建權重矩陣
        weights = np.zeros((len(topics), len(embeddings)))
        topic_indices = {}
        
        for topic_idx, topic in enumerate(topics):
            # 獲取該主題的文檔索引
            doc_indices = metadata.index[metadata['main_topic'] == topic].tolist()
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
    
    def compute_attention(self, embeddings, metadata, **kwargs):
        """計算組合型注意力權重"""
        # 獲取組合權重
        weights_dict = kwargs.get('weights', {})
        similarity_weight = weights_dict.get('similarity', 0.33)
        keyword_weight = weights_dict.get('keyword', 0.33)
        self_weight = weights_dict.get('self', 0.34)
        
        # 確保權重總和為1
        total = similarity_weight + keyword_weight + self_weight
        if total != 1.0:
            similarity_weight /= total
            keyword_weight /= total
            self_weight /= total
        
        # 初始化各種注意力機制
        similarity_attn = SimilarityAttention(self.logger)
        keyword_attn = KeywordGuidedAttention(self.logger)
        self_attn = SelfAttention(self.logger)
        
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