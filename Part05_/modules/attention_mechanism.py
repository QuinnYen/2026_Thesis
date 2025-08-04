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
import torch.nn as nn
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
        self.logger.debug(f"開始評估，共有 {len(topics)} 個面向")
        
        # 計算面向內聚度
        coherence = 0.0
        doc_count = 0
        topic_coherence_dict = {}
        
        self.logger.debug("開始計算面向內聚度...")
        
        for topic in topics:
            # 獲取該面向的文檔
            topic_docs = metadata[metadata['sentiment'] == topic] if 'sentiment' in metadata.columns else metadata[metadata.index.isin([topic])]
            
            if len(topic_docs) == 0:
                self.logger.debug(f"面向 '{topic}' 沒有對應的文檔")
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
                
                self.logger.debug(f"面向 '{topic}' 的內聚度: {topic_coherence:.4f} (文檔數: {len(topic_embeddings)})")
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
        
        self.logger.debug("開始計算面向間的分離度...")
        
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
                
                self.logger.debug(f"面向 '{topic_i}' 和 '{topic_j}' 的分離度: {distance:.4f}")
        
        if pair_count > 0:
            separation /= pair_count
            
        self.logger.info(f"總體分離度: {separation:.4f}")
            
        # 計算每個面向的平均分離度
        aspect_avg_separation = {}
        for topic in topics:
            if aspect_separation_values[topic]:
                aspect_avg_separation[topic] = float(np.mean(aspect_separation_values[topic]))
                self.logger.debug(f"面向 '{topic}' 的平均分離度: {aspect_avg_separation[topic]:.4f}")
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
            
            # 驗證索引有效性並過濾無效索引
            valid_doc_indices = []
            for idx in doc_indices:
                if 0 <= idx < len(embeddings):
                    valid_doc_indices.append(idx)
                else:
                    self.logger.warning(f"文檔索引 {idx} 超出嵌入向量範圍 [0, {len(embeddings)-1}]")
            
            topic_indices[topic] = valid_doc_indices
            
            # 設置均等權重
            if valid_doc_indices:
                uniform_weight = 1.0 / len(valid_doc_indices)
                for doc_idx in valid_doc_indices:
                    weights[topic_idx][doc_idx] = uniform_weight
        
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
            
            # 驗證索引有效性並過濾無效索引
            valid_doc_indices = []
            for idx in doc_indices:
                if 0 <= idx < len(embeddings):
                    valid_doc_indices.append(idx)
                else:
                    self.logger.warning(f"文檔索引 {idx} 超出嵌入向量範圍 [0, {len(embeddings)-1}]")
            
            # 更新有效的文檔索引
            topic_indices[topic] = valid_doc_indices
            
            if not valid_doc_indices:
                continue
                
            # 獲取該主題的嵌入向量
            topic_embeddings = embeddings[valid_doc_indices]
            
            # 計算中心向量
            center = np.mean(topic_embeddings, axis=0)
            
            # 計算每個文檔與中心的相似度
            similarities = []
            for embed in topic_embeddings:
                # 添加數值穩定性檢查
                center_norm = np.linalg.norm(center)
                embed_norm = np.linalg.norm(embed)
                
                if center_norm > 1e-8 and embed_norm > 1e-8:
                    similarity = np.dot(center, embed) / (center_norm * embed_norm)
                else:
                    similarity = 0.0
                similarities.append(similarity)
            
            # 轉換為 softmax 權重
            similarities = np.array(similarities)
            
            # 如果所有相似度都相同，使用均等權重
            if len(similarities) == 0 or np.all(similarities == similarities[0]):
                topic_weights = np.ones(len(valid_doc_indices)) / len(valid_doc_indices)
            else:
                # 避免數值不穩定
                similarities = similarities - np.max(similarities)
                exp_sim = np.exp(similarities)
                exp_sum = np.sum(exp_sim)
                if exp_sum > 1e-8:
                    topic_weights = exp_sim / exp_sum
                else:
                    topic_weights = np.ones(len(valid_doc_indices)) / len(valid_doc_indices)
            
            # 逐個填充權重矩陣
            for i, doc_idx in enumerate(valid_doc_indices):
                weights[topic_idx][doc_idx] = topic_weights[i]
        
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
                for doc_idx in doc_indices:
                    if 0 <= doc_idx < len(embeddings):
                        weights[topic_idx][doc_idx] = 1.0 / len(doc_indices)
                continue
            
            # 計算關鍵詞分數
            keyword_scores = []
            valid_doc_indices = []
            
            for idx in doc_indices:
                # 檢查索引有效性
                if not (0 <= idx < len(texts)):
                    self.logger.warning(f"文檔索引 {idx} 超出範圍 [0, {len(texts)-1}]")
                    continue
                    
                valid_doc_indices.append(idx)
                text = texts[idx].lower()
                score = 0
                
                # 計算關鍵詞出現次數的加權和
                for i, keyword in enumerate(keywords):
                    # 關鍵詞權重隨排名遞減
                    keyword_weight = 1.0 / (i + 1)
                    count = text.count(keyword.lower())
                    score += count * keyword_weight
                
                keyword_scores.append(score)
            
            # 更新有效的文檔索引
            topic_indices[topic] = valid_doc_indices
            
            if not valid_doc_indices:
                continue
            
            # 如果所有分數為零，使用平均權重
            if sum(keyword_scores) == 0:
                uniform_weight = 1.0 / len(valid_doc_indices)
                for doc_idx in valid_doc_indices:
                    if 0 <= doc_idx < len(embeddings):
                        weights[topic_idx][doc_idx] = uniform_weight
            else:
                # 正規化分數
                keyword_scores = np.array(keyword_scores)
                topic_weights = keyword_scores / np.sum(keyword_scores)
                
                # 逐個設置權重
                for i, doc_idx in enumerate(valid_doc_indices):
                    if 0 <= doc_idx < len(embeddings):
                        weights[topic_idx][doc_idx] = topic_weights[i]
        
        return weights, topic_indices


class SelfAttention(AttentionMechanism):
    """自注意力機制"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🔧 SelfAttention 使用設備: {self.device}")
        
        # 優化GPU記憶體使用
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
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
            
            # 驗證索引有效性並過濾無效索引
            valid_doc_indices = []
            for idx in doc_indices:
                if 0 <= idx < len(embeddings):
                    valid_doc_indices.append(idx)
                else:
                    self.logger.warning(f"文檔索引 {idx} 超出嵌入向量範圍 [0, {len(embeddings)-1}]")
            
            topic_indices[topic] = valid_doc_indices
            
            if not valid_doc_indices or len(valid_doc_indices) < 2:
                # 如果文檔數少於2，使用均等權重
                if valid_doc_indices:
                    uniform_weight = 1.0 / len(valid_doc_indices)
                    for doc_idx in valid_doc_indices:
                        weights[topic_idx][doc_idx] = uniform_weight
                continue
                
            # 獲取該主題的嵌入向量
            topic_embeddings = embeddings[valid_doc_indices]
            
            # 轉換為張量並確保在正確的設備上
            try:
                topic_embeddings_tensor = torch.tensor(topic_embeddings, dtype=torch.float32).to(self.device)
                
                # 計算注意力分數
                # 使用縮放點積注意力
                d_k = topic_embeddings_tensor.size(-1)
                scores = torch.matmul(topic_embeddings_tensor, topic_embeddings_tensor.transpose(-2, -1)) / math.sqrt(d_k)
                
                # 使用softmax轉換為權重
                attn_weights = F.softmax(scores, dim=-1)
                
                # 計算每個文檔的平均注意力得分
                avg_weights = torch.mean(attn_weights, dim=-1).cpu().numpy()
                
                # 逐個填充權重矩陣
                for i, doc_idx in enumerate(valid_doc_indices):
                    weights[topic_idx][doc_idx] = avg_weights[i]
                    
            except Exception as e:
                self.logger.warning(f"自注意力計算失敗，回退到均等權重: {str(e)}")
                # 回退到均等權重
                uniform_weight = 1.0 / len(valid_doc_indices)
                for doc_idx in valid_doc_indices:
                    weights[topic_idx][doc_idx] = uniform_weight
        
        return weights, topic_indices


class GatedFusionNetwork(nn.Module):
    """門控動態融合神經網路
    
    根據輸入文本特徵動態計算注意力機制的權重，
    解決靜態權重分配的問題
    """
    
    def __init__(self, input_dim: int, num_mechanisms: int, hidden_dim: int = 128):
        """初始化門控融合網路
        
        Args:
            input_dim: 輸入特徵維度（文本嵌入維度）
            num_mechanisms: 注意力機制數量
            hidden_dim: 隱藏層維度
        """
        super(GatedFusionNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_mechanisms = num_mechanisms
        self.hidden_dim = hidden_dim
        
        # 特徵提取網路
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 門控權重生成網路
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_mechanisms)
            # 移除 Softmax，將在 forward 方法中手動應用
        )
        
        # 互補性建模層
        self.complementarity_layer = nn.Linear(num_mechanisms, num_mechanisms)
        
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """前向傳播
        
        Args:
            text_features: 文本特徵張量 [batch_size, input_dim]
            
        Returns:
            dynamic_weights: 動態權重 [batch_size, num_mechanisms]
        """
        # 特徵提取
        extracted_features = self.feature_extractor(text_features)
        
        # 生成基礎門控權重 (移除這裡的 Softmax，避免雙重歸一化)
        base_logits = self.gate_network(extracted_features)  # 現在直接使用，因為已移除 Softmax 層
        
        # 互補性調整
        complementarity_weights = torch.sigmoid(self.complementarity_layer(
            torch.softmax(base_logits, dim=1)  # 暫時應用 softmax 用於互補性計算
        ))
        
        # 結合基礎權重和互補性權重
        # 使用加法而不是乘法，避免權重過小
        dynamic_logits = base_logits + torch.log(complementarity_weights + 1e-8)
        
        # 最終歸一化
        dynamic_weights = F.softmax(dynamic_logits, dim=1)
        
        return dynamic_weights


class DynamicCombinedAttention(AttentionMechanism):
    """動態組合注意力機制
    
    使用門控融合網路動態計算注意力權重，
    根據文本內容自動調整注意力組合
    """
    
    def __init__(self, config=None):
        """初始化動態組合注意力機制"""
        super().__init__(config)
        self.fusion_network = None
        # 設備選擇和管理
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"🔧 DynamicCombinedAttention 使用設備: {self.device}")
        self.mechanism_names = ['similarity', 'keyword', 'self']
        
        # 優化GPU記憶體使用
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            self.logger.info(f"   GPU記憶體已清理: {torch.cuda.get_device_name()}")
            self.logger.info(f"   可用GPU記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
    def _initialize_fusion_network(self, input_dim: int):
        """初始化融合網路"""
        if self.fusion_network is None:
            self.logger.info(f"   🔧 初始化GatedFusionNetwork (輸入維度: {input_dim}, 設備: {self.device})")
            
            try:
                self.fusion_network = GatedFusionNetwork(
                    input_dim=input_dim,
                    num_mechanisms=len(self.mechanism_names),
                    hidden_dim=128
                ).to(self.device)
                
                # 設置為評估模式以獲得一致的結果
                self.fusion_network.eval()
                
                self.logger.info(f"   ✅ GatedFusionNetwork 已成功初始化並移至 {self.device}")
                
                # GPU記憶體檢查
                if self.device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    self.logger.info(f"   GPU記憶體使用: {allocated:.1f}MB")
                    
            except Exception as e:
                self.logger.error(f"   ❌ GatedFusionNetwork 初始化失敗: {str(e)}")
                # 回退到CPU
                if self.device.type == 'cuda':
                    self.logger.warning("   ⚠️ 回退到CPU設備")
                    self.device = torch.device('cpu')
                    self.fusion_network = GatedFusionNetwork(
                        input_dim=input_dim,
                        num_mechanisms=len(self.mechanism_names),
                        hidden_dim=128
                    ).to(self.device)
                    self.fusion_network.eval()
                else:
                    raise
    
    def _extract_text_features(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> torch.Tensor:
        """從嵌入和元數據中提取文本特徵
        
        Args:
            embeddings: 文本嵌入
            metadata: 元數據
            
        Returns:
            text_features: 文本特徵張量
        """
        # 計算統計特徵
        batch_size = embeddings.shape[0]
        features_list = []
        
        for i in range(batch_size):
            # 基礎統計特徵
            embedding_mean = np.mean(embeddings[i])
            embedding_std = np.std(embeddings[i])
            embedding_max = np.max(embeddings[i])
            embedding_min = np.min(embeddings[i])
            
            # 文本長度特徵（如果有的話）
            text_length = len(metadata.iloc[i].get('text', '')) if 'text' in metadata.columns else 100
            text_length_norm = min(text_length / 500.0, 1.0)  # 正規化長度
            
            # 組合特徵
            features = [
                embedding_mean, embedding_std, embedding_max, embedding_min,
                text_length_norm
            ]
            
            # 加入嵌入的前幾個維度作為特徵
            top_dims = min(10, embeddings.shape[1])
            features.extend(embeddings[i][:top_dims].tolist())
            
            features_list.append(features)
        
        # 轉換為張量並確保在正確的設備上
        features_array = np.array(features_list)
        
        try:
            # 直接在目標設備上創建張量以避免CPU-GPU傳輸
            if self.device.type == 'cuda':
                text_features = torch.cuda.FloatTensor(features_array)
            else:
                text_features = torch.FloatTensor(features_array).to(self.device)
                
            self.logger.debug(f"   📊 文本特徵張量: 形狀={text_features.shape}, 設備={text_features.device}")
            return text_features
            
        except Exception as e:
            self.logger.warning(f"   ⚠️ GPU張量創建失敗，回退到CPU: {str(e)}")
            # 回退到CPU
            return torch.FloatTensor(features_array).to(torch.device('cpu'))
    
    def compute_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """計算動態組合注意力權重"""
        self.logger.info(f"🎯 開始動態注意力計算 (嵌入向量形狀: {embeddings.shape})")
        
        # 記錄初始GPU記憶體使用情況
        if self.device.type == 'cuda':
            initial_memory = torch.cuda.memory_allocated() / 1024**2
            self.logger.debug(f"   初始GPU記憶體: {initial_memory:.1f}MB")
        
        try:
            # 初始化融合網路
            feature_dim = 5 + min(10, embeddings.shape[1])  # 5個統計特徵 + 前10個嵌入維度
            self._initialize_fusion_network(feature_dim)
            
            # 提取文本特徵
            text_features = self._extract_text_features(embeddings, metadata)
            
            # 確保融合網路和文本特徵在同一設備上
            if text_features.device != next(self.fusion_network.parameters()).device:
                self.logger.warning(f"   ⚠️ 設備不匹配，調整文本特徵設備: {text_features.device} -> {next(self.fusion_network.parameters()).device}")
                text_features = text_features.to(next(self.fusion_network.parameters()).device)
            
            # 使用融合網路計算動態權重
            with torch.no_grad():
                # 確保在評估模式
                self.fusion_network.eval()
                
                dynamic_weights = self.fusion_network(text_features)
                
                # 確保結果在CPU上以避免後續處理的設備問題
                if dynamic_weights.device.type == 'cuda':
                    dynamic_weights_np = dynamic_weights.cpu().numpy()
                else:
                    dynamic_weights_np = dynamic_weights.numpy()
                    
                # 清理GPU記憶體
                if self.device.type == 'cuda':
                    del dynamic_weights  # 明確刪除GPU張量
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            self.logger.error(f"   ❌ 動態權重計算失敗: {str(e)}")
            # 清理GPU記憶體
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise
        
        # 將動態權重轉換為字典格式
        weights_dict = {}
        for i, mechanism_name in enumerate(self.mechanism_names):
            # 對每個樣本的權重取平均
            weights_dict[mechanism_name] = float(np.mean(dynamic_weights_np[:, i]))
        
        # 記錄最終GPU記憶體使用情況
        if self.device.type == 'cuda':
            final_memory = torch.cuda.memory_allocated() / 1024**2
            memory_diff = final_memory - initial_memory
            self.logger.debug(f"   最終GPU記憶體: {final_memory:.1f}MB (差異: {memory_diff:+.1f}MB)")
        
        # 添加詳細的調試日誌
        self.logger.info(f"🔍 GatedFusionNetwork 動態權重詳細資訊:")
        self.logger.info(f"   動態權重張量形狀: {dynamic_weights_np.shape}")
        self.logger.info(f"   平均後的權重字典: {weights_dict}")
        
        # 檢查權重是否有效
        total_weight = sum(weights_dict.values())
        min_weight = min(weights_dict.values())
        max_weight = max(weights_dict.values())
        self.logger.info(f"   權重統計 - 總和: {total_weight:.6f}, 最小值: {min_weight:.6f}, 最大值: {max_weight:.6f}")
        
        # 顯示前3個樣本的權重（如果有多個樣本）
        if dynamic_weights_np.shape[0] > 1:
            for sample_idx in range(min(3, dynamic_weights_np.shape[0])):
                sample_weights = {mechanism: dynamic_weights_np[sample_idx, i] 
                                for i, mechanism in enumerate(self.mechanism_names)}
                self.logger.debug(f"   樣本 {sample_idx} 權重: {sample_weights}")
        
        self.logger.info(f"✅ 動態權重計算完成")
        
        # 使用動態權重計算注意力
        weights, indices = self._compute_weighted_attention(embeddings, metadata, weights_dict, **kwargs)
        
        # 將動態權重信息添加到indices中，供GUI顯示使用
        if isinstance(indices, dict):
            indices['dynamic_weights'] = weights_dict
            indices['is_dynamic'] = True
        else:
            indices = {
                'topic_indices': indices,
                'dynamic_weights': weights_dict,
                'is_dynamic': True
            }
            
        return weights, indices
    
    def _compute_weighted_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, 
                                   weights_dict: Dict[str, float], **kwargs) -> Tuple[np.ndarray, Dict]:
        """使用權重計算注意力（與原CombinedAttention相同的邏輯）"""
        # 添加調試日誌
        self.logger.info(f"🔍 _compute_weighted_attention 輸入權重: {weights_dict}")
        
        # 只保留權重不為0的注意力機制
        # 使用更合理的閾值，因為浮點數精度問題可能導致小的正數被誤判為0
        min_threshold = 1e-6
        active_weights = {k: v for k, v in weights_dict.items() if v > min_threshold}
        
        self.logger.info(f"   過濾閾值: {min_threshold}")
        self.logger.info(f"   過濾後的有效權重: {active_weights}")
        
        if not active_weights:
            self.logger.warning("⚠️  沒有指定任何有效的注意力權重，使用均等權重")
            self.logger.warning(f"   原始權重字典: {weights_dict}")
            self.logger.warning(f"   所有權重值都 <= 0，將回退到 NoAttention")
            return NoAttention(self.config).compute_attention(embeddings, metadata, **kwargs)
        
        self.logger.info(f"✅ 使用有效權重進行動態注意力計算: {active_weights}")
        
        # 初始化權重矩陣和結果
        weights = None
        topic_indices = None
        
        # 根據權重計算對應的注意力機制
        for mechanism_name, weight in active_weights.items():
            if mechanism_name == 'similarity':
                mechanism = SimilarityAttention(self.config)
            elif mechanism_name == 'keyword':
                mechanism = KeywordGuidedAttention(self.config)
            elif mechanism_name == 'self':
                mechanism = SelfAttention(self.config)
            else:
                self.logger.warning(f"未知的注意力機制: {mechanism_name}，將被忽略")
                continue
            
            # 計算該機制的注意力權重
            mechanism_weights, indices = mechanism.compute_attention(embeddings, metadata, **kwargs)
            
            # 如果是第一個機制，初始化結果
            if weights is None:
                weights = weight * mechanism_weights
                topic_indices = indices
            else:
                # 累加加權結果
                weights += weight * mechanism_weights
        
        return weights, topic_indices


class CombinedAttention(AttentionMechanism):
    """組合型注意力機制（原有的靜態版本）"""
    
    def compute_attention(self, embeddings: np.ndarray, metadata: pd.DataFrame, **kwargs) -> Tuple[np.ndarray, Dict]:
        """計算組合型注意力權重"""
        # 獲取組合權重
        weights_dict = kwargs.get('weights', {})
        
        # 只保留權重不為0的注意力機制
        active_weights = {k: v for k, v in weights_dict.items() if v > 0}
        
        if not active_weights:
            self.logger.warning("沒有指定任何有效的注意力權重，使用均等權重")
            return NoAttention(self.config).compute_attention(embeddings, metadata, **kwargs)
        
        # 動態正規化權重確保總和為1
        total_weight = sum(active_weights.values())
        if abs(total_weight - 1.0) > 1e-5:
            self.logger.warning(f"注意力權重總和 {total_weight} 不為1，將進行歸一化")
            active_weights = {k: v/total_weight for k, v in active_weights.items()}
        
        # 初始化權重矩陣和結果
        weights = None
        topic_indices = None
        
        # 根據權重計算對應的注意力機制
        for mechanism_name, weight in active_weights.items():
            if mechanism_name == 'similarity':
                mechanism = SimilarityAttention(self.config)
            elif mechanism_name == 'keyword':
                mechanism = KeywordGuidedAttention(self.config)
            elif mechanism_name == 'self':
                mechanism = SelfAttention(self.config)
            else:
                self.logger.warning(f"未知的注意力機制: {mechanism_name}，將被忽略")
                continue
            
            # 計算該機制的注意力權重
            mechanism_weights, indices = mechanism.compute_attention(embeddings, metadata, **kwargs)
            
            # 如果是第一個機制，初始化結果
            if weights is None:
                weights = weight * mechanism_weights
                topic_indices = indices
            else:
                # 累加加權結果
                weights += weight * mechanism_weights
        
        return weights, topic_indices


def create_attention_mechanism(attention_type: str, config=None) -> AttentionMechanism:
    """創建指定類型的注意力機制
    
    Args:
        attention_type: 注意力機制類型，可以是'no', 'similarity', 'keyword', 'self', 'combined'或'dynamic'
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
    elif attention_type == 'dynamic' or attention_type == 'dynamic_combined':
        return DynamicCombinedAttention(config)
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
    
    # 顯示當前處理的注意力機制
    logger.debug(f"   🔄 正在計算 {attention_type} 注意力機制...")
    
    # 創建注意力機制
    attention_mech = create_attention_mechanism(attention_type, config)
    
    # 準備關鍵字參數
    kwargs = {}
    if topics_path:
        kwargs['topics_path'] = topics_path
    if weights:
        kwargs['weights'] = weights
    
    # 計算面向向量
    logger.debug(f"   📊 計算面向向量...")
    aspect_vectors, attention_data = attention_mech.compute_aspect_vectors(
        embeddings, metadata, **kwargs
    )
    
    # 評估面向向量
    logger.debug(f"   📏 評估指標...")
    metrics = attention_mech.evaluate(aspect_vectors, embeddings, metadata)
    
    # 顯示結果
    logger.info(f"   ✅ {attention_type} 注意力完成 - 內聚度: {metrics['coherence']:.4f}, "
               f"分離度: {metrics['separation']:.4f}, 綜合得分: {metrics['combined_score']:.4f}")
    
    result = {
        'aspect_vectors': aspect_vectors,
        'attention_data': attention_data,
        'metrics': metrics,
        'attention_type': attention_type
    }
    
    # 如果有動態權重信息，添加到結果中
    if isinstance(attention_data, dict) and 'dynamic_weights' in attention_data:
        result['dynamic_weights'] = attention_data['dynamic_weights']
        result['is_dynamic'] = attention_data.get('is_dynamic', False)
        logger.debug(f"   🎯 動態權重: {attention_data['dynamic_weights']}")
    
    return result


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
        if attn_type == 'combined':
            # 測試不同的組合權重
            test_weights = [
                {'similarity': 1.0},  # 單一注意力
                {'similarity': 0.5, 'keyword': 0.5},  # 雙重組合
                {'similarity': 0.33, 'keyword': 0.33, 'self': 0.34}  # 三重組合
            ]
            
            for weights in test_weights:
                logger.info(f"測試組合權重: {weights}")
                weights_array, topic_indices = attention_mech.compute_attention(
                    embeddings, metadata, weights=weights
                )
        else:
            weights_array, topic_indices = attention_mech.compute_attention(embeddings, metadata)
        
        # 測試計算面向向量
        aspect_vectors, _ = attention_mech.compute_aspect_vectors(embeddings, metadata)
        
        # 測試評估指標
        metrics = attention_mech.evaluate(aspect_vectors, embeddings, metadata)
        
        logger.info(f"內聚度: {metrics['coherence']:.4f}")
        logger.info(f"分離度: {metrics['separation']:.4f}")
        logger.info(f"綜合得分: {metrics['combined_score']:.4f}")
        logger.info("---")
    
    logger.info("注意力機制測試完成") 