"""
注意力融合網路 - 實現三種注意力機制的門控融合
包含特徵對齊、門控權重計算、權重歸一化和加權融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import math

logger = logging.getLogger(__name__)

class FeatureAligner(nn.Module):
    """特徵對齊模組，確保不同注意力機制輸出維度一致"""
    
    def __init__(self, input_dims: List[int], target_dim: int = 768):
        """
        初始化特徵對齊器
        
        Args:
            input_dims: 各注意力機制的輸入維度列表
            target_dim: 目標對齊維度
        """
        super(FeatureAligner, self).__init__()
        self.target_dim = target_dim
        self.aligners = nn.ModuleDict()
        
        # 為每個輸入維度創建對齊層
        for i, dim in enumerate(input_dims):
            if dim != target_dim:
                # 使用線性層進行維度對齊
                self.aligners[f'aligner_{i}'] = nn.Sequential(
                    nn.Linear(dim, target_dim),
                    nn.LayerNorm(target_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            else:
                # 維度已匹配，使用恆等映射
                self.aligners[f'aligner_{i}'] = nn.Identity()
    
    def forward(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        對齊特徵維度
        
        Args:
            features_list: 待對齊的特徵列表
            
        Returns:
            對齊後的特徵列表
        """
        aligned_features = []
        for i, features in enumerate(features_list):
            aligner = self.aligners[f'aligner_{i}']
            aligned = aligner(features)
            aligned_features.append(aligned)
        
        return aligned_features


class GatedFusionNetwork(nn.Module):
    """門控融合網路 - 三種門控機制的實現"""
    
    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256):
        """
        初始化門控融合網路
        
        Args:
            feature_dim: 特徵維度
            hidden_dim: 隱藏層維度
        """
        super(GatedFusionNetwork, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 相似度門控：計算特徵間的相似性
        self.similarity_gate = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),  # 三個注意力機制的特徵
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 輸出三個權重
            nn.Softmax(dim=-1)
        )
        
        # 關鍵詞門控：基於文本內容的關鍵詞密度
        self.keyword_gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softmax(dim=-1)
        )
        
        # 自注意力門控：計算特徵的自相關性
        self.self_attention_gate = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # 門控權重轉換
        self.gate_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # 最終融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, similarity_features: torch.Tensor,
                keyword_features: torch.Tensor,
                self_attention_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播
        
        Args:
            similarity_features: 相似度注意力特徵 [batch_size, feature_dim]
            keyword_features: 關鍵詞注意力特徵 [batch_size, feature_dim]
            self_attention_features: 自注意力特徵 [batch_size, feature_dim]
            
        Returns:
            fused_features: 融合後的特徵
            gate_weights: 各門控的權重
        """
        batch_size = similarity_features.size(0)
        
        # 1. 相似度門控計算
        concat_features = torch.cat([similarity_features, keyword_features, self_attention_features], dim=-1)
        similarity_weights = self.similarity_gate(concat_features)  # [batch_size, 3]
        
        # 2. 關鍵詞門控計算（使用平均特徵）
        avg_features = (similarity_features + keyword_features + self_attention_features) / 3
        keyword_weights = self.keyword_gate(avg_features)  # [batch_size, 3]
        
        # 3. 自注意力門控計算
        # 重塑為序列格式：[seq_len=3, batch_size, feature_dim]
        stacked_features = torch.stack([similarity_features, keyword_features, self_attention_features], dim=0)
        attended_features, attention_weights = self.self_attention_gate(
            stacked_features, stacked_features, stacked_features
        )
        # 平均注意力權重作為門控權重
        self_attention_weights = torch.mean(attention_weights, dim=1)  # [batch_size, 3]
        
        # 4. 權重歸一化 - 確保三個門控權重總和為1
        combined_weights = similarity_weights + keyword_weights + self_attention_weights
        normalized_weights = F.softmax(combined_weights, dim=-1)  # [batch_size, 3]
        
        # 5. 加權融合
        features_stack = torch.stack([similarity_features, keyword_features, self_attention_features], dim=1)  # [batch_size, 3, feature_dim]
        weighted_features = torch.sum(features_stack * normalized_weights.unsqueeze(-1), dim=1)  # [batch_size, feature_dim]
        
        # 6. 最終融合層
        fused_features = self.fusion_layer(weighted_features)
        
        # 準備輸出的權重信息
        gate_weights = {
            'similarity_weights': similarity_weights,
            'keyword_weights': keyword_weights,
            'self_attention_weights': self_attention_weights,
            'normalized_weights': normalized_weights,
            'final_weights': {
                'similarity': normalized_weights[:, 0],
                'keyword': normalized_weights[:, 1],
                'self_attention': normalized_weights[:, 2]
            }
        }
        
        return fused_features, gate_weights


class AttentionFusionProcessor:
    """注意力融合處理器 - 整合三種注意力機制並進行融合"""
    
    def __init__(self, feature_dim: int = 768, device: str = 'auto'):
        """
        初始化融合處理器
        
        Args:
            feature_dim: 特徵維度
            device: 計算設備
        """
        self.feature_dim = feature_dim
        
        # 設備選擇
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"注意力融合處理器使用設備: {self.device}")
        
        # 初始化特徵對齊器和融合網路
        self.feature_aligner = None
        self.fusion_network = None
        
        # GPU記憶體優化
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def _initialize_networks(self, input_dims: List[int]):
        """初始化網路結構"""
        if self.feature_aligner is None:
            self.feature_aligner = FeatureAligner(input_dims, self.feature_dim).to(self.device)
            self.fusion_network = GatedFusionNetwork(self.feature_dim).to(self.device)
            
            # 設為評估模式以獲得一致結果
            self.feature_aligner.eval()
            self.fusion_network.eval()
            
            logger.info(f"網路結構已初始化並移至 {self.device}")
    
    def fuse_attention_features(self, 
                              similarity_vectors: np.ndarray,
                              keyword_vectors: np.ndarray,
                              self_attention_vectors: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        融合三種注意力機制的特徵向量
        
        Args:
            similarity_vectors: 相似度注意力向量 [batch_size, dim1]
            keyword_vectors: 關鍵詞注意力向量 [batch_size, dim2]  
            self_attention_vectors: 自注意力向量 [batch_size, dim3]
            
        Returns:
            fused_features: 融合後的特徵 [batch_size, feature_dim]
            fusion_info: 融合過程的詳細信息
        """
        # 檢查輸入維度
        input_dims = [
            similarity_vectors.shape[-1],
            keyword_vectors.shape[-1], 
            self_attention_vectors.shape[-1]
        ]
        
        # 初始化網路（如果需要）
        self._initialize_networks(input_dims)
        
        try:
            with torch.no_grad():
                # 轉換為張量
                sim_tensor = torch.tensor(similarity_vectors, dtype=torch.float32).to(self.device)
                key_tensor = torch.tensor(keyword_vectors, dtype=torch.float32).to(self.device)
                self_tensor = torch.tensor(self_attention_vectors, dtype=torch.float32).to(self.device)
                
                # 特徵對齊
                aligned_features = self.feature_aligner([sim_tensor, key_tensor, self_tensor])
                
                # 門控融合
                fused_features, gate_weights = self.fusion_network(
                    aligned_features[0], aligned_features[1], aligned_features[2]
                )
                
                # 轉換回numpy格式
                fused_np = fused_features.cpu().numpy()
                
                # 準備融合信息
                fusion_info = {
                    'input_dims': input_dims,
                    'output_dim': fused_np.shape[-1],
                    'gate_weights': {
                        'similarity': gate_weights['final_weights']['similarity'].cpu().numpy(),
                        'keyword': gate_weights['final_weights']['keyword'].cpu().numpy(), 
                        'self_attention': gate_weights['final_weights']['self_attention'].cpu().numpy()
                    },
                    'average_weights': {
                        'similarity': float(torch.mean(gate_weights['final_weights']['similarity']).cpu()),
                        'keyword': float(torch.mean(gate_weights['final_weights']['keyword']).cpu()),
                        'self_attention': float(torch.mean(gate_weights['final_weights']['self_attention']).cpu())
                    }
                }
                
                logger.debug(f"融合完成 - 形狀:{fused_np.shape} 權重:(相似度:{fusion_info['average_weights']['similarity']:.3f} 關鍵詞:{fusion_info['average_weights']['keyword']:.3f} 自注意力:{fusion_info['average_weights']['self_attention']:.3f})")
                
                return fused_np, fusion_info
                
        except Exception as e:
            logger.error(f"注意力融合過程中發生錯誤: {str(e)}")
            # GPU記憶體清理
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise
        
        finally:
            # 清理GPU記憶體
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
    
    def compute_parallel_attention_features(self, 
                                          embeddings: np.ndarray,
                                          metadata,
                                          **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        並行計算三種注意力機制的特徵
        
        Args:
            embeddings: 文本嵌入向量
            metadata: 元數據
            **kwargs: 其他參數
            
        Returns:
            similarity_features: 相似度注意力特徵
            keyword_features: 關鍵詞注意力特徵
            self_attention_features: 自注意力特徵
            attention_info: 注意力計算信息
        """
        from .attention_mechanism import SimilarityAttention, KeywordGuidedAttention, SelfAttention
        
        logger.info("🔄 開始並行計算三種注意力機制特徵...")
        
        # 初始化三種注意力機制
        similarity_attn = SimilarityAttention()
        keyword_attn = KeywordGuidedAttention()
        self_attn = SelfAttention()
        
        attention_results = {}
        
        # 並行計算（在實際實現中可以使用多線程）
        mechanisms = [
            ('similarity', similarity_attn),
            ('keyword', keyword_attn), 
            ('self_attention', self_attn)
        ]
        
        features_list = []
        
        for name, mechanism in mechanisms:
            logger.debug(f"計算 {name} 注意力機制")
            
            try:
                # 計算注意力權重和面向向量
                aspect_vectors, attention_data = mechanism.compute_aspect_vectors(
                    embeddings, metadata, **kwargs
                )
                
                # 將aspect vectors轉換為document-level features
                if aspect_vectors:
                    # 為每個文檔計算與各aspect的相似度特徵
                    num_docs = len(embeddings)
                    doc_features = []
                    
                    # 獲取所有aspect名稱並排序以確保一致性
                    sorted_aspects = sorted(aspect_vectors.keys())
                    
                    for doc_idx in range(num_docs):
                        doc_embedding = embeddings[doc_idx]
                        
                        # 計算文檔與每個aspect的相似度分數
                        similarity_scores = []
                        for aspect_name in sorted_aspects:
                            aspect_vector = aspect_vectors[aspect_name]
                            
                            # 計算餘弦相似度
                            similarity = np.dot(doc_embedding, aspect_vector) / (
                                np.linalg.norm(doc_embedding) * np.linalg.norm(aspect_vector) + 1e-8
                            )
                            similarity_scores.append(similarity)
                        
                        # 使用相似度分數作為document-level特徵
                        # 同時加入原始embedding的統計特徵
                        doc_stats = [
                            np.mean(doc_embedding),    # 平均值
                            np.std(doc_embedding),     # 標準差  
                            np.max(doc_embedding),     # 最大值
                            np.min(doc_embedding)      # 最小值
                        ]
                        
                        # 組合特徵：相似度分數 + 文檔統計特徵
                        combined_feature = np.array(similarity_scores + doc_stats)
                        doc_features.append(combined_feature)
                    
                    features = np.array(doc_features)  # [num_docs, num_aspects + 4]
                    features_list.append(features)
                    
                    attention_results[name] = {
                        'aspect_vectors': aspect_vectors,
                        'attention_data': attention_data,
                        'features_shape': features.shape
                    }
                    
                    logger.debug(f"{name} 注意力特徵形狀: {features.shape}")
                else:
                    # 如果沒有aspect vectors，創建預設特徵
                    num_docs = len(embeddings)
                    # 使用文檔embedding的統計特徵作為預設特徵
                    doc_features = []
                    for doc_idx in range(num_docs):
                        doc_embedding = embeddings[doc_idx]
                        # 計算文檔統計特徵
                        doc_stats = [
                            np.mean(doc_embedding),    # 平均值
                            np.std(doc_embedding),     # 標準差  
                            np.max(doc_embedding),     # 最大值
                            np.min(doc_embedding),     # 最小值
                            0.0, 0.0, 0.0              # 3個預設相似度分數
                        ]
                        doc_features.append(np.array(doc_stats))
                    
                    features = np.array(doc_features)  # [num_docs, 7]
                    features_list.append(features)
                    logger.warning(f"{name} 注意力沒有產生有效aspect vectors，使用文檔統計特徵")
                    
            except Exception as e:
                logger.error(f"計算 {name} 注意力時發生錯誤: {str(e)}")
                # 回退到統計特徵
                num_docs = len(embeddings)
                doc_features = []
                for doc_idx in range(num_docs):
                    doc_embedding = embeddings[doc_idx]
                    # 使用文檔統計特徵作為回退方案
                    doc_stats = [
                        np.mean(doc_embedding),    # 平均值
                        np.std(doc_embedding),     # 標準差  
                        np.max(doc_embedding),     # 最大值
                        np.min(doc_embedding),     # 最小值
                        0.0, 0.0, 0.0              # 3個預設相似度分數
                    ]
                    doc_features.append(np.array(doc_stats))
                
                fallback_features = np.array(doc_features)  # [num_docs, 7]
                features_list.append(fallback_features)
        
        # 檢查特徵維度並進行對齊
        if len(features_list) >= 3:
            logger.debug("檢查特徵維度一致性...")
            dims = [f.shape for f in features_list]
            logger.debug(f"各機制特徵形狀: {dims}")
            
            # 確保所有特徵都有相同的樣本數
            min_samples = min(f.shape[0] for f in features_list)
            features_list = [f[:min_samples] for f in features_list]
            
            # 檢查特徵維度並進行對齊  
            feature_dims = [f.shape[1] for f in features_list]
            if len(set(feature_dims)) > 1:
                logger.info(f"檢測到不同特徵維度: {feature_dims}，進行特徵對齊...")
                # 統一到最大維度
                max_dim = max(feature_dims) 
                aligned_features = []
                for i, features in enumerate(features_list):
                    if features.shape[1] < max_dim:
                        # 零填充到最大維度
                        padding = np.zeros((features.shape[0], max_dim - features.shape[1]))
                        aligned_features.append(np.concatenate([features, padding], axis=1))
                    else:
                        aligned_features.append(features)
                features_list = aligned_features
                logger.info(f"特徵對齊完成 - 所有機制特徵維度統一為: {max_dim}")
            else:
                logger.info(f"所有注意力機制的特徵維度已一致: {feature_dims[0]}")
        
        attention_info = {
            'mechanisms_computed': len(features_list),
            'feature_shapes': [f.shape for f in features_list],
            'attention_results': attention_results
        }
        
        logger.info("✅ 三種注意力機制特徵計算完成，特徵已對齊為document-level格式")
        
        return features_list[0], features_list[1], features_list[2], attention_info


# 測試代碼
if __name__ == "__main__":
    # 配置日誌
    logging.basicConfig(level=logging.INFO)
    
    # 創建測試數據
    batch_size = 10
    feature_dim = 768
    
    # 模擬三種注意力特徵（不同維度以測試對齊）
    similarity_features = np.random.randn(batch_size, feature_dim)
    keyword_features = np.random.randn(batch_size, 512)  # 不同維度
    self_attention_features = np.random.randn(batch_size, 256)  # 不同維度
    
    # 測試融合處理器
    fusion_processor = AttentionFusionProcessor(feature_dim=feature_dim)
    
    # 執行融合
    fused_features, fusion_info = fusion_processor.fuse_attention_features(
        similarity_features, keyword_features, self_attention_features
    )
    
    print(f"融合結果形狀: {fused_features.shape}")
    print(f"平均權重: {fusion_info['average_weights']}")
    print("注意力融合網路測試完成！")