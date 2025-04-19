"""
注意力機制模組
此模組實現多種注意力機制處理文本表示
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import pandas as pd
import logging
import time
import sys

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 從utils模組導入工具類
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.config_manager import ConfigManager

class SelfAttention(nn.Module):
    """
    自注意力機制模型
    用於從多個文本嵌入向量中學習重要特徵
    """
    def __init__(self, hidden_dim: int, attention_dim: int = None):
        """
        初始化自注意力機制
        
        Args:
            hidden_dim: 輸入向量維度
            attention_dim: 注意力層維度，不指定則使用hidden_dim
        """
        super(SelfAttention, self).__init__()
        
        if attention_dim is None:
            attention_dim = hidden_dim
        
        # 定義注意力層
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
    
    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳遞
        
        Args:
            inputs: 輸入張量, shape: [batch_size, seq_len, hidden_dim]
            mask: 掩碼張量, shape: [batch_size, seq_len], 用於屏蔽填充位置
            
        Returns:
            (weighted_outputs, attention_weights): 加權后的輸出和注意力權重
        """
        # 計算注意力分數: [batch_size, seq_len, 1]
        attention_scores = self.attention(inputs)
        
        # 使用掩碼（如果提供）
        if mask is not None:
            # 擴展維度以便廣播
            mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            # 將填充位置的分數設為非常小的負數
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        # 按序列長度維度做softmax，得到注意力權重: [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加權平均: [batch_size, hidden_dim]
        weighted_outputs = torch.sum(attention_weights * inputs, dim=1)
        
        # 去掉最後一個維度，得到: [batch_size, seq_len]
        attention_weights = attention_weights.squeeze(-1)
        
        return weighted_outputs, attention_weights
    
class MultiHeadAttention(nn.Module):
    """
    多頭注意力機制模型
    用於從多個角度分析文本嵌入向量
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        初始化多頭注意力機制
        
        Args:
            hidden_dim: 輸入向量維度
            num_heads: 注意力頭數量
            dropout: dropout比例
        """
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "隱藏維度必須能被頭數整除"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 定義查詢、鍵、值的線性變換層
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # 輸出層
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** 0.5  # 縮放因子
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳遞
        
        Args:
            query: 查詢張量, shape: [batch_size, seq_len_q, hidden_dim]
            key: 鍵張量, shape: [batch_size, seq_len_k, hidden_dim]
            value: 值張量, shape: [batch_size, seq_len_v, hidden_dim]
            mask: 掩碼張量, shape: [batch_size, seq_len_q, seq_len_k], 用於屏蔽某些位置
            
        Returns:
            (output, attention_weights): 加權后的輸出和注意力權重
        """
        batch_size = query.size(0)
        
        # 線性變換和重塑
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 計算注意力分數
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # 應用掩碼（如果提供）
        if mask is not None:
            # 擴展掩碼維度以適應多頭
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 計算注意力權重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加權求和
        output = torch.matmul(attention_weights, v)
        
        # 重塑並合併多個頭
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 最終線性變換
        output = self.out_linear(output)
        
        return output, attention_weights
    
class AspectAttention(nn.Module):
    """
    面向注意力機制
    專門針對特定面向提取相關特徵
    """
    def __init__(self, hidden_dim: int, aspect_dim: int, attention_dim: int = 64):
        """
        初始化面向注意力機制
        
        Args:
            hidden_dim: 輸入向量維度
            aspect_dim: 面向向量維度
            attention_dim: 注意力層維度
        """
        super(AspectAttention, self).__init__()
        
        # 面向轉換層
        self.aspect_transform = nn.Linear(aspect_dim, attention_dim)
        
        # 輸入轉換層
        self.input_transform = nn.Linear(hidden_dim, attention_dim)
        
        # 注意力分數層
        self.attention_score = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, inputs: torch.Tensor, aspect: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳遞
        
        Args:
            inputs: 輸入張量, shape: [batch_size, seq_len, hidden_dim]
            aspect: 面向張量, shape: [batch_size, aspect_dim]
            mask: 掩碼張量, shape: [batch_size, seq_len], 用於屏蔽填充位置
            
        Returns:
            (weighted_outputs, attention_weights): 加權后的輸出和注意力權重
        """
        # 轉換面向表示
        aspect_vec = self.aspect_transform(aspect)  # [batch_size, attention_dim]
        
        # 擴展面向向量維度
        batch_size, seq_len, _ = inputs.size()
        aspect_vec = aspect_vec.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, attention_dim]
        
        # 轉換輸入表示
        inputs_trans = self.input_transform(inputs)  # [batch_size, seq_len, attention_dim]
        
        # 結合面向和輸入，使用tanh激活
        combined = torch.tanh(inputs_trans + aspect_vec)  # [batch_size, seq_len, attention_dim]
        
        # 計算注意力分數
        attention_scores = self.attention_score(combined).squeeze(-1)  # [batch_size, seq_len]
        
        # 使用掩碼（如果提供）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        # 計算注意力權重
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # 加權求和
        weighted_outputs = torch.sum(attention_weights * inputs, dim=1)  # [batch_size, hidden_dim]
        
        # 去掉最後一個維度
        attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]
        
        return weighted_outputs, attention_weights

class AttentionProcessor:
    """
    注意力處理器類
    提供各種注意力機制的高層接口
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化注意力處理器
        
        Args:
            config: 配置管理器
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 設置日誌
        log_dir = self.config.get('logging.log_dir', './Part03_/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'attention.log')
        
        self.logger = logging.getLogger('attention_processor')
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
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        # 默認使用CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用設備: {self.device}")
        
        # 初始化模型
        self.self_attention_model = None
        self.mha_model = None
        self.aspect_attention_model = None
    
    def apply_self_attention(self, embeddings: np.ndarray, 
                           hidden_dim: int = None,
                           attention_dim: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        應用自注意力機制處理文本嵌入
        
        Args:
            embeddings: 文本嵌入矩陣 [num_samples, seq_len, hidden_dim]
            hidden_dim: 隱藏層維度，默認使用輸入維度
            attention_dim: 注意力層維度，默認與hidden_dim相同
            
        Returns:
            (weighted_outputs, attention_weights): 加權后的輸出和注意力權重
        """
        # 如果未提供維度參數，從輸入推斷
        if hidden_dim is None:
            hidden_dim = embeddings.shape[2]
        
        if attention_dim is None:
            attention_dim = hidden_dim
        
        # 初始化模型
        if self.self_attention_model is None:
            self.self_attention_model = SelfAttention(hidden_dim, attention_dim).to(self.device)
            self.logger.info(f"初始化自注意力模型：hidden_dim={hidden_dim}, attention_dim={attention_dim}")
        
        # 轉換為張量
        inputs = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        
        # 模型設置為評估模式
        self.self_attention_model.eval()
        
        with torch.no_grad():
            weighted_outputs, attention_weights = self.self_attention_model(inputs)
        
        # 轉換回NumPy
        weighted_outputs_np = weighted_outputs.cpu().numpy()
        attention_weights_np = attention_weights.cpu().numpy()
        
        return weighted_outputs_np, attention_weights_np
    
    def apply_multi_head_attention(self, queries: np.ndarray, keys: np.ndarray = None, values: np.ndarray = None,
                                hidden_dim: int = None,
                                num_heads: int = 8,
                                dropout: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        應用多頭注意力機制處理文本嵌入
        
        Args:
            queries: 查詢矩陣 [batch_size, seq_len_q, hidden_dim]
            keys: 鍵矩陣 [batch_size, seq_len_k, hidden_dim]，如果為None則使用queries
            values: 值矩陣 [batch_size, seq_len_v, hidden_dim]，如果為None則使用keys
            hidden_dim: 隱藏層維度，默認使用輸入維度
            num_heads: 注意力頭數量
            dropout: dropout率
            
        Returns:
            (outputs, attention_weights): 輸出和注意力權重
        """
        # 如果未提供keys或values，使用queries
        if keys is None:
            keys = queries
        
        if values is None:
            values = keys
        
        # 如果未提供hidden_dim，從輸入推斷
        if hidden_dim is None:
            hidden_dim = queries.shape[2]
        
        # 初始化模型
        if self.mha_model is None or hidden_dim != self.mha_model.hidden_dim:
            self.mha_model = MultiHeadAttention(hidden_dim, num_heads, dropout).to(self.device)
            self.logger.info(f"初始化多頭注意力模型：hidden_dim={hidden_dim}, num_heads={num_heads}")
        
        # 轉換為張量
        q = torch.tensor(queries, dtype=torch.float32).to(self.device)
        k = torch.tensor(keys, dtype=torch.float32).to(self.device)
        v = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        # 模型設置為評估模式
        self.mha_model.eval()
        
        with torch.no_grad():
            outputs, attention_weights = self.mha_model(q, k, v)
        
        # 轉換回NumPy
        outputs_np = outputs.cpu().numpy()
        attention_weights_np = attention_weights.cpu().numpy()
        
        return outputs_np, attention_weights_np
    
    def apply_aspect_attention(self, embeddings: np.ndarray, aspects: np.ndarray, 
                             hidden_dim: int = None,
                             aspect_dim: int = None,
                             attention_dim: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        應用面向注意力機制處理文本嵌入
        
        Args:
            embeddings: 文本嵌入矩陣 [batch_size, seq_len, hidden_dim]
            aspects: 面向向量 [batch_size, aspect_dim]
            hidden_dim: 隱藏層維度，默認使用輸入維度
            aspect_dim: 面向向量維度，默認使用輸入維度
            attention_dim: 注意力層維度
            
        Returns:
            (weighted_outputs, attention_weights): 加權后的輸出和注意力權重
        """
        # 如果未提供維度參數，從輸入推斷
        if hidden_dim is None:
            hidden_dim = embeddings.shape[2]
        
        if aspect_dim is None:
            aspect_dim = aspects.shape[1]
        
        # 初始化模型
        if (self.aspect_attention_model is None or 
            hidden_dim != self.aspect_attention_model.input_transform.in_features or
            aspect_dim != self.aspect_attention_model.aspect_transform.in_features):
            self.aspect_attention_model = AspectAttention(hidden_dim, aspect_dim, attention_dim).to(self.device)
            self.logger.info(f"初始化面向注意力模型：hidden_dim={hidden_dim}, aspect_dim={aspect_dim}")
        
        # 轉換為張量
        inputs = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        aspect_tensors = torch.tensor(aspects, dtype=torch.float32).to(self.device)
        
        # 模型設置為評估模式
        self.aspect_attention_model.eval()
        
        with torch.no_grad():
            weighted_outputs, attention_weights = self.aspect_attention_model(inputs, aspect_tensors)
        
        # 轉換回NumPy
        weighted_outputs_np = weighted_outputs.cpu().numpy()
        attention_weights_np = attention_weights.cpu().numpy()
        
        return weighted_outputs_np, attention_weights_np
    
    def precompute_context_vectors(self, embeddings_dict: Dict[str, np.ndarray],
                                 method: str = 'self_attention',
                                 aspect_vectors: Dict[int, Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        預計算文本的上下文向量表示
        
        Args:
            embeddings_dict: ID到嵌入向量的字典 {id: [seq_len, hidden_dim]}
            method: 注意力方法，'self_attention'或'multi_head_attention'或'aspect_attention'
            aspect_vectors: 面向向量字典 {aspect_id: {'vector': vector}}，僅在method='aspect_attention'時需要
            
        Returns:
            ID到上下文向量的字典 {id: context_vector}
        """
        if not embeddings_dict:
            return {}
        
        # 獲取第一個樣本提取維度信息
        first_id = next(iter(embeddings_dict))
        first_emb = embeddings_dict[first_id]
        hidden_dim = first_emb.shape[-1]
        
        # 根據不同方法應用注意力機制
        context_vectors = {}
        
        if method == 'self_attention':
            self.logger.info(f"使用自注意力機制處理 {len(embeddings_dict)} 個樣本...")
            
            # 逐個處理樣本
            for idx, (id_val, embedding) in enumerate(embeddings_dict.items()):
                # 擴展batch維度
                emb_batch = np.expand_dims(embedding, axis=0)
                
                # 應用注意力
                weighted_output, _ = self.apply_self_attention(emb_batch, hidden_dim)
                
                # 存儲結果
                context_vectors[id_val] = weighted_output[0]  # 去除batch維度
                
                # 顯示進度
                if (idx + 1) % 100 == 0 or (idx + 1) == len(embeddings_dict):
                    self.logger.info(f"已處理 {idx + 1}/{len(embeddings_dict)} 個樣本")
            
        elif method == 'multi_head_attention':
            self.logger.info(f"使用多頭注意力機制處理 {len(embeddings_dict)} 個樣本...")
            
            # 逐個處理樣本
            for idx, (id_val, embedding) in enumerate(embeddings_dict.items()):
                # 擴展batch維度
                emb_batch = np.expand_dims(embedding, axis=0)
                
                # 自注意力形式：查詢=鍵=值
                output, _ = self.apply_multi_head_attention(emb_batch, hidden_dim=hidden_dim)
                
                # 通過平均或最大池化獲取上下文向量
                context_vector = np.mean(output[0], axis=0)
                
                # 存儲結果
                context_vectors[id_val] = context_vector
                
                # 顯示進度
                if (idx + 1) % 100 == 0 or (idx + 1) == len(embeddings_dict):
                    self.logger.info(f"已處理 {idx + 1}/{len(embeddings_dict)} 個樣本")
            
        elif method == 'aspect_attention':
            if not aspect_vectors:
                raise ValueError("使用面向注意力時必須提供面向向量")
                
            self.logger.info(f"使用面向注意力機制處理 {len(embeddings_dict)} 個樣本...")
            
            # 獲取所有面向ID
            aspect_ids = list(aspect_vectors.keys())
            
            # 對於每個樣本，計算與每個面向的加權表示
            for idx, (id_val, embedding) in enumerate(embeddings_dict.items()):
                # 擴展batch維度
                emb_batch = np.expand_dims(embedding, axis=0)
                
                # 初始化該樣本的所有面向表示
                aspect_context_vectors = []
                
                # 對每個面向計算注意力加權
                for aspect_id in aspect_ids:
                    aspect_vector = aspect_vectors[aspect_id]['vector']
                    # 擴展batch維度
                    aspect_batch = np.expand_dims(aspect_vector, axis=0)
                    
                    # 應用面向注意力
                    weighted_output, _ = self.apply_aspect_attention(emb_batch, aspect_batch)
                    
                    # 添加到列表
                    aspect_context_vectors.append(weighted_output[0])
                
                # 連接所有面向表示
                combined_vector = np.concatenate(aspect_context_vectors)
                
                # 存儲結果
                context_vectors[id_val] = combined_vector
                
                # 顯示進度
                if (idx + 1) % 100 == 0 or (idx + 1) == len(embeddings_dict):
                    self.logger.info(f"已處理 {idx + 1}/{len(embeddings_dict)} 個樣本")
        
        else:
            raise ValueError(f"不支持的注意力方法: {method}")
        
        self.logger.info(f"注意力處理完成，生成了 {len(context_vectors)} 個上下文向量")
        
        return context_vectors

# 使用示例
if __name__ == "__main__":
    # 初始化注意力處理器
    attention_processor = AttentionProcessor()
    
    # 模擬一些樣本數據
    batch_size = 3
    seq_len = 5
    hidden_dim = 10
    
    # 創建隨機嵌入矩陣
    embeddings = np.random.random((batch_size, seq_len, hidden_dim))
    
    # 應用自注意力機制
    print("測試自注意力機制...")
    weighted_outputs, attention_weights = attention_processor.apply_self_attention(embeddings)
    
    print(f"輸入形狀: {embeddings.shape}")
    print(f"輸出形狀: {weighted_outputs.shape}")
    print(f"注意力權重形狀: {attention_weights.shape}")
    
    # 創建一個字典模擬真實場景
    embeddings_dict = {
        "doc1": np.random.random((seq_len, hidden_dim)),
        "doc2": np.random.random((seq_len, hidden_dim)),
        "doc3": np.random.random((seq_len, hidden_dim))
    }
    
    # 預計算上下文向量
    context_vectors = attention_processor.precompute_context_vectors(embeddings_dict)
    
    # 顯示結果
    for doc_id, vector in context_vectors.items():
        print(f"{doc_id} 的上下文向量形狀: {vector.shape}")