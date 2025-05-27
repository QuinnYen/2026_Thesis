import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

class AttentionAnalyzer:
    def __init__(self, bert_attention_path, topic_labels_path):
        """
        初始化注意力分析器
        
        Args:
            bert_attention_path (str): BERT注意力權重的.npy檔案路徑
            topic_labels_path (str): 主題標籤的json檔案路徑
        """
        self.attention_weights = np.load(bert_attention_path)
        
        # 載入主題標籤
        with open(topic_labels_path, 'r', encoding='utf-8') as f:
            self.topic_labels = json.load(f)
        
        # 初始化注意力層
        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
        
    def process_attention_weights(self, layer_idx=-1):
        """
        處理BERT的注意力權重
        
        Args:
            layer_idx (int): 要使用的BERT層索引，預設使用最後一層
        
        Returns:
            torch.Tensor: 處理後的注意力權重
        """
        # 獲取指定層的注意力權重
        layer_weights = torch.from_numpy(self.attention_weights[layer_idx])
        return layer_weights
    
    def apply_attention_mechanism(self, features, attention_weights):
        """
        應用注意力機制
        
        Args:
            features (torch.Tensor): 輸入特徵
            attention_weights (torch.Tensor): 注意力權重
            
        Returns:
            torch.Tensor: 加權後的特徵
        """
        # 應用注意力權重
        attended_features = torch.matmul(attention_weights, features)
        return attended_features
    
    def analyze_topics(self, text_features, dataset_type="IMDB", language="zh"):
        """
        分析文本的主題分布
        
        Args:
            text_features (torch.Tensor): 文本特徵
            dataset_type (str): 資料集類型 (IMDB/AMAZON/YELP)
            language (str): 語言選擇 (zh/en)
            
        Returns:
            dict: 主題分布結果
        """
        # 獲取注意力權重
        attention_weights = self.process_attention_weights()
        
        # 應用注意力機制
        attended_features = self.apply_attention_mechanism(text_features, attention_weights)
        
        # 計算主題分數
        topic_scores = F.softmax(torch.mean(attended_features, dim=1), dim=-1)
        
        # 將分數對應到主題標籤
        results = {}
        for i in range(10):
            topic_label = self.topic_labels[dataset_type][language][str(i)]
            results[topic_label] = float(topic_scores[0][i])
            
        return results
    
    def get_topic_attention_weights(self, text_features):
        """
        獲取每個主題的注意力權重
        
        Args:
            text_features (torch.Tensor): 文本特徵
            
        Returns:
            torch.Tensor: 主題的注意力權重
        """
        attention_weights = self.process_attention_weights()
        topic_attention = torch.mean(attention_weights, dim=1)
        return topic_attention 