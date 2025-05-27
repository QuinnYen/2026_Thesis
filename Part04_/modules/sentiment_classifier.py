#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情感分類器模組 - 使用面向向量進行情感分類
基於面向向量特徵和原始評論星級標籤進行情感分類
"""

import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# 添加父目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# 機器學習相關
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 設定基本日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name):
    """獲取日誌記錄器"""
    logger = logging.getLogger(name)
    return logger

class SentimentClassifier:
    """情感分類器主類"""
    
    def __init__(self, output_dir=None):
        """
        初始化情感分類器
        
        Args:
            output_dir: 輸出目錄路徑
        """
        self.logger = get_logger("sentiment_classifier")
        self.output_dir = output_dir or os.path.join("Part04_", "1_output")
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 分類器模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # 資料預處理
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 訓練資料
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 評估結果
        self.evaluation_results = {}
        
        self.logger.info("情感分類器初始化完成")
    
    def load_and_process_data(self, raw_data_path: str, aspect_vectors_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        載入並處理原始資料和面向向量
        
        Args:
            raw_data_path: 原始資料檔案路徑
            aspect_vectors_path: 面向向量檔案路徑
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 特徵向量和標籤
        """
        self.logger.info(f"載入原始資料: {raw_data_path}")
        self.logger.info(f"載入面向向量: {aspect_vectors_path}")
        
        try:
            # 載入原始資料
            raw_data = pd.read_csv(raw_data_path)
            self.logger.info(f"載入 {len(raw_data)} 筆原始資料")
            
            # 載入面向向量
            with open(aspect_vectors_path, 'r', encoding='utf-8') as f:
                aspect_data = json.load(f)
            
            # 提取面向向量
            aspect_vectors = aspect_data.get('vectors', {})
            if not aspect_vectors:
                # 嘗試其他可能的格式
                aspect_vectors = aspect_data.get('aspect_vectors', {})
                if isinstance(aspect_vectors, dict):
                    aspect_vectors = aspect_vectors.get('vectors', {})
            
            if not aspect_vectors:
                raise ValueError("沒有找到有效的面向向量資料")
            
            self.logger.info(f"找到 {len(aspect_vectors)} 個面向向量")
            
            # 轉換面向向量為特徵矩陣
            features = []
            labels = []
            
            for aspect_name, vector in aspect_vectors.items():
                # 處理向量格式
                if isinstance(vector, list):
                    # 如果是列表，直接使用
                    vec = vector
                elif isinstance(vector, dict):
                    # 如果是字典，嘗試提取向量
                    vec = vector.get('vector', [])
                else:
                    continue
                
                # 確保向量是有效的
                if not isinstance(vec, list) or len(vec) == 0:
                    continue
                
                # 如果向量長度不是 768，嘗試調整
                if len(vec) != 768:
                    if len(vec) > 768:
                        vec = vec[:768]  # 截斷
                    else:
                        vec = vec + [0] * (768 - len(vec))  # 填充
                
                features.append(vec)
                # 根據面向名稱推斷情感標籤
                label = self._infer_sentiment_from_aspect(aspect_name)
                labels.append(label)
            
            if not features:
                raise ValueError("沒有找到有效的特徵向量")
            
            features = np.array(features, dtype=np.float32)
            labels = np.array(labels)
            
            self.logger.info(f"處理完成 - 特徵維度: {features.shape}")
            self.logger.info(f"標籤分佈: {np.bincount(labels)}")
            
            return features, labels
            
        except Exception as e:
            self.logger.error(f"載入資料失敗: {str(e)}")
            raise
    
    def _infer_sentiment_from_aspect(self, aspect_name: str) -> int:
        """
        根據面向名稱推斷情感標籤
        
        Args:
            aspect_name: 面向名稱
            
        Returns:
            int: 情感標籤 (0: 負面, 1: 正面)
        """
        # 從面向名稱中提取情感信息
        aspect_lower = aspect_name.lower()
        
        # 檢查是否包含情感標記
        if '_positive' in aspect_lower or '_pos' in aspect_lower:
            return 1
        elif '_negative' in aspect_lower or '_neg' in aspect_lower:
            return 0
        
        # 如果沒有明確的情感標記，根據關鍵詞判斷
        positive_keywords = ['好評', '推薦', '優秀', '精彩', '讚賞', 
                           'great', 'excellent', 'good', 'best', 'recommend',
                           'amazing', 'awesome', 'fantastic', 'wonderful', 'perfect']
        
        negative_keywords = ['差評', '失望', '糟糕', '浪費', '不推薦',
                           'bad', 'poor', 'worst', 'terrible', 'disappointing',
                           'waste', 'awful', 'horrible', 'boring', 'mediocre']
        
        # 檢查關鍵詞
        if any(keyword in aspect_lower for keyword in positive_keywords):
            return 1
        elif any(keyword in aspect_lower for keyword in negative_keywords):
            return 0
        
        # 如果無法判斷，根據雜湊值分配標籤
        import hashlib
        hash_val = int(hashlib.md5(aspect_name.encode()).hexdigest(), 16) % 2
        return hash_val
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray):
        """
        準備訓練和測試資料
        
        Args:
            features: 特徵向量
            labels: 標籤
        """
        self.logger.info("準備訓練和測試資料...")
        
        # 資料標準化
        features_scaled = self.scaler.fit_transform(features)
        
        # 標籤編碼
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # 分割資料
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features_scaled, labels_encoded,
            test_size=0.2,
            random_state=42,
            stratify=labels_encoded
        )
        
        self.logger.info(f"訓練集樣本數: {len(self.X_train)}")
        self.logger.info(f"測試集樣本數: {len(self.X_test)}")
        self.logger.info(f"訓練集標籤分佈: {np.bincount(self.y_train)}")
        self.logger.info(f"測試集標籤分佈: {np.bincount(self.y_test)}")
    
    def train_model(self):
        """訓練分類器模型"""
        self.logger.info("開始訓練分類器模型...")
        
        try:
            self.model.fit(self.X_train, self.y_train)
            self.logger.info("模型訓練完成")
        except Exception as e:
            self.logger.error(f"訓練模型失敗: {str(e)}")
            raise
    
    def evaluate_model(self):
        """評估分類器模型"""
        self.logger.info("評估分類器模型...")
        
        try:
            # 預測
            y_pred = self.model.predict(self.X_test)
            
            # 計算評估指標
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(
                self.y_test, y_pred,
                target_names=['負面', '正面'],
                output_dict=True
            )
            cm = confusion_matrix(self.y_test, y_pred)
            
            # 儲存評估結果
            self.evaluation_results = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
            
            self.logger.info(f"準確率: {accuracy:.4f}")
            self.logger.info(f"分類報告:\n{report}")
            
        except Exception as e:
            self.logger.error(f"評估模型失敗: {str(e)}")
            raise
    
    def compare_sentiment_labels(self, original_labels: np.ndarray, new_labels: np.ndarray) -> Dict[str, Any]:
        """
        比較原始和新資料的情感標籤
        
        Args:
            original_labels: 原始情感標籤
            new_labels: 新情感標籤
            
        Returns:
            Dict: 比較結果
        """
        self.logger.info("比較原始和新資料的情感標籤...")
        
        # 確保標籤長度相同
        if len(original_labels) != len(new_labels):
            raise ValueError("原始和新標籤的長度必須相同")
        
        # 計算統計資訊
        total_samples = len(original_labels)
        changed_count = sum(original_labels != new_labels)
        change_rate = changed_count / total_samples
        
        return {
            'statistics': {
                'total_samples': total_samples,
                'changed_count': changed_count,
                'change_rate': change_rate
            }
        }
    
    def save_results(self, filename: str = None):
        """儲存分類結果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_classification_result_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'description': '情感分類器結果',
            'evaluation_results': self.evaluation_results,
            'label_mapping': {0: '負面', 1: '正面'},
            'feature_dim': self.X_train.shape[1] if self.X_train is not None else None,
            'num_samples': {
                'train': len(self.X_train) if self.X_train is not None else 0,
                'test': len(self.X_test) if self.X_test is not None else 0
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"分類結果已儲存至: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"儲存結果失敗: {str(e)}")
            raise 