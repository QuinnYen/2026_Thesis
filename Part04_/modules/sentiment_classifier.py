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
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# 深度學習相關
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 設定基本日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_logger(name):
    """獲取日誌記錄器"""
    logger = logging.getLogger(name)
    return logger

class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment classification"""
    
    def __init__(self, features, labels):
        """
        初始化資料集
        
        Args:
            features: 面向向量特徵 (n_samples, 768)
            labels: 情感標籤 (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SentimentMLP(nn.Module):
    """多層感知器用於情感分類"""
    
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], num_classes=3, dropout=0.3):
        """
        初始化MLP模型
        
        Args:
            input_dim: 輸入維度 (面向向量維度，預設768)
            hidden_dims: 隱藏層維度列表
            num_classes: 分類數目 (預設3：正面/負面/中性)
            dropout: Dropout比率
        """
        super(SentimentMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 構建隱藏層
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 輸出層
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class SentimentClassifier:
    """情感分類器主類"""
    
    def __init__(self, config=None, output_dir=None):
        """
        初始化情感分類器
        
        Args:
            config: 配置對象（可選）
            output_dir: 輸出目錄路徑
        """
        self.logger = get_logger("sentiment_classifier")
        self.output_dir = output_dir or os.path.join("Part04_", "1_output")
        
        # 默認配置
        DEFAULT_CONFIG = {
            "label_mapping": {
                "positive": 1,
                "negative": 0
            },
            "data_settings": {
                "IMDB": {
                    "test_size": 0.2,
                    "validation_size": 0.1,
                    "random_state": 42,
                    "min_samples_per_class": 10,
                    "label_column": "sentiment"
                },
                "AMAZON": {
                    "test_size": 0.2,
                    "validation_size": 0.1,
                    "random_state": 42,
                    "min_samples_per_class": 10,
                    "label_column": "overall",
                    "rating_threshold": {
                        "negative": 3,
                        "positive": 4
                    }
                },
                "YELP": {
                    "test_size": 0.2,
                    "validation_size": 0.1,
                    "random_state": 42,
                    "min_samples_per_class": 10,
                    "label_column": "stars",
                    "rating_threshold": {
                        "negative": 3,
                        "positive": 4
                    }
                }
            }
        }
        
        # 使用提供的配置或默認配置
        self.config = config if isinstance(config, dict) else DEFAULT_CONFIG
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 分類器模型
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 訓練資料
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 評估結果
        self.evaluation_results = {}
        self.cross_validation_results = {}
        
        self.logger.info("情感分類器初始化完成")
    
    def load_aspect_vectors_data(self, aspect_vectors_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        載入面向向量資料
        
        Args:
            aspect_vectors_file: 面向向量結果檔案路徑
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 特徵向量和標籤
        """
        self.logger.info(f"載入面向向量資料: {aspect_vectors_file}")
        
        try:
            with open(aspect_vectors_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取最佳注意力機制的面向向量
            best_attention = data.get('best_attention', {})
            
            # 檢查機制格式
            if isinstance(best_attention.get('mechanism'), dict):
                best_mechanism = best_attention.get('mechanism', {}).get('type', 'similarity')
            else:
                best_mechanism = best_attention.get('mechanism', 'similarity')
            
            self.logger.info(f"使用最佳注意力機制: {best_mechanism}")
            
            # 檢查是否有直接的面向向量
            if 'aspect_vectors' in data:
                # 使用新格式的面向向量
                aspect_vectors_data = data.get('aspect_vectors', {}).get('vectors', {})
            else:
                # 使用舊格式的面向向量 - 從attention_mechanisms中獲取
                attention_data = data.get('attention_mechanisms', {}).get('mechanisms', {}).get(best_mechanism, {})
                aspect_vectors_data = attention_data.get('aspect_vectors', {})
            
            features = []
            labels = []
            self.aspects = []  # 保存面向名稱
            
            for aspect_name, vector in aspect_vectors_data.items():
                if isinstance(vector, list) and len(vector) == 768:
                    features.append(vector)
                    self.aspects.append(aspect_name)
                    
                    # 根據面向名稱推斷情感標籤
                    label = self._infer_sentiment_from_aspect(aspect_name)
                    labels.append(label)
            
            if not features:
                raise ValueError("沒有找到有效的面向向量資料")
            
            features = np.array(features, dtype=np.float32)
            labels = np.array(labels)
            
            self.logger.info(f"載入完成 - 樣本數: {len(features)}, 特徵維度: {features.shape[1]}")
            self.logger.info(f"標籤分佈: {np.bincount(labels)}")
            
            return features, labels
            
        except Exception as e:
            self.logger.error(f"載入面向向量資料失敗: {str(e)}")
            raise
    
    def _infer_sentiment_from_aspect(self, aspect_name: str) -> int:
        """
        根據面向名稱推斷情感標籤 (暫時實現)
        實際應該從原始評論的星級評分中獲取
        
        Args:
            aspect_name: 面向名稱
            
        Returns:
            int: 情感標籤 (0: 負面, 1: 中性, 2: 正面)
        """
        # 這是暫時的實現，實際應該根據原始資料的星級評分
        positive_keywords = ['視覺效果', '動作場面', '特效', 'visual', 'effect', 'action']
        negative_keywords = ['劇情', '故事', 'plot', 'story', '失望']
        neutral_keywords = ['演員', '音樂', 'actor', 'music', 'cast']
        
        aspect_lower = aspect_name.lower()
        
        # 為了測試，創建更均勻的標籤分佈
        import hashlib
        hash_val = int(hashlib.md5(aspect_name.encode()).hexdigest(), 16) % 3
        
        if any(keyword in aspect_lower for keyword in positive_keywords):
            return 2  # 正面
        elif any(keyword in aspect_lower for keyword in negative_keywords):
            return 0  # 負面
        elif any(keyword in aspect_lower for keyword in neutral_keywords):
            return 1  # 中性
        else:
            # 根據雜湊值分配標籤，確保有不同類別
            return hash_val
    
    def load_labeled_data(self, data_file: str, source: str = 'IMDB') -> Tuple[np.ndarray, np.ndarray]:
        """
        載入帶標籤的資料
        
        Args:
            data_file: 原始資料檔案路徑
            source: 數據來源 ('IMDB', 'AMAZON', 或 'YELP')
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 特徵向量和真實標籤
        """
        self.logger.info(f"載入帶標籤資料: {data_file}")
        
        try:
            # 獲取數據源特定的設定
            if source not in self.config['data_settings']:
                raise ValueError(f"未知的數據來源: {source}")
            
            source_settings = self.config['data_settings'][source]
            label_column = source_settings['label_column']
            
            # 根據檔案格式載入資料
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
            elif data_file.endswith('.json'):
                df = pd.read_json(data_file)
            else:
                raise ValueError(f"不支援的檔案格式: {data_file}")
            
            # 檢查必要欄位
            if label_column not in df.columns:
                raise ValueError(f"找不到標籤欄位: {label_column}")
            
            # 保存評論文字
            text_column = {
                'IMDB': 'review',
                'AMAZON': 'reviewText',
                'YELP': 'text'
            }.get(source)
            
            if text_column and text_column in df.columns:
                self.test_texts = df[text_column].tolist()
            else:
                self.test_texts = None
                self.logger.warning(f"找不到評論文字欄位: {text_column}")
            
            # 保存原始標籤（用於比較）
            self.original_labels_raw = df[label_column].tolist()
            
            # 使用配置中的標籤映射
            label_mapping = self.config.get('label_mapping', {
                'positive': 1,
                'negative': 0
            })
            
            if source in ['AMAZON', 'YELP']:
                # 對於 AMAZON 和 YELP，需要將評分轉換為情感標籤
                rating_threshold = source_settings.get('rating_threshold', {
                    'negative': 3,  # 1-3星為負面
                    'positive': 4   # 4-5星為正面
                })
                labels = self._convert_rating_to_sentiment(
                    df[label_column].values,
                    rating_threshold
                )
                self.logger.info(f"評分分佈: {np.bincount(df[label_column].values.astype(int), minlength=6)[1:]}")
            else:
                # 對於 IMDB，直接使用情感標籤
                labels = df[label_column].map(label_mapping).values
            
            # 檢查是否有無效的標籤
            if np.any(pd.isna(labels)):
                invalid_labels = df[pd.isna(labels)][label_column].unique()
                raise ValueError(f"發現無效的標籤值: {invalid_labels}")
            
            self.logger.info(f"載入完成 - 樣本數: {len(labels)}")
            self.logger.info(f"情感標籤分佈: {np.bincount(labels.astype(int))}")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"載入帶標籤資料失敗: {str(e)}")
            raise
    
    def _convert_rating_to_sentiment(self, ratings: np.ndarray, threshold: Dict[str, int]) -> np.ndarray:
        """
        將評分轉換為情感標籤
        
        Args:
            ratings: 評分陣列
            threshold: 評分閾值設定，包含 'negative' 和 'positive' 的閾值
            
        Returns:
            np.ndarray: 情感標籤 (0: 負面, 1: 正面)
        """
        labels = np.zeros(len(ratings), dtype=int)
        
        # 轉換邏輯：小於等於 negative_threshold 為負面，大於等於 positive_threshold 為正面
        negative_threshold = threshold.get('negative', 3)
        positive_threshold = threshold.get('positive', 4)
        
        # 1-3星為負面 (0)，4-5星為正面 (1)
        labels[ratings <= negative_threshold] = 0  # 負面
        labels[ratings >= positive_threshold] = 1  # 正面
        
        return labels
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray, source: str = 'IMDB'):
        """
        準備訓練和測試資料
        
        Args:
            features: 特徵向量
            labels: 標籤
            source: 數據來源 ('IMDB', 'AMAZON', 或 'YELP')
        """
        self.logger.info("準備訓練和測試資料...")
        
        # 從配置中獲取資料設定
        if source not in self.config['data_settings']:
            raise ValueError(f"未知的數據來源: {source}")
            
        data_settings = self.config['data_settings'][source]
        
        # 檢查每個類別的樣本數
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        min_samples = data_settings['min_samples_per_class']
        
        for label, count in zip(unique_labels, label_counts):
            if count < min_samples:
                self.logger.warning(f"類別 {label} 的樣本數 ({count}) 少於最小要求 ({min_samples})")
        
        # 資料標準化
        features_scaled = self.scaler.fit_transform(features)
        
        # 標籤編碼
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # 計算最小的類別樣本數
        min_class_samples = min(label_counts)
        
        # 根據數據集大小調整分割策略
        n_classes = len(unique_labels)
        if len(features) < n_classes * 3:  # 如果樣本總數小於類別數的3倍
            # 不進行分割，全部用於訓練
            self.logger.warning("樣本數太少，將全部用於訓練")
            self.X_train = features_scaled
            self.y_train = labels_encoded
            self.X_test = features_scaled
            self.y_test = labels_encoded
            
            # 保存對應的原始標籤和文字
            self.test_indices = np.arange(len(labels))
            
        else:
            # 確保測試集至少包含每個類別的一個樣本
            min_test_size = max(n_classes, int(len(features) * data_settings['test_size']))
            if min_test_size >= len(features):
                min_test_size = n_classes
            
            # 分割資料（同時分割原始標籤和文字的索引）
            self.X_train, self.X_test, self.y_train, self.y_test, train_indices, test_indices = train_test_split(
                features_scaled, labels_encoded, np.arange(len(labels)),
                test_size=min_test_size,
                random_state=data_settings['random_state'],
                stratify=labels_encoded if len(features) >= n_classes * 2 else None
            )
            
            # 保存測試集的原始索引
            self.test_indices = test_indices
        
        self.logger.info(f"訓練集樣本數: {len(self.X_train)}")
        self.logger.info(f"測試集樣本數: {len(self.X_test)}")
        self.logger.info(f"訓練集標籤分佈: {np.bincount(self.y_train)}")
        self.logger.info(f"測試集標籤分佈: {np.bincount(self.y_test)}")
    
    def initialize_models(self):
        """初始化各種分類器模型"""
        self.logger.info("初始化分類器模型...")
        
        # 計算類別權重以處理不平衡資料
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(self.y_train), 
            y=self.y_train
        )
        class_weight_dict = dict(zip(np.unique(self.y_train), class_weights))
        
        # 隨機森林
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # 支援向量機
        self.models['SVM'] = SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        # 邏輯回歸
        self.models['LogisticRegression'] = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        # 梯度提升
        self.models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # 多層感知器
        self.models['MLP'] = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        self.logger.info(f"已初始化 {len(self.models)} 個分類器模型")
    
    def train_models(self):
        """訓練所有分類器模型"""
        self.logger.info("開始訓練分類器模型...")
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"訓練 {model_name} 模型...")
                model.fit(self.X_train, self.y_train)
                self.logger.info(f"{model_name} 模型訓練完成")
            except Exception as e:
                self.logger.error(f"訓練 {model_name} 模型失敗: {str(e)}")
    
    def evaluate_models(self):
        """評估所有分類器模型"""
        self.logger.info("評估分類器模型...")
        
        best_score = 0
        
        for model_name, model in self.models.items():
            try:
                # 預測
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
                
                # 計算評估指標
                accuracy = accuracy_score(self.y_test, y_pred)
                
                # 處理只有一個類別的情況
                if len(np.unique(self.y_test)) == 1:
                    self.logger.warning(f"{model_name}: 測試集只包含一個類別，無法計算完整的評估指標")
                    precision = accuracy
                    recall = accuracy
                    f1 = accuracy
                else:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        self.y_test, y_pred, average='weighted'
                    )
                
                # AUC-ROC (只在二分類且有概率預測時計算)
                if y_pred_proba is not None and len(np.unique(self.y_test)) == 2:
                    try:
                        auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                    except:
                        auc_score = None
                else:
                    auc_score = None
                
                # 混淆矩陣
                cm = confusion_matrix(self.y_test, y_pred)
                
                # 分類報告
                try:
                    class_report = classification_report(
                        self.y_test, y_pred,
                        target_names=['負面', '正面'] if len(np.unique(self.y_test)) == 2 else ['負面', '中性', '正面'],
                        output_dict=True
                    )
                except:
                    class_report = {
                        'accuracy': accuracy,
                        'weighted avg': {
                            'precision': precision,
                            'recall': recall,
                            'f1-score': f1
                        }
                    }
                
                # 儲存評估結果
                self.evaluation_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_score': auc_score,
                    'confusion_matrix': cm.tolist(),
                    'classification_report': class_report,
                    'predictions': y_pred.tolist(),
                    'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None,
                    'is_small_dataset': len(self.X_train) < len(np.unique(self.y_train)) * 3
                }
                
                # 更新最佳模型
                if f1 > best_score:
                    best_score = f1
                    self.best_model = model
                    self.best_model_name = model_name
                
                self.logger.info(f"{model_name} - 準確率: {accuracy:.4f}, F1分數: {f1:.4f}")
                
            except Exception as e:
                self.logger.error(f"評估 {model_name} 模型失敗: {str(e)}")
        
        if self.best_model is not None:
            self.logger.info(f"最佳模型: {self.best_model_name} (F1分數: {best_score:.4f})")
        else:
            self.logger.warning("沒有找到最佳模型")
    
    def cross_validation(self, cv_folds: int = 5):
        """執行交叉驗證"""
        self.logger.info(f"執行 {cv_folds} 折交叉驗證...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for model_name, model in self.models.items():
            try:
                # 執行交叉驗證
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train, 
                    cv=cv, scoring='f1_weighted', n_jobs=-1
                )
                
                self.cross_validation_results[model_name] = {
                    'cv_scores': cv_scores.tolist(),
                    'mean_score': cv_scores.mean(),
                    'std_score': cv_scores.std()
                }
                
                self.logger.info(
                    f"{model_name} 交叉驗證 - "
                    f"平均F1分數: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"{model_name} 交叉驗證失敗: {str(e)}")
    
    def hyperparameter_tuning(self, model_name: str = None):
        """超參數調優"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            self.logger.error(f"找不到模型: {model_name}")
            return
        
        self.logger.info(f"為 {model_name} 執行超參數調優...")
        
        # 定義參數網格
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if model_name in param_grids:
            try:
                # 網格搜索
                grid_search = GridSearchCV(
                    self.models[model_name],
                    param_grids[model_name],
                    cv=3,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(self.X_train, self.y_train)
                
                # 更新模型
                self.models[model_name] = grid_search.best_estimator_
                
                self.logger.info(f"最佳參數: {grid_search.best_params_}")
                self.logger.info(f"最佳分數: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                self.logger.error(f"超參數調優失敗: {str(e)}")
    
    def predict_sentiment(self, features: np.ndarray, model_name: str = None) -> Dict[str, Any]:
        """
        預測情感標籤
        
        Args:
            features: 輸入特徵
            model_name: 指定使用的模型名稱，為None時使用最佳模型
            
        Returns:
            Dict: 預測結果
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"找不到模型: {model_name}")
        
        # 標準化特徵
        features_scaled = self.scaler.transform(features)
        
        # 預測
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled) if hasattr(model, 'predict_proba') else None
        
        # 轉換回原始標籤
        sentiment_labels = self.label_encoder.inverse_transform(predictions)
        
        return {
            'model_name': model_name,
            'predictions': predictions.tolist(),
            'sentiment_labels': sentiment_labels.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'label_mapping': {0: '負面', 1: '正面'}
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
            'best_model': self.best_model_name,
            'evaluation_results': self.evaluation_results,
            'cross_validation_results': self.cross_validation_results,
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
    
    def generate_visualizations(self, save_dir: str = None):
        """生成視覺化圖表"""
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 模型性能比較
        self._plot_model_comparison(save_dir)
        
        # 混淆矩陣
        self._plot_confusion_matrices(save_dir)
        
        # 交叉驗證結果
        self._plot_cross_validation_results(save_dir)
        
        self.logger.info(f"視覺化圖表已儲存至: {save_dir}")
    
    def _plot_model_comparison(self, save_dir: str):
        """繪製模型性能比較圖"""
        if not self.evaluation_results:
            return
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        models = list(self.evaluation_results.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.evaluation_results[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('模型')
        ax.set_ylabel('分數')
        ax.set_title('模型性能比較')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, save_dir: str):
        """繪製混淆矩陣"""
        if not self.evaluation_results:
            return
        
        n_models = len(self.evaluation_results)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 8))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        labels = ['負面', '正面']
        
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            if i >= len(axes):
                break
                
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=axes[i]
            )
            axes[i].set_title(f'{model_name} 混淆矩陣')
            axes[i].set_xlabel('預測標籤')
            axes[i].set_ylabel('真實標籤')
        
        # 隱藏多餘的子圖
        for i in range(len(self.evaluation_results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_validation_results(self, save_dir: str):
        """繪製交叉驗證結果"""
        if not self.cross_validation_results:
            return
        
        models = list(self.cross_validation_results.keys())
        means = [self.cross_validation_results[model]['mean_score'] for model in models]
        stds = [self.cross_validation_results[model]['std_score'] for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        
        ax.set_xlabel('模型')
        ax.set_ylabel('F1分數')
        ax.set_title('交叉驗證結果')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加數值標籤
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                   ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cross_validation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def compare_sentiment_labels(self, original_labels: np.ndarray, new_labels: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """
        比較原始和新資料的情感標籤
        
        Args:
            original_labels: 原始情感標籤（編碼後的測試集標籤）
            new_labels: 新情感標籤（模型預測標籤）
            max_samples: 最大顯示樣本數
            
        Returns:
            Dict: 比較結果
        """
        self.logger.info("比較原始和新資料的情感標籤...")
        
        # 確保標籤長度相同
        if len(original_labels) != len(new_labels):
            raise ValueError("原始和新標籤的長度必須相同")
        
        # 確保標籤是二分類（0和1）
        original_labels = np.array([1 if x > 0 else 0 for x in original_labels])
        new_labels = np.array([1 if x > 0 else 0 for x in new_labels])
        
        # 選擇要顯示的樣本索引
        total_test_samples = len(original_labels)
        display_count = min(max_samples, total_test_samples)
        
        # 隨機選擇樣本
        np.random.seed(42)  # 固定隨機種子以確保結果可復現
        sample_indices = np.random.choice(total_test_samples, display_count, replace=False)
        
        # 準備比較結果
        comparison_results = []
        for sample_idx in sample_indices:
            original_label = original_labels[sample_idx]
            new_label = new_labels[sample_idx]
            
            # 轉換標籤為文字
            original_sentiment = "positive" if original_label == 1 else "negative"
            new_sentiment = "positive" if new_label == 1 else "negative"
            
            # 獲取真實的原始資料索引
            if hasattr(self, 'test_indices') and self.test_indices is not None:
                real_data_idx = self.test_indices[sample_idx]
            else:
                real_data_idx = sample_idx
            
            # 獲取評論文字（使用真實的資料索引）
            text = ""
            if hasattr(self, 'test_texts') and self.test_texts is not None and real_data_idx < len(self.test_texts):
                text = self.test_texts[real_data_idx]
                # 限制文字長度為100字
                if len(text) > 100:
                    text = text[:97] + "..."
            
            # 獲取真實的原始標籤（從原始檔案）
            real_original_sentiment = original_sentiment  # 預設使用編碼後的標籤
            if hasattr(self, 'original_labels_raw') and self.original_labels_raw is not None and real_data_idx < len(self.original_labels_raw):
                raw_label = self.original_labels_raw[real_data_idx]
                if isinstance(raw_label, str):
                    real_original_sentiment = raw_label
                else:
                    # 對於數字評分，轉換為情感標籤
                    real_original_sentiment = "positive" if raw_label >= 4 else "negative"
            
            comparison_results.append({
                'index': int(real_data_idx),
                'original_sentiment': real_original_sentiment,
                'new_sentiment': new_sentiment,
                'text': text,
                'is_changed': bool(real_original_sentiment != new_sentiment)
            })
        
        # 計算統計資訊（基於全部測試集）
        total_samples = int(len(original_labels))
        changed_count = int(sum(1 for r in comparison_results if r['is_changed']))
        
        # 計算實際變化率（基於顯示的樣本）
        change_rate = float(changed_count / len(comparison_results) if len(comparison_results) > 0 else 0)
        
        return {
            'comparison_results': comparison_results,
            'statistics': {
                'total_samples': total_samples,
                'changed_count': changed_count,
                'change_rate': change_rate,
                'displayed_samples': len(comparison_results),
                'label_mapping': {0: 'negative', 1: 'positive'}
            }
        }

    def visualize_comparison(self, comparison_results: Dict[str, Any]) -> None:
        """
        視覺化比較結果
        
        Args:
            comparison_results: 比較結果字典
        """
        if not comparison_results:
            return
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 標籤分佈比較
        results = comparison_results['comparison_results']
        
        # 計算標籤分佈
        original_dist = {'positive': 0, 'negative': 0}
        new_dist = {'positive': 0, 'negative': 0}
        
        for result in results:
            original_dist[result['original_sentiment']] += 1
            new_dist[result['new_sentiment']] += 1
        
        # 繪製分佈比較圖
        x = np.arange(2)
        width = 0.35
        
        ax1.bar(x - width/2, [original_dist['negative'], original_dist['positive']], width, label='原始標籤')
        ax1.bar(x + width/2, [new_dist['negative'], new_dist['positive']], width, label='新標籤')
        
        ax1.set_ylabel('樣本數')
        ax1.set_title('標籤分佈比較')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['negative', 'positive'])
        ax1.legend()
        
        # 2. 變化率圖
        stats = comparison_results['statistics']
        labels = ['未變化', '已變化']
        sizes = [1 - stats['change_rate'], stats['change_rate']]
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax2.axis('equal')
        ax2.set_title('標籤變化率')
        
        plt.tight_layout()
        plt.show() 