#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感分類器模組 - 基於注意力機制特徵的情感分類
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import joblib
import os
import time
import torch
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

class SentimentClassifier:
    """基於注意力機制特徵的情感分類器"""
    
    def __init__(self, output_dir: Optional[str] = None, encoder_type: str = 'bert'):
        """
        初始化情感分類器
        
        Args:
            output_dir: 輸出目錄路徑
            encoder_type: 編碼器類型 (bert, gpt, t5, cnn, elmo)
        """
        self.output_dir = output_dir
        self.encoder_type = encoder_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_vectors = None
        self.labels = None
        self.model_type = 'xgboost'  # 修改：預設使用XGBoost（高準確率與性能）
        
        # 自動偵測GPU/CPU環境
        self.device_info = self._detect_compute_environment()
        logger.debug(f"計算環境: {self.device_info['description']}")
        
        # 支持的模型類型 - 優化配置
        self.available_models = self._init_models()
        
        logger.info("情感分類器已初始化")
        logger.debug(f"可用分類器: {list(self.available_models.keys())}")
    
    def _detect_compute_environment(self) -> Dict[str, Any]:
        """自動偵測計算環境（GPU/CPU）"""
        device_info = {
            'has_gpu': False,
            'gpu_name': None,
            'gpu_memory': None,
            'cuda_available': False,
            'device': 'cpu',
            'description': 'CPU Only'
        }
        
        try:
            # 檢測CUDA和GPU
            if torch.cuda.is_available():
                device_info['cuda_available'] = True
                device_info['has_gpu'] = True
                device_info['device'] = 'cuda'
                device_info['gpu_name'] = torch.cuda.get_device_name(0)
                device_info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                device_info['description'] = f"GPU: {device_info['gpu_name']} ({device_info['gpu_memory']:.1f}GB)"
                logger.debug(f"檢測到GPU: {device_info['gpu_name']}")
            else:
                logger.debug("未檢測到CUDA GPU，使用CPU模式")
                
        except Exception as e:
            logger.warning(f"GPU檢測過程中發生錯誤: {str(e)}")
            
        return device_info
    
    def _init_models(self) -> Dict[str, Any]:
        """初始化可用的模型"""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000, 
                C=1.0,
                n_jobs=-1  # 使用所有CPU核心
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            ),
            'svm_linear': SVC(
                kernel='linear', 
                C=1.0, 
                random_state=42, 
                probability=True
            ),
            'naive_bayes': GaussianNB()
        }
        
        # 嘗試載入XGBoost
        try:
            import xgboost as xgb
            
            # 檢查XGBoost版本並記錄
            xgb_version = xgb.__version__
            logger.debug(f"檢測到XGBoost版本: {xgb_version}")
            
            # 檢查是否為2.0.0+版本（使用新參數）
            xgb_major_version = int(xgb_version.split('.')[0])
            is_new_xgb = xgb_major_version >= 2
            
            if is_new_xgb:
                logger.debug("使用XGBoost 2.0.0+新版本參數配置")
            else:
                logger.debug("使用XGBoost 1.x版本參數配置")
            
            # GPU加速配置 - 根據版本和GPU可用性配置參數
            if self.device_info['has_gpu']:
                if is_new_xgb:
                    # XGBoost 2.0.0+ GPU配置 - 使用新參數
                    xgb_params = {
                        'tree_method': 'hist',      # 新版本統一使用 'hist'
                        'device': 'cuda',           # 使用 'device' 替代 'gpu_id'
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    logger.debug("XGBoost配置為GPU模式 (v2.0+: device='cuda', tree_method='hist')")
                else:
                    # XGBoost 1.x GPU配置 - 使用舊參數
                    xgb_params = {
                        'tree_method': 'gpu_hist',  # 舊版本使用 'gpu_hist'
                        'gpu_id': 0,                # 舊版本使用 'gpu_id'
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    logger.debug("XGBoost配置為GPU模式 (v1.x: gpu_id=0, tree_method='gpu_hist')")
                
                logger.info("🚀 GPU加速已啟用")
            else:
                # CPU配置（兩個版本都一樣）
                xgb_params = {
                    'tree_method': 'hist',      # CPU上最快的方法
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1  # 使用所有CPU核心
                }
                if is_new_xgb:
                    xgb_params['device'] = 'cpu'  # 新版本明確指定CPU設備
                    logger.info("XGBoost配置為CPU模式 (v2.0+: device='cpu')")
                else:
                    logger.info("XGBoost配置為CPU模式 (v1.x: tree_method='hist')")
            
            models['xgboost'] = xgb.XGBClassifier(**xgb_params)
            logger.info("XGBoost已成功載入並配置")
            
        except ImportError:
            logger.warning("XGBoost未安裝，請使用 'pip install xgboost' 安裝")
        except Exception as e:
            logger.error(f"XGBoost初始化失敗: {str(e)}")
        
        return models
    
    def get_available_models(self) -> List[str]:
        """獲取可用的模型列表"""
        return list(self.available_models.keys())
    
    def get_device_info(self) -> Dict[str, Any]:
        """獲取設備信息"""
        return self.device_info.copy()
    
    def _get_naive_bayes_classifier(self):
        """獲取Naive Bayes分類器"""
        return GaussianNB()
    
    def prepare_features(self, aspect_vectors: Dict, metadata: pd.DataFrame, 
                        original_embeddings: np.ndarray = None, 
                        fused_features: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        準備分類特徵，支援融合特徵和原始特徵
        
        Args:
            aspect_vectors: 面向特徵向量字典
            metadata: 包含真實標籤的元數據
            original_embeddings: 原始BERT嵌入向量
            fused_features: 融合後的特徵向量（新架構）
            
        Returns:
            features: 特徵矩陣
            labels: 標籤數組
        """
        start_time = time.time()
        logger.info("開始準備分類特徵...")
        
        # 優先使用數值化情感標籤
        if 'sentiment_numeric' in metadata.columns:
            logger.info("使用數值化情感標籤 (sentiment_numeric)")
            sentiments = metadata['sentiment_numeric'].values
            # 數值化標籤已經是編碼形式，直接使用
            encoded_labels = sentiments.astype(int)
            # 設定標籤編碼器的類別（0:負面, 1:中性, 2:正面）
            self.label_encoder.classes_ = np.array([0, 1, 2])
        elif 'sentiment' not in metadata.columns:
            if 'review_stars' in metadata.columns:
                logger.info("未找到 'sentiment' 欄位，根據 'review_stars' 生成數值化情感標籤...")
                # 根據評分生成數值化情感標籤：1-2星=0(負面), 3星=1(中性), 4-5星=2(正面)
                def map_stars_to_numeric(stars):
                    if stars <= 2:
                        return 0  # 負面
                    elif stars == 3:
                        return 1  # 中性
                    else:
                        return 2  # 正面
                
                sentiments = metadata['review_stars'].apply(map_stars_to_numeric).values
                encoded_labels = sentiments.astype(int)
                self.label_encoder.classes_ = np.array([0, 1, 2])
                logger.info(f"生成的數值化情感標籤分佈：{pd.Series(sentiments).value_counts().to_dict()}")
            else:
                raise ValueError("元數據中缺少 'sentiment_numeric'、'sentiment' 和 'review_stars' 欄位")
        else:
            # 提取情感標籤並轉換為數值
            sentiments = metadata['sentiment'].values
            # 編碼標籤
            encoded_labels = self.label_encoder.fit_transform(sentiments)
        
        # 處理特徵：優先使用融合特徵，否則使用原始特徵
        if fused_features is not None:
            logger.info("使用融合特徵進行分類")
            features = fused_features.copy()
            
            # 確保融合特徵數量與元數據匹配
            if len(features) != len(metadata):
                raise ValueError(f"融合特徵數量 ({len(features)}) 與元數據數量 ({len(metadata)}) 不匹配")
            
            logger.info(f"融合特徵形狀: {features.shape}")
        else:
            # 使用原始嵌入向量
            if original_embeddings is None:
                # 嘗試載入編碼器嵌入向量
                if self.output_dir:
                    # 使用通用的檔案檢測邏輯
                    try:
                        from .attention_processor import AttentionProcessor
                        temp_processor = AttentionProcessor(output_dir=self.output_dir, encoder_type=self.encoder_type)
                        embeddings_file = temp_processor._find_existing_embeddings(self.encoder_type)
                        
                        if embeddings_file and os.path.exists(embeddings_file):
                            original_embeddings = np.load(embeddings_file)
                            logger.info(f"已載入 {self.encoder_type.upper()} 嵌入向量，形狀: {original_embeddings.shape}")
                            logger.info(f"檔案來源: {embeddings_file}")
                        else:
                            raise ValueError(f"無法找到 {self.encoder_type.upper()} 嵌入向量文件。請提供 original_embeddings 參數或確保 {self.encoder_type.upper()} 特徵向量文件存在。")
                    except Exception as e:
                        raise ValueError(f"載入 {self.encoder_type.upper()} 嵌入向量時發生錯誤: {e}。請提供 original_embeddings 參數。")
                else:
                    raise ValueError(f"無法找到 {self.encoder_type.upper()} 嵌入向量。請提供 original_embeddings 參數。")
            
            # 確保嵌入向量數量與元數據匹配
            if len(original_embeddings) != len(metadata):
                raise ValueError(f"嵌入向量數量 ({len(original_embeddings)}) 與元數據數量 ({len(metadata)}) 不匹配")
            # 修正：使用原始BERT嵌入向量作為主要特徵
            features = original_embeddings.copy()
            
            # 修正：計算與面向向量的相似度作為額外特徵
            aspect_names = sorted(aspect_vectors.keys())
            similarity_features = []
            
            logger.info(f"計算每個文檔與 {len(aspect_names)} 個面向向量的相似度...")
            
            # 檢查維度相容性
            first_embedding = original_embeddings[0]
            first_aspect_vector = aspect_vectors[aspect_names[0]]
            
            if first_embedding.shape[0] != first_aspect_vector.shape[0]:
                error_msg = f"維度不匹配: 文檔嵌入向量維度 {first_embedding.shape[0]}, 面向向量維度 {first_aspect_vector.shape[0]}"
                logger.error(error_msg)
                logger.error(f"這通常是由於注意力分析和分類評估使用了不同的編碼器造成的")
                logger.error(f"當前編碼器類型: {self.encoder_type.upper()}")
                
                # 嘗試修復：如果面向向量維度是1024而文檔向量是768，說明面向向量來自不同編碼器
                if first_aspect_vector.shape[0] == 1024 and first_embedding.shape[0] == 768:
                    logger.warning("檢測到面向向量來自1024維編碼器（可能是GPT/T5），但文檔向量是768維（BERT）")
                    logger.warning("建議重新運行完整的流水線以確保編碼器一致性")
                elif first_aspect_vector.shape[0] == 768 and first_embedding.shape[0] == 1024:
                    logger.warning("檢測到面向向量來自768維編碼器（BERT），但文檔向量是1024維（可能是GPT/T5）")
                    logger.warning("建議重新運行完整的流水線以確保編碼器一致性")
                
                raise ValueError(f"{error_msg}。請確保注意力分析和分類評估使用相同的編碼器類型。")
            
            for i, embedding in enumerate(original_embeddings):
                doc_similarities = []
                for aspect_name in aspect_names:
                    aspect_vector = aspect_vectors[aspect_name]
                    # 計算餘弦相似度
                    similarity = np.dot(embedding, aspect_vector) / (
                        np.linalg.norm(embedding) * np.linalg.norm(aspect_vector) + 1e-8
                    )
                    doc_similarities.append(similarity)
                similarity_features.append(doc_similarities)
            
            similarity_features = np.array(similarity_features)
            
            # 組合原始特徵和相似度特徵
            features = np.concatenate([original_embeddings, similarity_features], axis=1)
        
        prepare_time = time.time() - start_time
        
        # 只顯示關鍵信息
        feature_type = "融合特徵" if fused_features is not None else f"{self.encoder_type.upper()}+相似度特徵"
        logger.debug(f"特徵準備完成 - {features.shape[0]}樣本 x {features.shape[1]}維 ({feature_type}, {prepare_time:.1f}s)")
        
        self.feature_vectors = features
        self.labels = encoded_labels
        
        return features, encoded_labels
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              model_type: str = None, test_size: float = 0.2,
              original_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        訓練分類模型
        
        Args:
            features: 特徵矩陣
            labels: 標籤數組
            model_type: 模型類型
            test_size: 測試集比例
            original_data: 原始數據DataFrame，用於保存測試集文本
            
        Returns:
            評估結果字典
        """
        # 如果沒有指定模型類型，使用預設值
        if model_type is None:
            model_type = self.model_type
            
        if model_type not in self.available_models:
            available_models = list(self.available_models.keys())
            raise ValueError(f"不支援的模型類型: {model_type}。可用的模型: {available_models}")
        
        start_time = time.time()
        logger.debug(f"開始訓練 {model_type} 模型 ({self.device_info['description']})")
        
        # 分割訓練和測試集
        split_start = time.time()
        if original_data is not None:
            # 如果有原始數據，同時分割原始數據以保持對應關係
            X_train, X_test, y_train, y_test, data_train, data_test = train_test_split(
                features, labels, original_data, test_size=test_size, random_state=42, stratify=labels
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=test_size, random_state=42, stratify=labels
            )
            data_test = None
        
        split_time = time.time() - split_start
        logger.debug(f"數據分割完成 - 訓練:{X_train.shape[0]} 測試:{X_test.shape[0]} ({split_time:.1f}s)")
        
        # 選擇模型
        self.model = self.available_models[model_type]
        self.model_type = model_type
        
        # 訓練模型
        train_start = time.time()
        
        # 針對XGBoost的GPU配置（簡化輸出）
        if model_type == 'xgboost' and self.device_info['has_gpu']:
            try:
                # 確保XGBoost使用GPU配置
                import xgboost as xgb
                xgb_version = xgb.__version__
                xgb_major_version = int(xgb_version.split('.')[0])
                
                if xgb_major_version >= 2:
                    current_device = getattr(self.model, 'device', None)
                    if current_device != 'cuda':
                        self.model.set_params(device='cuda')
                else:
                    current_tree_method = getattr(self.model, 'tree_method', None)
                    if current_tree_method != 'gpu_hist':
                        self.model.set_params(tree_method='gpu_hist', gpu_id=0)
                
                logger.debug(f"XGBoost使用GPU加速")
            except Exception as e:
                logger.debug(f"XGBoost GPU配置警告: {e}")
        
        self.model.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # 預測
        predict_start = time.time()
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_pred_proba = self.model.predict_proba(X_test)
        predict_time = time.time() - predict_start
        
        logger.debug(f"模型訓練+預測完成 ({train_time:.1f}s + {predict_time:.1f}s)")
        
        # 計算評估指標
        results = self._calculate_metrics(
            y_train, train_pred, y_test, test_pred, test_pred_proba
        )
        
        # 添加時間信息
        total_time = time.time() - start_time
        timing_info = {
            'total_time': total_time,
            'split_time': split_time,
            'train_time': train_time,
            'predict_time': predict_time,
            'model_type': model_type,
            'device_info': self.device_info
        }
        
        results['timing_info'] = timing_info
        # 為了向後兼容，也直接設定 training_time 鍵
        results['training_time'] = train_time
        
        # 保存預測結果詳細信息
        # 將編碼的標籤轉換回原始標籤名稱
        # 修正：檢查LabelEncoder是否已經fitted
        if hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_ is not None:
            true_label_names = self.label_encoder.inverse_transform(y_test)
            predicted_label_names = self.label_encoder.inverse_transform(test_pred)
            class_names_list = self.label_encoder.classes_.tolist()
        else:
            # 如果沒有fitted，使用數字標籤
            true_label_names = [f"label_{label}" for label in y_test]
            predicted_label_names = [f"label_{label}" for label in test_pred]
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            class_names_list = [f"label_{label}" for label in unique_labels]
        
        # 保存測試集的文本信息以便後續匹配
        test_texts = []
        if data_test is not None:
            # 獲取測試集的文本
            text_column = None
            for col in ['processed_text', 'text', 'review', 'content']:
                if col in data_test.columns:
                    text_column = col
                    break
            
            if text_column:
                test_texts = data_test[text_column].tolist()
        
        results['prediction_details'] = {
            'test_indices': None,  # 由於訓練時沒有原始索引，這裡設為None
            'true_labels': y_test.tolist(),  # 編碼後的標籤
            'predicted_labels': test_pred.tolist(),  # 編碼後的預測標籤
            'true_label_names': true_label_names.tolist() if hasattr(true_label_names, 'tolist') else list(true_label_names),  # 原始標籤名稱
            'predicted_label_names': predicted_label_names.tolist() if hasattr(predicted_label_names, 'tolist') else list(predicted_label_names),  # 預測標籤名稱
            'predicted_probabilities': test_pred_proba.tolist(),
            'class_names': class_names_list,
            'test_texts': test_texts  # 測試集文本，用於後續匹配
        }
        
        # 保存模型
        if self.output_dir:
            self._save_model()
        
        # 簡化的統計輸出
        logger.debug(f"訓練統計 - 總耗時:{total_time:.1f}s 準確率:{results['test_accuracy']:.4f} (分割:{split_time:.1f}s 訓練:{train_time:.1f}s 預測:{predict_time:.1f}s)")
        
        return results
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        預測情感標籤
        
        Args:
            features: 特徵矩陣
            
        Returns:
            predictions: 預測標籤
            probabilities: 預測機率
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先調用 train() 方法")
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        return predictions, probabilities
    
    def predict_with_details(self, features: np.ndarray, original_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        預測情感標籤並返回詳細結果
        
        Args:
            features: 特徵矩陣
            original_data: 原始數據（包含文本和真實標籤）
            
        Returns:
            包含預測結果和詳細信息的字典
        """
        if self.model is None:
            raise ValueError("模型尚未訓練，請先調用 train() 方法")
        
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        # 解碼預測標籤
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        result = {
            'predictions': predictions,
            'predicted_labels': predicted_labels.tolist(),
            'probabilities': probabilities.tolist(),
            'class_names': self.label_encoder.classes_.tolist()
        }
        
        # 如果有原始數據，添加詳細比對信息
        if original_data is not None:
            result['detailed_comparison'] = self._create_detailed_comparison(
                original_data, predicted_labels, probabilities
            )
        
        return result
    
    def evaluate_attention_mechanisms(self, attention_results: Dict[str, Any], 
                                    metadata: pd.DataFrame,
                                    original_embeddings: np.ndarray = None,
                                    model_type: str = None) -> Dict[str, Dict]:
        """
        評估不同注意力機制的分類性能
        
        Args:
            attention_results: 注意力機制分析結果
            metadata: 包含真實標籤的元數據
            original_embeddings: 原始BERT嵌入向量（修正：新增此參數）
            model_type: 指定分類器類型，如果None則使用預設值
            
        Returns:
            各注意力機制的分類性能結果
        """
        evaluation_results = {}
        
        # 修正：如果沒有提供original_embeddings，嘗試載入（支援多種編碼器）
        if original_embeddings is None:
            if self.output_dir:
                # 使用通用的檔案檢測邏輯
                try:
                    from .attention_processor import AttentionProcessor
                    temp_processor = AttentionProcessor(output_dir=self.output_dir, encoder_type=self.encoder_type)
                    embeddings_file = temp_processor._find_existing_embeddings(self.encoder_type)
                    
                    if embeddings_file and os.path.exists(embeddings_file):
                        original_embeddings = np.load(embeddings_file)
                        logger.info(f"已載入 {self.encoder_type.upper()} 嵌入向量用於分類評估，形狀: {original_embeddings.shape}")
                        logger.info(f"檔案來源: {embeddings_file}")
                    else:
                        logger.warning(f"未找到 {self.encoder_type.upper()} 嵌入向量文件，將嘗試從prepare_features方法中載入")
                except Exception as e:
                    logger.warning(f"載入 {self.encoder_type.upper()} 嵌入向量時發生錯誤: {e}，將嘗試從prepare_features方法中載入")
        
        # 過濾出有效的注意力機制
        valid_mechanisms = []
        print(f"🔍 檢查注意力結果，總共 {len(attention_results)} 個項目:")
        for mechanism_name, mechanism_result in attention_results.items():
            print(f"   - {mechanism_name}: {type(mechanism_result)}")
            if isinstance(mechanism_result, dict):
                print(f"     鍵: {list(mechanism_result.keys())}")
            
            if mechanism_name != 'comparison' and isinstance(mechanism_result, dict) and 'aspect_vectors' in mechanism_result:
                valid_mechanisms.append((mechanism_name, mechanism_result))
                print(f"   ✅ {mechanism_name} 有效，包含aspect_vectors")
            else:
                print(f"   ❌ {mechanism_name} 無效或缺少aspect_vectors")
        
        print(f"📊 開始評估 {len(valid_mechanisms)} 種注意力機制的分類性能...")
        
        # 如果沒有有效機制，提前返回
        if len(valid_mechanisms) == 0:
            print(f"⚠️  警告：沒有找到有效的注意力機制進行分類評估")
            return {'comparison': {'error': '沒有有效的注意力機制結果'}}
        
        for mechanism_name, mechanism_result in tqdm(valid_mechanisms, desc="評估注意力機制"):
            print(f"   🔍 正在評估 {mechanism_name} 注意力機制...")
            logger.info(f"評估 {mechanism_name} 注意力機制的分類性能...")
            
            try:
                # 準備特徵（修正：傳遞原始嵌入向量）
                print(f"      📋 準備特徵向量...")
                aspect_vectors = mechanism_result['aspect_vectors']
                features, labels = self.prepare_features(aspect_vectors, metadata, original_embeddings)
                
                # 訓練和評估
                print(f"      🤖 訓練分類器...")
                # 使用指定的model_type或預設值
                train_model_type = model_type if model_type is not None else self.model_type
                results = self.train(features, labels, model_type=train_model_type, original_data=metadata)
                
                # 添加注意力機制特定信息
                results['attention_mechanism'] = mechanism_name
                results['mechanism_metrics'] = mechanism_result.get('metrics', {})
                
                evaluation_results[mechanism_name] = results
                
                print(f"      ✅ {mechanism_name} 評估完成 - 準確率: {results['test_accuracy']:.4f}, "
                      f"F1分數: {results['test_f1']:.4f}")
                
            except Exception as e:
                print(f"      ❌ 評估 {mechanism_name} 時發生錯誤: {str(e)}")
                logger.error(f"評估 {mechanism_name} 時發生錯誤: {str(e)}")
                
                # 即使失敗也記錄錯誤信息
                evaluation_results[mechanism_name] = {
                    'error': str(e),
                    'attention_mechanism': mechanism_name,
                    'test_accuracy': 0.0,
                    'test_f1': 0.0,
                    'training_time': 0.0
                }
                continue
        
        # 比較不同注意力機制的性能
        print(f"   📈 比較不同注意力機制的性能...")
        comparison = self._compare_mechanisms(evaluation_results)
        evaluation_results['comparison'] = comparison
        
        print(f"✅ 分類性能評估完成！")
        if comparison and 'best_mechanism' in comparison:
            print(f"🏆 最佳分類性能機制: {comparison['best_mechanism']}")
            # 安全訪問 summary 鍵
            summary = comparison.get('summary', {})
            if summary:
                print(f"📊 最佳準確率: {summary.get('best_accuracy', 0):.4f}")
                print(f"📊 最佳F1分數: {summary.get('best_f1', 0):.4f}")
            else:
                print("📊 警告：無法獲取性能統計摘要")
        else:
            print("⚠️  警告：無法進行性能比較")
        
        return evaluation_results
    
    def _calculate_metrics(self, y_train: np.ndarray, train_pred: np.ndarray,
                          y_test: np.ndarray, test_pred: np.ndarray, 
                          test_pred_proba: np.ndarray) -> Dict[str, Any]:
        """計算評估指標"""
        # 準確率
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # 精確率、召回率、F1分數
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            y_train, train_pred, average='weighted'
        )
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, test_pred, average='weighted'
        )
        
        # 額外計算：也計算macro平均，用於比較
        test_precision_macro, test_recall_macro, test_f1_macro, _ = precision_recall_fscore_support(
            y_test, test_pred, average='macro'
        )
        
        print(f"🔍 加權平均 vs 宏平均比較：")
        print(f"   精確率 - 加權: {test_precision:.6f}, 宏: {test_precision_macro:.6f}")
        print(f"   召回率 - 加權: {test_recall:.6f}, 宏: {test_recall_macro:.6f}")
        print(f"   F1分數 - 加權: {test_f1:.6f}, 宏: {test_f1_macro:.6f}")
        
        # 解釋為什麼指標可能相同
        if abs(test_accuracy - test_precision) < 0.001 and abs(test_accuracy - test_recall) < 0.001:
            print(f"⚠️  注意：準確率、精確率、召回率非常接近，可能原因：")
            print(f"   1. 數據集高度平衡")
            print(f"   2. 模型性能極佳")
            print(f"   3. 使用加權平均導致指標趨同")
            print(f"   建議查看宏平均和各類別指標以獲得更詳細分析")
        
        # 混淆矩陣
        confusion_mat = confusion_matrix(y_test, test_pred)
        
        # 分類報告
        # 修正：檢查LabelEncoder是否已經fitted，避免AttributeError
        if hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_ is not None:
            class_names = self.label_encoder.classes_
        else:
            # 如果沒有類別名稱，使用唯一值
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            class_names = [f"class_{label}" for label in unique_labels]
        
        classification_rep = classification_report(
            y_test, test_pred, target_names=class_names, output_dict=True
        )
        
        # 調試輸出：詳細分析為什麼指標相同
        print(f"🔍 指標調試：")
        print(f"   測試準確率: {test_accuracy:.6f}")
        print(f"   測試精確率: {test_precision:.6f}")
        print(f"   測試召回率: {test_recall:.6f}")
        print(f"   測試F1分數: {test_f1:.6f}")
        
        # 檢查類別分佈
        unique_labels, counts = np.unique(y_test, return_counts=True)
        print(f"🔍 測試集類別分佈: {dict(zip(unique_labels, counts))}")
        
        # 檢查混淆矩陣對角線
        print(f"🔍 混淆矩陣:")
        print(confusion_mat)
        
        # 計算每個類別的精確率和召回率
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            y_test, test_pred, average=None
        )
        print(f"🔍 各類別精確率: {per_class_precision}")
        print(f"🔍 各類別召回率: {per_class_recall}")
        print(f"🔍 各類別F1分數: {per_class_f1}")
        print(f"🔍 各類別支持數: {support}")
        
        # 檢查是否所有預測都相同
        unique_preds = np.unique(test_pred)
        print(f"🔍 預測值種類: {unique_preds}")
        print(f"🔍 預測準確率分佈: {np.bincount(test_pred) / len(test_pred)}")
        
        return {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_precision': float(train_precision),
            'test_precision': float(test_precision),
            'train_recall': float(train_recall),
            'test_recall': float(test_recall),
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
            # 額外提供宏平均指標以供比較
            'test_precision_macro': float(test_precision_macro),
            'test_recall_macro': float(test_recall_macro),
            'test_f1_macro': float(test_f1_macro),
            'confusion_matrix': confusion_mat.tolist(),
            'classification_report': classification_rep,
            'class_names': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
            'model_type': self.model_type,
            # 每個類別的詳細指標
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'per_class_support': support.tolist()
        }
    
    def _compare_mechanisms(self, evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """比較不同注意力機制的性能"""
        # 如果沒有評估結果，返回空的但結構完整的比較結果
        if not evaluation_results:
            return {
                'best_mechanism': None,
                'accuracy_ranking': [],
                'f1_ranking': [],
                'summary': {
                    'best_accuracy': 0,
                    'best_f1': 0,
                    'mechanisms_tested': 0,
                    'error': 'No evaluation results available'
                }
            }
        
        # 提取性能指標
        mechanisms = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for mechanism, results in evaluation_results.items():
            if mechanism == 'comparison':
                continue
            
            # 跳過錯誤的結果，但記錄它們
            if 'error' in results:
                print(f"   ⚠️  跳過失敗的機制: {mechanism} (錯誤: {results['error']})")
                continue
            
            # 添加安全檢查，確保結果包含必要的鍵
            try:
                mechanisms.append(mechanism)
                accuracies.append(results.get('test_accuracy', 0))
                f1_scores.append(results.get('test_f1', 0))
                precisions.append(results.get('test_precision', 0))
                recalls.append(results.get('test_recall', 0))
            except (KeyError, TypeError) as e:
                logger.warning(f"機制 {mechanism} 的結果不完整，跳過: {str(e)}")
                continue
        
        # 確保有有效的結果才進行排序
        if not mechanisms:
            return {
                'best_mechanism': None,
                'accuracy_ranking': [],
                'f1_ranking': [],
                'summary': {
                    'best_accuracy': 0,
                    'best_f1': 0,
                    'mechanisms_tested': 0,
                    'error': 'No valid mechanism results found'
                }
            }
        
        # 排序
        accuracy_ranking = sorted(zip(mechanisms, accuracies), key=lambda x: x[1], reverse=True)
        f1_ranking = sorted(zip(mechanisms, f1_scores), key=lambda x: x[1], reverse=True)
        
        # 找出最佳機制
        best_mechanism = accuracy_ranking[0][0] if accuracy_ranking else None
        
        return {
            'best_mechanism': best_mechanism,
            'accuracy_ranking': accuracy_ranking,
            'f1_ranking': f1_ranking,
            'summary': {
                'best_accuracy': accuracy_ranking[0][1] if accuracy_ranking else 0,
                'best_f1': f1_ranking[0][1] if f1_ranking else 0,
                'mechanisms_tested': len(mechanisms)
            }
        }
    
    def _save_model(self):
        """保存訓練好的模型到run目錄根目錄"""
        if self.model is None or self.output_dir is None:
            return

        # 確保保存到run目錄的根目錄
        if any(subdir in self.output_dir for subdir in ["01_preprocessing", "02_bert_encoding", "03_attention_testing", "04_analysis"]):
            # 如果輸出目錄是子目錄，改為父目錄（run目錄根目錄）
            run_dir = os.path.dirname(self.output_dir)
        else:
            run_dir = self.output_dir

        model_path = os.path.join(run_dir, f"sentiment_classifier_{self.model_type}.pkl")
        encoder_path = os.path.join(run_dir, "label_encoder.pkl")

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)

        logger.info(f"模型已保存至: {model_path}")
        logger.info(f"標籤編碼器已保存至: {encoder_path}")
    
    def load_model(self, model_path: str, encoder_path: str):
        """載入預訓練模型"""
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        logger.info(f"模型已載入: {model_path}")
        logger.info(f"標籤編碼器已載入: {encoder_path}")
    
    def _create_detailed_comparison(self, original_data: pd.DataFrame, 
                                  predicted_labels: np.ndarray, 
                                  probabilities: np.ndarray) -> List[Dict]:
        """創建詳細的比對結果"""
        detailed_results = []
        
        for i in range(len(original_data)):
            row = original_data.iloc[i]
            
            # 獲取原始文本和標籤
            original_text = str(row.get('processed_text', row.get('text', row.get('review', ''))))
            original_label = str(row.get('sentiment', 'unknown'))
            
            # 截斷過長的文本
            if len(original_text) > 100:
                display_text = original_text[:97] + "..."
            else:
                display_text = original_text
            
            predicted_label = predicted_labels[i]
            is_correct = predicted_label == original_label
            confidence = float(np.max(probabilities[i]))
            
            detailed_results.append({
                'index': i,
                'original_text': display_text,
                'original_label': original_label,
                'predicted_label': predicted_label,
                'is_correct': is_correct,
                'confidence': confidence
            })
        
        return detailed_results 