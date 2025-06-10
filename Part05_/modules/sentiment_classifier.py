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
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

class SentimentClassifier:
    """基於注意力機制特徵的情感分類器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化情感分類器
        
        Args:
            output_dir: 輸出目錄路徑
        """
        self.output_dir = output_dir
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_vectors = None
        self.labels = None
        self.model_type = 'svm'  # 修改：預設使用SVM（更適合小數據集）
        
        # 支持的模型類型 - 針對小數據集優化
        self.available_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'svm': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),  # 優化SVM配置
            'naive_bayes': self._get_naive_bayes_classifier(),  # 新增Naive Bayes
            'svm_linear': SVC(kernel='linear', C=1.0, random_state=42, probability=True)  # 線性SVM選項
        }
        
        logger.info("情感分類器已初始化")
    
    def _get_naive_bayes_classifier(self):
        """獲取Naive Bayes分類器"""
        return GaussianNB()
    
    def prepare_features(self, aspect_vectors: Dict, metadata: pd.DataFrame, 
                        original_embeddings: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        準備分類特徵
        
        Args:
            aspect_vectors: 面向特徵向量字典
            metadata: 包含真實標籤的元數據
            original_embeddings: 原始BERT嵌入向量（修正：新增此參數）
            
        Returns:
            features: 特徵矩陣
            labels: 標籤數組
        """
        # 檢查情感標籤欄位
        if 'sentiment' not in metadata.columns:
            raise ValueError("元數據中缺少 'sentiment' 欄位")
        
        # 提取情感標籤
        sentiments = metadata['sentiment'].values
        
        # 編碼標籤
        encoded_labels = self.label_encoder.fit_transform(sentiments)
        
        # 修正：獲取原始BERT嵌入向量
        if original_embeddings is None:
            # 嘗試從輸出目錄載入BERT嵌入向量
            if self.output_dir:
                embeddings_file = os.path.join(self.output_dir, "02_bert_embeddings.npy")
                if os.path.exists(embeddings_file):
                    original_embeddings = np.load(embeddings_file)
                    logger.info(f"已載入原始BERT嵌入向量，形狀: {original_embeddings.shape}")
                else:
                    raise ValueError("無法找到原始BERT嵌入向量文件。請提供 original_embeddings 參數或確保BERT特徵向量文件存在於輸出目錄中。")
            else:
                raise ValueError("無法找到原始BERT嵌入向量。請提供 original_embeddings 參數。")
        
        # 確保嵌入向量數量與元數據匹配
        if len(original_embeddings) != len(metadata):
            raise ValueError(f"嵌入向量數量 ({len(original_embeddings)}) 與元數據數量 ({len(metadata)}) 不匹配")
        
        # 修正：使用原始BERT嵌入向量作為主要特徵
        features = original_embeddings.copy()
        
        # 修正：計算與面向向量的相似度作為額外特徵
        aspect_names = sorted(aspect_vectors.keys())
        similarity_features = []
        
        logger.info(f"計算每個文檔與 {len(aspect_names)} 個面向向量的相似度...")
        
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
        
        logger.info(f"準備了 {features.shape[0]} 個樣本，{features.shape[1]} 維特徵")
        logger.info(f"  - 原始BERT特徵: {original_embeddings.shape[1]} 維")
        logger.info(f"  - 面向相似度特徵: {similarity_features.shape[1]} 維")
        logger.info(f"使用的面向: {aspect_names}")
        logger.info(f"標籤分布: {dict(zip(*np.unique(sentiments, return_counts=True)))}")
        
        self.feature_vectors = features
        self.labels = encoded_labels
        
        return features, encoded_labels
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              model_type: str = 'svm', test_size: float = 0.2,
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
        if model_type not in self.available_models:
            raise ValueError(f"不支援的模型類型: {model_type}")
        
        # 分割訓練和測試集
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
        
        # 選擇模型
        self.model = self.available_models[model_type]
        self.model_type = model_type
        
        logger.info(f"開始訓練 {model_type} 模型...")
        logger.info(f"訓練集大小: {X_train.shape[0]}, 測試集大小: {X_test.shape[0]}")
        
        # 訓練模型
        self.model.fit(X_train, y_train)
        
        # 預測
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_pred_proba = self.model.predict_proba(X_test)
        
        # 計算評估指標
        results = self._calculate_metrics(
            y_train, train_pred, y_test, test_pred, test_pred_proba
        )
        
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
        
        logger.info(f"模型訓練完成！測試準確率: {results['test_accuracy']:.4f}")
        
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
                                    original_embeddings: np.ndarray = None) -> Dict[str, Dict]:
        """
        評估不同注意力機制的分類性能
        
        Args:
            attention_results: 注意力機制分析結果
            metadata: 包含真實標籤的元數據
            original_embeddings: 原始BERT嵌入向量（修正：新增此參數）
            
        Returns:
            各注意力機制的分類性能結果
        """
        evaluation_results = {}
        
        # 修正：如果沒有提供original_embeddings，嘗試載入
        if original_embeddings is None:
            if self.output_dir:
                embeddings_file = os.path.join(self.output_dir, "02_bert_embeddings.npy")
                if os.path.exists(embeddings_file):
                    original_embeddings = np.load(embeddings_file)
                    logger.info(f"已載入原始BERT嵌入向量用於分類評估，形狀: {original_embeddings.shape}")
                else:
                    logger.warning("未找到原始BERT嵌入向量文件，將嘗試從prepare_features方法中載入")
        
        # 過濾出有效的注意力機制
        valid_mechanisms = []
        for mechanism_name, mechanism_result in attention_results.items():
            if mechanism_name != 'comparison' and 'aspect_vectors' in mechanism_result:
                valid_mechanisms.append((mechanism_name, mechanism_result))
        
        print(f"📊 開始評估 {len(valid_mechanisms)} 種注意力機制的分類性能...")
        
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
                results = self.train(features, labels, model_type=self.model_type, original_data=metadata)
                
                # 添加注意力機制特定信息
                results['attention_mechanism'] = mechanism_name
                results['mechanism_metrics'] = mechanism_result.get('metrics', {})
                
                evaluation_results[mechanism_name] = results
                
                print(f"      ✅ {mechanism_name} 評估完成 - 準確率: {results['test_accuracy']:.4f}, "
                      f"F1分數: {results['test_f1']:.4f}")
                
            except Exception as e:
                print(f"      ❌ 評估 {mechanism_name} 時發生錯誤: {str(e)}")
                logger.error(f"評估 {mechanism_name} 時發生錯誤: {str(e)}")
                continue
        
        # 比較不同注意力機制的性能
        print(f"   📈 比較不同注意力機制的性能...")
        comparison = self._compare_mechanisms(evaluation_results)
        evaluation_results['comparison'] = comparison
        
        print(f"✅ 分類性能評估完成！")
        if comparison and 'best_mechanism' in comparison:
            print(f"🏆 最佳分類性能機制: {comparison['best_mechanism']}")
            print(f"📊 最佳準確率: {comparison['summary']['best_accuracy']:.4f}")
            print(f"📊 最佳F1分數: {comparison['summary']['best_f1']:.4f}")
        
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
        
        return {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'train_precision': float(train_precision),
            'test_precision': float(test_precision),
            'train_recall': float(train_recall),
            'test_recall': float(test_recall),
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
            'confusion_matrix': confusion_mat.tolist(),
            'classification_report': classification_rep,
            'class_names': class_names.tolist() if hasattr(class_names, 'tolist') else list(class_names),
            'model_type': self.model_type
        }
    
    def _compare_mechanisms(self, evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """比較不同注意力機制的性能"""
        if not evaluation_results:
            return {}
        
        # 提取性能指標
        mechanisms = []
        accuracies = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for mechanism, results in evaluation_results.items():
            if mechanism == 'comparison':
                continue
            
            mechanisms.append(mechanism)
            accuracies.append(results['test_accuracy'])
            f1_scores.append(results['test_f1'])
            precisions.append(results['test_precision'])
            recalls.append(results['test_recall'])
        
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
        """保存訓練好的模型"""
        if self.model is None or self.output_dir is None:
            return
        
        model_path = os.path.join(self.output_dir, f"sentiment_classifier_{self.model_type}.pkl")
        encoder_path = os.path.join(self.output_dir, "label_encoder.pkl")
        
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