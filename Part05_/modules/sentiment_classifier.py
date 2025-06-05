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
from sklearn.preprocessing import LabelEncoder
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
        self.model_type = 'random_forest'  # 預設使用隨機森林
        
        # 支持的模型類型
        self.available_models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True)
        }
        
        logger.info("情感分類器已初始化")
    
    def prepare_features(self, aspect_vectors: Dict, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        準備分類特徵
        
        Args:
            aspect_vectors: 面向特徵向量字典
            metadata: 包含真實標籤的元數據
            
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
        
        # 需要原始的BERT嵌入向量來計算與面向向量的相似度
        # 這裡假設在某處可以獲取原始嵌入向量
        # 為了完整性，我們將面向向量串聯作為特徵
        
        # 將面向向量合併為單一特徵向量（每個文檔都使用相同的面向向量）
        aspect_names = sorted(aspect_vectors.keys())  # 保證順序一致
        concatenated_features = []
        for aspect_name in aspect_names:
            concatenated_features.extend(aspect_vectors[aspect_name])
        
        # 為每個文檔複製相同的面向特徵（這是一個簡化的方法）
        # 在實際應用中，應該計算每個文檔與各面向向量的相似度
        features = np.tile(concatenated_features, (len(metadata), 1))
        
        logger.info(f"準備了 {features.shape[0]} 個樣本，{features.shape[1]} 維特徵")
        logger.info(f"使用的面向: {aspect_names}")
        logger.info(f"標籤分布: {dict(zip(*np.unique(sentiments, return_counts=True)))}")
        
        self.feature_vectors = features
        self.labels = encoded_labels
        
        return features, encoded_labels
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              model_type: str = 'random_forest', test_size: float = 0.2,
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
        true_label_names = self.label_encoder.inverse_transform(y_test)
        predicted_label_names = self.label_encoder.inverse_transform(test_pred)
        
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
            'true_label_names': true_label_names.tolist(),  # 原始標籤名稱
            'predicted_label_names': predicted_label_names.tolist(),  # 預測標籤名稱
            'predicted_probabilities': test_pred_proba.tolist(),
            'class_names': self.label_encoder.classes_.tolist(),
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
                                    metadata: pd.DataFrame) -> Dict[str, Dict]:
        """
        評估不同注意力機制的分類性能
        
        Args:
            attention_results: 注意力機制分析結果
            metadata: 包含真實標籤的元數據
            
        Returns:
            各注意力機制的分類性能結果
        """
        evaluation_results = {}
        
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
                # 準備特徵
                print(f"      📋 準備特徵向量...")
                aspect_vectors = mechanism_result['aspect_vectors']
                features, labels = self.prepare_features(aspect_vectors, metadata)
                
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
        class_names = self.label_encoder.classes_
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
            'class_names': class_names.tolist(),
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