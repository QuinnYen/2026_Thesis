#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉驗證模組 - 支援 K 折交叉驗證
為 BERT 情感分析系統提供穩健的模型評估
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import logging
import json
import os
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class CrossValidationEvaluator:
    """K 折交叉驗證評估器"""
    
    def __init__(self, output_dir: Optional[str] = None, n_folds: int = 5, random_state: int = 42):
        """
        初始化交叉驗證評估器
        
        Args:
            output_dir: 輸出目錄
            n_folds: 折數 (推薦 5 或 10)
            random_state: 隨機種子
        """
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_results = {}
        
        # 創建分層 K 折交叉驗證器
        self.skf = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=random_state
        )
        
        logger.info(f"交叉驗證評估器已初始化 - {n_folds} 折交叉驗證")
    
    def evaluate_single_model(self, 
                            features: np.ndarray, 
                            labels: np.ndarray, 
                            model, 
                            model_name: str,
                            label_encoder = None) -> Dict[str, Any]:
        """
        使用 K 折交叉驗證評估單個模型
        
        Args:
            features: 特徵矩陣
            labels: 標籤數組
            model: sklearn 兼容的模型
            model_name: 模型名稱
            label_encoder: 標籤編碼器
            
        Returns:
            交叉驗證結果字典
        """
        print(f"\n🔄 開始 {self.n_folds} 折交叉驗證: {model_name}")
        logger.info(f"開始交叉驗證: {model_name}")
        
        start_time = time.time()
        
        # 定義評估指標
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        # 執行交叉驗證
        cv_results = cross_validate(
            model, features, labels, 
            cv=self.skf, 
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1  # 使用所有CPU核心
        )
        
        # 手動計算每個 fold 的詳細結果
        fold_details = []
        fold_num = 1
        
        for train_idx, test_idx in self.skf.split(features, labels):
            print(f"   📊 Fold {fold_num}/{self.n_folds}")
            
            # 分割數據
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # 訓練模型
            model_copy = self._clone_model(model)
            model_copy.fit(X_train, y_train)
            
            # 預測
            train_pred = model_copy.predict(X_train)
            test_pred = model_copy.predict(X_test)
            test_pred_proba = model_copy.predict_proba(X_test)
            
            # 計算指標
            fold_metrics = self._calculate_fold_metrics(
                y_train, train_pred, y_test, test_pred, test_pred_proba, label_encoder
            )
            fold_metrics['fold'] = fold_num
            fold_metrics['train_size'] = len(X_train)
            fold_metrics['test_size'] = len(X_test)
            
            fold_details.append(fold_metrics)
            fold_num += 1
        
        # 計算總體統計
        summary_stats = self._calculate_summary_statistics(cv_results, fold_details)
        
        total_time = time.time() - start_time
        
        result = {
            'model_name': model_name,
            'n_folds': self.n_folds,
            'cv_scores': {
                'test_accuracy': cv_results['test_accuracy'].tolist(),
                'test_precision': cv_results['test_precision_weighted'].tolist(),
                'test_recall': cv_results['test_recall_weighted'].tolist(),
                'test_f1': cv_results['test_f1_weighted'].tolist(),
                'train_accuracy': cv_results['train_accuracy'].tolist(),
                'train_precision': cv_results['train_precision_weighted'].tolist(),
                'train_recall': cv_results['train_recall_weighted'].tolist(),
                'train_f1': cv_results['train_f1_weighted'].tolist()
            },
            'summary_statistics': summary_stats,
            'fold_details': fold_details,
            'total_evaluation_time': total_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # 顯示結果摘要
        self._print_cv_summary(model_name, summary_stats)
        
        return result
    
    def evaluate_multiple_models(self, 
                                features: np.ndarray, 
                                labels: np.ndarray, 
                                models_dict: Dict[str, Any],
                                label_encoder = None) -> Dict[str, Dict]:
        """
        評估多個模型的交叉驗證性能
        
        Args:
            features: 特徵矩陣
            labels: 標籤數組
            models_dict: 模型字典 {名稱: 模型實例}
            label_encoder: 標籤編碼器
            
        Returns:
            所有模型的交叉驗證結果
        """
        print(f"\n🔬 開始多模型 {self.n_folds} 折交叉驗證比較")
        print(f"   • 模型數量: {len(models_dict)}")
        print(f"   • 數據規模: {features.shape[0]} 樣本, {features.shape[1]} 特徵")
        print(f"   • 類別數量: {len(np.unique(labels))}")
        
        all_results = {}
        
        for model_name, model in models_dict.items():
            try:
                result = self.evaluate_single_model(
                    features, labels, model, model_name, label_encoder
                )
                all_results[model_name] = result
            except Exception as e:
                logger.error(f"評估模型 {model_name} 時發生錯誤: {e}")
                all_results[model_name] = {
                    'error': str(e),
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                }
        
        # 生成比較報告
        comparison = self._compare_models(all_results)
        
        final_results = {
            'individual_results': all_results,
            'comparison': comparison,
            'evaluation_config': {
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'total_samples': features.shape[0],
                'n_features': features.shape[1],
                'n_classes': len(np.unique(labels))
            }
        }
        
        # 保存結果
        if self.output_dir:
            self._save_cv_results(final_results)
        
        return final_results
    
    def evaluate_attention_mechanisms_cv(self, 
                                       attention_results: Dict[str, Any],
                                       metadata: pd.DataFrame,
                                       original_embeddings: np.ndarray,
                                       models_dict: Dict[str, Any],
                                       label_encoder = None) -> Dict[str, Dict]:
        """
        使用交叉驗證評估不同注意力機制的效果
        
        Args:
            attention_results: 注意力機制處理結果
            metadata: 元數據（包含標籤）
            original_embeddings: 原始嵌入向量
            models_dict: 模型字典
            label_encoder: 標籤編碼器
            
        Returns:
            各注意力機制的交叉驗證結果
        """
        print(f"\n🎯 開始注意力機制交叉驗證評估")
        
        # 檢查情感標籤欄位，如果沒有則根據review_stars生成
        if 'sentiment' not in metadata.columns:
            if 'review_stars' in metadata.columns:
                print("未找到 'sentiment' 欄位，根據 'review_stars' 生成情感標籤...")
                # 根據評分生成情感標籤：1-2星=負面, 3星=中性, 4-5星=正面
                def map_stars_to_sentiment(stars):
                    if stars <= 2:
                        return 'negative'
                    elif stars == 3:
                        return 'neutral'
                    else:
                        return 'positive'
                
                metadata = metadata.copy()
                metadata['sentiment'] = metadata['review_stars'].apply(map_stars_to_sentiment)
                print(f"生成的情感標籤分佈：{metadata['sentiment'].value_counts().to_dict()}")
            else:
                raise ValueError("元數據中缺少 'sentiment' 欄位，且無法找到 'review_stars' 欄位來生成情感標籤")
        
        # 編碼標籤
        if label_encoder is None:
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(metadata['sentiment'].values)
        else:
            labels = label_encoder.transform(metadata['sentiment'].values)
        
        attention_cv_results = {}
        
        # 為每個注意力機制進行交叉驗證
        for attention_type, attention_data in attention_results.items():
            if attention_type in ['processing_info', 'comparison']:
                continue
                
            print(f"\n🔍 評估注意力機制: {attention_type}")
            
            try:
                # 準備特徵
                if attention_type == 'no':
                    # 無注意力機制，使用原始嵌入向量
                    features = original_embeddings
                else:
                    # 使用面向特徵向量
                    aspect_vectors = attention_data.get('aspect_vectors', {})
                    features = self._prepare_attention_features(
                        aspect_vectors, original_embeddings
                    )
                
                # 對每個模型進行交叉驗證
                attention_model_results = {}
                for model_name, model in models_dict.items():
                    print(f"   📊 {model_name} + {attention_type}")
                    
                    cv_result = self.evaluate_single_model(
                        features, labels, model, 
                        f"{model_name}_{attention_type}", label_encoder
                    )
                    attention_model_results[model_name] = cv_result
                
                attention_cv_results[attention_type] = {
                    'model_results': attention_model_results,
                    'feature_shape': features.shape,
                    'attention_type': attention_type
                }
                
            except Exception as e:
                logger.error(f"評估注意力機制 {attention_type} 時發生錯誤: {e}")
                attention_cv_results[attention_type] = {
                    'error': str(e),
                    'attention_type': attention_type
                }
        
        # 生成注意力機制比較
        attention_comparison = self._compare_attention_mechanisms(attention_cv_results)
        
        final_results = {
            'attention_results': attention_cv_results,
            'attention_comparison': attention_comparison,
            'evaluation_config': {
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'n_attention_types': len([k for k in attention_results.keys() 
                                        if k not in ['processing_info', 'comparison']]),
                'n_models': len(models_dict)
            }
        }
        
        # 保存結果
        if self.output_dir:
            self._save_attention_cv_results(final_results)
        
        return final_results
    
    def _clone_model(self, model):
        """複製模型實例"""
        from sklearn.base import clone
        return clone(model)
    
    def _calculate_fold_metrics(self, y_train, train_pred, y_test, test_pred, 
                              test_pred_proba, label_encoder) -> Dict[str, Any]:
        """計算單個 fold 的詳細指標"""
        # 基本指標
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # 精確率、召回率、F1
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            y_train, train_pred, average='weighted'
        )
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, test_pred, average='weighted'
        )
        
        # 混淆矩陣
        cm = confusion_matrix(y_test, test_pred)
        
        # 分類報告
        if label_encoder and hasattr(label_encoder, 'classes_'):
            target_names = label_encoder.classes_
        else:
            target_names = None
        
        class_report = classification_report(
            y_test, test_pred, target_names=target_names, output_dict=True
        )
        
        return {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'train_precision': float(train_precision),
            'test_precision': float(test_precision),
            'train_recall': float(train_recall),
            'test_recall': float(test_recall),
            'train_f1': float(train_f1),
            'test_f1': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }
    
    def _calculate_summary_statistics(self, cv_results, fold_details) -> Dict[str, Any]:
        """計算交叉驗證的統計摘要"""
        stats = {}
        
        # 對每個指標計算平均值和標準差
        for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            metric_name = metric.replace('_weighted', '')
            stats[f'test_{metric_name}_mean'] = float(np.mean(test_scores))
            stats[f'test_{metric_name}_std'] = float(np.std(test_scores))
            stats[f'test_{metric_name}_min'] = float(np.min(test_scores))
            stats[f'test_{metric_name}_max'] = float(np.max(test_scores))
            
            stats[f'train_{metric_name}_mean'] = float(np.mean(train_scores))
            stats[f'train_{metric_name}_std'] = float(np.std(train_scores))
        
        # 過擬合指標 (訓練分數 - 測試分數)
        train_test_gap = np.mean(cv_results['train_accuracy']) - np.mean(cv_results['test_accuracy'])
        stats['overfitting_score'] = float(train_test_gap)
        
        # 穩定性指標 (標準差)
        stats['stability_score'] = float(np.std(cv_results['test_accuracy']))
        
        return stats
    
    def _prepare_attention_features(self, aspect_vectors, original_embeddings) -> np.ndarray:
        """準備注意力機制特徵"""
        # 這裡簡化處理，實際可能需要更複雜的特徵組合
        if not aspect_vectors:
            return original_embeddings
        
        # 使用第一個可用的面向向量
        first_aspect = list(aspect_vectors.keys())[0]
        aspect_features = aspect_vectors[first_aspect]
        
        # 組合原始嵌入和面向特徵
        if len(aspect_features) == len(original_embeddings):
            combined_features = np.hstack([original_embeddings, aspect_features])
            return combined_features
        else:
            return original_embeddings
    
    def _compare_models(self, all_results) -> Dict[str, Any]:
        """比較多個模型的性能"""
        comparison = {
            'ranking': [],
            'best_model': None,
            'performance_summary': {}
        }
        
        # 收集各模型的平均測試準確率
        model_scores = {}
        for model_name, result in all_results.items():
            if 'error' not in result:
                acc_mean = result['summary_statistics']['test_accuracy_mean']
                f1_mean = result['summary_statistics']['test_f1_mean']
                stability = result['summary_statistics']['stability_score']
                
                # 綜合分數 (準確率 70% + F1分數 20% + 穩定性 10%)
                composite_score = acc_mean * 0.7 + f1_mean * 0.2 + (1 - stability) * 0.1
                
                model_scores[model_name] = {
                    'accuracy_mean': acc_mean,
                    'f1_mean': f1_mean,
                    'stability_score': stability,
                    'composite_score': composite_score
                }
        
        # 排序
        sorted_models = sorted(model_scores.items(), 
                             key=lambda x: x[1]['composite_score'], 
                             reverse=True)
        
        comparison['ranking'] = [
            {
                'rank': i + 1,
                'model_name': model_name,
                **scores
            }
            for i, (model_name, scores) in enumerate(sorted_models)
        ]
        
        if sorted_models:
            comparison['best_model'] = sorted_models[0][0]
            comparison['performance_summary'] = {
                'best_accuracy': max(s['accuracy_mean'] for s in model_scores.values()),
                'best_f1': max(s['f1_mean'] for s in model_scores.values()),
                'most_stable': min(s['stability_score'] for s in model_scores.values()),
                'n_successful_models': len(model_scores)
            }
        
        return comparison
    
    def _compare_attention_mechanisms(self, attention_cv_results) -> Dict[str, Any]:
        """比較不同注意力機制的性能"""
        comparison = {
            'attention_ranking': [],
            'best_attention_model_combo': None,
            'mechanism_summary': {}
        }
        
        # 收集各注意力機制+模型組合的分數
        combo_scores = {}
        
        for attention_type, attention_data in attention_cv_results.items():
            if 'error' in attention_data:
                continue
                
            model_results = attention_data.get('model_results', {})
            for model_name, result in model_results.items():
                if 'error' not in result:
                    combo_name = f"{attention_type}+{model_name}"
                    acc_mean = result['summary_statistics']['test_accuracy_mean']
                    f1_mean = result['summary_statistics']['test_f1_mean']
                    stability = result['summary_statistics']['stability_score']
                    
                    combo_scores[combo_name] = {
                        'attention_type': attention_type,
                        'model_name': model_name,
                        'accuracy_mean': acc_mean,
                        'f1_mean': f1_mean,
                        'stability_score': stability,
                        'composite_score': acc_mean * 0.7 + f1_mean * 0.2 + (1 - stability) * 0.1
                    }
        
        # 排序組合
        sorted_combos = sorted(combo_scores.items(), 
                             key=lambda x: x[1]['composite_score'], 
                             reverse=True)
        
        comparison['attention_ranking'] = [
            {
                'rank': i + 1,
                'combination': combo_name,
                **scores
            }
            for i, (combo_name, scores) in enumerate(sorted_combos)
        ]
        
        if sorted_combos:
            comparison['best_attention_model_combo'] = sorted_combos[0][0]
        
        return comparison
    
    def _print_cv_summary(self, model_name: str, stats: Dict[str, Any]):
        """顯示交叉驗證結果摘要"""
        print(f"\n📊 {model_name} 交叉驗證結果:")
        print(f"   • 平均準確率: {stats['test_accuracy_mean']:.4f} ± {stats['test_accuracy_std']:.4f}")
        print(f"   • 平均 F1 分數: {stats['test_f1_mean']:.4f} ± {stats['test_f1_std']:.4f}")
        print(f"   • 穩定性 (σ): {stats['stability_score']:.4f}")
        print(f"   • 過擬合指標: {stats['overfitting_score']:.4f}")
    
    def _save_cv_results(self, results: Dict[str, Any]):
        """保存交叉驗證結果"""
        if not self.output_dir:
            return
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存詳細結果
        results_file = os.path.join(self.output_dir, f"cv_{self.n_folds}fold_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"交叉驗證結果已保存: {results_file}")
    
    def _save_attention_cv_results(self, results: Dict[str, Any]):
        """保存注意力機制交叉驗證結果"""
        if not self.output_dir:
            return
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存注意力機制交叉驗證結果
        results_file = os.path.join(self.output_dir, f"attention_cv_{self.n_folds}fold_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"注意力機制交叉驗證結果已保存: {results_file}") 