#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能動態權重學習器 - 自動優化注意力機制組合權重
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from abc import ABC, abstractmethod
import json
import os
from datetime import datetime

# 優化庫
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.optimize import minimize, differential_evolution

# 可選的優化庫
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna未安裝，貝葉斯優化功能將不可用")

# 機器學習模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# 項目內部模組
from .attention_mechanism import create_attention_mechanism, CombinedAttention
from .sentiment_classifier import SentimentClassifier

logger = logging.getLogger(__name__)

class BaseWeightLearner(ABC):
    """權重學習器基類"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.best_weights = None
        self.best_score = None
        self.learning_history = []
        
    @abstractmethod
    def learn_weights(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, 
                     metadata_train: pd.DataFrame, metadata_val: pd.DataFrame) -> Dict[str, float]:
        """學習最佳權重組合"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回學習器名稱"""
        pass
    
    def _print_learning_results(self, method_name: str, weights: Dict[str, float], score: float):
        """在終端機顯著打印權重學習結果"""
        if not weights:
            print(f"\n⚠️  {method_name} 學習失敗，無有效權重")
            return
        
        print("\n" + "🔥" * 50)
        print("🧠 智能權重學習完成！")
        print("🔥" * 50)
        print(f"🎯 學習方法: {method_name}")
        print(f"📈 學習分數: {score:.6f}")
        print(f"⭐ 最佳權重配置:")
        
        # 計算權重統計
        weights_list = list(weights.values())
        total_weight = sum(weights_list)
        
        # 按權重大小排序顯示
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for i, (mechanism, weight) in enumerate(sorted_weights):
            percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "🔸"
            print(f"   {rank_emoji} {mechanism}: {weight:.6f} ({percentage:.2f}%)")
        
        # 顯示權重分布分析
        if len(weights_list) > 1:
            max_weight = max(weights_list)
            min_weight = min(weights_list)
            weight_range = max_weight - min_weight
            dominant_mechanism = max(weights.items(), key=lambda x: x[1])
            
            print(f"\n📊 權重分布分析:")
            print(f"   • 主導機制: {dominant_mechanism[0]} ({dominant_mechanism[1]*100:.2f}%)")
            print(f"   • 權重範圍: {weight_range:.6f}")
            print(f"   • 權重分布: {'均衡' if weight_range < 0.3 else '偏向' if weight_range < 0.6 else '極端偏向'}")
        
        # 顯示權重建議
        print(f"\n💡 權重應用建議:")
        if max(weights_list) > 0.7:
            print(f"   • 系統偏向單一注意力機制，建議關注主導機制的性能")
        elif max(weights_list) < 0.4:
            print(f"   • 權重分布均衡，各機制協同工作效果較好")
        else:
            print(f"   • 存在明顯的主次關係，建議優化主導機制")
        
        print("🔥" * 50)
        
        # 記錄到日誌
        logger.info(f"智能權重學習結果 - {method_name}: {weights}, 分數: {score:.6f}")

class GridSearchWeightLearner(BaseWeightLearner):
    """網格搜索權重學習器"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.search_space = self._create_search_space()
        
    def _create_search_space(self) -> List[Dict[str, float]]:
        """創建搜索空間"""
        search_space = []
        
        # 定義各種權重組合
        weight_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # 單一機制
        for weight in weight_values:
            if weight > 0:
                search_space.append({'similarity': weight, 'keyword': 0, 'self': 0})
                search_space.append({'similarity': 0, 'keyword': weight, 'self': 0})
                search_space.append({'similarity': 0, 'keyword': 0, 'self': weight})
        
        # 雙重組合
        for w1 in weight_values:
            for w2 in weight_values:
                if w1 + w2 == 1.0 and w1 > 0 and w2 > 0:
                    search_space.append({'similarity': w1, 'keyword': w2, 'self': 0})
                    search_space.append({'similarity': w1, 'keyword': 0, 'self': w2})
                    search_space.append({'similarity': 0, 'keyword': w1, 'self': w2})
        
        # 三重組合
        for w1 in [0.2, 0.3, 0.4, 0.5]:
            for w2 in [0.2, 0.3, 0.4, 0.5]:
                w3 = 1.0 - w1 - w2
                if 0.1 <= w3 <= 0.6 and abs(w1 + w2 + w3 - 1.0) < 1e-6:
                    search_space.append({'similarity': w1, 'keyword': w2, 'self': w3})
        
        return search_space
    
    def get_name(self) -> str:
        return "GridSearchWeightLearner"
    
    def learn_weights(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, 
                     metadata_train: pd.DataFrame, metadata_val: pd.DataFrame) -> Dict[str, float]:
        """使用網格搜索學習最佳權重"""
        
        logger.info(f"開始網格搜索權重學習，搜索空間大小: {len(self.search_space)}")
        
        best_weights = None
        best_score = -1
        
        for i, weights in enumerate(self.search_space):
            try:
                # 數據驗證
                if X_train is None or X_val is None or metadata_train is None or metadata_val is None:
                    logger.warning("輸入數據包含None值")
                    continue
                
                # 計算組合注意力
                config = self.config or {}
                combined_attention = CombinedAttention(config)
                
                # 計算訓練集注意力權重
                train_attn_weights, _ = combined_attention.compute_attention(
                    X_train, metadata_train, weights=weights)
                
                if train_attn_weights is None:
                    logger.warning(f"權重組合 {weights} 訓練集注意力權重計算失敗")
                    continue
                
                # 確保權重形狀正確
                if train_attn_weights.ndim == 2:
                    train_attn_weights = train_attn_weights.mean(axis=0)
                
                # 應用注意力權重到特徵
                weighted_X_train = X_train * train_attn_weights.reshape(-1, 1)
                
                # 計算驗證集注意力權重  
                val_attn_weights, _ = combined_attention.compute_attention(
                    X_val, metadata_val, weights=weights)
                
                if val_attn_weights is None:
                    logger.warning(f"權重組合 {weights} 驗證集注意力權重計算失敗")
                    continue
                
                # 確保權重形狀正確
                if val_attn_weights.ndim == 2:
                    val_attn_weights = val_attn_weights.mean(axis=0)
                
                weighted_X_val = X_val * val_attn_weights.reshape(-1, 1)
                
                # 訓練分類器
                classifier = LogisticRegression(random_state=42, max_iter=1000)
                classifier.fit(weighted_X_train, y_train)
                
                # 評估性能
                y_pred = classifier.predict(weighted_X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                # 記錄結果
                self.learning_history.append({
                    'weights': weights.copy(),
                    'score': score,
                    'iteration': i
                })
                
                if score > best_score:
                    best_score = score
                    best_weights = weights.copy()
                
                if i % 50 == 0:
                    logger.info(f"進度: {i}/{len(self.search_space)}, 當前最佳分數: {best_score:.4f}")
                    
            except Exception as e:
                logger.warning(f"權重組合 {weights} 評估失敗: {str(e)}")
                import traceback
                logger.warning(f"錯誤追蹤: {traceback.format_exc()}")
                continue
        
        self.best_weights = best_weights
        self.best_score = best_score
        
        logger.info(f"網格搜索完成，最佳權重: {best_weights}, 最佳分數: {best_score:.4f}")
        
        # 在終端機顯著打印學習結果
        self._print_learning_results("網格搜索 (Grid Search)", best_weights, best_score)
        
        return best_weights

class GeneticAlgorithmWeightLearner(BaseWeightLearner):
    """遺傳算法權重學習器"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.population_size = (config or {}).get('population_size', 50)
        self.max_generations = (config or {}).get('max_generations', 100)
        
    def get_name(self) -> str:
        return "GeneticAlgorithmWeightLearner"
    
    def learn_weights(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, 
                     metadata_train: pd.DataFrame, metadata_val: pd.DataFrame) -> Dict[str, float]:
        """使用遺傳算法學習最佳權重"""
        
        def objective_function(weights_array):
            """目標函數：輸入權重數組，返回負的F1分數（因為minimize是最小化）"""
            try:
                # 數據驗證
                if X_train is None or X_val is None or metadata_train is None or metadata_val is None:
                    return 1.0
                
                # 歸一化權重
                weights_sum = np.sum(weights_array)
                if weights_sum == 0:
                    return 1.0  # 返回最差分數
                
                normalized_weights = weights_array / weights_sum
                weights_dict = {
                    'similarity': normalized_weights[0],
                    'keyword': normalized_weights[1], 
                    'self': normalized_weights[2]
                }
                
                # 計算組合注意力
                config = self.config or {}
                combined_attention = CombinedAttention(config)
                
                # 計算訓練集注意力權重
                train_attn_weights, _ = combined_attention.compute_attention(
                    X_train, metadata_train, weights=weights_dict)
                
                if train_attn_weights is None:
                    return 1.0
                
                # 確保權重形狀正確
                if train_attn_weights.ndim == 2:
                    train_attn_weights = train_attn_weights.mean(axis=0)
                
                weighted_X_train = X_train * train_attn_weights.reshape(-1, 1)
                
                # 計算驗證集注意力權重
                val_attn_weights, _ = combined_attention.compute_attention(
                    X_val, metadata_val, weights=weights_dict)
                
                if val_attn_weights is None:
                    return 1.0
                
                # 確保權重形狀正確
                if val_attn_weights.ndim == 2:
                    val_attn_weights = val_attn_weights.mean(axis=0)
                
                weighted_X_val = X_val * val_attn_weights.reshape(-1, 1)
                
                # 訓練分類器
                classifier = LogisticRegression(random_state=42, max_iter=500)
                classifier.fit(weighted_X_train, y_train)
                
                # 評估性能
                y_pred = classifier.predict(weighted_X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return -score  # 返回負值因為要最小化
                
            except Exception as e:
                logger.warning(f"目標函數評估失敗: {str(e)}")
                import traceback
                logger.warning(f"錯誤追蹤: {traceback.format_exc()}")
                return 1.0  # 返回最差分數
        
        logger.info("開始遺傳算法權重學習")
        
        # 使用scipy的differential_evolution進行優化
        bounds = [(0, 1), (0, 1), (0, 1)]  # 三個權重的範圍都是[0,1]
        
        result = differential_evolution(
            objective_function,
            bounds,
            seed=42,
            maxiter=self.max_generations,
            popsize=self.population_size,
            atol=1e-6,
            tol=1e-6
        )
        
        # 歸一化最佳權重
        best_weights_array = result.x / np.sum(result.x)
        best_weights = {
            'similarity': best_weights_array[0],
            'keyword': best_weights_array[1],
            'self': best_weights_array[2]
        }
        
        self.best_weights = best_weights
        self.best_score = -result.fun  # 轉回正值
        
        logger.info(f"遺傳算法完成，最佳權重: {best_weights}, 最佳分數: {self.best_score:.4f}")
        
        # 在終端機顯著打印學習結果
        self._print_learning_results("遺傳算法 (Genetic Algorithm)", best_weights, self.best_score)
        
        return best_weights

class BayesianOptimizationWeightLearner(BaseWeightLearner):
    """貝葉斯優化權重學習器"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.n_trials = (config or {}).get('n_trials', 100)
        
    def get_name(self) -> str:
        return "BayesianOptimizationWeightLearner"
    
    def learn_weights(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray, 
                     metadata_train: pd.DataFrame, metadata_val: pd.DataFrame) -> Dict[str, float]:
        """使用貝葉斯優化學習最佳權重"""
        
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna未安裝，使用網格搜索替代貝葉斯優化")
            # 退回到網格搜索
            grid_learner = GridSearchWeightLearner(self.config)
            return grid_learner.learn_weights(X_train, y_train, X_val, y_val, 
                                            metadata_train, metadata_val)
        
        def objective(trial):
            """Optuna目標函數"""
            try:
                # 數據驗證
                if X_train is None or X_val is None or metadata_train is None or metadata_val is None:
                    logger.warning("輸入數據包含None值")
                    return 0.0
                
                if len(X_train) == 0 or len(X_val) == 0:
                    logger.warning("輸入數據為空")
                    return 0.0
                
                # 採樣權重
                w1 = trial.suggest_float('similarity', 0.0, 1.0)
                w2 = trial.suggest_float('keyword', 0.0, 1.0)
                w3 = trial.suggest_float('self', 0.0, 1.0)
                
                # 歸一化
                total = w1 + w2 + w3
                if total == 0:
                    return 0.0
                
                weights_dict = {
                    'similarity': w1 / total,
                    'keyword': w2 / total,
                    'self': w3 / total
                }
                
                # 計算組合注意力
                config = self.config or {}
                combined_attention = CombinedAttention(config)
                
                # 計算訓練集注意力權重
                train_attn_weights, _ = combined_attention.compute_attention(
                    X_train, metadata_train, weights=weights_dict)
                
                if train_attn_weights is None:
                    logger.warning("訓練集注意力權重計算失敗")
                    return 0.0
                
                # 確保權重形狀正確
                if train_attn_weights.ndim == 2:
                    train_attn_weights = train_attn_weights.mean(axis=0)
                
                weighted_X_train = X_train * train_attn_weights.reshape(-1, 1)
                
                # 計算驗證集注意力權重
                val_attn_weights, _ = combined_attention.compute_attention(
                    X_val, metadata_val, weights=weights_dict)
                
                if val_attn_weights is None:
                    logger.warning("驗證集注意力權重計算失敗")
                    return 0.0
                
                # 確保權重形狀正確
                if val_attn_weights.ndim == 2:
                    val_attn_weights = val_attn_weights.mean(axis=0)
                
                weighted_X_val = X_val * val_attn_weights.reshape(-1, 1)
                
                # 訓練分類器
                classifier = LogisticRegression(random_state=42, max_iter=500)
                classifier.fit(weighted_X_train, y_train)
                
                # 評估性能
                y_pred = classifier.predict(weighted_X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                
                return score
                
            except Exception as e:
                logger.warning(f"貝葉斯優化目標函數評估失敗: {str(e)}")
                import traceback
                logger.warning(f"錯誤追蹤: {traceback.format_exc()}")
                return 0.0
        
        logger.info("開始貝葉斯優化權重學習")
        
        # 創建study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        # 獲取最佳參數
        best_params = study.best_params
        total = sum(best_params.values())
        
        best_weights = {
            'similarity': best_params['similarity'] / total,
            'keyword': best_params['keyword'] / total, 
            'self': best_params['self'] / total
        }
        
        self.best_weights = best_weights
        self.best_score = study.best_value
        
        logger.info(f"貝葉斯優化完成，最佳權重: {best_weights}, 最佳分數: {self.best_score:.4f}")
        
        # 在終端機顯著打印學習結果
        self._print_learning_results("貝葉斯優化 (Bayesian Optimization)", best_weights, self.best_score)
        
        return best_weights

class AdaptiveWeightLearner:
    """自適應權重學習器主類"""
    
    def __init__(self, output_dir: Optional[str] = None, config: Optional[Dict] = None):
        self.output_dir = output_dir
        self.config = config or {}
        
        # 初始化各種學習器（根據可用性）
        self.learners = [
            GridSearchWeightLearner(self.config),
            GeneticAlgorithmWeightLearner(self.config)
        ]
        
        # 只有在Optuna可用時才添加貝葉斯優化學習器
        if OPTUNA_AVAILABLE:
            self.learners.append(BayesianOptimizationWeightLearner(self.config))
        
        self.best_learner = None
        self.learning_results = {}
        
    def learn_optimal_weights(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            metadata_train: pd.DataFrame, metadata_val: pd.DataFrame,
                            learner_name: str = 'auto') -> Dict[str, Any]:
        """學習最佳權重組合
        
        Args:
            X_train: 訓練特徵
            y_train: 訓練標籤
            X_val: 驗證特徵
            y_val: 驗證標籤
            metadata_train: 訓練元數據
            metadata_val: 驗證元數據
            learner_name: 學習器名稱 ('grid', 'genetic', 'bayesian', 'auto')
            
        Returns:
            Dict: 包含最佳權重和學習結果的字典
        """
        
        logger.info(f"開始自適應權重學習，使用學習器: {learner_name}")
        
        if learner_name == 'auto':
            # 自動選擇：根據數據大小選擇合適的學習器
            data_size = len(X_train)
            if data_size < 1000:
                learner_name = 'grid'  # 小數據集使用網格搜索
            elif data_size < 5000:
                learner_name = 'genetic'  # 中等數據集使用遺傳算法
            else:
                learner_name = 'bayesian'  # 大數據集使用貝葉斯優化
            
            logger.info(f"根據數據大小 {data_size} 自動選擇學習器: {learner_name}")
        
        # 選擇學習器（根據可用性）
        learner_map = {
            'grid': GridSearchWeightLearner,
            'genetic': GeneticAlgorithmWeightLearner
        }
        
        # 只有在Optuna可用時才添加貝葉斯優化選項
        if OPTUNA_AVAILABLE:
            learner_map['bayesian'] = BayesianOptimizationWeightLearner
        elif learner_name == 'bayesian':
            logger.warning("貝葉斯優化不可用，使用遺傳算法替代")
            learner_name = 'genetic'
        
        if learner_name not in learner_map:
            logger.warning(f"未知的學習器名稱: {learner_name}，使用網格搜索")
            learner_name = 'grid'
        
        learner = learner_map[learner_name](self.config)
        
        # 學習權重
        start_time = datetime.now()
        best_weights = learner.learn_weights(X_train, y_train, X_val, y_val, 
                                           metadata_train, metadata_val)
        end_time = datetime.now()
        
        # 保存結果
        results = {
            'best_weights': best_weights,
            'best_score': learner.best_score,
            'learner_name': learner.get_name(),
            'learning_time': (end_time - start_time).total_seconds(),
            'learning_history': getattr(learner, 'learning_history', []),
            'timestamp': datetime.now().isoformat()
        }
        
        self.learning_results[learner_name] = results
        self.best_learner = learner
        
        # 保存到文件
        if self.output_dir:
            self._save_results(results, learner_name)
        
        logger.info(f"權重學習完成，耗時: {results['learning_time']:.2f}秒")
        
        # 如果有有效權重，在終端機顯示最終總結
        if best_weights:
            print("\n" + "⭐" * 60)
            print("🎉 智能權重學習任務完成！")
            print("⭐" * 60)
            print(f"🏁 總耗時: {results['learning_time']:.2f} 秒")
            print(f"🎯 最終採用權重:")
            for mechanism, weight in best_weights.items():
                print(f"   🔸 {mechanism}: {weight:.6f}")
            print(f"📈 最終學習分數: {learner.best_score:.6f}")
            print("⭐" * 60)
        
        return results
    
    def compare_learners(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        metadata_train: pd.DataFrame, metadata_val: pd.DataFrame) -> Dict[str, Any]:
        """比較不同學習器的性能"""
        
        logger.info("開始比較不同權重學習器的性能")
        
        comparison_results = {}
        
        for learner in self.learners:
            learner_name = learner.get_name()
            
            try:
                logger.info(f"測試學習器: {learner_name}")
                
                start_time = datetime.now()
                best_weights = learner.learn_weights(X_train, y_train, X_val, y_val,
                                                   metadata_train, metadata_val)
                end_time = datetime.now()
                
                comparison_results[learner_name] = {
                    'best_weights': best_weights,
                    'best_score': learner.best_score,
                    'learning_time': (end_time - start_time).total_seconds(),
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"學習器 {learner_name} 失敗: {str(e)}")
                comparison_results[learner_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 找出最佳學習器
        best_learner_name = None
        best_score = -1
        
        for name, result in comparison_results.items():
            if result.get('success', False) and result.get('best_score', -1) > best_score:
                best_score = result['best_score']
                best_learner_name = name
        
        comparison_results['summary'] = {
            'best_learner': best_learner_name,
            'best_score': best_score,
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        # 保存比較結果
        if self.output_dir:
            self._save_comparison_results(comparison_results)
        
        logger.info(f"學習器比較完成，最佳學習器: {best_learner_name}")
        
        return comparison_results
    
    def _save_results(self, results: Dict[str, Any], learner_name: str):
        """保存學習結果到文件"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            filename = f"weight_learning_{learner_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"權重學習結果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存權重學習結果失敗: {str(e)}")
    
    def _save_comparison_results(self, comparison_results: Dict[str, Any]):
        """保存比較結果到文件"""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            filename = f"learner_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"學習器比較結果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存學習器比較結果失敗: {str(e)}")

def create_adaptive_weight_learner(output_dir: Optional[str] = None, 
                                 config: Optional[Dict] = None) -> AdaptiveWeightLearner:
    """創建自適應權重學習器
    
    Args:
        output_dir: 輸出目錄
        config: 配置參數
        
    Returns:
        AdaptiveWeightLearner: 自適應權重學習器實例
    """
    return AdaptiveWeightLearner(output_dir, config)