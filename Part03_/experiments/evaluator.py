"""
評估工具模組
此模組負責評估面向分析結果的品質
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
import time

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 導入相關模組
from Part03_.utils.config_manager import ConfigManager
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.visualizers import Visualizer

class Evaluator:
    """
    評估器類
    用於評估面向分析結果的品質
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化評估器
        
        Args:
            config: 配置管理器，如果沒有提供則創建新的
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 設置日誌
        log_dir = self.config.get('logging.log_dir', './Part03_/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'evaluator.log')
        
        self.logger = logging.getLogger('evaluator')
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
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        # 配置 Matplotlib 中文字體支援
        try:
            self.logger.info("配置 Matplotlib 中文字體支援")
            
            # 列出可能的中文字體
            chinese_fonts = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'SimSun', 
                            'NSimSun', 'FangSong', 'KaiTi', 'DFKai-SB']
            
            # 測試是否存在這些字體，找到第一個可用的
            font_found = False
            for font_name in chinese_fonts:
                try:
                    test_font = fm.findfont(fm.FontProperties(family=font_name))
                    if test_font and 'ttf' in test_font.lower():
                        plt.rcParams['font.family'] = font_name
                        self.logger.info(f"已設定 Matplotlib 使用字體: {font_name}")
                        font_found = True
                        break
                except Exception as font_error:
                    self.logger.warning(f"測試字體 {font_name} 時出錯: {str(font_error)}")
            
            # 如果沒有找到內建字體，嘗試使用 Matplotlib 內建的中文字體
            if not font_found:
                try:
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans',
                                                    'Lucida Grande', 'Verdana', 'Geneva', 'Lucid',
                                                    'Arial', 'Helvetica', 'sans-serif']
                    plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
                    self.logger.info("已設定 Matplotlib 使用預設中文字體方案")
                except Exception as rc_error:
                    self.logger.warning(f"設定 Matplotlib 預設字體方案時出錯: {str(rc_error)}")
            
            # 嘗試創建測試圖形，確保中文字體設置生效
            try:
                test_fig = plt.figure(figsize=(1, 1))
                test_plot = test_fig.add_subplot(111)
                test_plot.set_title('中文測試')
                test_image_path = os.path.join(log_dir, 'evaluator_font_test.png')
                test_fig.savefig(test_image_path)
                plt.close(test_fig)
                self.logger.info(f"中文字體測試圖片已儲存至: {test_image_path}")
            except Exception as test_error:
                self.logger.warning(f"生成中文字體測試圖時出錯: {str(test_error)}")
                
        except ImportError:
            self.logger.warning("無法導入 Matplotlib，跳過字體配置")
        except Exception as font_config_error:
            self.logger.warning(f"配置 Matplotlib 字體時出錯: {str(font_config_error)}")
        
        # 初始化視覺化工具
        self.visualizer = Visualizer()
        
        # 輸出目錄
        self.output_dir = self.config.get('data_settings.output_directory', './Part03_/results/')
        self.eval_dir = os.path.join(self.output_dir, 'evaluation')
        os.makedirs(self.eval_dir, exist_ok=True)
    
    def evaluate_aspects(self, df: pd.DataFrame, aspect_results: Dict[str, Any],
                       sentiment_column: Optional[str] = 'sentiment',
                       console_output: bool = True) -> Dict[str, Any]:
        """
        評估面向分析結果
        
        Args:
            df: 原始數據框
            aspect_results: 面向分析結果
            sentiment_column: 情感列名稱，如果有的話
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            評估結果字典
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("面向評估")
            logger = ConsoleOutputManager.setup_console_logger("aspect_evaluation", log_file)
        else:
            logger = self.logger
        
        try:
            logger.info("開始評估面向分析結果...")
            
            # 提取面向信息
            aspect_vectors = aspect_results.get('aspect_vectors', {})
            aspect_labels = aspect_results.get('aspect_labels', {})
            text_aspect_matrix = aspect_results.get('text_aspect_matrix', None)
            
            if not aspect_vectors or not aspect_labels or text_aspect_matrix is None:
                error_msg = "面向結果中缺少必要資訊"
                logger.error(error_msg)
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                raise ValueError(error_msg)
            
            # 準備評估結果字典
            evaluation_results = {
                'metrics': {},
                'visualizations': {},
                'detailed_analysis': {}
            }
            
            # 1. 計算聚類評估指標
            logger.info("計算聚類評估指標...")
            clustering_metrics = self._evaluate_clustering(text_aspect_matrix)
            evaluation_results['metrics'].update(clustering_metrics)
            
            # 2. 評估面向的獨特性
            logger.info("評估面向獨特性...")
            uniqueness_metrics = self._evaluate_aspect_uniqueness(aspect_vectors)
            evaluation_results['metrics'].update(uniqueness_metrics)
            
            # 3. 評估面向的覆蓋率
            logger.info("評估面向覆蓋率...")
            coverage_metrics = self._evaluate_coverage(text_aspect_matrix)
            evaluation_results['metrics'].update(coverage_metrics)
            
            # 4. 如果有情感標籤，評估面向與情感的關聯
            if sentiment_column in df.columns:
                logger.info("評估面向與情感的關聯...")
                sentiment_metrics = self._evaluate_sentiment_correlation(
                    df, text_aspect_matrix, sentiment_column, aspect_results.get('id_to_index', {})
                )
                evaluation_results['metrics'].update(sentiment_metrics)
            
            # 5. 面向分布分析
            logger.info("分析文本的面向分布...")
            distribution_analysis = self._analyze_aspect_distribution(text_aspect_matrix, aspect_labels)
            evaluation_results['detailed_analysis']['distribution'] = distribution_analysis
            
            # 6. 生成面向評估圖表
            logger.info("生成評估圖表...")
            visualizations = self._generate_evaluation_charts(
                text_aspect_matrix, aspect_labels, df, sentiment_column if sentiment_column in df.columns else None
            )
            evaluation_results['visualizations'] = visualizations
            
            # 7. 面向詳細分析
            logger.info("進行面向詳細分析...")
            for aspect_id, aspect_info in aspect_vectors.items():
                # 計算每個面向的文本分布
                dominant_texts = self._get_dominant_texts_for_aspect(
                    aspect_id, text_aspect_matrix, df, 'processed_text', limit=5
                )
                
                evaluation_results['detailed_analysis'].setdefault('aspects', {})[aspect_id] = {
                    'label': aspect_labels.get(aspect_id, f"面向{aspect_id}"),
                    'keywords': aspect_info.get('keywords', []),
                    'dominant_texts': dominant_texts
                }
                
                # 如果有情感標籤，計算面向的情感分布
                if sentiment_column in df.columns:
                    sentiment_dist = self._calculate_aspect_sentiment_distribution(
                        aspect_id, text_aspect_matrix, df, sentiment_column, aspect_results.get('id_to_index', {})
                    )
                    evaluation_results['detailed_analysis']['aspects'][aspect_id]['sentiment_distribution'] = sentiment_dist
            
            # 8. 總體評分（基於各個指標的加權平均）
            overall_score = self._calculate_overall_score(evaluation_results['metrics'])
            evaluation_results['metrics']['overall_score'] = overall_score
            
            logger.info(f"評估完成，總體得分: {overall_score:.2f}/10.0")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"評估過程中發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def _evaluate_clustering(self, text_aspect_matrix: np.ndarray) -> Dict[str, float]:
        """
        評估文本-面向聚類質量
        
        Args:
            text_aspect_matrix: 文本-面向關聯矩陣
            
        Returns:
            聚類評估指標
        """
        metrics = {}
        
        # 使用KMeans對文本進行重新聚類
        n_clusters = text_aspect_matrix.shape[1]  # 使用與面向數相同的聚類數
        
        # 如果樣本數太少，調整聚類數
        if n_clusters > text_aspect_matrix.shape[0]:
            n_clusters = max(2, text_aspect_matrix.shape[0] // 2)
        
        # 確保聚類數至少為2
        n_clusters = max(2, n_clusters)
        
        try:
            # 檢查數據是否足夠進行聚類
            if text_aspect_matrix.shape[0] < 2 or n_clusters < 2:
                return {'silhouette_score': 0.0, 'davies_bouldin_score': 0.0, 'calinski_harabasz_score': 0.0}
                
            # 處理可能的 NaN 值
            if np.isnan(text_aspect_matrix).any():
                text_aspect_matrix = np.nan_to_num(text_aspect_matrix, nan=0.0)
            
            # 對文本-面向矩陣進行KMeans聚類
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(text_aspect_matrix)
            
            # 計算輪廓係數
            if text_aspect_matrix.shape[0] > n_clusters:
                try:
                    silhouette = silhouette_score(text_aspect_matrix, clusters)
                    metrics['silhouette_score'] = silhouette
                except Exception as silhouette_error:
                    self.logger.warning(f"計算輪廓係數時出錯: {str(silhouette_error)}")
                    metrics['silhouette_score'] = 0.0
            
            # Davies-Bouldin指數 (較低更好)
            try:
                davies_bouldin = davies_bouldin_score(text_aspect_matrix, clusters)
                metrics['davies_bouldin_score'] = davies_bouldin
            except Exception as db_error:
                self.logger.warning(f"計算Davies-Bouldin指數時出錯: {str(db_error)}")
                metrics['davies_bouldin_score'] = 0.0
            
            # Calinski-Harabasz指數 (較高更好)
            try:
                calinski_harabasz = calinski_harabasz_score(text_aspect_matrix, clusters)
                metrics['calinski_harabasz_score'] = calinski_harabasz
            except Exception as ch_error:
                self.logger.warning(f"計算Calinski-Harabasz指數時出錯: {str(ch_error)}")
                metrics['calinski_harabasz_score'] = 0.0
            
        except Exception as e:
            self.logger.warning(f"計算聚類指標時出錯: {str(e)}")
            metrics = {
                'silhouette_score': 0.0,
                'davies_bouldin_score': 0.0,
                'calinski_harabasz_score': 0.0
            }
        
        return metrics
    
    def _evaluate_aspect_uniqueness(self, aspect_vectors: Dict[int, Dict[str, Any]]) -> Dict[str, float]:
        """
        評估面向向量的獨特性
        
        Args:
            aspect_vectors: 面向向量字典 {aspect_id: {'vector': vector, ...}}
            
        Returns:
            獨特性評估指標
        """
        metrics = {}
        
        # 提取所有面向向量
        vectors = []
        for aspect_id, aspect_info in aspect_vectors.items():
            if 'vector' in aspect_info:
                vectors.append(aspect_info['vector'])
        
        if not vectors or len(vectors) < 2:
            return {
                'aspect_uniqueness': 0.0,
                'max_aspect_similarity': 0.0,
                'avg_aspect_similarity': 0.0
            }
        
        try:
            # 計算所有面向對之間的餘弦相似度，處理可能的零向量
            vectors = np.array(vectors)
            similarities = np.zeros((len(vectors), len(vectors)))
            
            for i in range(len(vectors)):
                for j in range(len(vectors)):
                    if i != j:
                        # 檢查向量是否為零向量
                        norm_i = np.linalg.norm(vectors[i])
                        norm_j = np.linalg.norm(vectors[j])
                        
                        if norm_i > 0 and norm_j > 0:
                            similarity = np.dot(vectors[i], vectors[j]) / (norm_i * norm_j)
                            similarities[i, j] = similarity
                        else:
                            similarities[i, j] = 0  # 零向量與其他向量的相似度定義為0
            
            # 計算平均相似度和最大相似度，排除對角線元素
            if len(vectors) > 1:
                similarities_flat = similarities[~np.eye(similarities.shape[0], dtype=bool)]
                if len(similarities_flat) > 0:
                    avg_similarity = np.mean(similarities_flat)
                    max_similarity = np.max(similarities_flat) if similarities_flat.size > 0 else 0.0
                else:
                    avg_similarity = 0.0
                    max_similarity = 0.0
            else:
                avg_similarity = 0.0
                max_similarity = 0.0
            
            # 獨特性 = 1 - 平均相似度
            uniqueness = 1 - avg_similarity
            
            metrics['aspect_uniqueness'] = uniqueness
            metrics['max_aspect_similarity'] = max_similarity
            metrics['avg_aspect_similarity'] = avg_similarity
        
        except Exception as e:
            self.logger.warning(f"計算面向獨特性時出錯: {str(e)}")
            metrics = {
                'aspect_uniqueness': 0.0,
                'max_aspect_similarity': 0.0,
                'avg_aspect_similarity': 0.0
            }
        
        return metrics
    
    def _evaluate_coverage(self, text_aspect_matrix: np.ndarray, threshold: float = 0.3) -> Dict[str, float]:
        """
        評估面向對文本的覆蓋率
        
        Args:
            text_aspect_matrix: 文本-面向關聯矩陣
            threshold: 面向重要性閾值
            
        Returns:
            覆蓋率評估指標
        """
        metrics = {}
        
        # 計算每個文本最高的面向權重
        max_weights = np.max(text_aspect_matrix, axis=1)
        
        # 計算平均最大權重
        avg_max_weight = np.mean(max_weights)
        metrics['avg_max_aspect_weight'] = avg_max_weight
        
        # 計算覆蓋率 (至少有一個面向權重大於閾值的文本比例)
        covered_docs = np.sum(max_weights >= threshold)
        coverage = covered_docs / len(max_weights)
        metrics['coverage_ratio'] = coverage
        
        # 計算每個文本明顯面向的數量 (面向權重大於閾值)
        significant_aspects_per_doc = np.sum(text_aspect_matrix >= threshold, axis=1)
        avg_significant_aspects = np.mean(significant_aspects_per_doc)
        metrics['avg_significant_aspects'] = avg_significant_aspects
        
        return metrics
    
    def _evaluate_sentiment_correlation(self, df: pd.DataFrame, text_aspect_matrix: np.ndarray,
                                     sentiment_column: str, id_to_index: Dict[str, int]) -> Dict[str, float]:
        """
        評估面向與情感的關聯度
        
        Args:
            df: 數據框
            text_aspect_matrix: 文本-面向關聯矩陣
            sentiment_column: 情感列名
            id_to_index: ID到索引的映射
            
        Returns:
            情感關聯評估指標
        """
        metrics = {}
        
        # 確保情感列存在
        if sentiment_column not in df.columns:
            return {'sentiment_correlation': 0.0}
        
        try:
            # 將情感標籤轉換為數值
            sentiment_map = {
                'positive': 1,
                'negative': -1,
                'neutral': 0
            }
            
            # 創建情感數值列表
            sentiments = []
            for idx, row in df.iterrows():
                id_val = row.get('id')
                if id_val in id_to_index:
                    sentiment = row[sentiment_column]
                    if isinstance(sentiment, str):
                        sentiment = sentiment_map.get(sentiment.lower(), 0)
                    sentiments.append(sentiment)
                    
            # 如果情感都一樣，那麼相關性為0
            if len(set(sentiments)) <= 1:
                return {'sentiment_correlation': 0.0}
            
            # 計算每個面向與情感的相關性
            sentiments = np.array(sentiments)
            n_aspects = text_aspect_matrix.shape[1]
            correlations = []
            
            for aspect_idx in range(n_aspects):
                aspect_weights = text_aspect_matrix[:len(sentiments), aspect_idx]
                correlation = np.corrcoef(aspect_weights, sentiments)[0, 1]
                correlations.append(abs(correlation))
            
            # 計算平均相關性和最大相關性
            avg_correlation = np.nanmean(correlations)
            max_correlation = np.nanmax(correlations)
            
            metrics['avg_sentiment_correlation'] = avg_correlation
            metrics['max_sentiment_correlation'] = max_correlation
            
        except Exception as e:
            self.logger.warning(f"計算情感相關性時發生錯誤: {str(e)}")
            metrics['sentiment_correlation'] = 0.0
        
        return metrics
    
    def _analyze_aspect_distribution(self, text_aspect_matrix: np.ndarray, 
                                  aspect_labels: Dict[int, str]) -> Dict[str, Any]:
        """
        分析文本的面向分布
        
        Args:
            text_aspect_matrix: 文本-面向關聯矩陣
            aspect_labels: 面向標籤字典
            
        Returns:
            面向分布分析結果
        """
        distribution = {}
        
        # 計算每個面向的平均權重
        avg_weights = np.mean(text_aspect_matrix, axis=0)
        
        # 計算每個面向作為主要面向的文本比例
        dominant_aspects = np.argmax(text_aspect_matrix, axis=1)
        aspect_counts = np.bincount(dominant_aspects, minlength=len(aspect_labels))
        aspect_percentages = aspect_counts / len(dominant_aspects)
        
        # 構建分布結果
        distribution['avg_weights'] = {aspect_id: float(avg_weights[int(aspect_id)]) 
                                     for aspect_id in aspect_labels}
        
        distribution['dominant_percentages'] = {aspect_id: float(aspect_percentages[int(aspect_id)]) 
                                             for aspect_id in aspect_labels}
        
        return distribution
    
    def _calculate_aspect_sentiment_distribution(self, aspect_id: int, text_aspect_matrix: np.ndarray,
                                              df: pd.DataFrame, sentiment_column: str, 
                                              id_to_index: Dict[str, int]) -> Dict[str, int]:
        """
        計算特定面向的情感分布
        
        Args:
            aspect_id: 面向ID
            text_aspect_matrix: 文本-面向關聯矩陣
            df: 數據框
            sentiment_column: 情感列名
            id_to_index: ID到索引的映射
            
        Returns:
            面向的情感分布
        """
        sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        # 獲取面向權重
        aspect_weights = text_aspect_matrix[:, int(aspect_id)]
        
        # 閾值
        threshold = 0.3
        
        # 遍歷數據框
        for idx, row in df.iterrows():
            id_val = row.get('id')
            if id_val in id_to_index:
                matrix_idx = id_to_index[id_val]
                if matrix_idx < len(aspect_weights) and aspect_weights[matrix_idx] >= threshold:
                    sentiment = row.get(sentiment_column, '').lower()
                    if sentiment in sentiment_distribution:
                        sentiment_distribution[sentiment] += 1
        
        return sentiment_distribution
    
    def _get_dominant_texts_for_aspect(self, aspect_id: int, text_aspect_matrix: np.ndarray,
                                    df: pd.DataFrame, text_column: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        獲取面向的主要文本示例
        
        Args:
            aspect_id: 面向ID
            text_aspect_matrix: 文本-面向關聯矩陣
            df: 數據框
            text_column: 文本列名
            limit: 最大返回數量
            
        Returns:
            面向的主要文本列表
        """
        # 獲取面向權重
        aspect_weights = text_aspect_matrix[:, int(aspect_id)]
        
        # 排序找出最重要的文本
        indices = np.argsort(aspect_weights)[::-1][:limit]
        
        # 提取文本及其權重
        texts = []
        for idx in indices:
            if idx < len(df):
                row = df.iloc[idx]
                texts.append({
                    'text': row[text_column] if text_column in df.columns else '',
                    'weight': float(aspect_weights[idx])
                })
        
        return texts
    
    def _generate_evaluation_charts(self, text_aspect_matrix: np.ndarray, 
                                 aspect_labels: Dict[int, str],
                                 df: pd.DataFrame, sentiment_column: Optional[str] = None) -> Dict[str, str]:
        """
        生成評估圖表
        
        Args:
            text_aspect_matrix: 文本-面向關聯矩陣
            aspect_labels: 面向標籤字典
            df: 數據框
            sentiment_column: 情感列名，如果有的話
            
        Returns:
            圖表文件路徑字典
        """
        visualizations = {}
        
        # 設置保存路徑
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            # 在非主線程中使用 Matplotlib 的非交互模式
            plt.ioff()
            
            # 1. 面向分布熱力圖
            heatmap_path = os.path.join(vis_dir, f'aspect_heatmap_{timestamp}.png')
            
            # 顯示前20行數據（或更少）
            sample_size = min(20, text_aspect_matrix.shape[0])
            sample_matrix = text_aspect_matrix[:sample_size]
            
            # 使用 Figure 和 Axes 對象，避免使用 pyplot 的交互功能
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            sns.heatmap(sample_matrix, cmap="YlGnBu", ax=ax,
                      xticklabels=[aspect_labels.get(i, f"面向{i+1}") for i in range(sample_matrix.shape[1])],
                      yticklabels=range(sample_size))
            ax.set_title("文本-面向關聯熱力圖 (前20項)")
            fig.tight_layout()
            fig.savefig(heatmap_path, dpi=300)
            plt.close(fig)
            
            visualizations['heatmap'] = heatmap_path
            
            # 2. 面向平均權重條形圖
            weights_path = os.path.join(vis_dir, f'aspect_weights_{timestamp}.png')
            
            avg_weights = np.mean(text_aspect_matrix, axis=0)
            aspects = [aspect_labels.get(i, f"面向{i+1}") for i in range(len(avg_weights))]
            
            # 按權重降序排序
            sorted_indices = np.argsort(avg_weights)[::-1]
            sorted_weights = avg_weights[sorted_indices]
            sorted_aspects = [aspects[i] for i in sorted_indices]
            
            # 同樣使用 Figure 和 Axes 對象
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            bars = ax.bar(sorted_aspects, sorted_weights)
            ax.set_title("面向平均權重分布")
            ax.set_xticklabels(sorted_aspects, rotation=45, ha='right')
            ax.set_ylabel("平均權重")
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # 添加數值標籤
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom')
            
            fig.tight_layout()
            fig.savefig(weights_path, dpi=300)
            plt.close(fig)
            
            visualizations['weights_chart'] = weights_path
            
            # 3. 如果有情感列，生成面向-情感對應圖
            if sentiment_column and sentiment_column in df.columns:
                sentiment_path = os.path.join(vis_dir, f'aspect_sentiment_{timestamp}.png')
                
                # 提取並轉換情感標籤
                sentiment_map = {
                    'positive': 1,
                    'neutral': 0,
                    'negative': -1
                }
                
                sentiments = []
                for idx, row in df.iterrows():
                    if idx < text_aspect_matrix.shape[0]:
                        sentiment = row[sentiment_column]
                        if isinstance(sentiment, str):
                            sentiment = sentiment_map.get(sentiment.lower(), 0)
                        sentiments.append(sentiment)
                
                sentiments = np.array(sentiments)
                
                # 計算每個面向與情感的相關性，使用安全的方式處理可能的錯誤
                correlations = []
                for aspect_idx in range(text_aspect_matrix.shape[1]):
                    if aspect_idx < text_aspect_matrix.shape[1]:
                        aspect_weights = text_aspect_matrix[:min(len(sentiments), text_aspect_matrix.shape[0]), aspect_idx]
                        
                        # 檢查數據是否足夠計算相關性
                        if len(aspect_weights) > 1 and len(np.unique(aspect_weights)) > 1 and len(np.unique(sentiments[:len(aspect_weights)])) > 1:
                            try:
                                # 使用 np.ma.masked_invalid 處理可能的 NaN 值
                                aspect_weights = np.ma.masked_invalid(aspect_weights)
                                sentiment_vals = np.ma.masked_invalid(sentiments[:len(aspect_weights)])
                                valid_indices = ~(aspect_weights.mask | sentiment_vals.mask)
                                
                                if np.sum(valid_indices) > 1:
                                    correlation = np.corrcoef(aspect_weights.compressed(), sentiment_vals.compressed())[0, 1]
                                    correlations.append(correlation)
                                else:
                                    correlations.append(0)
                            except Exception as corr_error:
                                self.logger.warning(f"計算相關性時出錯: {str(corr_error)}")
                                correlations.append(0)
                        else:
                            correlations.append(0)
                
                # 生成條形圖，使用 Figure 和 Axes 對象
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111)
                
                # 確保 correlations 長度與 aspects 相同
                if len(correlations) != len(aspects):
                    correlations = correlations + [0] * (len(aspects) - len(correlations))
                
                # 安全地將相關性值映射到顏色
                safe_correlations = np.clip(np.array(correlations), -1, 1)
                color_values = safe_correlations/2 + 0.5
                bars = ax.bar(aspects, correlations, color=plt.cm.coolwarm(color_values))
                
                ax.set_title("面向與情感的相關性")
                ax.set_xticklabels(aspects, rotation=45, ha='right')
                ax.set_ylabel("相關係數")
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # 添加數值標籤
                for bar in bars:
                    height = bar.get_height()
                    y_pos = height + 0.01 if height >= 0 else height - 0.05
                    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                           f'{height:.2f}', ha='center', va='bottom')
                
                fig.tight_layout()
                fig.savefig(sentiment_path, dpi=300)
                plt.close(fig)
                
                visualizations['sentiment_chart'] = sentiment_path
        
        except Exception as e:
            self.logger.warning(f"生成評估圖表時發生錯誤: {str(e)}")
        
        return visualizations
    
    def _generate_comparison_charts(self, evaluations: List[Dict[str, Any]], 
                                names: List[str]) -> Dict[str, str]:
        """
        生成模型比較圖表
        
        Args:
            evaluations: 評估結果列表
            names: 模型名稱列表
            
        Returns:
            圖表文件路徑字典
        """
        visualizations = {}
        
        # 設置保存路徑
        vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            # 在非主線程中使用 Matplotlib 的非交互模式
            plt.ioff()
            
            # 1. 主要指標雷達圖
            radar_path = os.path.join(vis_dir, f'model_comparison_radar_{timestamp}.png')
            
            # 選擇主要指標
            main_metrics = ['overall_score', 'silhouette_score', 'aspect_uniqueness', 
                          'coverage_ratio', 'avg_sentiment_correlation']
            
            # 準備數據
            metrics_data = []
            for eval_data in evaluations:
                metrics = {}
                for metric in main_metrics:
                    if metric in eval_data['evaluation']['metrics']:
                        # 歸一化指標值到 [0, 1] 區間
                        value = eval_data['evaluation']['metrics'][metric]
                        if metric == 'overall_score':
                            value = value / 10.0  # 0-10 -> 0-1
                        metrics[metric] = max(0, min(1, value))
                metrics_data.append(metrics)
            
            # 繪製雷達圖
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # 確保有數據
            if metrics_data and all(metrics_data):
                # 取得所有可用的指標
                available_metrics = set()
                for metrics in metrics_data:
                    available_metrics.update(metrics.keys())
                available_metrics = sorted(list(available_metrics))
                
                if available_metrics:
                    angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
                    angles += angles[:1]  # 閉合圖形
                    
                    # 繪製每個模型的雷達圖
                    for i, (metrics, name) in enumerate(zip(metrics_data, names)):
                        values = [metrics.get(metric, 0) for metric in available_metrics]
                        values += values[:1]  # 閉合圖形
                        
                        ax.plot(angles, values, linewidth=2, label=name)
                        ax.fill(angles, values, alpha=0.1)
                    
                    # 設置雷達圖標籤
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(available_metrics)
                    
                    # 添加圖例和標題
                    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                    ax.set_title("模型評估指標比較", size=15)
                    
                    fig.tight_layout()
                    fig.savefig(radar_path, dpi=300)
                    plt.close(fig)
                    
                    visualizations['radar_chart'] = radar_path
            
            # 2. 總體得分條形圖
            score_path = os.path.join(vis_dir, f'model_comparison_scores_{timestamp}.png')
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            
            scores = [eval_data['evaluation']['metrics'].get('overall_score', 0) for eval_data in evaluations]
            
            # 生成適當的顏色映射
            colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
            bars = ax.bar(names, scores, color=colors)
            
            ax.set_title("模型總體得分比較")
            ax.set_ylabel("得分 (0-10)")
            ax.set_ylim(0, 10)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # 添加數值標籤
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                       f'{height:.2f}', ha='center', va='bottom')
            
            fig.tight_layout()
            fig.savefig(score_path, dpi=300)
            plt.close(fig)
            
            visualizations['score_chart'] = score_path
            
        except Exception as e:
            self.logger.warning(f"生成比較圖表時發生錯誤: {str(e)}")
        
        return visualizations

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        計算總體評分
        
        Args:
            metrics: 指標字典
            
        Returns:
            總體評分 (0-10分)
        """
        score = 5.0  # 基礎分數
        
        # 基於聚類質量調整
        if 'silhouette_score' in metrics:
            # silhouette範圍是 [-1, 1]，轉換為 [0, 2] 並加權
            score += metrics['silhouette_score'] * 2
        
        if 'davies_bouldin_score' in metrics:
            # davies_bouldin通常在 [0, 無窮大]，越低越好，使用倒數並限制範圍
            db_score = metrics['davies_bouldin_score']
            if db_score > 0:
                score -= min(1, 1/db_score)  # 減少最多1分
        
        # 根據面向獨特性調整
        if 'aspect_uniqueness' in metrics:
            # uniqueness範圍是 [0, 1]，加權
            score += metrics['aspect_uniqueness'] * 2
        
        # 根據覆蓋率調整
        if 'coverage_ratio' in metrics:
            # coverage範圍是 [0, 1]，加權
            score += metrics['coverage_ratio'] * 1.5
        
        # 根據平均顯著面向數調整
        if 'avg_significant_aspects' in metrics:
            # 每個文檔平均有1-2個面向為理想狀態
            avg_aspects = metrics['avg_significant_aspects']
            if 1.0 <= avg_aspects <= 2.5:
                score += 0.5
        
        # 根據情感相關性調整
        if 'avg_sentiment_correlation' in metrics:
            # 相關性範圍是 [0, 1]，加權
            score += metrics['avg_sentiment_correlation'] * 1.5
        
        # 限制分數範圍
        score = max(0, min(10, score))
        
        return score

    def compare_models(self, model_results: List[Dict[str, Any]], df: pd.DataFrame, 
                     names: List[str], console_output: bool = True) -> Dict[str, Any]:
        """
        比較多個模型的面向分析結果
        
        Args:
            model_results: 多個模型的面向分析結果列表
            df: 原始數據框
            names: 模型名稱列表
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            模型比較結果
        """
        if len(model_results) < 2 or len(model_results) != len(names):
            return {"error": "比較至少需要兩個模型，且模型數量必須與名稱數量匹配"}
        
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("模型比較")
            logger = ConsoleOutputManager.setup_console_logger("model_comparison", log_file)
        else:
            logger = self.logger
        
        try:
            logger.info(f"開始比較 {len(names)} 個模型結果...")
            
            # 評估每個模型
            evaluations = []
            for i, (result, name) in enumerate(zip(model_results, names)):
                logger.info(f"評估模型 {i+1}: {name}...")
                evaluation = self.evaluate_aspects(df, result, console_output=False)
                evaluations.append({
                    'name': name,
                    'evaluation': evaluation
                })
            
            # 比較各個指標
            metrics_comparison = {}
            metric_names = set()
            
            # 收集所有指標名稱
            for eval_data in evaluations:
                metric_names.update(eval_data['evaluation']['metrics'].keys())
            
            # 構建比較表
            for metric in metric_names:
                metrics_comparison[metric] = []
                for eval_data in evaluations:
                    value = eval_data['evaluation']['metrics'].get(metric, None)
                    metrics_comparison[metric].append({
                        'name': eval_data['name'],
                        'value': value
                    })
            
            # 生成比較圖表
            visualizations = self._generate_comparison_charts(evaluations, names)
            
            # 構建返回結果
            comparison_result = {
                'models': names,
                'metrics_comparison': metrics_comparison,
                'visualizations': visualizations,
                'evaluations': evaluations
            }
            
            logger.info("模型比較完成")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"模型比較過程中發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise

# 使用示例
if __name__ == "__main__":
    # 初始化評估器
    evaluator = Evaluator()
    
    # 模擬數據和面向結果
    import numpy as np
    
    # 創建模擬數據框
    data = {
        'id': [f"text_{i}" for i in range(10)],
        'text': [f"Sample text {i}" for i in range(10)],
        'processed_text': [f"sample text {i}" for i in range(10)],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative',
                    'neutral', 'positive', 'negative', 'positive', 'neutral']
    }
    df = pd.DataFrame(data)
    
    # 創建模擬面向結果
    aspect_vectors = {
        0: {'vector': np.random.random(10), 'keywords': ['price', 'cost'], 'label': '價格'},
        1: {'vector': np.random.random(10), 'keywords': ['quality', 'good'], 'label': '質量'},
        2: {'vector': np.random.random(10), 'keywords': ['service', 'staff'], 'label': '服務'}
    }
    
    aspect_labels = {0: '價格', 1: '質量', 2: '服務'}
    
    text_aspect_matrix = np.random.random((10, 3))
    text_aspect_matrix = text_aspect_matrix / text_aspect_matrix.sum(axis=1, keepdims=True)
    
    id_to_index = {f"text_{i}": i for i in range(10)}
    
    aspect_results = {
        'aspect_vectors': aspect_vectors,
        'aspect_labels': aspect_labels,
        'text_aspect_matrix': text_aspect_matrix,
        'id_to_index': id_to_index
    }
    
    # 評估面向結果
    evaluation = evaluator.evaluate_aspects(df, aspect_results)
    
    # 輸出評估結果
    print(f"總體評分: {evaluation['metrics']['overall_score']:.2f}")
    print(f"面向獨特性: {evaluation['metrics']['aspect_uniqueness']:.2f}")
    print(f"覆蓋率: {evaluation['metrics']['coverage_ratio']:.2f}")
