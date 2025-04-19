"""
視覺化工具模組
此模組提供各種數據視覺化功能
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from gensim.models import LdaModel
import os
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import datetime

# 確保中文字體正確顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    """
    數據視覺化工具類
    提供各種視覺化方法
    """
    
    def __init__(self, output_dir: str = './Part03_/results/visualizations', style: str = 'seaborn-v0_8'):
        """
        初始化視覺化工具
        
        Args:
            output_dir: 視覺化圖像輸出目錄
            style: 視覺化風格 (可選：'seaborn-v0_8', 'ggplot', 'bmh', 'dark_background' 等)
        """
        self.output_dir = output_dir
        self.style = style
        
        # 設置風格
        try:
            plt.style.use(style)
        except (ValueError, IOError, OSError):
            self.logger.warning(f"無法使用指定風格 '{style}'，將使用默認風格")
            # 使用 seaborn 設置
            sns.set_theme()
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 設置日誌記錄
        self.logger = logging.getLogger('visualizer')
        self.logger.setLevel(logging.INFO)
        
        # 如果還沒有處理器，添加一個
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(level別名)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300) -> str:
        """
        保存圖像到文件
        
        Args:
            fig: matplotlib圖像對象
            filename: 文件名
            dpi: 圖像解析度
            
        Returns:
            保存的文件路徑
        """
        if not filename.endswith('.png') and not filename.endswith('.jpg') and not filename.endswith('.pdf'):
            filename += '.png'
        
        file_path = os.path.join(self.output_dir, filename)
        fig.savefig(file_path, dpi=dpi, bbox_inches='tight')
        self.logger.info(f"已保存圖像到: {file_path}")
        
        return file_path
    
    def plot_topic_distribution(self, topic_weights: List[float], topic_labels: Optional[List[str]] = None, 
                               title: str = "文檔主題分佈", save_as: Optional[str] = None) -> plt.Figure:
        """
        繪製文檔的主題分布圖
        
        Args:
            topic_weights: 主題權重列表
            topic_labels: 主題標籤列表（可選）
            title: 圖表標題
            save_as: 保存文件名（可選）
            
        Returns:
            生成的圖像對象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 準備標籤
        if not topic_labels:
            topic_labels = [f"主題 {i+1}" for i in range(len(topic_weights))]
        
        # 確保長度一致
        if len(topic_labels) != len(topic_weights):
            topic_labels = topic_labels[:len(topic_weights)] if len(topic_labels) > len(topic_weights) else \
                           topic_labels + [f"主題 {i+1}" for i in range(len(topic_labels), len(topic_weights))]
        
        # 排序數據（從大到小）
        sorted_indices = np.argsort(topic_weights)[::-1]
        sorted_weights = [topic_weights[i] for i in sorted_indices]
        sorted_labels = [topic_labels[i] for i in sorted_indices]
        
        # 創建水平條形圖
        bars = ax.barh(sorted_labels, sorted_weights, color=plt.cm.viridis(np.linspace(0, 1, len(topic_weights))))
        
        # 為每個條形添加數值標籤
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}',
                    ha='left', va='center')
        
        # 設置標題和標籤
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('主題權重')
        
        # 調整布局
        plt.tight_layout()
        
        # 保存圖像（如果需要）
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    def plot_topics_keywords(self, lda_model: LdaModel, feature_names: List[str], 
                           num_keywords: int = 10, title: str = "LDA主題關鍵詞", 
                           save_as: Optional[str] = None) -> plt.Figure:
        """
        繪製LDA主題模型的關鍵詞分布圖
        
        Args:
            lda_model: 訓練好的LDA模型
            feature_names: 特徵名稱列表（詞彙表）
            num_keywords: 每個主題顯示的關鍵詞數量
            title: 圖表標題
            save_as: 保存文件名（可選）
            
        Returns:
            生成的圖像對象
        """
        n_topics = lda_model.num_topics
        
        # 計算畫布高度（取決於關鍵詞數量和主題數量）
        fig_height = n_topics * 1.5 + 2
        fig, axes = plt.subplots(n_topics, 1, figsize=(10, fig_height), sharex=True)
        
        # 確保axes是數組
        if n_topics == 1:
            axes = [axes]
        
        # 獲取主題-詞分佈
        for i, ax in enumerate(axes):
            top_keywords_idx = lda_model.get_topic_terms(i, num_keywords)
            top_keywords = [(feature_names[idx], weight) for idx, weight in top_keywords_idx]
            
            # 排序關鍵詞（按權重降序）
            top_keywords.sort(key=lambda x: x[1], reverse=True)
            
            keywords = [kw for kw, _ in top_keywords]
            weights = [wt for _, wt in top_keywords]
            
            # 繪製水平條形圖
            ax.barh(keywords[::-1], weights[::-1], color=plt.cm.cool(i / n_topics))
            
            ax.set_title(f'主題 {i+1}', fontsize=12)
            ax.tick_params(axis='y', labelsize=10)
        
        plt.suptitle(title, fontsize=14, y=0.95)
        plt.xlabel('權重')
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 為標題留出空間
        
        # 保存圖像（如果需要）
        if save_as:
            self._save_figure(fig, save_as, dpi=300)
        
        return fig
    
    def plot_tsne_clusters(self, vectors: np.ndarray, labels: List[int], label_names: Optional[List[str]] = None,
                         title: str = "文本向量聚類可視化", figsize: Tuple[int, int] = (12, 10),
                         perplexity: int = 30, save_as: Optional[str] = None) -> plt.Figure:
        """
        使用t-SNE將高維向量可視化為2D聚類圖
        
        Args:
            vectors: 高維向量數組
            labels: 對應的標籤列表
            label_names: 標籤名稱（可選）
            title: 圖表標題
            figsize: 圖像尺寸
            perplexity: t-SNE的perplexity參數
            save_as: 保存文件名（可選）
            
        Returns:
            生成的圖像對象
        """
        # 檢查數據
        if len(vectors) != len(labels):
            raise ValueError(f"向量數量({len(vectors)})與標籤數量({len(labels)})不匹配")
        
        # 應用t-SNE降維
        self.logger.info("正在執行t-SNE降維...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced_vecs = tsne.fit_transform(vectors)
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 獲取唯一標籤
        unique_labels = sorted(set(labels))
        
        # 為每個標籤分配不同顏色
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # 繪製散點圖
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            label_text = label_names[i] if label_names and i < len(label_names) else f"類別 {label}"
            
            ax.scatter(
                reduced_vecs[mask, 0], reduced_vecs[mask, 1],
                c=[colors[i]], label=label_text,
                alpha=0.7, edgecolors='w', linewidths=0.5
            )
        
        # 添加圖例和標題
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('t-SNE維度 1')
        ax.set_ylabel('t-SNE維度 2')
        
        # 移除坐標軸刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        
        # 保存圖像（如果需要）
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    def plot_aspect_sentiment_distribution(self, aspects: List[str], 
                                         positive_counts: List[int], 
                                         negative_counts: List[int],
                                         neutral_counts: Optional[List[int]] = None,
                                         title: str = "面向情感分布",
                                         save_as: Optional[str] = None) -> plt.Figure:
        """
        繪製面向情感分布對比圖
        
        Args:
            aspects: 面向名稱列表
            positive_counts: 每個面向的正面評價數量
            negative_counts: 每個面向的負面評價數量
            neutral_counts: 每個面向的中性評價數量（可選）
            title: 圖表標題
            save_as: 保存文件名（可選）
            
        Returns:
            生成的圖像對象
        """
        # 檢查數據長度是否一致
        if len(aspects) != len(positive_counts) or len(aspects) != len(negative_counts):
            raise ValueError("面向名稱和計數列表長度不一致")
        
        if neutral_counts is not None and len(aspects) != len(neutral_counts):
            raise ValueError("中性計數列表長度與面向名稱不一致")
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 設置條形寬度
        bar_width = 0.25
        x = np.arange(len(aspects))
        
        # 繪製條形圖
        bars1 = ax.bar(x - bar_width, positive_counts, bar_width, label='正面', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x, negative_counts, bar_width, label='負面', color='#e74c3c', alpha=0.8)
        
        if neutral_counts:
            bars3 = ax.bar(x + bar_width, neutral_counts, bar_width, label='中性', color='#3498db', alpha=0.8)
        
        # 設置x軸標籤
        ax.set_xticks(x)
        ax.set_xticklabels(aspects, rotation=45, ha='right')
        
        # 添加數值標籤
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        add_labels(bars1)
        add_labels(bars2)
        if neutral_counts:
            add_labels(bars3)
        
        # 設置標題和標籤
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel('評價數量')
        ax.legend()
        
        plt.tight_layout()
        
        # 保存圖像（如果需要）
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    def plot_aspect_comparison(self, dataset_names: List[str], aspect_data: List[Dict[str, float]],
                             title: str = "不同數據集面向評價對比",
                             save_as: Optional[str] = None) -> plt.Figure:
        """
        繪製不同數據集的面向評價對比雷達圖
        
        Args:
            dataset_names: 數據集名稱列表
            aspect_data: 每個數據集的面向評分字典列表
            title: 圖表標題
            save_as: 保存文件名（可選）
            
        Returns:
            生成的圖像對象
        """
        # 獲取所有的面向
        all_aspects = set()
        for data in aspect_data:
            all_aspects.update(data.keys())
        
        all_aspects = sorted(list(all_aspects))
        
        # 角度計算
        angles = np.linspace(0, 2*np.pi, len(all_aspects), endpoint=False).tolist()
        angles += angles[:1]  # 閉合圖形
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        
        # 繪製每個數據集的雷達圖
        for i, (dataset, data) in enumerate(zip(dataset_names, aspect_data)):
            # 準備數據（按順序）
            values = [data.get(aspect, 0) for aspect in all_aspects]
            values += values[:1]  # 閉合圖形
            
            # 繪製雷達圖
            ax.plot(angles, values, linewidth=2, label=dataset)
            ax.fill(angles, values, alpha=0.1)
        
        # 設置雷達圖的標籤
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_aspects)
        
        # 設置y軸限制
        ax.set_ylim(0, max([max(data.values()) for data in aspect_data]) * 1.1)
        
        # 添加圖例和標題
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title(title, size=15)
        
        plt.tight_layout()
        
        # 保存圖像（如果需要）
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    def plot_sentiment_timeline(self, dates: List[str], positive_scores: List[float], 
                              negative_scores: List[float], neutral_scores: Optional[List[float]] = None,
                              title: str = "情感趨勢分析", save_as: Optional[str] = None) -> plt.Figure:
        """
        繪製隨時間變化的情感分數趨勢圖
        
        Args:
            dates: 日期列表
            positive_scores: 每個日期的正面情感分數
            negative_scores: 每個日期的負面情感分數
            neutral_scores: 每個日期的中性情感分數（可選）
            title: 圖表標題
            save_as: 保存文件名（可選）
            
        Returns:
            生成的圖像對象
        """
        # 創建圖形
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # 繪製折線圖
        ax.plot(dates, positive_scores, 'o-', color='#2ecc71', linewidth=2, label='正面')
        ax.plot(dates, negative_scores, 'o-', color='#e74c3c', linewidth=2, label='負面')
        
        if neutral_scores:
            ax.plot(dates, neutral_scores, 'o-', color='#3498db', linewidth=2, label='中性')
        
        # 設置x軸標籤
        if len(dates) > 10:
            # 如果日期太多，間隔顯示
            ax.set_xticks(np.arange(0, len(dates), len(dates) // 10))
            ax.set_xticklabels([dates[i] for i in range(0, len(dates), len(dates) // 10)], rotation=45, ha='right')
        else:
            ax.set_xticklabels(dates, rotation=45, ha='right')
        
        # 設置標題和標籤
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_ylabel('情感分數')
        ax.set_xlabel('日期')
        
        # 添加圖例
        ax.legend()
        
        # 添加網格線
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 保存圖像（如果需要）
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    def plot_wordcloud(self, word_freq: Dict[str, int], title: str = "詞頻雲圖",
                     max_words: int = 200, width: int = 800, height: int = 400,
                     save_as: Optional[str] = None) -> plt.Figure:
        """
        繪製詞頻雲圖
        
        Args:
            word_freq: 詞頻字典 {詞: 頻率}
            title: 圖表標題
            max_words: 最大顯示詞數
            width: 圖像寬度
            height: 圖像高度
            save_as: 保存文件名（可選）
            
        Returns:
            生成的圖像對象
        """
        try:
            from wordcloud import WordCloud
        except ImportError:
            self.logger.error("未安裝wordcloud庫，無法生成詞雲圖")
            raise ImportError("請安裝wordcloud庫: pip install wordcloud")
        
        # 創建詞雲生成器
        wc = WordCloud(
            font_path='C:/Windows/Fonts/msjh.ttc',  # 使用系統字體以支持中文
            width=width, 
            height=height,
            max_words=max_words,
            background_color='white',
            colormap='viridis',
            collocations=False  # 避免單詞重複出現
        )
        
        # 生成詞雲
        wc.generate_from_frequencies(word_freq)
        
        # 創建圖形
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        # 顯示詞雲
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=14)
        ax.axis('off')  # 隱藏坐標軸
        
        plt.tight_layout()
        
        # 保存圖像（如果需要）
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, classes: List[str], 
                            normalize: bool = False, title: str = '混淆矩陣',
                            cmap: Any = plt.cm.Blues, save_as: Optional[str] = None) -> plt.Figure:
        """
        繪製混淆矩陣
        
        Args:
            cm: 混淆矩陣
            classes: 類別名稱
            normalize: 是否歸一化
            title: 圖表標題
            cmap: 顏色映射
            save_as: 保存文件名（可選）
            
        Returns:
            生成的圖像對象
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        
        # 設置標題和標籤
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('預測標籤')
        ax.set_ylabel('實際標籤')
        
        # 設置刻度和標籤
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # 添加數值標籤
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        # 保存圖像（如果需要）
        if save_as:
            self._save_figure(fig, save_as)
        
        return fig
    
    def set_style(self, style: str) -> None:
        """
        設置繪圖風格
        
        Args:
            style: 風格名稱
        """
        self.style = style
        plt.style.use(style)
        self.logger.info(f"已設置繪圖風格為: {style}")

# 使用範例
if __name__ == "__main__":
    # 創建視覺化器
    viz = Visualizer()
    
    # 繪製主題分佈
    topic_weights = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
    topic_labels = ["價格", "質量", "服務", "包裝", "送貨", "其他"]
    fig1 = viz.plot_topic_distribution(topic_weights, topic_labels, "商品評論主題分佈", "topic_dist.png")
    
    # 繪製情感分布
    aspects = ["價格", "質量", "服務", "包裝", "送貨"]
    positive = [120, 85, 65, 45, 30]
    negative = [25, 35, 55, 15, 10]
    neutral = [40, 30, 25, 15, 20]
    fig2 = viz.plot_aspect_sentiment_distribution(aspects, positive, negative, neutral, 
                                               "商品評論面向情感分布", "sentiment_dist.png")
    
    print("已生成視覺化圖表")
