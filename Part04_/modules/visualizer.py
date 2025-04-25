"""
可視化模組 - 負責生成各種圖表和可視化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import logging
import time
from datetime import datetime

# 導入系統模組
from utils.logger import get_logger

# 獲取logger
logger = get_logger("visualizer")

class Visualizer:
    """提供各種可視化功能的類"""
    
    def __init__(self, config=None):
        """初始化可視化器
        
        Args:
            config: 配置參數字典，可包含以下鍵:
                - output_dir: 輸出目錄
                - dpi: 圖像解析度
                - figsize: 圖像尺寸元組 (width, height)
                - cmap: 顏色映射
                - show_values: 是否在圖表中顯示數值
        """
        self.config = config or {}
        self.logger = logger
        
        # 設置默認配置
        self.output_dir = self.config.get('output_dir', './output/visualizations')
        self.dpi = self.config.get('dpi', 300)
        self.figsize = self.config.get('figsize', (12, 8))
        self.cmap = self.config.get('cmap', 'viridis')
        self.show_values = self.config.get('show_values', True)
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_bar_chart(self, data, x_column, y_columns, title, output_filename=None, 
                      xlabel=None, ylabel=None, rotate_xlabels=False):
        """繪製條形圖
        
        Args:
            data: pandas DataFrame 或 字典
            x_column: X軸列名
            y_columns: Y軸列名列表，可以是單個列名或多個列名
            title: 圖表標題
            output_filename: 輸出文件名
            xlabel: X軸標籤
            ylabel: Y軸標籤
            rotate_xlabels: 是否旋轉X軸標籤
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成條形圖: {title}")
            
            # 轉換為DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
                
            # 確保y_columns是列表
            if isinstance(y_columns, str):
                y_columns = [y_columns]
                
            # 創建圖表
            plt.figure(figsize=self.figsize)
            
            # 獲取X值
            x_values = df[x_column].values
            width = 0.8 / len(y_columns)  # 調整條形寬度
            
            # 繪製多個條形
            for i, col in enumerate(y_columns):
                positions = np.arange(len(x_values)) + (i - len(y_columns)/2 + 0.5) * width
                bars = plt.bar(positions, df[col].values, width=width, label=col)
                
                # 顯示數值
                if self.show_values:
                    for bar in bars:
                        height = bar.get_height()
                        plt.annotate(f'{height:.4f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3點垂直偏移
                                   textcoords="offset points",
                                   ha='center', va='bottom', rotation=0)
            
            # 設置標籤和標題
            plt.title(title, fontsize=14)
            plt.xlabel(xlabel or x_column, fontsize=12)
            plt.ylabel(ylabel or ', '.join(y_columns), fontsize=12)
            
            # 設置X軸刻度標籤
            plt.xticks(np.arange(len(x_values)), x_values)
            if rotate_xlabels:
                plt.xticks(rotation=45, ha='right')
                
            # 添加圖例
            if len(y_columns) > 1:
                plt.legend()
                
            plt.tight_layout()
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"bar_chart_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"條形圖已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成條形圖時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_line_chart(self, data, x_column, y_columns, title, output_filename=None,
                       xlabel=None, ylabel=None, markers=True, grid=True):
        """繪製折線圖
        
        Args:
            data: pandas DataFrame 或 字典
            x_column: X軸列名
            y_columns: Y軸列名列表，可以是單個列名或多個列名
            title: 圖表標題
            output_filename: 輸出文件名
            xlabel: X軸標籤
            ylabel: Y軸標籤
            markers: 是否在數據點顯示標記
            grid: 是否顯示網格
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成折線圖: {title}")
            
            # 轉換為DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
                
            # 確保y_columns是列表
            if isinstance(y_columns, str):
                y_columns = [y_columns]
                
            # 創建圖表
            plt.figure(figsize=self.figsize)
            
            # 繪製多條折線
            for col in y_columns:
                if markers:
                    plt.plot(df[x_column], df[col], marker='o', label=col)
                else:
                    plt.plot(df[x_column], df[col], label=col)
                    
                # 顯示數值
                if self.show_values:
                    for i, value in enumerate(df[col]):
                        plt.annotate(f'{value:.4f}',
                                   (df[x_column].iloc[i], value),
                                   textcoords="offset points",
                                   xytext=(0, 5),
                                   ha='center')
            
            # 設置標籤和標題
            plt.title(title, fontsize=14)
            plt.xlabel(xlabel or x_column, fontsize=12)
            plt.ylabel(ylabel or ', '.join(y_columns), fontsize=12)
            
            # 添加網格
            if grid:
                plt.grid(True, linestyle='--', alpha=0.7)
                
            # 添加圖例
            if len(y_columns) > 1:
                plt.legend()
                
            plt.tight_layout()
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"line_chart_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"折線圖已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成折線圖時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_radar_chart(self, data, categories, title, output_filename=None,
                        labels=None, fill=True):
        """繪製雷達圖
        
        Args:
            data: 數據字典 {label1: [values1], label2: [values2], ...}
            categories: 各個軸的類別名稱
            title: 圖表標題
            output_filename: 輸出文件名
            labels: 數據系列的標籤
            fill: 是否填充雷達圖
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成雷達圖: {title}")
            
            # 處理數據格式
            if labels is None:
                labels = list(data.keys())
            
            values_list = [data[label] for label in labels]
            
            # 計算角度
            N = len(categories)
            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            
            # 閉合雷達圖
            angles += angles[:1]
            
            # 創建圖表
            plt.figure(figsize=self.figsize)
            ax = plt.subplot(111, polar=True)
            
            # 調整角度方向和起始位置
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # 設置徑向網格線
            plt.grid(True)
            
            # 設置角度刻度
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # 繪製每個數據系列
            for i, values in enumerate(values_list):
                # 確保數據閉合
                values_closed = values + values[:1]
                
                # 繪製折線
                ax.plot(angles, values_closed, linewidth=2, label=labels[i])
                
                # 填充區域
                if fill:
                    ax.fill(angles, values_closed, alpha=0.1)
            
            # 添加標題和圖例
            plt.title(title, size=15)
            plt.legend(loc='upper right')
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"radar_chart_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"雷達圖已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成雷達圖時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_heatmap(self, data, title, output_filename=None, xlabel=None, ylabel=None,
                    annot=True, cmap=None, fmt='.4f'):
        """繪製熱力圖
        
        Args:
            data: pandas DataFrame 或二維數組
            title: 圖表標題
            output_filename: 輸出文件名
            xlabel: X軸標籤
            ylabel: Y軸標籤
            annot: 是否顯示數值標註
            cmap: 顏色映射
            fmt: 數值格式
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成熱力圖: {title}")
            
            # 轉換為DataFrame
            if not isinstance(data, pd.DataFrame):
                df = pd.DataFrame(data)
            else:
                df = data
                
            # 創建圖表
            plt.figure(figsize=self.figsize)
            
            # 使用自定義顏色映射或默認映射
            if cmap is None:
                cmap = self.cmap
                
            # 繪製熱力圖
            sns.heatmap(df, annot=annot, cmap=cmap, fmt=fmt, linewidths=0.5)
            
            # 設置標籤和標題
            plt.title(title, fontsize=14)
            if xlabel:
                plt.xlabel(xlabel, fontsize=12)
            if ylabel:
                plt.ylabel(ylabel, fontsize=12)
                
            plt.tight_layout()
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"heatmap_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"熱力圖已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成熱力圖時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_3d_scatter(self, data, x_column, y_column, z_column, color_column=None, 
                      title=None, output_filename=None, xlabel=None, ylabel=None, zlabel=None):
        """繪製3D散點圖
        
        Args:
            data: pandas DataFrame
            x_column: X軸列名
            y_column: Y軸列名
            z_column: Z軸列名
            color_column: 用於著色的列名
            title: 圖表標題
            output_filename: 輸出文件名
            xlabel: X軸標籤
            ylabel: Y軸標籤
            zlabel: Z軸標籤
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成3D散點圖")
            
            # 創建圖表
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # 獲取數據
            x = data[x_column].values
            y = data[y_column].values
            z = data[z_column].values
            
            # 處理顏色
            if color_column is not None:
                colors = data[color_column].values
                scatter = ax.scatter(x, y, z, c=colors, cmap=self.cmap, s=50, alpha=0.6)
                
                # 添加顏色條
                plt.colorbar(scatter, ax=ax, label=color_column)
            else:
                ax.scatter(x, y, z, s=50, alpha=0.6)
            
            # 設置軸標籤
            ax.set_xlabel(xlabel or x_column)
            ax.set_ylabel(ylabel or y_column)
            ax.set_zlabel(zlabel or z_column)
            
            # 設置標題
            if title:
                plt.title(title)
            else:
                plt.title(f"3D散點圖: {x_column} vs {y_column} vs {z_column}")
            
            # 添加網格線
            ax.grid(True)
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"3d_scatter_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"3D散點圖已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成3D散點圖時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_tsne(self, embeddings, labels, title=None, output_filename=None, perplexity=30,
                n_iter=1000, point_size=20, alpha=0.6, figsize=None):
        """使用t-SNE降維繪製高維向量
        
        Args:
            embeddings: 高維向量數組
            labels: 向量標籤
            title: 圖表標題
            output_filename: 輸出文件名
            perplexity: t-SNE困惑度參數
            n_iter: t-SNE迭代次數
            point_size: 散點大小
            alpha: 散點透明度
            figsize: 圖像尺寸
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成t-SNE可視化")
            
            # 確定困惑度參數不大於樣本數
            perplexity = min(perplexity, len(embeddings) - 1)
            
            # 使用t-SNE降維
            tsne = TSNE(n_components=2, perplexity=perplexity, 
                       n_iter=n_iter, random_state=42)
            tsne_result = tsne.fit_transform(embeddings)
            
            # 創建圖表
            if figsize is None:
                figsize = self.figsize
            plt.figure(figsize=figsize)
            
            # 獲取唯一標籤
            unique_labels = sorted(list(set(labels)))
            
            # 建立標籤與顏色的映射
            label_to_color = {label: i for i, label in enumerate(unique_labels)}
            
            # 獲取每個點的顏色
            colors = [label_to_color[label] for label in labels]
            
            # 繪製散點圖
            scatter = plt.scatter(
                tsne_result[:, 0], tsne_result[:, 1],
                c=colors,
                cmap=self.cmap,
                s=point_size,
                alpha=alpha
            )
            
            # 添加標題
            if title:
                plt.title(title, fontsize=14)
            else:
                plt.title("t-SNE可視化", fontsize=14)
                
            # 添加軸標籤
            plt.xlabel('t-SNE維度1', fontsize=12)
            plt.ylabel('t-SNE維度2', fontsize=12)
            
            # 添加圖例
            legend1 = plt.legend(
                *scatter.legend_elements(),
                loc="upper right",
                title="標籤"
            )
            plt.gca().add_artist(legend1)
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"tsne_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"t-SNE可視化已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成t-SNE可視化時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_wordcloud(self, word_freq, title=None, output_filename=None, max_words=200,
                     width=800, height=400, background_color='white'):
        """生成詞雲圖
        
        Args:
            word_freq: 詞頻字典 {word: frequency}
            title: 圖表標題
            output_filename: 輸出文件名
            max_words: 最大詞數
            width: 圖像寬度
            height: 圖像高度
            background_color: 背景顏色
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成詞雲圖")
            
            # 創建詞雲
            wordcloud = WordCloud(
                width=width,
                height=height,
                max_words=max_words,
                background_color=background_color,
                colormap=self.cmap,
                random_state=42
            ).generate_from_frequencies(word_freq)
            
            # 創建圖表
            plt.figure(figsize=self.figsize)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # 添加標題
            if title:
                plt.title(title, fontsize=14)
                
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"wordcloud_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"詞雲圖已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成詞雲圖時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_aspect_comparison(self, metrics_df, title=None, output_filename=None):
        """繪製面向向量比較圖
        
        Args:
            metrics_df: 包含不同注意力機制評估指標的DataFrame
            title: 圖表標題
            output_filename: 輸出文件名
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成面向向量比較圖")
            
            # 獲取注意力類型和指標
            attention_types = metrics_df['attention_type'].tolist()
            coherence_scores = metrics_df['coherence'].tolist()
            separation_scores = metrics_df['separation'].tolist()
            combined_scores = metrics_df['combined_score'].tolist()
            
            # 設置柱的位置
            x = np.arange(len(attention_types))
            width = 0.25
            
            # 創建圖表
            plt.figure(figsize=self.figsize)
            
            # 繪製三組柱狀圖
            bars1 = plt.bar(x - width, coherence_scores, width, label='內聚度')
            bars2 = plt.bar(x, separation_scores, width, label='分離度')
            bars3 = plt.bar(x + width, combined_scores, width, label='綜合得分')
            
            # 添加標籤、標題等
            plt.xlabel('注意力機制類型', fontsize=12)
            plt.ylabel('評估指標分數', fontsize=12)
            
            if title:
                plt.title(title, fontsize=14)
            else:
                plt.title('不同注意力機制的評估指標比較', fontsize=14)
                
            plt.xticks(x, attention_types)
            plt.legend()
            
            # 添加數值標籤
            if self.show_values:
                def add_labels(bars):
                    for bar in bars:
                        height = bar.get_height()
                        plt.annotate(f'{height:.4f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom', rotation=45, fontsize=9)
                
                add_labels(bars1)
                add_labels(bars2)
                add_labels(bars3)
            
            plt.tight_layout()
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"aspect_comparison_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"面向向量比較圖已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成面向向量比較圖時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def plot_attention_weights(self, weights, topics, title=None, output_filename=None):
        """繪製注意力權重熱力圖
        
        Args:
            weights: 注意力權重矩陣
            topics: 主題標籤列表
            title: 圖表標題
            output_filename: 輸出文件名
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成注意力權重熱力圖")
            
            # 創建圖表
            plt.figure(figsize=self.figsize)
            
            # 計算每個主題的權重分布
            df = pd.DataFrame(weights, index=topics)
            
            # 正規化每行（主題），使權重總和為1
            df = df.div(df.sum(axis=1), axis=0)
            
            # 繪製熱力圖
            sns.heatmap(df, cmap=self.cmap, annot=True, fmt=".3f")
            
            # 添加標籤
            plt.xlabel('文檔索引', fontsize=12)
            plt.ylabel('主題', fontsize=12)
            
            if title:
                plt.title(title, fontsize=14)
            else:
                plt.title('注意力權重分布', fontsize=14)
                
            plt.tight_layout()
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"attention_weights_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"注意力權重熱力圖已保存至: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成注意力權重熱力圖時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def create_visualization(self, vis_type, data, **kwargs):
        """創建指定類型的可視化
        
        Args:
            vis_type: 可視化類型，可以是'bar', 'line', 'radar', 'heatmap',
                     '3d', 'tsne', 'wordcloud', 'aspect_comparison', 'attention_weights'
            data: 可視化數據
            **kwargs: 其他參數，傳遞給對應的可視化函數
            
        Returns:
            str: 生成的圖片路徑
        """
        vis_funcs = {
            'bar': self.plot_bar_chart,
            'line': self.plot_line_chart,
            'radar': self.plot_radar_chart,
            'heatmap': self.plot_heatmap,
            '3d': self.plot_3d_scatter,
            'tsne': self.plot_tsne,
            'wordcloud': self.plot_wordcloud,
            'aspect_comparison': self.plot_aspect_comparison,
            'attention_weights': self.plot_attention_weights
        }
        
        if vis_type in vis_funcs:
            return vis_funcs[vis_type](data, **kwargs)
        else:
            self.logger.error(f"不支持的可視化類型: {vis_type}")
            return None

    def create_topic_distribution(self, topics, vectors, show_labels=True, interactive=True, use_3d=False, output_dir=None):
        """生成主題分佈可視化
        
        Args:
            topics: 主題詞字典，格式 {topic_id: [word1, word2, ...]}
            vectors: 面向向量字典，格式 {topic_id: vector}
            show_labels: 是否顯示主題標籤
            interactive: 是否生成互動式圖表
            use_3d: 是否使用3D視圖
            output_dir: 輸出目錄
            
        Returns:
            tuple: (html_content, img_path, data_html)
        """
        try:
            self.logger.info("生成主題分佈可視化")
            
            # 如果提供了輸出目錄，則臨時更改輸出位置
            old_output_dir = None
            if output_dir:
                old_output_dir = self.output_dir
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)
            
            # 準備數據
            vectors_list = list(vectors.values())
            topic_ids = list(vectors.keys())
            
            # 創建主題標籤
            if show_labels:
                # 從主題詞創建簡短標籤
                labels = []
                for topic_id in topic_ids:
                    if topic_id in topics:
                        # 獲取前3個關鍵詞作為標籤
                        keywords = topics[topic_id][:3]
                        label = f"主題 {topic_id}: {', '.join(keywords)}"
                    else:
                        label = f"主題 {topic_id}"
                    labels.append(label)
            else:
                labels = [f"主題 {t_id}" for t_id in topic_ids]
            
            # 創建數據表格HTML
            import pandas as pd
            df = pd.DataFrame({
                "主題ID": topic_ids,
                "標籤": labels,
                "向量維度": [len(vec) for vec in vectors_list]
            })
            
            data_html = df.to_html(index=False)
            
            # 將向量降維可視化
            html_content = ""
            img_path = None
            
            if use_3d and len(vectors_list) > 2:
                # 3D可視化需要至少3個向量
                try:
                    from sklearn.manifold import TSNE
                    import plotly.express as px
                    import numpy as np
                    
                    # 降維到3D
                    tsne_3d = TSNE(n_components=3, perplexity=min(30, len(vectors_list)-1), 
                             random_state=42, n_iter=1000)
                    vectors_array = np.array(vectors_list)
                    embeddings_3d = tsne_3d.fit_transform(vectors_array)
                    
                    if interactive:
                        # 創建互動式3D圖
                        df_3d = pd.DataFrame({
                            'x': embeddings_3d[:, 0],
                            'y': embeddings_3d[:, 1],
                            'z': embeddings_3d[:, 2],
                            'label': labels,
                            'topic_id': topic_ids
                        })
                        
                        fig = px.scatter_3d(
                            df_3d, x='x', y='y', z='z',
                            text='label', color='topic_id',
                            title='主題分佈 (3D)'
                        )
                        
                        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
                    
                    # 同時生成靜態圖像
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    
                    fig = plt.figure(figsize=self.figsize)
                    ax = fig.add_subplot(111, projection='3d')
                    
                    # 繪製散點圖
                    scatter = ax.scatter(
                        embeddings_3d[:, 0], 
                        embeddings_3d[:, 1], 
                        embeddings_3d[:, 2],
                        s=100, alpha=0.8
                    )
                    
                    # 添加標籤
                    if show_labels:
                        for i, label in enumerate(labels):
                            ax.text(
                                embeddings_3d[i, 0], 
                                embeddings_3d[i, 1], 
                                embeddings_3d[i, 2],
                                label, fontsize=8
                            )
                    
                    ax.set_title('主題分佈 (3D)')
                    
                    # 保存圖片
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_filename = f"topic_distribution_3d_{timestamp}.png"
                    img_path = os.path.join(self.output_dir, img_filename)
                    plt.savefig(img_path, dpi=self.dpi)
                    plt.close()
                    
                except Exception as e:
                    self.logger.warning(f"3D可視化生成失敗，回退到2D: {str(e)}")
                    use_3d = False
            
            if not use_3d or img_path is None:
                # 2D可視化
                try:
                    from sklearn.manifold import TSNE
                    import numpy as np
                    
                    # 降維到2D
                    tsne = TSNE(n_components=2, perplexity=min(30, len(vectors_list)-1), 
                           random_state=42, n_iter=1000)
                    vectors_array = np.array(vectors_list)
                    embeddings_2d = tsne.fit_transform(vectors_array)
                    
                    if interactive:
                        # 創建互動式2D圖
                        try:
                            import plotly.express as px
                            import pandas as pd
                            
                            df_2d = pd.DataFrame({
                                'x': embeddings_2d[:, 0],
                                'y': embeddings_2d[:, 1],
                                'label': labels,
                                'topic_id': topic_ids
                            })
                            
                            fig = px.scatter(
                                df_2d, x='x', y='y',
                                text='label', hover_data=['topic_id'],
                                title='主題分佈 (2D)'
                            )
                            
                            html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
                            
                        except ImportError:
                            self.logger.warning("未安裝plotly，無法生成互動式圖表")
                    
                    # 生成靜態圖像
                    import matplotlib.pyplot as plt
                    
                    plt.figure(figsize=self.figsize)
                    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.8)
                    
                    # 添加標籤
                    if show_labels:
                        for i, label in enumerate(labels):
                            plt.annotate(
                                label,
                                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                                fontsize=9,
                                alpha=0.8
                            )
                    
                    plt.title('主題分佈 (2D)')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    # 保存圖片
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_filename = f"topic_distribution_2d_{timestamp}.png"
                    img_path = os.path.join(self.output_dir, img_filename)
                    plt.savefig(img_path, dpi=self.dpi)
                    plt.close()
                    
                except Exception as e:
                    self.logger.error(f"2D可視化生成失敗: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
            
            # 如果互動式圖表生成失敗，但靜態圖片成功，則創建簡單HTML
            if not html_content and img_path:
                html_content = f"<h2>主題分佈</h2><img src='{img_path}' width='100%'>"
                
            # 恢復原始輸出目錄
            if old_output_dir:
                self.output_dir = old_output_dir
                
            return html_content, img_path, data_html
            
        except Exception as e:
            self.logger.error(f"生成主題分佈可視化時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"<h2>錯誤</h2><p>{str(e)}</p>", None, f"<h2>錯誤</h2><p>{str(e)}</p>"


# 測試代碼
if __name__ == "__main__":
    # 配置日誌級別
    logger.setLevel(logging.INFO)
    
    # 創建可視化器
    visualizer = Visualizer()
    
    # 測試條形圖
    data = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'value1': [10, 20, 15, 25],
        'value2': [12, 15, 18, 22]
    })
    
    visualizer.plot_bar_chart(
        data,
        x_column='category',
        y_columns=['value1', 'value2'],
        title='測試條形圖'
    )
    
    # 測試雷達圖
    categories = ['內聚度', '分離度', '準確率', '覆蓋率', '效率']
    data = {
        '算法A': [0.8, 0.6, 0.9, 0.7, 0.75],
        '算法B': [0.7, 0.8, 0.75, 0.8, 0.9]
    }
    
    visualizer.plot_radar_chart(
        data,
        categories=categories,
        title='算法性能雷達圖'
    )
    
    # 測試使用通用接口
    visualizer.create_visualization(
        'bar',
        data,
        x_column='category',
        y_columns=['value1', 'value2'],
        title='通用接口測試'
    )