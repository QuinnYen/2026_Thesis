"""
可視化模組 - 負責生成各種圖表和可視化
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import logging
import time
from datetime import datetime
from matplotlib.font_manager import FontProperties
import tkinter as tk
from tkinter import messagebox

# 導入系統模組
from utils.logger import get_logger

# 獲取logger
logger = get_logger("visualizer")

# 設置中文字體支援
def set_chinese_font():
    """設置matplotlib支援中文字體"""
    # 設置負號顯示
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 導入必要的模組
    import matplotlib.font_manager as fm
    from pathlib import Path
    import platform
    
    # 根據不同操作系統設置默認字體路徑
    system = platform.system()
    font_found = False
    
    if system == 'Windows':
        # Windows系統中文字體位置
        potential_fonts = [
            r"C:\Windows\Fonts\simsun.ttc",               # 宋體
            r"C:\Windows\Fonts\simhei.ttf",               # 黑體
            r"C:\Windows\Fonts\msyh.ttc",                 # 微軟雅黑
            r"C:\Windows\Fonts\msjh.ttc",                 # 微軟正黑體
            r"C:\Windows\Fonts\simkai.ttf",               # 楷體
            r"C:\Windows\Fonts\DengXian.ttf",             # 等線體
            r"C:\Windows\Fonts\Deng.ttf",                 # 等線體
            r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc",  # Noto Sans CJK
        ]
    elif system == 'Darwin':  # macOS
        # macOS系統中文字體位置
        potential_fonts = [
            "/Library/Fonts/Songti.ttc",
            "/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
    else:  # Linux和其他系統
        # Linux系統中文字體位置
        potential_fonts = [
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        ]
    
    # 嘗試設置字體
    for font_path in potential_fonts:
        if Path(font_path).exists():
            logger.info(f"使用中文字體文件: {font_path}")
            prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [prop.get_name()] + plt.rcParams['font.sans-serif']
            font_found = True
            break
    
    # 如果找不到系統字體，嘗試使用matplotlib內建字體
    if not font_found:
        logger.warning("找不到系統中文字體，嘗試使用Matplotlib內建字體")
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK TC', 
                                          'Noto Sans CJK SC', 'Noto Sans CJK HK', 
                                          'Microsoft YaHei', 'Microsoft JhengHei', 
                                          'SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    
    # 確保字體設置生效
    plt.rcParams['axes.titlesize'] = 14  # 標題字體大小
    plt.rcParams['axes.labelsize'] = 12  # 軸標籤字體大小
    plt.rcParams['xtick.labelsize'] = 10  # x軸刻度標籤字體大小
    plt.rcParams['ytick.labelsize'] = 10  # y軸刻度標籤字體大小
    
    logger.info("中文字體設置完成")

# 設置中文字體
set_chinese_font()

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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"條形圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
            
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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"折線圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"雷達圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"熱力圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"3D散點圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"t-SNE視覺化已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"詞雲圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"面向向量比較圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
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
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"注意力權重熱力圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
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

    def create_topic_network(self, topics, vectors, show_weights=True, edge_threshold=0.3, interactive=True, output_dir=None):
        """生成主題關係網絡可視化
        
        Args:
            topics: 主題詞字典，格式 {topic_id: [word1, word2, ...]}
            vectors: 面向向量字典，格式 {topic_id: vector}
            show_weights: 是否顯示連接權重
            edge_threshold: 邊閾值，用於過濾弱連接
            interactive: 是否生成互動式圖表
            output_dir: 輸出目錄
            
        Returns:
            tuple: (html_content, img_path, data_html)
        """
        try:
            self.logger.info("生成主題關係網絡可視化")
            
            # 如果提供了輸出目錄，則臨時更改輸出位置
            old_output_dir = None
            if output_dir:
                old_output_dir = self.output_dir
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)
            
            # 準備數據
            topic_ids = list(vectors.keys())
            vectors_list = list(vectors.values())
            
            # 計算主題間的相似度矩陣
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectors_array = np.array(vectors_list)
            similarity_matrix = cosine_similarity(vectors_array)
            
            # 應用閾值過濾
            similarity_matrix[similarity_matrix < edge_threshold] = 0
            
            # 創建網絡數據
            import pandas as pd
            import networkx as nx
            
            # 創建圖
            G = nx.Graph()
            
            # 添加節點
            for i, topic_id in enumerate(topic_ids):
                # 獲取主題關鍵詞
                keywords = topics[topic_id][:3] if topic_id in topics else []
                keyword_text = ", ".join(keywords) if keywords else ""
                
                # 添加節點，包含關鍵詞信息
                G.add_node(
                    topic_id, 
                    keywords=keyword_text,
                    label=f"主題 {topic_id}"
                )
            
            # 添加邊
            edges_data = []
            for i in range(len(topic_ids)):
                for j in range(i+1, len(topic_ids)):
                    if similarity_matrix[i, j] > 0:
                        G.add_edge(
                            topic_ids[i], 
                            topic_ids[j], 
                            weight=similarity_matrix[i, j]
                        )
                        
                        # 收集邊數據用於表格顯示
                        edges_data.append({
                            "主題1": topic_ids[i],
                            "主題2": topic_ids[j],
                            "相似度": similarity_matrix[i, j]
                        })
            
            # 創建邊數據表格HTML
            edges_df = pd.DataFrame(edges_data)
            data_html = edges_df.to_html(index=False) if not edges_df.empty else "<p>沒有符合閾值的主題連接</p>"
            
            # 生成圖像和HTML內容
            html_content = ""
            img_path = None
            
            # 使用networkx和matplotlib創建靜態圖像
            plt.figure(figsize=self.figsize)
            
            # 計算節點位置 (使用spring_layout以獲得較好的可視化效果)
            pos = nx.spring_layout(G, seed=42)
            
            # 繪製節點
            nx.draw_networkx_nodes(
                G, pos,
                node_size=700,
                node_color='lightblue',
                alpha=0.8
            )
            
            # 繪製邊
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            if edge_weights:
                max_weight = max(edge_weights)
                normalized_weights = [w / max_weight * 5 for w in edge_weights]
                
                nx.draw_networkx_edges(
                    G, pos,
                    width=normalized_weights,
                    alpha=0.5,
                    edge_color='gray'
                )
                
                # 顯示邊權重
                if show_weights:
                    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
                    nx.draw_networkx_edge_labels(
                        G, pos,
                        edge_labels=edge_labels,
                        font_size=8
                    )
            
            # 繪製節點標籤
            labels = {node: f"主題 {node}" for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=10)
            
            # 設置標題和佈局
            plt.title("主題關係網絡", fontsize=15)
            plt.axis('off')
            plt.tight_layout()
            
            # 保存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"topic_network_{timestamp}.png"
            img_path = os.path.join(self.output_dir, img_filename)
            plt.savefig(img_path, dpi=self.dpi)
            plt.close()
            
            # 創建互動式視覺化
            if interactive:
                try:
                    import plotly.graph_objects as go
                    import networkx as nx
                    import numpy as np
                    
                    # 將networkx圖轉換為plotly格式
                    edge_x = []
                    edge_y = []
                    edge_text = []
                    
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        weight = G[edge[0]][edge[1]]['weight']
                        
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])
                        edge_text.append(f"相似度: {weight:.3f}")
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=0.8, color='#888'),
                        hoverinfo='text',
                        text=edge_text,
                        mode='lines'
                    )
                    
                    # 節點數據
                    node_x = []
                    node_y = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                    
                    # 節點懸停文本
                    node_text = []
                    for node in G.nodes():
                        keywords = G.nodes[node]['keywords']
                        text = f"主題 {node}<br>關鍵詞: {keywords}"
                        node_text.append(text)
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        hoverinfo='text',
                        text=[f"T{n}" for n in G.nodes()],
                        hovertext=node_text,
                        marker=dict(
                            showscale=True,
                            colorscale='YlGnBu',
                            size=20,
                            colorbar=dict(
                                thickness=15,
                                title='節點連接數',
                                xanchor='left',
                            ),
                            line=dict(width=2)
                        )
                    )
                    
                    # 連接數作為節點顏色
                    node_adjacencies = []
                    for node in G.nodes():
                        node_adjacencies.append(len(list(G.neighbors(node))))
                    
                    node_trace.marker.color = node_adjacencies
                    
                    # 創建圖形
                    fig = go.Figure(
                        data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="主題關係網絡",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                    )
                    
                    html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
                    
                except ImportError as e:
                    self.logger.warning(f"缺少生成互動式圖表的必要模組 ({str(e)})，僅生成靜態圖像")
                    html_content = f"<h2>主題關係網絡</h2><img src='{img_path}' width='100%'>"
                except Exception as e:
                    self.logger.error(f"生成互動式網絡圖時出錯: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    html_content = f"<h2>主題關係網絡</h2><img src='{img_path}' width='100%'>"
            else:
                # 僅使用靜態圖像
                html_content = f"<h2>主題關係網絡</h2><img src='{img_path}' width='100%'>"
            
            # 恢復原始輸出目錄
            if old_output_dir:
                self.output_dir = old_output_dir
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"主題關係網絡視覺化已成功生成！\n\n保存路徑：\n{img_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
            return html_content, img_path, data_html
            
        except Exception as e:
            self.logger.error(f"生成主題關係網絡視覺化時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"<h2>錯誤</h2><p>{str(e)}</p>", None, f"<h2>錯誤</h2><p>{str(e)}</p>"

    def create_attention_heatmap(self, attention_matrix, row_labels=None, col_labels=None, title="注意力權重熱力圖", 
                               output_filename=None, cmap="YlGnBu", annot=True, output_dir=None):
        """生成注意力權重熱力圖
        
        Args:
            attention_matrix: 注意力權重矩陣，形狀為 (n, m)
            row_labels: 行標籤，對應於熱力圖的Y軸
            col_labels: 列標籤，對應於熱力圖的X軸
            title: 熱力圖標題
            output_filename: 輸出檔案名稱
            cmap: 顏色映射
            annot: 是否在熱力圖上顯示數值
            output_dir: 輸出目錄
            
        Returns:
            tuple: (html_content, img_path, data_html)
        """
        try:
            self.logger.info(f"生成注意力熱力圖: {title}")
            
            # 如果提供了輸出目錄，則臨時更改輸出位置
            old_output_dir = None
            if output_dir:
                old_output_dir = self.output_dir
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)
            
            # 轉換為DataFrame以便更好地顯示標籤
            import pandas as pd
            if row_labels is None:
                row_labels = [f"行 {i+1}" for i in range(attention_matrix.shape[0])]
            if col_labels is None:
                col_labels = [f"列 {i+1}" for i in range(attention_matrix.shape[1])]
                
            df = pd.DataFrame(attention_matrix, index=row_labels, columns=col_labels)
            
            # 創建圖表
            plt.figure(figsize=self.figsize)
            
            # 繪製熱力圖
            ax = sns.heatmap(df, annot=annot, cmap=cmap, fmt=".3f", linewidths=0.5)
            
            # 調整標籤方向，避免重疊
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # 設置標題
            plt.title(title, fontsize=14, pad=20)
            
            # 調整布局，確保不裁剪標籤
            plt.tight_layout()
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"attention_heatmap_{timestamp}.png"
                
            img_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(img_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"注意力熱力圖已保存至: {img_path}")
            
            # 生成HTML內容
            html_content = f"<h2>{title}</h2><img src='{img_path}' width='100%'>"
            
            # 生成數據表格HTML
            data_html = df.to_html(classes="table table-bordered table-hover", 
                                  float_format="%.4f")
            
            # 恢復原始輸出目錄
            if old_output_dir:
                self.output_dir = old_output_dir
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"注意力熱力圖已成功生成！\n\n保存路徑：\n{img_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
            return html_content, img_path, data_html
            
        except Exception as e:
            self.logger.error(f"生成注意力熱力圖時出錯: {str(e)}")
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