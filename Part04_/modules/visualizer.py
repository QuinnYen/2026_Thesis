"""
可視化模組 - 負責生成各種圖表和可視化
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import traceback  # 添加 traceback 模組用於詳細錯誤報告
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
        # 獲取當前檔案所在的Part04_目錄
        current_dir = os.path.dirname(os.path.abspath(__file__))
        part04_dir = os.path.dirname(current_dir)
        self.output_dir = self.config.get('output_dir', os.path.join(part04_dir, '1_output', 'visualizations'))
        self.dpi = self.config.get('dpi', 300)
        self.figsize = self.config.get('figsize', (12, 8))
        self.cmap = self.config.get('cmap', 'viridis')
        self.show_values = self.config.get('show_values', True)
        
        # 延遲創建輸出目錄，直到實際需要時才創建
        # os.makedirs(self.output_dir, exist_ok=True)
        
    # 添加新方法：檢查並清理面向向量
    def _clean_aspect_vectors(self, aspect_vectors):
        """檢查面向向量並移除空向量
        
        Args:
            aspect_vectors: 面向向量字典 {aspect_name: vector_data}
            
        Returns:
            dict: 清理後的面向向量字典
        """
        if not aspect_vectors:
            self.logger.warning("輸入的面向向量為空")
            return {}
            
        cleaned_vectors = {}
        for aspect, vector in aspect_vectors.items():
            # 檢查向量是否為空
            if vector and len(vector) > 0:
                cleaned_vectors[aspect] = vector
            else:
                self.logger.warning(f"檢測到空向量: {aspect}，已忽略")
                
        if len(cleaned_vectors) == 0:
            self.logger.error("所有向量均為空，無法繪製圖表")
            
        return cleaned_vectors

    def _normalize_metrics(self, metrics):
        """標準化評估指標格式
        
        Args:
            metrics: 評估指標字典
        
        Returns:
            dict: 標準化後的指標字典
        """
        if not metrics:
            self.logger.warning("提供的評估指標為空")
            return {}
        
        normalized_metrics = {}
        base_metrics = ['coherence', 'separation', 'combined_score']
        topic_level_metrics = ['topic_coherence', 'topic_separation']
        
        # 列印實際收到的評估指標供調試
        self.logger.info(f"原始評估指標: {metrics}")
        
        # 檢查是否有基本指標
        has_base_metrics = any(metric in metrics for metric in base_metrics)
        
        # 檢查是否有主題級別指標
        has_topic_metrics = any(metric in metrics for metric in topic_level_metrics)
        
        if has_base_metrics:
            # 使用基本指標
            for metric in base_metrics:
                if metric in metrics:
                    normalized_metrics[metric] = metrics[metric]
        elif has_topic_metrics:
            # 處理主題級別指標
            if 'topic_coherence' in metrics:
                topic_value = metrics['topic_coherence']
                if isinstance(topic_value, dict) and topic_value:
                    # 如果是字典，計算平均值
                    normalized_metrics['coherence'] = sum(topic_value.values()) / len(topic_value)
                    self.logger.info(f"從字典計算得到 coherence: {normalized_metrics['coherence']}")
                else:
                    # 如果是單一值，直接使用
                    normalized_metrics['coherence'] = float(topic_value)
                    self.logger.info(f"直接使用單一值 topic_coherence 作為 coherence: {topic_value}")
            
            if 'topic_separation' in metrics:
                topic_value = metrics['topic_separation']
                if isinstance(topic_value, dict) and topic_value:
                    # 如果是字典，計算平均值
                    normalized_metrics['separation'] = sum(topic_value.values()) / len(topic_value)
                    self.logger.info(f"從字典計算得到 separation: {normalized_metrics['separation']}")
                else:
                    # 如果是單一值，直接使用
                    normalized_metrics['separation'] = float(topic_value)
                    self.logger.info(f"直接使用單一值 topic_separation 作為 separation: {topic_value}")
            
            # 處理組合得分
            if 'combined_score' in metrics:
                normalized_metrics['combined_score'] = metrics['combined_score']
            elif 'coherence' in normalized_metrics and 'separation' in normalized_metrics:
                normalized_metrics['combined_score'] = (normalized_metrics['coherence'] + normalized_metrics['separation']) / 2
                self.logger.info(f"計算得到的 combined_score: {normalized_metrics['combined_score']}")
        
        # 如果標準化後沒有必需的指標，嘗試提取更多信息
        if not ('coherence' in normalized_metrics and 'separation' in normalized_metrics):
            self.logger.warning(f"標準化後缺少必要指標，嘗試其它方式提取")
            
            # 嘗試一些常見的替代鍵名
            alt_keys = {
                'coherence': ['topic_coherence_avg', 'avg_coherence', 'coherence_score'],
                'separation': ['topic_separation_avg', 'avg_separation', 'separation_score'],
                'combined_score': ['overall_score', 'total_score', 'final_score']
            }
            
            for target_key, alt_key_list in alt_keys.items():
                if target_key not in normalized_metrics:
                    for alt_key in alt_key_list:
                        if alt_key in metrics:
                            normalized_metrics[target_key] = metrics[alt_key]
                            self.logger.info(f"使用替代鍵 {alt_key} 作為 {target_key}: {metrics[alt_key]}")
                            break
        
        # 最終檢查結果
        if not ('coherence' in normalized_metrics and 'separation' in normalized_metrics):
            self.logger.error(f"標準化後仍缺少必要指標: 標準化結果={normalized_metrics}")
        else:
            self.logger.info(f"標準化後的指標: {normalized_metrics}")
            
        return normalized_metrics

    def plot_aspect_vectors_quality(self, aspect_vectors, metrics, title=None, output_filename=None):
        """繪製面向向量質量評估圖
        
        Args:
            aspect_vectors: 面向向量字典 {aspect_name: vector_data}
            metrics: 評估指標字典，包含 coherence, separation, combined_score 等指標
            title: 圖表標題
            output_filename: 輸出文件名
            
        Returns:
            str: 生成的圖片路徑
        """
        try:
            self.logger.info(f"生成面向向量質量評估圖")
            
            # 清理面向向量，移除空向量
            cleaned_vectors = self._clean_aspect_vectors(aspect_vectors)
            if not cleaned_vectors:
                raise ValueError("所有面向向量均為空，無法生成評估圖")
                
            # 檢查評估指標並進行格式轉換
            norm_metrics = self._normalize_metrics(metrics)
            if not norm_metrics or not ('coherence' in norm_metrics and 'separation' in norm_metrics):
                self.logger.error(f"提供的評估指標不完整: {metrics}")
                raise ValueError("缺少內容度或分離度數據，無法生成評估圖")
                
            # 提取指標數據
            coherence = norm_metrics.get('coherence')
            separation = norm_metrics.get('separation')
            combined_score = norm_metrics.get('combined_score', (coherence + separation) / 2)
            
            # 創建圖表：雙軸圖表
            fig, ax1 = plt.subplots(figsize=self.figsize)
            
            # 設置主軸（左側）- 評估指標
            ax1.set_xlabel('評估指標')
            ax1.set_ylabel('指標分數', color='tab:blue')
            
            # 繪製評估指標條形圖
            metrics_labels = ['內聚度', '分離度', '綜合得分']
            metrics_values = [coherence, separation, combined_score]
            x = np.arange(len(metrics_labels))
            
            bars = ax1.bar(x, metrics_values, width=0.4, color=['tab:blue', 'tab:orange', 'tab:green'])
            
            # 添加數值標籤
            if self.show_values:
                for bar in bars:
                    height = bar.get_height()
                    ax1.annotate(f'{height:.4f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),  # 3點垂直偏移
                             textcoords="offset points",
                             ha='center', va='bottom', fontsize=9)
            
            # 設置刻度和標籤
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics_labels)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # 設置Y軸範圍（假設評分在0到1之間）
            ax1.set_ylim(0, 1.05)
            
            # 創建第二個Y軸（右側）- 面向數量
            ax2 = ax1.twinx()
            ax2.set_ylabel('面向數量', color='tab:red')
            
            # 添加面向數量標記
            ax2.plot([-0.5, 2.5], [len(cleaned_vectors), len(cleaned_vectors)], 'r--', label=f'面向數量: {len(cleaned_vectors)}')
            ax2.text(1.5, len(cleaned_vectors) + 0.2, f'面向數量: {len(cleaned_vectors)}', 
                   color='tab:red', ha='center', fontsize=10)
            
            # 設置第二個Y軸的範圍
            ax2.set_ylim(0, max(10, len(cleaned_vectors) * 1.5))  # 確保有足夠的空間顯示標籤
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            # 添加標題
            if title:
                plt.title(title, fontsize=14)
            else:
                plt.title('面向向量質量評估', fontsize=14)
            
            # 添加網格線（僅針對左側Y軸）
            ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 添加面向名稱列表
            plt.figtext(0.02, 0.02, f"面向列表: {', '.join(cleaned_vectors.keys())}", 
                     wrap=True, fontsize=8, ha='left')
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"aspect_vectors_quality_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"面向向量質量評估圖已保存至: {output_path}")
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"面向向量質量評估圖已成功生成！\n\n保存路徑：\n{output_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"生成面向向量質量評估圖時出錯: {str(e)}")
            # 顯示錯誤通知
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showerror("錯誤", f"無法生成面向向量質量評估圖：{str(e)}")
                root.destroy()
            except Exception as msg_e:
                self.logger.error(f"顯示錯誤通知視窗時出錯: {str(msg_e)}")
                
            self.logger.error(traceback.format_exc())
            return None

    def update_config(self, config):
        """更新視覺化器配置
        
        Args:
            config: 新的配置參數字典
            
        Returns:
            None
        """
        if not config:
            self.logger.warning("收到空配置，未進行更新")
            return
            
        self.logger.info("更新視覺化器配置")
        try:
            # 更新配置字典
            self.config.update(config)
            
            # 更新各項設置
            self.output_dir = self.config.get('output_dir', self.output_dir)
            self.dpi = self.config.get('dpi', self.dpi)
            self.figsize = self.config.get('figsize', self.figsize)
            self.cmap = self.config.get('cmap', self.cmap)
            self.show_values = self.config.get('show_values', self.show_values)
            
            # 延遲創建輸出目錄，直到實際需要時才創建
            # os.makedirs(self.output_dir, exist_ok=True)
            
            self.logger.info(f"視覺化器配置已更新: output_dir={self.output_dir}, dpi={self.dpi}, "
                             f"figsize={self.figsize}, cmap={self.cmap}, show_values={self.show_values}")
        except Exception as e:
            self.logger.error(f"更新視覺化器配置時出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            
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
            self.logger.info(f"生成t-SNE視覺化: {title}")
            
            # 使用 scikit-learn 的 t-SNE 進行降維
            from sklearn.manifold import TSNE
            from datetime import datetime
            
            # 設置輸出目錄
            if figsize is None:
                figsize = self.figsize
                
            # 將輸入轉換為NumPy數組
            embeddings_array = np.array(embeddings)
            
            # 確保數據維度正確
            if len(embeddings_array.shape) == 1:
                self.logger.warning("輸入向量為一維，嘗試重塑為二維")
                # 如果是一維數組，重塑為一個樣本的二維數組
                embeddings_array = embeddings_array.reshape(1, -1)
            
            # 應用t-SNE降維
            tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings_array)-1), 
                  n_iter=n_iter, random_state=42)
            tsne_result = tsne.fit_transform(embeddings_array)
            
            # 創建圖表
            plt.figure(figsize=figsize)
            
            # 獲取唯一標籤並分配顏色
            unique_labels = list(set(labels))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            # 繪製散點圖
            for i, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       c=[colors[i]], label=label, s=point_size, alpha=alpha)
            
            # 添加標題和圖例
            if title:
                plt.title(title, fontsize=14)
            else:
                plt.title("t-SNE降維視覺化", fontsize=14)
                
            plt.legend(loc='best')
            
            # 去除坐標軸刻度
            plt.xticks([])
            plt.yticks([])
            
            # 添加網格
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存圖片
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"tsne_{timestamp}.png"
                
            output_path = os.path.join(self.output_dir, output_filename)
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.dpi)
            plt.close()
            
            self.logger.info(f"t-SNE視覺化已保存至: {output_path}")
            
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
            self.logger.error(f"t-SNE視覺化生成出錯: {str(e)}")
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
            
            # 轉換英文注意力機制名稱為中文
            chinese_attention_types = []
            for att_type in attention_types:
                if 'similarity' in att_type.lower():
                    chinese_attention_types.append("相似度注意力")
                elif 'keyword' in att_type.lower():
                    chinese_attention_types.append("關鍵詞注意力")
                elif 'self' in att_type.lower():
                    chinese_attention_types.append("自注意力")
                else:
                    chinese_attention_types.append(att_type)
            
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
                
            plt.xticks(x, chinese_attention_types)
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

    def create_evaluation_visualization(self, evaluation_results, show_all=True, show_chart=True, output_dir=None):
        """生成評估指標可視化
        
        Args:
            evaluation_results: 評估結果字典，格式可能是 
                              {'coherence': float, 'separation': float, 'combined_score': float} 或
                              {'topic_coherence': {...}, 'topic_separation': {...}, 'combined_score': float}
            show_all: 是否顯示所有指標
            show_chart: 是否顯示圖表
            output_dir: 輸出目錄
            
        Returns:
            tuple: (html_content, img_path, data_html)
        """
        try:
            self.logger.info("生成評估指標視覺化")
            
            # 如果提供了輸出目錄，則臨時更改輸出位置
            old_output_dir = None
            if output_dir:
                old_output_dir = self.output_dir
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)
            
            # 檢查評估結果數據格式
            if not evaluation_results:
                raise ValueError("評估結果為空")
                
            # 拆分評估結果，支持不同格式
            metrics_data = {}
            
            # 處理基礎指標
            base_metrics = ['coherence', 'separation', 'combined_score']
            topic_level_metrics = ['topic_coherence', 'topic_separation']
            
            # 檢查是否有基本指標
            has_base_metrics = any(metric in evaluation_results for metric in base_metrics)
            
            # 檢查是否有主題級別指標
            has_topic_metrics = any(metric in evaluation_results for metric in topic_level_metrics)
            
            # 根據格式提取數據
            if has_base_metrics:
                # 使用基本指標
                for metric in base_metrics:
                    if metric in evaluation_results:
                        metrics_data[metric] = evaluation_results[metric]
            elif has_topic_metrics:
                # 計算主題級別指標的平均值
                for metric in topic_level_metrics:
                    if metric in evaluation_results:
                        topic_values = evaluation_results[metric]
                        if isinstance(topic_values, dict):
                            # 計算平均值
                            metric_name = metric.replace('topic_', '')
                            metrics_data[metric_name] = sum(topic_values.values()) / len(topic_values)
                
                # 獲取組合得分
                if 'combined_score' in evaluation_results:
                    metrics_data['combined_score'] = evaluation_results['combined_score']
                elif 'coherence' in metrics_data and 'separation' in metrics_data:
                    # 如果沒有給定組合得分，但有內聚度和分離度，則計算簡單的組合得分
                    metrics_data['combined_score'] = (metrics_data['coherence'] + metrics_data['separation']) / 2
            
            # 如果仍然沒有數據，檢查其他可能的鍵名
            if not metrics_data:
                # 嘗試一些常見的替代鍵名
                alt_keys = {
                    'coherence': ['topic_coherence_avg', 'avg_coherence', 'coherence_score'],
                    'separation': ['topic_separation_avg', 'avg_separation', 'separation_score'],
                    'combined_score': ['overall_score', 'total_score', 'final_score']
                }
                
                for target_key, alt_key_list in alt_keys.items():
                    for alt_key in alt_key_list:
                        if alt_key in evaluation_results:
                            metrics_data[target_key] = evaluation_results[alt_key]
                            break
            
            # 確保數據包含至少一個指標
            if not metrics_data:
                raise ValueError("無法從評估結果中提取有效的指標數據")
                
            self.logger.info(f"提取的指標數據: {metrics_data}")
            
            # 生成數據表格
            import pandas as pd
            
            # 創建基本數據表格
            metrics_df = pd.DataFrame([metrics_data])
            
            # 視覺化處理
            img_path = None
            
            if show_chart and len(metrics_data) > 0:
                # 使用條形圖展示各指標
                try:
                    # 確保我們有數據可視化
                    data_to_viz = {k: v for k, v in metrics_data.items()}
                    
                    # 設置圖表標題
                    chart_title = '面向模型評估指標'
                    
                    # 在數據很少的情況下，可以使用極坐標圖更直觀
                    if len(data_to_viz) <= 3:
                        # 創建極坐標圖
                        fig = plt.figure(figsize=self.figsize)
                        ax = plt.subplot(111, polar=True)
                        
                        # 計算角度
                        categories = list(data_to_viz.keys())
                        N = len(categories)
                        angles = [n / float(N) * 2 * np.pi for n in range(N)]
                        angles += angles[:1]  # 閉合圖形
                        
                        # 提取數值
                        values = list(data_to_viz.values())
                        values += values[:1]  # 閉合數據
                        
                        # 繪製極坐標圖
                        ax.plot(angles, values, linewidth=2, linestyle='solid', label='評估指標')
                        ax.fill(angles, values, alpha=0.25)
                        
                        # 設置極坐標標籤
                        plt.xticks(angles[:-1], categories, fontsize=12)
                        
                        # 添加標題和圖例
                        plt.title(chart_title, size=15, y=1.08)
                        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                        
                    else:
                        # 標準條形圖
                        fig, ax = plt.subplots(figsize=self.figsize)
                        bars = ax.bar(data_to_viz.keys(), data_to_viz.values(), color='steelblue')
                        
                        # 添加數值標籤
                        for bar in bars:
                            height = bar.get_height()
                            ax.annotate(f'{height:.4f}',
                                      xy=(bar.get_x() + bar.get_width() / 2, height),
                                      xytext=(0, 3),  # 3點垂直偏移
                                      textcoords="offset points",
                                      ha='center', va='bottom', rotation=0)
                        
                        # 設置標題和標籤
                        plt.title(chart_title, fontsize=14)
                        plt.ylabel('評估得分', fontsize=12)
                        plt.xticks(rotation=15, ha='right')
                        plt.ylim(0, 1.0)  # 假設評分在0-1範圍內
                        
                    plt.tight_layout()
                    
                    # 保存圖像
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_filename = f"evaluation_metrics_{timestamp}.png"
                    img_path = os.path.join(self.output_dir, img_filename)
                    plt.savefig(img_path, dpi=self.dpi)
                    plt.close()
                    
                    self.logger.info(f"評估指標圖表已保存至: {img_path}")
                    
                except Exception as viz_error:
                    self.logger.error(f"生成評估指標圖表時出錯: {str(viz_error)}")
                    self.logger.error(traceback.format_exc())
            
            # 生成HTML內容
            html_content = "<h2>模型評估指標</h2>"
            
            if img_path:
                html_content += f"<img src='{img_path}' width='100%'>"
            
            # 生成數據表格HTML
            data_html = metrics_df.to_html(index=False, float_format=lambda x: f"{x:.5f}")
            
            # 完善HTML內容
            html_content += f"<div class='metrics-data'>{data_html}</div>"
            
            # 恢復原始輸出目錄
            if old_output_dir:
                self.output_dir = old_output_dir
            
            # 顯示成功通知視窗
            try:
                root = tk.Tk()
                root.withdraw()  # 隱藏主視窗
                messagebox.showinfo("輸出成功", f"評估指標視覺化已成功生成！\n\n保存路徑：\n{img_path}")
                root.destroy()
            except Exception as e:
                self.logger.error(f"顯示通知視窗時出錯: {str(e)}")
                
            return html_content, img_path, data_html
            
        except Exception as e:
            self.logger.error(f"生成評估指標視覺化時出錯: {str(e)}")
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