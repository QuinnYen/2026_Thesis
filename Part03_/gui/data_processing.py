"""
數據處理標籤頁模組
此模組負責數據處理界面的邏輯
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any
import json
import datetime
import time

# 導入PyQt6模組
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, 
    QSpinBox, QLineEdit, QTextEdit, QFileDialog, QMessageBox, QProgressBar, 
    QGroupBox, QTableWidget, QTableWidgetItem, QSplitter, QTabWidget
)
from PyQt6.QtCore import Qt, QSize, QUrl, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QIcon, QFont, QPixmap

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 導入相關模組
from Part03_.utils.config_manager import ConfigManager
from Part03_.utils.result_manager import ResultManager
from Part03_.core.data_importer import DataImporter
from Part03_.utils.console_output import ConsoleOutputManager

class DataProcessingTab(QWidget):
    """數據處理標籤頁類"""
    
    def __init__(self, config: ConfigManager, result_manager: ResultManager):
        """初始化數據處理標籤頁"""
        super().__init__()
        
        self.config = config
        self.result_manager = result_manager
        self.data_importer = DataImporter(config=self.config)
        
        # 當前載入的結果ID和數據
        self.current_result_id = None
        self.current_data = None
        self.current_meta = None
        
        # 初始化UI
        self.init_ui()
    
    def init_ui(self):
        """初始化使用者界面"""
        # 主佈局
        main_layout = QVBoxLayout(self)
        
        # 創建上方控制面板
        control_panel = QGroupBox("數據控制")
        control_layout = QHBoxLayout(control_panel)
        
        # 載入數據按鈕
        load_data_btn = QPushButton("載入數據")
        load_data_btn.clicked.connect(self.load_data_file)
        control_layout.addWidget(load_data_btn)
        
        # 資料集選擇下拉選單
        dataset_label = QLabel("數據集:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["IMDB", "Yelp", "Amazon"])
        control_layout.addWidget(dataset_label)
        control_layout.addWidget(self.dataset_combo)
        
        # 抽樣設置
        sample_label = QLabel("樣本大小:")
        self.sample_size = QSpinBox()
        self.sample_size.setRange(100, 100000)
        self.sample_size.setValue(1000)
        self.sample_size.setSingleStep(100)
        control_layout.addWidget(sample_label)
        control_layout.addWidget(self.sample_size)
        
        # 刷新按鈕
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_view)
        control_layout.addWidget(refresh_btn)
        
        # 處理按鈕
        process_btn = QPushButton("預處理數據")
        process_btn.clicked.connect(self.preprocess_data)
        control_layout.addWidget(process_btn)
        
        control_layout.addStretch()
        
        # 創建數據TabWidget，包含數據表格、統計和視圖
        data_tabs = QTabWidget()
        
        # 數據表格標籤頁
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)
        self.data_table = QTableWidget()
        table_layout.addWidget(self.data_table)
        
        # 添加表格操作按鈕
        table_buttons = QHBoxLayout()
        export_csv_btn = QPushButton("導出CSV")
        export_csv_btn.clicked.connect(self.export_to_csv)
        table_buttons.addWidget(export_csv_btn)
        
        view_column_stats_btn = QPushButton("查看列統計")
        view_column_stats_btn.clicked.connect(self.view_column_statistics)
        table_buttons.addWidget(view_column_stats_btn)
        
        table_buttons.addStretch()
        table_layout.addLayout(table_buttons)
        
        # 統計資訊標籤頁
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        
        # 視覺化標籤頁
        visual_tab = QWidget()
        visual_layout = QVBoxLayout(visual_tab)
        
        # 視覺化控制面板
        visual_control = QHBoxLayout()
        
        plot_type_label = QLabel("圖表類型:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["長度分佈", "情感分佈", "詞雲", "詞頻"])
        visual_control.addWidget(plot_type_label)
        visual_control.addWidget(self.plot_type_combo)
        
        column_label = QLabel("數據列:")
        self.column_combo = QComboBox()
        visual_control.addWidget(column_label)
        visual_control.addWidget(self.column_combo)
        
        generate_plot_btn = QPushButton("生成圖表")
        generate_plot_btn.clicked.connect(self.generate_plot)
        visual_control.addWidget(generate_plot_btn)
        
        visual_control.addStretch()
        visual_layout.addLayout(visual_control)
        
        # 視覺化顯示區域
        self.plot_area = QLabel("尚無圖表")
        self.plot_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_area.setMinimumHeight(400)
        self.plot_area.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        visual_layout.addWidget(self.plot_area)
        
        # 添加標籤頁到TabWidget
        data_tabs.addTab(table_tab, "數據表格")
        data_tabs.addTab(stats_tab, "統計資訊")
        data_tabs.addTab(visual_tab, "視覺化")
        
        # 添加日誌輸出區域
        log_group = QGroupBox("處理日誌")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # 添加進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        
        # 將元件添加到主佈局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(data_tabs, 3)
        main_layout.addWidget(log_group, 1)
        main_layout.addWidget(self.progress_bar)
    
    def load_data_file(self):
        """載入數據文件"""
        # 選擇數據文件
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇數據文件", "", "CSV文件 (*.csv);;JSON文件 (*.json);;所有文件 (*)")
        
        if not file_path:
            return
        
        try:
            self.log_message(f"開始載入檔案: {file_path}")
            self.progress_bar.setValue(10)
            
            # 根據文件類型載入數據
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                self.log_message(f"成功載入CSV檔案，共 {len(df)} 行 {len(df.columns)} 列")
            elif file_ext == '.json':
                # 根據數據集類型進行載入
                dataset_type = self.dataset_combo.currentText().lower()
                if dataset_type == 'yelp':
                    # 載入Yelp數據集
                    self.log_message("正在載入Yelp數據...")
                    df = self.data_importer.load_yelp_data(file_path, sample_size=self.sample_size.value())
                else:
                    # 一般JSON載入
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        QMessageBox.warning(self, "警告", "JSON格式不支援或不是列表格式。")
                        return
                
                self.log_message(f"成功載入JSON檔案，共 {len(df)} 行 {len(df.columns)} 列")
            else:
                QMessageBox.warning(self, "警告", "不支援的檔案類型。")
                return
            
            # 顯示數據
            self.current_data = df
            self.display_data(df)
            self.compute_statistics(df)
            self.update_column_combo()
            
            # 恢復進度條
            self.progress_bar.setValue(100)
            self.log_message("數據載入完成。")
            
        except Exception as e:
            self.log_message(f"載入檔案時發生錯誤: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"載入檔案時發生錯誤: {str(e)}")
            self.progress_bar.setValue(0)
    
    def load_result(self, result_id: str):
        """載入特定結果ID的數據"""
        try:
            self.log_message(f"正在載入結果 {result_id} 的數據...")
            self.progress_bar.setValue(10)
            
            # 從結果管理器獲取數據和元數據
            result_set = self.result_manager.get_result_set(result_id)
            if not result_set:
                self.log_message(f"找不到結果 {result_id}")
                return
            
            # 檢查是否有處理好的數據
            if 'files' in result_set and 'processed_data' in result_set['files']:
                # 獲取最新的數據文件信息
                data_files = result_set['files']['processed_data']
                if isinstance(data_files, list) and data_files:
                    # 排序並獲取最新的文件
                    data_file_info = sorted(data_files, key=lambda x: x.get('created_at', ''), reverse=True)[0]
                    data_file = data_file_info['path']
                else:
                    # 保持向後兼容
                    data_file = data_files
                
                if os.path.exists(data_file):
                    # 根據文件類型載入
                    file_ext = os.path.splitext(data_file)[1].lower()
                    if file_ext == '.csv':
                        df = pd.read_csv(data_file)
                    elif file_ext == '.json':
                        df = pd.read_json(data_file, orient='records', lines=True)
                    elif file_ext == '.pickle' or file_ext == '.pkl':
                        df = pd.read_pickle(data_file)
                    else:
                        self.log_message(f"不支援的檔案類型: {file_ext}")
                        return
                    
                    # 獲取元數據
                    meta = {}
                    if 'metadata' in result_set.get('files', {}):
                        meta_files = result_set['files']['metadata']
                        if isinstance(meta_files, list) and meta_files:
                            meta_file_info = sorted(meta_files, key=lambda x: x.get('created_at', ''), reverse=True)[0]
                            meta_file = meta_file_info['path']
                            if os.path.exists(meta_file):
                                try:
                                    with open(meta_file, 'r', encoding='utf-8') as f:
                                        meta = json.load(f)
                                except Exception as e:
                                    self.log_message(f"無法載入元數據文件：{str(e)}")
                    
                    # 更新數據和元數據
                    self.current_result_id = result_id
                    self.current_data = df
                    self.current_meta = meta
                    
                    # 顯示數據
                    self.display_data(df)
                    self.compute_statistics(df)
                    self.update_column_combo()
                    
                    self.progress_bar.setValue(100)
                    self.log_message(f"成功載入結果 {result_id} 的數據，共 {len(df)} 行 {len(df.columns)} 列")
                else:
                    self.log_message(f"數據檔案不存在: {data_file}")
            else:
                self.log_message(f"結果 {result_id} 沒有可用的處理數據")
                
        except Exception as e:
            self.log_message(f"載入結果數據時發生錯誤: {str(e)}")
            self.progress_bar.setValue(0)
    
    def display_data(self, df: pd.DataFrame):
        """在表格中顯示數據"""
        # 清空表格
        self.data_table.clear()
        
        # 設置行數和列數
        row_count = min(1000, len(df))  # 限制最多顯示1000行，避免過度消耗記憶體
        self.data_table.setRowCount(row_count)
        self.data_table.setColumnCount(len(df.columns))
        
        # 設置表頭
        self.data_table.setHorizontalHeaderLabels(df.columns)
        
        # 填充數據
        for row in range(row_count):
            for col, column_name in enumerate(df.columns):
                value = str(df.iloc[row, col])
                item = QTableWidgetItem(value)
                self.data_table.setItem(row, col, item)
        
        # 調整列寬
        self.data_table.resizeColumnsToContents()
    
    def compute_statistics(self, df: pd.DataFrame):
        """計算並顯示數據統計資訊"""
        stats_text = "數據統計摘要:\n\n"
        
        # 基本統計信息
        stats_text += f"總行數: {len(df)}\n"
        stats_text += f"總列數: {len(df.columns)}\n\n"
        
        # 計算各列統計信息
        stats_text += "列統計:\n"
        for column in df.columns:
            stats_text += f"- {column}:\n"
            
            # 檢查是否為數值列
            if pd.api.types.is_numeric_dtype(df[column]):
                # 數值統計
                stats_text += f"  類型: 數值\n"
                stats_text += f"  平均值: {df[column].mean():.2f}\n"
                stats_text += f"  中位數: {df[column].median():.2f}\n"
                stats_text += f"  標準差: {df[column].std():.2f}\n"
                stats_text += f"  最小值: {df[column].min():.2f}\n"
                stats_text += f"  最大值: {df[column].max():.2f}\n"
                stats_text += f"  空值數: {df[column].isna().sum()}\n"
            elif pd.api.types.is_string_dtype(df[column]):
                # 字符串統計
                stats_text += f"  類型: 文本\n"
                
                # 字符串長度統計
                if not df[column].isna().all():
                    len_mean = df[column].str.len().mean()
                    len_min = df[column].str.len().min()
                    len_max = df[column].str.len().max()
                    stats_text += f"  平均長度: {len_mean:.2f}\n"
                    stats_text += f"  最小長度: {len_min}\n"
                    stats_text += f"  最大長度: {len_max}\n"
                
                # 唯一值數量
                unique_count = df[column].nunique()
                stats_text += f"  唯一值數量: {unique_count}\n"
                stats_text += f"  空值數: {df[column].isna().sum()}\n"
                
                # 如果唯一值較少，顯示出現頻率最高的幾個值
                if 1 < unique_count < 20:
                    top_values = df[column].value_counts().head(5)
                    stats_text += "  最常見值:\n"
                    for val, count in top_values.items():
                        stats_text += f"    {val}: {count} 次\n"
            else:
                # 其他類型
                stats_text += f"  類型: {df[column].dtype}\n"
                stats_text += f"  唯一值數量: {df[column].nunique()}\n"
                stats_text += f"  空值數: {df[column].isna().sum()}\n"
            
            stats_text += "\n"
        
        # 設置統計文本
        self.stats_text.setText(stats_text)
    
    def update_column_combo(self):
        """更新列選擇下拉選單"""
        if self.current_data is None:
            return
        
        # 保存當前選擇的列
        current_column = self.column_combo.currentText()
        
        # 清空並重新填充列選擇
        self.column_combo.clear()
        self.column_combo.addItems(self.current_data.columns)
        
        # 恢復之前的選擇
        if current_column:
            index = self.column_combo.findText(current_column)
            if index >= 0:
                self.column_combo.setCurrentIndex(index)
    
    def refresh_view(self):
        """刷新數據視圖"""
        if self.current_data is not None:
            self.display_data(self.current_data)
    
    def view_column_statistics(self):
        """檢視選中列的統計資訊"""
        if self.current_data is None:
            QMessageBox.information(self, "提示", "請先載入數據")
            return
        
        # 獲取選中的列
        selected_items = self.data_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "請先選擇一個數據列")
            return
        
        # 獲取選中的列名
        col_index = selected_items[0].column()
        if col_index < 0 or col_index >= len(self.current_data.columns):
            return
        
        column_name = self.current_data.columns[col_index]
        column_data = self.current_data[column_name]
        
        # 統計資訊
        stats_msg = f"列 '{column_name}' 的統計信息:\n\n"
        
        if pd.api.types.is_numeric_dtype(column_data):
            stats_msg += f"數據類型: 數值 ({column_data.dtype})\n"
            stats_msg += f"計數: {column_data.count()}\n"
            stats_msg += f"平均值: {column_data.mean():.4f}\n"
            stats_msg += f"標準差: {column_data.std():.4f}\n"
            stats_msg += f"最小值: {column_data.min():.4f}\n"
            stats_msg += f"25% 分位數: {column_data.quantile(0.25):.4f}\n"
            stats_msg += f"中位數: {column_data.median():.4f}\n"
            stats_msg += f"75% 分位數: {column_data.quantile(0.75):.4f}\n"
            stats_msg += f"最大值: {column_data.max():.4f}\n"
            stats_msg += f"空值數量: {column_data.isna().sum()}\n"
        elif pd.api.types.is_string_dtype(column_data):
            stats_msg += f"數據類型: 文本 ({column_data.dtype})\n"
            stats_msg += f"計數: {column_data.count()}\n"
            stats_msg += f"唯一值數量: {column_data.nunique()}\n"
            stats_msg += f"空值數量: {column_data.isna().sum()}\n"
            
            # 計算文本長度統計
            if not column_data.isna().all():
                length_stats = column_data.str.len().describe()
                stats_msg += f"\n文本長度統計:\n"
                stats_msg += f"平均長度: {length_stats['mean']:.2f}\n"
                stats_msg += f"最小長度: {length_stats['min']:.0f}\n"
                stats_msg += f"25% 分位數: {length_stats['25%']:.0f}\n"
                stats_msg += f"中位長度: {length_stats['50%']:.0f}\n"
                stats_msg += f"75% 分位數: {length_stats['75%']:.0f}\n"
                stats_msg += f"最大長度: {length_stats['max']:.0f}\n"
            
            # 顯示最常見的值
            top_values = column_data.value_counts().head(10)
            if len(top_values) > 0:
                stats_msg += f"\n最常見的值 (前10項):\n"
                for val, count in top_values.items():
                    # 截斷過長的值
                    display_val = val if len(str(val)) < 50 else str(val)[:47] + "..."
                    stats_msg += f"{display_val}: {count} 次 ({count/len(column_data)*100:.2f}%)\n"
        else:
            stats_msg += f"數據類型: {column_data.dtype}\n"
            stats_msg += f"計數: {column_data.count()}\n"
            stats_msg += f"唯一值數量: {column_data.nunique()}\n"
            stats_msg += f"空值數量: {column_data.isna().sum()}\n"
        
        # 顯示統計資訊
        QMessageBox.information(self, f"列統計 - {column_name}", stats_msg)
    
    def generate_plot(self):
        """生成選定類型的視覺化圖表"""
        if self.current_data is None:
            QMessageBox.information(self, "提示", "請先載入數據")
            return
        
        plot_type = self.plot_type_combo.currentText()
        column = self.column_combo.currentText()
        
        if not column:
            QMessageBox.information(self, "提示", "請選擇一個數據列")
            return
        
        try:
            self.log_message(f"正在生成 {plot_type} 圖表，基於列 '{column}'...")
            
            # 創建臨時檔案路徑用於保存圖片
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"plot_{int(time.time())}.png")
            
            plt.figure(figsize=(10, 6))
            
            # 根據圖表類型生成不同的可視化
            if plot_type == "長度分佈":
                if pd.api.types.is_string_dtype(self.current_data[column]):
                    # 計算文本長度
                    length_series = self.current_data[column].str.len()
                    length_series = length_series[~length_series.isna()]
                    
                    # 繪製直方圖
                    plt.hist(length_series, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.xlabel('文本長度')
                    plt.ylabel('頻率')
                    plt.title(f'{column} 的長度分佈')
                    plt.grid(True, alpha=0.3)
                else:
                    self.log_message("此列不是文本類型，無法計算長度分佈")
                    return
                
            elif plot_type == "情感分佈":
                if 'sentiment' in self.current_data.columns:
                    # 統計情感標籤
                    sentiment_counts = self.current_data['sentiment'].value_counts()
                    
                    # 繪製條形圖
                    sentiment_counts.plot(kind='bar', color=['red', 'gray', 'green'])
                    plt.xlabel('情感')
                    plt.ylabel('計數')
                    plt.title('情感分佈')
                    plt.xticks(rotation=0)
                    plt.grid(True, axis='y', alpha=0.3)
                else:
                    self.log_message("數據中沒有情感列")
                    return
                
            elif plot_type == "詞雲":
                if pd.api.types.is_string_dtype(self.current_data[column]):
                    try:
                        from wordcloud import WordCloud
                        import jieba
                        
                        # 合併所有文本
                        text = ' '.join(self.current_data[column].astype(str).fillna(''))
                        
                        # 檢查是否含有中文
                        has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in text)
                        
                        if has_chinese:
                            # 使用結巴進行中文分詞
                            words = ' '.join(jieba.cut(text))
                            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                               font_path='C:/Windows/Fonts/msjh.ttc').generate(words)
                        else:
                            # 英文直接生成
                            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                        
                        plt.imshow(wordcloud, interpolation='bilinear')
                        plt.axis('off')
                        plt.title(f'{column} 詞雲')
                    except ImportError:
                        self.log_message("無法生成詞雲，請確保已安裝 wordcloud 和 jieba 套件")
                        return
                else:
                    self.log_message("此列不是文本類型，無法生成詞雲")
                    return
                
            elif plot_type == "詞頻":
                if pd.api.types.is_string_dtype(self.current_data[column]):
                    try:
                        import jieba
                        from collections import Counter
                        
                        # 合併所有文本
                        text = ' '.join(self.current_data[column].astype(str).fillna(''))
                        
                        # 檢查是否含有中文
                        has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in text)
                        
                        if has_chinese:
                            # 中文分詞
                            words = list(jieba.cut(text))
                        else:
                            # 英文分詞
                            words = text.lower().split()
                        
                        # 過濾掉空白和標點符號
                        words = [word.strip() for word in words if len(word.strip()) > 1]
                        
                        # 計算詞頻
                        word_counts = Counter(words).most_common(20)
                        
                        # 繪製圖表
                        words, counts = zip(*word_counts)
                        plt.barh(words, counts, color='skyblue')
                        plt.xlabel('出現次數')
                        plt.ylabel('詞彙')
                        plt.title(f'{column} 中最常見的20個詞彙')
                        plt.tight_layout()
                    except ImportError:
                        self.log_message("無法生成詞頻圖，請確保已安裝 jieba 套件")
                        return
                else:
                    self.log_message("此列不是文本類型，無法生成詞頻圖")
                    return
            
            # 保存圖片
            plt.tight_layout()
            plt.savefig(temp_file, dpi=100)
            plt.close()
            
            # 在GUI中顯示圖片
            pixmap = QPixmap(temp_file)
            self.plot_area.setPixmap(pixmap.scaled(
                self.plot_area.width(), 
                self.plot_area.height(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            self.log_message(f"{plot_type} 圖表生成完成")
            
        except Exception as e:
            self.log_message(f"生成圖表時發生錯誤: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"生成圖表時發生錯誤: {str(e)}")
    
    def preprocess_data(self):
        """對數據進行預處理"""
        if self.current_data is None:
            QMessageBox.information(self, "提示", "請先載入數據")
            return
        
        # 創建預處理線程
        preprocess_thread = PreprocessingThread(self.current_data, self.config)
        
        # 連接信號
        preprocess_thread.progress_update.connect(self.update_progress)
        preprocess_thread.log_message.connect(self.log_message)
        preprocess_thread.processing_finished.connect(self.on_preprocessing_complete)
        
        # 啟動線程
        self.log_message("開始預處理數據...")
        preprocess_thread.start()
    
    def update_progress(self, value):
        """更新進度條"""
        self.progress_bar.setValue(value)
    
    def log_message(self, message):
        """添加日誌消息"""
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{current_time}] {message}")
        
        # 滾動到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def on_preprocessing_complete(self, result):
        """預處理完成後的回調"""
        if result['success']:
            # 更新數據
            self.current_data = result['data']
            self.display_data(self.current_data)
            self.compute_statistics(self.current_data)
            self.update_column_combo()
            
            self.log_message("數據預處理完成")
            QMessageBox.information(self, "完成", "數據預處理已完成")
        else:
            self.log_message(f"數據預處理失敗: {result['error']}")
            QMessageBox.critical(self, "錯誤", f"數據預處理失敗: {result['error']}")
    
    def export_to_csv(self):
        """將當前數據導出為CSV文件"""
        if self.current_data is None:
            QMessageBox.information(self, "提示", "沒有可導出的數據")
            return
        
        # 選擇保存路徑
        file_path, _ = QFileDialog.getSaveFileName(
            self, "導出CSV", "", "CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if file_path:
            try:
                self.current_data.to_csv(file_path, index=False)
                self.log_message(f"數據已導出到: {file_path}")
                QMessageBox.information(self, "成功", f"數據已成功導出到: {file_path}")
            except Exception as e:
                self.log_message(f"導出CSV時發生錯誤: {str(e)}")
                QMessageBox.critical(self, "錯誤", f"導出CSV時發生錯誤: {str(e)}")


class PreprocessingThread(QThread):
    """數據預處理線程"""
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    processing_finished = pyqtSignal(dict)
    
    def __init__(self, data: pd.DataFrame, config: ConfigManager):
        """初始化預處理線程"""
        super().__init__()
        self.data = data.copy()
        self.config = config
    
    def run(self):
        """執行預處理"""
        try:
            # 發送初始進度
            self.progress_update.emit(0)
            self.log_message.emit("開始數據預處理...")
            
            # 檢查數據是否為空
            if self.data is None or len(self.data) == 0:
                self.processing_finished.emit({
                    'success': False,
                    'error': "數據為空，無法進行預處理"
                })
                return
            
            # 預處理步驟1: 移除空值行
            self.progress_update.emit(10)
            self.log_message.emit("步驟1: 移除空值行...")
            
            # 檢查文本列
            text_columns = [col for col in self.data.columns if pd.api.types.is_string_dtype(self.data[col])]
            
            if 'text' in text_columns:
                primary_text_col = 'text'
            elif 'review' in text_columns:
                primary_text_col = 'review'
            elif len(text_columns) > 0:
                primary_text_col = text_columns[0]
            else:
                self.log_message.emit("警告: 找不到文本列，無法繼續處理")
                self.processing_finished.emit({
                    'success': False,
                    'error': "找不到文本列，無法繼續處理"
                })
                return
            
            # 移除文本列為空的行
            initial_rows = len(self.data)
            self.data = self.data[~self.data[primary_text_col].isna()]
            self.data = self.data[self.data[primary_text_col].str.strip() != '']
            removed_rows = initial_rows - len(self.data)
            self.log_message.emit(f"移除了 {removed_rows} 行空值或空白文本，剩餘 {len(self.data)} 行")
            
            # 預處理步驟2: 移除重複行
            self.progress_update.emit(20)
            self.log_message.emit("步驟2: 移除重複行...")
            
            # 根據文本列移除重複
            initial_rows = len(self.data)
            self.data = self.data.drop_duplicates(subset=[primary_text_col])
            removed_dups = initial_rows - len(self.data)
            self.log_message.emit(f"移除了 {removed_dups} 行重複文本，剩餘 {len(self.data)} 行")
            
            # 預處理步驟3: 文本清理
            self.progress_update.emit(30)
            self.log_message.emit("步驟3: 文本清理...")
            
            # 添加處理後的文本列
            self.data['processed_text'] = self.data[primary_text_col].str.lower()
            
            # 清理文本中的HTML標籤
            self.data['processed_text'] = self.data['processed_text'].str.replace(r'<.*?>', ' ', regex=True)
            
            # 清理URL
            self.data['processed_text'] = self.data['processed_text'].str.replace(r'https?\S+|www\.\S+', ' ', regex=True)
            
            # 清理標點符號
            self.data['processed_text'] = self.data['processed_text'].str.replace(r'[^\w\s]', ' ', regex=True)
            
            # 清理多餘的空白
            self.data['processed_text'] = self.data['processed_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
            
            self.log_message.emit("文本清理完成")
            
            # 預處理步驟4: 處理情感標籤（如果存在）
            self.progress_update.emit(50)
            
            if 'sentiment' in self.data.columns:
                self.log_message.emit("步驟4: 處理情感標籤...")
                
                # 將情感標籤標準化為 'positive', 'negative', 'neutral'
                sentiment_map = {
                    'positive': 'positive',
                    'negative': 'negative',
                    'neutral': 'neutral',
                    'pos': 'positive',
                    'neg': 'negative',
                    'neu': 'neutral',
                    '1': 'positive',
                    '0': 'neutral',
                    '-1': 'negative',
                    '2': 'positive',
                    '1': 'neutral',
                    '0': 'negative',
                }
                
                # 對數值型情感進行映射
                if pd.api.types.is_numeric_dtype(self.data['sentiment']):
                    # 將大於0的值映射為正面，等於0為中性，小於0為負面
                    self.data['sentiment'] = self.data['sentiment'].apply(
                        lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral')
                    )
                else:
                    # 對字符串情感進行標準化
                    self.data['sentiment'] = self.data['sentiment'].str.lower().map(sentiment_map)
                    
                    # 填充未映射的值
                    if self.data['sentiment'].isna().any():
                        self.log_message.emit(f"警告: {self.data['sentiment'].isna().sum()} 個情感標籤無法識別，設為中性")
                        self.data['sentiment'] = self.data['sentiment'].fillna('neutral')
                
                self.log_message.emit("情感標籤處理完成")
                
                # 統計情感分佈
                sentiment_counts = self.data['sentiment'].value_counts()
                self.log_message.emit(f"情感分佈: {dict(sentiment_counts)}")
            
            # 預處理步驟5: 添加長度特徵
            self.progress_update.emit(70)
            self.log_message.emit("步驟5: 添加文本長度特徵...")
            
            self.data['text_length'] = self.data[primary_text_col].str.len()
            self.data['word_count'] = self.data['processed_text'].str.split().str.len()
            
            # 預處理步驟6: 最終清理
            self.progress_update.emit(90)
            self.log_message.emit("步驟6: 最終數據清理...")
            
            # 移除文本長度過短的樣本（小於5個字）
            initial_rows = len(self.data)
            self.data = self.data[self.data['word_count'] > 5]
            removed_short = initial_rows - len(self.data)
            
            if removed_short > 0:
                self.log_message.emit(f"移除了 {removed_short} 行過短的文本，剩餘 {len(self.data)} 行")
            
            # 重置索引
            self.data.reset_index(drop=True, inplace=True)
            
            # 添加ID列（如果不存在）
            if 'id' not in self.data.columns:
                self.data['id'] = [f"text_{i}" for i in range(len(self.data))]
            
            # 完成
            self.progress_update.emit(100)
            self.log_message.emit(f"預處理完成，最終數據包含 {len(self.data)} 行")
            
            # 發送結果
            self.processing_finished.emit({
                'success': True,
                'data': self.data
            })
            
        except Exception as e:
            self.log_message.emit(f"預處理過程中發生錯誤: {str(e)}")
            self.processing_finished.emit({
                'success': False,
                'error': str(e)
            })
