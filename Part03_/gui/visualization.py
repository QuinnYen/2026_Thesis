"""
視覺化標籤頁模組
此模組負責視覺化界面的邏輯
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Optional, Dict, List, Any, Tuple
import datetime
import tempfile

# 導入PyQt6模組
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, 
    QSpinBox, QLineEdit, QTextEdit, QFileDialog, QMessageBox, QProgressBar, 
    QGroupBox, QTableWidget, QTableWidgetItem, QSplitter, QTabWidget,
    QRadioButton, QButtonGroup, QCheckBox, QSlider, QDoubleSpinBox, QGridLayout,
    QAbstractScrollArea
)
from PyQt6.QtCore import Qt, QSize, QUrl, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QIcon, QFont, QPixmap, QColor
from PyQt6.QtWebEngineWidgets import QWebEngineView

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 導入相關模組
from Part03_.utils.config_manager import ConfigManager
from Part03_.utils.result_manager import ResultManager
from Part03_.utils.visualizers import Visualizer

class VisualizationTab(QWidget):
    """視覺化標籤頁類"""
    
    def __init__(self, config: ConfigManager, result_manager: ResultManager):
        """初始化視覺化標籤頁"""
        super().__init__()
        
        self.config = config
        self.result_manager = result_manager
        
        # 當前結果ID和數據
        self.current_result_id = None
        self.current_data = None
        self.current_meta = None
        self.current_model = None
        
        # 最近生成的圖表路徑
        self.last_chart_path = None
        
        # 初始化UI
        self.init_ui()
    
    def init_ui(self):
        """初始化使用者界面"""
        # 主佈局
        main_layout = QVBoxLayout(self)
        
        # 建立上方控制面板
        control_panel = QGroupBox("視覺化控制")
        control_layout = QVBoxLayout(control_panel)
        
        # 結果選擇
        result_layout = QHBoxLayout()
        result_label = QLabel("結果集:")
        self.result_combo = QComboBox()
        self.result_combo.setMinimumWidth(250)
        self.refresh_result_btn = QPushButton("刷新")
        self.refresh_result_btn.clicked.connect(self.refresh_results)
        self.load_result_btn = QPushButton("載入選擇的結果")
        self.load_result_btn.clicked.connect(self.load_selected_result)
        
        result_layout.addWidget(result_label)
        result_layout.addWidget(self.result_combo)
        result_layout.addWidget(self.refresh_result_btn)
        result_layout.addWidget(self.load_result_btn)
        result_layout.addStretch()
        
        control_layout.addLayout(result_layout)
        
        # 視覺化類型選擇
        visualization_layout = QHBoxLayout()
        visual_type_label = QLabel("視覺化類型:")
        self.visual_type_combo = QComboBox()
        self.visual_type_combo.addItems([
            "主題分佈", "詞雲", "情感分析", "文本長度分佈",
            "主題關聯熱圖", "詞向量視覺化", "模型性能指標"
        ])
        self.visual_type_combo.currentIndexChanged.connect(self.update_visual_options)
        
        visualization_layout.addWidget(visual_type_label)
        visualization_layout.addWidget(self.visual_type_combo)
        
        # 數據選擇
        data_label = QLabel("資料來源:")
        self.data_source_combo = QComboBox()
        self.data_source_combo.setMinimumWidth(200)
        visualization_layout.addWidget(data_label)
        visualization_layout.addWidget(self.data_source_combo)
        
        visualization_layout.addStretch()
        control_layout.addLayout(visualization_layout)
        
        # 視覺化選項（動態更新）
        self.options_group = QGroupBox("視覺化選項")
        self.options_layout = QGridLayout(self.options_group)
        
        control_layout.addWidget(self.options_group)
        
        # 操作按鈕
        buttons_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("生成視覺化")
        self.generate_btn.setIcon(QIcon.fromTheme("document-new"))
        self.generate_btn.clicked.connect(self.generate_visualization)
        
        self.export_btn = QPushButton("匯出")
        self.export_btn.setIcon(QIcon.fromTheme("document-save"))
        self.export_btn.clicked.connect(self.export_visualization)
        
        buttons_layout.addWidget(self.generate_btn)
        buttons_layout.addWidget(self.export_btn)
        buttons_layout.addStretch()
        
        control_layout.addLayout(buttons_layout)
        
        # 設置視覺化顯示區域
        display_group = QGroupBox("視覺化結果")
        display_layout = QVBoxLayout(display_group)
        
        # 創建標籤頁來顯示不同類型的視覺化
        self.display_tabs = QTabWidget()
        
        # 圖像視圖頁面（用於一般圖像）
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        self.image_label = QLabel("尚無視覺化圖表")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd;")
        self.image_label.setMinimumHeight(400)
        image_layout.addWidget(self.image_label)
        
        # 網頁視圖頁面（用於互動式視覺化）
        web_tab = QWidget()
        web_layout = QVBoxLayout(web_tab)
        self.web_view = QWebEngineView()
        self.web_view.setMinimumHeight(400)
        web_layout.addWidget(self.web_view)
        
        # 數據視圖頁面（用於數據表格）
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        self.data_table = QTableWidget()
        self.data_table.setMinimumHeight(400)
        data_layout.addWidget(self.data_table)
        
        # 添加標籤頁到TabWidget
        self.display_tabs.addTab(image_tab, "圖像視圖")
        self.display_tabs.addTab(web_tab, "互動視圖")
        self.display_tabs.addTab(data_tab, "數據視圖")
        
        display_layout.addWidget(self.display_tabs)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        
        # 日誌區域
        log_group = QGroupBox("處理日誌")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # 添加元件到主佈局
        main_layout.addWidget(control_panel)
        main_layout.addWidget(display_group, 3)  # 分配更多空間給顯示區域
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(log_group, 1)
        
        # 初始化視覺化選項
        self.update_visual_options()
        
        # 刷新結果列表
        self.refresh_results()
    
    def refresh_results(self):
        """刷新可用結果列表"""
        try:
            # 保存當前選擇
            current_result = self.result_combo.currentText()
            
            # 清空下拉選單
            self.result_combo.clear()
            
            # 取得所有結果
            results_index = self.result_manager.get_results_index()
            
            if not results_index:
                self.log_message("沒有找到任何處理結果")
                return
            
            # 將結果添加到下拉選單
            for result_id, result_info in results_index.items():
                display_name = f"{result_id} - {result_info.get('dataset', '未知數據集')} ({result_info.get('created_at', '未知日期')})"
                self.result_combo.addItem(display_name, result_id)
            
            # 恢復之前的選擇（如果存在）
            if current_result:
                index = self.result_combo.findText(current_result)
                if index >= 0:
                    self.result_combo.setCurrentIndex(index)
            
            self.log_message(f"找到 {len(results_index)} 個結果集")
            
        except Exception as e:
            self.log_message(f"刷新結果列表時發生錯誤: {str(e)}")
    
    def load_selected_result(self):
        """載入選擇的結果集"""
        if self.result_combo.count() == 0:
            QMessageBox.information(self, "提示", "沒有可用的結果集")
            return
        
        # 獲取選擇的結果ID
        result_id = self.result_combo.currentData()
        if not result_id:
            QMessageBox.information(self, "提示", "請選擇一個結果集")
            return
        
        self.load_result(result_id)
    
    def load_result(self, result_id: str):
        """載入特定結果ID的數據"""
        try:
            self.log_message(f"正在載入結果 {result_id}...")
            self.progress_bar.setValue(10)
            
            # 從結果管理器獲取結果集信息
            result_set = self.result_manager.get_result_set(result_id)
            if not result_set:
                self.log_message(f"找不到結果 {result_id}")
                QMessageBox.warning(self, "錯誤", f"找不到結果 {result_id}")
                return
            
            # 更新當前結果ID
            self.current_result_id = result_id
            self.current_meta = result_set
            
            # 更新可用的數據源
            self.update_data_sources(result_set)
            
            # 載入處理好的數據（如果存在）
            if 'files' in result_set and 'processed_data' in result_set['files']:
                data_file = result_set['files']['processed_data']
                if os.path.exists(data_file):
                    # 根據檔案類型載入數據
                    file_ext = os.path.splitext(data_file)[1].lower()
                    if file_ext == '.csv':
                        self.current_data = pd.read_csv(data_file)
                    elif file_ext == '.json':
                        self.current_data = pd.read_json(data_file, orient='records', lines=True)
                    elif file_ext == '.pickle' or file_ext == '.pkl':
                        self.current_data = pd.read_pickle(data_file)
                    else:
                        self.log_message(f"不支援的數據檔案類型: {file_ext}")
            
            self.progress_bar.setValue(100)
            self.log_message(f"成功載入結果 {result_id}")
            
        except Exception as e:
            self.log_message(f"載入結果時發生錯誤: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"載入結果時發生錯誤: {str(e)}")
            self.progress_bar.setValue(0)
    
    def update_data_sources(self, result_set: Dict[str, Any]):
        """更新可用數據源列表"""
        try:
            # 清空現有數據源
            self.data_source_combo.clear()
            
            if 'files' not in result_set:
                self.log_message("結果集中沒有檔案信息")
                return
            
            # 添加可用的數據文件
            for file_key, file_path in result_set['files'].items():
                if os.path.exists(file_path):
                    display_name = file_key.replace('_', ' ').title()
                    self.data_source_combo.addItem(display_name, file_path)
            
            # 如果有項目，選擇第一個
            if self.data_source_combo.count() > 0:
                self.data_source_combo.setCurrentIndex(0)
            
        except Exception as e:
            self.log_message(f"更新數據源列表時發生錯誤: {str(e)}")
    
    def update_visual_options(self):
        """更新視覺化選項"""
        # 清空現有選項
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # 獲取當前選擇的視覺化類型
        visual_type = self.visual_type_combo.currentText()
        
        # 根據視覺化類型添加相應的選項
        if visual_type == "詞雲":
            # 詞雲選項
            row = 0
            
            # 文本列選擇
            text_col_label = QLabel("文本列:")
            self.text_column_combo = QComboBox()
            self.text_column_combo.addItems(["text", "processed_text", "review"])
            self.options_layout.addWidget(text_col_label, row, 0)
            self.options_layout.addWidget(self.text_column_combo, row, 1)
            
            row += 1
            
            # 停用詞選項
            self.remove_stopwords = QCheckBox("移除停用詞")
            self.remove_stopwords.setChecked(True)
            self.options_layout.addWidget(self.remove_stopwords, row, 0, 1, 2)
            
            row += 1
            
            # 最大詞數
            max_words_label = QLabel("最大詞數:")
            self.max_words_spin = QSpinBox()
            self.max_words_spin.setRange(50, 1000)
            self.max_words_spin.setValue(200)
            self.max_words_spin.setSingleStep(50)
            self.options_layout.addWidget(max_words_label, row, 0)
            self.options_layout.addWidget(self.max_words_spin, row, 1)
            
            row += 1
            
            # 背景色選擇
            bg_color_label = QLabel("背景顏色:")
            self.bg_color_combo = QComboBox()
            self.bg_color_combo.addItems(["白色", "黑色", "灰色", "淺藍"])
            self.options_layout.addWidget(bg_color_label, row, 0)
            self.options_layout.addWidget(self.bg_color_combo, row, 1)
            
        elif visual_type == "主題分佈":
            # 主題分佈選項
            row = 0
            
            # 主題數選擇
            topics_label = QLabel("主題數量:")
            self.topics_spin = QSpinBox()
            self.topics_spin.setRange(2, 50)
            self.topics_spin.setValue(10)
            self.options_layout.addWidget(topics_label, row, 0)
            self.options_layout.addWidget(self.topics_spin, row, 1)
            
            row += 1
            
            # 顯示類型
            display_label = QLabel("顯示類型:")
            self.topic_display_combo = QComboBox()
            self.topic_display_combo.addItems(["條形圖", "圓餅圖", "雷達圖", "樹狀圖"])
            self.options_layout.addWidget(display_label, row, 0)
            self.options_layout.addWidget(self.topic_display_combo, row, 1)
            
            row += 1
            
            # 是否顯示前N個詞
            self.show_top_words = QCheckBox("顯示每個主題的前N個詞")
            self.show_top_words.setChecked(True)
            self.options_layout.addWidget(self.show_top_words, row, 0, 1, 2)
            
            row += 1
            
            # 前N個詞數量
            top_words_label = QLabel("每個主題顯示詞數:")
            self.top_words_spin = QSpinBox()
            self.top_words_spin.setRange(5, 20)
            self.top_words_spin.setValue(10)
            self.options_layout.addWidget(top_words_label, row, 0)
            self.options_layout.addWidget(self.top_words_spin, row, 1)
            
        elif visual_type == "情感分析":
            # 情感分析選項
            row = 0
            
            # 顯示類型
            display_label = QLabel("顯示類型:")
            self.sentiment_display_combo = QComboBox()
            self.sentiment_display_combo.addItems(["條形圖", "圓餅圖", "堆疊條形圖"])
            self.options_layout.addWidget(display_label, row, 0)
            self.options_layout.addWidget(self.sentiment_display_combo, row, 1)
            
            row += 1
            
            # 分組選項
            group_label = QLabel("分組依據:")
            self.sentiment_group_combo = QComboBox()
            self.sentiment_group_combo.addItems(["無", "主題", "長度區間", "日期"])
            self.options_layout.addWidget(group_label, row, 0)
            self.options_layout.addWidget(self.sentiment_group_combo, row, 1)
            
            row += 1
            
            # 顏色選擇
            color_label = QLabel("顏色方案:")
            self.sentiment_color_combo = QComboBox()
            self.sentiment_color_combo.addItems(["紅綠藍", "紅黃綠", "藍橙紫"])
            self.options_layout.addWidget(color_label, row, 0)
            self.options_layout.addWidget(self.sentiment_color_combo, row, 1)
            
        elif visual_type == "文本長度分佈":
            # 文本長度分佈選項
            row = 0
            
            # 長度類型
            length_type_label = QLabel("長度類型:")
            self.length_type_combo = QComboBox()
            self.length_type_combo.addItems(["字符數", "詞數", "句子數"])
            self.options_layout.addWidget(length_type_label, row, 0)
            self.options_layout.addWidget(self.length_type_combo, row, 1)
            
            row += 1
            
            # 顯示類型
            display_label = QLabel("顯示類型:")
            self.length_display_combo = QComboBox()
            self.length_display_combo.addItems(["直方圖", "密度圖", "箱型圖", "累積分佈"])
            self.options_layout.addWidget(display_label, row, 0)
            self.options_layout.addWidget(self.length_display_combo, row, 1)
            
            row += 1
            
            # 箱子數量
            bins_label = QLabel("箱子數量:")
            self.bins_spin = QSpinBox()
            self.bins_spin.setRange(5, 100)
            self.bins_spin.setValue(30)
            self.options_layout.addWidget(bins_label, row, 0)
            self.options_layout.addWidget(self.bins_spin, row, 1)
            
            row += 1
            
            # 分組選項
            group_label = QLabel("分組依據:")
            self.length_group_combo = QComboBox()
            self.length_group_combo.addItems(["無", "情感", "主題", "日期"])
            self.options_layout.addWidget(group_label, row, 0)
            self.options_layout.addWidget(self.length_group_combo, row, 1)
            
        elif visual_type == "主題關聯熱圖":
            # 主題關聯熱圖選項
            row = 0
            
            # 主題數選擇
            topics_label = QLabel("主題數量:")
            self.heatmap_topics_spin = QSpinBox()
            self.heatmap_topics_spin.setRange(2, 30)
            self.heatmap_topics_spin.setValue(10)
            self.options_layout.addWidget(topics_label, row, 0)
            self.options_layout.addWidget(self.heatmap_topics_spin, row, 1)
            
            row += 1
            
            # 指標類型
            metric_label = QLabel("關聯指標:")
            self.heatmap_metric_combo = QComboBox()
            self.heatmap_metric_combo.addItems(["餘弦相似度", "互信息", "主題共現", "傑卡德係數"])
            self.options_layout.addWidget(metric_label, row, 0)
            self.options_layout.addWidget(self.heatmap_metric_combo, row, 1)
            
            row += 1
            
            # 顏色映射
            color_label = QLabel("顏色映射:")
            self.heatmap_color_combo = QComboBox()
            self.heatmap_color_combo.addItems(["紅藍", "彩虹", "熱力", "冷色調", "地形圖"])
            self.options_layout.addWidget(color_label, row, 0)
            self.options_layout.addWidget(self.heatmap_color_combo, row, 1)
            
            row += 1
            
            # 顯示數值
            self.show_heatmap_values = QCheckBox("顯示關聯數值")
            self.show_heatmap_values.setChecked(True)
            self.options_layout.addWidget(self.show_heatmap_values, row, 0, 1, 2)
            
        elif visual_type == "詞向量視覺化":
            # 詞向量視覺化選項
            row = 0
            
            # 降維方法
            reduction_label = QLabel("降維方法:")
            self.reduction_combo = QComboBox()
            self.reduction_combo.addItems(["t-SNE", "PCA", "UMAP"])
            self.options_layout.addWidget(reduction_label, row, 0)
            self.options_layout.addWidget(self.reduction_combo, row, 1)
            
            row += 1
            
            # 維度
            dimensions_label = QLabel("輸出維度:")
            self.dimensions_combo = QComboBox()
            self.dimensions_combo.addItems(["2D", "3D"])
            self.options_layout.addWidget(dimensions_label, row, 0)
            self.options_layout.addWidget(self.dimensions_combo, row, 1)
            
            row += 1
            
            # 顯示標籤
            self.show_vector_labels = QCheckBox("顯示詞彙標籤")
            self.show_vector_labels.setChecked(True)
            self.options_layout.addWidget(self.show_vector_labels, row, 0, 1, 2)
            
            row += 1
            
            # 顯示點數
            points_label = QLabel("最大點數:")
            self.max_points_spin = QSpinBox()
            self.max_points_spin.setRange(100, 5000)
            self.max_points_spin.setValue(1000)
            self.max_points_spin.setSingleStep(100)
            self.options_layout.addWidget(points_label, row, 0)
            self.options_layout.addWidget(self.max_points_spin, row, 1)
            
            row += 1
            
            # 著色方式
            color_label = QLabel("著色方式:")
            self.vector_color_combo = QComboBox()
            self.vector_color_combo.addItems(["詞頻", "主題", "情感", "單一顏色"])
            self.options_layout.addWidget(color_label, row, 0)
            self.options_layout.addWidget(self.vector_color_combo, row, 1)
            
        elif visual_type == "模型性能指標":
            # 模型性能指標選項
            row = 0
            
            # 指標類型
            metric_label = QLabel("指標類型:")
            self.perf_metric_combo = QComboBox()
            self.perf_metric_combo.addItems(["準確率", "精確率/召回率", "F1分數", "混淆矩陣", "ROC曲線", "學習曲線"])
            self.options_layout.addWidget(metric_label, row, 0)
            self.options_layout.addWidget(self.perf_metric_combo, row, 1)
            
            row += 1
            
            # 模型選擇
            model_label = QLabel("選擇模型:")
            self.model_combo = QComboBox()
            # 這裡會動態填充可用的模型
            self.options_layout.addWidget(model_label, row, 0)
            self.options_layout.addWidget(self.model_combo, row, 1)
            
            row += 1
            
            # 比較模型
            self.compare_models = QCheckBox("與其他模型比較")
            self.options_layout.addWidget(self.compare_models, row, 0, 1, 2)
            
            # 更新可用的模型
            self.update_available_models()
    
    def update_available_models(self):
        """更新可用的模型列表"""
        if not hasattr(self, 'model_combo'):
            return
        
        self.model_combo.clear()
        
        if not self.current_result_id:
            return
        
        result_set = self.current_meta
        if not result_set:
            return
        
        # 檢查結果集中是否有模型
        if 'models' in result_set and result_set['models']:
            for model_name in result_set['models']:
                self.model_combo.addItem(model_name)
    
    def generate_visualization(self):
        """生成選定類型的視覺化"""
        if not self.current_result_id:
            QMessageBox.information(self, "提示", "請先載入結果集")
            return
        
        if not self.current_data is not None:
            QMessageBox.information(self, "提示", "沒有可用的數據")
            return
        
        # 獲取當前選擇的視覺化類型
        visual_type = self.visual_type_combo.currentText()
        
        # 創建視覺化線程
        visualization_thread = VisualizationThread(
            visual_type=visual_type,
            options=self.get_visualization_options(),
            data=self.current_data,
            result_set=self.current_meta,
            config=self.config
        )
        
        # 連接信號
        visualization_thread.progress_update.connect(self.update_progress)
        visualization_thread.log_message.connect(self.log_message)
        visualization_thread.visualization_complete.connect(self.on_visualization_complete)
        
        # 啟動線程
        self.log_message(f"正在生成 {visual_type} 視覺化圖表...")
        visualization_thread.start()
        
        # 禁用生成按鈕，避免重複點擊
        self.generate_btn.setEnabled(False)
    
    def get_visualization_options(self) -> Dict[str, Any]:
        """獲取當前視覺化選項"""
        options = {}
        visual_type = self.visual_type_combo.currentText()
        
        # 根據視覺化類型獲取相應選項
        if visual_type == "詞雲":
            if hasattr(self, 'text_column_combo'):
                options['text_column'] = self.text_column_combo.currentText()
            if hasattr(self, 'remove_stopwords'):
                options['remove_stopwords'] = self.remove_stopwords.isChecked()
            if hasattr(self, 'max_words_spin'):
                options['max_words'] = self.max_words_spin.value()
            if hasattr(self, 'bg_color_combo'):
                bg_colors = {
                    '白色': 'white',
                    '黑色': 'black',
                    '灰色': '#f0f0f0',
                    '淺藍': '#e8f4f8'
                }
                options['background_color'] = bg_colors[self.bg_color_combo.currentText()]
                
        elif visual_type == "主題分佈":
            if hasattr(self, 'topics_spin'):
                options['num_topics'] = self.topics_spin.value()
            if hasattr(self, 'topic_display_combo'):
                options['display_type'] = self.topic_display_combo.currentText()
            if hasattr(self, 'show_top_words'):
                options['show_top_words'] = self.show_top_words.isChecked()
            if hasattr(self, 'top_words_spin'):
                options['top_n_words'] = self.top_words_spin.value()
                
        elif visual_type == "情感分析":
            if hasattr(self, 'sentiment_display_combo'):
                options['display_type'] = self.sentiment_display_combo.currentText()
            if hasattr(self, 'sentiment_group_combo'):
                options['group_by'] = self.sentiment_group_combo.currentText()
            if hasattr(self, 'sentiment_color_combo'):
                color_schemes = {
                    '紅綠藍': ['#e74c3c', '#3498db', '#2ecc71'],
                    '紅黃綠': ['#e74c3c', '#f1c40f', '#2ecc71'],
                    '藍橙紫': ['#3498db', '#e67e22', '#9b59b6']
                }
                options['colors'] = color_schemes[self.sentiment_color_combo.currentText()]
                
        elif visual_type == "文本長度分佈":
            if hasattr(self, 'length_type_combo'):
                options['length_type'] = self.length_type_combo.currentText()
            if hasattr(self, 'length_display_combo'):
                options['display_type'] = self.length_display_combo.currentText()
            if hasattr(self, 'bins_spin'):
                options['bins'] = self.bins_spin.value()
            if hasattr(self, 'length_group_combo'):
                options['group_by'] = self.length_group_combo.currentText()
                
        elif visual_type == "主題關聯熱圖":
            if hasattr(self, 'heatmap_topics_spin'):
                options['num_topics'] = self.heatmap_topics_spin.value()
            if hasattr(self, 'heatmap_metric_combo'):
                options['metric'] = self.heatmap_metric_combo.currentText()
            if hasattr(self, 'heatmap_color_combo'):
                cmap_mapping = {
                    '紅藍': 'coolwarm',
                    '彩虹': 'rainbow',
                    '熱力': 'hot',
                    '冷色調': 'Blues',
                    '地形圖': 'terrain'
                }
                options['colormap'] = cmap_mapping[self.heatmap_color_combo.currentText()]
            if hasattr(self, 'show_heatmap_values'):
                options['show_values'] = self.show_heatmap_values.isChecked()
                
        elif visual_type == "詞向量視覺化":
            if hasattr(self, 'reduction_combo'):
                options['reduction_method'] = self.reduction_combo.currentText()
            if hasattr(self, 'dimensions_combo'):
                options['dimensions'] = 2 if self.dimensions_combo.currentText() == '2D' else 3
            if hasattr(self, 'show_vector_labels'):
                options['show_labels'] = self.show_vector_labels.isChecked()
            if hasattr(self, 'max_points_spin'):
                options['max_points'] = self.max_points_spin.value()
            if hasattr(self, 'vector_color_combo'):
                options['color_by'] = self.vector_color_combo.currentText()
                
        elif visual_type == "模型性能指標":
            if hasattr(self, 'perf_metric_combo'):
                options['metric'] = self.perf_metric_combo.currentText()
            if hasattr(self, 'model_combo') and self.model_combo.currentText():
                options['model'] = self.model_combo.currentText()
            if hasattr(self, 'compare_models'):
                options['compare'] = self.compare_models.isChecked()
        
        return options
    
    def on_visualization_complete(self, result):
        """視覺化完成後的回調"""
        # 重新啟用生成按鈕
        self.generate_btn.setEnabled(True)
        
        if not result['success']:
            self.log_message(f"視覺化失敗: {result.get('error', '未知錯誤')}")
            QMessageBox.critical(self, "錯誤", f"視覺化失敗: {result.get('error', '未知錯誤')}")
            return
        
        self.log_message("視覺化生成成功")
        
        # 獲取視覺化結果
        vis_type = result.get('type')
        output_path = result.get('path')
        
        # 保存最近的圖表路徑
        self.last_chart_path = output_path
        
        # 根據視覺化類型顯示結果
        if vis_type == 'image':
            # 圖像類型
            self.display_tabs.setCurrentIndex(0)  # 切換到圖像視圖
            
            # 顯示圖像
            if output_path and os.path.exists(output_path):
                pixmap = QPixmap(output_path)
                self.image_label.setPixmap(pixmap.scaled(
                    self.image_label.width(), 
                    self.image_label.height(), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                ))
            else:
                self.log_message("錯誤: 找不到輸出的圖像文件")
                
        elif vis_type == 'html':
            # HTML互動式視覺化
            self.display_tabs.setCurrentIndex(1)  # 切換到網頁視圖
            
            # 載入HTML文件
            if output_path and os.path.exists(output_path):
                url = QUrl.fromLocalFile(output_path)
                self.web_view.load(url)
            else:
                self.log_message("錯誤: 找不到輸出的HTML文件")
                
        elif vis_type == 'data':
            # 數據表格
            self.display_tabs.setCurrentIndex(2)  # 切換到數據視圖
            
            # 顯示數據
            if 'data' in result:
                self.display_data_table(result['data'])
            else:
                self.log_message("錯誤: 沒有可用的數據結果")
    
    def display_data_table(self, data):
        """在表格中顯示數據結果"""
        # 清空表格
        self.data_table.clear()
        
        if isinstance(data, pd.DataFrame):
            # 設置行數和列數
            row_count = len(data)
            col_count = len(data.columns)
            self.data_table.setRowCount(row_count)
            self.data_table.setColumnCount(col_count)
            
            # 設置表頭
            self.data_table.setHorizontalHeaderLabels(data.columns)
            
            # 填充數據
            for row in range(row_count):
                for col, column_name in enumerate(data.columns):
                    value = str(data.iloc[row, col])
                    item = QTableWidgetItem(value)
                    self.data_table.setItem(row, col, item)
                    
        elif isinstance(data, dict):
            # 如果是字典，以鍵值對形式顯示
            keys = list(data.keys())
            self.data_table.setRowCount(len(keys))
            self.data_table.setColumnCount(2)
            self.data_table.setHorizontalHeaderLabels(["指標", "值"])
            
            for row, key in enumerate(keys):
                key_item = QTableWidgetItem(str(key))
                value_item = QTableWidgetItem(str(data[key]))
                self.data_table.setItem(row, 0, key_item)
                self.data_table.setItem(row, 1, value_item)
                
        elif isinstance(data, list):
            # 如果是列表，且每個元素是字典
            if all(isinstance(item, dict) for item in data):
                if len(data) > 0:
                    # 提取所有鍵
                    keys = set()
                    for item in data:
                        keys.update(item.keys())
                    keys = list(keys)
                    
                    # 設置表格
                    self.data_table.setRowCount(len(data))
                    self.data_table.setColumnCount(len(keys))
                    self.data_table.setHorizontalHeaderLabels(keys)
                    
                    # 填充數據
                    for row, item in enumerate(data):
                        for col, key in enumerate(keys):
                            value = str(item.get(key, ''))
                            self.data_table.setItem(row, col, QTableWidgetItem(value))
            else:
                # 簡單列表
                self.data_table.setRowCount(len(data))
                self.data_table.setColumnCount(1)
                self.data_table.setHorizontalHeaderLabels(["值"])
                
                for row, value in enumerate(data):
                    self.data_table.setItem(row, 0, QTableWidgetItem(str(value)))
        
        # 調整列寬
        self.data_table.resizeColumnsToContents()
    
    def export_visualization(self):
        """匯出當前視覺化圖表"""
        if not self.last_chart_path or not os.path.exists(self.last_chart_path):
            QMessageBox.information(self, "提示", "沒有可匯出的視覺化圖表")
            return
        
        # 確定文件類型
        file_ext = os.path.splitext(self.last_chart_path)[1].lower()
        
        # 選擇保存路徑
        if file_ext == '.html':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "匯出視覺化", "", "HTML文件 (*.html);;所有文件 (*)"
            )
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "匯出視覺化", "", "圖像文件 (*.png *.jpg);;所有文件 (*)"
            )
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "匯出視覺化", "", "所有文件 (*)"
            )
        
        if file_path:
            try:
                # 複製文件
                import shutil
                shutil.copy2(self.last_chart_path, file_path)
                self.log_message(f"視覺化已匯出到: {file_path}")
                QMessageBox.information(self, "成功", f"視覺化已成功匯出到: {file_path}")
            except Exception as e:
                self.log_message(f"匯出視覺化時發生錯誤: {str(e)}")
                QMessageBox.critical(self, "錯誤", f"匯出視覺化時發生錯誤: {str(e)}")
    
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


class VisualizationThread(QThread):
    """視覺化線程"""
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)
    visualization_complete = pyqtSignal(dict)
    
    def __init__(self, visual_type: str, options: Dict[str, Any], data: pd.DataFrame, 
                 result_set: Dict[str, Any], config: ConfigManager):
        """初始化視覺化線程"""
        super().__init__()
        self.visual_type = visual_type
        self.options = options
        self.data = data
        self.result_set = result_set
        self.config = config
    
    def run(self):
        """執行視覺化生成"""
        try:
            # 發送初始進度
            self.progress_update.emit(0)
            self.log_message.emit(f"開始生成 {self.visual_type} 視覺化...")
            
            # 檢查數據
            if self.data is None or len(self.data) == 0:
                raise Exception("沒有可用的數據")
            
            # 創建輸出目錄
            output_dir = self.result_set.get('dirs', {}).get('visualizations')
            if not output_dir:
                # 使用默認目錄
                output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'results', 'visualizations')
            
            # 確保目錄存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名
            filename_base = f"{self.visual_type.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 根據視覺化類型進行不同的處理
            if self.visual_type == "詞雲":
                self.progress_update.emit(10)
                self.log_message.emit("生成詞雲...")
                
                # 獲取選項
                text_column = self.options.get('text_column', 'text')
                remove_stopwords = self.options.get('remove_stopwords', True)
                max_words = self.options.get('max_words', 200)
                background_color = self.options.get('background_color', 'white')
                
                # 檢查文本列是否存在
                if text_column not in self.data.columns:
                    text_columns = [col for col in self.data.columns if pd.api.types.is_string_dtype(self.data[col])]
                    if text_columns:
                        text_column = text_columns[0]
                        self.log_message.emit(f"指定的文本列不存在，使用 {text_column} 列")
                    else:
                        raise Exception("找不到有效的文本列")
                
                # 準備文本
                self.progress_update.emit(20)
                text_data = self.data[text_column].dropna().astype(str).tolist()
                
                # 生成詞雲
                self.progress_update.emit(40)
                output_path = os.path.join(output_dir, f"{filename_base}.png")
                
                # 調用詞雲生成函數
                Visualizer.generate_wordcloud(
                    text_data=text_data,
                    output_path=output_path,
                    max_words=max_words,
                    remove_stopwords=remove_stopwords,
                    background_color=background_color
                )
                
                self.progress_update.emit(100)
                self.log_message.emit(f"詞雲已生成: {output_path}")
                
                # 返回結果
                self.visualization_complete.emit({
                    'success': True,
                    'type': 'image',
                    'path': output_path
                })
                
            elif self.visual_type == "主題分佈":
                self.progress_update.emit(10)
                self.log_message.emit("生成主題分佈...")
                
                # 獲取LDA模型和主題數據
                lda_model_path = self.result_set.get('files', {}).get('lda_model')
                topics_data_path = self.result_set.get('files', {}).get('topics_data')
                
                if not lda_model_path or not os.path.exists(lda_model_path):
                    raise Exception("找不到LDA模型文件")
                
                if not topics_data_path or not os.path.exists(topics_data_path):
                    raise Exception("找不到主題數據文件")
                
                # 獲取選項
                num_topics = self.options.get('num_topics', 10)
                display_type = self.options.get('display_type', '條形圖')
                show_top_words = self.options.get('show_top_words', True)
                top_n_words = self.options.get('top_n_words', 10)
                
                # 載入主題數據
                self.progress_update.emit(30)
                topic_distribution = None
                
                try:
                    with open(topics_data_path, 'r', encoding='utf-8') as f:
                        topics_data = json.load(f)
                        if 'topic_distribution' in topics_data:
                            topic_distribution = topics_data['topic_distribution']
                        if 'topic_keywords' in topics_data:
                            topic_keywords = topics_data['topic_keywords']
                except Exception as e:
                    self.log_message.emit(f"載入主題數據時發生錯誤: {str(e)}")
                    # 嘗試從模型中提取主題分佈
                    topic_distribution = None
                
                # 生成主題分佈視覺化
                self.progress_update.emit(50)
                output_path = os.path.join(output_dir, f"{filename_base}.html")
                
                # 調用主題分佈生成函數
                Visualizer.generate_topic_distribution(
                    topic_data=topic_distribution,
                    topic_keywords=topic_keywords if show_top_words else None,
                    output_path=output_path,
                    num_topics=num_topics,
                    display_type=display_type.lower().replace('圖', ''),
                    top_n_words=top_n_words
                )
                
                self.progress_update.emit(100)
                self.log_message.emit(f"主題分佈已生成: {output_path}")
                
                # 返回結果
                self.visualization_complete.emit({
                    'success': True,
                    'type': 'html',  # 使用HTML格式
                    'path': output_path
                })
                
            # 其他視覺化類型...
            else:
                # 生成一個臨時的示例圖表
                import matplotlib.pyplot as plt
                import numpy as np
                
                self.log_message.emit(f"生成演示圖表 ({self.visual_type})...")
                self.progress_update.emit(30)
                
                plt.figure(figsize=(10, 6))
                
                # 生成示例數據
                x = np.linspace(0, 10, 100)
                y = np.sin(x)
                
                plt.plot(x, y)
                plt.title(f"{self.visual_type} 演示圖")
                plt.xlabel("X軸")
                plt.ylabel("Y軸")
                plt.grid(True)
                
                output_path = os.path.join(output_dir, f"{filename_base}.png")
                plt.savefig(output_path)
                plt.close()
                
                self.progress_update.emit(100)
                self.log_message.emit(f"演示圖表已生成: {output_path}")
                self.log_message.emit(f"注意: {self.visual_type} 功能尚未完全實現，顯示的是演示圖表")
                
                # 返回結果
                self.visualization_complete.emit({
                    'success': True,
                    'type': 'image',
                    'path': output_path
                })
            
        except Exception as e:
            self.log_message.emit(f"視覺化生成過程中發生錯誤: {str(e)}")
            self.progress_update.emit(0)
            self.visualization_complete.emit({
                'success': False,
                'error': str(e)
            })
