"""
可視化頁面模組 - 實現各種數據可視化和圖表展示功能
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTabWidget, QFileDialog, QGroupBox, QCheckBox,
    QRadioButton, QButtonGroup, QSplitter, QScrollArea,
    QSlider, QSpinBox, QListWidget, QListWidgetItem, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QTextCursor
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView

# 導入模組
from modules.visualizer import Visualizer
from modules.evaluator import Evaluator

# 導入工具類
from utils.logger import get_logger
from utils.file_manager import FileManager

# 獲取logger
logger = get_logger("visualization_tab")

class VisualizationTab(QWidget):
    """可視化頁面類，實現各種數據可視化和圖表展示功能"""
    
    # 定義信號
    status_message = pyqtSignal(str, int)  # 狀態欄訊息信號，參數：訊息, 顯示時間(毫秒)
    visualization_completed = pyqtSignal(dict)  # 可視化完成信號，參數：結果字典
    
    def __init__(self, config, file_manager):
        """初始化可視化頁面
        
        Args:
            config: 配置管理器
            file_manager: 文件管理器
        """
        super().__init__()
        
        # 保存引用
        self.config = config
        self.file_manager = file_manager
        
        # 檢查並確保 NLTK 資源已就緒
        self._ensure_nltk_resources()
        
        # 初始化成員變數
        self.current_dataset = None  # 當前數據集
        self.topics = None  # 主題詞
        self.aspect_vectors = None  # 面向向量
        self.evaluation_results = None  # 評估結果
        self.visualization_results = {}  # 可視化結果
        self.result_file_path = None  # 結果文件路徑
        
        # 初始化可視化模組
        self.visualizer = Visualizer(self.config.get("visualization"))
        
        # 創建UI
        self._init_ui()
        
        # 初始化完畢後發送狀態訊息
        self.status_message.emit("可視化頁面已準備就緒", 3000)
    
    def _ensure_nltk_resources(self):
        """確保必要的 NLTK 資源可用"""
        try:
            import nltk
            import os
            
            # 設定 NLTK 數據目錄
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            nltk_data_path = os.path.join(current_dir, "nltk_data")
            
            # 確保路徑在 NLTK 搜尋路徑中
            if nltk_data_path not in nltk.data.path:
                nltk.data.path.append(nltk_data_path)
                
            # 檢查必要的資源
            required_resources = ['punkt', 'stopwords', 'wordnet']
            missing_resources = []
            
            for resource in required_resources:
                try:
                    nltk.data.find(f'{resource}')
                except LookupError:
                    missing_resources.append(resource)
            
            # 如果有缺少的資源，嘗試下載
            if missing_resources:
                for resource in missing_resources:
                    try:
                        nltk.download(resource, download_dir=nltk_data_path, quiet=True)
                    except Exception as e:
                        print(f"下載 NLTK 資源 {resource} 時出錯: {str(e)}")
        
        except ImportError as e:
            print(f"無法導入 NLTK: {str(e)}")
        except Exception as e:
            print(f"NLTK 資源處理時出錯: {str(e)}")
    
    def _init_ui(self):
        """初始化UI界面"""
        # 主佈局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 創建上部控制區
        control_layout = self._create_control_panel()
        main_layout.addLayout(control_layout)
        
        # 創建內容分割器
        self.content_splitter = QSplitter(Qt.Vertical)
        
        # 創建可視化選項區域
        self._create_options_panel()
        
        # 創建可視化結果區域
        self._create_result_panel()
        
        # 添加分割器到主佈局
        main_layout.addWidget(self.content_splitter, 1)  # 1表示拉伸系數
        
        # 創建底部控制區
        bottom_layout = self._create_bottom_panel()
        main_layout.addLayout(bottom_layout)
    
    def _create_control_panel(self):
        """創建控制面板"""
        control_layout = QHBoxLayout()
        
        # 結果選擇區域
        results_group = QGroupBox("分析結果")
        results_layout = QVBoxLayout(results_group)
        
        # 創建結果選擇組合框
        result_selector_layout = QHBoxLayout()
        
        self.result_combo = QComboBox()
        self.result_combo.setMinimumWidth(300)
        available_results = self._get_available_results()
        for result in available_results:
            self.result_combo.addItem(result["name"], result["path"])
        
        result_selector_layout.addWidget(QLabel("選擇結果:"))
        result_selector_layout.addWidget(self.result_combo, 1)
        
        # 創建載入、刷新按鈕
        load_btn = QPushButton("載入")
        load_btn.clicked.connect(self.load_selected_result)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_result_list)
        
        browse_btn = QPushButton("瀏覽...")
        browse_btn.clicked.connect(self.browse_result_file)
        
        result_selector_layout.addWidget(load_btn)
        result_selector_layout.addWidget(refresh_btn)
        result_selector_layout.addWidget(browse_btn)
        results_layout.addLayout(result_selector_layout)
        
        # 添加結果信息標籤
        self.result_info_label = QLabel("當前結果: 未載入")
        results_layout.addWidget(self.result_info_label)
        
        control_layout.addWidget(results_group)
        
        return control_layout
    
    def _create_options_panel(self):
        """創建可視化選項面板"""
        # 選項面板容器
        self.options_widget = QWidget()
        options_layout = QVBoxLayout(self.options_widget)
        
        # 選項分組
        group_viz_type = QGroupBox("可視化類型")
        viz_type_layout = QVBoxLayout(group_viz_type)
        
        # 創建可視化類型選項
        self.viz_type_group = QButtonGroup(self)
        
        self.rb_topic_distribution = QRadioButton("主題分佈")
        self.rb_topic_distribution.setChecked(True)
        self.viz_type_group.addButton(self.rb_topic_distribution, 1)
        viz_type_layout.addWidget(self.rb_topic_distribution)
        
        self.rb_vector_clustering = QRadioButton("向量聚類")
        self.viz_type_group.addButton(self.rb_vector_clustering, 2)
        viz_type_layout.addWidget(self.rb_vector_clustering)
        
        self.rb_topic_network = QRadioButton("主題關係網絡")
        self.viz_type_group.addButton(self.rb_topic_network, 3)
        viz_type_layout.addWidget(self.rb_topic_network)
        
        self.rb_word_cloud = QRadioButton("主題詞雲")
        self.viz_type_group.addButton(self.rb_word_cloud, 4)
        viz_type_layout.addWidget(self.rb_word_cloud)
        
        self.rb_attention_heatmap = QRadioButton("注意力熱圖")
        self.viz_type_group.addButton(self.rb_attention_heatmap, 5)
        viz_type_layout.addWidget(self.rb_attention_heatmap)
        
        self.rb_evaluation = QRadioButton("評估指標")
        self.viz_type_group.addButton(self.rb_evaluation, 6)
        viz_type_layout.addWidget(self.rb_evaluation)
        
        options_layout.addWidget(group_viz_type)
        
        # 各類型的細節選項
        self.viz_options_stack = QTabWidget()
        
        # 主題分佈選項
        self.topic_options = QWidget()
        topic_options_layout = QVBoxLayout(self.topic_options)
        
        self.cb_show_topic_labels = QCheckBox("顯示主題標籤")
        self.cb_show_topic_labels.setChecked(True)
        topic_options_layout.addWidget(self.cb_show_topic_labels)
        
        self.cb_interactive_topic = QCheckBox("互動式圖表")
        self.cb_interactive_topic.setChecked(True)
        topic_options_layout.addWidget(self.cb_interactive_topic)
        
        self.cb_3d_topic = QCheckBox("3D視圖 (如果可用)")
        topic_options_layout.addWidget(self.cb_3d_topic)
        
        topic_options_layout.addStretch()
        
        # 向量聚類選項
        self.cluster_options = QWidget()
        cluster_options_layout = QVBoxLayout(self.cluster_options)
        
        cluster_algorithm_layout = QHBoxLayout()
        cluster_algorithm_layout.addWidget(QLabel("聚類算法:"))
        self.cluster_algorithm_combo = QComboBox()
        self.cluster_algorithm_combo.addItems(["K-Means", "DBSCAN", "層次聚類"])
        cluster_algorithm_layout.addWidget(self.cluster_algorithm_combo)
        cluster_options_layout.addLayout(cluster_algorithm_layout)
        
        cluster_count_layout = QHBoxLayout()
        cluster_count_layout.addWidget(QLabel("聚類數:"))
        self.cluster_count_spin = QSpinBox()
        self.cluster_count_spin.setRange(2, 20)
        self.cluster_count_spin.setValue(5)
        cluster_count_layout.addWidget(self.cluster_count_spin)
        cluster_options_layout.addLayout(cluster_count_layout)
        
        self.cb_interactive_cluster = QCheckBox("互動式圖表")
        self.cb_interactive_cluster.setChecked(True)
        cluster_options_layout.addWidget(self.cb_interactive_cluster)
        
        cluster_options_layout.addStretch()
        
        # 關係網絡選項
        self.network_options = QWidget()
        network_options_layout = QVBoxLayout(self.network_options)
        
        self.cb_show_weights = QCheckBox("顯示連接權重")
        self.cb_show_weights.setChecked(True)
        network_options_layout.addWidget(self.cb_show_weights)
        
        edge_threshold_layout = QHBoxLayout()
        edge_threshold_layout.addWidget(QLabel("邊閾值:"))
        self.edge_threshold_slider = QSlider(Qt.Horizontal)
        self.edge_threshold_slider.setRange(1, 100)
        self.edge_threshold_slider.setValue(30)
        edge_threshold_layout.addWidget(self.edge_threshold_slider)
        edge_threshold_layout.addWidget(QLabel("0.30"))
        network_options_layout.addLayout(edge_threshold_layout)
        
        self.cb_interactive_network = QCheckBox("互動式網絡")
        self.cb_interactive_network.setChecked(True)
        network_options_layout.addWidget(self.cb_interactive_network)
        
        network_options_layout.addStretch()
        
        # 詞雲選項
        self.wordcloud_options = QWidget()
        wordcloud_options_layout = QVBoxLayout(self.wordcloud_options)
        
        self.topic_select_layout = QHBoxLayout()
        self.topic_select_layout.addWidget(QLabel("選擇主題:"))
        self.topic_select_combo = QComboBox()
        self.topic_select_combo.addItem("所有主題")
        self.topic_select_layout.addWidget(self.topic_select_combo)
        wordcloud_options_layout.addLayout(self.topic_select_layout)
        
        wordcloud_color_layout = QHBoxLayout()
        wordcloud_color_layout.addWidget(QLabel("配色方案:"))
        self.wordcloud_color_combo = QComboBox()
        self.wordcloud_color_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis"])
        wordcloud_color_layout.addWidget(self.wordcloud_color_combo)
        wordcloud_options_layout.addLayout(wordcloud_color_layout)
        
        wordcloud_options_layout.addStretch()
        
        # 注意力熱圖選項
        self.heatmap_options = QWidget()
        heatmap_options_layout = QVBoxLayout(self.heatmap_options)
        
        self.attention_type_layout = QHBoxLayout()
        self.attention_type_layout.addWidget(QLabel("注意力類型:"))
        self.attention_type_combo = QComboBox()
        self.attention_type_combo.addItems(["相似度注意力", "關鍵詞注意力", "自注意力", "綜合注意力"])
        self.attention_type_layout.addWidget(self.attention_type_combo)
        heatmap_options_layout.addLayout(self.attention_type_layout)
        
        self.sample_id_layout = QHBoxLayout()
        self.sample_id_layout.addWidget(QLabel("樣本ID:"))
        self.sample_id_spin = QSpinBox()
        self.sample_id_spin.setRange(0, 100)
        self.sample_id_spin.setValue(0)
        self.sample_id_layout.addWidget(self.sample_id_spin)
        heatmap_options_layout.addLayout(self.sample_id_layout)
        
        heatmap_options_layout.addStretch()
        
        # 評估指標選項
        self.eval_options = QWidget()
        eval_options_layout = QVBoxLayout(self.eval_options)
        
        self.cb_show_all_metrics = QCheckBox("顯示所有指標")
        self.cb_show_all_metrics.setChecked(True)
        eval_options_layout.addWidget(self.cb_show_all_metrics)
        
        self.cb_show_chart = QCheckBox("圖表顯示")
        self.cb_show_chart.setChecked(True)
        eval_options_layout.addWidget(self.cb_show_chart)
        
        eval_options_layout.addStretch()
        
        # 添加選項卡
        self.viz_options_stack.addTab(self.topic_options, "主題分佈")
        self.viz_options_stack.addTab(self.cluster_options, "向量聚類")
        self.viz_options_stack.addTab(self.network_options, "關係網絡")
        self.viz_options_stack.addTab(self.wordcloud_options, "詞雲")
        self.viz_options_stack.addTab(self.heatmap_options, "注意力熱圖")
        self.viz_options_stack.addTab(self.eval_options, "評估指標")
        
        options_layout.addWidget(self.viz_options_stack)
        
        # 連接可視化類型選擇的信號
        self.viz_type_group.buttonClicked.connect(self._on_viz_type_changed)
        
        # 生成可視化按鈕
        self.generate_btn = QPushButton("生成可視化")
        self.generate_btn.setMinimumHeight(30)
        self.generate_btn.clicked.connect(self.generate_visualization)
        options_layout.addWidget(self.generate_btn)
        
        # 添加到分割器
        self.content_splitter.addWidget(self.options_widget)
    
    def _create_result_panel(self):
        """創建可視化結果面板"""
        # 創建結果顯示區域
        self.result_widget = QWidget()
        result_layout = QVBoxLayout(self.result_widget)
        
        # 添加標籤
        self.viz_title_label = QLabel("可視化結果")
        self.viz_title_label.setAlignment(Qt.AlignCenter)
        self.viz_title_label.setFont(QFont("Arial", 12, QFont.Bold))
        result_layout.addWidget(self.viz_title_label)
        
        # 創建結果標籤頁
        self.result_tabs = QTabWidget()
        
        # 互動式可視化標籤頁 (使用QWebEngineView)
        self.interactive_view = QWebEngineView()
        self.interactive_view.setHtml("<center><h3>尚未生成互動式可視化</h3></center>")
        self.result_tabs.addTab(self.interactive_view, "互動式視圖")
        
        # 靜態圖片標籤頁
        self.static_scroll = QScrollArea()
        self.static_scroll.setWidgetResizable(True)
        self.static_container = QWidget()
        self.static_layout = QVBoxLayout(self.static_container)
        
        self.static_image_label = QLabel("尚未生成靜態可視化")
        self.static_image_label.setAlignment(Qt.AlignCenter)
        self.static_layout.addWidget(self.static_image_label)
        self.static_layout.addStretch()
        
        self.static_scroll.setWidget(self.static_container)
        self.result_tabs.addTab(self.static_scroll, "靜態視圖")
        
        # 數據表格標籤頁 (僅在某些可視化中顯示)
        self.data_scroll = QScrollArea()
        self.data_scroll.setWidgetResizable(True)
        self.data_container = QWidget()
        self.data_layout = QVBoxLayout(self.data_container)
        
        self.data_label = QLabel("尚未生成數據視圖")
        self.data_label.setAlignment(Qt.AlignCenter)
        self.data_layout.addWidget(self.data_label)
        self.data_layout.addStretch()
        
        self.data_scroll.setWidget(self.data_container)
        self.result_tabs.addTab(self.data_scroll, "數據視圖")
        
        result_layout.addWidget(self.result_tabs, 1)
        
        # 添加到分割器
        self.content_splitter.addWidget(self.result_widget)
    
    def _create_bottom_panel(self):
        """創建底部控制面板"""
        bottom_layout = QHBoxLayout()
        
        # 左側空間
        bottom_layout.addStretch(1)
        
        # 保存按鈕
        self.save_image_btn = QPushButton("保存圖片")
        self.save_image_btn.clicked.connect(self.save_visualization_image)
        bottom_layout.addWidget(self.save_image_btn)
        
        # 保存HTML按鈕
        self.save_html_btn = QPushButton("保存HTML")
        self.save_html_btn.clicked.connect(self.save_visualization_html)
        bottom_layout.addWidget(self.save_html_btn)
        
        # 匯出報告按鈕
        self.export_report_btn = QPushButton("匯出報告")
        self.export_report_btn.clicked.connect(self.export_report_dialog)
        bottom_layout.addWidget(self.export_report_btn)
        
        # 禁用按鈕（直到生成可視化）
        self.save_image_btn.setEnabled(False)
        self.save_html_btn.setEnabled(False)
        self.export_report_btn.setEnabled(False)
        
        return bottom_layout

    def _get_available_results(self):
        """獲取可用的結果文件列表"""
        results = []
        
        try:
            # 從結果目錄中獲取
            # 安全地從不同類型的配置對象中獲取路徑
            results_dir = None
            
            # 嘗試不同的方式獲取配置值
            try:
                if isinstance(self.config, dict):
                    results_dir = self.config.get("paths", {}).get("output_dir", "./output")
                elif hasattr(self.config, "get"):
                    # 嘗試直接獲取配置路徑
                    try:
                        paths = self.config.get("paths")
                        if isinstance(paths, dict):
                            results_dir = paths.get("output_dir", "./output")
                        else:
                            results_dir = self.config.get(("paths", "output_dir"), "./output")
                    except TypeError:
                        # 如果上述方法失敗，嘗試一次直接訪問完整路徑
                        results_dir = self.config.get("paths.output_dir", "./output")
                else:
                    results_dir = "./output"  # 默認值
            except Exception as config_error:
                logger.warning(f"讀取配置時出現錯誤，使用默認值: {str(config_error)}")
                results_dir = "./output"
            
            if not os.path.exists(results_dir):
                return results
                
            # 列出所有結果JSON文件
            for file in os.listdir(results_dir):
                if file.startswith('result_') and file.endswith('.json'):
                    file_path = os.path.join(results_dir, file)
                    
                    # 嘗試解析文件名和時間
                    try:
                        # 從文件名解析數據集名稱和時間
                        # 格式: result_DATASETNAME_TIMESTAMP.json
                        parts = file[7:-5].split('_')
                        timestamp_parts = parts[-2:]
                        dataset_parts = parts[:-2]
                        
                        dataset_name = '_'.join(dataset_parts)
                        timestamp = '_'.join(timestamp_parts)
                        
                        # 格式化顯示名稱
                        display_name = f"{dataset_name} ({timestamp})"
                        
                        results.append({
                            "name": display_name,
                            "path": file_path,
                            "timestamp": timestamp
                        })
                    except Exception:
                        # 如果解析失敗，直接使用文件名
                        results.append({
                            "name": file,
                            "path": file_path,
                            "timestamp": ""
                        })
        except Exception as e:
            logger.error(f"獲取結果文件列表出錯: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 按時間戳排序，最新的在前面
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # 添加一個空選項
        results.insert(0, {"name": "-- 選擇結果文件 --", "path": "", "timestamp": ""})
        
        return results

    def refresh_result_list(self):
        """刷新結果文件列表"""
        current_selection = self.result_combo.currentText()
        
        self.result_combo.clear()
        available_results = self._get_available_results()
        
        for result in available_results:
            self.result_combo.addItem(result["name"], result["path"])
            
        # 嘗試恢復之前選中的項
        index = self.result_combo.findText(current_selection)
        if index >= 0:
            self.result_combo.setCurrentIndex(index)
            
        self.status_message.emit("結果列表已刷新", 3000)

    def browse_result_file(self):
        """瀏覽選擇結果文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇結果文件",
            self.config.get("paths", {}).get("output_dir", "./output"),
            "結果文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            self.load_results(file_path)

    def load_selected_result(self):
        """載入選中的結果文件"""
        result_path = self.result_combo.currentData()
        
        if not result_path or self.result_combo.currentIndex() == 0:
            QMessageBox.warning(self, "選擇結果", "請選擇一個有效的結果文件")
            return
            
        self.load_results(result_path)

    def load_results(self, file_path):
        """載入指定路徑的結果文件
        
        Args:
            file_path: 結果文件路徑
        """
        try:
            # 重置數據和結果
            self.current_dataset = None
            self.topics = None
            self.aspect_vectors = None
            self.evaluation_results = None
            self.visualization_results = {}
            
            # 讀取結果文件
            with open(file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 設置結果文件路徑
            self.result_file_path = file_path
            
            # 解析結果數據
            self.current_dataset = result_data.get("dataset_name", "未知數據集")
            self.topics = result_data.get("topics", {})
            self.evaluation_results = result_data.get("evaluation", {})
            
            # 讀取面向向量（如果有）
            vectors_file = result_data.get("vectors_file")
            if vectors_file:
                vectors_path = os.path.join(os.path.dirname(file_path), vectors_file)
                if os.path.exists(vectors_path):
                    npz_data = np.load(vectors_path)
                    self.aspect_vectors = npz_data.get("aspect_vectors")
            
            # 更新主題選擇下拉框
            self._update_topic_selector()
            
            # 更新樣本ID範圍
            if self.aspect_vectors is not None:
                self.sample_id_spin.setMaximum(len(self.aspect_vectors) - 1)
            
            # 更新UI狀態
            self._update_result_info()
            self.generate_btn.setEnabled(True)
            
            # 清除現有的可視化
            self.interactive_view.setHtml("<center><h3>請點擊「生成可視化」按鈕</h3></center>")
            self.static_image_label.setText("尚未生成靜態可視化")
            self.data_label.setText("尚未生成數據視圖")
            
            # 禁用保存按鈕
            self.save_image_btn.setEnabled(False)
            self.save_html_btn.setEnabled(False)
            self.export_report_btn.setEnabled(True)
            
            # 提示信息
            self.status_message.emit(f"已載入結果文件: {os.path.basename(file_path)}", 3000)
            
        except Exception as e:
            logger.error(f"載入結果文件出錯: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "載入出錯", f"載入結果文件時出錯:\n{str(e)}")

    def _update_topic_selector(self):
        """更新主題選擇下拉框"""
        self.topic_select_combo.clear()
        self.topic_select_combo.addItem("所有主題")
        
        if self.topics:
            for topic_id in self.topics.keys():
                # 添加前5個關鍵詞作為主題標籤
                keywords = self.topics[topic_id][:5]
                topic_label = f"主題 {topic_id}: {', '.join(keywords)}"
                self.topic_select_combo.addItem(topic_label, topic_id)

    def _update_result_info(self):
        """更新結果信息標籤"""
        if self.current_dataset:
            info_text = f"當前數據集: {self.current_dataset}"
            
            if self.topics:
                info_text += f" | 主題數: {len(self.topics)}"
                
            if self.aspect_vectors is not None:
                info_text += f" | 向量數: {len(self.aspect_vectors)}"
                
            self.result_info_label.setText(info_text)
        else:
            self.result_info_label.setText("當前結果: 未載入")

    def _on_viz_type_changed(self, button):
        """可視化類型變更處理"""
        # 獲取選擇的按鈕ID
        button_id = self.viz_type_group.id(button)
        
        # 將標籤頁切換到相應的選項卡
        if button_id == 1:  # 主題分佈
            self.viz_options_stack.setCurrentIndex(0)
        elif button_id == 2:  # 向量聚類
            self.viz_options_stack.setCurrentIndex(1)
        elif button_id == 3:  # 主題關係網絡
            self.viz_options_stack.setCurrentIndex(2)
        elif button_id == 4:  # 詞雲
            self.viz_options_stack.setCurrentIndex(3)
        elif button_id == 5:  # 注意力熱圖
            self.viz_options_stack.setCurrentIndex(4)
        elif button_id == 6:  # 評估指標
            self.viz_options_stack.setCurrentIndex(5)

    def generate_visualization(self):
        """生成可視化"""
        if not self.current_dataset:
            QMessageBox.warning(self, "數據未載入", "請先載入結果文件後再生成可視化")
            return
            
        # 獲取選定的可視化類型
        viz_type = self.viz_type_group.checkedId()
        
        # 根據選擇類型進行相應的可視化
        try:
            self.status_message.emit("正在生成可視化...", 0)
            
            if viz_type == 1:  # 主題分佈
                self._generate_topic_distribution()
            elif viz_type == 2:  # 向量聚類
                self._generate_vector_clustering()
            elif viz_type == 3:  # 主題關係網絡
                self._generate_topic_network()
            elif viz_type == 4:  # 詞雲
                self._generate_word_cloud()
            elif viz_type == 5:  # 注意力熱圖
                self._generate_attention_heatmap()
            elif viz_type == 6:  # 評估指標
                self._generate_evaluation_viz()
            else:
                QMessageBox.warning(self, "無效選擇", "請選擇一種可視化類型")
                return
                
            # 更新UI狀態
            self.save_image_btn.setEnabled(True)
            self.save_html_btn.setEnabled(True)
            
            # 提示信息
            self.status_message.emit("可視化生成完成", 3000)
            
        except Exception as e:
            logger.error(f"生成可視化出錯: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "生成出錯", f"生成可視化時出錯:\n{str(e)}")

    def _generate_topic_distribution(self):
        """生成主題分佈可視化"""
        if not self.topics or self.aspect_vectors is None:
            QMessageBox.warning(self, "缺少數據", "缺少主題或向量數據，無法生成主題分佈")
            return
            
        # 取得選項
        show_labels = self.cb_show_topic_labels.isChecked()
        interactive = self.cb_interactive_topic.isChecked()
        use_3d = self.cb_3d_topic.isChecked()
        
        # 設置標題
        self.viz_title_label.setText("主題分佈可視化")
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_topic_distribution(
            topics=self.topics,
            vectors=self.aspect_vectors,
            show_labels=show_labels,
            interactive=interactive,
            use_3d=use_3d,
            output_dir=self.config.get("paths", {}).get("visualizations_dir", "./visualizations")
        )
        
        # 保存結果
        self.visualization_results["topic_distribution"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("topic_distribution")

    def _generate_vector_clustering(self):
        """生成向量聚類可視化"""
        if self.aspect_vectors is None:
            QMessageBox.warning(self, "缺少數據", "缺少向量數據，無法生成聚類可視化")
            return
            
        # 取得選項
        algorithm = self.cluster_algorithm_combo.currentText()
        n_clusters = self.cluster_count_spin.value()
        interactive = self.cb_interactive_cluster.isChecked()
        
        # 設置標題
        self.viz_title_label.setText("向量聚類可視化")
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_vector_clustering(
            vectors=self.aspect_vectors,
            algorithm=algorithm,
            n_clusters=n_clusters,
            interactive=interactive,
            output_dir=self.config.get("paths", {}).get("visualizations_dir", "./visualizations")
        )
        
        # 保存結果
        self.visualization_results["vector_clustering"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("vector_clustering")

    def _generate_topic_network(self):
        """生成主題關係網絡可視化"""
        if not self.topics or self.aspect_vectors is None:
            QMessageBox.warning(self, "缺少數據", "缺少主題或向量數據，無法生成關係網絡")
            return
            
        # 取得選項
        show_weights = self.cb_show_weights.isChecked()
        edge_threshold = self.edge_threshold_slider.value() / 100.0
        interactive = self.cb_interactive_network.isChecked()
        
        # 設置標題
        self.viz_title_label.setText("主題關係網絡可視化")
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_topic_network(
            topics=self.topics,
            vectors=self.aspect_vectors,
            show_weights=show_weights,
            edge_threshold=edge_threshold,
            interactive=interactive,
            output_dir=self.config.get("paths", {}).get("visualizations_dir", "./visualizations")
        )
        
        # 保存結果
        self.visualization_results["topic_network"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("topic_network")

    def _generate_word_cloud(self):
        """生成詞雲可視化"""
        if not self.topics:
            QMessageBox.warning(self, "缺少數據", "缺少主題數據，無法生成詞雲")
            return
            
        # 取得選項
        topic_idx = self.topic_select_combo.currentData()  # 如果是"所有主題"，則為None
        color_scheme = self.wordcloud_color_combo.currentText()
        
        # 設置標題
        topic_text = f"主題 {topic_idx}" if topic_idx else "所有主題"
        self.viz_title_label.setText(f"{topic_text} 詞雲可視化")
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_word_cloud(
            topics=self.topics,
            topic_idx=topic_idx,
            color_scheme=color_scheme,
            output_dir=self.config.get("paths", {}).get("visualizations_dir", "./visualizations")
        )
        
        # 保存結果
        self.visualization_results["word_cloud"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("word_cloud")

    def _generate_attention_heatmap(self):
        """生成注意力熱圖可視化"""
        # 注意：此功能可能需要額外數據，但為了示例，我們假設可直接生成
        
        # 取得選項
        attention_type = self.attention_type_combo.currentText()
        sample_id = self.sample_id_spin.value()
        
        # 設置標題
        self.viz_title_label.setText(f"{attention_type} (樣本 {sample_id}) 熱圖可視化")
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_attention_heatmap(
            attention_type=attention_type,
            sample_id=sample_id,
            output_dir=self.config.get("paths", {}).get("visualizations_dir", "./visualizations")
        )
        
        # 保存結果
        self.visualization_results["attention_heatmap"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("attention_heatmap")

    def _generate_evaluation_viz(self):
        """生成評估指標可視化"""
        if not self.evaluation_results:
            QMessageBox.warning(self, "缺少數據", "缺少評估數據，無法生成評估可視化")
            return
            
        # 取得選項
        show_all = self.cb_show_all_metrics.isChecked()
        show_chart = self.cb_show_chart.isChecked()
        
        # 設置標題
        self.viz_title_label.setText("模型評估指標可視化")
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_evaluation_visualization(
            evaluation_results=self.evaluation_results,
            show_all=show_all,
            show_chart=show_chart,
            output_dir=self.config.get("paths", {}).get("visualizations_dir", "./visualizations")
        )
        
        # 保存結果
        self.visualization_results["evaluation"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("evaluation")

    def _update_visualization_display(self, viz_key):
        """更新可視化顯示
        
        Args:
            viz_key: 可視化結果的鍵名
        """
        if viz_key not in self.visualization_results:
            return
            
        viz_data = self.visualization_results[viz_key]
        
        # 更新互動式視圖
        if "html" in viz_data and viz_data["html"]:
            self.interactive_view.setHtml(viz_data["html"])
            self.result_tabs.setCurrentIndex(0)  # 切換到互動視圖標籤頁
        
        # 更新靜態圖片
        if "image_path" in viz_data and viz_data["image_path"] and os.path.exists(viz_data["image_path"]):
            pixmap = QPixmap(viz_data["image_path"])
            # 根據窗口大小縮放圖片
            pixmap = pixmap.scaled(
                self.static_image_label.width() - 20,
                self.static_image_label.height() - 20,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.static_image_label.setPixmap(pixmap)
            self.static_image_label.setAlignment(Qt.AlignCenter)
        
        # 更新數據視圖
        if "data_html" in viz_data and viz_data["data_html"]:
            self.data_label.setText("")
            self.data_label.setHtml(viz_data["data_html"])
            
    def save_visualization(self):
        """保存當前可視化（可由外部調用）"""
        viz_type = self.viz_type_group.checkedId()
        
        if viz_type == 1:  # 主題分佈
            viz_key = "topic_distribution"
        elif viz_type == 2:  # 向量聚類
            viz_key = "vector_clustering" 
        elif viz_type == 3:  # 主題關係網絡
            viz_key = "topic_network"
        elif viz_type == 4:  # 詞雲
            viz_key = "word_cloud"
        elif viz_type == 5:  # 注意力熱圖
            viz_key = "attention_heatmap"
        elif viz_type == 6:  # 評估指標
            viz_key = "evaluation"
        else:
            QMessageBox.warning(self, "保存失敗", "未選擇可視化類型或尚未生成可視化")
            return False
        
        if viz_key in self.visualization_results and "image_path" in self.visualization_results[viz_key]:
            return self.visualization_results[viz_key]["image_path"]
        else:
            QMessageBox.warning(self, "保存失敗", "當前沒有可用的可視化結果")
            return False

    def save_visualization_image(self):
        """保存可視化圖片"""
        # 檢查是否有當前可視化類型的結果
        viz_type = self.viz_type_group.checkedId()
        
        if viz_type == 1:  # 主題分佈
            viz_key = "topic_distribution"
            default_name = "topic_distribution"
        elif viz_type == 2:  # 向量聚類
            viz_key = "vector_clustering"
            default_name = "vector_clustering"
        elif viz_type == 3:  # 主題關係網絡
            viz_key = "topic_network"
            default_name = "topic_network"
        elif viz_type == 4:  # 詞雲
            viz_key = "word_cloud"
            default_name = "word_cloud"
        elif viz_type == 5:  # 注意力熱圖
            viz_key = "attention_heatmap"
            default_name = "attention_heatmap"
        elif viz_type == 6:  # 評估指標
            viz_key = "evaluation"
            default_name = "evaluation"
        else:
            QMessageBox.warning(self, "保存失敗", "未選擇可視化類型或尚未生成可視化")
            return
        
        # 檢查是否有圖片可保存
        if viz_key not in self.visualization_results or "image_path" not in self.visualization_results[viz_key]:
            QMessageBox.warning(self, "保存失敗", "當前可視化沒有可用的圖片")
            return
            
        # 獲取源圖片路徑
        source_path = self.visualization_results[viz_key]["image_path"]
        if not os.path.exists(source_path):
            QMessageBox.warning(self, "保存失敗", "找不到源圖片文件")
            return
            
        # 選擇保存位置
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_name = f"{default_name}_{self.current_dataset}_{timestamp}.png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存圖片",
            os.path.join(self.config.get("paths", {}).get("exports_dir", "./exports"), suggested_name),
            "PNG圖片 (*.png);;JPEG圖片 (*.jpg);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 保存圖片
            import shutil
            shutil.copy2(source_path, file_path)
            
            self.status_message.emit(f"圖片已保存至: {file_path}", 3000)
            
        except Exception as e:
            logger.error(f"保存圖片出錯: {str(e)}")
            QMessageBox.critical(self, "保存出錯", f"保存圖片時出錯:\n{str(e)}")

    def save_visualization_html(self):
        """保存可視化HTML"""
        # 檢查是否有當前可視化類型的結果
        viz_type = self.viz_type_group.checkedId()
        
        if viz_type == 1:  # 主題分佈
            viz_key = "topic_distribution"
            default_name = "topic_distribution"
        elif viz_type == 2:  # 向量聚類
            viz_key = "vector_clustering"
            default_name = "vector_clustering"
        elif viz_type == 3:  # 主題關係網絡
            viz_key = "topic_network"
            default_name = "topic_network"
        elif viz_type == 4:  # 詞雲
            viz_key = "word_cloud"
            default_name = "word_cloud"
        elif viz_type == 5:  # 注意力熱圖
            viz_key = "attention_heatmap"
            default_name = "attention_heatmap"
        elif viz_type == 6:  # 評估指標
            viz_key = "evaluation"
            default_name = "evaluation"
        else:
            QMessageBox.warning(self, "保存失敗", "未選擇可視化類型或尚未生成可視化")
            return
        
        # 檢查是否有HTML可保存
        if viz_key not in self.visualization_results or "html" not in self.visualization_results[viz_key]:
            QMessageBox.warning(self, "保存失敗", "當前可視化沒有可用的HTML內容")
            return
            
        # 選擇保存位置
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_name = f"{default_name}_{self.current_dataset}_{timestamp}.html"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存HTML",
            os.path.join(self.config.get("paths", {}).get("exports_dir", "./exports"), suggested_name),
            "HTML文件 (*.html);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 保存HTML
            html_content = self.visualization_results[viz_key]["html"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.status_message.emit(f"HTML已保存至: {file_path}", 3000)
            
        except Exception as e:
            logger.error(f"保存HTML出錯: {str(e)}")
            QMessageBox.critical(self, "保存出錯", f"保存HTML時出錯:\n{str(e)}")

    def export_report_dialog(self):
        """打開導出報告對話框"""
        if not self.current_dataset:
            QMessageBox.warning(self, "無法導出", "請先載入結果文件")
            return
        
        # 選擇保存位置
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_name = f"report_{self.current_dataset}_{timestamp}.html"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "導出分析報告",
            os.path.join(self.config.get("paths", {}).get("exports_dir", "./exports"), suggested_name),
            "HTML報告 (*.html);;PDF報告 (*.pdf);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            self.export_report(file_path)
            
        except Exception as e:
            logger.error(f"導出報告出錯: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "導出出錯", f"導出報告時出錯:\n{str(e)}")

    def export_report(self, file_path):
        """導出分析報告
        
        Args:
            file_path: 報告文件保存路徑
        """
        if not self.current_dataset or not self.topics:
            QMessageBox.warning(self, "無法導出", "缺少必要的數據，無法導出完整報告")
            return
            
        try:
            # 讀取原始結果數據
            with open(self.result_file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # 使用可視化模組生成報告
            self.visualizer.export_report(
                file_path=file_path,
                dataset_name=self.current_dataset,
                topics=self.topics,
                aspect_vectors=self.aspect_vectors,
                evaluation=self.evaluation_results,
                params=result_data.get("parameters", {}),
                visualization_results=self.visualization_results
            )
            
            # 成功提示
            self.status_message.emit(f"報告已導出至 {file_path}", 5000)
            
            # 嘗試打開報告
            if file_path.endswith('.html'):
                import webbrowser
                webbrowser.open(file_path)
            
        except Exception as e:
            logger.error(f"導出報告出錯: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "導出出錯", f"導出分析報告時出錯:\n{str(e)}")
            raise e  # 重新拋出異常，讓調用者知道出錯了

    def show_topic_visualization(self):
        """顯示主題可視化（可由外部調用）"""
        # 選擇主題分佈可視化
        self.rb_topic_distribution.setChecked(True)
        self._on_viz_type_changed(self.rb_topic_distribution)
        
        # 生成可視化
        if self.current_dataset and self.topics:
            self.generate_visualization()
            
    def show_attention_visualization(self):
        """顯示注意力可視化（可由外部調用）"""
        # 選擇注意力熱圖可視化
        self.rb_attention_heatmap.setChecked(True)
        self._on_viz_type_changed(self.rb_attention_heatmap)
        
        # 生成可視化
        if self.current_dataset:
            self.generate_visualization()

    def generate_visualizations(self):
        """生成所有可視化（可由外部調用）"""
        if not self.current_dataset or not self.topics:
            QMessageBox.warning(self, "無法生成", "請先載入結果文件")
            return
            
        # 先生成主題分佈
        self.rb_topic_distribution.setChecked(True)
        self._on_viz_type_changed(self.rb_topic_distribution)
        self.generate_visualization()
        
        # 再生成詞雲
        self.rb_word_cloud.setChecked(True)
        self._on_viz_type_changed(self.rb_word_cloud)
        self.generate_visualization()
        
        # 如果有評估結果，生成評估可視化
        if self.evaluation_results:
            self.rb_evaluation.setChecked(True)
            self._on_viz_type_changed(self.rb_evaluation)
            self.generate_visualization()
            
        # 返回到主題分佈
        self.rb_topic_distribution.setChecked(True)
        self._on_viz_type_changed(self.rb_topic_distribution)
        
        self.status_message.emit("已生成多個可視化圖表", 3000)

    def on_settings_changed(self):
        """處理設定變更"""
        # 重新載入配置到可視化模組
        self.visualizer.update_config(self.config.get("visualization"))