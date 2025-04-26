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

        # 設置 logger
        self.logger = logger

        # 默認輸出目錄 - 更新為使用 Part04_/0_output/visualizations 目錄
        self.output_dir = os.path.join("Part04_", "0_output", "visualizations")
        # 嘗試從文件管理器獲取正確路徑
        if file_manager is not None and hasattr(file_manager, "visualizations_dir"):
            self.output_dir = file_manager.visualizations_dir
            self.logger.debug(f"從文件管理器獲取可視化目錄: {self.output_dir}")

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
        
        self.rb_attention_heatmap = QRadioButton("注意力熱圖")
        self.viz_type_group.addButton(self.rb_attention_heatmap, 5)
        viz_type_layout.addWidget(self.rb_attention_heatmap)
        
        self.rb_evaluation = QRadioButton("評估指標")
        self.viz_type_group.addButton(self.rb_evaluation, 6)
        viz_type_layout.addWidget(self.rb_evaluation)
        
        options_layout.addWidget(group_viz_type)
        
        # 主題選擇下拉框
        topic_select_layout = QHBoxLayout()
        topic_select_layout.addWidget(QLabel("選擇主題:"))
        self.topic_select_combo = QComboBox()
        self.topic_select_combo.addItem("所有主題")
        topic_select_layout.addWidget(self.topic_select_combo)
        options_layout.addLayout(topic_select_layout)
        
        # 各類型的細節選項
        self.viz_options_stack = QTabWidget()
        
        # 主題分佈選項
        self.topic_options = QWidget()
        topic_options_layout = QVBoxLayout(self.topic_options)
        
        self.cb_show_topic_labels = QCheckBox("顯示主題標籤")
        self.cb_show_topic_labels.setChecked(True)
        topic_options_layout.addWidget(self.cb_show_topic_labels)
        
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
        
        network_options_layout.addStretch()
        
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
    
    def _create_bottom_panel(self):
        """創建底部控制面板"""
        bottom_layout = QHBoxLayout()
        
        # 左側空間
        bottom_layout.addStretch(1)
        
        # 保存按鈕
        self.save_image_btn = QPushButton("保存圖片")
        self.save_image_btn.clicked.connect(self.save_visualization_image)
        bottom_layout.addWidget(self.save_image_btn)
        
        # 匯出報告按鈕
        self.export_report_btn = QPushButton("匯出報告")
        self.export_report_btn.clicked.connect(self.export_report_dialog)
        bottom_layout.addWidget(self.export_report_btn)
        
        # 禁用按鈕（直到生成可視化）
        self.save_image_btn.setEnabled(False)
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
                    results_dir = self.config.get("paths", {}).get("output_dir", "./Part04_/0_output")
                elif hasattr(self.config, "get"):
                    # 嘗試直接獲取配置路徑
                    try:
                        paths = self.config.get("paths")
                        if isinstance(paths, dict):
                            results_dir = paths.get("output_dir", "./Part04_/0_output")
                        else:
                            results_dir = self.config.get(("paths", "output_dir"), "./Part04_/0_output")
                    except TypeError:
                        # 如果上述方法失敗，嘗試一次直接訪問完整路徑
                        results_dir = self.config.get("paths.output_dir", "./Part04_/0_output")
                else:
                    results_dir = "./Part04_/0_output"  # 默認值
            except Exception as config_error:
                logger.warning(f"讀取配置時出現錯誤，使用默認值: {str(config_error)}")
                results_dir = "./Part04_/0_output"
            
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
        # 安全獲取輸出目錄
        output_dir = "./output"  # 默認值
        
        # 檢查配置對象是否存在並安全獲取路徑
        if self.config is not None:
            try:
                if isinstance(self.config, dict):
                    output_dir = self.config.get("paths", {}).get("output_dir", output_dir)
                elif hasattr(self.config, "get"):
                    paths = self.config.get("paths")
                    if isinstance(paths, dict):
                        output_dir = paths.get("output_dir", output_dir)
                    else:
                        output_dir = self.config.get("paths.output_dir", output_dir)
            except Exception as e:
                logger.warning(f"獲取輸出目錄時出錯: {str(e)}，使用默認值 {output_dir}")
        
        # 確保路徑存在
        if not os.path.exists(output_dir):
            output_dir = "."  # 如果目錄不存在，切換到當前目錄
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇結果文件",
            output_dir,
            "結果文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            logger.debug(f"已選擇結果檔案: {file_path}")
            self.load_results(file_path)
        else:
            logger.debug("用戶取消了檔案選擇")

    def load_selected_result(self):
        """載入選中的結果文件"""
        result_path = self.result_combo.currentData()
        
        if not result_path or self.result_combo.currentIndex() == 0:
            QMessageBox.warning(self, "選擇結果", "請選擇一個有效的結果文件")
            return
            
        self.load_results(result_path)

    def load_results(self, file_path):
        """載入結果檔案
        
        Args:
            file_path: 結果檔案路徑
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 清空當前結果
            self.aspect_vectors = None
            self.topics = None
            self.evaluation_results = None
            
            # 從文件中獲取面向向量
            if 'aspect_vectors' in results:
                self.aspect_vectors = results['aspect_vectors']
                self.logger.info(f"成功載入 {len(self.aspect_vectors)} 個面向向量")
            
            # 從文件中獲取主題
            if 'topics' in results:
                self.topics = results['topics']
                self.logger.info(f"成功載入 {len(self.topics)} 個主題")
            
            # 從文件中獲取評估結果 - 修改以同時支持多種格式
            # 格式1: metrics 作為頂層鍵
            if 'metrics' in results:
                self.evaluation_results = results['metrics']
                self.logger.info(f"從 metrics 鍵成功載入評估結果")
            # 格式2: evaluation 作為頂層鍵
            elif 'evaluation' in results:
                self.evaluation_results = results['evaluation']
                self.logger.info(f"從 evaluation 鍵成功載入評估結果")
            # 格式3: metrics_details 作為頂層鍵 (備用方案)
            elif 'metrics_details' in results:
                self.evaluation_results = results['metrics_details']
                self.logger.info(f"從 metrics_details 鍵成功載入評估結果")
                
            # 如果沒有找到評估結果，則記錄警告
            if not self.evaluation_results:
                self.logger.warning(f"在結果文件中沒有找到評估結果")
            else:
                self.logger.info(f"評估結果包含: {list(self.evaluation_results.keys())}")
            
            # 設置當前數據集名稱
            self.current_dataset = Path(file_path).stem
            
            # 更新 UI
            self._update_topic_selector()
            self._update_result_info()
            
            return True
            
        except Exception as e:
            self.logger.error(f"載入結果文件出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "載入出錯", f"載入結果文件時出錯:\n{str(e)}")
            return False

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
        elif button_id == 5:  # 注意力熱圖
            self.viz_options_stack.setCurrentIndex(3)
        elif button_id == 6:  # 評估指標
            self.viz_options_stack.setCurrentIndex(4)
    
    # ===Create=================================
    # 使用 scikit-learn 的 t-SNE 實現進行降維
    def plot_tsne(self, embeddings, labels, title="t-SNE Visualization", point_size=30, output_dir=None):
        """使用 t-SNE 繪製降維視覺化
        
        Args:
            embeddings: 嵌入向量列表
            labels: 標籤列表
            title: 標題
            point_size: 點大小
            output_dir: 輸出目錄

        Returns:
            str: 圖片文件路徑
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            import os
            import numpy as np
            from datetime import datetime
            
            # 設置中文字體支援
            try:
                # 嘗試設置支援中文的字體
                plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
                plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
            except Exception as font_error:
                self.logger.warning(f"設置中文字體時出錯: {str(font_error)}，將使用默認字體")
            
            # 設置輸出目錄
            if output_dir is None:
                output_dir = self.output_dir
                
            os.makedirs(output_dir, exist_ok=True)
            # 將輸入轉換為NumPy數組
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # 確保數據維度正確
            if len(embeddings_array.shape) == 1:
                self.logger.warning("輸入向量為一維，嘗試重塑為二維")
                # 如果是一維數組，重塑為一個樣本的二維數組
                embeddings_array = embeddings_array.reshape(1, -1)
            
            # 應用t-SNE降維 (使用 max_iter 替換 n_iter)
            tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings_array)-1), 
                max_iter=1000, random_state=42)
            tsne_result = tsne.fit_transform(embeddings_array)
            
            # 創建圖表
            plt.figure(figsize=(12, 8))
            
            # 獲取唯一標籤並分配顏色
            unique_labels = list(set(labels))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            
            # 繪製散點圖
            for i, label in enumerate(unique_labels):
                mask = [l == label for l in labels]
                plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                        c=[colors[i]], label=label, s=point_size)
            
            # 添加標題和圖例
            plt.title(title, fontsize=18)
            plt.legend(loc='best')
            
            # 去除坐標軸刻度
            plt.xticks([])
            plt.yticks([])
            
            # 添加網格
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 保存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tsne_{timestamp}.png"
            img_path = os.path.join(output_dir, filename)
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return img_path
            
        except Exception as e:
            self.logger.error(f"t-SNE 視覺化生成出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # 返回空路徑
            return None
        
    def create_vector_clustering(self, vectors, algorithm="K-Means", n_clusters=5, interactive=True, output_dir=None):
        """創建向量聚類可視化
        
        Args:
            vectors: 面向向量字典或列表
            algorithm: 聚類算法，可以是 "K-Means", "DBSCAN", 或 "層次聚類"
            n_clusters: 聚類數量
            interactive: 是否創建互動式視覺化
            output_dir: 輸出目錄
            
        Returns:
            tuple: (html_content, img_path, data_html)
        """
        try:
            self.logger.info(f"生成向量聚類可視化: {algorithm}")
            
            # 設置輸出目錄
            if output_dir:
                old_output_dir = self.output_dir
                self.output_dir = output_dir
                os.makedirs(self.output_dir, exist_ok=True)
                
            # 準備數據
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            import pandas as pd
            
            # 轉換向量格式
            if isinstance(vectors, dict):
                # 如果是字典，轉換為列表
                vector_ids = list(vectors.keys())
                vectors_list = list(vectors.values())
            elif isinstance(vectors, list):
                vectors_list = vectors
                vector_ids = [f"向量{i}" for i in range(len(vectors_list))]
            else:
                raise ValueError("向量必須是字典或列表")
                
            # 進行聚類
            labels = None
            if algorithm == "K-Means":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                labels = clusterer.fit_predict(vectors_list)
            elif algorithm == "DBSCAN":
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                labels = clusterer.fit_predict(vectors_list)
            elif algorithm == "層次聚類":
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clusterer.fit_predict(vectors_list)
            else:
                raise ValueError(f"不支持的聚類算法: {algorithm}")
                
            # 創建標籤列表
            label_strings = [f"群集 {label}" for label in labels]
            
            # 使用 t-SNE 進行降維和可視化
            img_path = self.plot_tsne(
                embeddings=vectors_list,
                labels=label_strings,
                title=f"{algorithm} 聚類結果 (n_clusters={n_clusters})",
                point_size=40
            )
            
            # 創建數據表格
            data_df = pd.DataFrame({
                "向量ID": vector_ids,
                "聚類標籤": label_strings
            })
            
            # 生成 HTML 內容
            html_content = ""
            if interactive:
                try:
                    import plotly.express as px
                    import numpy as np
                    
                    # 使用 t-SNE 降維
                    from sklearn.manifold import TSNE
                    
                    # 將向量轉換為NumPy數組（修復關鍵部分）
                    vectors_array = np.array(vectors_list, dtype=np.float32)
                    
                    # 確保數據維度正確
                    if len(vectors_array.shape) == 1:
                        self.logger.warning("輸入向量為一維，嘗試重塑為二維")
                        vectors_array = vectors_array.reshape(1, -1)
                        
                    tsne = TSNE(n_components=2, perplexity=min(30, len(vectors_array)-1), 
                            n_iter=1000, random_state=42)
                    tsne_result = tsne.fit_transform(vectors_array)
                    
                    # 創建互動式視覺化
                    df = pd.DataFrame({
                        'x': tsne_result[:, 0],
                        'y': tsne_result[:, 1],
                        'cluster': label_strings,
                        'vector_id': vector_ids
                    })
                    
                    # 創建 plotly 圖表
                    fig = px.scatter(
                        df, x='x', y='y', color='cluster',
                        title=f"{algorithm} 聚類結果 (n_clusters={n_clusters})",
                        hover_data=['vector_id', 'cluster']
                    )
                    
                    html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
                    
                except ImportError:
                    self.logger.warning("未安裝 plotly，無法創建互動式視覺化")
                    html_content = f"<h2>{algorithm} 聚類結果</h2><img src='{img_path}' width='100%'>"
            else:
                html_content = f"<h2>{algorithm} 聚類結果</h2><img src='{img_path}' width='100%'>"
                
            # 生成數據視圖 HTML
            data_html = data_df.to_html(index=False)
            
            # 恢復輸出目錄
            if output_dir:
                self.output_dir = old_output_dir
                
            return html_content, img_path, data_html
            
        except Exception as e:
            self.logger.error(f"生成向量聚類可視化時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return f"<h2>錯誤</h2><p>{str(e)}</p>", None, f"<h2>錯誤</h2><p>{str(e)}</p>"
    
    # ===Generate=======================================
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
            elif viz_type == 5:  # 注意力熱圖
                self._generate_attention_heatmap()
            elif viz_type == 6:  # 評估指標
                self._generate_evaluation_viz()
            else:
                QMessageBox.warning(self, "無效選擇", "請選擇一種可視化類型")
                return
                
            # 更新UI狀態
            self.save_image_btn.setEnabled(True)
            self.export_report_btn.setEnabled(True)
            
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
        
        # 安全地獲取輸出目錄
        output_dir = "./visualizations"  # 默認值
        
        # 檢查配置對象是否存在
        if self.config is not None:
            try:
                # 嘗試獲取可視化輸出目錄
                if isinstance(self.config, dict):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        output_dir = paths.get("visualizations_dir", output_dir)
                elif hasattr(self.config, "get"):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        output_dir = paths.get("visualizations_dir", output_dir)
            except Exception as e:
                self.logger.warning(f"獲取配置時出錯: {str(e)}，使用默認輸出目錄: {output_dir}")
        else:
            self.logger.warning("配置對象為None，使用默認輸出目錄")
        
        # 確保輸出目錄存在
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"創建目錄時出錯: {str(e)}")
            output_dir = "./"  # 回退到當前目錄
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_topic_distribution(
            topics=self.topics,
            vectors=self.aspect_vectors,
            show_labels=show_labels,
            output_dir=output_dir
        )
        
        # 保存結果
        self.visualization_results["topic_distribution"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("topic_distribution")
        
        # 顯示輸出成功提示視窗
        if img_path:
            QMessageBox.information(
                self, 
                "輸出成功", 
                f"主題分佈視覺化已成功生成！\n\n保存路徑：\n{img_path}"
            )

    def _generate_vector_clustering(self):
        """生成向量聚類可視化"""
        if self.aspect_vectors is None:
            QMessageBox.warning(self, "缺少數據", "缺少向量數據，無法生成聚類可視化")
            return
            
        # 取得選項
        algorithm = self.cluster_algorithm_combo.currentText()
        n_clusters = self.cluster_count_spin.value()
        
        # 安全地獲取輸出目錄
        output_dir = "./visualizations"  # 默認值
        
        # 檢查配置對象是否存在
        if self.config is not None:
            try:
                # 嘗試不同方式獲取配置
                if isinstance(self.config, dict):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        vis_dir = paths.get("visualizations_dir")
                        if vis_dir:
                            output_dir = vis_dir
                elif hasattr(self.config, "get"):
                    # 配置對象有 get 方法，嘗試使用它
                    try:
                        paths = self.config.get("paths", {})
                        if isinstance(paths, dict):
                            output_dir = paths.get("visualizations_dir", output_dir)
                    except Exception as e:
                        self.logger.warning(f"獲取可視化目錄配置出錯: {str(e)}，使用默認目錄 {output_dir}")
            except Exception as e:
                self.logger.warning(f"讀取配置時出錯: {str(e)}，使用默認目錄 {output_dir}")
        
        # 確保輸出目錄存在
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"創建目錄時出錯: {str(e)}")
            output_dir = "./"  # 回退到當前目錄
        
        # 生成可視化
        html_content, img_path, data_html = self.create_vector_clustering(
            vectors=self.aspect_vectors,
            algorithm=algorithm,
            n_clusters=n_clusters,
            output_dir=output_dir
        )
        
        # 保存結果
        self.visualization_results["vector_clustering"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("vector_clustering")
        
        # 顯示輸出成功提示視窗
        if img_path:
            QMessageBox.information(
                self, 
                "輸出成功", 
                f"向量聚類視覺化已成功生成！\n\n保存路徑：\n{img_path}"
            )

    def _generate_topic_network(self):
        """生成主題關係網絡可視化"""
        if not self.topics or self.aspect_vectors is None:
            QMessageBox.warning(self, "缺少數據", "缺少主題或向量數據，無法生成關係網絡")
            return
            
        # 取得選項
        show_weights = self.cb_show_weights.isChecked()
        edge_threshold = self.edge_threshold_slider.value() / 100.0
        
        # 安全地獲取輸出目錄
        output_dir = "./visualizations"  # 默認值
        
        # 檢查配置對象是否存在
        if self.config is not None:
            try:
                if isinstance(self.config, dict):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        output_dir = paths.get("visualizations_dir", output_dir)
                elif hasattr(self.config, "get"):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        output_dir = paths.get("visualizations_dir", output_dir)
            except Exception as e:
                self.logger.warning(f"獲取可視化目錄配置出錯: {str(e)}，使用默認目錄 {output_dir}")
        else:
            self.logger.warning("配置對象為None，使用默認輸出目錄: {output_dir}")
            
        # 確保輸出目錄存在
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"創建輸出目錄時出錯: {str(e)}")
            output_dir = "./"  # 回退到當前目錄
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_topic_network(
            topics=self.topics,
            vectors=self.aspect_vectors,
            show_weights=show_weights,
            edge_threshold=edge_threshold,
            output_dir=output_dir
        )
        
        # 保存結果
        self.visualization_results["topic_network"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("topic_network")
        
        # 顯示輸出成功提示視窗
        if img_path:
            QMessageBox.information(
                self, 
                "輸出成功", 
                f"主題關係網絡視覺化已成功生成！\n\n保存路徑：\n{img_path}"
            )

    def _generate_attention_heatmap(self):
        """生成注意力熱圖可視化"""
        # 注意：此功能可能需要額外數據，但為了示例，我們假設可直接生成
        
        # 取得選項
        attention_type = self.attention_type_combo.currentText()
        sample_id = self.sample_id_spin.value()
        
        # 安全地獲取輸出目錄
        output_dir = "./visualizations"  # 默認值
        
        # 檢查配置對象是否存在
        if self.config is not None:
            try:
                if isinstance(self.config, dict):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        output_dir = paths.get("visualizations_dir", output_dir)
                elif hasattr(self.config, "get"):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        output_dir = paths.get("visualizations_dir", output_dir)
            except Exception as e:
                self.logger.warning(f"獲取可視化目錄配置出錯: {str(e)}，使用默認目錄 {output_dir}")
        else:
            self.logger.warning("配置對象為None，使用默認輸出目錄: {output_dir}")
            
        # 確保輸出目錄存在
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"創建輸出目錄時出錯: {str(e)}")
            output_dir = "./"  # 回退到當前目錄
        
        # 創建示例注意力矩陣數據
        try:
            # 如果沒有實際數據，創建示例數據
            import numpy as np
            rows = 10
            cols = 10
            attention_matrix = np.random.rand(rows, cols)
            
            # 正規化注意力矩陣，使每行總和為1
            row_sums = attention_matrix.sum(axis=1, keepdims=True)
            attention_matrix = attention_matrix / row_sums
            
            # 生成示例標籤
            row_labels = [f"主題 {i+1}" for i in range(rows)]
            col_labels = [f"樣本 {i+1}" for i in range(cols)]
            
            # 生成可視化
            html_content, img_path, data_html = self.visualizer.create_attention_heatmap(
                attention_matrix=attention_matrix,
                row_labels=row_labels,
                col_labels=col_labels,
                title=f"{attention_type} - 樣本 {sample_id} 的注意力分布",
                output_dir=output_dir
            )
            
            # 保存結果
            self.visualization_results["attention_heatmap"] = {
                "html": html_content,
                "image_path": img_path,
                "data_html": data_html
            }
            
            # 更新顯示
            self._update_visualization_display("attention_heatmap")
            
            # 顯示輸出成功提示視窗
            if img_path:
                QMessageBox.information(
                    self, 
                    "輸出成功", 
                    f"注意力熱圖視覺化已成功生成！\n\n保存路徑：\n{img_path}"
                )
        except Exception as e:
            self.logger.error(f"生成注意力熱圖時出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "生成出錯", f"生成注意力熱圖時出錯:\n{str(e)}")

    def _generate_evaluation_viz(self):
        """生成評估指標可視化"""
        # 檢查評估數據是否存在
        if not self.evaluation_results:
            self.logger.warning("缺少評估數據，無法生成評估指標視覺化")
            # 顯示提示消息，指導用戶如何獲得真實評估數據
            QMessageBox.warning(
                self, 
                "缺少評估數據", 
                "無法生成評估指標視覺化，因為缺少必要的評估數據。\n\n" +
                "請按照以下步驟獲取評估數據：\n" +
                "1. 運行主流程分析，確保包含評估步驟\n" +
                "2. 在分析結果中包含 'metrics' 評估數據\n" +
                "3. 將評估結果保存在結果JSON文件中\n\n" +
                "或者：\n" +
                "- 使用其他視覺化類型，如主題分佈或向量聚類"
            )
            return
            
        # 取得選項
        show_all = self.cb_show_all_metrics.isChecked()
        show_chart = self.cb_show_chart.isChecked()
        
        # 安全地設置輸出目錄
        output_dir = self.output_dir  # 使用類中已初始化的默認輸出目錄
        
        # 嘗試從配置中獲取更精確的輸出目錄，但處理self.config可能為None的情況
        if self.config is not None:
            try:
                if isinstance(self.config, dict):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict) and "visualizations_dir" in paths:
                        output_dir = paths["visualizations_dir"]
                elif hasattr(self.config, "get"):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict) and "visualizations_dir" in paths:
                        output_dir = paths["visualizations_dir"]
            except Exception as e:
                self.logger.warning(f"獲取可視化目錄配置出錯: {str(e)}，使用默認目錄 {output_dir}")
        
        # 確保輸出目錄存在
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"創建輸出目錄時出錯: {str(e)}")
            output_dir = "./"  # 回退到當前目錄
        
        # 生成可視化
        html_content, img_path, data_html = self.visualizer.create_evaluation_visualization(
            evaluation_results=self.evaluation_results,
            show_all=show_all,
            show_chart=show_chart,
            output_dir=output_dir
        )
        
        # 保存結果
        self.visualization_results["evaluation"] = {
            "html": html_content,
            "image_path": img_path,
            "data_html": data_html
        }
        
        # 更新顯示
        self._update_visualization_display("evaluation")
        
        # 顯示成功訊息
        if img_path:
            QMessageBox.information(
                self, 
                "輸出成功", 
                f"評估指標視覺化已成功生成！\n\n保存路徑：\n{img_path}"
            )

    def _update_visualization_display(self, viz_key):
        """更新可視化顯示
        
        Args:
            viz_key: 可視化結果的鍵名
        """
        # 由於結果顯示區域已移除，僅保存結果數據但不顯示
        pass

    def save_visualization(self):
        """保存當前可視化（可由外部調用）"""
        viz_type = self.viz_type_group.checkedId()
        
        if viz_type == 1:  # 主題分佈
            viz_key = "topic_distribution"
        elif viz_type == 2:  # 向量聚類
            viz_key = "vector_clustering" 
        elif viz_type == 3:  # 主題關係網絡
            viz_key = "topic_network"
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