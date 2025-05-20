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
import matplotlib.pyplot as plt  # 添加 matplotlib.pyplot 導入
import seaborn as sns  # 添加 seaborn 導入，因為代碼中多處使用
import random
import plotly.graph_objects as go

# 導入 PyQt 模組
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTabWidget, QFileDialog, QGroupBox, QCheckBox,
    QRadioButton, QButtonGroup, QSplitter, QScrollArea,
    QSlider, QSpinBox, QListWidget, QListWidgetItem, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QTextCursor
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QSize, QUrl

# 嘗試導入 QWebEngineView，但不讓它阻止程式運行
WEB_ENGINE_AVAILABLE = False
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    WEB_ENGINE_AVAILABLE = True
except ImportError:
    print("PyQt5.QtWebEngineWidgets 或其依賴庫無法導入，互動式圖表功能將被禁用")

# 導入模組
from modules.visualizer import Visualizer
from modules.evaluator import Evaluator

# 導入工具類
from utils.logger import get_logger
from utils.file_manager import FileManager

# 獲取logger
logger = get_logger("visualization_tab")

# 修改 Plotly 相關函數，以支援無 WebEngine 環境
def write_figure_safely(fig, html_path=None, img_path=None, width=1200, height=800):
    """安全地將 Plotly 圖形儲存為檔案
    
    根據環境支持情況，儲存為 HTML 和/或 PNG 格式
    
    Args:
        fig: Plotly 圖形對象
        html_path: HTML 輸出路徑，如果為 None 則不儲存 HTML
        img_path: 圖片輸出路徑，如果為 None 則不儲存圖片
        width: 圖片寬度，預設 1200
        height: 圖片高度，預設 800
        
    Returns:
        str: 成功儲存的檔案路徑，優先返回 img_path
    """
    saved_path = None
    
    # 儲存 PNG 版本（總是嘗試）
    if img_path:
        try:
            fig.write_image(img_path, width=width, height=height)
            saved_path = img_path
        except Exception as e:
            logger.error(f"無法將圖表儲存為圖片: {str(e)}")
            
    # 如果環境支持，儲存 HTML 版本
    if html_path and WEB_ENGINE_AVAILABLE:
        try:
            fig.write_html(html_path)
            # 如果沒有成功儲存圖片，則返回 HTML 路徑
            if not saved_path:
                saved_path = html_path
        except Exception as e:
            logger.error(f"無法將圖表儲存為 HTML: {str(e)}")
    
    return saved_path

# 改為用一個函數處理通用的儲存功能
def save_plotly_figure(fig, directory, base_name, timestamp=None):
    """將 Plotly 圖形儲存到指定目錄
    
    Args:
        fig: Plotly 圖形對象
        directory: 儲存目錄
        base_name: 基本檔名
        timestamp: 時間戳，如果為 None 則自動產生
        
    Returns:
        tuple: (img_path, html_path)，如果儲存失敗則為 None
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    html_path = os.path.join(directory, f"{base_name}_{timestamp}.html")
    img_path = os.path.join(directory, f"{base_name}_{timestamp}.png")
    
    saved_path = write_figure_safely(fig, html_path, img_path)
    
    if saved_path:
        return img_path, (html_path if WEB_ENGINE_AVAILABLE else None)
    return None, None

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

        # 默認輸出目錄 - 更新為直接使用 1_output/visualizations 目錄，避免多層 Part04_ 路徑
        self.output_dir = os.path.join("1_output", "visualizations")
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
        
        # 初始化數據來源相關屬性
        self.data_source = None  # 數據來源
        self.data_sources = {}  # 數據來源字典
        self.attention_type = None  # 注意力類型
        self.vector_type = None  # 向量類型
        
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
        
        # 創建瀏覽按鈕
        browse_btn = QPushButton("瀏覽...")
        browse_btn.clicked.connect(self.browse_result_file)
        
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
        
        # 創建可視化區塊標籤頁
        self.viz_options_stack = QTabWidget()
        
        # 1. 面向向量質量評估指標（內部指標）
        self.vector_quality_tab = QWidget()
        vector_quality_layout = QVBoxLayout(self.vector_quality_tab)
        
        # 內聚度與分離度區域
        cohesion_separation_group = QGroupBox("內聚度與分離度")
        cohesion_layout = QVBoxLayout(cohesion_separation_group)
        
        # 內聚度與分離度選項
        cohesion_chart_layout = QHBoxLayout()
        cohesion_chart_layout.addWidget(QLabel("圖表類型:"))
        self.cohesion_chart_combo = QComboBox()
        self.cohesion_chart_combo.addItems(["條形圖", "散點圖", "樹狀圖"])
        cohesion_chart_layout.addWidget(self.cohesion_chart_combo)
        cohesion_layout.addLayout(cohesion_chart_layout)
        
        vector_quality_layout.addWidget(cohesion_separation_group)
        
        # 綜合得分區域
        combined_score_group = QGroupBox("綜合得分")
        combined_score_layout = QVBoxLayout(combined_score_group)
        
        # 綜合得分選項
        combined_chart_layout = QHBoxLayout()
        combined_chart_layout.addWidget(QLabel("圖表類型:"))
        self.combined_chart_combo = QComboBox()
        self.combined_chart_combo.addItems(["條形圖", "熱力圖"])
        combined_chart_layout.addWidget(self.combined_chart_combo)
        combined_score_layout.addLayout(combined_chart_layout)
        
        vector_quality_layout.addWidget(combined_score_group)
        
        # 輪廓係數區域
        silhouette_group = QGroupBox("輪廓係數")
        silhouette_layout = QVBoxLayout(silhouette_group)
        
        # 輪廓係數選項
        silhouette_chart_layout = QHBoxLayout()
        silhouette_chart_layout.addWidget(QLabel("圖表類型:"))
        self.silhouette_chart_combo = QComboBox()
        self.silhouette_chart_combo.addItems(["輪廓圖", "小提琴圖"])
        silhouette_chart_layout.addWidget(self.silhouette_chart_combo)
        silhouette_layout.addLayout(silhouette_chart_layout)
        
        vector_quality_layout.addWidget(silhouette_group)
        
        # 困惑度區域
        perplexity_group = QGroupBox("困惑度")
        perplexity_layout = QVBoxLayout(perplexity_group)
        
        # 困惑度選項
        topics_range_layout = QHBoxLayout()
        topics_range_layout.addWidget(QLabel("主題數範圍:"))
        self.min_topics_spin = QSpinBox()
        self.min_topics_spin.setRange(2, 50)
        self.min_topics_spin.setValue(2)
        topics_range_layout.addWidget(self.min_topics_spin)
        topics_range_layout.addWidget(QLabel("至"))
        self.max_topics_spin = QSpinBox()
        self.max_topics_spin.setRange(5, 100)
        self.max_topics_spin.setValue(20)
        topics_range_layout.addWidget(self.max_topics_spin)
        perplexity_layout.addLayout(topics_range_layout)
        
        vector_quality_layout.addWidget(perplexity_group)
        
        # 2. 情感分析性能指標（外部指標）
        self.sentiment_tab = QWidget()
        sentiment_layout = QVBoxLayout(self.sentiment_tab)
        
        # 準確率、精確率、召回率、F1分數區域
        metrics_group = QGroupBox("準確率、精確率、召回率、F1分數")
        metrics_layout = QVBoxLayout(metrics_group)
        
        # 指標圖表選項
        metrics_chart_layout = QHBoxLayout()
        metrics_chart_layout.addWidget(QLabel("圖表類型:"))
        self.metrics_chart_combo = QComboBox()
        self.metrics_chart_combo.addItems(["分組條形圖", "雷達圖", "面積圖"])
        metrics_chart_layout.addWidget(self.metrics_chart_combo)
        metrics_layout.addLayout(metrics_chart_layout)
        
        sentiment_layout.addWidget(metrics_group)
        
        # 宏平均F1和微平均F1區域
        f1_group = QGroupBox("宏平均F1和微平均F1")
        f1_layout = QVBoxLayout(f1_group)
        
        # F1指標圖表選項
        f1_chart_layout = QHBoxLayout()
        f1_chart_layout.addWidget(QLabel("圖表類型:"))
        self.f1_chart_combo = QComboBox()
        self.f1_chart_combo.addItems(["條形圖", "熱力圖"])
        f1_chart_layout.addWidget(self.f1_chart_combo)
        f1_layout.addLayout(f1_chart_layout)
        
        sentiment_layout.addWidget(f1_group)
        
        # 3. 注意力機制評估
        self.attention_tab = QWidget()
        attention_layout = QVBoxLayout(self.attention_tab)
        
        # 注意力分布區域
        attention_dist_group = QGroupBox("注意力分布")
        attention_dist_layout = QVBoxLayout(attention_dist_group)
        
        # 注意力分布圖表選項
        attention_chart_layout = QHBoxLayout()
        attention_chart_layout.addWidget(QLabel("圖表類型:"))
        self.attention_chart_combo = QComboBox()
        self.attention_chart_combo.addItems(["熱力圖", "文本注釋圖", "弦圖"])
        attention_chart_layout.addWidget(self.attention_chart_combo)
        attention_dist_layout.addLayout(attention_chart_layout)
        
        # 樣本選擇（用於文本注釋圖）
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("樣本ID:"))
        self.sample_id_spin = QSpinBox()
        self.sample_id_spin.setRange(0, 100)
        self.sample_id_spin.setValue(0)
        sample_layout.addWidget(self.sample_id_spin)
        attention_dist_layout.addLayout(sample_layout)
        
        attention_layout.addWidget(attention_dist_group)
        
        # 注意力權重比較區域
        weight_compare_group = QGroupBox("注意力權重比較")
        weight_compare_layout = QVBoxLayout(weight_compare_group)
        
        # 權重比較圖表選項
        weight_chart_layout = QHBoxLayout()
        weight_chart_layout.addWidget(QLabel("圖表類型:"))
        self.weight_chart_combo = QComboBox()
        self.weight_chart_combo.addItems(["平行坐標圖", "相關性熱力圖"])
        weight_chart_layout.addWidget(self.weight_chart_combo)
        weight_compare_layout.addLayout(weight_chart_layout)
        
        attention_layout.addWidget(weight_compare_group)
        
        # 4. 主題模型評估
        self.topic_tab = QWidget()
        topic_layout = QVBoxLayout(self.topic_tab)
        
        # 主題連貫性區域
        coherence_group = QGroupBox("主題連貫性")
        coherence_layout = QVBoxLayout(coherence_group)
        
        # 主題連貫性圖表選項
        coherence_chart_layout = QHBoxLayout()
        coherence_chart_layout.addWidget(QLabel("圖表類型:"))
        self.coherence_chart_combo = QComboBox()
        self.coherence_chart_combo.addItems(["條形圖", "詞雲"])
        coherence_chart_layout.addWidget(self.coherence_chart_combo)
        coherence_layout.addLayout(coherence_chart_layout)
        
        topic_layout.addWidget(coherence_group)
        
        # 主題分布區域
        topic_dist_group = QGroupBox("主題分布")
        topic_dist_layout = QVBoxLayout(topic_dist_group)
        
        # 主題分布圖表選項
        topic_dist_chart_layout = QHBoxLayout()
        topic_dist_chart_layout.addWidget(QLabel("圖表類型:"))
        self.topic_dist_chart_combo = QComboBox()
        self.topic_dist_chart_combo.addItems(["堆疊柱狀圖", "交互式氣泡圖"])
        topic_dist_chart_layout.addWidget(self.topic_dist_chart_combo)
        topic_dist_layout.addLayout(topic_dist_chart_layout)
        
        topic_layout.addWidget(topic_dist_group)
        
        # 5. 綜合比較視覺化
        self.comprehensive_tab = QWidget()
        comprehensive_layout = QVBoxLayout(self.comprehensive_tab)
        
        # 降維可視化區域
        dim_reduction_group = QGroupBox("降維可視化")
        dim_reduction_layout = QVBoxLayout(dim_reduction_group)
        
        # 降維方法選項
        dim_method_layout = QHBoxLayout()
        dim_method_layout.addWidget(QLabel("降維方法:"))
        self.dim_method_combo = QComboBox()
        self.dim_method_combo.addItems(["t-SNE", "UMAP", "PCA", "3D散點圖"])
        dim_method_layout.addWidget(self.dim_method_combo)
        dim_reduction_layout.addLayout(dim_method_layout)
        
        # 顏色標記選項
        color_by_layout = QHBoxLayout()
        color_by_layout.addWidget(QLabel("著色依據:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["主題", "注意力機制", "聚類結果"])
        color_by_layout.addWidget(self.color_by_combo)
        dim_reduction_layout.addLayout(color_by_layout)
        
        comprehensive_layout.addWidget(dim_reduction_group)
        
        # 多指標綜合視圖區域
        multi_metrics_group = QGroupBox("多指標綜合視圖")
        multi_metrics_layout = QVBoxLayout(multi_metrics_group)
        
        # 多指標圖表選項
        multi_chart_layout = QHBoxLayout()
        multi_chart_layout.addWidget(QLabel("圖表類型:"))
        self.multi_chart_combo = QComboBox()
        self.multi_chart_combo.addItems(["雷達圖組", "交互式儀表板"])
        multi_chart_layout.addWidget(self.multi_chart_combo)
        multi_metrics_layout.addLayout(multi_chart_layout)
        
        # 選擇要包括的指標
        self.cb_include_cohesion = QCheckBox("包含內聚度")
        self.cb_include_cohesion.setChecked(True)
        multi_metrics_layout.addWidget(self.cb_include_cohesion)
        
        self.cb_include_separation = QCheckBox("包含分離度")
        self.cb_include_separation.setChecked(True)
        multi_metrics_layout.addWidget(self.cb_include_separation)
        
        self.cb_include_f1 = QCheckBox("包含F1分數")
        self.cb_include_f1.setChecked(True)
        multi_metrics_layout.addWidget(self.cb_include_f1)
        
        comprehensive_layout.addWidget(multi_metrics_group)
        
        # 添加所有標籤頁
        self.viz_options_stack.addTab(self.vector_quality_tab, "面向向量質量")
        self.viz_options_stack.addTab(self.sentiment_tab, "情感分析性能")
        self.viz_options_stack.addTab(self.attention_tab, "注意力機制評估")
        self.viz_options_stack.addTab(self.topic_tab, "主題模型評估")
        self.viz_options_stack.addTab(self.comprehensive_tab, "綜合比較")
        
        options_layout.addWidget(self.viz_options_stack)
        
        # 生成可視化按鈕
        self.generate_btn = QPushButton("生成並保存圖片")
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
        
        # 創建結果區域顯示
        result_info_layout = QHBoxLayout()
        self.result_status_label = QLabel("結果狀態:")
        result_info_layout.addWidget(self.result_status_label)
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v%")
        self.progress_bar.setFixedWidth(200)
        result_info_layout.addWidget(self.progress_bar)
        
        bottom_layout.addLayout(result_info_layout)
        
        # 右側空間
        bottom_layout.addStretch(1)
        
        return bottom_layout

    def browse_result_file(self):
        """瀏覽選擇結果文件或資料夾"""
        # 優先使用file_manager的路徑
        if self.file_manager is not None and hasattr(self.file_manager, "output_dir"):
            output_dir = self.file_manager.output_dir
        else:
            # 退回到安全獲取輸出目錄的方式
            output_dir = "./1_output"
            
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
                    self.logger.warning(f"獲取輸出目錄時出錯: {str(e)}，使用默認值 {output_dir}")
        
        # 確保路徑存在
        if not os.path.exists(output_dir):
            output_dir = "."  # 如果目錄不存在，切換到當前目錄
            
        # 顯示選擇對話框，讓用戶可以選擇文件或文件夾
        select_type_dialog = QMessageBox()
        select_type_dialog.setWindowTitle("選擇載入方式")
        select_type_dialog.setText("請選擇要載入的資料類型：")
        select_type_dialog.addButton("單一結果檔案", QMessageBox.AcceptRole)
        select_type_dialog.addButton("整個run資料夾", QMessageBox.AcceptRole)
        select_type_dialog.addButton("取消", QMessageBox.RejectRole)
        
        # 執行對話框並獲取用戶選擇
        ret = select_type_dialog.exec_()
        selected_button = select_type_dialog.clickedButton().text()
        
        if selected_button == "取消":
            return
        
        if selected_button == "單一結果檔案":
            # 選擇單一JSON文件
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "選擇結果文件",
                output_dir,
                "JSON 文件 (*.json)"
            )
            
            if file_path:
                self.load_results(file_path)
        else:
            # 選擇整個run資料夾
            folder_path = QFileDialog.getExistingDirectory(
                self,
                "選擇run資料夾",
                output_dir
            )
            
            if folder_path:
                self.load_results_from_folder(folder_path)

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
                self.logger.info(f"成功載入面向向量，共 {len(self.aspect_vectors)} 個面向")
            
            # 從文件中獲取評估結果
            if 'evaluation_metrics' in results:
                self.evaluation_results = results['evaluation_metrics']
                self.logger.info(f"成功載入評估結果，包含: {list(self.evaluation_results.keys())}")
            elif 'metrics' in results:
                self.evaluation_results = results['metrics']
                self.logger.info(f"成功載入評估結果，包含: {list(self.evaluation_results.keys())}")
            
            # 從文件中獲取主題信息
            if 'topics' in results:
                self.topics = results['topics']
                self.logger.info(f"成功載入 {len(self.topics)} 個主題")
            
            # 從文件中獲取注意力權重
            if 'attention_weights' in results:
                self.attention_weights = results['attention_weights']
                self.logger.info(f"成功載入注意力權重，形狀: {self.attention_weights.shape if hasattr(self.attention_weights, 'shape') else '未知'}")
            
            # 從文件中獲取其他重要信息
            self.data_source = results.get('data_source', '未知')
            self.attention_type = results.get('attention_type', '未知')
            self.vector_type = results.get('vector_type', '未知')
            
            # 設置當前數據集名稱
            self.current_dataset = os.path.basename(file_path).replace('.json', '')
            
            # 更新結果信息
            self._update_result_info()
            
            # 成功載入提示
            self.status_message.emit(f"已成功載入結果文件: {os.path.basename(file_path)}", 3000)
            
            # 更新結果文件路徑
            self.result_file_path = file_path
            
            # 記錄詳細信息
            self.logger.info(f"數據來源: {self.data_source}")
            self.logger.info(f"注意力類型: {self.attention_type}")
            self.logger.info(f"向量類型: {self.vector_type}")
            
            return True
                
        except Exception as e:
            self.logger.error(f"載入結果文件出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "載入出錯", f"載入結果文件時出錯:\n{str(e)}")
            return False

    def load_results_from_folder(self, folder_path):
        """從run資料夾載入結果
        
        Args:
            folder_path: run資料夾路徑
        
        Returns:
            bool: 是否成功載入
        """
        try:
            # 檢查這是否是一個run資料夾
            if not os.path.isdir(folder_path):
                QMessageBox.critical(self, "載入出錯", "選擇的路徑不是一個有效的目錄")
                return False
                
            # 清空當前結果
            self.aspect_vectors = None
            self.topics = None
            self.evaluation_results = None
            self.data_sources = {}  # 儲存每個數據集的來源位置
            
            self.logger.info(f"開始從資料夾載入結果: {folder_path}")
            self.status_message.emit(f"正在載入run資料夾: {os.path.basename(folder_path)}...", 3000)
            
            # 檢查是否有aspect_vectors_result.json文件
            main_result_path = os.path.join(folder_path, "aspect_vectors_result.json")
            if os.path.exists(main_result_path):
                self.logger.info(f"在根目錄找到主要結果文件: {main_result_path}")
                self.load_results(main_result_path)
                self.data_sources["主要結果"] = os.path.basename(main_result_path)
            
            # 檢查並載入各個子目錄的專用結果
            
            # 1. 檢查評估結果 (05_evaluation)
            evaluation_dir = os.path.join(folder_path, "05_evaluation")
            if os.path.exists(evaluation_dir) and os.path.isdir(evaluation_dir):
                self.logger.info(f"找到評估結果目錄: {evaluation_dir}")
                # 尋找評估結果文件
                evaluation_files = [f for f in os.listdir(evaluation_dir) if f.endswith('.json')]
                for eval_file in evaluation_files:
                    eval_path = os.path.join(evaluation_dir, eval_file)
                    try:
                        with open(eval_path, 'r', encoding='utf-8') as f:
                            eval_results = json.load(f)
                            
                        # 如果還沒有評估結果或當前評估結果比較新，則更新
                        if self.evaluation_results is None or len(eval_results) > len(self.evaluation_results):
                            self.evaluation_results = eval_results
                            self.data_sources["評估結果"] = os.path.join("05_evaluation", eval_file)
                            self.logger.info(f"載入評估結果: {eval_path}")
                    except Exception as e:
                        self.logger.warning(f"載入評估文件 {eval_path} 時出錯: {str(e)}")
            
            # 2. 檢查LDA主題 (03_lda_topics)
            topics_dir = os.path.join(folder_path, "03_lda_topics")
            if os.path.exists(topics_dir) and os.path.isdir(topics_dir):
                self.logger.info(f"找到LDA主題目錄: {topics_dir}")
                # 尋找主題文件
                topic_files = [f for f in os.listdir(topics_dir) if f.endswith('.json')]
                for topic_file in topic_files:
                    topic_path = os.path.join(topics_dir, topic_file)
                    try:
                        with open(topic_path, 'r', encoding='utf-8') as f:
                            topic_results = json.load(f)
                            
                        if 'topics' in topic_results:
                            self.topics = topic_results['topics']
                            self.data_sources["主題模型"] = os.path.join("03_lda_topics", topic_file)
                            self.logger.info(f"載入主題模型: {topic_path}")
                            break  # 找到主題就跳出循環
                    except Exception as e:
                        self.logger.warning(f"載入主題文件 {topic_path} 時出錯: {str(e)}")
            
            # 3. 檢查面向向量 (04_aspect_vectors)
            vectors_dir = os.path.join(folder_path, "04_aspect_vectors")
            if os.path.exists(vectors_dir) and os.path.isdir(vectors_dir):
                self.logger.info(f"找到面向向量目錄: {vectors_dir}")
                # 尋找向量文件
                vector_files = [f for f in os.listdir(vectors_dir) if f.endswith('.json')]
                for vector_file in vector_files:
                    vector_path = os.path.join(vectors_dir, vector_file)
                    try:
                        with open(vector_path, 'r', encoding='utf-8') as f:
                            vector_results = json.load(f)
                            
                        if 'aspect_vectors' in vector_results:
                            self.aspect_vectors = vector_results['aspect_vectors']
                            self.data_sources["面向向量"] = os.path.join("04_aspect_vectors", vector_file)
                            self.logger.info(f"載入面向向量: {vector_path}")
                            break  # 找到向量就跳出循環
                    except Exception as e:
                        self.logger.warning(f"載入向量文件 {vector_path} 時出錯: {str(e)}")
            
            # 4. 檢查注意力權重
            # 注意力權重可能在主要結果文件或專門的文件中
            if not hasattr(self, 'attention_weights') or self.attention_weights is None:
                attention_files = []
                # 檢查評估目錄中的文件
                if os.path.exists(evaluation_dir) and os.path.isdir(evaluation_dir):
                    attention_files += [os.path.join(evaluation_dir, f) for f in os.listdir(evaluation_dir) if 'attention' in f.lower() and f.endswith('.json')]
                
                # 檢查根目錄中的文件
                attention_files += [os.path.join(folder_path, f) for f in os.listdir(folder_path) if 'attention' in f.lower() and f.endswith('.json')]
                
                for att_path in attention_files:
                    try:
                        with open(att_path, 'r', encoding='utf-8') as f:
                            att_results = json.load(f)
                            
                        if 'attention_weights' in att_results:
                            self.attention_weights = att_results['attention_weights']
                            rel_path = os.path.relpath(att_path, folder_path)
                            self.data_sources["注意力權重"] = rel_path
                            self.logger.info(f"載入注意力權重: {att_path}")
                            break  # 找到權重就跳出循環
                    except Exception as e:
                        self.logger.warning(f"載入注意力文件 {att_path} 時出錯: {str(e)}")
            
            # 從文件中獲取其他重要信息
            # 這些信息應該在主要結果文件或評估文件中
            if hasattr(self, 'data_source') and not self.data_source:
                # 嘗試從資料夾名字推斷數據來源
                folder_name = os.path.basename(folder_path)
                if 'run_' in folder_name:
                    run_date = folder_name.replace('run_', '')
                    self.data_source = f"Run {run_date}"
                else:
                    self.data_source = folder_name
            
            # 設置當前數據集名稱
            self.current_dataset = os.path.basename(folder_path)
            
            # 更新結果信息
            self._update_result_info()
            
            # 統計載入的數據
            loaded_data = []
            if self.aspect_vectors:
                loaded_data.append("面向向量")
            if self.topics:
                loaded_data.append("主題模型")
            if self.evaluation_results:
                loaded_data.append("評估結果")
            if hasattr(self, 'attention_weights') and self.attention_weights:
                loaded_data.append("注意力權重")
            
            # 顯示載入的數據源提示
            data_source_text = ""
            for key, path in self.data_sources.items():
                data_source_text += f"• {key}: {path}\n"
            
            # 成功載入提示
            if loaded_data:
                self.status_message.emit(f"成功載入資料夾: {os.path.basename(folder_path)}", 5000)
                QMessageBox.information(
                    self, 
                    "載入成功", 
                    f"已成功從run資料夾載入以下數據：\n\n{', '.join(loaded_data)}\n\n數據來源：\n{data_source_text}"
                )
            else:
                QMessageBox.warning(
                    self, 
                    "載入警告", 
                    f"從run資料夾載入數據時未找到任何可用的評估結果、面向向量或主題數據。\n請檢查資料夾內容是否完整。"
                )
                return False
            
            # 更新結果文件路徑
            self.result_file_path = folder_path
            
            return True
                
        except Exception as e:
            self.logger.error(f"載入run資料夾出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "載入出錯", f"載入run資料夾時出錯:\n{str(e)}")
            return False

    def _update_result_info(self):
        """更新結果信息標籤"""
        if self.current_dataset:
            info_text = f"當前結果: {self.current_dataset}"
            
            if self.data_source:
                info_text += f" | 數據來源: {self.data_source}"
            
            if self.attention_type:
                info_text += f" | 注意力類型: {self.attention_type}"
            
            if self.vector_type:
                info_text += f" | 向量類型: {self.vector_type}"
            
            if self.aspect_vectors is not None:
                info_text += f" | 面向數: {len(self.aspect_vectors)}"
            
            if self.topics is not None:
                info_text += f" | 主題數: {len(self.topics)}"
            
            if self.evaluation_results is not None:
                info_text += f" | 評估指標: {len(self.evaluation_results)}"
            
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
    
    def generate_visualization(self):
        """生成可視化並自動保存"""
        try:
            # 檢查數據是否已載入
            if not self.current_dataset or self.aspect_vectors is None or self.evaluation_results is None:
                QMessageBox.warning(self, "數據未載入", "請先載入結果文件後再生成可視化")
                return
            
            # 獲取當前選中的標籤頁索引
            current_tab_index = self.viz_options_stack.currentIndex()
            
            # 根據當前選中的標籤頁生成相應的可視化
            try:
                import os
                import re
                
                self.status_message.emit("正在生成可視化...", 0)
                
                # 決定要生成的可視化類型和保存文件名
                viz_type = ""
                default_name = ""
                all_img_paths = []  # 存儲所有生成的圖片路徑
                
                if current_tab_index == 0:  # 面向向量質量
                    viz_type = "vector_quality"
                    default_name = "vector_quality"
                    img_path = self._generate_vector_quality_viz()
                    
                    # 尋找同一時間生成的所有圖表
                    if img_path:
                        # 從路徑中提取時間戳部分
                        timestamp_match = re.search(r'_(\d{8}_\d{6})\.png$', img_path)
                        if timestamp_match:
                            timestamp = timestamp_match.group(1)
                            directory = os.path.dirname(img_path)
                            # 找到相同時間戳的所有文件
                            for file in os.listdir(directory):
                                if timestamp in file and file.startswith("vector_quality_"):
                                    all_img_paths.append(os.path.join(directory, file))
                            
                            # 排序圖片路徑
                            all_img_paths.sort()
                elif current_tab_index == 1:  # 情感分析性能
                    viz_type = "sentiment_analysis"
                    default_name = "sentiment_analysis" 
                    img_path = self._generate_sentiment_viz()
                    if img_path:
                        all_img_paths.append(img_path)
                elif current_tab_index == 2:  # 注意力機制評估
                    viz_type = "attention_evaluation"
                    default_name = "attention_evaluation"
                    img_path = self._generate_attention_viz()
                    if img_path:
                        all_img_paths.append(img_path)
                elif current_tab_index == 3:  # 主題模型評估
                    viz_type = "topic_evaluation"
                    default_name = "topic_evaluation"
                    img_path = self._generate_topic_viz()
                    if img_path:
                        all_img_paths.append(img_path)
                elif current_tab_index == 4:  # 綜合比較視覺化
                    viz_type = "comprehensive"
                    default_name = "comprehensive_viz"
                    img_path = self._generate_comprehensive_viz()
                    if img_path:
                        all_img_paths.append(img_path)
                else:
                    QMessageBox.warning(self, "無效選擇", "請選擇一種可視化類型")
                    return
                
                # 如果成功生成圖片，顯示成功訊息
                if img_path and os.path.exists(img_path):
                    # 提示用戶所有生成的圖表
                    if len(all_img_paths) > 1 and viz_type == "vector_quality":
                        img_types = []
                        for path in all_img_paths:
                            base_name = os.path.basename(path)
                            if "cohesion" in base_name:
                                img_types.append("內聚度與分離度")
                            elif "combined" in base_name:
                                img_types.append("綜合得分")
                            elif "silhouette" in base_name:
                                img_types.append("輪廓係數")
                            elif "perplexity" in base_name:
                                img_types.append("困惑度")
                                
                        img_folder = os.path.dirname(img_path)
                        multi_img_message = f"已成功生成 {len(all_img_paths)} 種圖表：\n\n"
                        for i, img_type in enumerate(img_types):
                            multi_img_message += f"{i+1}. {img_type}\n"
                        multi_img_message += f"\n所有圖表已保存在目錄：\n{img_folder}"
                        
                        self.status_message.emit(f"已生成 {len(all_img_paths)} 種面向向量質量圖表", 5000)
                        QMessageBox.information(self, "生成成功", multi_img_message)
                    else:
                        # 提示用戶
                        img_folder = os.path.dirname(img_path)
                        self.status_message.emit(f"可視化已生成並保存至: {img_folder}", 5000)
                        QMessageBox.information(
                            self, 
                            "生成成功", 
                            f"可視化已成功生成！\n\n保存路徑：\n{img_folder}"
                        )
                else:
                    self.status_message.emit("可視化生成失敗或未生成圖片文件", 3000)
                    
            except Exception as e:
                import traceback
                self.logger.error(f"生成可視化出錯: {str(e)}")
                self.logger.error(traceback.format_exc())
                QMessageBox.critical(self, "生成出錯", f"生成可視化時出錯:\n{str(e)}")
                
        except Exception as e:
            self.status_message.emit(f"生成可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "生成出錯", f"生成可視化時出錯:\n{str(e)}")

    def _generate_vector_quality_viz(self):
        """生成面向向量質量可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 檢查是否有必要的數據
            if self.aspect_vectors is None or self.evaluation_results is None:
                QMessageBox.warning(self, "數據缺失", "缺少必要的面向向量或評估結果數據")
                return None
            
            # 導入必要的模組
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            import os
            import re
            import traceback
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "面向向量質量")
            os.makedirs(output_dir, exist_ok=True)
            
            self.status_message.emit("開始生成面向向量質量可視化...", 3000)
            self.progress_bar.setValue(5)
            
            # 獲取當前選項
            cohesion_chart_type = self.cohesion_chart_combo.currentText()
            combined_chart_type = self.combined_chart_combo.currentText()
            silhouette_chart_type = self.silhouette_chart_combo.currentText()
            min_topics = self.min_topics_spin.value()
            max_topics = self.max_topics_spin.value()
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 從評估結果中獲取指標
            metrics = self.evaluation_results
            aspect_names = list(self.aspect_vectors.keys())

            # 記錄評估結果中的所有可用欄位，以幫助診斷問題
            self.logger.debug(f"評估結果包含以下欄位: {list(metrics.keys())}")

            # 1. 內聚度與分離度圖表
            self.status_message.emit(f"1. 開始處理「內聚度與分離度」區域（{cohesion_chart_type}）...", 3000)
            self.progress_bar.setValue(15)

            # 從評估結果中獲取內聚度和分離度
            cohesion_values = []
            separation_values = []

            # 優先檢查是否存在面向特定的內聚度與分離度數據
            if 'topic_coherence' in metrics and isinstance(metrics['topic_coherence'], dict):
                self.logger.info("找到面向特定的內聚度數據")
                # 從字典中獲取每個面向的內聚度
                for aspect in aspect_names:
                    if aspect in metrics['topic_coherence']:
                        cohesion_values.append(metrics['topic_coherence'][aspect])
                    else:
                        self.logger.warning(f"面向 {aspect} 的內聚度數據未找到，使用默認值 0.5")
                        cohesion_values.append(0.5)
            # 次優先檢查是否存在 cohesion 字典
            elif 'cohesion' in metrics and isinstance(metrics['cohesion'], dict):
                self.logger.info("找到 cohesion 字典數據")
                for aspect in aspect_names:
                    if aspect in metrics['cohesion']:
                        cohesion_values.append(metrics['cohesion'][aspect])
                    else:
                        self.logger.warning(f"面向 {aspect} 的內聚度數據未找到，使用默認值 0.5")
                        cohesion_values.append(0.5)
            # 再次檢查是否有單一值的 coherence 或 cohesion
            elif 'coherence' in metrics and isinstance(metrics['coherence'], (int, float)):
                self.logger.info(f"使用全局內聚度值: {metrics['coherence']}")
                cohesion_values = [metrics['coherence']] * len(aspect_names)
            elif 'cohesion' in metrics and isinstance(metrics['cohesion'], (int, float)):
                self.logger.info(f"使用全局內聚度值: {metrics['cohesion']}")
                cohesion_values = [metrics['cohesion']] * len(aspect_names)
            else:
                self.logger.warning("未找到任何內聚度數據，所有面向將使用默認值 0.5")
                cohesion_values = [0.5] * len(aspect_names)
            
            # 優先檢查 topic_separation 字典
            # 注意：topic_separation 可能存儲的是面向對之間的分離度
            # 我們需要計算每個面向與其他所有面向的平均分離度
            if 'topic_separation' in metrics and isinstance(metrics['topic_separation'], dict):
                self.logger.info("找到面向對的分離度數據")
                aspect_separation_values = {aspect: [] for aspect in aspect_names}
                
                # 收集每個面向的所有分離度值
                for pair_key, separation_value in metrics['topic_separation'].items():
                    parts = pair_key.split('_')
                    if len(parts) >= 2:  # 確保格式為 "aspect1_aspect2"
                        aspect1 = parts[0]
                        aspect2 = '_'.join(parts[1:])  # 處理面向名稱中可能包含下劃線的情況
                        
                        if aspect1 in aspect_names and aspect2 in aspect_names:
                            aspect_separation_values[aspect1].append(separation_value)
                            aspect_separation_values[aspect2].append(separation_value)
                
                # 計算每個面向的平均分離度
                for aspect in aspect_names:
                    if aspect_separation_values[aspect]:
                        separation_values.append(np.mean(aspect_separation_values[aspect]))
                    else:
                        self.logger.warning(f"面向 {aspect} 的分離度數據未找到，使用默認值 0.3")
                        separation_values.append(0.3)
            # 檢查 separation 字典
            elif 'separation' in metrics and isinstance(metrics['separation'], dict):
                self.logger.info("找到 separation 字典數據")
                for aspect in aspect_names:
                    if aspect in metrics['separation']:
                        separation_values.append(metrics['separation'][aspect])
                    else:
                        self.logger.warning(f"面向 {aspect} 的分離度數據未找到，使用默認值 0.3")
                        separation_values.append(0.3)
            # 檢查單一值的 separation
            elif 'separation' in metrics and isinstance(metrics['separation'], (int, float)):
                self.logger.info(f"使用全局分離度值: {metrics['separation']}")
                separation_values = [metrics['separation']] * len(aspect_names)
            else:
                self.logger.warning("未找到任何分離度數據，所有面向將使用默認值 0.3")
                separation_values = [0.3] * len(aspect_names)

            # 記錄提取的數據，以幫助診斷問題
            self.logger.info(f"共提取了 {len(cohesion_values)} 個內聚度值和 {len(separation_values)} 個分離度值")
            self.logger.debug(f"內聚度值: {cohesion_values}")
            self.logger.debug(f"分離度值: {separation_values}")

            # 確保兩個列表有值，如果仍然沒有值則報錯
            if not cohesion_values or not separation_values:
                QMessageBox.warning(self, "數據缺失", "缺少內聚度或分離度數據")
                self.logger.error(f"未找到內聚度或分離度數據，可用指標: {list(metrics.keys())}")
                return None

            # 創建「內聚度與分離度」子目錄
            cohesion_dir = os.path.join(output_dir, "內聚度與分離度")
            os.makedirs(cohesion_dir, exist_ok=True)
            
            # 生成內聚度與分離度圖表
            if cohesion_chart_type == "條形圖":
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(aspect_names))
                width = 0.35
                
                ax.bar(x - width/2, cohesion_values, width, label='內聚度')
                ax.bar(x + width/2, separation_values, width, label='分離度')
                
                ax.set_ylabel('分數')
                ax.set_title('不同面向的內聚度與分離度')
                ax.set_xticks(x)
                ax.set_xticklabels(aspect_names, rotation=45, ha='right')
                ax.legend()
                
            elif cohesion_chart_type == "散點圖":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(cohesion_values, separation_values, s=100)
                
                # 添加標籤
                for i, aspect in enumerate(aspect_names):
                    ax.annotate(aspect, (cohesion_values[i], separation_values[i]),
                              textcoords="offset points", xytext=(0,10), ha='center')
                
                ax.set_xlabel('內聚度')
                ax.set_ylabel('分離度')
                ax.set_title('內聚度與分離度的關係散點圖')
                ax.grid(True)
                
            elif cohesion_chart_type == "樹狀圖":
                # 使用條形圖模擬樹狀圖
                combined_scores = [0.5 * c + 0.5 * s for c, s in zip(cohesion_values, separation_values)]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(aspect_names))
                width = 0.7
                
                ax.bar(x, combined_scores, width, label='綜合得分')
                ax.set_ylabel('綜合得分')
                ax.set_title('內聚度與分離度的綜合得分')
                ax.set_xticks(x)
                ax.set_xticklabels(aspect_names, rotation=45, ha='right')
                ax.legend()
            
            # 添加數據來源信息
            self._add_data_source_caption(fig, plt, title="內聚度與分離度")
            
            # 保存內聚度分離度圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cs_img_path = os.path.join(cohesion_dir, f"vector_quality_cohesion_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(cs_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「內聚度與分離度」區域處理完成", 3000)
            self.progress_bar.setValue(30)
            
            # 2. 綜合得分圖表
            self.status_message.emit(f"2. 開始處理「綜合得分」區域（{combined_chart_type}）...", 3000)
            
            # 創建「綜合得分」子目錄
            combined_dir = os.path.join(output_dir, "綜合得分")
            os.makedirs(combined_dir, exist_ok=True)
            
            # 從評估結果中獲取所有相關指標
            all_metrics = {}
            
            # 處理內聚度指標
            if 'topic_coherence' in metrics and isinstance(metrics['topic_coherence'], dict):
                self.logger.info("使用面向特定的內聚度數據")
                all_metrics['coherence'] = [metrics['topic_coherence'].get(aspect, 0) for aspect in aspect_names]
            elif 'cohesion' in metrics and isinstance(metrics['cohesion'], dict):
                all_metrics['coherence'] = [metrics['cohesion'].get(aspect, 0) for aspect in aspect_names]
            elif 'coherence' in metrics and isinstance(metrics['coherence'], (int, float)):
                all_metrics['coherence'] = [metrics['coherence']] * len(aspect_names)
            
            # 處理分離度指標
            if 'topic_separation' in metrics and isinstance(metrics['topic_separation'], dict):
                self.logger.info("使用面向特定的分離度數據")
                # 需要計算每個面向的平均分離度
                aspect_separation_values = {aspect: [] for aspect in aspect_names}
                
                # 收集每個面向的所有分離度值
                for pair_key, separation_value in metrics['topic_separation'].items():
                    parts = pair_key.split('_')
                    if len(parts) >= 2:
                        aspect1 = parts[0]
                        aspect2 = '_'.join(parts[1:])
                        
                        if aspect1 in aspect_names and aspect2 in aspect_names:
                            aspect_separation_values[aspect1].append(separation_value)
                            aspect_separation_values[aspect2].append(separation_value)
                
                # 計算每個面向的平均分離度
                all_metrics['separation'] = []
                for aspect in aspect_names:
                    if aspect_separation_values[aspect]:
                        all_metrics['separation'].append(np.mean(aspect_separation_values[aspect]))
                    else:
                        all_metrics['separation'].append(0.3)
            elif 'separation' in metrics and isinstance(metrics['separation'], dict):
                all_metrics['separation'] = [metrics['separation'].get(aspect, 0) for aspect in aspect_names]
            elif 'separation' in metrics and isinstance(metrics['separation'], (int, float)):
                all_metrics['separation'] = [metrics['separation']] * len(aspect_names)
            
            # 處理其他指標
            for metric_name in ['silhouette', 'f1']:
                if metric_name in metrics:
                    if isinstance(metrics[metric_name], dict):
                        all_metrics[metric_name] = [metrics[metric_name].get(aspect, 0) for aspect in aspect_names]
                    else:
                        all_metrics[metric_name] = [metrics[metric_name]] * len(aspect_names)
            
            if not all_metrics:
                QMessageBox.warning(self, "數據缺失", "缺少必要的評估指標數據")
                return None
            
            if combined_chart_type == "條形圖":
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 計算綜合得分
                combined_scores = []
                for i in range(len(aspect_names)):
                    scores = [values[i] for values in all_metrics.values()]
                    combined_scores.append(np.mean(scores))
                
                bars = ax.bar(aspect_names, combined_scores, color='skyblue')
                
                # 添加數值標籤
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom')
                
                ax.set_ylim(0, 1.0)
                ax.set_ylabel('綜合得分')
                ax.set_title('不同面向的綜合性能得分')
                ax.set_xticklabels(aspect_names, rotation=45, ha='right')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            elif combined_chart_type == "熱力圖":
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 創建指標矩陣
                metrics_matrix = np.array([values for values in all_metrics.values()])
                
                # 創建熱力圖
                sns.heatmap(metrics_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                          xticklabels=aspect_names, yticklabels=list(all_metrics.keys()), ax=ax)
                
                ax.set_title('不同面向在各指標上的得分熱力圖')
            
            # 添加數據來源信息
            self._add_data_source_caption(fig, plt, title="綜合得分")
            
            # 保存綜合得分圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_img_path = os.path.join(combined_dir, f"vector_quality_combined_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(combined_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「綜合得分」區域處理完成", 3000)
            self.progress_bar.setValue(50)
            
            # 3. 輪廓係數圖表
            self.status_message.emit(f"3. 開始處理「輪廓係數」區域（{silhouette_chart_type}）...", 3000)
            
            # 創建「輪廓係數」子目錄
            silhouette_dir = os.path.join(output_dir, "輪廓係數")
            os.makedirs(silhouette_dir, exist_ok=True)
            
            # 從評估結果中獲取輪廓係數
            silhouette_values = []
            
            # 檢查是否有輪廓係數數據
            if 'silhouette' in metrics:
                if isinstance(metrics['silhouette'], dict):
                    silhouette_values = [metrics['silhouette'].get(aspect, 0) for aspect in aspect_names]
                else:
                    # 如果是單一值，則所有面向共用相同的值
                    self.logger.info(f"檢測到單一值的輪廓係數: {metrics['silhouette']}")
                    silhouette_values = [metrics['silhouette']] * len(aspect_names)
            else:
                # 如果找不到輪廓係數數據，使用默認值
                self.logger.warning("未找到輪廓係數數據，所有面向將使用默認值 0.4")
                silhouette_values = [0.4] * len(aspect_names)
            
            # 記錄提取的數據
            self.logger.info(f"共提取了 {len(silhouette_values)} 個輪廓係數值")
            self.logger.debug(f"輪廓係數值: {silhouette_values}")
            
            if silhouette_chart_type == "輪廓圖":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 繪製輪廓係數分布
                sns.kdeplot(silhouette_values, ax=ax)
                
                # 添加平均值線
                mean_value = np.mean(silhouette_values)
                ax.axvline(mean_value, color='r', linestyle='--', label=f'平均值: {mean_value:.2f}')
                
                ax.set_xlabel('輪廓係數')
                ax.set_ylabel('密度')
                ax.set_title('輪廓係數分布')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
            elif silhouette_chart_type == "小提琴圖":
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 創建小提琴圖
                sns.violinplot(y=silhouette_values, ax=ax)
                
                ax.set_title('輪廓係數分布')
                ax.set_ylabel('輪廓係數')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 添加數據來源信息
            self._add_data_source_caption(fig, plt, title="輪廓係數")
            
            # 保存輪廓係數圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            silhouette_img_path = os.path.join(silhouette_dir, f"vector_quality_silhouette_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(silhouette_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「輪廓係數」區域處理完成", 3000)
            self.progress_bar.setValue(75)
            
            # 4. 困惑度圖表
            self.status_message.emit(f"4. 開始處理「困惑度」區域 (範圍: {min_topics}-{max_topics})...", 3000)
            
            # 創建「困惑度」子目錄
            perplexity_dir = os.path.join(output_dir, "困惑度")
            os.makedirs(perplexity_dir, exist_ok=True)
            
            # 從評估結果中獲取困惑度
            if 'perplexity' not in metrics:
                QMessageBox.warning(self, "數據缺失", "缺少困惑度數據")
                return None
            
            perplexity_data = metrics['perplexity']
            topic_nums = list(range(min_topics, max_topics + 1))
            
            # 繪製困惑度曲線
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for aspect in aspect_names:
                if aspect in perplexity_data:
                    perplexities = [perplexity_data[aspect].get(str(t), 0) for t in topic_nums]
                    ax.plot(topic_nums, perplexities, marker='o', label=aspect)
            
            ax.set_xlabel('主題數量')
            ax.set_ylabel('困惑度')
            ax.set_title(f'主題數{min_topics}至{max_topics}的困惑度變化')
            ax.legend()
            ax.grid(True)
            
            # 添加數據來源信息
            self._add_data_source_caption(fig, plt, title="困惑度")
            
            # 保存困惑度圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            perplexity_img_path = os.path.join(perplexity_dir, f"vector_quality_perplexity_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(perplexity_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「困惑度」區域處理完成", 3000)
            self.progress_bar.setValue(100)
            
            # 返回第一張圖片的路徑
            self.status_message.emit(f"✓ 所有面向向量質量可視化已完成", 5000)
            return cs_img_path
            
        except Exception as e:
            self.status_message.emit(f"生成面向向量質量可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成面向向量質量可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _generate_sentiment_viz(self):
        """生成情感分析可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 檢查是否有必要的數據
            if self.aspect_vectors is None or self.evaluation_results is None:
                QMessageBox.warning(self, "數據缺失", "缺少必要的面向向量或評估結果數據")
                return None
                
            # 導入必要的模組
            import os
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            import traceback
            import plotly.graph_objects as go
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "情感分析性能")
            os.makedirs(output_dir, exist_ok=True)
            
            self.status_message.emit("開始生成情感分析性能可視化...", 3000)
            self.progress_bar.setValue(5)
            
            # 獲取當前選項
            metrics_chart_type = self.metrics_chart_combo.currentText()
            f1_chart_type = self.f1_chart_combo.currentText()
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 提取面向向量數據
            aspects = list(self.aspect_vectors.keys())
            
            # 創建情感分析圖表
            # 如果評估結果中有情感分析相關指標
            sentiment_metrics = {}
            if 'sentiment_metrics' in self.evaluation_results:
                sentiment_metrics = self.evaluation_results['sentiment_metrics']
            elif 'accuracy' in self.evaluation_results or 'precision' in self.evaluation_results:
                sentiment_metrics = {
                    'accuracy': self.evaluation_results.get('accuracy', {}),
                    'precision': self.evaluation_results.get('precision', {}),
                    'recall': self.evaluation_results.get('recall', {}),
                    'f1': self.evaluation_results.get('f1', {})
                }
                
            # 1. 生成準確率、精確率、召回率、F1分數圖表
            self.status_message.emit(f"1. 開始處理「準確率、精確率、召回率、F1分數」區域（{metrics_chart_type}）...", 3000)
            self.progress_bar.setValue(20)
            
            # 創建性能指標圖表子目錄
            metrics_dir = os.path.join(output_dir, "準確率_精確率_召回率_F1")
            os.makedirs(metrics_dir, exist_ok=True)
            
            if metrics_chart_type == "分組條形圖":
                # 從評估結果中提取各個面向的性能指標
                accuracy_values = []
                precision_values = []
                recall_values = []
                f1_values = []
                
                for aspect in aspects:
                    accuracy_values.append(sentiment_metrics.get('accuracy', {}).get(aspect, 0))
                    precision_values.append(sentiment_metrics.get('precision', {}).get(aspect, 0))
                    recall_values.append(sentiment_metrics.get('recall', {}).get(aspect, 0))
                    f1_values.append(sentiment_metrics.get('f1', {}).get(aspect, 0))
                
                # 創建分組條形圖
                fig, ax = plt.subplots(figsize=(12, 8))
                x = np.arange(len(aspects))
                width = 0.2
                
                ax.bar(x - width*1.5, accuracy_values, width, label='準確率')
                ax.bar(x - width/2, precision_values, width, label='精確率')
                ax.bar(x + width/2, recall_values, width, label='召回率')
                ax.bar(x + width*1.5, f1_values, width, label='F1分數')
                
                ax.set_ylabel('分數')
                ax.set_title('各面向的情感分析性能指標')
                ax.set_xticks(x)
                ax.set_xticklabels(aspects, rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
            elif metrics_chart_type == "雷達圖":
                # 創建雷達圖
                fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
                
                # 為每個面向創建一個雷達圖
                n_metrics = 4  # 準確率、精確率、召回率、F1
                theta = np.linspace(0, 2*np.pi, n_metrics, endpoint=False)
                
                # 重複第一個點以閉合多邊形
                theta = np.append(theta, theta[0])
                
                # 性能指標名稱
                metrics_names = ['準確率', '精確率', '召回率', 'F1分數']
                metrics_names.append(metrics_names[0])  # 重複第一個點以閉合多邊形
                
                # 繪製各面向的雷達圖
                for i, aspect in enumerate(aspects):
                    values = [
                        sentiment_metrics.get('accuracy', {}).get(aspect, 0),
                        sentiment_metrics.get('precision', {}).get(aspect, 0),
                        sentiment_metrics.get('recall', {}).get(aspect, 0),
                        sentiment_metrics.get('f1', {}).get(aspect, 0)
                    ]
                    values.append(values[0])  # 重複第一個點以閉合多邊形
                    
                    ax.plot(theta, values, '-', linewidth=2, label=aspect)
                    ax.fill(theta, values, alpha=0.1)
                
                ax.set_xticks(theta[:-1])
                ax.set_xticklabels(metrics_names[:-1])
                ax.set_ylim(0, 1)
                ax.set_title('各面向的情感分析性能雷達圖')
                ax.grid(True)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                
            elif metrics_chart_type == "面積圖":
                # 創建面積圖
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 為每個指標創建一條面積線
                metrics = {
                    '準確率': [sentiment_metrics.get('accuracy', {}).get(aspect, 0) for aspect in aspects],
                    '精確率': [sentiment_metrics.get('precision', {}).get(aspect, 0) for aspect in aspects],
                    '召回率': [sentiment_metrics.get('recall', {}).get(aspect, 0) for aspect in aspects],
                    'F1分數': [sentiment_metrics.get('f1', {}).get(aspect, 0) for aspect in aspects]
                }
                
                x = np.arange(len(aspects))
                
                # 繪製堆疊面積圖
                ax.stackplot(x, metrics.values(), labels=metrics.keys(), alpha=0.7)
                
                ax.set_xticks(x)
                ax.set_xticklabels(aspects, rotation=45, ha='right')
                ax.set_ylabel('分數')
                ax.set_title('各面向的情感分析性能累積分數')
                ax.legend(loc='upper left')
                ax.set_ylim(0, 4)  # 4個指標，最大值為4
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 添加數據來源信息
            self._add_data_source_caption(fig, plt, title="準確率、精確率、召回率、F1分數")
            
            # 保存性能指標圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_img_path = os.path.join(metrics_dir, f"sentiment_metrics_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(metrics_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「準確率、精確率、召回率、F1分數」區域處理完成", 3000)
            self.progress_bar.setValue(50)
            
            # 2. 宏平均F1和微平均F1圖表
            self.status_message.emit(f"2. 開始處理「宏平均F1和微平均F1」區域（{f1_chart_type}）...", 3000)
            
            # 創建F1指標圖表子目錄
            f1_dir = os.path.join(output_dir, "宏平均F1和微平均F1")
            os.makedirs(f1_dir, exist_ok=True)
            
            # 檢查是否有宏平均F1和微平均F1數據
            if 'macro_f1' in sentiment_metrics or 'micro_f1' in sentiment_metrics:
                # 從評估結果中提取宏平均F1和微平均F1數據
                macro_f1_values = {}
                micro_f1_values = {}
                
                for aspect in aspects:
                    macro_f1_values[aspect] = sentiment_metrics.get('macro_f1', {}).get(aspect, 0)
                    micro_f1_values[aspect] = sentiment_metrics.get('micro_f1', {}).get(aspect, 0)
                
                if f1_chart_type == "條形圖":
                    # 創建對比條形圖
                    fig, ax = plt.subplots(figsize=(12, 8))
                    x = np.arange(len(aspects))
                    width = 0.35
                    
                    ax.bar(x - width/2, list(macro_f1_values.values()), width, label='宏平均F1')
                    ax.bar(x + width/2, list(micro_f1_values.values()), width, label='微平均F1')
                    
                    ax.set_ylabel('F1分數')
                    ax.set_title('各面向的宏平均F1和微平均F1')
                    ax.set_xticks(x)
                    ax.set_xticklabels(aspects, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                elif f1_chart_type == "熱力圖":
                    # 創建熱力圖數據
                    f1_data = []
                    for aspect in aspects:
                        f1_data.append([
                            macro_f1_values.get(aspect, 0),
                            micro_f1_values.get(aspect, 0)
                        ])
                    
                    f1_df = pd.DataFrame(f1_data, index=aspects, columns=['宏平均F1', '微平均F1'])
                    
                    # 創建熱力圖
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(f1_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
                    ax.set_title('各面向的宏平均F1和微平均F1熱力圖')
                
                # 添加數據來源信息
                self._add_data_source_caption(fig, plt, title="宏平均F1和微平均F1")
                
                # 保存F1指標圖表
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                f1_img_path = os.path.join(f1_dir, f"sentiment_f1_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(f1_img_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # 如果沒有宏平均F1和微平均F1數據，生成一張提示圖
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "此數據集缺少宏平均F1和微平均F1數據", ha='center', va='center', fontsize=14)
                ax.axis('off')
                
                # 保存提示圖表
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                f1_img_path = os.path.join(f1_dir, f"sentiment_f1_missing_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(f1_img_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            self.status_message.emit(f"✓ 「宏平均F1和微平均F1」區域處理完成", 3000)
            self.progress_bar.setValue(100)
            
            # 返回第一張圖片的路徑
            self.status_message.emit("✓ 情感分析性能可視化已完成", 5000)
            return metrics_img_path
            
        except Exception as e:
            self.status_message.emit(f"生成情感分析可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成情感分析可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _generate_attention_viz(self):
        """生成注意力機制評估可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 檢查是否有必要的數據
            if self.aspect_vectors is None or self.evaluation_results is None:
                QMessageBox.warning(self, "數據缺失", "缺少必要的面向向量或評估結果數據")
                return None
            
            # 導入必要的模組
            import os
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            import traceback
            import plotly.graph_objects as go
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "注意力機制評估")
            os.makedirs(output_dir, exist_ok=True)
            
            self.status_message.emit("開始生成注意力機制評估可視化...", 3000)
            self.progress_bar.setValue(5)
            
            # 獲取當前選項
            attention_chart_type = self.attention_chart_combo.currentText()
            weight_chart_type = self.weight_chart_combo.currentText()
            sample_id = self.sample_id_spin.value()
            
            # 從評估結果中獲取注意力機制相關指標
            attention_mechanisms = []
            topic_coherence = {}
            topic_separation = {}
            combined_score = {}
            
            # 提取面向向量數據
            aspects = list(self.aspect_vectors.keys())
            
            # 檢查評估結果中是否有注意力機制相關指標
            if 'attention_mechanisms' in self.evaluation_results:
                attention_mechanisms = self.evaluation_results['attention_mechanisms']
            elif 'attention_type' in self.evaluation_results:
                attention_mechanisms = [self.evaluation_results['attention_type']]
            else:
                # 如果沒有明確的注意力機制信息，使用默認值
                attention_mechanisms = ['default_attention']
            
            # 提取主題連貫性、分離度和綜合得分
            if 'topic_coherence' in self.evaluation_results:
                # 如果是單一值，轉換為字典形式
                if isinstance(self.evaluation_results['topic_coherence'], (int, float)):
                    for mechanism in attention_mechanisms:
                        topic_coherence[mechanism] = self.evaluation_results['topic_coherence']
                else:
                    topic_coherence = self.evaluation_results['topic_coherence']
            
            if 'topic_separation' in self.evaluation_results:
                # 如果是單一值，轉換為字典形式
                if isinstance(self.evaluation_results['topic_separation'], (int, float)):
                    for mechanism in attention_mechanisms:
                        topic_separation[mechanism] = self.evaluation_results['topic_separation']
                else:
                    topic_separation = self.evaluation_results['topic_separation']
            
            if 'combined_score' in self.evaluation_results:
                # 如果是單一值，轉換為字典形式
                if isinstance(self.evaluation_results['combined_score'], (int, float)):
                    for mechanism in attention_mechanisms:
                        combined_score[mechanism] = self.evaluation_results['combined_score']
                else:
                    combined_score = self.evaluation_results['combined_score']
            
            # 創建注意力機制評估儀表盤
            self.status_message.emit("1. 開始創建注意力機制評估儀表盤...", 3000)
            self.progress_bar.setValue(20)
            
            # 創建儀表盤子目錄
            dashboard_dir = os.path.join(output_dir, "評估儀表盤")
            os.makedirs(dashboard_dir, exist_ok=True)
            
            # 提取注意力機制和對應的評估指標
            mechanisms = list(topic_coherence.keys()) if topic_coherence else attention_mechanisms
            
            # 創建評估儀表盤
            fig = go.Figure()
            
            # 如果有多個注意力機制，創建一個綜合比較
            if len(mechanisms) > 1:
                for i, mechanism in enumerate(mechanisms):
                    coherence_val = topic_coherence.get(mechanism, 0)
                    separation_val = topic_separation.get(mechanism, 0)
                    combined_val = combined_score.get(mechanism, 0)
                    
                    # 創建子圖布局
                    if i == 0:
                        domain_x = [0, 0.45]
                        domain_y = [0.5, 1]
                        title = "主題連貫性"
                    elif i == 1:
                        domain_x = [0.55, 1]
                        domain_y = [0.5, 1]
                        title = "主題分離度"
                    else:
                        domain_x = [0.25, 0.75]
                        domain_y = [0, 0.4]
                        title = "綜合評分"
                    
                    # 添加儀表盤
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=coherence_val if i == 0 else (separation_val if i == 1 else combined_val),
                        title={'text': f"{title}<br><span style='font-size:0.8em'>{mechanism}</span>"},
                        domain={'x': domain_x, 'y': domain_y},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightgray"},
                                {'range': [0.3, 0.6], 'color': "gray"},
                                {'range': [0.6, 1], 'color': "darkgray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
            else:
                # 如果只有一個注意力機制，創建三個指標的儀表盤
                mechanism = mechanisms[0]
                coherence_val = topic_coherence.get(mechanism, 0)
                separation_val = topic_separation.get(mechanism, 0)
                combined_val = combined_score.get(mechanism, 0)
                
                # 主題連貫性儀表盤
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=coherence_val,
                    title={'text': "主題連貫性"},
                    domain={'x': [0, 0.45], 'y': [0.5, 1]},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.6], 'color': "gray"},
                            {'range': [0.6, 1], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                
                # 主題分離度儀表盤
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=separation_val,
                    title={'text': "主題分離度"},
                    domain={'x': [0.55, 1], 'y': [0.5, 1]},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.6], 'color': "gray"},
                            {'range': [0.6, 1], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                
                # 綜合評分儀表盤
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=combined_val,
                    title={'text': "綜合評分"},
                    domain={'x': [0.25, 0.75], 'y': [0, 0.4]},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.3], 'color': "lightgray"},
                            {'range': [0.3, 0.6], 'color': "gray"},
                            {'range': [0.6, 1], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
            
            # 更新圖表布局
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = self.current_dataset if self.current_dataset else "未知數據集"
            
            fig.update_layout(
                title=f'注意力機制評估結果 - {dataset_name}',
                template='plotly_white',
                height=600,
                annotations=[
                    dict(
                        text=f"評估時間: {timestamp}<br>" +
                             f"數據集: {dataset_name}<br>" +
                             f"注意力機制: {', '.join(mechanisms)}",
                        xref="paper",
                        yref="paper",
                        x=0.02,
                        y=0.02,
                        showarrow=False,
                        font=dict(size=10)
                    )
                ]
            )
            
            # 保存圖表為HTML和PNG（根據環境支援情況）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dashboard_dir = os.path.join(output_dir, "評估儀表盤")
            os.makedirs(dashboard_dir, exist_ok=True)
            
            img_path, html_path = save_plotly_figure(
                fig, 
                dashboard_dir, 
                "attention_evaluation",
                timestamp
            )
            
            if not img_path:
                self.logger.error("無法儲存注意力機制評估儀表盤圖表")
                self.status_message.emit("無法儲存注意力機制評估儀表盤圖表", 5000)
                return None
            
            self.status_message.emit(f"✓ 注意力機制評估儀表盤已生成", 3000)
            self.progress_bar.setValue(50)
            
            # 2. 注意力分布可視化
            self.status_message.emit(f"2. 開始處理「注意力分布」區域（{attention_chart_type}）...", 3000)
            
            # 創建注意力分布子目錄
            attention_dist_dir = os.path.join(output_dir, "注意力分布")
            os.makedirs(attention_dist_dir, exist_ok=True)
            
            # 生成注意力分布圖表
            # 如果有注意力權重數據
            attention_weights = {}
            if 'attention_weights' in self.evaluation_results:
                attention_weights = self.evaluation_results['attention_weights']
            elif hasattr(self, 'attention_weights'):
                attention_weights = self.attention_weights
                
            if attention_weights:
                if attention_chart_type == "熱力圖":
                    # 創建注意力權重熱力圖
                    weights_matrix = []
                    for aspect in aspects:
                        if aspect in attention_weights:
                            weights_matrix.append(attention_weights[aspect])
                    
                    if weights_matrix:
                        # 轉換為numpy數組
                        weights_matrix = np.array(weights_matrix)
                        
                        # 如果是多維數組，提取第一個樣本或平均值
                        if len(weights_matrix.shape) > 2:
                            weights_matrix = weights_matrix[0]  # 提取第一個樣本
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(weights_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                                  xticklabels=range(weights_matrix.shape[1]),
                                  yticklabels=aspects, ax=ax)
                        
                        ax.set_title('不同面向的注意力權重熱力圖')
                        ax.set_xlabel('詞元位置')
                        ax.set_ylabel('面向')
                    else:
                        # 如果沒有權重矩陣數據，創建提示圖
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.text(0.5, 0.5, "缺少注意力權重數據", ha='center', va='center', fontsize=14)
                        ax.axis('off')
                
                # 添加數據來源信息
                self._add_data_source_caption(fig, plt, title="注意力分布")
                
                # 保存注意力分布圖表
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                attention_dist_img_path = os.path.join(attention_dist_dir, f"attention_distribution_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(attention_dist_img_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # 如果沒有注意力權重數據，創建提示圖
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "此數據集缺少注意力權重數據", ha='center', va='center', fontsize=14)
                ax.axis('off')
                
                # 保存提示圖表
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                attention_dist_img_path = os.path.join(attention_dist_dir, f"attention_distribution_missing_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(attention_dist_img_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            self.status_message.emit(f"✓ 「注意力分布」區域處理完成", 3000)
            self.progress_bar.setValue(100)
            
            # 返回第一張圖片的路徑
            self.status_message.emit("✓ 注意力機制評估可視化已完成", 5000)
            return img_path
            
        except Exception as e:
            self.status_message.emit(f"生成注意力機制評估可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成注意力機制評估可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _generate_topic_viz(self):
        """生成主題模型評估可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 檢查是否有必要的數據
            if self.aspect_vectors is None or self.evaluation_results is None or self.topics is None:
                QMessageBox.warning(self, "數據缺失", "缺少必要的面向向量、主題或評估結果數據")
                return None
            
            # 導入必要的模組
            import os
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            import traceback
            import plotly.graph_objects as go
            from wordcloud import WordCloud
            import matplotlib.colors as mcolors
            import itertools
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "主題模型評估")
            os.makedirs(output_dir, exist_ok=True)
            
            self.status_message.emit("開始生成主題模型評估可視化...", 3000)
            self.progress_bar.setValue(5)
            
            # 獲取當前選項
            coherence_chart_type = self.coherence_chart_combo.currentText()
            topic_dist_chart_type = self.topic_dist_chart_combo.currentText()
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 從評估結果和主題數據中獲取主題相關信息
            topics = self.topics
            topic_ids = list(topics.keys()) if isinstance(topics, dict) else list(range(len(topics)))
            
            # 1. 主題連貫性可視化
            self.status_message.emit(f"1. 開始處理「主題連貫性」區域（{coherence_chart_type}）...", 3000)
            self.progress_bar.setValue(20)
            
            # 創建主題連貫性子目錄
            coherence_dir = os.path.join(output_dir, "主題連貫性")
            os.makedirs(coherence_dir, exist_ok=True)
            
            # 提取主題連貫性分數
            coherence_scores = {}
            if 'topic_coherence' in self.evaluation_results:
                if isinstance(self.evaluation_results['topic_coherence'], dict):
                    for topic_id in topic_ids:
                        topic_id_str = str(topic_id)
                        if topic_id_str in self.evaluation_results['topic_coherence']:
                            coherence_scores[topic_id_str] = self.evaluation_results['topic_coherence'][topic_id_str]
                else:
                    # 如果主題連貫性是單一值，創建一個固定值的字典
                    for topic_id in topic_ids:
                        coherence_scores[str(topic_id)] = self.evaluation_results['topic_coherence']
            
            if coherence_chart_type == "條形圖":
                if coherence_scores:
                    # 將主題連貫性得分轉為列表
                    topics_list = list(coherence_scores.keys())
                    scores_list = list(coherence_scores.values())
                    
                    # 根據分數排序
                    sorted_indices = np.argsort(scores_list)
                    sorted_topics = [topics_list[i] for i in sorted_indices]
                    sorted_scores = [scores_list[i] for i in sorted_indices]
                    
                    # 創建條形圖
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(sorted_topics, sorted_scores, color='skyblue')
                    
                    # 添加數值標籤
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                              ha='left', va='center')
                    
                    ax.set_xlabel('連貫性分數')
                    ax.set_ylabel('主題')
                    ax.set_title('主題連貫性評分')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                else:
                    # 如果沒有連貫性數據，創建提示圖
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.text(0.5, 0.5, "缺少主題連貫性數據", ha='center', va='center', fontsize=14)
                    ax.axis('off')
                
            elif coherence_chart_type == "詞雲":
                # 創建主題詞雲
                if isinstance(topics, dict) and len(topics) > 0:
                    # 為每個主題創建詞雲
                    n_topics = len(topics)
                    
                    # 計算網格大小
                    grid_size = int(np.ceil(np.sqrt(n_topics)))
                    fig, axes = plt.subplots(grid_size, grid_size, figsize=(5*grid_size, 4*grid_size))
                    axes = axes.flatten()
                    
                    # 生成顏色
                    colors = list(mcolors.TABLEAU_COLORS.values())
                    if len(colors) < n_topics:
                        colors = colors * (n_topics // len(colors) + 1)
                    
                    # 為每個主題繪製詞雲
                    for i, (topic_id, topic_words) in enumerate(topics.items()):
                        if i < len(axes):
                            ax = axes[i]
                            
                            if isinstance(topic_words, list):
                                if isinstance(topic_words[0], str):
                                    # 如果是詞列表，直接使用
                                    words = topic_words
                                    # 創建權重，假設權重遞減
                                    weights = [1.0 - 0.05*j for j in range(len(words))]
                                    # 創建詞-權重字典
                                    word_weights = {words[j]: weights[j] for j in range(len(words))}
                                elif isinstance(topic_words[0], (list, tuple)) and len(topic_words[0]) >= 2:
                                    # 如果是(詞,權重)列表
                                    word_weights = {word: weight for word, weight in topic_words}
                            elif isinstance(topic_words, dict):
                                # 如果已經是詞-權重字典
                                word_weights = topic_words
                            else:
                                # 如果是其他格式，創建一個示例字典
                                word_weights = {"未知格式": 1.0}
                            
                            # 生成詞雲
                            wordcloud = WordCloud(
                                width=400, height=300,
                                background_color='white',
                                color_func=lambda *args, **kwargs: colors[i % len(colors)],
                                max_words=30,
                                font_path='simhei.ttf' if os.path.exists('simhei.ttf') else None,
                                contour_width=1, contour_color='black'
                            ).generate_from_frequencies(word_weights)
                            
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.set_title(f'主題 {topic_id}')
                            ax.axis('off')
                        
                    # 隱藏多餘的子圖
                    for j in range(i+1, len(axes)):
                        axes[j].axis('off')
                    
                    plt.tight_layout()
                else:
                    # 如果沒有主題詞數據，創建提示圖
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.text(0.5, 0.5, "主題詞數據格式不支持創建詞雲", ha='center', va='center', fontsize=14)
                    ax.axis('off')
            
            # 添加數據來源信息
            self._add_data_source_caption(fig, plt, title="主題連貫性")
            
            # 保存主題連貫性圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            coherence_img_path = os.path.join(coherence_dir, f"topic_coherence_{timestamp}.png")
            plt.savefig(coherence_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「主題連貫性」區域處理完成", 3000)
            self.progress_bar.setValue(50)
            
            # 2. 主題分布可視化
            self.status_message.emit(f"2. 開始處理「主題分布」區域（{topic_dist_chart_type}）...", 3000)
            
            # 創建主題分布子目錄
            topic_dist_dir = os.path.join(output_dir, "主題分布")
            os.makedirs(topic_dist_dir, exist_ok=True)
            
            # 從評估結果中提取主題分布數據
            topic_distribution = {}
            if 'topic_distribution' in self.evaluation_results:
                topic_distribution = self.evaluation_results['topic_distribution']
            
            if topic_dist_chart_type == "堆疊柱狀圖" and topic_distribution:
                # 處理主題分布數據
                aspect_names = list(self.aspect_vectors.keys())
                
                # 創建堆疊柱狀圖
                if isinstance(topic_distribution, dict) and len(topic_distribution) > 0:
                    # 將主題分布按比例堆疊
                    dist_data = []
                    labels = []
                    
                    for aspect, dist in topic_distribution.items():
                        if isinstance(dist, dict):
                            dist_data.append(list(dist.values()))
                            labels.append(aspect)
                    
                    if dist_data:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # 創建x軸刻度
                        x = np.arange(len(labels))
                        
                        # 生成顏色
                        colors = plt.colormaps['tab20'](np.linspace(0, 1, len(dist_data[0])))
                        
                        # 設置底部初始值
                        bottoms = np.zeros(len(labels))
                        
                        # 為每個主題繪製堆疊柱形
                        for i in range(len(dist_data[0])):
                            heights = [data[i] for data in dist_data]
                            ax.bar(x, heights, bottom=bottoms, label=f'主題 {i}', color=colors[i])
                            bottoms += heights
                        
                        ax.set_xlabel('面向')
                        ax.set_ylabel('主題比例')
                        ax.set_title('不同面向的主題分布')
                        ax.set_xticks(x)
                        ax.set_xticklabels(labels, rotation=45, ha='right')
                        ax.legend(loc='upper right')
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                    else:
                        # 如果沒有主題分布數據，創建提示圖
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.text(0.5, 0.5, "主題分布數據格式不支持創建堆疊柱狀圖", ha='center', va='center', fontsize=14)
                        ax.axis('off')
                else:
                    # 如果沒有主題分布數據，創建提示圖
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.text(0.5, 0.5, "缺少主題分布數據", ha='center', va='center', fontsize=14)
                    ax.axis('off')
                    
            elif topic_dist_chart_type == "交互式氣泡圖" and topic_distribution:
                # 創建交互式氣泡圖
                if isinstance(topic_distribution, dict) and len(topic_distribution) > 0:
                    # 提取數據
                    aspect_names = []
                    topic_values = []
                    topic_ids = []
                    aspect_ids = []
                    
                    for aspect_idx, (aspect, dist) in enumerate(topic_distribution.items()):
                        if isinstance(dist, dict):
                            for topic_idx, (topic, value) in enumerate(dist.items()):
                                aspect_names.append(aspect)
                                topic_values.append(value)
                                topic_ids.append(f'主題 {topic}')
                                aspect_ids.append(aspect_idx)
                    
                    if topic_values:
                        # 創建氣泡圖的配色方案
                        colors = plt.colormaps['tab20'](np.linspace(0, 1, len(set(topic_ids))))
                        
                        # 為每個主題分配一個顏色
                        unique_topics = list(set(topic_ids))
                        color_map = {topic: colors[i] for i, topic in enumerate(unique_topics)}
                        
                        # 轉換顏色為rgba格式的字符串
                        marker_colors = [f'rgba({int(255*c[0])},{int(255*c[1])},{int(255*c[2])},{c[3]})' 
                                       for c in [color_map[t] for t in topic_ids]]
                        
                        # 創建plotly圖表
                        fig = go.Figure()
                        
                        # 為每個主題創建一個跟蹤對象
                        for topic in unique_topics:
                            indices = [i for i, t in enumerate(topic_ids) if t == topic]
                            
                            fig.add_trace(go.Scatter(
                                x=[aspect_names[i] for i in indices],
                                y=[topic_values[i] for i in indices],
                                mode='markers',
                                marker=dict(
                                    size=[80 * topic_values[i] for i in indices],
                                    color=[marker_colors[i] for i in indices],
                                    line=dict(width=2, color='DarkSlateGrey')
                                ),
                                name=topic,
                                text=[f'面向: {aspect_names[i]}<br>主題: {topic_ids[i]}<br>比例: {topic_values[i]:.2f}' for i in indices],
                                hoverinfo='text'
                            ))
                        
                        # 更新布局
                        fig.update_layout(
                            title='主題分布交互式氣泡圖',
                            xaxis=dict(
                                title='面向',
                                tickangle=45,
                                gridcolor='lightgray'
                            ),
                            yaxis=dict(
                                title='主題比例',
                                gridcolor='lightgray'
                            ),
                            hovermode='closest',
                            template='plotly_white'
                        )
                        
                        # 保存為HTML和PNG（根據環境支援情況）
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        topic_dist_dir = os.path.join(output_dir, "主題分布")
                        os.makedirs(topic_dist_dir, exist_ok=True)
                        
                        img_path, html_path = save_plotly_figure(
                            fig, 
                            topic_dist_dir, 
                            "topic_distribution_bubble",
                            timestamp
                        )
                        
                        if img_path:
                            self.status_message.emit(f"✓ 「主題分布」區域處理完成", 3000)
                            self.progress_bar.setValue(100)
                            
                            # 返回主題連貫性圖片路徑
                            self.status_message.emit("✓ 主題模型評估可視化已完成", 5000)
                            return coherence_img_path
                        else:
                            # 如果沒有成功儲存圖片，顯示錯誤
                            self.logger.error("無法儲存主題分布氣泡圖")
                            self.status_message.emit("無法儲存主題分布氣泡圖，使用靜態圖表代替", 5000)
                else:
                    # 如果沒有主題分布數據，創建提示圖
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.text(0.5, 0.5, "主題分布數據格式不支持創建交互式氣泡圖", ha='center', va='center', fontsize=14)
                    ax.axis('off')
            else:
                # 如果沒有主題分布數據，創建提示圖
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "缺少主題分布數據或選擇了不支持的圖表類型", ha='center', va='center', fontsize=14)
                ax.axis('off')
            
            # 添加數據來源信息
            self._add_data_source_caption(fig, plt, title="主題分布")
            
            # 保存主題分布圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_dist_img_path = os.path.join(topic_dist_dir, f"topic_distribution_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(topic_dist_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「主題分布」區域處理完成", 3000)
            self.progress_bar.setValue(100)
            
            # 返回主題連貫性圖片路徑
            self.status_message.emit("✓ 主題模型評估可視化已完成", 5000)
            return coherence_img_path
            
        except Exception as e:
            self.status_message.emit(f"生成主題模型評估可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成主題模型評估可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _generate_comprehensive_viz(self):
        """生成綜合比較可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 檢查是否有必要的數據
            if self.aspect_vectors is None or self.evaluation_results is None:
                QMessageBox.warning(self, "數據缺失", "缺少必要的面向向量或評估結果數據")
                return None
            
            # 導入必要的模組
            from datetime import datetime
            from mpl_toolkits.mplot3d import Axes3D
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "綜合比較")
            os.makedirs(output_dir, exist_ok=True)
            
            self.status_message.emit("開始生成綜合比較可視化...", 3000)
            self.progress_bar.setValue(5)
            
            # 獲取當前選項
            dim_method = self.dim_method_combo.currentText()
            color_by = self.color_by_combo.currentText()
            multi_chart_type = self.multi_chart_combo.currentText()
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 從評估結果中獲取分析模型相關指標
            metrics = self.evaluation_results
            
            # 提取面向向量數據
            aspect_vectors = self.aspect_vectors
            aspect_names = list(aspect_vectors.keys())
            
            # 嘗試提取面向向量的值進行降維
            vectors = []
            for aspect, vector in aspect_vectors.items():
                if isinstance(vector, list) or isinstance(vector, np.ndarray):
                    vectors.append(vector)
                elif isinstance(vector, dict) and 'vector' in vector:
                    vectors.append(vector['vector'])
                else:
                    self.logger.warning(f"無法處理的面向向量格式: {type(vector)} 對於面向 {aspect}")
            
            # 1. 降維可視化
            self.status_message.emit(f"1. 開始處理「降維可視化」區域（{dim_method}）...", 3000)
            self.progress_bar.setValue(20)
            
            # 創建降維可視化子目錄
            dim_reduction_dir = os.path.join(output_dir, "降維可視化")
            os.makedirs(dim_reduction_dir, exist_ok=True)
            
            # 如果有面向向量數據，執行降維可視化
            if vectors and len(vectors) > 0:
                # 將向量轉換為numpy數組
                vectors_array = np.array(vectors)
                
                # 檢查向量維度是否足夠進行降維
                if vectors_array.shape[0] < 3:
                    self.logger.warning(f"面向數量過少 ({vectors_array.shape[0]})，難以進行有意義的降維可視化")
                    self.status_message.emit(f"面向數量過少 ({vectors_array.shape[0]})，難以進行有意義的降維可視化", 5000)
                    
                    # 創建提示圖
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.text(0.5, 0.5, f"面向數量過少 ({vectors_array.shape[0]})，難以進行有意義的降維可視化", 
                           ha='center', va='center', fontsize=14)
                    ax.axis('off')
                else:
                    # 根據選擇的降維方法執行降維
                    if dim_method == "t-SNE":
                        from sklearn.manifold import TSNE
                        model = TSNE(n_components=2, perplexity=min(30, len(vectors_array)-1), random_state=42)
                        reduced_data = model.fit_transform(vectors_array)
                    elif dim_method == "PCA":
                        from sklearn.decomposition import PCA
                        model = PCA(n_components=2)
                        reduced_data = model.fit_transform(vectors_array)
                    elif dim_method == "UMAP":
                        try:
                            import umap
                            model = umap.UMAP(n_components=2, random_state=42)
                            reduced_data = model.fit_transform(vectors_array)
                        except ImportError:
                            self.logger.error("UMAP模組未安裝，切換到PCA")
                            self.status_message.emit("UMAP模組未安裝，切換到PCA", 5000)
                            from sklearn.decomposition import PCA
                            model = PCA(n_components=2)
                            reduced_data = model.fit_transform(vectors_array)
                    else:
                        # 默認使用PCA
                        from sklearn.decomposition import PCA
                        model = PCA(n_components=2)
                        reduced_data = model.fit_transform(vectors_array)
                    
                    # 創建散點圖
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # 獲取著色數據
                    if color_by == "內聚度" and 'topic_coherence' in metrics:
                        if isinstance(metrics['topic_coherence'], dict):
                            colors = [metrics['topic_coherence'].get(aspect, 0) for aspect in aspect_names]
                        else:
                            self.logger.warning("主題內聚度不是字典格式，無法用於著色")
                            colors = None
                    elif color_by == "分離度" and 'topic_separation' in metrics:
                        if isinstance(metrics['topic_separation'], dict):
                            colors = [metrics['topic_separation'].get(aspect, 0) for aspect in aspect_names]
                        else:
                            self.logger.warning("主題分離度不是字典格式，無法用於著色")
                            colors = None
                    else:
                        colors = None
                    
                    # 繪製散點圖
                    scatter = ax.scatter(
                        reduced_data[:, 0], 
                        reduced_data[:, 1],
                        c=colors,
                        cmap='viridis' if colors is not None else None,
                        s=100,
                        alpha=0.7
                    )
                    
                    # 添加顏色條
                    if colors is not None:
                        plt.colorbar(scatter, ax=ax, label=color_by)
                    
                    # 為每個點添加標籤
                    for i, aspect in enumerate(aspect_names):
                        ax.annotate(
                            aspect,
                            (reduced_data[i, 0], reduced_data[i, 1]),
                            fontsize=9,
                            ha='center',
                            va='bottom',
                            xytext=(0, 5),
                            textcoords='offset points'
                        )
                    
                    # 添加標題和軸標籤
                    plt.title(f"面向向量{dim_method}降維可視化", fontsize=14)
                    plt.xlabel("第一維度", fontsize=12)
                    plt.ylabel("第二維度", fontsize=12)
                    plt.grid(True, linestyle='--', alpha=0.7)
                
                # 添加數據來源信息
                self._add_data_source_caption(fig, plt, title="降維可視化")
                
                # 保存圖片
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dim_reduction_img_path = os.path.join(dim_reduction_dir, f"dim_reduction_{timestamp}.png")
                plt.savefig(dim_reduction_img_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.status_message.emit(f"✓ 「降維可視化」區域處理完成", 3000)
                self.progress_bar.setValue(50)
                
                # 返回降維可視化圖片路徑
                self.status_message.emit("✓ 綜合比較可視化已完成", 5000)
                return dim_reduction_img_path
            else:
                self.logger.error("無法提取有效的面向向量數據進行降維可視化")
                self.status_message.emit("無法提取有效的面向向量數據進行降維可視化", 5000)
                
                # 創建提示圖
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "無法提取有效的面向向量數據進行降維可視化", 
                       ha='center', va='center', fontsize=14)
                ax.axis('off')
                
                # 添加數據來源信息
                self._add_data_source_caption(fig, plt, title="錯誤")
                
                # 保存圖片
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                error_img_path = os.path.join(dim_reduction_dir, f"error_{timestamp}.png")
                plt.savefig(error_img_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.progress_bar.setValue(0)
                return None
            
        except Exception as e:
            self.status_message.emit(f"生成綜合比較可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成綜合比較可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _add_data_source_caption(self, fig, plt, title=None):
        """在圖表下方添加數據來源提示
        
        Args:
            fig: 圖表對象
            plt: matplotlib.pyplot 對象
            title: 額外的標題信息
        """
        if hasattr(self, 'data_sources') and self.data_sources:
            # 根據當前視覺化類型選擇相應的數據來源
            source_text = ""
            
            current_tab_index = self.viz_options_stack.currentIndex()
            source_type = ""
            
            if current_tab_index == 0:  # 面向向量質量
                source_type = "面向向量"
                if "評估結果" in self.data_sources:
                    source_text = f"數據來源: {self.data_sources['評估結果']}"
                elif "面向向量" in self.data_sources:
                    source_text = f"數據來源: {self.data_sources['面向向量']}"
            elif current_tab_index == 1:  # 情感分析性能
                source_type = "評估結果"
                if "評估結果" in self.data_sources:
                    source_text = f"數據來源: {self.data_sources['評估結果']}"
            elif current_tab_index == 2:  # 注意力機制評估
                source_type = "注意力權重"
                if "注意力權重" in self.data_sources:
                    source_text = f"數據來源: {self.data_sources['注意力權重']}"
                elif "評估結果" in self.data_sources:
                    source_text = f"數據來源: {self.data_sources['評估結果']}"
            elif current_tab_index == 3:  # 主題模型評估
                source_type = "主題模型"
                if "主題模型" in self.data_sources:
                    source_text = f"數據來源: {self.data_sources['主題模型']}"
            elif current_tab_index == 4:  # 綜合比較
                # 綜合比較可能使用多個數據源
                sources = []
                if "面向向量" in self.data_sources:
                    sources.append(f"面向向量: {self.data_sources['面向向量']}")
                if "評估結果" in self.data_sources:
                    sources.append(f"評估結果: {self.data_sources['評估結果']}")
                if sources:
                    source_text = "數據來源:\n" + "\n".join(sources)
            
            # 如果沒有找到特定的數據來源但有主要結果
            if not source_text and "主要結果" in self.data_sources:
                source_text = f"數據來源: {self.data_sources['主要結果']}"
                
            # 如果有標題信息，添加到來源文本中
            if title:
                source_text = f"{title}\n{source_text}"
                
            # 如果有來源信息，添加到圖表底部
            if source_text:
                # 添加來源文本到底部
                plt.figtext(0.02, 0.01, source_text, fontsize=8, color='gray',
                          ha='left', va='bottom')
                
                # 調整布局，為底部文本留出空間
                plt.subplots_adjust(bottom=0.15)