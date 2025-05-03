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

        # 默認輸出目錄 - 更新為使用 Part04_/1_output/visualizations 目錄
        self.output_dir = os.path.join("Part04_", "1_output", "visualizations")
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
        """瀏覽選擇結果文件"""
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
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇結果文件",
            output_dir,
            "結果文件 (*.json);;所有文件 (*.*)"
        )
        
        if file_path:
            self.logger.info(f"已選擇結果檔案: {file_path}")
            self.load_results(file_path)
        else:
            self.logger.debug("用戶取消了檔案選擇")

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
                if isinstance(results['aspect_vectors'], dict):
                    self.aspect_vectors = results['aspect_vectors']
                    self.logger.info(f"成功載入面向向量字典: {len(self.aspect_vectors)} 個項目")
                elif isinstance(results['aspect_vectors'], list):
                    self.aspect_vectors = results['aspect_vectors']
                    self.logger.info(f"成功載入面向向量列表, 長度: {len(self.aspect_vectors)}")
                else:
                    self.logger.warning(f"無法識別的面向向量格式")
            
            # 從文件中獲取主題
            if 'topics' in results:
                self.topics = results['topics']
                self.logger.info(f"成功載入 {len(self.topics)} 個主題")
            
            # 從文件中獲取評估結果 - 修改以同時支持多種格式
            # 格式1: metrics 作為頂層鍵
            if 'metrics' in results:
                self.evaluation_results = results['metrics']
                self.logger.info(f"從 metrics 鍵載入評估結果")
            # 格式2: evaluation 作為頂層鍵
            elif 'evaluation' in results:
                self.evaluation_results = results['evaluation']
                self.logger.info(f"從 evaluation 鍵載入評估結果")
            # 格式3: metrics_details 作為頂層鍵 (備用方案)
            elif 'metrics_details' in results:
                self.evaluation_results = results['metrics_details']
                self.logger.info(f"從 metrics_details 鍵載入評估結果")
                
            # 如果沒有找到評估結果，則記錄警告
            if not self.evaluation_results:
                self.logger.warning(f"在結果文件中沒有找到評估結果")
            else:
                self.logger.info(f"評估結果包含: {list(self.evaluation_results.keys()) if isinstance(self.evaluation_results, dict) else '非字典格式'}")
            
            # 設置當前數據集名稱
            self.current_dataset = os.path.basename(file_path).replace('.json', '')
            
            # 更新結果信息
            self._update_result_info()
            
            # 成功載入提示
            self.status_message.emit(f"已成功載入結果文件: {os.path.basename(file_path)}", 3000)
            
            # 更新結果文件路徑
            self.result_file_path = file_path
            
            return True
                
        except Exception as e:
            self.logger.error(f"載入結果文件出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "載入出錯", f"載入結果文件時出錯:\n{str(e)}")
            return False

    def _update_result_info(self):
        """更新結果信息標籤"""
        if self.current_dataset:
            info_text = f"當前結果: {self.current_dataset}"
            
            if isinstance(self.topics, dict):
                info_text += f" | 主題數: {len(self.topics)}"
            
            if self.aspect_vectors is not None:
                if isinstance(self.aspect_vectors, dict):
                    info_text += f" | 向量數: {len(self.aspect_vectors)}"
                elif isinstance(self.aspect_vectors, list):
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
    
    def generate_visualization(self):
        """生成可視化並自動保存"""
        if not self.current_dataset:
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
            
    def _generate_vector_quality_viz(self):
        """生成面向向量質量可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 導入必要的模組
            import os
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
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
            
            # 創建示例數據
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os
            
            # 創建圖表類型選擇
            viz_type_options = {
                "內聚度與分離度": cohesion_chart_type,
                "綜合得分": combined_chart_type,
                "輪廓係數": silhouette_chart_type,
                "困惑度": "line_chart"  # 困惑度固定使用折線圖
            }
            
            self.status_message.emit(f"處理「面向向量質量」標籤頁: 已選取 {len(viz_type_options)} 個視覺化區域", 3000)
            
            # 根據用戶選擇決定要生成的圖表類型
            # 這裡我們默認生成所有類型的圖表併返回第一張
            img_paths = []
            
            # 生成內聚度與分離度圖表
            self.status_message.emit(f"1. 開始處理「內聚度與分離度」區域（{cohesion_chart_type}）...", 3000)
            self.progress_bar.setValue(15)
            
            attention_mechanisms = ['相似度注意力', '關鍵詞注意力', '自注意力', '綜合注意力']
            cohesion_values = np.random.uniform(0.6, 0.9, len(attention_mechanisms))
            separation_values = np.random.uniform(0.5, 0.8, len(attention_mechanisms))
            
            # 1. 內聚度與分離度圖表
            if cohesion_chart_type == "條形圖":
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(attention_mechanisms))
                width = 0.35
                
                ax.bar(x - width/2, cohesion_values, width, label='內聚度')
                ax.bar(x + width/2, separation_values, width, label='分離度')
                
                ax.set_ylabel('分數')
                ax.set_title('不同注意力機制的內聚度與分離度')
                ax.set_xticks(x)
                ax.set_xticklabels(attention_mechanisms)
                ax.legend()
                
            elif cohesion_chart_type == "散點圖":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(cohesion_values, separation_values, s=100)
                
                # 添加標籤
                for i, mechanism in enumerate(attention_mechanisms):
                    ax.annotate(mechanism, (cohesion_values[i], separation_values[i]),
                              textcoords="offset points", xytext=(0,10), ha='center')
                
                ax.set_xlabel('內聚度')
                ax.set_ylabel('分離度')
                ax.set_title('內聚度與分離度的關係散點圖')
                ax.grid(True)
                
            elif cohesion_chart_type == "樹狀圖":
                # 使用条形图模拟树状图
                combined_scores = cohesion_values * 0.5 + separation_values * 0.5
                
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(attention_mechanisms))
                width = 0.7
                
                ax.bar(x, combined_scores, width, label='綜合得分')
                ax.set_ylabel('綜合得分')
                ax.set_title('內聚度與分離度的綜合得分')
                ax.set_xticks(x)
                ax.set_xticklabels(attention_mechanisms)
                ax.legend()
            
            # 創建「內聚度與分離度」子目錄
            cohesion_dir = os.path.join(output_dir, "內聚度與分離度")
            os.makedirs(cohesion_dir, exist_ok=True)
            
            # 保存內聚度分離度圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cs_img_path = os.path.join(cohesion_dir, f"vector_quality_cohesion_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(cs_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            img_paths.append(cs_img_path)
            
            self.status_message.emit(f"✓ 「內聚度與分離度」區域處理完成", 3000)
            self.progress_bar.setValue(30)
            
            # 2. 綜合得分圖表
            self.status_message.emit(f"2. 開始處理「綜合得分」區域（{combined_chart_type}）...", 3000)
            # 創建示例數據
            mechanisms = ['相似度注意力', '關鍵詞注意力', '自注意力', '綜合注意力']
            metrics = ['內聚度', '分離度', '輪廓係數', 'F1分數']
            
            # 創建隨機得分矩陣
            scores_matrix = np.random.uniform(0.5, 0.9, (len(mechanisms), len(metrics)))
            
            if combined_chart_type == "條形圖":
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 創建組合得分
                combined_scores = np.mean(scores_matrix, axis=1)
                
                bars = ax.bar(mechanisms, combined_scores, color='skyblue')
                
                # 添加數值標籤
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}', ha='center', va='bottom')
                
                ax.set_ylim(0, 1.0)
                ax.set_ylabel('綜合得分')
                ax.set_title('不同注意力機制的綜合性能得分')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
            elif combined_chart_type == "熱力圖":
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 創建熱力圖
                sns.heatmap(scores_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                          xticklabels=metrics, yticklabels=mechanisms, ax=ax)
                
                ax.set_title('不同注意力機制在各指標上的得分熱力圖')
            
            # 創建「綜合得分」子目錄
            combined_dir = os.path.join(output_dir, "綜合得分")
            os.makedirs(combined_dir, exist_ok=True)
                
            # 保存綜合得分圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_img_path = os.path.join(combined_dir, f"vector_quality_combined_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(combined_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            img_paths.append(combined_img_path)
            
            self.status_message.emit(f"✓ 「綜合得分」區域處理完成", 3000)
            self.progress_bar.setValue(50)
            
            # 3. 輪廓係數圖表
            self.status_message.emit(f"3. 開始處理「輪廓係數」區域（{silhouette_chart_type}）...", 3000)
            # 創建示例數據
            n_samples = 200
            n_clusters = 5
            
            # 創建隨機輪廓係數值 (模擬不同機制的結果)
            silhouette_values = []
            cluster_labels = []
            
            for mechanism_id in range(len(mechanisms)):
                # 為每個機制創建隨機輪廓係數值 (-1到1之間，通常好的聚類大於0)
                mech_values = np.random.uniform(-0.1, 0.8, n_samples // len(mechanisms))
                silhouette_values.extend(mech_values)
                
                # 為每個樣本分配聚類標籤
                cluster_ids = np.random.randint(0, n_clusters, size=len(mech_values))
                cluster_labels.extend(cluster_ids)
                
                # 為每個樣本記錄相應的機制
                mechanism_labels = [mechanisms[mechanism_id]] * len(mech_values)
                if mechanism_id == 0:
                    all_mechanism_labels = mechanism_labels
                else:
                    all_mechanism_labels.extend(mechanism_labels)
            
            # 創建DataFrame用於可視化
            silhouette_df = pd.DataFrame({
                '輪廓係數': silhouette_values,
                '聚類': [f'聚類{c}' for c in cluster_labels],
                '注意力機制': all_mechanism_labels
            })
            
            if silhouette_chart_type == "輪廓圖":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 為每個機制創建輪廓圖 (簡化版)
                for mechanism in mechanisms:
                    mech_data = silhouette_df[silhouette_df['注意力機制'] == mechanism]
                    sns.kdeplot(mech_data['輪廓係數'], label=mechanism, ax=ax)
                
                ax.set_xlabel('輪廓係數')
                ax.set_ylabel('密度')
                ax.set_title('不同注意力機制的輪廓係數分布')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # 添加垂直線表示平均值
                for mechanism in mechanisms:
                    mech_data = silhouette_df[silhouette_df['注意力機制'] == mechanism]
                    avg = mech_data['輪廓係數'].mean()
                    ax.axvline(avg, linestyle='--', alpha=0.6)
                    ax.text(avg, 0.1, f'{avg:.2f}', horizontalalignment='center')
                    
            elif silhouette_chart_type == "小提琴圖":
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 創建小提琴圖
                sns.violinplot(x='注意力機制', y='輪廓係數', data=silhouette_df,
                             inner='box', ax=ax)
                
                ax.set_title('不同注意力機制的輪廓係數分布')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 創建「輪廓係數」子目錄
            silhouette_dir = os.path.join(output_dir, "輪廓係數")
            os.makedirs(silhouette_dir, exist_ok=True)
            
            # 保存輪廓係數圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            silhouette_img_path = os.path.join(silhouette_dir, f"vector_quality_silhouette_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(silhouette_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            img_paths.append(silhouette_img_path)
            
            self.status_message.emit(f"✓ 「輪廓係數」區域處理完成", 3000)
            self.progress_bar.setValue(75)
            
            # 4. 困惑度圖表 (只用折線圖)
            self.status_message.emit(f"4. 開始處理「困惑度」區域 (範圍: {min_topics}-{max_topics})...", 3000)
            # 創建困惑度數據
            topic_nums = list(range(min_topics, max_topics + 1))
            perplexities = []
            
            # 為每個機制創建困惑度值
            for mechanism in mechanisms:
                # 使用對數衰減函數模擬困惑度隨主題數增加而降低的情況
                base_perp = 1000 + np.random.uniform(-200, 200)  # 基準困惑度
                mech_perplexities = [base_perp * np.exp(-0.05 * i) + 100 + np.random.uniform(-20, 20) 
                                   for i in range(len(topic_nums))]
                perplexities.append(mech_perplexities)
            
            # 繪製困惑度曲線
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for i, mechanism in enumerate(mechanisms):
                ax.plot(topic_nums, perplexities[i], marker='o', label=mechanism)
            
            ax.set_xlabel('主題數量')
            ax.set_ylabel('困惑度')
            ax.set_title(f'主題數{min_topics}至{max_topics}的困惑度變化')
            ax.legend()
            ax.grid(True)
            
            # 創建「困惑度」子目錄
            perplexity_dir = os.path.join(output_dir, "困惑度")
            os.makedirs(perplexity_dir, exist_ok=True)
            
            # 保存困惑度圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            perplexity_img_path = os.path.join(perplexity_dir, f"vector_quality_perplexity_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(perplexity_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            img_paths.append(perplexity_img_path)
            
            self.status_message.emit(f"✓ 「困惑度」區域處理完成", 3000)
            self.progress_bar.setValue(95)
            
            # 返回第一張圖片的路徑
            if img_paths:
                self.status_message.emit(f"✓ 所有面向向量質量可視化已完成，共生成 {len(img_paths)} 個圖表", 5000)
                self.progress_bar.setValue(100)
                return img_paths[0]  # 返回第一張圖片的路徑
            return None
            
        except Exception as e:
            self.status_message.emit(f"生成面向向量質量可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成面向向量質量可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _generate_sentiment_viz(self):
        """生成情感分析性能指標可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 導入必要的模組
            import os
            import numpy as np
            import matplotlib.pyplot as plt
            from datetime import datetime
            import traceback
            
            # 獲取選項
            chart_type = self.metrics_chart_combo.currentText()
            
            # 創建示例數據（實際項目中應從模型結果獲取）
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "情感分析性能")
            os.makedirs(output_dir, exist_ok=True)
            
            # 模擬不同注意力機制的指標數據
            attention_mechanisms = ['相似度注意力', '關鍵詞注意力', '自注意力', '綜合注意力']
            accuracy_values = np.random.uniform(0.7, 0.9, len(attention_mechanisms))
            precision_values = np.random.uniform(0.65, 0.85, len(attention_mechanisms))
            recall_values = np.random.uniform(0.6, 0.9, len(attention_mechanisms))
            f1_values = np.random.uniform(0.65, 0.88, len(attention_mechanisms))
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 更新狀態和進度條
            self.status_message.emit("開始生成情感分析性能指標可視化...", 3000)
            self.progress_bar.setValue(10)
            
            # 根據選擇的圖表類型生成不同的圖表
            self.status_message.emit(f"1. 開始處理「準確率、精確率、召回率、F1分數」區域（{chart_type}）...", 3000)
            self.progress_bar.setValue(30)
            
            if chart_type == "分組條形圖":
                fig, ax = plt.subplots(figsize=(12, 6))
                x = np.arange(len(attention_mechanisms))
                width = 0.2
                
                ax.bar(x - width*1.5, accuracy_values, width, label='準確率')
                ax.bar(x - width/2, precision_values, width, label='精確率')
                ax.bar(x + width/2, recall_values, width, label='召回率')
                ax.bar(x + width*1.5, f1_values, width, label='F1分數')
                
                ax.set_ylabel('分數')
                ax.set_title('不同注意力機制的性能指標')
                ax.set_xticks(x)
                ax.set_xticklabels(attention_mechanisms)
                ax.legend()
                
            elif chart_type == "雷達圖":
                # 雷達圖需要閉合的數據
                metrics = ['準確率', '精確率', '召回率', 'F1分數']
                
                # 創建子圖
                fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
                axes = axes.flatten()
                
                for i, mechanism in enumerate(attention_mechanisms):
                    ax = axes[i]
                    # 獲取當前機制的指標數據
                    values = [accuracy_values[i], precision_values[i], recall_values[i], f1_values[i]]
                    # 閉合數據
                    values = np.append(values, values[0])
                    
                    # 創建角度均勻分布
                    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
                    # 閉合角度
                    angles = np.append(angles, angles[0])
                    
                    # 繪製雷達圖
                    ax.plot(angles, values, 'o-', linewidth=2)
                    ax.fill(angles, values, alpha=0.25)
                    ax.set_title(mechanism)
                    
                    # 設置角度標籤
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(metrics)
                    
                    # 設置y軸範圍
                    ax.set_ylim(0.5, 1)
                
                plt.tight_layout()
                fig.suptitle('不同注意力機制的性能雷達圖', fontsize=16, y=1.05)
                
            elif chart_type == "面積圖":
                fig, ax = plt.subplots(figsize=(10, 6))
                
                x = np.arange(len(attention_mechanisms))
                
                ax.fill_between(x, accuracy_values, label='準確率', alpha=0.7)
                ax.fill_between(x, precision_values, label='精確率', alpha=0.7)
                ax.fill_between(x, recall_values, label='召回率', alpha=0.7)
                ax.fill_between(x, f1_values, label='F1分數', alpha=0.7)
                
                ax.set_ylabel('分數')
                ax.set_title('不同注意力機制的性能指標面積圖')
                ax.set_xticks(x)
                ax.set_xticklabels(attention_mechanisms)
                ax.legend()
            
            self.status_message.emit(f"✓ 「準確率、精確率、召回率、F1分數」區域處理完成", 3000)
            self.progress_bar.setValue(60)
            
            # 根據F1圖表類型創建子目錄
            f1_chart_type = self.f1_chart_combo.currentText()
            metrics_dir = os.path.join(output_dir, "準確率精確率召回率F1")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # 保存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(metrics_dir, f"sentiment_analysis_{chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 創建宏平均F1和微平均F1圖表
            self.status_message.emit(f"2. 開始處理「宏平均F1和微平均F1」區域（{f1_chart_type}）...", 3000)
            self.progress_bar.setValue(70)
            
            macro_micro_dir = os.path.join(output_dir, "宏平均微平均F1")
            os.makedirs(macro_micro_dir, exist_ok=True)
            
            # 生成宏平均F1和微平均F1數據
            mechanisms = ['相似度注意力', '關鍵詞注意力', '自注意力', '綜合注意力']
            macro_f1_values = np.random.uniform(0.6, 0.85, len(mechanisms))
            micro_f1_values = np.random.uniform(0.65, 0.9, len(mechanisms))
            
            if f1_chart_type == "條形圖":
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(mechanisms))
                width = 0.35
                
                ax.bar(x - width/2, macro_f1_values, width, label='宏平均F1')
                ax.bar(x + width/2, micro_f1_values, width, label='微平均F1')
                
                ax.set_ylabel('F1分數')
                ax.set_title('不同注意力機制的宏平均F1和微平均F1')
                ax.set_xticks(x)
                ax.set_xticklabels(mechanisms)
                ax.legend()
                
            elif f1_chart_type == "熱力圖":
                # 創建包含兩種F1分數的熱力圖
                data = np.vstack([macro_f1_values, micro_f1_values])
                fig, ax = plt.subplots(figsize=(12, 4))
                
                im = ax.imshow(data, cmap="YlGnBu")
                
                # 設置坐標軸
                ax.set_xticks(np.arange(len(mechanisms)))
                ax.set_yticks([0, 1])
                ax.set_xticklabels(mechanisms)
                ax.set_yticklabels(['宏平均F1', '微平均F1'])
                
                # 添加顏色條和數值標籤
                plt.colorbar(im)
                
                # 添加數值標籤
                for i in range(len(['宏平均F1', '微平均F1'])):
                    for j in range(len(mechanisms)):
                        text = ax.text(j, i, f"{data[i, j]:.2f}",
                                     ha="center", va="center", color="black")
                
                ax.set_title('不同注意力機制的宏平均F1和微平均F1熱力圖')
            
            # 保存F1圖表
            f1_img_path = os.path.join(macro_micro_dir, f"sentiment_f1_{f1_chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(f1_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「宏平均F1和微平均F1」區域處理完成", 3000)
            self.progress_bar.setValue(100)
            
            # 返回第一張圖片的路徑
            self.status_message.emit(f"✓ 所有情感分析性能可視化已完成", 5000)
            return img_path
            
        except Exception as e:
            self.status_message.emit(f"生成情感分析性能指標可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成情感分析性能指標可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
            
    def _generate_attention_viz(self):
        """生成注意力機制評估可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 獲取選項
            chart_type = self.attention_chart_combo.currentText()
            sample_id = self.sample_id_spin.value()
            weight_chart_type = self.weight_chart_combo.currentText()
            
            # 創建示例數據（實際項目中應從模型結果獲取）
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import os
            import seaborn as sns
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "注意力機制評估")
            os.makedirs(output_dir, exist_ok=True)
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            self.status_message.emit("開始生成注意力機制評估可視化...", 3000)
            self.progress_bar.setValue(5)
            
            # 1. 注意力分布可視化
            self.status_message.emit(f"1. 開始處理「注意力分布」區域（{chart_type}）...", 3000)
            self.progress_bar.setValue(20)
            
            # 創建「注意力分布」子目錄
            attention_dist_dir = os.path.join(output_dir, "注意力分布")
            os.makedirs(attention_dist_dir, exist_ok=True)
            
            if chart_type == "熱力圖":
                # 生成模擬注意力矩陣數據
                n_aspects = 5
                n_words = 10
                attention_matrix = np.random.rand(n_aspects, n_words)
                # 正規化
                attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)
                
                # 創建標籤
                aspect_labels = [f"面向{i+1}" for i in range(n_aspects)]
                word_labels = [f"詞{i+1}" for i in range(n_words)]
                
                # 創建熱力圖
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(attention_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                           xticklabels=word_labels, yticklabels=aspect_labels, ax=ax)
                ax.set_title(f"樣本 {sample_id} 的注意力分布熱力圖")
                ax.set_xlabel("詞彙")
                ax.set_ylabel("面向")
                
            elif chart_type == "文本注釋圖":
                # 創建示例句子和注意力權重
                sentence = "這是一個示例句子用於展示文本的注意力權重分布"
                words = sentence.split()
                attention_weights = np.random.rand(len(words))
                attention_weights = attention_weights / attention_weights.sum()
                
                # 創建圖表
                fig, ax = plt.subplots(figsize=(12, 4))
                
                # 定義顏色映射
                cmap = plt.cm.YlOrRd
                
                # 繪製文本注釋
                for i, (word, weight) in enumerate(zip(words, attention_weights)):
                    color = cmap(weight)
                    ax.text(i, 0.5, word, ha='center', va='center', color='black',
                          bbox=dict(facecolor=color, alpha=0.7, boxstyle='round,pad=0.5'),
                          fontsize=12 + weight * 15)  # 根據注意力權重調整字體大小
                
                ax.set_xlim(-0.5, len(words) - 0.5)
                ax.set_ylim(0, 1)
                ax.set_title(f"樣本 {sample_id} 的注意力文本注釋圖")
                ax.axis('off')  # 隱藏坐標軸
                
                # 添加顏色條
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array(attention_weights)
                cbar = plt.colorbar(sm)
                cbar.set_label('注意力權重')
                
            elif chart_type == "弦圖":
                # 由於弦圖複雜，我們繪製一個簡化版本
                # 使用矩形面積圖模擬弦圖
                n_aspects = 5
                n_mechanisms = 4
                
                # 生成模擬數據
                flow_matrix = np.random.rand(n_aspects, n_mechanisms)
                flow_matrix = flow_matrix / flow_matrix.sum() * 100  # 轉換為百分比
                
                aspect_labels = [f"面向{i+1}" for i in range(n_aspects)]
                mechanism_labels = ['相似度注意力', '關鍵詞注意力', '自注意力', '綜合注意力']
                
                # 創建堆疊條形圖
                fig, ax = plt.subplots(figsize=(12, 6))
                
                bottom = np.zeros(n_mechanisms)
                for i, aspect in enumerate(aspect_labels):
                    ax.bar(mechanism_labels, flow_matrix[i], bottom=bottom, label=aspect)
                    bottom += flow_matrix[i]
                
                ax.set_title("面向與注意力機制間的關係強度")
                ax.set_ylabel("關係強度 (%)")
                ax.legend(title="面向")
            
            # 保存注意力分布圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dist_img_path = os.path.join(attention_dist_dir, f"attention_dist_{chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(dist_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「注意力分布」區域處理完成", 3000)
            self.progress_bar.setValue(60)
            
            # 2. 注意力權重比較可視化
            self.status_message.emit(f"2. 開始處理「注意力權重比較」區域（{weight_chart_type}）...", 3000)
            
            # 創建「注意力權重比較」子目錄
            weight_compare_dir = os.path.join(output_dir, "注意力權重比較")
            os.makedirs(weight_compare_dir, exist_ok=True)
            
            # 生成注意力權重比較數據
            mechanisms = ['相似度注意力', '關鍵詞注意力', '自注意力']
            topics = [f"主題{i+1}" for i in range(5)]
            
            if weight_chart_type == "平行坐標圖":
                # 創建模擬數據
                weight_data = []
                for topic in topics:
                    # 為每個主題創建隨機權重，但確保總和為1
                    weights = np.random.rand(len(mechanisms))
                    weights = weights / weights.sum()
                    
                    row = {'主題': topic}
                    for i, mechanism in enumerate(mechanisms):
                        row[mechanism] = weights[i]
                    
                    weight_data.append(row)
                
                weight_df = pd.DataFrame(weight_data)
                
                # 繪製平行坐標圖
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 獲取顏色映射
                colors = plt.cm.tab10(np.linspace(0, 1, len(topics)))
                
                # 繪製每個主題的線
                for i, topic in enumerate(topics):
                    row = weight_df[weight_df['主題'] == topic]
                    values = [row[mechanism].values[0] for mechanism in mechanisms]
                    ax.plot(mechanisms, values, marker='o', label=topic, color=colors[i])
                
                ax.set_title("各主題中不同注意力機制的權重分布")
                ax.set_ylabel("權重值")
                ax.set_ylim(0, 1)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(title="主題")
                
            elif weight_chart_type == "相關性熱力圖":
                # 創建模擬相關性矩陣
                n_mechanisms = len(mechanisms)
                correlation_matrix = np.random.uniform(-0.5, 1, (n_mechanisms, n_mechanisms))
                np.fill_diagonal(correlation_matrix, 1)  # 對角線為1
                
                # 確保矩陣對稱
                correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
                
                # 創建熱力圖
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                          xticklabels=mechanisms, yticklabels=mechanisms, ax=ax, vmin=-1, vmax=1)
                ax.set_title("注意力機制間的相關性熱力圖")
            
            # 保存權重比較圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weight_img_path = os.path.join(weight_compare_dir, f"attention_weight_{weight_chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(weight_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「注意力權重比較」區域處理完成", 3000)
            self.progress_bar.setValue(100)
            
            # 返回第一張圖片的路徑
            self.status_message.emit(f"✓ 所有注意力機制評估可視化已完成", 5000)
            return dist_img_path
            
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
            # 獲取選項
            chart_type = self.coherence_chart_combo.currentText()
            topic_dist_chart_type = self.topic_dist_chart_combo.currentText()
            
            # 創建示例數據（實際項目中應從模型結果獲取）
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import os
            from wordcloud import WordCloud
            import seaborn as sns
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "主題模型評估")
            os.makedirs(output_dir, exist_ok=True)
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            self.status_message.emit("開始生成主題模型評估可視化...", 3000)
            self.progress_bar.setValue(5)
            
            # 1. 主題連貫性圖表
            self.status_message.emit(f"1. 開始處理「主題連貫性」區域（{chart_type}）...", 3000)
            self.progress_bar.setValue(20)
            
            # 創建「主題連貫性」子目錄
            coherence_dir = os.path.join(output_dir, "主題連貫性")
            os.makedirs(coherence_dir, exist_ok=True)
            
            if chart_type == "條形圖":
                # 模擬不同主題的連貫性分數
                n_topics = 8
                topic_ids = [f"主題{i+1}" for i in range(n_topics)]
                coherence_scores = np.random.uniform(0.3, 0.8, n_topics)
                
                # 創建條形圖
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(topic_ids, coherence_scores, color='skyblue')
                
                # 在條形上添加具體數值
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.2f}', ha='center', va='bottom')
                
                ax.set_ylim(0, max(coherence_scores) + 0.1)
                ax.set_ylabel('連貫性分數')
                ax.set_title('各主題連貫性得分比較')
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
            elif chart_type == "詞雲":
                # 創建模擬的主題關鍵詞數據
                topics_keywords = {
                    "主題1": {"服務": 0.9, "態度": 0.8, "員工": 0.7, "專業": 0.6, "熱情": 0.5, "禮貌": 0.4},
                    "主題2": {"價格": 0.85, "便宜": 0.75, "實惠": 0.65, "優惠": 0.55, "划算": 0.45, "貴": 0.35},
                    "主題3": {"味道": 0.95, "好吃": 0.85, "美味": 0.75, "口感": 0.65, "可口": 0.55, "鮮": 0.45},
                    "主題4": {"環境": 0.9, "整潔": 0.8, "舒適": 0.7, "裝修": 0.6, "安靜": 0.5, "氛圍": 0.4}
                }
                
                # 創建2x2子圖佈局繪製詞雲
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                for i, (topic, keywords) in enumerate(topics_keywords.items()):
                    ax = axes[i]
                    # 創建詞雲
                    wc = WordCloud(
                        background_color='white',
                        width=400,
                        height=300,
                        font_path='simhei.ttf' if os.path.exists('simhei.ttf') else None
                    ).generate_from_frequencies(keywords)
                    
                    # 顯示詞雲
                    ax.imshow(wc, interpolation='bilinear')
                    ax.set_title(topic)
                    ax.axis('off')
                
                fig.suptitle('主題關鍵詞詞雲', fontsize=16)
            
            # 保存主題連貫性圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            coherence_img_path = os.path.join(coherence_dir, f"topic_coherence_{chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(coherence_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「主題連貫性」區域處理完成", 3000)
            self.progress_bar.setValue(60)
            
            # 2. 主題分布圖表
            self.status_message.emit(f"2. 開始處理「主題分布」區域（{topic_dist_chart_type}）...", 3000)
            
            # 創建「主題分布」子目錄
            topic_dist_dir = os.path.join(output_dir, "主題分布")
            os.makedirs(topic_dist_dir, exist_ok=True)
            
            # 生成模擬數據
            n_topics = 5
            n_docs = 4
            topic_names = [f"主題{i+1}" for i in range(n_topics)]
            document_names = [f"文檔集{i+1}" for i in range(n_docs)]
            
            # 創建主題分布數據（確保每個文檔集中主題分布總和為1）
            topic_dist_data = []
            for doc in document_names:
                # 創建隨機分布
                dist = np.random.rand(n_topics)
                dist = dist / dist.sum()  # 正規化
                
                for i, topic in enumerate(topic_names):
                    topic_dist_data.append({
                        '文檔集': doc,
                        '主題': topic,
                        '比例': dist[i]
                    })
            
            # 轉換為DataFrame
            topic_dist_df = pd.DataFrame(topic_dist_data)
            
            if topic_dist_chart_type == "堆疊柱狀圖":
                # 繪製堆疊柱狀圖
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 重塑數據以便繪圖
                pivot_df = topic_dist_df.pivot(index='文檔集', columns='主題', values='比例')
                
                # 繪製堆疊條形圖
                pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')
                
                ax.set_title('各文檔集的主題分布')
                ax.set_ylabel('比例')
                ax.set_ylim(0, 1.0)
                ax.legend(title='主題')
                
            elif topic_dist_chart_type == "交互式氣泡圖":
                # 由於交互式圖表在靜態圖像中無法實現，我們創建一個靜態的氣泡圖作為替代
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 為每個主題創建不同顏色
                colors = plt.cm.tab10(np.linspace(0, 1, len(topic_names)))
                
                # 繪製氣泡圖
                for i, topic in enumerate(topic_names):
                    topic_data = topic_dist_df[topic_dist_df['主題'] == topic]
                    
                    # 計算氣泡大小（比例轉換為點大小）
                    sizes = topic_data['比例'] * 1000
                    
                    # 獲取文檔索引作為x值
                    x_positions = [document_names.index(doc) for doc in topic_data['文檔集']]
                    
                    ax.scatter(x_positions, [i] * len(x_positions), s=sizes, 
                             color=colors[i], alpha=0.7, label=topic)
                    
                    # 添加文本標籤
                    for j, (x, proportion) in enumerate(zip(x_positions, topic_data['比例'])):
                        ax.text(x, i, f"{proportion:.2f}", ha='center', va='center')
                
                ax.set_title('各文檔集的主題分布')
                ax.set_xlabel('文檔集')
                ax.set_ylabel('主題')
                ax.set_xticks(range(len(document_names)))
                ax.set_xticklabels(document_names)
                ax.set_yticks(range(len(topic_names)))
                ax.set_yticklabels(topic_names)
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # 保存主題分布圖表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dist_img_path = os.path.join(topic_dist_dir, f"topic_distribution_{topic_dist_chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(dist_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「主題分布」區域處理完成", 3000)
            self.progress_bar.setValue(100)
            
            # 返回第一張圖片的路徑
            self.status_message.emit(f"✓ 所有主題模型評估可視化已完成", 5000)
            return coherence_img_path
            
        except Exception as e:
            self.status_message.emit(f"生成主題模型評估可視化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成主題模型評估可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
            
    def _generate_comprehensive_viz(self):
        """生成綜合比較視覺化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 獲取選項
            dim_method = self.dim_method_combo.currentText()
            color_by = self.color_by_combo.currentText()
            
            # 創建示例數據（實際項目中應從模型結果獲取）
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import os
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
            
            # 創建對應的中文子目錄
            output_dir = os.path.join(self.output_dir, "綜合比較視覺化")
            os.makedirs(output_dir, exist_ok=True)
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            self.status_message.emit("開始生成綜合比較視覺化...", 3000)
            self.progress_bar.setValue(5)
            
            # 1. 降維可視化
            self.status_message.emit(f"1. 開始處理「降維可視化」區域（{dim_method}，著色依據：{color_by}）...", 3000)
            self.progress_bar.setValue(20)
            
            # 創建「降維可視化」子目錄
            dim_reduction_dir = os.path.join(output_dir, "降維可視化")
            os.makedirs(dim_reduction_dir, exist_ok=True)
            
            # 生成隨機向量數據
            n_samples = 200
            n_features = 20
            n_topics = 5
            
            # 生成隨機高維向量
            vectors = np.random.rand(n_samples, n_features)
            
            # 生成隨機標籤
            if color_by == "主題":
                labels = [f"主題{np.random.randint(1, n_topics+1)}" for _ in range(n_samples)]
            elif color_by == "注意力機制":
                mechanisms = ['相似度注意力', '關鍵詞注意力', '自注意力', '綜合注意力']
                labels = [mechanisms[np.random.randint(0, len(mechanisms))] for _ in range(n_samples)]
            else:  # 聚類結果
                labels = [f"群集{np.random.randint(1, 6)}" for _ in range(n_samples)]
            
            # 降維處理
            if dim_method == "t-SNE":
                reducer = TSNE(n_components=2, perplexity=30, random_state=42)
                reduced_data = reducer.fit_transform(vectors)
                title = "t-SNE降維視覺化"
            elif dim_method == "PCA":
                reducer = PCA(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(vectors)
                title = "PCA降維視覺化"
            elif dim_method == "UMAP":
                # 如果沒有UMAP，退回到PCA
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2, random_state=42)
                    reduced_data = reducer.fit_transform(vectors)
                    title = "UMAP降維視覺化"
                except ImportError:
                    reducer = PCA(n_components=2, random_state=42)
                    reduced_data = reducer.fit_transform(vectors)
                    title = "PCA降維視覺化 (UMAP不可用)"
            elif dim_method == "3D散點圖":
                if dim_method == "t-SNE":
                    reducer = TSNE(n_components=3, perplexity=30, random_state=42)
                else:  # 默認使用PCA
                    reducer = PCA(n_components=3, random_state=42)
                reduced_data = reducer.fit_transform(vectors)
                title = "3D降維視覺化"
            
            # 繪製圖形
            if dim_method == "3D散點圖":
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # 獲取唯一標籤
                unique_labels = list(set(labels))
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = [l == label for l in labels]
                    ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1], reduced_data[mask, 2],
                             c=[colors[i]], label=label, s=50, alpha=0.7)
                
                ax.set_title(title)
                ax.set_xlabel('維度 1')
                ax.set_ylabel('維度 2')
                ax.set_zlabel('維度 3')
                ax.legend()
            else:
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # 獲取唯一標籤
                unique_labels = list(set(labels))
                colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
                
                for i, label in enumerate(unique_labels):
                    mask = [l == label for l in labels]
                    ax.scatter(reduced_data[mask, 0], reduced_data[mask, 1],
                             c=[colors[i]], label=label, s=50, alpha=0.7)
                
                ax.set_title(title)
                ax.set_xlabel('維度 1')
                ax.set_ylabel('維度 2')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
            
            # 保存降維可視化圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dim_img_path = os.path.join(dim_reduction_dir, f"comprehensive_{dim_method}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(dim_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「降維可視化」區域處理完成", 3000)
            self.progress_bar.setValue(60)
            
            # 2. 多指標綜合視圖
            multi_chart_type = self.multi_chart_combo.currentText()
            self.status_message.emit(f"2. 開始處理「多指標綜合視圖」區域（{multi_chart_type}）...", 3000)
            
            # 創建「多指標綜合視圖」子目錄
            multi_metrics_dir = os.path.join(output_dir, "多指標綜合視圖")
            os.makedirs(multi_metrics_dir, exist_ok=True)
            
            # 獲取用戶選擇包含的指標
            include_cohesion = self.cb_include_cohesion.isChecked()
            include_separation = self.cb_include_separation.isChecked()
            include_f1 = self.cb_include_f1.isChecked()
            
            # 統計實際包含的指標數量
            included_metrics = []
            if include_cohesion:
                included_metrics.append("內聚度")
            if include_separation:
                included_metrics.append("分離度")
            if include_f1:
                included_metrics.append("F1分數")
            
            if not included_metrics:
                included_metrics = ["內聚度", "分離度", "F1分數"]  # 如果用戶沒有選擇，默認包含所有指標
            
            # 生成示例指標數據
            mechanisms = ['相似度注意力', '關鍵詞注意力', '自注意力', '綜合注意力']
            metrics_data = {}
            
            for metric in included_metrics:
                # 為每個機制創建隨機分數
                metrics_data[metric] = np.random.uniform(0.5, 0.9, len(mechanisms))
            
            if multi_chart_type == "雷達圖組":
                # 創建2x2子圖佈局繪製雷達圖
                fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
                axes = axes.flatten()
                
                for i, mechanism in enumerate(mechanisms):
                    if i >= len(axes):  # 防止超出子圖數量
                        break
                        
                    ax = axes[i]
                    # 準備數據
                    values = [metrics_data[metric][i] for metric in included_metrics]
                    # 確保閉合
                    values = np.append(values, values[0])
                    
                    # 創建角度均勻分布
                    angles = np.linspace(0, 2*np.pi, len(included_metrics), endpoint=False)
                    angles = np.append(angles, angles[0])  # 閉合
                    
                    # 繪製雷達圖
                    ax.plot(angles, values, 'o-', linewidth=2)
                    ax.fill(angles, values, alpha=0.25)
                    ax.set_title(mechanism)
                    
                    # 設置角度標籤
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(included_metrics)
                    
                    # 設置y軸範圍
                    ax.set_ylim(0, 1)
                
                # 處理剩餘的空子圖
                for i in range(len(mechanisms), len(axes)):
                    axes[i].axis('off')
                
                fig.suptitle('不同注意力機制的多指標評估', fontsize=16)
                
            elif multi_chart_type == "交互式儀表板":
                # 由於交互式儀表板在靜態圖片中無法實現，我們創建一個靜態組合圖表作為替代
                fig, axes = plt.subplots(len(included_metrics), 1, figsize=(12, 4 * len(included_metrics)))
                
                # 如果只有一個指標，確保axes是一個數組
                if len(included_metrics) == 1:
                    axes = [axes]
                
                for i, metric in included_metrics:
                    ax = axes[i]
                    # 繪製條形圖
                    bars = ax.bar(mechanisms, metrics_data[metric], color='skyblue')
                    
                    # 添加數值標籤
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.2f}', ha='center', va='bottom')
                    
                    ax.set_ylim(0, 1.0)
                    ax.set_ylabel(metric)
                    ax.set_title(f'{metric}評估')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                fig.suptitle('多指標綜合評估儀表板', fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # 為標題留出空間
            
            # 保存多指標圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            multi_img_path = os.path.join(multi_metrics_dir, f"comprehensive_multi_{multi_chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(multi_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.status_message.emit(f"✓ 「多指標綜合視圖」區域處理完成", 3000)
            self.progress_bar.setValue(100)
            
            # 返回第一張圖片的路徑
            self.status_message.emit(f"✓ 所有綜合比較視覺化已完成", 5000)
            return dim_img_path
            
        except Exception as e:
            self.status_message.emit(f"生成綜合比較視覺化出錯: {str(e)}", 5000)
            self.progress_bar.setValue(0)
            self.logger.error(f"生成綜合比較視覺化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None