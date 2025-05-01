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
            self.status_message.emit("正在生成可視化...", 0)
            
            # 決定要生成的可視化類型和保存文件名
            viz_type = ""
            default_name = ""
            
            if current_tab_index == 0:  # 面向向量質量
                viz_type = "vector_quality"
                default_name = "vector_quality"
                img_path = self._generate_vector_quality_viz()
            elif current_tab_index == 1:  # 情感分析性能
                viz_type = "sentiment_analysis"
                default_name = "sentiment_analysis" 
                img_path = self._generate_sentiment_viz()
            elif current_tab_index == 2:  # 注意力機制評估
                viz_type = "attention_evaluation"
                default_name = "attention_evaluation"
                img_path = self._generate_attention_viz()
            elif current_tab_index == 3:  # 主題模型評估
                viz_type = "topic_evaluation"
                default_name = "topic_evaluation"
                img_path = self._generate_topic_viz()
            elif current_tab_index == 4:  # 綜合比較視覺化
                viz_type = "comprehensive"
                default_name = "comprehensive_viz"
                img_path = self._generate_comprehensive_viz()
            else:
                QMessageBox.warning(self, "無效選擇", "請選擇一種可視化類型")
                return
            
            # 如果成功生成，自動保存圖片
            if img_path and os.path.exists(img_path):
                # 創建保存目錄 - 確保在 1_output/exports 目錄下
                exports_dir = os.path.join("Part04_", "1_output", "exports")
                
                # 嘗試從文件管理器獲取正確路徑
                if self.file_manager is not None:
                    if hasattr(self.file_manager, "export_dir"):
                        exports_dir = self.file_manager.export_dir
                        self.logger.debug(f"從文件管理器獲取導出目錄: {exports_dir}")
                    elif hasattr(self.file_manager, "get_path"):
                        try:
                            exports_dir = self.file_manager.get_path("exports")
                            self.logger.debug(f"從文件管理器的get_path獲取導出目錄: {exports_dir}")
                        except:
                            pass
                    # 確保路徑包含 1_output
                    if "1_output" not in exports_dir and os.path.exists(os.path.join("Part04_", "1_output")):
                        exports_dir = os.path.join("Part04_", "1_output", "exports")
                
                # 安全地檢查配置對象並獲取路徑，確保包含 1_output
                if self.config is not None:
                    if isinstance(self.config, dict):
                        paths = self.config.get("paths", {})
                        if "exports_dir" in paths:
                            exports_dir = paths.get("exports_dir")
                            # 確保路徑在 1_output 下
                            if "1_output" not in exports_dir:
                                # 檢查是否有 output_dir 設定
                                output_dir = paths.get("output_dir", os.path.join("Part04_", "1_output"))
                                exports_dir = os.path.join(output_dir, "exports")
                    elif hasattr(self.config, "get"):
                        try:
                            paths = self.config.get("paths", {})
                            if isinstance(paths, dict) and "exports_dir" in paths:
                                exports_dir = paths.get("exports_dir")
                                if "1_output" not in exports_dir:
                                    output_dir = paths.get("output_dir", os.path.join("Part04_", "1_output"))
                                    exports_dir = os.path.join(output_dir, "exports")
                        except Exception as e:
                            self.logger.warning(f"從配置獲取導出目錄時出錯: {str(e)}")
                
                # 最後確認路徑是否包含 1_output，如果不包含則強制設置
                if "1_output" not in exports_dir:
                    # 絕對路徑處理
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
                    exports_dir = os.path.join(project_root, "1_output", "exports")
                    self.logger.debug(f"修正導出目錄到1_output下: {exports_dir}")
                
                # 確保目錄存在
                if not os.path.exists(exports_dir):
                    os.makedirs(exports_dir, exist_ok=True)
                    self.logger.debug(f"創建導出目錄: {exports_dir}")
                
                # 生成目標文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_filename = f"{default_name}_{self.current_dataset}_{timestamp}.png"
                target_path = os.path.join(exports_dir, target_filename)
                
                # 複製檔案
                import shutil
                shutil.copy2(img_path, target_path)
                
                # 提示用戶
                self.status_message.emit(f"可視化已生成並保存至: {target_path}", 5000)
                QMessageBox.information(
                    self, 
                    "生成並保存成功", 
                    f"可視化已成功生成並保存！\n\n保存路徑：\n{target_path}"
                )
            else:
                self.status_message.emit("可視化生成失敗或未生成圖片文件", 3000)
                
        except Exception as e:
            logger.error(f"生成可視化出錯: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "生成出錯", f"生成可視化時出錯:\n{str(e)}")
            
    def _generate_vector_quality_viz(self):
        """生成面向向量質量可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 獲取選項
            chart_type = self.cohesion_chart_combo.currentText()
            
            # 創建示例數據（實際項目中應從模型結果獲取）
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import os
            
            # 確保輸出目錄存在
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # 模擬不同注意力機制的內聚度和分離度數據
            attention_mechanisms = ['相似度注意力', '關鍵詞注意力', '自注意力', '綜合注意力']
            cohesion_values = np.random.uniform(0.6, 0.9, len(attention_mechanisms))
            separation_values = np.random.uniform(0.5, 0.8, len(attention_mechanisms))
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 根據選擇的圖表類型生成不同的圖表
            if chart_type == "條形圖":
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
                
            elif chart_type == "散點圖":
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
                
            elif chart_type == "樹狀圖":
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
            
            # 保存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(output_dir, f"vector_quality_{chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return img_path
            
        except Exception as e:
            self.logger.error(f"生成面向向量質量可視化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
            
    def _generate_sentiment_viz(self):
        """生成情感分析性能指標可視化
        
        Returns:
            str: 圖片路徑，如果失敗則返回None
        """
        try:
            # 獲取選項
            chart_type = self.metrics_chart_combo.currentText()
            
            # 創建示例數據（實際項目中應從模型結果獲取）
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import os
            
            # 確保輸出目錄存在
            output_dir = self.output_dir
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
            
            # 根據選擇的圖表類型生成不同的圖表
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
            
            # 保存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(output_dir, f"sentiment_analysis_{chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return img_path
            
        except Exception as e:
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
            
            # 創建示例數據（實際項目中應從模型結果獲取）
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import os
            import seaborn as sns
            
            # 確保輸出目錄存在
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
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
            
            # 保存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(output_dir, f"attention_{chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return img_path
            
        except Exception as e:
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
            
            # 創建示例數據（實際項目中應從模型結果獲取）
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import os
            from wordcloud import WordCloud
            
            # 確保輸出目錄存在
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
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
            
            # 保存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(output_dir, f"topic_model_{chart_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return img_path
            
        except Exception as e:
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
            
            # 確保輸出目錄存在
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # 設置中文字體
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
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
            
            # 保存圖片
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = os.path.join(output_dir, f"comprehensive_{dim_method}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return img_path
            
        except Exception as e:
            self.logger.error(f"生成綜合比較視覺化出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None