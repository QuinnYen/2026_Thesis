"""
分析頁面模組 - 實現數據載入、預處理和分析的相關功能
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import threading
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTabWidget, QTableView, QHeaderView, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QTextEdit,
    QProgressBar, QSplitter, QScrollArea, QMessageBox, QGridLayout,
    QRadioButton, QButtonGroup, QSlider
)
from PyQt5.QtGui import QFont, QTextCursor, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QSize

# 導入模組
from modules.data_processor import DataProcessor
from modules.bert_embedder import BertEmbedder
from modules.lda_modeler import LDAModeler
from modules.aspect_calculator import AspectCalculator
from modules.evaluator import AttentionEvaluator
from modules.visualizer import Visualizer

# 導入工具類
from utils.logger import get_logger
from utils.file_manager import FileManager

# 獲取logger
logger = get_logger("analysis_tab")

class AnalysisTab(QWidget):
    """分析頁面類，實現數據的載入、分析和處理功能"""
    
    # 定義信號
    status_message = pyqtSignal(str, int)  # 狀態欄訊息信號，參數：訊息, 顯示時間(毫秒)
    progress_updated = pyqtSignal(int, int)  # 進度更新信號，參數：當前值, 最大值
    analysis_completed = pyqtSignal(dict)  # 分析完成信號，參數：結果字典
    
    def __init__(self, config, file_manager):
        """初始化分析頁面
        
        Args:
            config: 配置管理器
            file_manager: 文件管理器
        """
        super().__init__()
        
        # 保存引用
        self.config = config
        self.file_manager = file_manager
        
        # 初始化成員變數
        self.data = None  # 原始數據
        self.processed_data = None  # 預處理後的數據
        self.bert_embeddings = None  # BERT嵌入
        self.lda_model = None  # LDA模型
        self.topics = None  # 主題詞
        self.aspect_vectors = None  # 面向向量
        self.is_processing = False  # 是否正在處理
        self.current_dataset_name = ""  # 當前數據集名稱
        
        # 初始化處理模組
        self._init_modules()
        
        # 創建UI
        self._init_ui()
        
        # 初始化完畢後發送狀態訊息
        self.status_message.emit("分析頁面已準備就緒", 3000)

    def _init_modules(self):
        """初始化處理模組"""
        try:
            # 創建數據處理模組
            self.data_processor = DataProcessor(self.config.get("data_processing"))
            
            # 創建BERT嵌入模組
            self.bert_embedder = BertEmbedder(self.config.get("bert"))
            
            # 創建LDA模型模組
            self.lda_modeler = LDAModeler(self.config.get("lda"))
            
            # 創建面向向量計算模組
            self.aspect_calculator = AspectCalculator(
                self.config.get("attention")
            )
            
            # 創建評估模組
            self.evaluator = AttentionEvaluator(self.config.get("evaluation"))
            
            # 創建可視化模組
            self.visualizer = Visualizer(self.config.get("visualization"))
            
            logger.info("所有分析模組初始化完成")
        except Exception as e:
            logger.error(f"初始化模組時出錯: {str(e)}")
            logger.error(traceback.format_exc())
    
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
        
        # 創建數據視圖區域
        self._create_data_view()
        
        # 創建分析結果視圖區域
        self._create_result_view()
        
        # 添加分割器到主佈局
        main_layout.addWidget(self.content_splitter, 1)  # 1表示拉伸系數
        
        # 創建狀態區
        self._create_status_panel()
        
        # 設置初始狀態
        self._update_ui_state()
    
    def _create_control_panel(self):
        """創建控制面板"""
        control_layout = QHBoxLayout()
        
        # 資料集選擇區域
        dataset_group = QGroupBox("資料集")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # 創建資料集選擇組合框
        dataset_selector_layout = QHBoxLayout()
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(200)
        available_datasets = self._get_available_datasets()
        for dataset in available_datasets:
            self.dataset_combo.addItem(dataset)
        
        dataset_selector_layout.addWidget(QLabel("選擇資料集:"))
        dataset_selector_layout.addWidget(self.dataset_combo, 1)
        
        # 創建載入、刷新按鈕
        load_btn = QPushButton("載入")
        load_btn.clicked.connect(self.load_selected_data)
        
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_dataset_list)
        
        import_btn = QPushButton("導入...")
        import_btn.clicked.connect(self.import_data)
        
        dataset_selector_layout.addWidget(load_btn)
        dataset_selector_layout.addWidget(refresh_btn)
        dataset_selector_layout.addWidget(import_btn)
        dataset_layout.addLayout(dataset_selector_layout)
        
        # 添加數據集信息標籤
        self.dataset_info_label = QLabel("當前資料集: 未載入")
        dataset_layout.addWidget(self.dataset_info_label)
        
        control_layout.addWidget(dataset_group, 2)  # 佔據2/5的空間
        
        # 分析控制區域
        analysis_group = QGroupBox("分析選項")
        analysis_layout = QHBoxLayout(analysis_group)
        
        # LDA參數設置
        lda_options_layout = QVBoxLayout()
        
        # 添加主題數選擇
        topic_layout = QHBoxLayout()
        topic_layout.addWidget(QLabel("主題數:"))
        self.topic_count_spin = QSpinBox()
        self.topic_count_spin.setMinimum(2)
        self.topic_count_spin.setMaximum(100)
        
        # 從配置中獲取 n_topics 的值，提供默認值 10
        lda_config = self.config.get("lda")
        n_topics = 10  # 默認值
        if isinstance(lda_config, dict):
            n_topics = lda_config.get("n_topics", 10)
        self.topic_count_spin.setValue(n_topics)
        
        topic_layout.addWidget(self.topic_count_spin)
        lda_options_layout.addLayout(topic_layout)
        
        # 添加迭代次數選擇
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("迭代:"))
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setMinimum(10)
        self.max_iter_spin.setMaximum(500)
        
        # 從配置中獲取 max_iter 的值，提供默認值 50
        max_iter = 50  # 默認值
        if isinstance(lda_config, dict):
            max_iter = lda_config.get("max_iter", 50)
        self.max_iter_spin.setValue(max_iter)
        iter_layout.addWidget(self.max_iter_spin)
        lda_options_layout.addLayout(iter_layout)
        
        analysis_layout.addLayout(lda_options_layout)
        
        # 注意力機制選項
        attention_options_layout = QVBoxLayout()
        
        self.similarity_attention_check = QCheckBox("相似度注意力")
        self.similarity_attention_check.setChecked(True)
        attention_options_layout.addWidget(self.similarity_attention_check)
        
        self.keyword_attention_check = QCheckBox("關鍵詞注意力")
        self.keyword_attention_check.setChecked(True)
        attention_options_layout.addWidget(self.keyword_attention_check)
        
        self.self_attention_check = QCheckBox("自注意力機制")
        self.self_attention_check.setChecked(True)
        attention_options_layout.addWidget(self.self_attention_check)
        
        analysis_layout.addLayout(attention_options_layout)
        
        # 運行按鈕
        run_options_layout = QVBoxLayout()
        
        self.run_all_btn = QPushButton("運行全部")
        self.run_all_btn.clicked.connect(self.start_analysis)
        run_options_layout.addWidget(self.run_all_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_analysis)
        self.stop_btn.setEnabled(False)
        run_options_layout.addWidget(self.stop_btn)
        
        self.save_results_btn = QPushButton("保存結果")
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)
        run_options_layout.addWidget(self.save_results_btn)
        
        analysis_layout.addLayout(run_options_layout)
        
        control_layout.addWidget(analysis_group, 3)  # 佔據3/5的空間
        
        return control_layout
    
    def _create_data_view(self):
        """創建數據視圖區域"""
        # 數據顯示區
        self.data_widget = QWidget()
        data_layout = QVBoxLayout(self.data_widget)
        
        # 創建表格標籤頁
        self.data_tabs = QTabWidget()
        
        # 原始數據標籤頁
        self.original_data_table = QTableView()
        self.data_tabs.addTab(self.original_data_table, "原始數據")
        
        # 處理後數據標籤頁
        self.processed_data_table = QTableView()
        self.data_tabs.addTab(self.processed_data_table, "預處理數據")
        
        data_layout.addWidget(self.data_tabs)
        
        # 將數據視圖添加到分割器
        self.content_splitter.addWidget(self.data_widget)
    
    def _create_result_view(self):
        """創建分析結果視圖區域"""
        # 結果顯示區
        self.result_widget = QWidget()
        result_layout = QVBoxLayout(self.result_widget)
        
        # 創建結果標籤頁
        self.result_tabs = QTabWidget()
        
        # LDA主題標籤頁
        lda_tab = QWidget()
        lda_layout = QVBoxLayout(lda_tab)
        self.lda_topic_view = QTextEdit()
        self.lda_topic_view.setReadOnly(True)
        lda_layout.addWidget(self.lda_topic_view)
        self.result_tabs.addTab(lda_tab, "LDA主題")
        
        # 面向向量標籤頁
        aspect_tab = QWidget()
        aspect_layout = QVBoxLayout(aspect_tab)
        self.aspect_vector_view = QTextEdit()
        self.aspect_vector_view.setReadOnly(True)
        aspect_layout.addWidget(self.aspect_vector_view)
        self.result_tabs.addTab(aspect_tab, "面向向量")
        
        # 評估結果標籤頁
        eval_tab = QWidget()
        eval_layout = QVBoxLayout(eval_tab)
        self.eval_result_view = QTextEdit()
        self.eval_result_view.setReadOnly(True)
        eval_layout.addWidget(self.eval_result_view)
        self.result_tabs.addTab(eval_tab, "評估結果")
        
        result_layout.addWidget(self.result_tabs)
        
        # 將結果視圖添加到分割器
        self.content_splitter.addWidget(self.result_widget)
    
    def _create_status_panel(self):
        """創建狀態面板"""
        # 狀態區域
        status_layout = QHBoxLayout()
        
        # 進度條
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v/%m (%p%)")
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.progress_bar, 1)  # 1是拉伸因子
        
        # 狀態標籤
        self.status_label = QLabel("就緒")
        status_layout.addWidget(self.status_label)
        
        # 添加狀態區域到主佈局
        self.layout().addLayout(status_layout)
    
    def _get_available_datasets(self):
        """獲取可用的數據集列表"""
        datasets = ["-- 選擇數據集 --"]
        
        try:
            # 從處理後的數據目錄中獲取
            processed_dir = Path(self.config.get("paths", {}).get("data_dir", "./data"))
            if processed_dir.exists():
                for file in processed_dir.glob("*.csv"):
                    datasets.append(file.stem)
        except Exception as e:
            logger.error(f"獲取數據集列表出錯: {str(e)}")
        
        return datasets
    
    def _update_ui_state(self):
        """更新UI狀態"""
        has_data = self.data is not None
        has_results = self.aspect_vectors is not None
        
        # 更新按鈕狀態
        self.run_all_btn.setEnabled(has_data and not self.is_processing)
        self.stop_btn.setEnabled(self.is_processing)
        self.save_results_btn.setEnabled(has_results and not self.is_processing)
        
        # 更新資料集信息
        if has_data:
            rows = len(self.data)
            self.dataset_info_label.setText(f"當前資料集: {self.current_dataset_name} ({rows} 筆資料)")
        else:
            self.dataset_info_label.setText("當前資料集: 未載入")
    
    def _create_data_model(self, data, table_view):
        """創建數據模型並顯示到表格視圖
        
        Args:
            data: pandas DataFrame
            table_view: QTableView實例
        """
        if data is None:
            return
            
        model = QStandardItemModel()
        
        # 設置表頭
        model.setHorizontalHeaderLabels(data.columns)
        
        # 添加數據
        for row in range(min(1000, len(data))):  # 限制最多顯示1000行，避免性能問題
            items = [QStandardItem(str(data.iloc[row, col])) for col in range(len(data.columns))]
            model.appendRow(items)
            
        table_view.setModel(model)
        
        # 調整列寬
        table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        
        # 如果有超過1000行，顯示提示
        if len(data) > 1000:
            model.setHeaderData(0, Qt.Horizontal, f"顯示前1000行 (共 {len(data)} 行)", Qt.ToolTipRole)
    
    def refresh_dataset_list(self):
        """刷新數據集列表"""
        current_text = self.dataset_combo.currentText()
        
        self.dataset_combo.clear()
        available_datasets = self._get_available_datasets()
        for dataset in available_datasets:
            self.dataset_combo.addItem(dataset)
            
        # 嘗試恢復之前選中的項
        index = self.dataset_combo.findText(current_text)
        if index >= 0:
            self.dataset_combo.setCurrentIndex(index)
            
        self.status_message.emit("數據集列表已刷新", 3000)
    
    def import_data(self):
        """導入新數據"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "導入數據檔案",
            "",
            "文本檔案 (*.csv *.txt *.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
            
        try:
            # 讀取文件
            data = pd.read_csv(file_path, encoding='utf-8')
            
            # 提取文件名作為數據集名稱
            file_name = Path(file_path).stem
            
            # 獲取數據目錄 - 使用安全方式
            try:
                # 方法1：直接從配置對象獲取路徑字符串
                if isinstance(self.config, dict):
                    data_dir = self.config.get("paths", {}).get("data_dir", "./data")
                elif hasattr(self.config, "config") and isinstance(self.config.config, dict):
                    data_dir = self.config.config.get("paths", {}).get("data_dir", "./data")
                else:
                    data_dir = "./data"
            except Exception:
                data_dir = "./data"
            
            # 確保目錄存在
            os.makedirs(data_dir, exist_ok=True)
            
            # 保存到數據目錄
            output_path = os.path.join(data_dir, f"{file_name}.csv")
            
            # 確保目錄存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存數據
            data.to_csv(output_path, index=False)
            
            # 刷新數據集列表並選中新導入的數據集
            self.refresh_dataset_list()
            index = self.dataset_combo.findText(file_name)
            if index >= 0:
                self.dataset_combo.setCurrentIndex(index)
                
            # 載入數據
            self.load_data(output_path)
            
            self.status_message.emit(f"已導入數據集 {file_name}", 3000)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            # 使用全局logger而不是self.logger
            logger = get_logger("analysis_tab")
            logger.error(f"導入數據出錯: {str(e)}")
            logger.error(error_details)
            QMessageBox.critical(self, "導入出錯", f"導入數據時出錯:\n{str(e)}")
    
    def load_selected_data(self):
        """載入選中的數據集"""
        dataset_name = self.dataset_combo.currentText()
        
        if dataset_name == "-- 選擇數據集 --" or not dataset_name:
            QMessageBox.warning(self, "選擇數據集", "請選擇一個有效的數據集")
            return
            
        data_path = os.path.join(
            self.config.get("paths", {}).get("data_dir", "./data"),
            f"{dataset_name}.csv"
        )
        
        self.load_data(data_path)
        
    def load_data(self, file_path):
        """載入指定路徑的數據
        
        Args:
            file_path: 數據文件路徑
        """
        try:
            # 重置數據和結果
            self.data = None
            self.processed_data = None
            self.bert_embeddings = None
            self.lda_model = None
            self.topics = None
            self.aspect_vectors = None
            
            # 清除結果視圖
            self.lda_topic_view.clear()
            self.aspect_vector_view.clear()
            self.eval_result_view.clear()
            
            # 讀取數據
            self.data = pd.read_csv(file_path)
            
            # 設置數據集名稱
            self.current_dataset_name = Path(file_path).stem
            
            # 顯示數據
            self._create_data_model(self.data, self.original_data_table)
            
            # 切換到原始數據標籤頁
            self.data_tabs.setCurrentIndex(0)
            
            # 更新UI狀態
            self._update_ui_state()
            
            # 提示信息
            self.status_message.emit(f"已載入數據集 {self.current_dataset_name}", 3000)
            self.status_label.setText(f"已載入: {self.current_dataset_name}")
            
        except Exception as e:
            logger.error(f"載入數據出錯: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "載入出錯", f"載入數據時出錯:\n{str(e)}")
            
            self.data = None
            self._update_ui_state()
    
    def start_analysis(self):
        """開始分析處理"""
        if self.data is None:
            QMessageBox.warning(self, "資料未載入", "請先載入數據後再進行分析")
            return
            
        # 設置處理中狀態
        self.is_processing = True
        self._update_ui_state()
        
        # 獲取分析參數
        params = self._get_analysis_params()
        
        # 創建並啟動分析線程
        self.analysis_thread = AnalysisThread(
            data=self.data,
            params=params,
            modules={
                'data_processor': self.data_processor,
                'bert_embedder': self.bert_embedder,
                'lda_modeler': self.lda_modeler,
                'aspect_calculator': self.aspect_calculator,
                'evaluator': self.evaluator
            }
        )
        
        # 連接信號
        self.analysis_thread.progress_updated.connect(self.update_progress)
        self.analysis_thread.status_updated.connect(self.update_status)
        self.analysis_thread.error_occurred.connect(self.handle_error)
        self.analysis_thread.analysis_completed.connect(self.handle_analysis_completed)
        
        # 啟動線程
        self.analysis_thread.start()
        
        self.status_label.setText("正在分析中...")
        self.progress_bar.setValue(0)
        self.status_message.emit("正在進行數據分析...", 0)  # 0表示不自動隱藏
    
    def _get_analysis_params(self):
        """獲取分析參數"""
        # 收集界面上的參數設置
        params = {
            'n_topics': self.topic_count_spin.value(),
            'max_iter': self.max_iter_spin.value(),
            'dataset_name': self.current_dataset_name,
            'attention_mechanisms': []
        }
        
        # 收集選中的注意力機制
        if self.similarity_attention_check.isChecked():
            params['attention_mechanisms'].append('similarity')
            
        if self.keyword_attention_check.isChecked():
            params['attention_mechanisms'].append('keyword')
            
        if self.self_attention_check.isChecked():
            params['attention_mechanisms'].append('self')
            
        return params
    
    def stop_analysis(self):
        """停止分析處理"""
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            # 標記為需要停止
            self.analysis_thread.should_stop = True
            self.status_label.setText("正在停止...")
            self.status_message.emit("正在停止分析處理...", 0)
    
    def update_progress(self, current, total, step_name=""):
        """更新進度
        
        Args:
            current: 當前進度
            total: 總進度
            step_name: 步驟名稱
        """
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        # 發送進度信號到主窗口
        self.progress_updated.emit(current, total)
        
        if step_name:
            percentage = int(current / total * 100) if total > 0 else 0
            self.status_label.setText(f"{step_name}: {percentage}%")
    
    def update_status(self, message):
        """更新狀態
        
        Args:
            message: 狀態信息
        """
        self.status_label.setText(message)
        self.status_message.emit(message, 0)
    
    def handle_error(self, error_msg):
        """處理錯誤
        
        Args:
            error_msg: 錯誤信息
        """
        self.is_processing = False
        self._update_ui_state()
        
        logger.error(f"分析處理錯誤: {error_msg}")
        QMessageBox.critical(self, "處理錯誤", f"分析過程中發生錯誤:\n{error_msg}")
        
        self.status_label.setText("處理出錯")
        self.status_message.emit("分析處理出錯，請查看日誌", 0)
    
    def handle_analysis_completed(self, results):
        """處理分析完成
        
        Args:
            results: 分析結果字典
        """
        self.is_processing = False
        
        # 保存結果
        self.processed_data = results.get('processed_data')
        self.bert_embeddings = results.get('bert_embeddings')
        self.lda_model = results.get('lda_model')
        self.topics = results.get('topics')
        self.aspect_vectors = results.get('aspect_vectors')
        self.evaluation_results = results.get('evaluation')
        
        # 顯示處理後數據
        self._create_data_model(self.processed_data, self.processed_data_table)
        
        # 顯示LDA主題
        self._display_topics()
        
        # 顯示面向向量
        self._display_aspect_vectors()
        
        # 顯示評估結果
        self._display_evaluation_results()
        
        # 更新UI狀態
        self._update_ui_state()
        
        # 切換到LDA主題標籤頁
        self.result_tabs.setCurrentIndex(0)
        
        # 提示信息
        self.status_label.setText("分析完成")
        self.status_message.emit("數據分析完成", 5000)
        
        # 發送分析完成信號
        self.analysis_completed.emit(results)
    
    def _display_topics(self):
        """顯示LDA主題"""
        if self.topics is None:
            return
            
        self.lda_topic_view.clear()
        
        topics_html = "<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #dddddd; text-align: left; padding: 8px;} th {background-color: #f2f2f2;} tr:nth-child(even) {background-color: #f9f9f9;}</style>"
        topics_html += "<h3>LDA 主題詞</h3>"
        topics_html += "<table><tr><th>主題編號</th><th>關鍵詞</th></tr>"
        
        for topic_id, words in self.topics.items():
            words_str = ", ".join(words)
            topics_html += f"<tr><td>{topic_id}</td><td>{words_str}</td></tr>"
            
        topics_html += "</table>"
        
        self.lda_topic_view.setHtml(topics_html)
    
    def _display_aspect_vectors(self):
        """顯示面向向量"""
        if self.aspect_vectors is None:
            return
            
        self.aspect_vector_view.clear()
        
        vectors_html = "<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #dddddd; text-align: left; padding: 8px;} th {background-color: #f2f2f2;} tr:nth-child(even) {background-color: #f9f9f9;}</style>"
        vectors_html += "<h3>面向向量概述</h3>"
        
        # 檢查 aspect_vectors 是字典還是 NumPy 數組
        if isinstance(self.aspect_vectors, dict):
            # 處理字典格式的 aspect_vectors
            topics = list(self.aspect_vectors.keys())
            vectors_count = len(topics)
            if vectors_count > 0:
                vector_dim = len(next(iter(self.aspect_vectors.values())))
                vectors_html += f"<p>向量維度: ({vectors_count}, {vector_dim})</p>"
                vectors_html += f"<p>向量數量: {vectors_count}</p>"
                
                # 顯示前5個面向向量的範例
                vectors_html += "<h4>前5個面向向量示例:</h4>"
                vectors_html += "<table><tr><th>主題</th><th>向量值(前10維)</th></tr>"
                
                for i, topic in enumerate(topics[:5]):
                    vector = self.aspect_vectors[topic]
                    vector_preview = ", ".join(f"{val:.4f}" for val in vector[:10])
                    if len(vector) > 10:
                        vector_preview += "..."
                    vectors_html += f"<tr><td>{topic}</td><td>{vector_preview}</td></tr>"
            else:
                vectors_html += "<p>未找到面向向量</p>"
        else:
            # 處理 NumPy 數組格式的 aspect_vectors
            shape = self.aspect_vectors.shape
            vectors_html += f"<p>向量維度: {shape}</p>"
            vectors_html += f"<p>向量數量: {shape[0]}</p>"
            
            # 顯示前5個面向向量的範例
            vectors_html += "<h4>前5個面向向量示例:</h4>"
            vectors_html += "<table><tr><th>向量索引</th><th>向量值(前10維)</th></tr>"
            
            for i in range(min(5, shape[0])):
                vector_preview = ", ".join(f"{val:.4f}" for val in self.aspect_vectors[i][:10])
                if shape[1] > 10:
                    vector_preview += "..."
                vectors_html += f"<tr><td>{i}</td><td>{vector_preview}</td></tr>"
            
        vectors_html += "</table>"
        
        self.aspect_vector_view.setHtml(vectors_html)
    
    def _display_evaluation_results(self):
        """顯示評估結果"""
        if not hasattr(self, 'evaluation_results') or self.evaluation_results is None:
            return
            
        self.eval_result_view.clear()
        
        eval_html = "<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid #dddddd; text-align: left; padding: 8px;} th {background-color: #f2f2f2;} tr:nth-child(even) {background-color: #f9f9f9;}</style>"
        eval_html += "<h3>模型評估結果</h3>"
        
        # 顯示主題一致性
        if 'topic_coherence' in self.evaluation_results:
            eval_html += "<h4>主題一致性</h4>"
            eval_html += f"<p>平均一致性分數: {self.evaluation_results['topic_coherence']:.4f}</p>"
        
        # 顯示主題分離度
        if 'topic_separation' in self.evaluation_results:
            eval_html += "<h4>主題分離度</h4>"
            eval_html += f"<p>平均分離度分數: {self.evaluation_results['topic_separation']:.4f}</p>"
        
        # 顯示綜合評分
        if 'combined_score' in self.evaluation_results:
            eval_html += "<h4>綜合評分</h4>"
            eval_html += f"<p>綜合性能分數: {self.evaluation_results['combined_score']:.4f}</p>"
        
        # 顯示詳細指標（如果有）
        if 'details' in self.evaluation_results:
            eval_html += "<h4>詳細指標</h4>"
            eval_html += "<table><tr><th>指標名稱</th><th>數值</th></tr>"
            
            for metric, value in self.evaluation_results['details'].items():
                if isinstance(value, float):
                    eval_html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
                else:
                    eval_html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                    
            eval_html += "</table>"
        
        self.eval_result_view.setHtml(eval_html)
    
    def save_results(self):
        """保存分析結果"""
        if self.aspect_vectors is None:
            QMessageBox.warning(self, "結果未生成", "尚未生成分析結果，請先運行分析")
            return
            
        try:
            # 安全地獲取配置值
            results_dir = './output'  # 默認值
            
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
                            results_dir = "./output"
                    except TypeError:
                        # 如果上述方法失敗，使用默認路徑
                        results_dir = "./output"
                else:
                    # 配置對象不可用，使用默認路徑
                    results_dir = "./output"
            except Exception as config_error:
                logger.warning(f"讀取配置時出現錯誤，使用默認值: {str(config_error)}")
                results_dir = "./output"
            
            # 確保結果目錄存在
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成結果文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file_name = f"result_{self.current_dataset_name}_{timestamp}.json"
            result_file_path = os.path.join(results_dir, result_file_name)
            
            # 準備保存的結果數據
            result_data = {
                "dataset_name": self.current_dataset_name,
                "timestamp": timestamp,
                "parameters": self._get_analysis_params(),
                "topics": self.topics,
                "evaluation": self.evaluation_results,
                "metadata": {
                    "processed_rows": len(self.processed_data) if self.processed_data is not None else 0,
                    "num_topics": len(self.topics) if self.topics else 0,
                    "vector_shape": list(self.aspect_vectors.shape) if self.aspect_vectors is not None else []
                }
            }
            
            # 另存面向向量（單獨存為 NPZ 格式）
            if self.aspect_vectors is not None:
                vectors_file_name = f"vectors_{self.current_dataset_name}_{timestamp}.npz"
                vectors_file_path = os.path.join(results_dir, vectors_file_name)
                np.savez_compressed(vectors_file_path, aspect_vectors=self.aspect_vectors)
                result_data["vectors_file"] = vectors_file_name
            
            # 保存結果數據
            self.file_manager.write_json(result_file_path, result_data)
            
            # 成功提示
            self.status_message.emit(f"結果已保存至 {result_file_path}", 5000)
            QMessageBox.information(self, "保存成功", f"分析結果已成功保存")
            
        except Exception as e:
            logger.error(f"保存結果出錯: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "保存出錯", f"保存分析結果時出錯:\n{str(e)}")
    
    def export_report(self, file_path):
        """導出分析報告
        
        Args:
            file_path: 報告文件保存路徑
        """
        if self.aspect_vectors is None or self.topics is None:
            QMessageBox.warning(self, "結果未生成", "尚未生成分析結果，無法導出報告")
            return
            
        try:
            # 使用可視化模組生成報告
            self.visualizer.export_report(
                file_path=file_path,
                dataset_name=self.current_dataset_name,
                data=self.processed_data,
                topics=self.topics,
                aspect_vectors=self.aspect_vectors,
                evaluation=self.evaluation_results,
                params=self._get_analysis_params()
            )
            
            # 成功提示
            self.status_message.emit(f"報告已導出至 {file_path}", 5000)
            
        except Exception as e:
            logger.error(f"導出報告出錯: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "導出出錯", f"導出分析報告時出錯:\n{str(e)}")
            raise e  # 重新拋出異常，讓調用者知道出錯了
    
    def on_settings_changed(self):
        """處理設定變更"""
        # 重新載入配置到各模組
        self.data_processor.update_config(self.config.get("data_processing"))
        self.bert_embedder.update_config(self.config.get("bert"))
        self.lda_modeler.update_config(self.config.get("lda"))
        self.aspect_calculator.update_config(self.config.get("attention"))
        self.evaluator.update_config(self.config.get("evaluation"))
        self.visualizer.update_config(self.config.get("visualization"))
        
        # 更新界面上的參數設置
        self.topic_count_spin.setValue(self.config.get("lda", {}).get("n_topics", 10))
        self.max_iter_spin.setValue(self.config.get("lda", {}).get("max_iter", 50))
        
        # 注意力機制勾選狀態
        enabled_mechanisms = self.config.get("attention", {}).get("enabled_mechanisms", [])
        self.similarity_attention_check.setChecked("similarity" in enabled_mechanisms)
        self.keyword_attention_check.setChecked("keyword" in enabled_mechanisms)
        self.self_attention_check.setChecked("self" in enabled_mechanisms)


class AnalysisThread(QThread):
    """分析處理線程類"""
    
    progress_updated = pyqtSignal(int, int, str)  # 進度更新信號，參數：當前值, 最大值, 步驟名稱
    status_updated = pyqtSignal(str)  # 狀態更新信號
    error_occurred = pyqtSignal(str)  # 錯誤信號
    analysis_completed = pyqtSignal(dict)  # 分析完成信號，參數：結果字典
    
    def __init__(self, data, params, modules):
        """初始化線程
        
        Args:
            data: 原始數據
            params: 分析參數
            modules: 處理模組字典
        """
        super().__init__()
        self.data = data
        self.params = params
        self.modules = modules
        self.should_stop = False
        self.logger = get_logger("analysis_thread")
    
    def run(self):
        """運行分析處理"""
        try:
            results = {}
            
            # 檢查是否有文本列
            if 'text' not in self.data.columns and 'review' not in self.data.columns:
                raise ValueError("數據中找不到文本列，請確保數據中包含'text'或'review'列")
            
            # 1. 數據預處理
            self.status_updated.emit("正在進行數據預處理...")
            self.logger.info("開始數據預處理")
            
            text_col = 'text' if 'text' in self.data.columns else 'review'
            processor = self.modules['data_processor']
            
            total_rows = len(self.data)
            processed_texts = []
            
            for i, text in enumerate(self.data[text_col]):
                if self.should_stop:
                    return
                    
                processed_text = processor.preprocess(text)
                processed_texts.append(processed_text)
                
                if i % 10 == 0 or i == total_rows - 1:
                    self.progress_updated.emit(i + 1, total_rows, "數據預處理")
            
            # 創建處理後的數據框
            processed_data = self.data.copy()
            processed_data['processed_text'] = processed_texts
            results['processed_data'] = processed_data
            
            # 2. BERT嵌入
            self.status_updated.emit("正在進行BERT編碼...")
            self.logger.info("開始BERT編碼")
            
            embedder = self.modules['bert_embedder']
            
            embeddings = []
            for i, text in enumerate(processed_texts):
                if self.should_stop:
                    return
                    
                embedding = embedder.get_embeddings(text)
                embeddings.append(embedding)
                
                if i % 5 == 0 or i == total_rows - 1:
                    self.progress_updated.emit(i + 1, total_rows, "BERT編碼")
            
            # 將嵌入轉換為numpy數組
            bert_embeddings = np.array(embeddings)
            results['bert_embeddings'] = bert_embeddings
            
            # 3. LDA主題建模
            self.status_updated.emit("正在進行LDA主題建模...")
            self.logger.info("開始LDA主題建模")
            
            modeler = self.modules['lda_modeler']
            
            # 設置主題數和迭代次數
            modeler.n_topics = self.params['n_topics']
            modeler.max_iter = self.params['max_iter']
            
            # 訓練LDA模型
            lda_model, topics = modeler.train(processed_texts, progress_callback=self._lda_progress_callback)
            
            results['lda_model'] = lda_model
            results['topics'] = topics
            
            # 獲取文檔主題分佈 (這部分是新增的，用於獲取doc_main_topics)
            if lda_model is not None:
                # 準備文檔-主題分佈所需的向量化文本
                if hasattr(modeler, 'vectorizer') and hasattr(modeler, 'feature_names'):
                    vectorizer = modeler.vectorizer
                    X = vectorizer.transform(processed_texts)
                    doc_topic_dist = lda_model.transform(X)
                    # 獲取每個文檔的主要主題索引
                    doc_main_topics = np.argmax(doc_topic_dist, axis=1)
                else:
                    # 如果無法獲取向量化器，則將所有文檔分配到主題0
                    self.logger.warning("無法獲取LDA模型的向量化器，將所有文檔分配到主題0")
                    doc_main_topics = np.zeros(len(processed_texts), dtype=int)
            else:
                # LDA模型訓練失敗，將所有文檔分配到主題0
                self.logger.warning("LDA模型訓練失敗，將所有文檔分配到主題0")
                doc_main_topics = np.zeros(len(processed_texts), dtype=int)
            
            # 4. 面向向量計算
            self.status_updated.emit("正在進行面向向量計算...")
            self.logger.info("開始面向向量計算")
            
            calculator = self.modules['aspect_calculator']
            
            # 定義進度回調函數
            def _aspect_progress_callback(message_or_processed, percentage_or_total=None):
                """面向向量計算進度回調
                
                這個回調函數兼容兩種不同的參數格式:
                1. (message, percentage) - 用於來自aspect_calculator的回調
                2. (processed, total) - 用於舊版本的回調格式
                """
                # 檢查是否為(message, percentage)格式
                if isinstance(message_or_processed, str):
                    # 是字符串，表示是message參數
                    message = message_or_processed
                    percentage = percentage_or_total or 0
                    
                    # 將百分比轉換為進度值
                    processed = int(percentage)
                    total = 100
                    
                    # 更新進度條
                    self.progress_updated.emit(processed, total, "面向向量計算")
                    
                    # 更新狀態訊息
                    self.status_updated.emit(message)
                else:
                    # 是數字，表示是processed參數
                    processed = message_or_processed
                    total = percentage_or_total or 100
                    
                    # 更新進度條
                    self.progress_updated.emit(processed, total, "面向向量計算")
                    
                    # 每20個向量更新一次狀態
                    if processed % 20 == 0 or processed == total:
                        self.status_updated.emit(f"計算面向向量: {processed}/{total}")
            
            # 使用新的參數格式調用
            aspect_result = calculator.calculate_aspect_vectors(
                embeddings=bert_embeddings,
                metadata=pd.DataFrame({
                    'main_topic': [f"Topic_{topic_idx+1}" for topic_idx in doc_main_topics],
                    'text': processed_texts
                }),
                topics=topics,
                progress_callback=_aspect_progress_callback
            )
            
            # 從返回結果中獲取面向向量
            if aspect_result and 'aspect_vectors' in aspect_result:
                aspect_vectors = aspect_result['aspect_vectors']
                results['aspect_vectors'] = aspect_vectors
            else:
                self.logger.error("無法獲取面向向量結果")
                if self.should_stop:
                    return
                else:
                    raise ValueError("面向向量計算失敗")
            
            # 5. 評估模型
            self.status_updated.emit("正在評估模型...")
            self.logger.info("開始評估模型")
            
            evaluator = self.modules['evaluator']
            
            # 評估結果
            evaluation = evaluator.evaluate_model(
                topics=topics,
                vectors=aspect_vectors,
                texts=processed_texts
            )
            
            results['evaluation'] = evaluation
            
            # 發送完成信號
            self.status_updated.emit("分析完成")
            self.logger.info("分析處理完成")
            self.analysis_completed.emit(results)
            
        except Exception as e:
            self.logger.error(f"分析處理出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.error_occurred.emit(str(e))
    
    def _lda_progress_callback(self, iteration, total, perplexity=None):
        message = f"LDA迭代: {iteration}/{total}"
        if perplexity is not None:
            if isinstance(perplexity, (float, int)):
                message += f", 困惑度: {perplexity:.2f}"
            else:
                message += f", 困惑度: {perplexity}"
        self.progress_updated.emit(iteration, total, "LDA訓練")
        if iteration % 10 == 0 or iteration == total:
            self.status_updated.emit(message)
    
    def _aspect_progress_callback(self, processed, total):
        """面向向量計算進度回調"""
        self.progress_updated.emit(processed, total, "面向向量計算")
        
        # 每20個向量更新一次狀態
        if processed % 20 == 0 or processed == total:
            self.status_updated.emit(f"計算面向向量: {processed}/{total}")