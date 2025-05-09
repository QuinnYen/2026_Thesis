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
import json
from pathlib import Path
import pickle

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTabWidget, QTableView, QHeaderView, QFileDialog,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QTextEdit,
    QProgressBar, QSplitter, QScrollArea, QMessageBox, QGridLayout,
    QRadioButton, QButtonGroup, QSlider, QInputDialog, QApplication
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

        # 確保有日誌記錄器可用
        try:
            from utils.logger import get_logger
            self.logger = get_logger("analysis_tab")
        except Exception:
            import logging
            self.logger = logging.getLogger("analysis_tab")
            self.logger.setLevel(logging.INFO)
            
        # 設置輸出目錄，同時處理config可能無效的情況
        try:
            if hasattr(self.config, 'get'):
                paths_config = self.config.get("paths", {})
                if isinstance(paths_config, dict):
                    self.output_dir = paths_config.get("output_dir", "./1_output")
                else:
                    # 默認輸出目錄
                    app_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
                    self.output_dir = os.path.join(app_dir, "1_output")
            else:
                # 默認輸出目錄
                app_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
                self.output_dir = os.path.join(app_dir, "1_output")
                
            # 確保輸出目錄存在
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"設置輸出目錄時出錯: {str(e)}")
            self.output_dir = "./1_output"  # 使用相對路徑作為最後的備用選項
            os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化成員變數
        self.data = None
        self.processed_data = None
        self.bert_embeddings = None
        self.lda_model = None
        self.topics = None
        self.aspect_vectors = None
        self.evaluation_results = None
        self.is_processing = False
        self.current_dataset_name = ""
        
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
        
        # 創建匯入按鈕
        import_btn = QPushButton("導入CSV/JSON...")
        import_btn.clicked.connect(self.import_data)
        
        yelp_btn = QPushButton("Yelp合併導入")
        yelp_btn.clicked.connect(self.import_yelp_data)
        yelp_btn.setToolTip("合併Yelp的business和review數據文件")
        
        dataset_selector_layout.addWidget(import_btn, 1)  # 使用1作為伸展因子
        dataset_selector_layout.addWidget(yelp_btn)
        dataset_layout.addLayout(dataset_selector_layout)
        
        # 添加數據處理數量輸入框
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("處理數據數量:"))
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setMinimum(100)
        self.sample_size_spin.setMaximum(100000)
        self.sample_size_spin.setSingleStep(100)
        self.sample_size_spin.setValue(50000)
        self.sample_size_spin.setToolTip("設定要處理的數據記錄數量")
        sample_layout.addWidget(self.sample_size_spin)
        dataset_layout.addLayout(sample_layout)
        
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
        datasets = []
        
        try:
            # 從處理後的數據目錄中獲取
            data_dir = "./data"  # 默認值
            
            # 安全地獲取配置
            if self.config is not None:
                if isinstance(self.config, dict):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        data_dir = paths.get("data_dir", "./data")
                elif hasattr(self.config, 'get'):
                    try:
                        paths = self.config.get("paths", {})
                        if isinstance(paths, dict):
                            data_dir = paths.get("data_dir", "./data")
                    except (TypeError, AttributeError):
                        logger.warning("無法從config獲取paths配置，使用默認data_dir")
            
            # 確保目錄存在
            processed_dir = Path(data_dir)
            os.makedirs(processed_dir, exist_ok=True)
            
            # 獲取CSV文件
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
        # 不再修改當前資料集標籤文字，保留原始的 "當前資料集" 標示
        available_datasets = self._get_available_datasets()
        
        # 僅將刷新結果記錄到日誌，不再更新UI
        if available_datasets:
            logger.info(f"可用數據集: {', '.join(available_datasets)}")
        else:
            logger.info("目前沒有可用數據集")
            
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
            # 檢查是否為Yelp數據
            file_name = Path(file_path).stem
            is_yelp = "yelp_business" in file_name or "yelp_review" in file_name
            
            # 根據文件擴展名選擇讀取方式
            file_ext = Path(file_path).suffix.lower()
            
            if is_yelp and file_ext == '.json':
                # 如果是Yelp文件，詢問用戶是否要合併處理
                reply = QMessageBox.question(
                    self, 
                    "Yelp數據",
                    "檢測到Yelp格式數據，是否需要合併business和review文件?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.import_yelp_data()
                    return
            
            # 標準數據文件處理流程
            if file_ext == '.csv':
                # 使用on_bad_lines='skip'參數忽略解析錯誤的行
                data = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            elif file_ext == '.json':
                try:
                    # 嘗試常規JSON讀取
                    data = pd.read_json(file_path)
                except:
                    # 嘗試逐行讀取JSON (JSONL格式)
                    data = pd.read_json(file_path, lines=True)
            elif file_ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                data = pd.DataFrame({'text': lines})
            else:
                raise ValueError(f"不支持的文件類型: {file_ext}")
            
            # 直接載入資料，不再保存副本到 ReviewsDataBase 目錄
            self.data = data
            
            # 設置數據集名稱
            self.current_dataset_name = file_name
            
            # 顯示數據
            self._create_data_model(self.data, self.original_data_table)
            
            # 切換到原始數據標籤頁
            self.data_tabs.setCurrentIndex(0)
            
            # 更新UI狀態
            self._update_ui_state()
            
            # 刷新數據集列表
            self.refresh_dataset_list()
            
            # 顯示成功信息
            self.status_message.emit(f"已導入數據集 {file_name}", 3000)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger = get_logger("analysis_tab")
            logger.error(f"導入數據出錯: {str(e)}")
            logger.error(error_details)
            QMessageBox.critical(self, "導入出錯", f"導入數據時出錯:\n{str(e)}")
    
    def import_yelp_data(self):
        """專用的Yelp數據導入方法"""
        progress = None
        try:
            # 提示用戶選擇business文件
            business_file, _ = QFileDialog.getOpenFileName(
                self,
                "選擇Yelp business檔案",
                "",
                "JSON檔案 (*.json);;所有檔案 (*.*)"
            )
            
            if not business_file:
                return
            
            # 提示用戶選擇review文件
            review_file, _ = QFileDialog.getOpenFileName(
                self,
                "選擇Yelp review檔案",
                str(Path(business_file).parent),  # 使用business文件所在的目錄作為起始目錄
                "JSON檔案 (*.json);;所有檔案 (*.*)"
            )
            
            if not review_file:
                return
            
            # 創建進度對話框
            progress = QMessageBox()
            progress.setWindowTitle("處理中")
            progress.setText("正在處理Yelp數據，這可能需要一些時間...")
            progress.setStandardButtons(QMessageBox.NoButton)
            progress.show()
            QApplication.processEvents()
            
            # 使用YelpProcessor處理數據
            from modules.data_processor import YelpProcessor
            
            # 設置輸出目錄 - 安全地處理config為None的情況
            data_dir = "./data"  # 預設值
            try:
                if self.config is None:
                    pass  # 使用預設值
                elif isinstance(self.config, dict):
                    paths = self.config.get("paths", {})
                    if isinstance(paths, dict):
                        data_dir = paths.get("data_dir", "./data")
                elif hasattr(self.config, 'get'):
                    try:
                        paths = self.config.get("paths", {})
                        if isinstance(paths, dict):
                            data_dir = paths.get("data_dir", "./data")
                    except:
                        pass  # 使用預設值
            except:
                pass  # 使用預設值
                
            # 確保目錄存在
            os.makedirs(data_dir, exist_ok=True)
            
            # 生成輸出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(data_dir, f"yelp_merged_{timestamp}.csv")
            
            # 取得樣本大小
            sample_size = self.sample_size_spin.value()
            
            # 創建處理器並處理數據
            processor = YelpProcessor()
            df = processor.process_yelp_files(
                business_path=business_file,
                review_path=review_file,
                output_path=output_file,
                sample_size=sample_size
            )
            
            # 關閉處理對話框
            if progress:
                progress.close()
                progress = None
            
            # 安全地更新數據集列表
            QApplication.processEvents()  # 確保UI更新
            try:
                self.refresh_dataset_list()
            except Exception as e:
                logger.warning(f"刷新數據集列表失敗: {str(e)}")
            
            # 安全地載入數據
            try:
                self.load_data(output_file)
            except Exception as e:
                logger.warning(f"載入新數據失敗: {str(e)}")
                # 顯示備用消息
                QMessageBox.information(
                    self, 
                    "處理完成", 
                    f"Yelp數據已成功合併處理，共 {len(df) if df is not None else '未知'} 筆資料\n"
                    f"儲存位置: {output_file}"
                )
            
            # 顯示成功消息
            self.status_message.emit(f"已處理並導入Yelp合併數據 ({len(df) if df is not None else '未知'} 筆資料)", 5000)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger = get_logger("analysis_tab")
            logger.error(f"處理Yelp數據出錯: {str(e)}")
            logger.error(error_details)
            
            # 確保進度對話框關閉
            if progress:
                try:
                    progress.close()
                except:
                    pass
                    
            QMessageBox.critical(self, "處理出錯", f"處理Yelp數據時出錯:\n{str(e)}")
    
    def load_selected_data(self):
        """載入選中的數據集"""
        dataset_name = self.dataset_info_label.text()
        
        if not dataset_name:
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
            
            # 讀取數據 - 添加on_bad_lines='skip'參數解決欄位不一致問題
            self.data = pd.read_csv(file_path, on_bad_lines='skip')
            
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
            'attention_mechanisms': [],
            'sample_size': self.sample_size_spin.value()  # 新增樣本大小參數
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
        
        # 導入主題標籤轉換函數
        from utils.settings.topic_labels import convert_topic_key_to_chinese
        
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
                    # 將英文主題標籤轉換為中文格式
                    chinese_topic = convert_topic_key_to_chinese(topic, self.current_dataset_name, self.topic_count_spin.value())
                    
                    vector = self.aspect_vectors[topic]
                    vector_preview = ", ".join(f"{val:.4f}" for val in vector[:10])
                    if len(vector) > 10:
                        vector_preview += "..."
                    vectors_html += f"<tr><td>{chinese_topic}</td><td>{vector_preview}</td></tr>"
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
        try:
            # 檢查是否有結果可以保存
            if self.aspect_vectors is None:
                QMessageBox.warning(self, "保存失敗", "沒有可保存的向量結果")
                return
    
            # 確定文件名
            default_name = "aspect_vectors_result"
            default_path = os.path.join(self.output_dir, f"{default_name}.json")
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存結果", default_path, "JSON文件 (*.json)"
            )
            
            if not file_path:
                return
                
            # 準備要保存的數據
            results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_source": self.current_dataset_name,
            }
            
            # 正確地設置 attention_type (從分析結果中獲取)
            if hasattr(self, 'analysis_thread') and hasattr(self.analysis_thread, 'aspect_result') and self.analysis_thread.aspect_result:
                results["attention_type"] = self.analysis_thread.aspect_result.get('attention_type', "未知類型")
            elif hasattr(self, 'analysis_thread') and hasattr(self.analysis_thread, 'params'):
                # 備用方案：從參數中獲取
                results["attention_type"] = self.analysis_thread.params.get('attention_mechanisms', "未知類型")
            else:
                # 這裡我們嘗試獲取最佳注意力類型，而不是使用「未知類型」
                best_attention_type = None
                if hasattr(self, 'evaluation_results') and self.evaluation_results and 'details' in self.evaluation_results:
                    details = self.evaluation_results['details']
                    if 'best_attention_type' in details:
                        best_attention_type = details['best_attention_type']
                results["attention_type"] = best_attention_type or "未知類型"
            
            # 正確處理不同類型的 aspect_vectors
            if isinstance(self.aspect_vectors, dict):
                # 字典類型的 aspect_vectors
                results["vector_type"] = "dictionary"
                results["vector_shape"] = [len(self.aspect_vectors)]
                if len(self.aspect_vectors) > 0:
                    first_key = next(iter(self.aspect_vectors))
                    first_vector = self.aspect_vectors[first_key]
                    if hasattr(first_vector, "shape"):
                        results["vector_shape"].append(first_vector.shape[0])
                    elif isinstance(first_vector, (list, np.ndarray)):
                        results["vector_shape"].append(len(first_vector))
                        
                # 將向量轉換為可序列化的格式，確保使用中文標籤
                serialized_vectors = {}
                
                # 導入主題標籤轉換函數
                from utils.settings.topic_labels import convert_topic_key_to_chinese
                
                for topic, vector in self.aspect_vectors.items():
                    # 將英文主題標籤轉換為中文格式
                    chinese_topic = convert_topic_key_to_chinese(topic, self.current_dataset_name, self.topic_count_spin.value())
                    
                    if isinstance(vector, np.ndarray):
                        serialized_vectors[chinese_topic] = vector.tolist()
                    else:
                        serialized_vectors[chinese_topic] = vector
                results["aspect_vectors"] = serialized_vectors
            else:
                # NumPy 陣列類型的 aspect_vectors
                results["vector_type"] = "numpy_array"
                results["vector_shape"] = list(self.aspect_vectors.shape) if self.aspect_vectors is not None else []
                
                # 將 NumPy 陣列轉換為列表
                if self.aspect_vectors is not None:
                    results["aspect_vectors"] = self.aspect_vectors.tolist()
                else:
                    results["aspect_vectors"] = []
            
            # 保存評估指標
            if hasattr(self, 'evaluation_results') and self.evaluation_results is not None:
                results["metrics"] = {
                    "coherence": self.evaluation_results.get('topic_coherence', 0.0),
                    "separation": self.evaluation_results.get('topic_separation', 0.0),
                    "combined_score": self.evaluation_results.get('combined_score', 0.0)
                }
                
                # 如果有詳細指標，也一併保存
                if 'details' in self.evaluation_results:
                    results["metrics_details"] = self.evaluation_results['details']
            
            # 保存結果到JSON文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"結果已保存至: {file_path}")
            QMessageBox.information(self, "保存成功", f"結果已保存至:\n{file_path}")
            
        except Exception as e:
            self.logger.error(f"保存結果出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "保存失敗", f"保存結果時出錯:\n{str(e)}")
    
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
        # 確保 self.config 不為 None
        if self.config is None:
            self.logger.warning("配置對象為空，無法更新模組配置")
            return
            
        # 安全地更新各模組配置
        try:
            # 重新載入配置到各模組
            self.data_processor.update_config(self.config.get("data_processing", {}))
            self.bert_embedder.update_config(self.config.get("bert", {}))
            self.lda_modeler.update_config(self.config.get("lda", {}))
            self.aspect_calculator.update_config(self.config.get("attention", {}))
            self.evaluator.update_config(self.config.get("evaluation", {}))
            self.visualizer.update_config(self.config.get("visualization", {}))
            
            # 更新界面上的參數設置
            lda_config = self.config.get("lda", {})
            if isinstance(lda_config, dict):
                self.topic_count_spin.setValue(lda_config.get("n_topics", 10))
                self.max_iter_spin.setValue(lda_config.get("max_iter", 50))
            else:
                self.logger.warning("LDA配置無效，使用默認值")
                self.topic_count_spin.setValue(10)
                self.max_iter_spin.setValue(50)
            
            # 注意力機制勾選狀態
            attention_config = self.config.get("attention", {})
            if isinstance(attention_config, dict):
                enabled_mechanisms = attention_config.get("enabled_mechanisms", [])
                self.similarity_attention_check.setChecked("similarity" in enabled_mechanisms)
                self.keyword_attention_check.setChecked("keyword" in enabled_mechanisms)
                self.self_attention_check.setChecked("self" in enabled_mechanisms)
            else:
                self.logger.warning("注意力機制配置無效，使用默認值")
                self.similarity_attention_check.setChecked(True)
                self.keyword_attention_check.setChecked(True)
                self.self_attention_check.setChecked(True)
        except Exception as e:
            self.logger.error(f"更新配置時發生錯誤: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())


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
        
        # 生成當前時間戳
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 設置輸出基礎目錄路徑 - 使用相對路徑，確保使用 1_output
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_base_dir = os.path.join("Part04_", "1_output", f"run_{current_timestamp}")
        
        # 定義各階段的輸出子目錄路徑 - 僅定義路徑，不創建目錄
        self.output_dirs = {
            'processed_data': os.path.join(self.output_base_dir, "01_processed_data"),
            'bert_embeddings': os.path.join(self.output_base_dir, "02_bert_embeddings"),
            'lda_topics': os.path.join(self.output_base_dir, "03_lda_topics"),
            'aspect_vectors': os.path.join(self.output_base_dir, "04_aspect_vectors"),
            'evaluation': os.path.join(self.output_base_dir, "05_evaluation")
        }
        
        # 確保基礎輸出目錄存在
        os.makedirs(self.output_base_dir, exist_ok=True)
    
    def run(self):
        """運行分析處理"""
        try:
            results = {}
            
            # 檢查是否有文本列 - 擴展支援的文本列名稱
            common_text_columns = [
                'text', 'review', 'review_text', 'reviewText', 
                'content', 'comment', 'description',
                'Review', 'TEXT', 'REVIEW', 'Content'
            ]
            
            # 尋找可能的文本列
            found_text_col = None
            for col in common_text_columns:
                if col in self.data.columns:
                    found_text_col = col
                    break
                    
            # 如果沒找到常見文本列，則嘗試查找包含關鍵字的列名
            if found_text_col is None:
                text_keywords = ['text', 'review', 'comment', 'content', 'description']
                for col in self.data.columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in text_keywords):
                        found_text_col = col
                        break
                        
            # 如果還是找不到，嘗試檢查 Amazon 格式的數據
            if found_text_col is None and 'asin' in self.data.columns and 'reviewText' in self.data.columns:
                found_text_col = 'reviewText'
                
            # 如果仍然無法找到文本列，則報錯
            if found_text_col is None:
                raise ValueError("數據中找不到文本列，請確保數據中包含'text'或'review'列")
                
            self.logger.info(f"使用文本列: {found_text_col}")
            
            # 準備文件名基礎部分（使用數據集名稱和時間戳）
            dataset_name = self.params.get('dataset_name', 'dataset')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_base_name = f"{dataset_name}_{timestamp}"
            
            # 應用樣本大小限制 - 檢查參數中是否有樣本大小設定
            sample_size = self.params.get('sample_size', 100000)  # 默認值為100000
            if len(self.data) > sample_size:
                self.logger.info(f"數據總量 {len(self.data)} 筆，根據設定將限制處理量為 {sample_size} 筆")
                # 隨機抽樣，使用固定的隨機種子以確保結果可重現
                self.data = self.data.sample(n=sample_size, random_state=42)
                self.logger.info(f"已隨機抽樣 {sample_size} 筆數據進行處理")
            
            # 1. 數據預處理
            self.status_updated.emit("正在進行數據預處理...")
            self.logger.info("開始數據預處理")
            
            text_col = found_text_col
            processor = self.modules['data_processor']
            
            total_rows = len(self.data)
            processed_texts = []
            
            for i, text in enumerate(self.data[text_col]):
                if self.should_stop:
                    return
                    
                # 處理可能的 NaN 值和浮點數
                if pd.isna(text):
                    # 如果是 NaN 值，直接使用空字串
                    processed_text = ""
                else:
                    try:
                        # 將文本轉換為字串並清理
                        text_str = str(text)
                        # 使用清洗文本和標記化函數，避免調用整個 preprocess
                        clean_text = processor._clean_text(text_str)
                        tokens = processor._tokenize_and_lemmatize(clean_text)
                        processed_text = " ".join(tokens) if isinstance(tokens, list) else ""
                    except Exception as e:
                        self.logger.warning(f"處理文本時出錯: {str(e)}, 行號: {i}")
                        processed_text = ""  # 如果處理出錯，使用空字串
                
                processed_texts.append(processed_text)
                
                if i % 10 == 0 or i == total_rows - 1:
                    self.progress_updated.emit(i + 1, total_rows, "數據預處理")
            
            # 創建處理後的數據框
            processed_data = self.data.copy()
            processed_data['processed_text'] = processed_texts
            results['processed_data'] = processed_data
            
            # 保存預處理結果到CSV
            processed_data_path = os.path.join(self.output_dirs['processed_data'], f"{file_base_name}_processed.csv")
            # 確保保存目錄存在
            os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
            processed_data.to_csv(processed_data_path, index=False)
            self.logger.info(f"預處理數據已保存至: {processed_data_path}")
            self.status_updated.emit(f"預處理數據已保存至: {os.path.basename(processed_data_path)}")
            
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
            
            # 保存BERT嵌入結果到NPY文件
            embeddings_path = os.path.join(self.output_dirs['bert_embeddings'], f"{file_base_name}_embeddings.npy")
            # 確保保存目錄存在
            os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
            np.save(embeddings_path, bert_embeddings)
            self.logger.info(f"BERT嵌入數據已保存至: {embeddings_path}")
            self.status_updated.emit(f"BERT嵌入數據已保存至: {os.path.basename(embeddings_path)}")
            
            # 同時保存一份元數據索引
            embeddings_meta = pd.DataFrame({
                'index': range(len(bert_embeddings)),
                'text_col': processed_data[text_col].tolist() if text_col in processed_data.columns else [''] * len(bert_embeddings),
                'processed_text': processed_texts
            })
            embeddings_meta_path = os.path.join(self.output_dirs['bert_embeddings'], f"{file_base_name}_embeddings_meta.csv")
            embeddings_meta.to_csv(embeddings_meta_path, index=False)
            
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
            
            # 保存LDA主題結果
            topics_path = os.path.join(self.output_dirs['lda_topics'], f"{file_base_name}_topics.json")
            # 確保保存目錄存在
            os.makedirs(os.path.dirname(topics_path), exist_ok=True)
            with open(topics_path, 'w', encoding='utf-8') as f:
                # 將topics轉換為可JSON序列化的格式
                serializable_topics = {str(k): v for k, v in topics.items()}
                json.dump(serializable_topics, f, ensure_ascii=False, indent=2)
            self.logger.info(f"LDA主題已保存至: {topics_path}")
            self.status_updated.emit(f"LDA主題已保存至: {os.path.basename(topics_path)}")
            
            # 如果模型對象可以被pickle，也保存模型對象
            try:
                model_path = os.path.join(self.output_dirs['lda_topics'], f"{file_base_name}_lda_model.pkl")
                # 目錄已在上面確保存在
                with open(model_path, 'wb') as f:
                    pickle.dump(lda_model, f)
                self.logger.info(f"LDA模型已保存至: {model_path}")
            except Exception as e:
                self.logger.warning(f"無法保存LDA模型對象: {str(e)}")
            
            # 獲取文檔主題分佈 (這部分是新增的，用於獲取doc_main_topics)
            if lda_model is not None:
                # 準備文檔-主題分佈所需的向量化文本
                if hasattr(modeler, 'vectorizer') and hasattr(modeler, 'feature_names'):
                    vectorizer = modeler.vectorizer
                    X = vectorizer.transform(processed_texts)
                    doc_topic_dist = lda_model.transform(X)
                    # 獲取每個文檔的主要主題索引
                    doc_main_topics = np.argmax(doc_topic_dist, axis=1)
                    
                    # 保存文檔-主題分佈
                    doc_topics_path = os.path.join(self.output_dirs['lda_topics'], f"{file_base_name}_doc_topics.csv")
                    doc_topics_df = pd.DataFrame({
                        'doc_index': range(len(doc_topic_dist)),
                        'main_topic': [f"Topic_{i+1}" for i in doc_main_topics]
                    })
                    # 添加每個主題的分佈概率
                    for i in range(doc_topic_dist.shape[1]):
                        doc_topics_df[f'topic_{i+1}_prob'] = doc_topic_dist[:, i]
                    
                    doc_topics_df.to_csv(doc_topics_path, index=False)
                    self.logger.info(f"文檔-主題分佈已保存至: {doc_topics_path}")
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
            
            # 保存aspect_result（包含面向向量和其他結果）
            self.aspect_result = aspect_result  # 保存引用以供later_use
            
            # 從返回結果中獲取面向向量
            if aspect_result and 'aspect_vectors' in aspect_result:
                aspect_vectors = aspect_result['aspect_vectors']
                results['aspect_vectors'] = aspect_vectors
                
                # 保存面向向量結果
                vectors_path = os.path.join(self.output_dirs['aspect_vectors'], f"{file_base_name}_aspect_vectors.json")
                
                # 確保目錄存在
                os.makedirs(os.path.dirname(vectors_path), exist_ok=True)
                
                # 準備可序列化的數據
                serializable_vectors = {}
                if isinstance(aspect_vectors, dict):
                    # 如果是字典類型，將每個向量轉為列表
                    for topic, vector in aspect_vectors.items():
                        if isinstance(vector, np.ndarray):
                            serializable_vectors[str(topic)] = vector.tolist()
                        else:
                            serializable_vectors[str(topic)] = vector
                else:
                    # 如果是numpy數組，直接轉為列表
                    serializable_vectors = {
                        'format': 'numpy_array',
                        'shape': list(aspect_vectors.shape) if hasattr(aspect_vectors, 'shape') else [],
                        'data': aspect_vectors.tolist() if hasattr(aspect_vectors, 'tolist') else []
                    }
                
                with open(vectors_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_vectors, f, ensure_ascii=False, indent=2)
                    
                self.logger.info(f"面向向量已保存至: {vectors_path}")
                self.status_updated.emit(f"面向向量已保存至: {os.path.basename(vectors_path)}")
                
                # 保存完整的aspect_result，包含所有相關信息
                result_path = os.path.join(self.output_dirs['aspect_vectors'], f"{file_base_name}_aspect_result.json")
                
                # 準備可序列化的完整結果 - 改進版本，確保所有嵌套的numpy數組都被轉換
                def convert_to_serializable(obj):
                    """遞迴轉換所有numpy數組為列表"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list) or isinstance(obj, tuple):
                        return [convert_to_serializable(item) for item in obj]
                    else:
                        return obj
                
                # 使用遞迴轉換函數處理完整結果
                serializable_result = convert_to_serializable(aspect_result)
                
                # 保存為JSON
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_result, f, ensure_ascii=False, indent=2)
                    
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
            
            # 保存評估結果
            eval_path = os.path.join(self.output_dirs['evaluation'], f"{file_base_name}_evaluation.json")
            
            # 確保保存目錄存在
            os.makedirs(os.path.dirname(eval_path), exist_ok=True)
            
            with open(eval_path, 'w', encoding='utf-8') as f:
                # 將評估結果轉換為可JSON序列化的格式
                serializable_eval = {}
                for key, value in evaluation.items():
                    if isinstance(value, np.ndarray):
                        serializable_eval[key] = value.tolist()
                    elif isinstance(value, dict):
                        serializable_eval[key] = {}
                        for k, v in value.items():
                            if isinstance(v, np.ndarray):
                                serializable_eval[key][k] = v.tolist()
                            else:
                                serializable_eval[key][k] = v
                    else:
                        serializable_eval[key] = value
                        
                json.dump(serializable_eval, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"評估結果已保存至: {eval_path}")
            self.status_updated.emit(f"評估結果已保存至: {os.path.basename(eval_path)}")
            
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