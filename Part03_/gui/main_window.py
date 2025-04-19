"""
主視窗模組
此模組負責應用程式的主視窗邏輯
"""

import os
import sys
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
import datetime
import json
import time
import traceback
import threading
import webbrowser
import platform

# 導入PyQt6模組
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QSpinBox, QLineEdit, QTextEdit, QFileDialog,
    QMessageBox, QProgressBar, QGroupBox, QTableWidget, QTableWidgetItem, QSplitter,
    QInputDialog, QFrame, QCheckBox, QRadioButton, QDialog, QListWidget
)
from PyQt6.QtCore import Qt, QSize, QUrl, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QIcon, QFont, QPixmap, QDesktopServices

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 導入相關模組
from Part03_.utils.config_manager import ConfigManager
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.result_manager import ResultManager
from Part03_.experiments.pipeline import Pipeline
from Part03_.gui.data_processing import DataProcessingTab
from Part03_.gui.visualization import VisualizationTab
from Part03_.core.aspect_calculator import AspectCalculator

class MainWindow(QMainWindow):
    """
    應用程式主視窗類
    """
    
    def __init__(self):
        """初始化主視窗"""
        super().__init__()
        
        # 設置視窗基本屬性
        self.setWindowTitle("面向分析系統")
        self.setMinimumSize(1200, 800)
        
        # 初始化組件
        self.config = ConfigManager()
        # 從配置中獲取結果目錄路徑
        results_dir = self.config.get('data_settings.output_directory', './Part03_/results')
        self.result_manager = ResultManager(base_dir=results_dir)
        self.pipeline = Pipeline(config=self.config)
        
        # 設置日誌
        log_dir = self.config.get('logging.log_dir', './Part03_/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'app.log')
        
        self.logger = logging.getLogger('main_app')
        self.logger.setLevel(logging.INFO)
        
        # 移除所有處理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 添加文件處理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # 結果集緩存
        self.result_sets = {}
        self.current_result_id = None
        
        # 初始化UI
        self.init_ui()
        self.load_result_sets()
        
        self.logger.info("應用程式啟動")
    
    def init_ui(self):
        """初始化使用者界面"""
        # 創建中央小部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主佈局
        main_layout = QVBoxLayout(central_widget)
        
        # 創建頂部信息面板
        info_panel = QFrame()
        info_panel.setFrameShape(QFrame.Shape.StyledPanel)
        info_panel.setFrameShadow(QFrame.Shadow.Raised)
        info_panel_layout = QHBoxLayout(info_panel)
        
        # 應用標題和狀態
        title_label = QLabel("面向分析系統")
        title_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        status_label = QLabel("就緒")
        status_label.setFont(QFont("Arial", 10))
        self.status_label = status_label
        
        # 當前結果選擇器
        result_label = QLabel("當前結果集:")
        self.result_selector = QComboBox()
        self.result_selector.setMinimumWidth(300)
        self.result_selector.currentIndexChanged.connect(self.on_result_selected)
        
        # 頂部按鈕
        refresh_button = QPushButton("刷新結果列表")
        refresh_button.clicked.connect(self.load_result_sets)
        
        new_analysis_button = QPushButton("執行新分析")
        new_analysis_button.clicked.connect(self.start_new_analysis)
        
        # 添加頂部控件到面板
        info_panel_layout.addWidget(title_label)
        info_panel_layout.addStretch()
        info_panel_layout.addWidget(result_label)
        info_panel_layout.addWidget(self.result_selector)
        info_panel_layout.addWidget(refresh_button)
        info_panel_layout.addWidget(new_analysis_button)
        info_panel_layout.addStretch()
        info_panel_layout.addWidget(status_label)
        
        # 創建標籤頁
        self.tab_widget = QTabWidget()
        
        # 數據處理標籤頁
        self.data_processing_tab = DataProcessingTab(self.config, self.result_manager)
        self.tab_widget.addTab(self.data_processing_tab, "數據處理")
        
        # 視覺化標籤頁
        self.visualization_tab = VisualizationTab(self.config, self.result_manager)
        self.tab_widget.addTab(self.visualization_tab, "視覺化")
        
        # 面向組合標籤頁
        aspect_combination_tab = QWidget()
        self.tab_widget.addTab(aspect_combination_tab, "面向組合分析")
        self.setup_aspect_combination_tab(aspect_combination_tab)
        
        # 結果管理標籤頁
        results_tab = QWidget()
        self.tab_widget.addTab(results_tab, "結果管理")
        self.setup_results_tab(results_tab)
        
        # 模型比較標籤頁
        compare_tab = QWidget()
        self.tab_widget.addTab(compare_tab, "模型比較")
        self.setup_compare_tab(compare_tab)
        
        # 設置標籤頁
        settings_tab = QWidget()
        self.tab_widget.addTab(settings_tab, "設置")
        self.setup_settings_tab(settings_tab)
        
        # 將組件添加到主佈局
        main_layout.addWidget(info_panel)
        main_layout.addWidget(self.tab_widget, 1)
        
        # 創建狀態欄
        self.statusBar().showMessage("就緒")
    
    def setup_aspect_combination_tab(self, tab):
        """設置面向組合分析標籤頁"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        # 引入面向計算器
        from Part03_.core.aspect_calculator import AspectCalculator
        
        # 主佈局
        tab_layout = QVBoxLayout(tab)
        
        # 頂部控制區
        control_frame = QFrame()
        control_frame.setFrameShape(QFrame.Shape.StyledPanel)
        control_layout = QVBoxLayout(control_frame)
        
        # 第一行：選擇面向並添加
        aspect_select_layout = QHBoxLayout()
        aspect_select_layout.addWidget(QLabel("可用面向:"))
        
        # 可用面向選擇器
        self.available_aspects_combo = QComboBox()
        aspect_select_layout.addWidget(self.available_aspects_combo)
        
        # 添加到組合按鈕
        add_aspect_btn = QPushButton("添加到組合")
        add_aspect_btn.clicked.connect(self.add_aspect_to_combination)
        aspect_select_layout.addWidget(add_aspect_btn)
        
        # 添加間隔
        aspect_select_layout.addStretch(1)
        
        # 注意力機制選擇
        aspect_select_layout.addWidget(QLabel("組合方法:"))
        self.combination_method_combo = QComboBox()
        self.combination_method_combo.addItems([
            "注意力加權 (自注意力)",
            "多頭注意力",
            "簡單平均"
        ])
        aspect_select_layout.addWidget(self.combination_method_combo)
        
        control_layout.addLayout(aspect_select_layout)
        
        # 第二行：當前組合和按鈕
        combination_layout = QHBoxLayout()
        combination_layout.addWidget(QLabel("當前組合:"))
        
        # 當前組合顯示
        self.current_combination_label = QLabel("")
        self.current_combination_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px;")
        self.current_combination_label.setMinimumWidth(300)
        combination_layout.addWidget(self.current_combination_label)
        
        # 清空組合按鈕
        clear_btn = QPushButton("清空組合")
        clear_btn.clicked.connect(self.clear_combination)
        combination_layout.addWidget(clear_btn)
        
        # 分析組合按鈕
        analyze_btn = QPushButton("分析組合")
        analyze_btn.clicked.connect(self.analyze_current_combination)
        combination_layout.addWidget(analyze_btn)
        
        # 添加到比較按鈕
        add_to_compare_btn = QPushButton("添加到比較")
        add_to_compare_btn.clicked.connect(self.add_combination_to_comparison)
        combination_layout.addWidget(add_to_compare_btn)
        
        control_layout.addLayout(combination_layout)
        
        # 第三行：組合比較區域
        comparison_layout = QHBoxLayout()
        comparison_layout.addWidget(QLabel("組合比較:"))
        
        # 組合列表
        self.combinations_list = QListWidget()
        self.combinations_list.setMinimumHeight(100)
        comparison_layout.addWidget(self.combinations_list, 2)
        
        # 比較操作按鈕
        comparison_buttons_layout = QVBoxLayout()
        
        remove_btn = QPushButton("移除組合")
        remove_btn.clicked.connect(self.remove_selected_combination)
        comparison_buttons_layout.addWidget(remove_btn)
        
        compare_btn = QPushButton("比較組合")
        compare_btn.clicked.connect(self.compare_combinations)
        comparison_buttons_layout.addWidget(compare_btn)
        
        clear_all_btn = QPushButton("清空所有")
        clear_all_btn.clicked.connect(self.clear_all_combinations)
        comparison_buttons_layout.addWidget(clear_all_btn)
        
        comparison_layout.addLayout(comparison_buttons_layout)
        control_layout.addLayout(comparison_layout)
        
        # 添加控制區
        tab_layout.addWidget(control_frame)
        
        # 創建分隔器以分隔文本分析和可視化區域
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左側：文本分析區
        text_analysis_group = QGroupBox("文本面向分析")
        text_layout = QVBoxLayout(text_analysis_group)
        
        # 輸入文本
        text_layout.addWidget(QLabel("輸入要分析的文本:"))
        self.analysis_text_edit = QTextEdit()
        text_layout.addWidget(self.analysis_text_edit)
        
        # 分析按鈕
        analyze_text_btn = QPushButton("分析文本")
        analyze_text_btn.clicked.connect(self.analyze_text_with_combinations)
        text_layout.addWidget(analyze_text_btn)
        
        # 分析結果
        text_layout.addWidget(QLabel("分析結果:"))
        self.analysis_result_text = QTextEdit()
        self.analysis_result_text.setReadOnly(True)
        text_layout.addWidget(self.analysis_result_text)
        
        splitter.addWidget(text_analysis_group)
        
        # 右側：可視化區
        vis_group = QGroupBox("組合可視化")
        vis_layout = QVBoxLayout(vis_group)
        
        # 創建matplotlib圖形區域
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        vis_layout.addWidget(self.canvas)
        
        # 可視化控制區
        vis_controls = QHBoxLayout()
        
        # 可視化類型選擇
        vis_controls.addWidget(QLabel("可視化類型:"))
        self.vis_type_combo = QComboBox()
        self.vis_type_combo.addItems([
            "相似度矩陣",
            "組合向量分布",
            "關鍵詞雲"
        ])
        vis_controls.addWidget(self.vis_type_combo)
        
        # 更新可視化按鈕
        update_vis_btn = QPushButton("更新可視化")
        update_vis_btn.clicked.connect(self.update_visualization)
        vis_controls.addWidget(update_vis_btn)
        
        vis_layout.addLayout(vis_controls)
        
        # 結果說明文本
        self.vis_description = QTextEdit()
        self.vis_description.setReadOnly(True)
        self.vis_description.setMaximumHeight(100)
        vis_layout.addWidget(self.vis_description)
        
        splitter.addWidget(vis_group)
        
        # 設置分隔器比例
        splitter.setSizes([400, 400])
        
        # 添加分隔器到主佈局
        tab_layout.addWidget(splitter, 1)
        
        # 初始化數據
        self.current_aspects = []  # 當前選中的面向ID
        self.combinations_to_compare = []  # 待比較的面向組合列表
        self.aspect_calculator = None  # 面向計算器實例
        self.aspect_vectors = {}  # 面向向量數據
        self.aspect_labels = {}  # 面向標籤數據
        
        # 記錄當前結果已加載狀態
        self.combinations_result_loaded = False
    
    def load_aspect_combinations_tab(self, result_id):
        """為面向組合標籤頁加載特定結果數據"""
        if not result_id:
            return
        
        try:
            # 初始化面向計算器
            if self.aspect_calculator is None:
                self.aspect_calculator = AspectCalculator(config=self.config)
            
            # 獲取結果集詳細信息
            result_set = self.result_manager.get_result_set(result_id)
            if not result_set:
                return
            
            # 獲取面向向量文件路徑
            aspect_file = None
            if 'files' in result_set and 'aspect_vectors' in result_set['files']:
                aspect_file = result_set['files']['aspect_vectors']
            
            if not aspect_file or not os.path.exists(aspect_file):
                self.aspect_vectors = {}
                self.aspect_labels = {}
                self.available_aspects_combo.clear()
                self.current_combination_label.setText("未找到面向向量文件")
                return
            
            # 載入面向向量
            aspect_result = self.aspect_calculator.load_aspect_vectors(aspect_file)
            
            # 獲取面向數據
            self.aspect_vectors = aspect_result.get('aspect_vectors', {})
            self.aspect_labels = aspect_result.get('aspect_labels', {})
            
            # 更新面向下拉列表
            self.available_aspects_combo.clear()
            for aspect_id, info in self.aspect_vectors.items():
                label = f"{aspect_id}: {info['label']} ({', '.join(info['keywords'][:3])}...)"
                self.available_aspects_combo.addItem(label, aspect_id)
            
            # 清空當前選擇
            self.current_aspects = []
            self.update_combination_label()
            
            # 更新已加載狀態
            self.combinations_result_loaded = True
            
        except Exception as e:
            import traceback
            error_msg = f"載入面向向量時出錯: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "錯誤", f"載入面向向量時出錯: {str(e)}")
            self.combinations_result_loaded = False
    
    def setup_results_tab(self, tab):
        """設置結果管理標籤頁"""
        tab_layout = QVBoxLayout(tab)
        
        # 結果列表
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["ID", "數據集", "創建時間", "狀態", "描述", "操作"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        # 按鈕面板
        buttons_panel = QFrame()
        buttons_layout = QHBoxLayout(buttons_panel)
        
        view_report_button = QPushButton("查看報告")
        view_report_button.clicked.connect(self.view_selected_report)
        
        delete_result_button = QPushButton("刪除結果")
        delete_result_button.clicked.connect(self.delete_selected_result)
        
        export_result_button = QPushButton("導出結果")
        export_result_button.clicked.connect(self.export_selected_result)
        
        compare_button = QPushButton("比較所選結果")
        compare_button.clicked.connect(self.compare_selected_results)
        
        # 添加按鈕到面板
        buttons_layout.addWidget(view_report_button)
        buttons_layout.addWidget(export_result_button)
        buttons_layout.addWidget(delete_result_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(compare_button)
        
        # 添加組件到佈局
        tab_layout.addWidget(self.results_table)
        tab_layout.addWidget(buttons_panel)
    
    def setup_compare_tab(self, tab):
        """設置比較標籤頁"""
        tab_layout = QVBoxLayout(tab)
        
        # 比較配置區域
        config_group = QGroupBox("比較配置")
        config_layout = QVBoxLayout(config_group)
        
        # 選擇要比較的結果
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("選擇要比較的結果:"))
        
        # 第一個結果選擇器
        self.compare_selector1 = QComboBox()
        selection_layout.addWidget(self.compare_selector1)
        
        # 第二個結果選擇器
        self.compare_selector2 = QComboBox()
        selection_layout.addWidget(self.compare_selector2)
        
        # 比較按鈕
        compare_btn = QPushButton("執行比較")
        compare_btn.clicked.connect(self.execute_comparison)
        selection_layout.addWidget(compare_btn)
        
        config_layout.addLayout(selection_layout)
        
        # 比較選項
        options_layout = QHBoxLayout()
        
        # 比較指標選項
        self.metrics_checkbox = QCheckBox("評估指標")
        self.metrics_checkbox.setChecked(True)
        options_layout.addWidget(self.metrics_checkbox)
        
        self.aspects_checkbox = QCheckBox("面向詞彙")
        self.aspects_checkbox.setChecked(True)
        options_layout.addWidget(self.aspects_checkbox)
        
        self.visual_checkbox = QCheckBox("可視化比較")
        self.visual_checkbox.setChecked(True)
        options_layout.addWidget(self.visual_checkbox)
        
        options_layout.addStretch()
        
        config_layout.addLayout(options_layout)
        
        # 比較結果區
        results_group = QGroupBox("比較結果")
        results_layout = QVBoxLayout(results_group)
        
        self.comparison_text = QTextEdit()
        self.comparison_text.setReadOnly(True)
        
        results_layout.addWidget(self.comparison_text)
        
        # 比較操作按鈕
        action_layout = QHBoxLayout()
        
        view_report_btn = QPushButton("查看比較報告")
        view_report_btn.clicked.connect(self.view_comparison_report)
        action_layout.addWidget(view_report_btn)
        
        export_btn = QPushButton("導出比較結果")
        export_btn.clicked.connect(self.export_comparison_result)
        action_layout.addWidget(export_btn)
        
        action_layout.addStretch()
        
        results_layout.addLayout(action_layout)
        
        # 添加組件到佈局
        tab_layout.addWidget(config_group)
        tab_layout.addWidget(results_group, 1)
    
    def setup_settings_tab(self, tab):
        """設置設置標籤頁"""
        tab_layout = QVBoxLayout(tab)
        
        # 模型設置
        model_group = QGroupBox("模型設置")
        model_layout = QVBoxLayout(model_group)
        
        # BERT模型設置
        bert_layout = QHBoxLayout()
        bert_layout.addWidget(QLabel("BERT模型:"))
        self.bert_model_selector = QComboBox()
        self.bert_model_selector.addItems(["bert-base-uncased", "bert-large-uncased", "roberta-base", "distilbert-base-uncased"])
        bert_layout.addWidget(self.bert_model_selector)
        
        bert_layout.addWidget(QLabel("使用CUDA:"))
        self.use_cuda_checkbox = QCheckBox()
        self.use_cuda_checkbox.setChecked(True)
        bert_layout.addWidget(self.use_cuda_checkbox)
        
        bert_layout.addStretch()
        model_layout.addLayout(bert_layout)
        
        # LDA設置
        lda_layout = QHBoxLayout()
        lda_layout.addWidget(QLabel("LDA主題數:"))
        self.lda_topics_selector = QSpinBox()
        self.lda_topics_selector.setRange(3, 50)
        self.lda_topics_selector.setValue(10)
        lda_layout.addWidget(self.lda_topics_selector)
        
        lda_layout.addWidget(QLabel("最佳主題自動選擇:"))
        self.auto_topics_checkbox = QCheckBox()
        self.auto_topics_checkbox.setChecked(True)
        lda_layout.addWidget(self.auto_topics_checkbox)
        
        lda_layout.addStretch()
        model_layout.addLayout(lda_layout)
        
        # 數據設置
        data_group = QGroupBox("數據設置")
        data_layout = QVBoxLayout(data_group)
        
        # 輸入目錄
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("數據輸入目錄:"))
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setText(self.config.get("data_settings.input_directories.imdb", ""))
        input_layout.addWidget(self.input_dir_edit)
        
        browse_input_btn = QPushButton("瀏覽...")
        browse_input_btn.clicked.connect(lambda: self.browse_directory(self.input_dir_edit))
        input_layout.addWidget(browse_input_btn)
        
        data_layout.addLayout(input_layout)
        
        # 輸出目錄
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("結果輸出目錄:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setText(self.config.get("data_settings.output_directory", ""))
        output_layout.addWidget(self.output_dir_edit)
        
        browse_output_btn = QPushButton("瀏覽...")
        browse_output_btn.clicked.connect(lambda: self.browse_directory(self.output_dir_edit))
        output_layout.addWidget(browse_output_btn)
        
        data_layout.addLayout(output_layout)
        
        # 保存按鈕
        save_layout = QHBoxLayout()
        save_layout.addStretch()
        save_settings_btn = QPushButton("保存設置")
        save_settings_btn.clicked.connect(self.save_settings)
        save_layout.addWidget(save_settings_btn)
        
        # 添加組件到佈局
        tab_layout.addWidget(model_group)
        tab_layout.addWidget(data_group)
        tab_layout.addLayout(save_layout)
        tab_layout.addStretch()
    
    def load_result_sets(self):
        """從結果管理器載入結果集"""
        self.logger.info("載入結果集")
        
        # 獲取所有結果集
        try:
            result_sets = self.result_manager.get_result_sets()
            self.result_sets = {rs['id']: rs for rs in result_sets}
            
            # 更新結果選擇器
            self.result_selector.blockSignals(True)
            self.result_selector.clear()
            self.result_selector.addItem("請選擇結果集...", None)
            
            # 添加結果到下拉選擇器
            for result_id, result_set in self.result_sets.items():
                display_text = f"{result_id} - {result_set.get('dataset_name', '')} - {result_set.get('creation_time', '')}"
                self.result_selector.addItem(display_text, result_id)
            
            # 如果有當前選中的結果，嘗試重新選擇它
            if self.current_result_id:
                for i in range(self.result_selector.count()):
                    if self.result_selector.itemData(i) == self.current_result_id:
                        self.result_selector.setCurrentIndex(i)
                        break
            
            self.result_selector.blockSignals(False)
            
            # 更新比較標籤頁中的選擇器
            self.update_comparison_selectors()
            
            # 刷新結果表格
            self.refresh_results_table()
            
            self.logger.info(f"已載入 {len(result_sets)} 個結果集")
            
        except Exception as e:
            self.logger.error(f"載入結果集時出錯: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"載入結果集時出錯: {str(e)}")
    
    def refresh_results_table(self):
        """更新結果表格"""
        self.results_table.setRowCount(0)
        
        # 按創建時間倒序排列結果
        sorted_results = sorted(
            self.result_sets.values(), 
            key=lambda x: x.get('creation_time', ''), 
            reverse=True
        )
        
        for row, result in enumerate(sorted_results):
            result_id = result.get('id', '')
            dataset = result.get('dataset_name', '')
            creation_time = result.get('creation_time', '')
            status = result.get('status', '')
            description = result.get('description', '')
            
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(result_id))
            self.results_table.setItem(row, 1, QTableWidgetItem(dataset))
            self.results_table.setItem(row, 2, QTableWidgetItem(creation_time))
            self.results_table.setItem(row, 3, QTableWidgetItem(status))
            self.results_table.setItem(row, 4, QTableWidgetItem(description))
            
            # 創建操作按鈕
            actions_cell = QWidget()
            actions_layout = QHBoxLayout(actions_cell)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            view_btn = QPushButton("查看")
            view_btn.setProperty("result_id", result_id)
            view_btn.clicked.connect(lambda checked, rid=result_id: self.select_result(rid))
            actions_layout.addWidget(view_btn)
            
            self.results_table.setCellWidget(row, 5, actions_cell)
        
        # 調整列寬
        self.results_table.resizeColumnsToContents()
    
    def update_comparison_selectors(self):
        """更新比較選擇器的內容"""
        # 保存當前選擇的值
        selector1_value = self.compare_selector1.currentData()
        selector2_value = self.compare_selector2.currentData()
        
        # 清空並重新填充選擇器
        self.compare_selector1.clear()
        self.compare_selector2.clear()
        
        self.compare_selector1.addItem("請選擇結果...", None)
        self.compare_selector2.addItem("請選擇結果...", None)
        
        # 按創建時間倒序排列結果
        sorted_results = sorted(
            self.result_sets.values(), 
            key=lambda x: x.get('creation_time', ''), 
            reverse=True
        )
        
        # 只添加已完成的結果
        for result in sorted_results:
            if result.get('status') == 'completed':
                result_id = result.get('id')
                display_text = f"{result_id} - {result.get('dataset_name', '')}"
                self.compare_selector1.addItem(display_text, result_id)
                self.compare_selector2.addItem(display_text, result_id)
        
        # 恢復之前選擇的值
        if selector1_value:
            index = self.compare_selector1.findData(selector1_value)
            if index >= 0:
                self.compare_selector1.setCurrentIndex(index)
        
        if selector2_value:
            index = self.compare_selector2.findData(selector2_value)
            if index >= 0:
                self.compare_selector2.setCurrentIndex(index)
    
    def on_result_selected(self, index):
        """當結果選擇器選擇變更時調用"""
        result_id = self.result_selector.itemData(index)
        if result_id:
            self.select_result(result_id)
    
    def select_result(self, result_id):
        """選擇並顯示指定的結果"""
        if not result_id or result_id not in self.result_sets:
            return
        
        self.logger.info(f"選擇結果: {result_id}")
        self.current_result_id = result_id
        
        # 更新結果選擇器
        for i in range(self.result_selector.count()):
            if self.result_selector.itemData(i) == result_id:
                self.result_selector.blockSignals(True)
                self.result_selector.setCurrentIndex(i)
                self.result_selector.blockSignals(False)
                break
        
        # 更新各個標籤頁
        self.data_processing_tab.load_result(result_id)
        self.visualization_tab.load_result(result_id)
        
        # 加載面向組合標籤頁
        self.load_aspect_combinations_tab(result_id)
        
        # 更新狀態顯示
        result_set = self.result_sets.get(result_id, {})
        status = result_set.get('status', '未知')
        self.status_label.setText(f"當前結果: {result_id} ({status})")
        
        # 切換到視覺化標籤頁
        self.tab_widget.setCurrentIndex(1)
    
    def view_selected_report(self):
        """查看選中結果的報告"""
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "請先選擇一個結果")
            return
        
        row = selected_items[0].row()
        result_id = self.results_table.item(row, 0).text()
        
        # 查找最新的報告文件
        result_set = self.result_sets.get(result_id)
        if not result_set:
            QMessageBox.warning(self, "警告", f"無法找到結果: {result_id}")
            return
        
        if 'files' in result_set and 'report' in result_set['files']:
            report_file = result_set['files']['report']
            if os.path.exists(report_file):
                # 用瀏覽器打開報告
                QDesktopServices.openUrl(QUrl.fromLocalFile(report_file))
            else:
                QMessageBox.warning(self, "警告", f"報告文件不存在: {report_file}")
        else:
            QMessageBox.information(self, "提示", "該結果沒有關聯的報告")
    
    def delete_selected_result(self):
        """刪除選中的結果"""
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "請先選擇一個結果")
            return
        
        row = selected_items[0].row()
        result_id = self.results_table.item(row, 0).text()
        
        reply = QMessageBox.question(
            self, "確認刪除", 
            f"確定要刪除結果 '{result_id}' 嗎？此操作不可撤銷。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.result_manager.delete_result_set(result_id)
                self.logger.info(f"已刪除結果: {result_id}")
                
                # 刷新結果列表
                self.load_result_sets()
                
                # 如果當前選中的結果被刪除，清空當前結果
                if self.current_result_id == result_id:
                    self.current_result_id = None
                    self.status_label.setText("就緒")
                
                QMessageBox.information(self, "提示", f"已成功刪除結果 '{result_id}'")
            except Exception as e:
                self.logger.error(f"刪除結果時出錯: {str(e)}")
                QMessageBox.critical(self, "錯誤", f"刪除結果時出錯: {str(e)}")
    
    def export_selected_result(self):
        """導出選中的結果"""
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "請先選擇一個結果")
            return
        
        row = selected_items[0].row()
        result_id = self.results_table.item(row, 0).text()
        
        # 選擇導出目錄
        export_dir = QFileDialog.getExistingDirectory(self, "選擇導出目錄")
        if not export_dir:
            return
        
        try:
            # 執行導出操作
            exported_files = self.result_manager.export_result_set(result_id, export_dir)
            self.logger.info(f"已導出結果 {result_id} 到 {export_dir}")
            
            QMessageBox.information(self, "提示", f"已成功導出結果 '{result_id}' 到 {export_dir}")
        except Exception as e:
            self.logger.error(f"導出結果時出錯: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"導出結果時出錯: {str(e)}")
    
    def compare_selected_results(self):
        """比較表格中選中的結果"""
        # 獲取所有選中的結果ID
        selected_rows = set()
        for item in self.results_table.selectedItems():
            selected_rows.add(item.row())
        
        result_ids = []
        for row in selected_rows:
            result_ids.append(self.results_table.item(row, 0).text())
        
        if len(result_ids) < 2:
            QMessageBox.information(self, "提示", "請至少選擇兩個結果進行比較")
            return
        
        # 切換到比較標籤頁
        self.tab_widget.setCurrentIndex(3)
        
        # 設置比較選擇器
        if len(result_ids) >= 1:
            index = self.compare_selector1.findData(result_ids[0])
            if index >= 0:
                self.compare_selector1.setCurrentIndex(index)
        
        if len(result_ids) >= 2:
            index = self.compare_selector2.findData(result_ids[1])
            if index >= 0:
                self.compare_selector2.setCurrentIndex(index)
    
    def execute_comparison(self):
        """執行結果比較"""
        result_id1 = self.compare_selector1.currentData()
        result_id2 = self.compare_selector2.currentData()
        
        if not result_id1 or not result_id2 or result_id1 == result_id2:
            QMessageBox.information(self, "提示", "請選擇兩個不同的結果進行比較")
            return
        
        # 執行比較
        try:
            self.comparison_text.setText("正在比較結果，請稍候...")
            self.statusBar().showMessage("正在比較結果...")
            
            # 創建比較線程
            compare_thread = CompareThread(self.pipeline, [result_id1, result_id2])
            compare_thread.comparison_complete.connect(self.on_comparison_complete)
            compare_thread.comparison_error.connect(self.on_comparison_error)
            compare_thread.start()
            
        except Exception as e:
            self.logger.error(f"執行比較時出錯: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"執行比較時出錯: {str(e)}")
            self.statusBar().showMessage("就緒")
    
    def on_comparison_complete(self, result):
        """比較完成的回調"""
        self.statusBar().showMessage("比較完成")
        self.comparison_result = result
        
        # 顯示比較結果
        if result.get("success", False):
            # 構建比較結果文本
            report_text = "比較結果摘要:\n\n"
            
            # 添加比較的模型名稱
            models = [r.get("dataset_name", f"模型 {i+1}") for i, r in enumerate(result.get("results", []))]
            report_text += f"比較模型: {', '.join(models)}\n\n"
            
            # 添加指標比較
            report_text += "評估指標比較:\n"
            metrics_comparison = result.get("metrics_comparison", {})
            for metric, values in metrics_comparison.items():
                report_text += f"- {metric}: "
                metric_values = []
                for value_data in values:
                    name = value_data.get("name", "")
                    value = value_data.get("value")
                    if value is not None:
                        metric_values.append(f"{name}: {value:.4f}")
                report_text += ", ".join(metric_values) + "\n"
            
            # 添加報告文件信息
            if "report_file" in result:
                report_text += f"\n比較報告已生成: {result['report_file']}\n"
                report_text += "點擊「查看比較報告」按鈕查看詳細報告。"
            
            self.comparison_text.setText(report_text)
        else:
            error_msg = result.get("error", "未知錯誤")
            self.comparison_text.setText(f"比較失敗: {error_msg}")
    
    def on_comparison_error(self, error_msg):
        """比較出錯的回調"""
        self.statusBar().showMessage("比較失敗")
        QMessageBox.critical(self, "錯誤", f"比較過程中發生錯誤: {error_msg}")
        self.comparison_text.setText(f"比較失敗: {error_msg}")
    
    def view_comparison_report(self):
        """查看比較報告"""
        if hasattr(self, 'comparison_result') and self.comparison_result.get("success", False):
            report_file = self.comparison_result.get("report_file")
            if report_file and os.path.exists(report_file):
                # 用瀏覽器打開報告
                QDesktopServices.openUrl(QUrl.fromLocalFile(report_file))
            else:
                QMessageBox.warning(self, "警告", "比較報告文件不存在")
        else:
            QMessageBox.information(self, "提示", "請先執行比較")
    
    def export_comparison_result(self):
        """導出比較結果"""
        if hasattr(self, 'comparison_result') and self.comparison_result.get("success", False):
            # 選擇導出目錄
            export_dir = QFileDialog.getExistingDirectory(self, "選擇導出目錄")
            if not export_dir:
                return
            
            try:
                # 導出比較報告文件
                report_file = self.comparison_result.get("report_file")
                if report_file and os.path.exists(report_file):
                    import shutil
                    dest_file = os.path.join(export_dir, os.path.basename(report_file))
                    shutil.copy2(report_file, dest_file)
                    
                    # 導出比較數據
                    comparison_data = self.comparison_result.copy()
                    if 'report_file' in comparison_data:
                        comparison_data['report_file'] = os.path.basename(comparison_data['report_file'])
                    
                    data_file = os.path.join(export_dir, "comparison_data.json")
                    with open(data_file, 'w', encoding='utf-8') as f:
                        json.dump(comparison_data, f, indent=2)
                    
                    QMessageBox.information(self, "提示", f"比較結果已成功導出到 {export_dir}")
                else:
                    QMessageBox.warning(self, "警告", "比較報告文件不存在")
                    
            except Exception as e:
                self.logger.error(f"導出比較結果時出錯: {str(e)}")
                QMessageBox.critical(self, "錯誤", f"導出比較結果時出錯: {str(e)}")
        else:
            QMessageBox.information(self, "提示", "請先執行比較")
    
    def browse_directory(self, line_edit):
        """瀏覽並選擇目錄"""
        directory = QFileDialog.getExistingDirectory(self, "選擇目錄")
        if directory:
            line_edit.setText(directory)
    
    def save_settings(self):
        """保存配置設置"""
        try:
            # 保存BERT模型設置
            bert_model = self.bert_model_selector.currentText()
            use_cuda = self.use_cuda_checkbox.isChecked()
            
            self.config.set("model_settings.bert.model_name", bert_model)
            self.config.set("processing.use_cuda", use_cuda)
            
            # 保存LDA設置
            lda_topics = self.lda_topics_selector.value()
            auto_topics = self.auto_topics_checkbox.isChecked()
            
            self.config.set("model_settings.lda.num_topics", lda_topics)
            self.config.set("model_settings.lda.auto_topics", auto_topics)
            
            # 保存目錄設置
            input_dir = self.input_dir_edit.text()
            output_dir = self.output_dir_edit.text()
            
            self.config.set("data_settings.input_directories.imdb", input_dir)
            self.config.set("data_settings.output_directory", output_dir)
            
            # 保存配置文件
            self.config.save_user_config()
            
            QMessageBox.information(self, "提示", "設置已保存")
            self.logger.info("已保存配置設置")
            
        except Exception as e:
            self.logger.error(f"保存設置時出錯: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"保存設置時出錯: {str(e)}")
    
    def start_new_analysis(self):
        """開始新的面向分析任務"""
        dialog = NewAnalysisDialog(self.config, self)
        if dialog.exec():
            # 取得分析參數
            params = dialog.get_parameters()
            
            # 執行面向分析
            try:
                self.statusBar().showMessage("正在執行面向分析...")
                
                # 創建分析線程
                analysis_thread = AnalysisThread(self.pipeline, params)
                analysis_thread.analysis_complete.connect(self.on_analysis_complete)
                analysis_thread.analysis_error.connect(self.on_analysis_error)
                analysis_thread.start()
                
            except Exception as e:
                self.logger.error(f"開始面向分析時出錯: {str(e)}")
                QMessageBox.critical(self, "錯誤", f"開始面向分析時出錯: {str(e)}")
                self.statusBar().showMessage("就緒")
    
    def on_analysis_complete(self, result):
        """分析完成的回調"""
        self.statusBar().showMessage("分析完成")
        
        # 重新載入結果列表
        self.load_result_sets()
        
        # 選擇新的結果
        result_id = result.get("result_id")
        if result_id:
            self.select_result(result_id)
            QMessageBox.information(self, "提示", f"面向分析完成，結果ID: {result_id}")
    
    def on_analysis_error(self, error_msg):
        """分析出錯的回調"""
        self.statusBar().showMessage("分析失敗")
        QMessageBox.critical(self, "錯誤", f"面向分析過程中發生錯誤: {error_msg}")
    
    def add_aspect_to_combination(self):
        """添加選中的面向到當前組合"""
        if not self.combinations_result_loaded:
            QMessageBox.information(self, "提示", "請先載入結果集")
            return
        
        current_index = self.available_aspects_combo.currentIndex()
        if current_index < 0:
            return
        
        aspect_id = self.available_aspects_combo.currentData()
        
        # 檢查是否已添加
        if aspect_id in self.current_aspects:
            QMessageBox.information(self, "提示", "該面向已在當前組合中")
            return
        
        # 添加到當前組合
        self.current_aspects.append(aspect_id)
        
        # 更新顯示
        self.update_combination_label()

    def clear_combination(self):
        """清空當前面向組合"""
        self.current_aspects = []
        self.update_combination_label()

    def update_combination_label(self):
        """更新當前組合的顯示標籤"""
        if not self.current_aspects:
            self.current_combination_label.setText("尚未選擇任何面向")
            return
        
        # 構建組合文字
        aspects_text = []
        for aspect_id in self.current_aspects:
            if aspect_id in self.aspect_vectors:
                info = self.aspect_vectors[aspect_id]
                label = info.get('label', f'面向 {aspect_id}')
                aspects_text.append(label)
        
        self.current_combination_label.setText(", ".join(aspects_text))

    def analyze_current_combination(self):
        """分析當前選擇的面向組合"""
        if not self.current_aspects or len(self.current_aspects) < 2:
            QMessageBox.information(self, "提示", "請至少選擇兩個面向進行組合分析")
            return
        
        try:
            # 獲取選擇的組合方法
            method_index = self.combination_method_combo.currentIndex()
            method = ["attention", "multihead", "average"][method_index]
            
            # 分析組合
            if self.aspect_calculator:
                result = self.aspect_calculator.combine_aspects(
                    self.current_aspects,
                    method=method
                )
                
                # 更新可視化
                self.update_visualization()
                
                # 顯示分析結果
                result_text = f"組合分析結果:\n\n"
                result_text += f"組合方法: {self.combination_method_combo.currentText()}\n"
                result_text += f"包含的面向: {', '.join([self.aspect_vectors[aid]['label'] for aid in self.current_aspects])}\n\n"
                
                if 'keywords' in result:
                    result_text += f"組合關鍵詞: {', '.join(result['keywords'])}\n\n"
                
                if 'similarity_matrix' in result:
                    result_text += "面向相似度矩陣已計算，請查看可視化區域\n"
                
                self.vis_description.setText(result_text)
                
            else:
                QMessageBox.warning(self, "警告", "面向計算器未初始化")
        
        except Exception as e:
            self.logger.error(f"分析面向組合時出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "錯誤", f"分析面向組合時出錯: {str(e)}")

    def add_combination_to_comparison(self):
        """將當前組合添加到比較列表"""
        if not self.current_aspects or len(self.current_aspects) < 2:
            QMessageBox.information(self, "提示", "請至少選擇兩個面向進行組合")
            return
        
        # 構建組合描述
        method_text = self.combination_method_combo.currentText()
        aspects_text = ", ".join([self.aspect_vectors[aid]['label'] for aid in self.current_aspects])
        combination_text = f"{method_text}: {aspects_text}"
        
        # 構建組合對象
        combination = {
            "method": ["attention", "multihead", "average"][self.combination_method_combo.currentIndex()],
            "aspects": self.current_aspects.copy(),
            "description": combination_text
        }
        
        # 檢查是否已有相同組合
        for existing in self.combinations_to_compare:
            if existing["aspects"] == combination["aspects"] and existing["method"] == combination["method"]:
                QMessageBox.information(self, "提示", "該組合已在比較列表中")
                return
        
        # 添加到比較列表
        self.combinations_to_compare.append(combination)
        self.combinations_list.addItem(combination_text)

    def remove_selected_combination(self):
        """從比較列表中移除選中的組合"""
        current_row = self.combinations_list.currentRow()
        if current_row >= 0:
            self.combinations_to_compare.pop(current_row)
            self.combinations_list.takeItem(current_row)

    def clear_all_combinations(self):
        """清空所有待比較的組合"""
        self.combinations_to_compare = []
        self.combinations_list.clear()

    def compare_combinations(self):
        """比較列表中的所有面向組合"""
        if len(self.combinations_to_compare) < 2:
            QMessageBox.information(self, "提示", "請至少添加兩個組合到比較列表")
            return
        
        try:
            if self.aspect_calculator:
                # 獲取所有組合的向量
                comparison_results = {}
                
                for i, combo in enumerate(self.combinations_to_compare):
                    result = self.aspect_calculator.combine_aspects(
                        combo["aspects"],
                        method=combo["method"]
                    )
                    comparison_results[f"組合 {i+1}"] = {
                        "vector": result.get("combined_vector"),
                        "description": combo["description"],
                        "method": combo["method"]
                    }
                
                # 計算組合之間的相似度
                similarity_matrix = {}
                for name1, data1 in comparison_results.items():
                    similarity_matrix[name1] = {}
                    for name2, data2 in comparison_results.items():
                        if name1 != name2:
                            from scipy import spatial
                            similarity = 1 - spatial.distance.cosine(data1["vector"], data2["vector"])
                            similarity_matrix[name1][name2] = similarity
                
                # 創建比較報告
                report_text = "組合比較結果:\n\n"
                
                for name, data in comparison_results.items():
                    report_text += f"{name}: {data['description']}\n"
                
                report_text += "\n相似度矩陣:\n"
                for name1, similarities in similarity_matrix.items():
                    for name2, sim in similarities.items():
                        report_text += f"{name1} 與 {name2} 的相似度: {sim:.4f}\n"
                
                # 顯示結果
                self.analysis_result_text.setText(report_text)
                
                # 更新可視化
                self.visualize_combinations_similarity(comparison_results, similarity_matrix)
                
            else:
                QMessageBox.warning(self, "警告", "面向計算器未初始化")
        
        except Exception as e:
            self.logger.error(f"比較面向組合時出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "錯誤", f"比較面向組合時出錯: {str(e)}")

    def analyze_text_with_combinations(self):
        """使用面向組合分析輸入文本"""
        text = self.analysis_text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "提示", "請輸入要分析的文本")
            return
        
        if not self.current_aspects:
            QMessageBox.information(self, "提示", "請選擇至少一個面向")
            return
        
        try:
            if self.aspect_calculator:
                # 獲取選擇的組合方法
                method_index = self.combination_method_combo.currentIndex()
                method = ["attention", "multihead", "average"][method_index]
                
                # 分析文本
                result = self.aspect_calculator.analyze_text(
                    text,
                    aspect_ids=self.current_aspects,
                    combination_method=method
                )
                
                # 構建分析報告
                report_text = "文本面向分析結果:\n\n"
                report_text += f"輸入文本: {text}\n\n"
                
                # 添加各面向的分數
                report_text += "面向分數:\n"
                for aspect_id, score in result.get("aspect_scores", {}).items():
                    if aspect_id in self.aspect_vectors:
                        label = self.aspect_vectors[aspect_id]["label"]
                        report_text += f"{label}: {score:.4f}\n"
                
                # 添加情感極性
                if "sentiment" in result:
                    report_text += f"\n整體情感: {result['sentiment']:.4f} "
                    if result["sentiment"] > 0.1:
                        report_text += "(正面)"
                    elif result["sentiment"] < -0.1:
                        report_text += "(負面)"
                    else:
                        report_text += "(中性)"
                
                # 添加關鍵片段
                if "key_phrases" in result and result["key_phrases"]:
                    report_text += "\n\n關鍵片段:\n"
                    for phrase in result["key_phrases"]:
                        report_text += f"- {phrase}\n"
                
                self.analysis_result_text.setText(report_text)
                
            else:
                QMessageBox.warning(self, "警告", "面向計算器未初始化")
        
        except Exception as e:
            self.logger.error(f"分析文本時出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "錯誤", f"分析文本時出錯: {str(e)}")

    def update_visualization(self):
        """更新面向組合可視化"""
        if not self.current_aspects:
            return
        
        try:
            # 獲取可視化類型
            vis_type = self.vis_type_combo.currentIndex()
            
            if vis_type == 0:  # 相似度矩陣
                self.visualize_similarity_matrix()
            elif vis_type == 1:  # 組合向量分布
                self.visualize_vectors_distribution()
            elif vis_type == 2:  # 關鍵詞雲
                self.visualize_keywords_cloud()
        
        except Exception as e:
            self.logger.error(f"更新可視化時出錯: {str(e)}")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "錯誤", f"更新可視化時出錯: {str(e)}")

    def visualize_similarity_matrix(self):
        """可視化面向相似度矩陣"""
        if not self.aspect_calculator or len(self.current_aspects) < 2:
            return
        
        # 清除當前圖形
        self.figure.clear()
        
        # 獲取面向標籤
        labels = []
        for aspect_id in self.current_aspects:
            if aspect_id in self.aspect_vectors:
                label = self.aspect_vectors[aspect_id]["label"]
                labels.append(label)
        
        # 計算相似度矩陣
        similarity_matrix = self.aspect_calculator.calculate_similarity_matrix(self.current_aspects)
        
        # 創建熱力圖
        ax = self.figure.add_subplot(111)
        cax = ax.matshow(similarity_matrix, cmap='coolwarm')
        self.figure.colorbar(cax)
        
        # 設置軸標籤
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='left')
        ax.set_yticklabels(labels)
        
        # 添加數值標籤
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{similarity_matrix[i][j]:.2f}',
                       ha="center", va="center", color="black")
        
        ax.set_title('面向相似度矩陣')
        self.figure.tight_layout()
        
        # 更新畫布
        self.canvas.draw()
        
        # 更新描述
        desc_text = "相似度矩陣顯示了各個面向之間的相似程度。\n"
        desc_text += "值越接近1表示面向越相似，值越接近0表示面向越不相關。"
        self.vis_description.setText(desc_text)

    def visualize_vectors_distribution(self):
        """可視化面向向量分布"""
        if not self.aspect_calculator or not self.current_aspects:
            return
        
        # 清除當前圖形
        self.figure.clear()
        
        # 使用PCA降維
        from sklearn.decomposition import PCA
        
        # 獲取面向向量
        vectors = []
        labels = []
        for aspect_id in self.current_aspects:
            if aspect_id in self.aspect_vectors:
                vector = self.aspect_vectors[aspect_id]["vector"]
                vectors.append(vector)
                labels.append(self.aspect_vectors[aspect_id]["label"])
        
        if not vectors:
            return
        
        # 使用PCA降到2維
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(vectors)
        
        # 繪製散點圖
        ax = self.figure.add_subplot(111)
        ax.scatter(vectors_2d[:,0], vectors_2d[:,1])
        
        # 添加標籤
        for i, label in enumerate(labels):
            ax.annotate(label, (vectors_2d[i,0], vectors_2d[i,1]))
        
        # 如果有超過一個面向，嘗試計算組合向量
        if len(self.current_aspects) > 1:
            try:
                # 獲取選擇的組合方法
                method_index = self.combination_method_combo.currentIndex()
                method = ["attention", "multihead", "average"][method_index]
                
                # 計算組合向量
                result = self.aspect_calculator.combine_aspects(
                    self.current_aspects,
                    method=method
                )
                
                if "combined_vector" in result:
                    combined_vector = result["combined_vector"]
                    combined_vector_2d = pca.transform([combined_vector])
                    
                    # 添加組合向量到圖表
                    ax.scatter(combined_vector_2d[:,0], combined_vector_2d[:,1], c='red', s=100)
                    ax.annotate("組合", (combined_vector_2d[0,0], combined_vector_2d[0,1]), color='red')
            except Exception as e:
                self.logger.warning(f"計算組合向量時出錯: {str(e)}")
        
        ax.set_title('面向向量分布 (PCA降維)')
        ax.set_xlabel(f'主成分1 (解釋方差: {pca.explained_variance_ratio_[0]:.2f})')
        ax.set_ylabel(f'主成分2 (解釋方差: {pca.explained_variance_ratio_[1]:.2f})')
        ax.grid(True)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # 更新描述
        desc_text = "此圖顯示了各個面向在二維空間中的分布情況。\n"
        desc_text += "面向之間的距離反映了它們的相似程度，距離越近表示越相似。\n"
        desc_text += "紅色點表示使用選定方法組合的面向向量位置。"
        self.vis_description.setText(desc_text)

    def visualize_keywords_cloud(self):
        """可視化面向關鍵詞雲"""
        if not self.aspect_vectors or not self.current_aspects:
            return
        
        try:
            # 檢查是否安裝了wordcloud
            import wordcloud
            import matplotlib.pyplot as plt
            from collections import Counter
            
            # 清除當前圖形
            self.figure.clear()
            
            # 收集所有關鍵詞和權重
            all_keywords = []
            for aspect_id in self.current_aspects:
                if aspect_id in self.aspect_vectors:
                    keywords = self.aspect_vectors[aspect_id].get("keywords", [])
                    weights = self.aspect_vectors[aspect_id].get("weights", [])
                    
                    # 如果沒有權重，使用均勻權重
                    if not weights or len(weights) != len(keywords):
                        weights = [1] * len(keywords)
                    
                    # 添加到總列表
                    for kw, w in zip(keywords, weights):
                        all_keywords.extend([kw] * int(w * 20 + 1))  # 根據權重重複關鍵詞
            
            # 創建詞頻計數
            word_counts = Counter(all_keywords)
            
            # 生成詞雲
            wc = wordcloud.WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                contour_width=1,
                contour_color='steelblue'
            )
            
            # 如果詞雲為空，添加一個默認詞
            if not word_counts:
                word_counts = {"無關鍵詞": 1}
            
            wc.generate_from_frequencies(word_counts)
            
            # 顯示詞雲
            ax = self.figure.add_subplot(111)
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('面向關鍵詞雲')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # 更新描述
            desc_text = "詞雲顯示了所選面向中的關鍵詞，字體大小代表關鍵詞的重要性。"
            self.vis_description.setText(desc_text)
            
        except ImportError:
            self.logger.warning("無法導入wordcloud模組")
            self.vis_description.setText("無法創建詞雲可視化，請安裝wordcloud模組。")
        except Exception as e:
            self.logger.error(f"創建詞雲時出錯: {str(e)}")
            self.vis_description.setText(f"創建詞雲時出錯: {str(e)}")

    def visualize_combinations_similarity(self, combinations, similarity_matrix):
        """可視化組合之間的相似度"""
        if not combinations or not similarity_matrix:
            return
        
        # 清除當前圖形
        self.figure.clear()
        
        # 提取標籤和相似度值
        labels = list(combinations.keys())
        similarities = []
        for name1 in labels:
            sim_row = []
            for name2 in labels:
                if name1 == name2:
                    sim_row.append(1.0)  # 自身相似度為1
                else:
                    sim_row.append(similarity_matrix[name1].get(name2, 0))
            similarities.append(sim_row)
        
        # 創建熱力圖
        ax = self.figure.add_subplot(111)
        cax = ax.matshow(similarities, cmap='coolwarm', vmin=0, vmax=1)
        self.figure.colorbar(cax)
        
        # 設置軸標籤
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # 添加數值標籤
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f'{similarities[i][j]:.2f}',
                       ha="center", va="center", color="black")
        
        ax.set_title('組合相似度矩陣')
        self.figure.tight_layout()
        
        # 更新畫布
        self.canvas.draw()


class NewAnalysisDialog(QDialog):
    """新面向分析任務對話框"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        
        self.setWindowTitle("新面向分析")
        self.setMinimumSize(600, 400)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化對話框UI"""
        layout = QVBoxLayout(self)
        
        # 數據集選擇
        dataset_group = QGroupBox("數據集選擇")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # 數據集類型
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("數據集類型:"))
        self.dataset_type = QComboBox()
        self.dataset_type.addItems(["IMDB", "Amazon", "Yelp", "自定義"])
        self.dataset_type.currentIndexChanged.connect(self.on_dataset_type_changed)
        type_layout.addWidget(self.dataset_type)
        dataset_layout.addLayout(type_layout)
        
        # 數據集文件
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("數據文件:"))
        self.file_path = QLineEdit()
        file_layout.addWidget(self.file_path)
        self.browse_btn = QPushButton("瀏覽...")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)
        dataset_layout.addLayout(file_layout)
        
        # 抽樣設置
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(QLabel("樣本大小:"))
        self.sample_size = QSpinBox()
        self.sample_size.setRange(0, 100000)
        self.sample_size.setValue(1000)
        self.sample_size.setSpecialValueText("全部")
        sample_layout.addWidget(self.sample_size)
        dataset_layout.addLayout(sample_layout)
        
        # 實驗描述
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("實驗描述:"))
        self.description = QLineEdit()
        desc_layout.addWidget(self.description)
        
        # 添加主要組件
        layout.addWidget(dataset_group)
        layout.addLayout(desc_layout)
        
        # 模型參數
        model_group = QGroupBox("模型參數")
        model_layout = QVBoxLayout(model_group)
        
        # 主題數
        topics_layout = QHBoxLayout()
        topics_layout.addWidget(QLabel("LDA主題數:"))
        self.num_topics = QSpinBox()
        self.num_topics.setRange(3, 50)
        self.num_topics.setValue(int(self.config.get("model_settings.lda.num_topics", 10)))
        topics_layout.addWidget(self.num_topics)
        
        self.auto_topics = QCheckBox("自動選擇最佳主題數")
        self.auto_topics.setChecked(self.config.get("model_settings.lda.auto_topics", True))
        self.auto_topics.toggled.connect(self.on_auto_topics_toggled)
        topics_layout.addWidget(self.auto_topics)
        topics_layout.addStretch()
        model_layout.addLayout(topics_layout)
        
        # 使用CUDA
        cuda_layout = QHBoxLayout()
        self.use_cuda = QCheckBox("使用CUDA加速")
        self.use_cuda.setChecked(self.config.get("processing.use_cuda", True))
        cuda_layout.addWidget(self.use_cuda)
        cuda_layout.addStretch()
        model_layout.addLayout(cuda_layout)
        
        layout.addWidget(model_group)
        
        # 按鈕
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        start_btn = QPushButton("開始分析")
        start_btn.clicked.connect(self.accept)
        button_layout.addWidget(start_btn)
        
        layout.addLayout(button_layout)
        
        # 初始化UI狀態
        self.on_dataset_type_changed(0)
        self.on_auto_topics_toggled(self.auto_topics.isChecked())
    
    def on_dataset_type_changed(self, index):
        """當數據集類型變更時更新UI"""
        dataset_type = self.dataset_type.currentText().lower()
        
        if dataset_type == "imdb":
            default_path = self.config.get("data_settings.input_directories.imdb", "")
            if default_path:
                default_path = os.path.join(default_path, "IMDB Dataset.csv")
            self.file_path.setText(default_path)
        elif dataset_type == "amazon":
            default_path = self.config.get("data_settings.input_directories.amazon", "")
            self.file_path.setText(default_path)
        elif dataset_type == "yelp":
            default_path = self.config.get("data_settings.input_directories.yelp", "")
            if default_path:
                default_path = os.path.join(default_path, "yelp_review.json")
            self.file_path.setText(default_path)
        else:
            self.file_path.clear()
    
    def on_auto_topics_toggled(self, checked):
        """當自動主題選擇切換時更新UI"""
        self.num_topics.setEnabled(not checked)
    
    def browse_file(self):
        """瀏覽文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇數據文件")
        if file_path:
            self.file_path.setText(file_path)
    
    def get_parameters(self) -> Dict[str, Any]:
        """獲取面向分析參數"""
        dataset_type = self.dataset_type.currentText().lower()
        file_path = self.file_path.text()
        
        sample_size = self.sample_size.value()
        if sample_size == 0:  # "全部" 選項
            sample_size = None
        
        # 數據集參數
        dataset_params = {"file_path": file_path}
        
        if dataset_type == "yelp":
            # 假設業務文件在相同目錄
            business_file = os.path.join(os.path.dirname(file_path), "yelp_business.json")
            if os.path.exists(business_file):
                dataset_params["business_file"] = business_file
            dataset_params["category_filter"] = "Restaurants"
        
        # 返回完整參數
        return {
            "dataset_type": dataset_type,
            "dataset_params": dataset_params,
            "sample_size": sample_size,
            "description": self.description.text(),
            "num_topics": self.num_topics.value() if not self.auto_topics.isChecked() else None,
            "auto_topics": self.auto_topics.isChecked(),
            "use_cuda": self.use_cuda.isChecked()
        }


class AnalysisThread(QThread):
    """面向分析線程"""
    analysis_complete = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)
    analysis_progress = pyqtSignal(str, int)  # 進度信號
    
    def __init__(self, pipeline, params):
        try:
            super().__init__()
            self.pipeline = pipeline
            self.params = params
            
            # 確保日誌目錄存在
            log_dir = os.path.join('Part03_', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            # 配置日誌記錄器
            self.logger = logging.getLogger('analysis_thread')
            self.logger.setLevel(logging.DEBUG)
            
            # 清除舊的處理器
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
            
            # 添加新的檔案處理器
            log_file = os.path.join(log_dir, 'analysis_thread.log')
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
            # 同時配置標準錯誤輸出處理器
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levellevel)s - %(message)s'))
            self.logger.addHandler(console_handler)
            
            # 配置 Matplotlib 中文字體支援
            self.configure_matplotlib_chinese_fonts()
            
            self.logger.info("分析線程初始化成功")
        except Exception as e:
            # 如果在初始化時發生錯誤，直接寫入緊急日誌文件
            emergency_log_file = os.path.join('Part03_', 'logs', 'thread_init_error.log')
            os.makedirs(os.path.dirname(emergency_log_file), exist_ok=True)
            with open(emergency_log_file, 'w', encoding='utf-8') as f:
                f.write(f"線程初始化錯誤: {str(e)}\n")
                f.write(traceback.format_exc())
    
    def get_system_fonts(self):
        """根據作業系統取得適合的中文字體列表"""
        system = 'Windows'
        
        # 根據不同操作系統返回適合的中文字體
        if system == 'Windows':
            # Windows 中文字體優先順序
            return [
                'Microsoft JhengHei', 'Microsoft YaHei', 
                'DFKai-SB', 'MingLiU', 'SimHei', 'SimSun', 'NSimSun',
                'FangSong', 'KaiTi', 'STKaiti', 'STSong'
            ]

    def configure_matplotlib_chinese_fonts(self):
        """配置 Matplotlib 支援中文顯示"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.font_manager as fm
            from matplotlib.font_manager import FontProperties

            self.logger.info("配置 Matplotlib 中文字體支援")
            
            # 獲取系統特定的字體
            system_fonts = self.get_system_fonts()
            
            # 一般使用的字體優先順序
            general_fonts = system_fonts + [
                'Arial Unicode MS', 'DejaVu Sans', 'FreeSans', 'Droid Sans Fallback',
                'Source Han Sans TC', 'Source Han Sans SC', 'Noto Sans CJK JP'
            ]
            
            # 查找系統中存在的字體
            available_fonts = []
            for font_name in general_fonts:
                try:
                    # 檢查字體是否可用
                    font_path = fm.findfont(fm.FontProperties(family=font_name), fallback_to_default=False)
                    if font_path and os.path.exists(font_path) and font_path.endswith((".ttf", ".ttc", ".otf")):
                        available_fonts.append(font_name)
                        self.logger.info(f"找到可用的中文字體: {font_name} ({font_path})")
                except Exception:
                    continue
            
            if available_fonts:
                # 設置找到的中文字體
                plt.rcParams['font.sans-serif'] = available_fonts + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
                plt.rcParams['font.family'] = 'sans-serif'
                
                self.logger.info(f"中文字體配置成功，使用字體: {', '.join(available_fonts[:3])}")
                
                # 創建一個測試圖形確認設置生效
                test_fig = plt.figure(figsize=(1, 1))
                test_ax = test_fig.add_subplot(111)
                test_ax.set_title('中文測試')  # 中文標題
                test_image_path = os.path.join('Part03_', 'logs', 'chinese_font_test.png')
                plt.savefig(test_image_path)
                plt.close(test_fig)
                self.logger.info(f"中文字體測試圖片已儲存至: {test_image_path}")
            else:
                # 如果沒有找到任何可用中文字體，使用備用方案
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + general_fonts
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.family'] = 'sans-serif'
                self.logger.warning("找不到可用的中文字體，使用系統默認字體")
                
                # 嘗試直接嵌入字型：將 Arial Unicode MS 字體檔案放置在專案目錄中
                fonts_dir = os.path.join('Part03_', 'fonts')
                if not os.path.exists(fonts_dir):
                    os.makedirs(fonts_dir, exist_ok=True)
                
                # 查找用戶系統中已知的字體目錄
                potential_font_paths = []
                if 'Windows':
                    potential_font_paths = [
                        os.path.join(os.environ.get('SystemRoot', 'C:\\Windows'), 'Fonts'),
                        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts')
                    ]
                
                # 嘗試在系統中找到 simsun.ttc 或其他已知中文字體檔案
                for path in potential_font_paths:
                    if os.path.exists(path):
                        for font_file in ['simsun.ttc', 'simhei.ttf', 'msyh.ttc', 'msjh.ttc']:
                            font_path = os.path.join(path, font_file)
                            if os.path.exists(font_path):
                                self.logger.info(f"在系統目錄找到中文字體: {font_path}")
                                # 直接使用該字體設定
                                plt.rcParams['font.sans-serif'] = ['SimSun' if 'simsun' in font_file.lower() else 
                                                                'SimHei' if 'simhei' in font_file.lower() else 
                                                                'Microsoft YaHei' if 'msyh' in font_file.lower() else
                                                                'Microsoft JhengHei'] + plt.rcParams['font.sans-serif']
                                
                                self.logger.info(f"直接設定 font.sans-serif = {plt.rcParams['font.sans-serif'][0]}")
                                
                                # 驗證字體設置
                                test_fig = plt.figure(figsize=(1, 1))
                                test_ax = test_fig.add_subplot(111)
                                test_ax.set_title('中文測試')  # 中文標題
                                test_image_path = os.path.join('Part03_', 'logs', 'chinese_font_test_direct.png')
                                plt.savefig(test_image_path)
                                plt.close(test_fig)
                                self.logger.info(f"直接設定字體後，中文字體測試圖片已儲存至: {test_image_path}")
                                break
                        else:
                            continue
                        break
                
        except ImportError as ie:
            self.logger.warning(f"無法導入 Matplotlib 相關模組: {str(ie)}")
        except Exception as e:
            self.logger.warning(f"配置 Matplotlib 中文字體時出錯: {str(e)}")
            self.logger.warning(traceback.format_exc())
    
    def run(self):
        # 創建一個臨時日誌文件用於緊急錯誤
        emergency_log_file = os.path.join('Part03_', 'logs', 'emergency_run_error.log')
        
        try:
            # 記錄開始執行
            self.logger.info(f"開始面向分析，參數: {self.params}")
            self.analysis_progress.emit("正在初始化分析...", 0)
            
            # 記錄系統環境
            import platform
            import psutil
            
            self.logger.info(f"作業系統: {platform.platform()}")
            self.logger.info(f"Python版本: {platform.python_version()}")
            self.logger.info(f"系統總記憶體: {psutil.virtual_memory().total / (1024**3):.2f} GB")
            self.logger.info(f"系統可用記憶體: {psutil.virtual_memory().available / (1024**3):.2f} GB")
            
            # 優先檢查CUDA相關設置
            if self.params.get("use_cuda", True):
                self.logger.info("檢查CUDA可用性...")
                self.analysis_progress.emit("檢查CUDA可用性...", 5)
                
                try:
                    import torch
                    self.logger.info(f"PyTorch版本: {torch.__version__}")
                    
                    if torch.cuda.is_available():
                        self.logger.info(f"CUDA可用，設備: {torch.cuda.get_device_name(0)}")
                        self.logger.info(f"CUDA版本: {torch.version.cuda}")
                        self.logger.info(f"GPU總記憶體: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
                        self.logger.info(f"GPU可用記憶體: {torch.cuda.memory_reserved(0) / (1024**3):.2f} GB 已保留")
                        
                        # 預先清理GPU記憶體
                        torch.cuda.empty_cache()
                        self.logger.info("已清理GPU快取記憶體")
                    else:
                        self.logger.warning("CUDA不可用，將使用CPU模式")
                        self.logger.info(f"CPU數量: {os.cpu_count()}")
                        self.params["use_cuda"] = False
                except ImportError as ie:
                    self.logger.warning(f"無法導入PyTorch: {str(ie)}")
                    self.params["use_cuda"] = False
                except Exception as cuda_e:
                    self.logger.warning(f"檢查CUDA時出錯: {cuda_e}")
                    self.logger.warning(traceback.format_exc())
                    self.params["use_cuda"] = False
            
            # 更新進度
            self.analysis_progress.emit("正在載入數據...", 10)
            
            # 檢查數據文件是否存在
            dataset_params = self.params.get("dataset_params", {})
            file_path = dataset_params.get("file_path", "")
            
            self.logger.info(f"檢查數據文件: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到數據文件: {file_path}")
            
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            self.logger.info(f"數據文件大小: {file_size:.2f} MB")
            
            # 檢查Pipeline對象是否正確
            self.logger.info(f"Pipeline類型: {type(self.pipeline).__name__}")
            self.logger.info("即將開始執行面向分析流程")
            
            # 執行面向分析
            result = self.pipeline.run_full_pipeline(
                dataset_type=self.params["dataset_type"],
                dataset_params=self.params["dataset_params"],
                sample_size=self.params["sample_size"],
                description=self.params["description"],
                console_output=True
            )
            
            # 記錄完成
            self.logger.info(f"面向分析完成，結果ID: {result.get('result_id', 'unknown')}")
            
            # 完成後清理CUDA記憶體
            if self.params.get("use_cuda", True):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.logger.info("面向分析完成後清理GPU記憶體成功")
                except Exception as e:
                    self.logger.warning(f"清理CUDA記憶體時出錯: {str(e)}")
                
            self.analysis_complete.emit(result)
            
        except FileNotFoundError as fnf:
            error_msg = f"檔案不存在: {str(fnf)}"
            self.logger.error(error_msg)
            self.analysis_error.emit(error_msg)
        except ImportError as ie:
            error_msg = f"缺少必要的模組: {str(ie)}"
            self.logger.error(error_msg)
            self.analysis_error.emit(error_msg)
        except MemoryError as me:
            error_msg = f"記憶體不足: {str(me)}"
            self.logger.error(error_msg)
            self.analysis_error.emit(error_msg)
        except Exception as e:
            # 詳細記錄錯誤信息
            error_msg = f"面向分析發生異常: {str(e)}"
            stack_trace = traceback.format_exc()
            
            # 記錄到日誌
            self.logger.error(error_msg)
            self.logger.error(stack_trace)
            
            # 同時寫入緊急錯誤日誌
            try:
                with open(emergency_log_file, 'w', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                    f.write(stack_trace)
            except:
                pass  # 如果緊急日誌也無法寫入，就放棄
            
            # 出錯後清理CUDA記憶體
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("錯誤後清理GPU記憶體成功")
            except Exception as cuda_e:
                if hasattr(self, 'logger'):
                    self.logger.error(f"清理CUDA記憶體時出錯: {str(cuda_e)}")
            
            # 向UI發送錯誤訊息
            self.analysis_error.emit(error_msg)


class CompareThread(QThread):
    """結果比較線程"""
    comparison_complete = pyqtSignal(dict)
    comparison_error = pyqtSignal(str)
    
    def __init__(self, pipeline, result_ids):
        super().__init__()
        self.pipeline = pipeline
        self.result_ids = result_ids
    
    def run(self):
        try:
            # 執行結果比較
            result = self.pipeline.compare_results(self.result_ids)
            
            self.comparison_complete.emit(result)
        except Exception as e:
            self.comparison_error.emit(str(e))


def main():
    """主函數"""
    app = QApplication(sys.argv)
    app.setApplicationName("面向分析系統")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
