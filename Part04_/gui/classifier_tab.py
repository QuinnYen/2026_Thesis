#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分類器分頁模組 - 情感分類器的GUI界面
使用面向向量進行情感分類的用戶界面
"""

import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit,
    QFileDialog, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

# 導入系統模組
from modules.sentiment_classifier import SentimentClassifier
from utils.logger import get_logger
from utils.file_manager import FileManager

class ClassifierWorker(QThread):
    """分類器處理工作線程"""
    
    progress_updated = pyqtSignal(int, str)  # 進度值, 狀態訊息
    status_message = pyqtSignal(str)  # 狀態訊息
    error_occurred = pyqtSignal(str)  # 錯誤訊息
    finished = pyqtSignal(dict)  # 完成信號，傳遞結果
    
    def __init__(self, classifier, operation, **kwargs):
        super().__init__()
        self.classifier = classifier
        self.operation = operation
        self.kwargs = kwargs
        self.logger = get_logger("classifier_worker")
    
    def run(self):
        """執行分類器操作"""
        try:
            if self.operation == "train_and_evaluate":
                self._train_and_evaluate()
            else:
                self.error_occurred.emit(f"未知操作: {self.operation}")
                
        except Exception as e:
            self.logger.error(f"工作線程執行失敗: {str(e)}")
            self.error_occurred.emit(str(e))
    
    def _train_and_evaluate(self):
        """訓練和評估模型"""
        self.progress_updated.emit(30, "訓練模型...")
        self.classifier.train_model()
        
        self.progress_updated.emit(70, "評估模型...")
        self.classifier.evaluate_model()
        
        self.progress_updated.emit(100, "完成")
        self.finished.emit({"operation": "train_and_evaluate"})

class ClassifierTab(QWidget):
    """分類器分頁類"""
    
    status_message = pyqtSignal(str, int)  # 狀態訊息, 顯示時間
    progress_updated = pyqtSignal(int, int)  # 當前值, 最大值
    
    def __init__(self, config=None, file_manager=None):
        super().__init__()
        
        self.logger = get_logger("classifier_tab")
        self.config = config
        self.file_manager = file_manager or FileManager(config)
        
        # 分類器實例
        self.classifier = None
        self.worker = None
        
        # 初始化UI
        self._init_ui()
        
        self.logger.info("分類器分頁初始化完成")
    
    def _init_ui(self):
        """初始化用戶界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 資料準備區域
        data_group = QGroupBox("資料準備")
        data_layout = QGridLayout(data_group)
        
        # 面向向量檔案選擇
        data_layout.addWidget(QLabel("面向向量檔案:"), 0, 0)
        self.aspect_vectors_edit = QLineEdit()
        self.aspect_vectors_edit.setPlaceholderText("選擇面向向量結果檔案 (.json)")
        data_layout.addWidget(self.aspect_vectors_edit, 0, 1)
        
        browse_aspect_btn = QPushButton("瀏覽...")
        browse_aspect_btn.clicked.connect(self._browse_aspect_vectors_file)
        data_layout.addWidget(browse_aspect_btn, 0, 2)
        
        # 原始資料檔案選擇
        data_layout.addWidget(QLabel("原始資料檔案:"), 1, 0)
        self.raw_data_edit = QLineEdit()
        self.raw_data_edit.setPlaceholderText("選擇原始資料檔案 (.csv)")
        data_layout.addWidget(self.raw_data_edit, 1, 1)
        
        browse_raw_btn = QPushButton("瀏覽...")
        browse_raw_btn.clicked.connect(self._browse_raw_data_file)
        data_layout.addWidget(browse_raw_btn, 1, 2)
        
        # 載入按鈕
        load_btn = QPushButton("載入資料")
        load_btn.clicked.connect(self._load_data)
        data_layout.addWidget(load_btn, 2, 1)
        
        main_layout.addWidget(data_group)
        
        # 資料資訊顯示
        info_group = QGroupBox("資料資訊")
        info_layout = QVBoxLayout(info_group)
        
        self.data_info_text = QTextEdit()
        self.data_info_text.setReadOnly(True)
        self.data_info_text.setMaximumHeight(150)
        info_layout.addWidget(self.data_info_text)
        
        main_layout.addWidget(info_group)
        
        # 訓練控制區域
        train_group = QGroupBox("模型訓練")
        train_layout = QVBoxLayout(train_group)
        
        # 訓練按鈕
        self.train_btn = QPushButton("訓練模型")
        self.train_btn.clicked.connect(self._train_models)
        self.train_btn.setEnabled(False)
        train_layout.addWidget(self.train_btn)
        
        # 訓練日誌
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumHeight(150)
        train_layout.addWidget(self.training_log)
        
        main_layout.addWidget(train_group)
        
        # 結果顯示區域
        results_group = QGroupBox("分類結果")
        results_layout = QVBoxLayout(results_group)
        
        # 性能表格
        self.performance_table = QTableWidget()
        self.performance_table.setColumnCount(4)
        self.performance_table.setHorizontalHeaderLabels([
            "準確率", "精確率", "召回率", "F1分數"
        ])
        self.performance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.performance_table)
        
        # 詳細結果
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        main_layout.addWidget(results_group)
        
        # 進度條和狀態
        progress_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就緒")
        self.status_label.setStyleSheet("QLabel { color: green; }")
        progress_layout.addWidget(self.status_label)
        
        main_layout.addLayout(progress_layout)
    
    def _browse_aspect_vectors_file(self):
        """瀏覽面向向量檔案"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇面向向量檔案", 
            "", "JSON files (*.json)"
        )
        if file_path:
            self.aspect_vectors_edit.setText(file_path)
    
    def _browse_raw_data_file(self):
        """瀏覽原始資料檔案"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "選擇原始資料檔案", 
            "", "CSV files (*.csv)"
        )
        if file_path:
            self.raw_data_edit.setText(file_path)
    
    def _load_data(self):
        """載入資料"""
        try:
            aspect_vectors_file = self.aspect_vectors_edit.text().strip()
            raw_data_file = self.raw_data_edit.text().strip()
            
            if not aspect_vectors_file or not raw_data_file:
                QMessageBox.warning(self, "警告", "請選擇所有必要的檔案")
                return
            
            # 初始化分類器
            if self.config:
                output_dir = self.config.get("paths", "output_dir")
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                part04_dir = os.path.dirname(current_dir)
                output_dir = os.path.join(part04_dir, "1_output")
            
            self.classifier = SentimentClassifier(output_dir)
            
            # 載入資料
            features, labels = self.classifier.load_and_process_data(
                raw_data_path=raw_data_file,
                aspect_vectors_path=aspect_vectors_file
            )
            
            # 準備資料
            self.classifier.prepare_data(features, labels)
            
            # 顯示資料資訊
            info = f"""資料載入成功！

特徵維度: {features.shape[1]}
總樣本數: {len(features)}
訓練集樣本數: {len(self.classifier.X_train)}
測試集樣本數: {len(self.classifier.X_test)}

標籤分佈:
- negative: {sum(labels == 0)} 樣本
- positive: {sum(labels == 1)} 樣本
"""
            
            self.data_info_text.setPlainText(info)
            
            # 啟用訓練按鈕
            self.train_btn.setEnabled(True)
            
            self.status_message.emit("資料載入完成", 3000)
            
        except Exception as e:
            self.logger.error(f"載入資料失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"載入資料失敗:\n{str(e)}")
    
    def _train_models(self):
        """訓練模型"""
        if not self.classifier:
            QMessageBox.warning(self, "警告", "請先載入資料")
            return
        
        # 開始訓練
        self._start_operation("train_and_evaluate")
    
    def _start_operation(self, operation, **kwargs):
        """開始執行操作"""
        self.worker = ClassifierWorker(self.classifier, operation, **kwargs)
        
        self.worker.progress_updated.connect(self._update_progress)
        self.worker.status_message.connect(self._update_status)
        self.worker.error_occurred.connect(self._handle_error)
        self.worker.finished.connect(self._operation_finished)
        
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.worker.start()
    
    def _update_progress(self, value, message):
        """更新進度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] {message}")
    
    def _update_status(self, message):
        """更新狀態"""
        self.status_label.setText(message)
    
    def _handle_error(self, error_message):
        """處理錯誤"""
        self.status_label.setText("發生錯誤")
        self.status_label.setStyleSheet("QLabel { color: red; }")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] 錯誤: {error_message}")
        
        QMessageBox.critical(self, "錯誤", f"操作失敗:\n{error_message}")
        
        self._operation_finished({})
    
    def _operation_finished(self, result):
        """操作完成"""
        self.progress_bar.setVisible(False)
        self.train_btn.setEnabled(True)
        
        if result.get("operation") == "train_and_evaluate":
            self._update_results_display()
        
        self.status_label.setText("就緒")
        self.status_label.setStyleSheet("QLabel { color: green; }")
        
        self.status_message.emit("操作完成", 3000)
    
    def _update_results_display(self):
        """更新結果顯示"""
        if not self.classifier or not self.classifier.evaluation_results:
            return
        
        results = self.classifier.evaluation_results
        
        # 更新性能表格
        self.performance_table.setRowCount(1)
        self.performance_table.setColumnCount(4)
        
        # 從分類報告中提取指標
        report = results.get('classification_report', {})
        accuracy = results.get('accuracy', 0)
        
        # 計算宏觀平均指標
        macro_avg = report.get('macro avg', {})
        precision = macro_avg.get('precision', 0)
        recall = macro_avg.get('recall', 0)
        f1_score = macro_avg.get('f1-score', 0)
        
        # 更新表格
        metrics = [
            ('準確率', accuracy),
            ('精確率', precision),
            ('召回率', recall),
            ('F1分數', f1_score)
        ]
        
        for j, (label, value) in enumerate(metrics):
            item = QTableWidgetItem(f"{value:.4f}")
            item.setTextAlignment(Qt.AlignCenter)
            self.performance_table.setItem(0, j, item)
        
        # 更新詳細結果
        text_parts = [
            "=== 分類結果 ===",
            f"準確率: {accuracy:.4f}",
            "",
            "=== 各類別詳細結果 ==="
        ]
        
        # 添加每個類別的詳細結果
        for label in ['負面', '正面']:
            if label in report:
                metrics = report[label]
                text_parts.extend([
                    f"\n{label}:",
                    f"精確率: {metrics.get('precision', 0):.4f}",
                    f"召回率: {metrics.get('recall', 0):.4f}",
                    f"F1分數: {metrics.get('f1-score', 0):.4f}",
                    f"樣本數: {metrics.get('support', 0)}"
                ])
        
        # 添加宏觀平均結果
        text_parts.extend([
            "\n=== 宏觀平均 ===",
            f"精確率: {precision:.4f}",
            f"召回率: {recall:.4f}",
            f"F1分數: {f1_score:.4f}"
        ])
        
        self.results_text.setPlainText("\n".join(text_parts))