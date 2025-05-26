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
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QSplitter, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon

# 導入系統模組
from modules.sentiment_classifier import SentimentClassifier
from utils.logger import get_logger
from utils.file_manager import FileManager

class ClassifierWorker(QThread):
    """分類器處理工作線程"""
    
    # 信號定義
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
            elif self.operation == "cross_validation":
                self._cross_validation()
            elif self.operation == "hyperparameter_tuning":
                self._hyperparameter_tuning()
            else:
                self.error_occurred.emit(f"未知操作: {self.operation}")
                
        except Exception as e:
            self.logger.error(f"工作線程執行失敗: {str(e)}")
            self.error_occurred.emit(str(e))
    
    def _train_and_evaluate(self):
        """訓練和評估模型"""
        self.progress_updated.emit(10, "初始化模型...")
        self.classifier.initialize_models()
        
        self.progress_updated.emit(30, "訓練模型...")
        self.classifier.train_models()
        
        self.progress_updated.emit(70, "評估模型...")
        self.classifier.evaluate_models()
        
        self.progress_updated.emit(90, "生成視覺化...")
        self.classifier.generate_visualizations()
        
        self.progress_updated.emit(100, "完成")
        self.finished.emit({"operation": "train_and_evaluate"})
    
    def _cross_validation(self):
        """執行交叉驗證"""
        cv_folds = self.kwargs.get('cv_folds', 5)
        
        self.progress_updated.emit(20, "執行交叉驗證...")
        self.classifier.cross_validation(cv_folds)
        
        self.progress_updated.emit(100, "完成")
        self.finished.emit({"operation": "cross_validation"})
    
    def _hyperparameter_tuning(self):
        """執行超參數調優"""
        model_name = self.kwargs.get('model_name', None)
        
        self.progress_updated.emit(20, "執行超參數調優...")
        self.classifier.hyperparameter_tuning(model_name)
        
        self.progress_updated.emit(100, "完成")
        self.finished.emit({"operation": "hyperparameter_tuning"})

class ClassifierTab(QWidget):
    """分類器分頁類"""
    
    # 信號定義
    status_message = pyqtSignal(str, int)  # 狀態訊息, 顯示時間
    progress_updated = pyqtSignal(int, int)  # 當前值, 最大值
    
    def __init__(self, config=None, file_manager=None):
        """
        初始化分類器分頁
        
        Args:
            config: 配置對象
            file_manager: 文件管理器
        """
        super().__init__()
        
        self.logger = get_logger("classifier_tab")
        self.config = config
        self.file_manager = file_manager or FileManager(config)
        
        # 分類器實例
        self.classifier = None
        self.worker = None
        
        # 結果資料
        self.classification_results = {}
        
        # 初始化UI
        self._init_ui()
        
        self.logger.info("分類器分頁初始化完成")
    
    def _init_ui(self):
        """初始化用戶界面"""
        # 主佈局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 創建分頁標籤
        self.tab_widget = QTabWidget()
        
        # 資料準備分頁
        self.data_tab = self._create_data_tab()
        self.tab_widget.addTab(self.data_tab, "資料準備")
        
        # 模型訓練分頁
        self.training_tab = self._create_training_tab()
        self.tab_widget.addTab(self.training_tab, "模型訓練")
        
        # 標籤比較結果分頁
        self.comparison_tab = self._create_comparison_tab()
        self.tab_widget.addTab(self.comparison_tab, "標籤比較結果")
        
        # 結果分析分頁
        self.results_tab = self._create_results_tab()
        self.tab_widget.addTab(self.results_tab, "結果分析")
        
        main_layout.addWidget(self.tab_widget)
        
        # 進度條和狀態
        progress_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就緒")
        self.status_label.setStyleSheet("QLabel { color: green; }")
        progress_layout.addWidget(self.status_label)
        
        main_layout.addLayout(progress_layout)
    
    def _create_data_tab(self):
        """創建資料準備分頁"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 創建三個數據集的分頁
        dataset_tabs = QTabWidget()
        
        # IMDB 數據集分頁
        imdb_tab = self._create_dataset_tab("IMDB", "sentiment")
        dataset_tabs.addTab(imdb_tab, "IMDB")
        
        # Amazon 數據集分頁
        amazon_tab = self._create_dataset_tab("AMAZON", "overall")
        dataset_tabs.addTab(amazon_tab, "Amazon")
        
        # Yelp 數據集分頁
        yelp_tab = self._create_dataset_tab("YELP", "stars")
        dataset_tabs.addTab(yelp_tab, "Yelp")
        
        layout.addWidget(dataset_tabs)
        
        # 共用的資料資訊顯示區域
        info_group = QGroupBox("資料資訊")
        info_layout = QVBoxLayout(info_group)
        
        self.data_info_text = QTextEdit()
        self.data_info_text.setReadOnly(True)
        self.data_info_text.setMaximumHeight(200)
        info_layout.addWidget(self.data_info_text)
        
        layout.addWidget(info_group)
        
        return widget
    
    def _create_dataset_tab(self, dataset_name: str, label_column: str):
        """創建數據集特定的分頁"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 檔案選擇組
        file_group = QGroupBox("檔案選擇")
        file_layout = QGridLayout(file_group)
        
        # 面向向量檔案
        file_layout.addWidget(QLabel("面向向量檔案:"), 0, 0)
        aspect_vectors_edit = QLineEdit()
        aspect_vectors_edit.setObjectName(f"{dataset_name.lower()}_aspect_vectors_edit")
        aspect_vectors_edit.setPlaceholderText("選擇面向向量結果檔案 (.json)")
        file_layout.addWidget(aspect_vectors_edit, 0, 1)
        
        browse_aspect_btn = QPushButton("瀏覽...")
        browse_aspect_btn.setObjectName(f"{dataset_name.lower()}_browse_aspect_btn")
        browse_aspect_btn.clicked.connect(lambda: self._browse_aspect_vectors_file(dataset_name))
        file_layout.addWidget(browse_aspect_btn, 0, 2)
        
        # 標籤資料檔案
        file_layout.addWidget(QLabel("標籤資料檔案:"), 1, 0)
        labels_edit = QLineEdit()
        labels_edit.setObjectName(f"{dataset_name.lower()}_labels_edit")
        labels_edit.setPlaceholderText(f"選擇包含{label_column}的資料檔案 (.csv/.json)")
        file_layout.addWidget(labels_edit, 1, 1)
        
        browse_labels_btn = QPushButton("瀏覽...")
        browse_labels_btn.setObjectName(f"{dataset_name.lower()}_browse_labels_btn")
        browse_labels_btn.clicked.connect(lambda: self._browse_labels_file(dataset_name))
        file_layout.addWidget(browse_labels_btn, 1, 2)
        
        layout.addWidget(file_group)
        
        # 資料設定組
        data_group = QGroupBox("資料設定")
        data_layout = QGridLayout(data_group)
        
        # 標籤欄位名稱（唯讀）
        data_layout.addWidget(QLabel("標籤欄位名稱:"), 0, 0)
        label_column_edit = QLineEdit(label_column)
        label_column_edit.setObjectName(f"{dataset_name.lower()}_label_column_edit")
        label_column_edit.setReadOnly(True)
        data_layout.addWidget(label_column_edit, 0, 1)
        
        # 測試集比例
        data_layout.addWidget(QLabel("測試集比例:"), 1, 0)
        test_size_spin = QDoubleSpinBox()
        test_size_spin.setObjectName(f"{dataset_name.lower()}_test_size_spin")
        test_size_spin.setRange(0.1, 0.5)
        test_size_spin.setValue(0.2)
        test_size_spin.setSingleStep(0.05)
        data_layout.addWidget(test_size_spin, 1, 1)
        
        # 隨機種子
        data_layout.addWidget(QLabel("隨機種子:"), 2, 0)
        random_seed_spin = QSpinBox()
        random_seed_spin.setObjectName(f"{dataset_name.lower()}_random_seed_spin")
        random_seed_spin.setRange(0, 1000)
        random_seed_spin.setValue(42)
        data_layout.addWidget(random_seed_spin, 2, 1)
        
        layout.addWidget(data_group)
        
        # 載入按鈕
        load_btn = QPushButton("載入資料")
        load_btn.setObjectName(f"{dataset_name.lower()}_load_btn")
        load_btn.clicked.connect(lambda: self._load_data(dataset_name))
        layout.addWidget(load_btn)
        
        # 保存控件引用
        setattr(self, f"{dataset_name.lower()}_aspect_vectors_edit", aspect_vectors_edit)
        setattr(self, f"{dataset_name.lower()}_labels_edit", labels_edit)
        setattr(self, f"{dataset_name.lower()}_label_column_edit", label_column_edit)
        setattr(self, f"{dataset_name.lower()}_test_size_spin", test_size_spin)
        setattr(self, f"{dataset_name.lower()}_random_seed_spin", random_seed_spin)
        setattr(self, f"{dataset_name.lower()}_load_btn", load_btn)
        
        layout.addStretch()
        return widget
    
    def _browse_aspect_vectors_file(self, dataset_name: str):
        """瀏覽面向向量檔案"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"選擇 {dataset_name} 面向向量檔案", 
            "", "JSON files (*.json)"
        )
        if file_path:
            edit = getattr(self, f"{dataset_name.lower()}_aspect_vectors_edit")
            edit.setText(file_path)
    
    def _browse_labels_file(self, dataset_name: str):
        """瀏覽標籤資料檔案"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"選擇 {dataset_name} 標籤資料檔案", 
            "", "Data files (*.csv *.json)"
        )
        if file_path:
            edit = getattr(self, f"{dataset_name.lower()}_labels_edit")
            edit.setText(file_path)
    
    def _load_data(self, dataset_name: str):
        """載入資料"""
        try:
            # 獲取相關控件
            aspect_vectors_edit = getattr(self, f"{dataset_name.lower()}_aspect_vectors_edit")
            labels_edit = getattr(self, f"{dataset_name.lower()}_labels_edit")
            label_column_edit = getattr(self, f"{dataset_name.lower()}_label_column_edit")
            test_size_spin = getattr(self, f"{dataset_name.lower()}_test_size_spin")
            random_seed_spin = getattr(self, f"{dataset_name.lower()}_random_seed_spin")
            
            # 檢查檔案路徑
            aspect_vectors_file = aspect_vectors_edit.text().strip()
            labels_file = labels_edit.text().strip()
            
            if not aspect_vectors_file:
                QMessageBox.warning(self, "警告", f"請選擇 {dataset_name} 面向向量檔案")
                return
            
            if not os.path.exists(aspect_vectors_file):
                QMessageBox.warning(self, "警告", f"{dataset_name} 面向向量檔案不存在")
                return
            
            # 初始化分類器
            if self.config:
                output_dir = self.config.get("paths", "output_dir")
            else:
                # 獲取當前檔案所在的Part04_目錄
                current_dir = os.path.dirname(os.path.abspath(__file__))
                part04_dir = os.path.dirname(current_dir)
                output_dir = os.path.join(part04_dir, "1_output")
            self.classifier = SentimentClassifier(self.config, output_dir)
            
            # 載入面向向量資料
            features, labels = self.classifier.load_aspect_vectors_data(aspect_vectors_file)
            
            # 如果有標籤檔案，載入真實標籤
            if labels_file and os.path.exists(labels_file):
                try:
                    true_labels = self.classifier.load_labeled_data(
                        labels_file,
                        source=dataset_name
                    )
                    if len(true_labels) == len(features):
                        labels = true_labels
                        self.logger.info(f"使用 {dataset_name} 真實標籤資料")
                    else:
                        self.logger.warning(f"{dataset_name} 標籤數量與特徵數量不匹配，使用推斷標籤")
                except Exception as e:
                    self.logger.warning(f"載入 {dataset_name} 標籤檔案失敗，使用推斷標籤: {str(e)}")
            
            # 準備資料
            test_size = test_size_spin.value()
            random_state = random_seed_spin.value()
            
            self.classifier.prepare_data(
                features,
                labels,
                source=dataset_name
            )
            
            # 顯示資料資訊
            info = f"""{dataset_name} 資料載入成功！

特徵維度: {features.shape[1]}
總樣本數: {len(features)}
訓練集樣本數: {len(self.classifier.X_train)}
測試集樣本數: {len(self.classifier.X_test)}

標籤分佈:
- negative: {sum(labels == 0)} 樣本
- positive: {sum(labels == 1)} 樣本

測試集比例: {test_size:.1%}
隨機種子: {random_state}
"""
            
            self.data_info_text.setPlainText(info)
            
            # 啟用訓練按鈕
            self.train_btn.setEnabled(True)
            self.cv_btn.setEnabled(True)
            
            self.status_message.emit(f"{dataset_name} 資料載入完成", 3000)
            
        except Exception as e:
            self.logger.error(f"載入 {dataset_name} 資料失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"載入 {dataset_name} 資料失敗:\n{str(e)}")
    
    def _train_models(self):
        """訓練模型"""
        if not self.classifier:
            QMessageBox.warning(self, "警告", "請先載入資料")
            return
        
        # 開始訓練
        self._start_operation("train_and_evaluate")
    
    def _cross_validation(self):
        """執行交叉驗證"""
        if not self.classifier:
            QMessageBox.warning(self, "警告", "請先載入資料")
            return
        
        cv_folds = self.cv_folds_spin.value()
        self._start_operation("cross_validation", cv_folds=cv_folds)
    
    def _hyperparameter_tuning(self):
        """執行超參數調優"""
        if not self.classifier or not self.classifier.best_model_name:
            QMessageBox.warning(self, "警告", "請先訓練模型")
            return
        
        self._start_operation("hyperparameter_tuning")
    
    def _start_operation(self, operation, **kwargs):
        """開始執行操作"""
        # 創建工作線程
        self.worker = ClassifierWorker(self.classifier, operation, **kwargs)
        
        # 連接信號
        self.worker.progress_updated.connect(self._update_progress)
        self.worker.status_message.connect(self._update_status)
        self.worker.error_occurred.connect(self._handle_error)
        self.worker.finished.connect(self._operation_finished)
        
        # 禁用按鈕
        self._set_buttons_enabled(False)
        
        # 顯示進度條
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 開始執行
        self.worker.start()
    
    def _update_progress(self, value, message):
        """更新進度"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
        # 更新訓練日誌
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] {message}")
    
    def _update_status(self, message):
        """更新狀態"""
        self.status_label.setText(message)
        
    def _handle_error(self, error_message):
        """處理錯誤"""
        self.status_label.setText("發生錯誤")
        self.status_label.setStyleSheet("QLabel { color: red; }")
        
        # 更新訓練日誌
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_log.append(f"[{timestamp}] 錯誤: {error_message}")
        
        QMessageBox.critical(self, "錯誤", f"操作失敗:\n{error_message}")
        
        # 恢復按鈕狀態
        self._operation_finished({})
    
    def _operation_finished(self, result):
        """操作完成"""
        # 隱藏進度條
        self.progress_bar.setVisible(False)
        
        # 恢復按鈕狀態
        self._set_buttons_enabled(True)
        
        # 啟用儲存按鈕
        self.save_btn.setEnabled(True)
        
        # 更新結果顯示
        if result.get("operation") == "train_and_evaluate":
            self._update_results_display()
        
        # 重設狀態標籤
        self.status_label.setText("就緒")
        self.status_label.setStyleSheet("QLabel { color: green; }")
        
        self.status_message.emit("操作完成", 3000)
    
    def _set_buttons_enabled(self, enabled):
        """設定按鈕啟用狀態"""
        self.train_btn.setEnabled(enabled and self.classifier is not None)
        self.cv_btn.setEnabled(enabled and self.classifier is not None)
        self.tune_btn.setEnabled(enabled and self.classifier is not None and hasattr(self.classifier, 'best_model_name'))
        
        # 啟用比較按鈕（需要有分類器且已完成訓練）
        has_trained_model = (self.classifier is not None and 
                            hasattr(self.classifier, 'best_model') and 
                            self.classifier.best_model is not None)
        self.compare_button.setEnabled(enabled and has_trained_model)
    
    def _update_results_display(self):
        """更新結果顯示"""
        if not self.classifier or not self.classifier.evaluation_results:
            return
        
        # 更新性能表格
        self._update_performance_table()
        
        # 更新詳細結果
        self._update_results_text()
    
    def _update_performance_table(self):
        """更新性能表格"""
        results = self.classifier.evaluation_results
        
        if not results:
            return
        
        # 設定表格
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        self.performance_table.setRowCount(len(models))
        self.performance_table.setColumnCount(len(metrics))
        self.performance_table.setHorizontalHeaderLabels(['準確率', '精確率', '召回率', 'F1分數'])
        self.performance_table.setVerticalHeaderLabels(models)
        
        # 填充資料
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                value = results[model].get(metric, 0)
                item = QTableWidgetItem(f"{value:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                
                # 最佳模型高亮顯示
                if model == self.classifier.best_model_name:
                    item.setBackground(Qt.lightGray)
                
                self.performance_table.setItem(i, j, item)
        
        # 調整列寬
        self.performance_table.resizeColumnsToContents()
        self.performance_table.horizontalHeader().setStretchLastSection(True)
    
    def _update_results_text(self):
        """更新詳細結果文字"""
        if not self.classifier:
            return
        
        text_parts = []
        
        # 最佳模型資訊
        if self.classifier.best_model_name:
            text_parts.append(f"最佳模型: {self.classifier.best_model_name}")
            text_parts.append("")
        
        # 各模型詳細結果
        for model_name, results in self.classifier.evaluation_results.items():
            text_parts.append(f"=== {model_name} ===")
            text_parts.append(f"準確率: {results['accuracy']:.4f}")
            text_parts.append(f"精確率: {results['precision']:.4f}")
            text_parts.append(f"召回率: {results['recall']:.4f}")
            text_parts.append(f"F1分數: {results['f1_score']:.4f}")
            
            if results.get('auc_score'):
                text_parts.append(f"AUC分數: {results['auc_score']:.4f}")
            
            text_parts.append("")
        
        # 交叉驗證結果
        if self.classifier.cross_validation_results:
            text_parts.append("=== 交叉驗證結果 ===")
            for model_name, cv_results in self.classifier.cross_validation_results.items():
                mean_score = cv_results['mean_score']
                std_score = cv_results['std_score']
                text_parts.append(f"{model_name}: {mean_score:.4f} ± {std_score:.4f}")
            text_parts.append("")
        
        self.results_text.setPlainText("\n".join(text_parts))
    
    def _save_results(self):
        """儲存結果"""
        if not self.classifier:
            QMessageBox.warning(self, "警告", "沒有可儲存的結果")
            return
        
        try:
            # 儲存結果檔案
            result_file = self.classifier.save_results()
            
            # 顯示成功訊息
            QMessageBox.information(self, "成功", f"結果已儲存至:\n{result_file}")
            
            self.status_message.emit("結果已儲存", 3000)
            
        except Exception as e:
            self.logger.error(f"儲存結果失敗: {str(e)}")
            QMessageBox.critical(self, "錯誤", f"儲存結果失敗:\n{str(e)}")

    def _create_training_tab(self):
        """創建模型訓練分頁"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 模型選擇組
        model_group = QGroupBox("模型選擇")
        model_layout = QGridLayout(model_group)
        
        # 啟用的模型
        self.model_checkboxes = {}
        models = ["RandomForest", "SVM", "LogisticRegression", "GradientBoosting", "MLP"]
        
        for i, model in enumerate(models):
            checkbox = QCheckBox(model)
            checkbox.setChecked(True)
            self.model_checkboxes[model] = checkbox
            model_layout.addWidget(checkbox, i // 2, i % 2)
        
        layout.addWidget(model_group)
        
        # 交叉驗證設定
        cv_group = QGroupBox("交叉驗證設定")
        cv_layout = QGridLayout(cv_group)
        
        cv_layout.addWidget(QLabel("交叉驗證折數:"), 0, 0)
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(3, 10)
        self.cv_folds_spin.setValue(5)
        cv_layout.addWidget(self.cv_folds_spin, 0, 1)
        
        layout.addWidget(cv_group)
        
        # 操作按鈕
        button_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("訓練模型")
        self.train_btn.clicked.connect(self._train_models)
        self.train_btn.setEnabled(False)
        button_layout.addWidget(self.train_btn)
        
        self.cv_btn = QPushButton("交叉驗證")
        self.cv_btn.clicked.connect(self._cross_validation)
        self.cv_btn.setEnabled(False)
        button_layout.addWidget(self.cv_btn)
        
        self.tune_btn = QPushButton("超參數調優")
        self.tune_btn.clicked.connect(self._hyperparameter_tuning)
        self.tune_btn.setEnabled(False)
        button_layout.addWidget(self.tune_btn)
        
        button_layout.addStretch()
        
        self.save_btn = QPushButton("儲存結果")
        self.save_btn.clicked.connect(self._save_results)
        self.save_btn.setEnabled(False)
        button_layout.addWidget(self.save_btn)
        
        layout.addLayout(button_layout)
        
        # 訓練日誌
        log_group = QGroupBox("訓練日誌")
        log_layout = QVBoxLayout(log_group)
        
        self.training_log = QTextEdit()
        self.training_log.setMaximumHeight(200)
        log_layout.addWidget(self.training_log)
        
        layout.addWidget(log_group)
        
        layout.addStretch()
        return widget
    
    def _create_comparison_tab(self):
        """創建標籤比較結果分頁"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 控制面板
        control_group = QGroupBox("比較控制")
        control_layout = QHBoxLayout(control_group)
        
        # 比較按鈕
        self.compare_button = QPushButton("比較原始和新標籤")
        self.compare_button.clicked.connect(self._compare_labels)
        self.compare_button.setEnabled(False)
        control_layout.addWidget(self.compare_button)
        
        # 最大樣本數設定
        control_layout.addWidget(QLabel("最大顯示樣本數:"))
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(10, 1000)
        self.max_samples_spin.setValue(100)
        control_layout.addWidget(self.max_samples_spin)
        
        control_layout.addStretch()
        
        # 儲存比較結果按鈕
        self.save_comparison_btn = QPushButton("儲存比較結果")
        self.save_comparison_btn.clicked.connect(self._save_comparison_results)
        self.save_comparison_btn.setEnabled(False)
        control_layout.addWidget(self.save_comparison_btn)
        
        layout.addWidget(control_group)
        
        # 比較統計資訊
        stats_group = QGroupBox("比較統計")
        stats_layout = QGridLayout(stats_group)
        
        stats_layout.addWidget(QLabel("總樣本數:"), 0, 0)
        self.total_samples_label = QLabel("0")
        stats_layout.addWidget(self.total_samples_label, 0, 1)
        
        stats_layout.addWidget(QLabel("變化數量:"), 0, 2)
        self.changed_count_label = QLabel("0")
        stats_layout.addWidget(self.changed_count_label, 0, 3)
        
        stats_layout.addWidget(QLabel("變化率:"), 1, 0)
        self.change_rate_label = QLabel("0.0%")
        stats_layout.addWidget(self.change_rate_label, 1, 1)
        
        stats_layout.addWidget(QLabel("一致準確率:"), 1, 2)
        self.accuracy_label = QLabel("0.0%")
        stats_layout.addWidget(self.accuracy_label, 1, 3)
        
        layout.addWidget(stats_group)
        
        # 比較結果表格
        table_group = QGroupBox("詳細比較結果")
        table_layout = QVBoxLayout(table_group)
        
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(4)
        self.comparison_table.setHorizontalHeaderLabels([
            "索引", "評論文字", "原始情感", "新情感"
        ])
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.setSortingEnabled(True)
        table_layout.addWidget(self.comparison_table)
        
        layout.addWidget(table_group)
        
        # 說明文字
        info_text = QLabel(
            "說明：黃色高亮顯示的行表示情感標籤發生變化的樣本。\n"
            "您可以點擊表格標題進行排序，以便更好地分析結果。"
        )
        info_text.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        layout.addWidget(info_text)
        
        return widget
    
    def _create_results_tab(self):
        """創建結果分析分頁"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左側：模型性能表格
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        left_layout.addWidget(QLabel("模型性能比較"))
        
        self.performance_table = QTableWidget()
        self.performance_table.setAlternatingRowColors(True)
        left_layout.addWidget(self.performance_table)
        
        splitter.addWidget(left_widget)
        
        # 右側：詳細結果
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("詳細結果"))
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        right_layout.addWidget(self.results_text)
        
        splitter.addWidget(right_widget)
        
        # 設定分割比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        return widget
    
    def _compare_labels(self):
        """比較原始和新標籤"""
        try:
            if not hasattr(self, 'classifier') or not self.classifier:
                QMessageBox.warning(self, "警告", "請先載入資料並訓練模型")
                return
            
            # 獲取最大樣本數設定
            max_samples = self.max_samples_spin.value()
            
            # 獲取原始標籤
            original_labels = self.classifier.y_test
            
            # 使用最佳模型預測新標籤
            new_predictions = self.classifier.predict_sentiment(self.classifier.X_test)
            new_labels = np.array(new_predictions['predictions'])
            
            # 比較標籤
            comparison_results = self.classifier.compare_sentiment_labels(
                original_labels, new_labels, max_samples=max_samples
            )
            
            # 保存比較結果
            self.comparison_results = comparison_results
            
            # 更新統計資訊
            stats = comparison_results['statistics']
            self.total_samples_label.setText(str(stats['total_samples']))
            self.changed_count_label.setText(str(stats['changed_count']))
            self.change_rate_label.setText(f"{stats['change_rate']:.2%}")
            
            # 計算準確率 (一致率)
            accuracy = 1 - stats['change_rate']
            self.accuracy_label.setText(f"{accuracy:.2%}")
            
            # 更新表格
            results = comparison_results['comparison_results']
            self.comparison_table.setRowCount(len(results))
            
            for i, result in enumerate(results):
                self.comparison_table.setItem(i, 0, QTableWidgetItem(str(result['index'])))
                self.comparison_table.setItem(i, 1, QTableWidgetItem(result['text']))
                self.comparison_table.setItem(i, 2, QTableWidgetItem(result['original_sentiment']))
                self.comparison_table.setItem(i, 3, QTableWidgetItem(result['new_sentiment']))
                
                # 如果情感標籤發生變化，將整行標記為黃色
                if result['is_changed']:
                    for j in range(4):
                        item = self.comparison_table.item(i, j)
                        if item:
                            item.setBackground(Qt.yellow)
            
            # 調整列寬
            self.comparison_table.resizeColumnsToContents()
            
            # 啟用儲存按鈕
            self.save_comparison_btn.setEnabled(True)
            
            # 顯示視覺化結果
            self.classifier.visualize_comparison(comparison_results)
            
            # 切換到標籤比較結果分頁
            self.tab_widget.setCurrentIndex(2)  # 標籤比較結果分頁索引
            
            # 顯示完成訊息
            self.status_message.emit(f"標籤比較完成 - 變化率: {stats['change_rate']:.2%}", 5000)
            
        except Exception as e:
            error_msg = f"比較標籤時發生錯誤：{str(e)}"
            QMessageBox.critical(self, "錯誤", error_msg)
            self.logger.error(error_msg) 
    
    def _save_comparison_results(self):
        """儲存比較結果"""
        if not hasattr(self, 'comparison_results') or not self.comparison_results:
            QMessageBox.warning(self, "警告", "沒有可儲存的比較結果")
            return
        
        try:
            # 選擇儲存路徑
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"sentiment_comparison_result_{timestamp}.json"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "儲存比較結果", 
                default_filename, 
                "JSON files (*.json);;All files (*.*)"
            )
            
            if not file_path:
                return
            
            # 深拷貝並轉換數據以確保JSON序列化兼容性
            def convert_for_json(obj):
                """遞歸轉換對象為JSON兼容格式"""
                if isinstance(obj, dict):
                    return {key: convert_for_json(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                elif isinstance(obj, bool):
                    return obj  # JSON原生支持布林值
                elif isinstance(obj, (int, float, str, type(None))):
                    return obj
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif hasattr(obj, 'tolist'):  # numpy arrays
                    return obj.tolist()
                else:
                    return str(obj)  # 其他類型轉為字符串
            
            # 準備儲存資料並轉換為JSON兼容格式
            save_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'description': '情感標籤比較結果',
                'parameters': {
                    'max_samples': self.max_samples_spin.value(),
                    'model_used': self.classifier.best_model_name if self.classifier.best_model_name else 'Unknown'
                },
                'statistics': convert_for_json(self.comparison_results['statistics']),
                'comparison_results': convert_for_json(self.comparison_results['comparison_results'])
            }
            
            # 儲存到檔案
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "成功", f"比較結果已儲存至:\n{file_path}")
            self.status_message.emit("比較結果已儲存", 3000)
            
        except Exception as e:
            error_msg = f"儲存比較結果失敗：{str(e)}"
            QMessageBox.critical(self, "錯誤", error_msg)
            self.logger.error(error_msg)
            # 記錄更詳細的錯誤信息以便調試
            import traceback
            self.logger.error(f"詳細錯誤追蹤: {traceback.format_exc()}")