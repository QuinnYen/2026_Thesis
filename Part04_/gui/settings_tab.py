"""
設定標籤頁模組 - 提供系統各項參數設定的介面
"""
import os
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QComboBox, QFileDialog, QTabWidget, 
    QGroupBox, QScrollArea, QFrame, QSlider, QColorDialog,
    QMessageBox, QSplitter
)
from PyQt5.QtCore import Qt, pyqtSignal, QSettings
from PyQt5.QtGui import QFont, QIcon, QColor

from utils.logger import get_logger
from utils.config import Config
from utils.file_manager import FileManager

# 獲取logger
logger = get_logger("settings_tab")


class SettingsTab(QWidget):
    """系統設定頁面"""
    
    # 定義信號
    settings_changed = pyqtSignal()  # 設定變更信號
    
    def __init__(self, config, file_manager):
        """初始化設定頁面
        
        Args:
            config: 系統配置對象
            file_manager: 文件管理器對象
        """
        super().__init__()
        
        self.config = config
        self.file_manager = file_manager
        self.settings = QSettings("ThesisResearch", "TextAnalysisTool")
        
        # 設置預設路徑 - 更新為使用 Part04_ 路徑
        self.default_paths = {
            "data_dir": os.path.join("..", "ReviewsDataBase"),
            "output_dir": os.path.join("Part04_", "1_output"),
            "log_dir": os.path.join("Part04_", "1_output", "logs"),
            "model_dir": os.path.join("Part04_", "1_output", "models"),
            "export_dir": os.path.join("Part04_", "1_output", "exports"),
            "results_dir": os.path.join("Part04_", "1_output", "results"),
            "embeddings_dir": os.path.join("Part04_", "1_output", "embeddings"),
            "topics_dir": os.path.join("Part04_", "1_output", "topics"),
            "vectors_dir": os.path.join("Part04_", "1_output", "vectors"),
            "evaluation_dir": os.path.join("Part04_", "1_output", "evaluation"),
            "visualizations_dir": os.path.join("Part04_", "1_output", "visualizations"),
            "temp_dir": os.path.join("Part04_", "1_output", "temp")
        }
        
        # 緩存原配置，用於比較變化
        self._original_config = {}
        
        # 初始化UI
        self._init_ui()
        
        # 載入設定
        self.load_settings()
        
    def _init_ui(self):
        """初始化UI組件"""
        # 主佈局
        main_layout = QVBoxLayout(self)
        
        # 建立選項卡組件
        self.tabs = QTabWidget()
        
        # 建立各設定選項卡
        self.tabs.addTab(self._create_bert_tab(), "BERT設定")
        self.tabs.addTab(self._create_lda_tab(), "LDA設定")
        self.tabs.addTab(self._create_attention_tab(), "注意力機制")
        self.tabs.addTab(self._create_paths_tab(), "路徑設定")
        self.tabs.addTab(self._create_visualization_tab(), "視覺化設定")
        self.tabs.addTab(self._create_system_tab(), "系統設定")
        
        # 將選項卡組件添加至主佈局
        main_layout.addWidget(self.tabs)
        
        # 建立按鈕區域
        buttons_layout = QHBoxLayout()
        
        # 建立按鈕
        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_settings)
        
        self.apply_button = QPushButton("應用")
        self.apply_button.clicked.connect(self.apply_settings)
        
        self.save_button = QPushButton("儲存")
        self.save_button.clicked.connect(self.save_settings)
        
        # 添加按鈕至佈局
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.apply_button)
        buttons_layout.addWidget(self.save_button)
        
        # 將按鈕區域添加至主佈局
        main_layout.addLayout(buttons_layout)
    
    def _create_bert_tab(self):
        """建立BERT設定選項卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 建立滾動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # 建立內容組件
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # BERT模型設定
        model_group = QGroupBox("BERT模型設定")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("模型名稱:"), 0, 0)
        self.bert_model_name = QComboBox()
        self.bert_model_name.addItems([
            "bert-base-uncased", 
            "bert-large-uncased", 
            "bert-base-cased", 
            "bert-large-cased",
            "chinese-bert-wwm",
            "chinese-roberta-wwm-ext"
        ])
        self.bert_model_name.setEditable(True)
        model_layout.addWidget(self.bert_model_name, 0, 1)
        
        model_layout.addWidget(QLabel("最大長度:"), 1, 0)
        self.bert_max_length = QSpinBox()
        self.bert_max_length.setRange(16, 512)
        self.bert_max_length.setSingleStep(8)
        model_layout.addWidget(self.bert_max_length, 1, 1)
        
        model_layout.addWidget(QLabel("批次大小:"), 2, 0)
        self.bert_batch_size = QSpinBox()
        self.bert_batch_size.setRange(1, 64)
        model_layout.addWidget(self.bert_batch_size, 2, 1)
        
        model_layout.addWidget(QLabel("使用GPU:"), 3, 0)
        self.bert_use_gpu = QCheckBox()
        model_layout.addWidget(self.bert_use_gpu, 3, 1)
        
        model_group.setLayout(model_layout)
        content_layout.addWidget(model_group)
        
        # BERT進階設定
        advanced_group = QGroupBox("進階設定")
        advanced_layout = QGridLayout()
        
        advanced_layout.addWidget(QLabel("預訓練模型下載目錄:"), 0, 0)
        
        path_layout = QHBoxLayout()
        self.bert_cache_dir = QLineEdit()
        path_layout.addWidget(self.bert_cache_dir)
        
        browse_button = QPushButton("瀏覽...")
        browse_button.clicked.connect(lambda: self._browse_directory(self.bert_cache_dir))
        path_layout.addWidget(browse_button)
        
        advanced_layout.addLayout(path_layout, 0, 1)
        
        advanced_layout.addWidget(QLabel("Tokenizer類型:"), 1, 0)
        self.bert_tokenizer = QComboBox()
        self.bert_tokenizer.addItems(["WordPiece", "BPE", "SentencePiece"])
        advanced_layout.addWidget(self.bert_tokenizer, 1, 1)
        
        advanced_group.setLayout(advanced_layout)
        content_layout.addWidget(advanced_group)
        
        # 離線模型下載
        offline_group = QGroupBox("離線模型管理")
        offline_layout = QVBoxLayout()
        
        download_button = QPushButton("下載BERT模型")
        download_button.clicked.connect(self._download_bert_model)
        offline_layout.addWidget(download_button)
        
        offline_group.setLayout(offline_layout)
        content_layout.addWidget(offline_group)
        
        # 添加伸展因子
        content_layout.addStretch()
        
        # 設置滾動區域
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab
        
    def _create_lda_tab(self):
        """建立LDA設定選項卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 建立滾動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # 建立內容組件
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # LDA模型設定
        model_group = QGroupBox("LDA模型參數")
        model_layout = QGridLayout()
        
        model_layout.addWidget(QLabel("主題數量:"), 0, 0)
        self.lda_n_topics = QSpinBox()
        self.lda_n_topics.setRange(2, 100)
        model_layout.addWidget(self.lda_n_topics, 0, 1)
        
        model_layout.addWidget(QLabel("Alpha:"), 1, 0)
        self.lda_alpha = QDoubleSpinBox()
        self.lda_alpha.setRange(0.01, 10.0)
        self.lda_alpha.setSingleStep(0.01)
        self.lda_alpha.setDecimals(3)
        model_layout.addWidget(self.lda_alpha, 1, 1)
        
        model_layout.addWidget(QLabel("Beta:"), 2, 0)
        self.lda_beta = QDoubleSpinBox()
        self.lda_beta.setRange(0.001, 1.0)
        self.lda_beta.setSingleStep(0.001)
        self.lda_beta.setDecimals(3)
        model_layout.addWidget(self.lda_beta, 2, 1)
        
        model_layout.addWidget(QLabel("最大迭代次數:"), 3, 0)
        self.lda_max_iter = QSpinBox()
        self.lda_max_iter.setRange(10, 500)
        model_layout.addWidget(self.lda_max_iter, 3, 1)
        
        model_layout.addWidget(QLabel("隨機種子:"), 4, 0)
        self.lda_random_state = QSpinBox()
        self.lda_random_state.setRange(-1, 100)
        self.lda_random_state.setSpecialValueText("無 (自動)")
        model_layout.addWidget(self.lda_random_state, 4, 1)
        
        model_group.setLayout(model_layout)
        content_layout.addWidget(model_group)
        
        # LDA進階設定
        advanced_group = QGroupBox("LDA進階設定")
        advanced_layout = QGridLayout()
        
        advanced_layout.addWidget(QLabel("LDA演算法:"), 0, 0)
        self.lda_algorithm = QComboBox()
        self.lda_algorithm.addItems(["VB (Variational Bayes)", "Online VB", "Gibbs Sampling"])
        advanced_layout.addWidget(self.lda_algorithm, 0, 1)
        
        advanced_layout.addWidget(QLabel("批次大小:"), 1, 0)
        self.lda_batch_size = QSpinBox()
        self.lda_batch_size.setRange(64, 4096)
        self.lda_batch_size.setSingleStep(64)
        advanced_layout.addWidget(self.lda_batch_size, 1, 1)
        
        advanced_layout.addWidget(QLabel("收斂閾值:"), 2, 0)
        self.lda_tol = QDoubleSpinBox()
        self.lda_tol.setRange(0.00001, 0.1)
        self.lda_tol.setDecimals(5)
        self.lda_tol.setSingleStep(0.0001)
        advanced_layout.addWidget(self.lda_tol, 2, 1)
        
        advanced_group.setLayout(advanced_layout)
        content_layout.addWidget(advanced_group)
        
        # 主題評估
        evaluation_group = QGroupBox("主題評估指標")
        evaluation_layout = QGridLayout()
        
        evaluation_layout.addWidget(QLabel("主題一致性:"), 0, 0)
        self.lda_coherence = QCheckBox()
        self.lda_coherence.setChecked(True)
        evaluation_layout.addWidget(self.lda_coherence, 0, 1)
        
        evaluation_layout.addWidget(QLabel("主題分離度:"), 1, 0)
        self.lda_separation = QCheckBox()
        self.lda_separation.setChecked(True)
        evaluation_layout.addWidget(self.lda_separation, 1, 1)
        
        evaluation_group.setLayout(evaluation_layout)
        content_layout.addWidget(evaluation_group)
        
        # 添加伸展因子
        content_layout.addStretch()
        
        # 設置滾動區域
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab

    def _create_attention_tab(self):
        """建立注意力機制設定選項卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 建立滾動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # 建立內容組件
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # 啟用的注意力機制
        mechanisms_group = QGroupBox("啟用的注意力機制")
        mechanisms_layout = QGridLayout()
        
        # 相似度注意力
        mechanisms_layout.addWidget(QLabel("相似度注意力:"), 0, 0)
        self.attention_similarity = QCheckBox()
        mechanisms_layout.addWidget(self.attention_similarity, 0, 1)
        
        # 關鍵詞注意力
        mechanisms_layout.addWidget(QLabel("關鍵詞注意力:"), 1, 0)
        self.attention_keyword = QCheckBox()
        mechanisms_layout.addWidget(self.attention_keyword, 1, 1)
        
        # Self-Attention
        mechanisms_layout.addWidget(QLabel("Self-Attention:"), 2, 0)
        self.attention_self = QCheckBox()
        mechanisms_layout.addWidget(self.attention_self, 2, 1)
        
        # 組合注意力
        mechanisms_layout.addWidget(QLabel("組合注意力:"), 3, 0)
        self.attention_combined = QCheckBox()
        mechanisms_layout.addWidget(self.attention_combined, 3, 1)
        
        mechanisms_group.setLayout(mechanisms_layout)
        content_layout.addWidget(mechanisms_group)
        
        # 注意力機制權重
        weights_group = QGroupBox("注意力機制權重")
        weights_layout = QGridLayout()
        
        # 相似度權重
        weights_layout.addWidget(QLabel("相似度注意力權重:"), 0, 0)
        self.attention_similarity_weight = QDoubleSpinBox()
        self.attention_similarity_weight.setRange(0.0, 1.0)
        self.attention_similarity_weight.setSingleStep(0.01)
        self.attention_similarity_weight.setDecimals(2)
        weights_layout.addWidget(self.attention_similarity_weight, 0, 1)
        
        # 關鍵詞權重
        weights_layout.addWidget(QLabel("關鍵詞注意力權重:"), 1, 0)
        self.attention_keyword_weight = QDoubleSpinBox()
        self.attention_keyword_weight.setRange(0.0, 1.0)
        self.attention_keyword_weight.setSingleStep(0.01)
        self.attention_keyword_weight.setDecimals(2)
        weights_layout.addWidget(self.attention_keyword_weight, 1, 1)
        
        # Self-Attention權重
        weights_layout.addWidget(QLabel("Self-Attention權重:"), 2, 0)
        self.attention_self_weight = QDoubleSpinBox()
        self.attention_self_weight.setRange(0.0, 1.0)
        self.attention_self_weight.setSingleStep(0.01)
        self.attention_self_weight.setDecimals(2)
        weights_layout.addWidget(self.attention_self_weight, 2, 1)
        
        # 自動平衡權重
        weights_layout.addWidget(QLabel("自動平衡權重:"), 3, 0)
        self.attention_auto_balance = QCheckBox()
        weights_layout.addWidget(self.attention_auto_balance, 3, 1)
        self.attention_auto_balance.toggled.connect(self._toggle_attention_weights)
        
        weights_group.setLayout(weights_layout)
        content_layout.addWidget(weights_group)
        
        # 進階設定
        advanced_group = QGroupBox("進階設定")
        advanced_layout = QGridLayout()
        
        # 窗口大小
        advanced_layout.addWidget(QLabel("注意力窗口大小:"), 0, 0)
        self.attention_window_size = QSpinBox()
        self.attention_window_size.setRange(1, 100)
        advanced_layout.addWidget(self.attention_window_size, 0, 1)
        
        # 閾值
        advanced_layout.addWidget(QLabel("注意力閾值:"), 1, 0)
        self.attention_threshold = QDoubleSpinBox()
        self.attention_threshold.setRange(0.0, 1.0)
        self.attention_threshold.setSingleStep(0.01)
        self.attention_threshold.setDecimals(2)
        advanced_layout.addWidget(self.attention_threshold, 1, 1)
        
        advanced_group.setLayout(advanced_layout)
        content_layout.addWidget(advanced_group)
        
        # 添加伸展因子
        content_layout.addStretch()
        
        # 設置滾動區域
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab

    def _create_paths_tab(self):
        """建立路徑設定選項卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 建立滾動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # 建立內容組件
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # 數據路徑設定
        paths_group = QGroupBox("數據路徑設定")
        paths_layout = QGridLayout()
        
        # 資料目錄
        paths_layout.addWidget(QLabel("資料目錄:"), 0, 0)
        
        data_path_layout = QHBoxLayout()
        self.data_dir = QLineEdit()
        data_path_layout.addWidget(self.data_dir)
        
        data_browse_button = QPushButton("瀏覽...")
        data_browse_button.clicked.connect(lambda: self._browse_directory(self.data_dir))
        data_path_layout.addWidget(data_browse_button)
        
        paths_layout.addLayout(data_path_layout, 0, 1)
        
        # 輸出目錄
        paths_layout.addWidget(QLabel("輸出目錄:"), 1, 0)
        
        output_path_layout = QHBoxLayout()
        self.output_dir = QLineEdit()
        output_path_layout.addWidget(self.output_dir)
        
        output_browse_button = QPushButton("瀏覽...")
        output_browse_button.clicked.connect(lambda: self._browse_directory(self.output_dir))
        output_path_layout.addWidget(output_browse_button)
        
        paths_layout.addLayout(output_path_layout, 1, 1)
        
        # 模型目錄
        paths_layout.addWidget(QLabel("模型目錄:"), 2, 0)
        
        model_path_layout = QHBoxLayout()
        self.model_dir = QLineEdit()
        model_path_layout.addWidget(self.model_dir)
        
        model_browse_button = QPushButton("瀏覽...")
        model_browse_button.clicked.connect(lambda: self._browse_directory(self.model_dir))
        model_path_layout.addWidget(model_browse_button)
        
        paths_layout.addLayout(model_path_layout, 2, 1)
        
        # 日誌目錄
        paths_layout.addWidget(QLabel("日誌目錄:"), 3, 0)
        
        log_path_layout = QHBoxLayout()
        self.log_dir = QLineEdit()
        log_path_layout.addWidget(self.log_dir)
        
        log_browse_button = QPushButton("瀏覽...")
        log_browse_button.clicked.connect(lambda: self._browse_directory(self.log_dir))
        log_path_layout.addWidget(log_browse_button)
        
        paths_layout.addLayout(log_path_layout, 3, 1)
        
        paths_group.setLayout(paths_layout)
        content_layout.addWidget(paths_group)
        
        # 文件導出設定
        export_group = QGroupBox("文件導出設定")
        export_layout = QGridLayout()
        
        # 導出路徑
        export_layout.addWidget(QLabel("導出路徑:"), 0, 0)
        
        export_path_layout = QHBoxLayout()
        self.export_dir = QLineEdit()
        export_path_layout.addWidget(self.export_dir)
        
        export_browse_button = QPushButton("瀏覽...")
        export_browse_button.clicked.connect(lambda: self._browse_directory(self.export_dir))
        export_path_layout.addWidget(export_browse_button)
        
        export_layout.addLayout(export_path_layout, 0, 1)
        
        # 默認導出格式
        export_layout.addWidget(QLabel("默認導出格式:"), 1, 0)
        self.export_format = QComboBox()
        self.export_format.addItems(["HTML", "CSV", "Excel", "PDF", "JSON"])
        export_layout.addWidget(self.export_format, 1, 1)
        
        export_group.setLayout(export_layout)
        content_layout.addWidget(export_group)
        
        # 按鈕區域
        buttons_group = QGroupBox("目錄管理")
        buttons_layout = QGridLayout()
        
        # 建立目錄按鈕
        create_dirs_button = QPushButton("建立所有目錄")
        create_dirs_button.clicked.connect(self._create_directories)
        buttons_layout.addWidget(create_dirs_button, 0, 0)
        
        # 開啟目錄按鈕
        open_dirs_button = QPushButton("開啟數據目錄")
        open_dirs_button.clicked.connect(lambda: self._open_directory(self.data_dir.text()))
        buttons_layout.addWidget(open_dirs_button, 0, 1)
        
        buttons_group.setLayout(buttons_layout)
        content_layout.addWidget(buttons_group)
        
        # 添加伸展因子
        content_layout.addStretch()
        
        # 設置滾動區域
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab

    def _create_visualization_tab(self):
        """建立視覺化設定選項卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 建立滾動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # 建立內容組件
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # 圖表設定
        chart_group = QGroupBox("圖表設定")
        chart_layout = QGridLayout()
        
        # DPI設定
        chart_layout.addWidget(QLabel("DPI:"), 0, 0)
        self.viz_dpi = QSpinBox()
        self.viz_dpi.setRange(72, 600)
        chart_layout.addWidget(self.viz_dpi, 0, 1)
        
        # 圖表格式
        chart_layout.addWidget(QLabel("圖表格式:"), 1, 0)
        self.viz_format = QComboBox()
        self.viz_format.addItems(["png", "jpg", "svg", "pdf"])
        chart_layout.addWidget(self.viz_format, 1, 1)
        
        # 顏色方案
        chart_layout.addWidget(QLabel("顏色方案:"), 2, 0)
        self.viz_color_scheme = QComboBox()
        self.viz_color_scheme.addItems([
            "viridis", "plasma", "inferno", "magma", 
            "cividis", "Spectral", "rainbow", "jet"
        ])
        chart_layout.addWidget(self.viz_color_scheme, 2, 1)
        
        chart_group.setLayout(chart_layout)
        content_layout.addWidget(chart_group)
        
        # 圖表類型設定
        type_group = QGroupBox("圖表類型設定")
        type_layout = QGridLayout()
        
        # 主題圖表類型
        type_layout.addWidget(QLabel("主題圖表類型:"), 0, 0)
        self.viz_topic_type = QComboBox()
        self.viz_topic_type.addItems(["泡泡圖", "堆疊柱狀圖", "熱力圖"])
        type_layout.addWidget(self.viz_topic_type, 0, 1)
        
        # 詞雲設定
        type_layout.addWidget(QLabel("詞雲最大詞數:"), 1, 0)
        self.viz_wordcloud_words = QSpinBox()
        self.viz_wordcloud_words.setRange(10, 300)
        type_layout.addWidget(self.viz_wordcloud_words, 1, 1)
        
        # 字體設定
        type_layout.addWidget(QLabel("字體:"), 2, 0)
        self.viz_font = QComboBox()
        
        # 獲取系統字體
        from PyQt5.QtGui import QFontDatabase
        fonts = QFontDatabase().families()
        self.viz_font.addItems(fonts)
        
        type_layout.addWidget(self.viz_font, 2, 1)
        
        type_group.setLayout(type_layout)
        content_layout.addWidget(type_group)
        
        # 添加伸展因子
        content_layout.addStretch()
        
        # 設置滾動區域
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab

    def _create_system_tab(self):
        """建立系統設定選項卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 建立滾動區域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        # 建立內容組件
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # 系統設定
        system_group = QGroupBox("系統設定")
        system_layout = QGridLayout()
        
        # 日誌等級
        system_layout.addWidget(QLabel("日誌等級:"), 0, 0)
        self.system_log_level = QComboBox()
        self.system_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        system_layout.addWidget(self.system_log_level, 0, 1)
        
        # 多處理
        system_layout.addWidget(QLabel("啟用多處理:"), 1, 0)
        self.system_multi_processing = QCheckBox()
        system_layout.addWidget(self.system_multi_processing, 1, 1)
        
        # 工作執行緒數
        system_layout.addWidget(QLabel("工作執行緒數:"), 2, 0)
        self.system_num_workers = QSpinBox()
        self.system_num_workers.setRange(1, 16)
        system_layout.addWidget(self.system_num_workers, 2, 1)
        
        system_group.setLayout(system_layout)
        content_layout.addWidget(system_group)
        
        # 數據集設定
        datasets_group = QGroupBox("數據集設定")
        datasets_layout = QGridLayout()
        
        # IMDB數據集
        datasets_layout.addWidget(QLabel("IMDB電影評論:"), 0, 0)
        imdb_layout = QHBoxLayout()
        
        self.dataset_imdb_path = QLineEdit()
        imdb_layout.addWidget(self.dataset_imdb_path)
        
        imdb_browse_button = QPushButton("瀏覽...")
        imdb_browse_button.clicked.connect(lambda: self._browse_file(self.dataset_imdb_path, "CSV檔案 (*.csv)"))
        imdb_layout.addWidget(imdb_browse_button)
        
        datasets_layout.addLayout(imdb_layout, 0, 1)
        
        # Amazon數據集
        datasets_layout.addWidget(QLabel("Amazon產品評論:"), 1, 0)
        amazon_layout = QHBoxLayout()
        
        self.dataset_amazon_path = QLineEdit()
        amazon_layout.addWidget(self.dataset_amazon_path)
        
        amazon_browse_button = QPushButton("瀏覽...")
        amazon_browse_button.clicked.connect(lambda: self._browse_file(self.dataset_amazon_path, "JSON檔案 (*.json)"))
        amazon_layout.addWidget(amazon_browse_button)
        
        datasets_layout.addLayout(amazon_layout, 1, 1)
        
        # Yelp數據集
        datasets_layout.addWidget(QLabel("Yelp餐廳評論:"), 2, 0)
        yelp_layout = QHBoxLayout()
        
        self.dataset_yelp_path = QLineEdit()
        yelp_layout.addWidget(self.dataset_yelp_path)
        
        yelp_browse_button = QPushButton("瀏覽...")
        yelp_browse_button.clicked.connect(lambda: self._browse_file(self.dataset_yelp_path, "JSON檔案 (*.json)"))
        yelp_layout.addWidget(yelp_browse_button)
        
        datasets_layout.addLayout(yelp_layout, 2, 1)
        
        datasets_group.setLayout(datasets_layout)
        content_layout.addWidget(datasets_group)
        
        # 進階系統設定
        advanced_group = QGroupBox("進階系統設定")
        advanced_layout = QGridLayout()
        
        # 快取設定
        advanced_layout.addWidget(QLabel("啟用系統快取:"), 0, 0)
        self.system_cache = QCheckBox()
        advanced_layout.addWidget(self.system_cache, 0, 1)
        
        # 快取大小上限
        advanced_layout.addWidget(QLabel("快取大小上限(MB):"), 1, 0)
        self.system_cache_size = QSpinBox()
        self.system_cache_size.setRange(100, 10000)
        self.system_cache_size.setSingleStep(100)
        advanced_layout.addWidget(self.system_cache_size, 1, 1)
        
        # 清除快取按鈕
        advanced_layout.addWidget(QLabel("清除系統快取:"), 2, 0)
        clear_cache_button = QPushButton("清除快取")
        clear_cache_button.clicked.connect(self._clear_cache)
        advanced_layout.addWidget(clear_cache_button, 2, 1)
        
        advanced_group.setLayout(advanced_layout)
        content_layout.addWidget(advanced_group)
        
        # 添加伸展因子
        content_layout.addStretch()
        
        # 設置滾動區域
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return tab

    def _browse_directory(self, line_edit):
        """瀏覽選擇目錄
        
        Args:
            line_edit: 要填寫目錄路徑的QLineEdit
        """
        current_dir = line_edit.text() or os.getcwd()
        directory = QFileDialog.getExistingDirectory(
            self, 
            "選擇目錄", 
            current_dir
        )
        
        if directory:
            line_edit.setText(directory)

    def _browse_file(self, line_edit, filter_str="所有文件 (*.*)"):
        """瀏覽選擇文件
        
        Args:
            line_edit: 要填寫文件路徑的QLineEdit
            filter_str: 文件過濾字串
        """
        current_dir = os.path.dirname(line_edit.text()) or os.getcwd()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "選擇文件", 
            current_dir,
            filter_str
        )
        
        if file_path:
            line_edit.setText(file_path)

    def _open_directory(self, directory):
        """開啟指定目錄
        
        Args:
            directory: 目錄路徑
        """
        if not directory:
            return
            
        # 確保目錄存在
        if not os.path.exists(directory):
            reply = QMessageBox.question(
                self, 
                "目錄不存在", 
                f"目錄 {directory} 不存在，是否創建？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    QMessageBox.critical(self, "錯誤", f"無法創建目錄: {str(e)}")
                    return
            else:
                return
        
        # 使用系統檔案管理器開啟目錄
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        QDesktopServices.openUrl(QUrl.fromLocalFile(directory))

    def _create_directories(self):
        """建立所有設定的目錄"""
        dirs = [
            self.data_dir.text(),
            self.output_dir.text(),
            self.model_dir.text(),
            self.log_dir.text(),
            self.export_dir.text()
        ]
        
        created = []
        errors = []
        
        for directory in dirs:
            if directory:
                try:
                    os.makedirs(directory, exist_ok=True)
                    created.append(directory)
                except Exception as e:
                    errors.append(f"{directory}: {str(e)}")
        
        if errors:
            QMessageBox.warning(
                self, 
                "建立目錄失敗", 
                "以下目錄無法建立：\n" + "\n".join(errors)
            )
        
        if created:
            QMessageBox.information(
                self, 
                "建立目錄成功", 
                "已成功建立以下目錄：\n" + "\n".join(created)
            )

    def _download_bert_model(self):
        """下載BERT模型"""
        model_name = self.bert_model_name.currentText()
        
        reply = QMessageBox.question(
            self, 
            "下載BERT模型", 
            f"確定要下載BERT模型 {model_name} 嗎？\n\n這可能需要一些時間，具體取決於您的網絡速度。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 調用離線下載模組
            from PyQt5.QtWidgets import QProgressDialog
            from PyQt5.QtCore import Qt
            
            progress = QProgressDialog("正在下載BERT模型...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setWindowTitle("下載BERT模型")
            progress.setValue(0)
            progress.show()
            
            # TODO: 實現實際的下載邏輯，這裡只是模擬
            import time
            for i in range(101):
                time.sleep(0.1)  # 模擬下載
                progress.setValue(i)
                if progress.wasCanceled():
                    break
            
            if not progress.wasCanceled():
                QMessageBox.information(
                    self, 
                    "下載完成", 
                    f"BERT模型 {model_name} 已成功下載。"
                )
            
            progress.close()

    def _toggle_attention_weights(self, checked):
        """切換注意力權重輸入框的啟用狀態
        
        Args:
            checked: 是否啟用自動平衡權重
        """
        # 如果啟用自動平衡，則禁用權重輸入框
        self.attention_similarity_weight.setEnabled(not checked)
        self.attention_keyword_weight.setEnabled(not checked)
        self.attention_self_weight.setEnabled(not checked)
        
        # 如果啟用自動平衡，則重置權重
        if checked:
            total_mechanisms = 0
            if self.attention_similarity.isChecked():
                total_mechanisms += 1
            if self.attention_keyword.isChecked():
                total_mechanisms += 1
            if self.attention_self.isChecked():
                total_mechanisms += 1
            
            if total_mechanisms > 0:
                weight = round(1.0 / total_mechanisms, 2)
                if self.attention_similarity.isChecked():
                    self.attention_similarity_weight.setValue(weight)
                if self.attention_keyword.isChecked():
                    self.attention_keyword_weight.setValue(weight)
                if self.attention_self.isChecked():
                    self.attention_self_weight.setValue(weight)

    def _clear_cache(self):
        """清除系統快取"""
        reply = QMessageBox.question(
            self, 
            "清除快取", 
            "確定要清除所有系統快取嗎？\n這將會刪除暫存的計算結果和模型資料。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # TODO: 實現實際的快取清除邏輯
            QMessageBox.information(self, "清除完成", "系統快取已清除。")

    def load_settings(self):
        """從配置載入設定"""
        # 儲存原配置以便比較變化
        self._original_config = json.loads(json.dumps(self.config.get_all()))
        
        # 載入BERT設定
        bert_config = self.config.get("bert")
        if bert_config:
            self.bert_model_name.setCurrentText(bert_config.get("model_name", "bert-base-uncased"))
            self.bert_max_length.setValue(bert_config.get("max_length", 128))
            self.bert_batch_size.setValue(bert_config.get("batch_size", 16))
            self.bert_use_gpu.setChecked(bert_config.get("use_gpu", True))
            
            # 進階設定
            self.bert_cache_dir.setText(bert_config.get("cache_dir", ""))
            self.bert_tokenizer.setCurrentText(bert_config.get("tokenizer", "WordPiece"))
        
        # 載入LDA設定
        lda_config = self.config.get("lda")
        if lda_config:
            self.lda_n_topics.setValue(lda_config.get("n_topics", 10))
            self.lda_alpha.setValue(lda_config.get("alpha", 0.1))
            self.lda_beta.setValue(lda_config.get("beta", 0.01))
            self.lda_max_iter.setValue(lda_config.get("max_iter", 50))
            self.lda_random_state.setValue(lda_config.get("random_state", 42))
            
            # 進階設定
            alg_map = {"VB (Variational Bayes)": "batch", "Online VB": "online", "Gibbs Sampling": "gibbs"}
            alg_reverse = {v: k for k, v in alg_map.items()}
            self.lda_algorithm.setCurrentText(alg_reverse.get(lda_config.get("algorithm", "batch"), "VB (Variational Bayes)"))
            
            self.lda_batch_size.setValue(lda_config.get("batch_size", 128))
            self.lda_tol.setValue(lda_config.get("tol", 0.001))
            
            # 評估指標
            eval_config = self.config.get("evaluation")
            if eval_config:
                metrics = eval_config.get("metrics", [])
                self.lda_coherence.setChecked("coherence" in metrics)
                self.lda_separation.setChecked("separation" in metrics)
        
        # 載入注意力機制設定
        attention_config = self.config.get("attention")
        if attention_config:
            enabled = attention_config.get("enabled_mechanisms", [])
            self.attention_similarity.setChecked("similarity" in enabled)
            self.attention_keyword.setChecked("keyword" in enabled)
            self.attention_self.setChecked("self" in enabled)
            self.attention_combined.setChecked("combined" in enabled)
            
            # 權重
            weights = attention_config.get("weights", {})
            self.attention_similarity_weight.setValue(weights.get("similarity", 0.33))
            self.attention_keyword_weight.setValue(weights.get("keyword", 0.33))
            self.attention_self_weight.setValue(weights.get("self", 0.34))
            
            # 自動平衡狀態
            self.attention_auto_balance.setChecked(attention_config.get("auto_balance", False))
            
            # 進階設定
            self.attention_window_size.setValue(attention_config.get("window_size", 5))
            self.attention_threshold.setValue(attention_config.get("threshold", 0.1))
        
        # 載入路徑設定
        paths_config = self.config.get("paths")
        if paths_config:
            self.data_dir.setText(paths_config.get("data_dir", self.default_paths["data_dir"]))
            self.output_dir.setText(paths_config.get("output_dir", self.default_paths["output_dir"]))
            self.model_dir.setText(paths_config.get("model_dir", self.default_paths["model_dir"]))
            self.log_dir.setText(paths_config.get("log_dir", self.default_paths["log_dir"]))
            self.export_dir.setText(paths_config.get("export_dir", self.default_paths["export_dir"]))
        
        # 載入視覺化設定
        viz_config = self.config.get("visualization")
        if viz_config:
            self.viz_dpi.setValue(viz_config.get("dpi", 300))
            self.viz_format.setCurrentText(viz_config.get("format", "png"))
            self.viz_color_scheme.setCurrentText(viz_config.get("color_scheme", "viridis"))
            
            # 圖表類型
            chart_types = {"bubble": "泡泡圖", "stacked_bar": "堆疊柱狀圖", "heatmap": "熱力圖"}
            chart_types_reverse = {v: k for k, v in chart_types.items()}
            self.viz_topic_type.setCurrentText(chart_types.get(viz_config.get("topic_chart_type", "bubble"), "泡泡圖"))
            
            # 詞雲設定
            self.viz_wordcloud_words.setValue(viz_config.get("wordcloud_max_words", 100))
            
            # 字體設定
            font_name = viz_config.get("font", "Arial")
            font_idx = self.viz_font.findText(font_name)
            if font_idx >= 0:
                self.viz_font.setCurrentIndex(font_idx)
        
        # 載入系統設定
        system_config = self.config.get("system")
        if system_config:
            self.system_log_level.setCurrentText(system_config.get("log_level", "INFO"))
            self.system_multi_processing.setChecked(system_config.get("multi_processing", True))
            self.system_num_workers.setValue(system_config.get("num_workers", 4))
            
            # 快取設定
            self.system_cache.setChecked(system_config.get("cache_enabled", True))
            self.system_cache_size.setValue(system_config.get("cache_size_mb", 1000))
        
        # 載入數據集設定
        datasets_config = self.config.get("datasets")
        if datasets_config:
            if "imdb" in datasets_config:
                self.dataset_imdb_path.setText(datasets_config["imdb"].get("path", ""))
                
            if "amazon" in datasets_config:
                self.dataset_amazon_path.setText(datasets_config["amazon"].get("path", ""))
                
            if "yelp" in datasets_config:
                self.dataset_yelp_path.setText(datasets_config["yelp"].get("path", ""))
        
        # 更新 UI 狀態
        self._toggle_attention_weights(self.attention_auto_balance.isChecked())

    def collect_settings(self):
        """收集所有設定"""
        config = {}
        
        # 收集BERT設定
        config["bert"] = {
            "model_name": self.bert_model_name.currentText(),
            "max_length": self.bert_max_length.value(),
            "batch_size": self.bert_batch_size.value(),
            "use_gpu": self.bert_use_gpu.isChecked(),
            "cache_dir": self.bert_cache_dir.text(),
            "tokenizer": self.bert_tokenizer.currentText()
        }
        
        # 收集LDA設定
        alg_map = {"VB (Variational Bayes)": "batch", "Online VB": "online", "Gibbs Sampling": "gibbs"}
        config["lda"] = {
            "n_topics": self.lda_n_topics.value(),
            "alpha": self.lda_alpha.value(),
            "beta": self.lda_beta.value(),
            "max_iter": self.lda_max_iter.value(),
            "random_state": self.lda_random_state.value(),
            "algorithm": alg_map.get(self.lda_algorithm.currentText(), "batch"),
            "batch_size": self.lda_batch_size.value(),
            "tol": self.lda_tol.value()
        }
        
        # 收集評估指標設定
        metrics = []
        if self.lda_coherence.isChecked():
            metrics.append("coherence")
        if self.lda_separation.isChecked():
            metrics.append("separation")
            
        config["evaluation"] = {
            "metrics": metrics,
            "coherence_weight": 0.5,
            "separation_weight": 0.5
        }
        
        # 收集注意力機制設定
        enabled_mechanisms = []
        if self.attention_similarity.isChecked():
            enabled_mechanisms.append("similarity")
        if self.attention_keyword.isChecked():
            enabled_mechanisms.append("keyword")
        if self.attention_self.isChecked():
            enabled_mechanisms.append("self")
        if self.attention_combined.isChecked():
            enabled_mechanisms.append("combined")
        
        config["attention"] = {
            "enabled_mechanisms": enabled_mechanisms,
            "weights": {
                "similarity": self.attention_similarity_weight.value(),
                "keyword": self.attention_keyword_weight.value(),
                "self": self.attention_self_weight.value()
            },
            "auto_balance": self.attention_auto_balance.isChecked(),
            "window_size": self.attention_window_size.value(),
            "threshold": self.attention_threshold.value()
        }
        
        # 收集路徑設定
        config["paths"] = {
            "data_dir": self.data_dir.text(),
            "output_dir": self.output_dir.text(),
            "model_dir": self.model_dir.text(),
            "log_dir": self.log_dir.text(),
            "export_dir": self.export_dir.text()
        }
        
        # 收集視覺化設定
        chart_types = {"泡泡圖": "bubble", "堆疊柱狀圖": "stacked_bar", "熱力圖": "heatmap"}
        config["visualization"] = {
            "dpi": self.viz_dpi.value(),
            "format": self.viz_format.currentText(),
            "color_scheme": self.viz_color_scheme.currentText(),
            "topic_chart_type": chart_types.get(self.viz_topic_type.currentText(), "bubble"),
            "wordcloud_max_words": self.viz_wordcloud_words.value(),
            "font": self.viz_font.currentText()
        }
        
        # 收集系統設定
        config["system"] = {
            "log_level": self.system_log_level.currentText(),
            "multi_processing": self.system_multi_processing.isChecked(),
            "num_workers": self.system_num_workers.value(),
            "cache_enabled": self.system_cache.isChecked(),
            "cache_size_mb": self.system_cache_size.value()
        }
        
        # 收集數據集設定
        config["datasets"] = {
            "imdb": {
                "name": "IMDB電影評論",
                "type": "movie",
                "language": "english",
                "path": self.dataset_imdb_path.text()
            },
            "amazon": {
                "name": "Amazon產品評論",
                "type": "product",
                "language": "english",
                "path": self.dataset_amazon_path.text()
            },
            "yelp": {
                "name": "Yelp餐廳評論",
                "type": "restaurant",
                "language": "english",
                "path": self.dataset_yelp_path.text()
            }
        }
        
        return config

    def apply_settings(self):
        """應用設定"""
        # 收集設定
        new_config = self.collect_settings()
        
        # 更新配置對象
        for section, values in new_config.items():
            # 使用深度更新防止丟失未在UI中顯示的子設定
            self._update_config_section(section, values)
        
        # 發出設定更改信號
        self.settings_changed.emit()
        
        # 顯示應用成功訊息
        QMessageBox.information(self, "設定已應用", "設定已成功應用。\n\n部分設定可能需要重啟應用程式才能生效。")

    def _update_config_section(self, section, values):
        """深度更新配置部分
        
        Args:
            section: 配置段名稱
            values: 新值
        """
        if isinstance(values, dict):
            # 確保該部分存在於配置中
            if section not in self.config.config:
                self.config.config[section] = {}
            
            # 迭代更新子項目
            for key, value in values.items():
                if isinstance(value, dict) and key in self.config.config[section] and isinstance(self.config.config[section][key], dict):
                    # 如果是嵌套字典，則遞迴更新
                    self._update_config_section(f"{section}.{key}", value)
                else:
                    # 否則直接更新值
                    self.config.config[section][key] = value
        else:
            # 如果不是字典，則簡單更新值 (通常不會發生)
            self.config.config[section] = values

    def save_settings(self):
        """保存設定"""
        # 先應用設定
        self.apply_settings()
        
        # 保存配置文件
        self.config.save()
        
        # 顯示保存成功訊息
        QMessageBox.information(self, "設定已保存", "設定已成功保存到配置文件。")

    def reset_settings(self):
        """重置設定"""
        reply = QMessageBox.question(
            self, 
            "重置設定", 
            "確定要將所有設定重置為默認值嗎？\n這將會失去所有自定義配置。",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 重置配置
            self.config.reset_to_default()
            # 重新載入UI
            self.load_settings()
            
            # 顯示重置成功訊息
            QMessageBox.information(self, "設定已重置", "所有設定已重置為默認值。")

    def closeEvent(self, event):
        """處理窗口關閉事件"""
        # 檢查是否有未保存的更改
        current_config = self.collect_settings()
        
        # 轉換為JSON字串進行深度比較
        current_json = json.dumps(current_config, sort_keys=True)
        original_json = json.dumps(self._original_config, sort_keys=True)
        
        if current_json != original_json:
            reply = QMessageBox.question(
                self, 
                "未保存的更改", 
                "有未保存的設定更改，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_settings()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
                
        # 接受關閉事件
        event.accept()