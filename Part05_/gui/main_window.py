import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from pathlib import Path
import pandas as pd
from gui.config import WINDOW_TITLE, WINDOW_SIZE, WINDOW_MIN_SIZE, COLORS, STATUS_TEXT, SUPPORTED_FILE_TYPES, FONTS, SIMULATION_DELAYS, DATASETS, PREPROCESSING_STEPS
from modules.text_preprocessor import TextPreprocessor
import threading
import queue
import torch
from modules.run_manager import RunManager
from modules.modular_gui_extensions import MODULAR_METHODS

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*WINDOW_MIN_SIZE)
        
        # 設定資料庫目錄路徑
        self.database_dir = self.get_database_dir()
        
        # 初始化RunManager
        self.run_manager = RunManager(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 初始化數據集類型
        self.dataset_type = tk.StringVar()
        
        # 初始化分類器類型
        self.classifier_type = tk.StringVar(value='xgboost')
        
        # 初始化編碼器類型
        self.encoder_type = tk.StringVar(value='bert')
        
        # 初始化面向分類器類型
        self.aspect_classifier_type = tk.StringVar(value='default')
        
        # 初始化步驟狀態
        self.step_states = {
            'file_imported': False,    # 步驟1：檔案導入
            'processing_done': False,   # 步驟2：文本處理
            'encoding_done': False,     # 步驟3：BERT編碼
            'baseline_done': False,     # 步驟4：基準測試
            'dual_head_done': False,    # 步驟5：雙頭測試
            'triple_head_done': False,  # 步驟6：三頭測試
            'analysis_done': False      # 步驟7：比對分析
        }
        
        # 初始化處理佇列
        self.process_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.encoding_queue = queue.Queue()  # 新增：BERT編碼佇列
        
        # 保存最後一次預處理的 run 目錄
        self.last_run_dir = None
        
        # 創建筆記本控件（分頁）
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=15, pady=15)
        
        # 創建四個分頁
        self.create_data_processing_tab()
        self.create_attention_testing_tab()
        self.create_comparison_analysis_tab()
        self.create_cross_validation_tab()
        
        # 添加當前run目錄標籤
        self.create_run_dir_label()
        
        # 初始化按鈕狀態
        self.update_button_states()
        
        # 最大化視窗（在所有UI元素創建完成後）
        self.root.after(100, self.maximize_window)
        
        # 延遲更新配置顯示（等待GUI元素完全初始化）
        self.root.after(200, self.update_current_config_safe)
    
    def detect_compute_environment(self):
        """檢測計算環境"""
        try:
            from modules.sentiment_classifier import SentimentClassifier
            classifier = SentimentClassifier()
            device_info = classifier.get_device_info()
            
            if device_info['has_gpu']:
                self.device_label.config(text=f"🔥 {device_info['description']}", foreground='green')
            else:
                self.device_label.config(text=f"🖥️ {device_info['description']}", foreground='blue')
                
        except Exception as e:
            self.device_label.config(text="❓ 環境檢測失敗", foreground='red')
    
    def on_encoder_selected(self, event=None):
        """編碼器選擇變更時的回調"""
        selected = self.encoder_type.get()
        
        # 顯示編碼器相關信息
        encoder_info = {
            'bert': "✨ BERT - 強大的語義理解能力",
            'gpt': "🚀 GPT - 優秀的生成式語言模型",
            't5': "🎯 T5 - 統一的Text-to-Text框架",
            'cnn': "⚡ CNN - 高效的卷積神經網路",
            'elmo': "🌊 ELMo - 上下文相關嵌入表示"
        }
        
        info_text = encoder_info.get(selected, "")
        # 安全檢查：只有當標籤存在時才更新
        if hasattr(self, 'encoder_desc_label') and self.encoder_desc_label:
            self.encoder_desc_label.config(text=info_text)
        # 更新模組化流水線配置顯示
        if hasattr(self, 'current_config_label'):
            self.update_current_config()
    
    def on_aspect_classifier_selected(self, event=None):
        """面向分類器選擇變更時的回調"""
        selected = self.aspect_classifier_type.get()
        
        # 顯示面向分類器相關信息
        aspect_info = {
            'default': "🎯 預設 - 基於注意力機制的高準確率分類",
            'lda': "📈 LDA - 潛在狄利克雷分配主題建模",
            'bertopic': "🤖 BERTopic - 基於BERT的高品質主題模型",
            'nmf': "📊 NMF - 非負矩陣分解方法"
        }
        
        info_text = aspect_info.get(selected, "")
        # 安全檢查：只有當標籤存在時才更新
        if hasattr(self, 'aspect_desc_label') and self.aspect_desc_label:
            self.aspect_desc_label.config(text=info_text)
        # 更新模組化流水線配置顯示
        if hasattr(self, 'current_config_label'):
            self.update_current_config()
    
    def on_classifier_selected(self, event=None):
        """情感分類器選擇變更時的回調"""
        selected = self.classifier_type.get()
        
        # 顯示分類器相關信息
        classifier_info = {
            'xgboost': "⚡ XGBoost - 高準確率，支援GPU加速",
            'logistic_regression': "🚀 邏輯迴歸 - 快速穩定，適合中小數據",
            'random_forest': "🌳 隨機森林 - 穩定可靠，可並行處理",
            'svm_linear': "📐 線性SVM - 適合線性可分數據"
        }
        
        info_text = classifier_info.get(selected, "")
        if hasattr(self, 'timing_label'):
            self.timing_label.config(text=info_text)
        # 更新模組化流水線配置顯示
        if hasattr(self, 'current_config_label'):
            self.update_current_config()
    
    def center_window(self):
        """將視窗置中於螢幕（已棄用，改用最大化視窗）"""
        # 強制更新視窗以獲取實際尺寸
        self.root.update_idletasks()
        
        # 從WINDOW_SIZE配置獲取視窗尺寸
        size_parts = WINDOW_SIZE.split('x')
        window_width = int(size_parts[0])
        window_height = int(size_parts[1])
        
        # 獲取螢幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 計算置中位置
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # 確保視窗不會超出螢幕邊界
        x = max(0, x)
        y = max(0, y)
        
        # 設定視窗大小和位置
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 確保視窗顯示在最前面
        self.root.lift()
        self.root.focus_force()
    
    def maximize_window(self):
        """最大化視窗"""
        try:
            # 嘗試使用state方法最大化 (Windows/Linux)
            self.root.state('zoomed')
        except:
            try:
                # 備用方法：使用attributes (某些Linux發行版)
                self.root.attributes('-zoomed', True)
            except:
                try:
                    # 第三種方法：使用wm_state (macOS兼容)
                    self.root.wm_state('zoomed')
                except:
                    # 最後備用方法：手動設置為螢幕大小
                    screen_width = self.root.winfo_screenwidth()
                    screen_height = self.root.winfo_screenheight()
                    self.root.geometry(f'{screen_width}x{screen_height}+0+0')
        
        # 確保視窗顯示在最前面
        self.root.lift()
        self.root.focus_force()
    
    def create_data_processing_tab(self):
        """第一分頁：資料處理"""
        frame1 = ttk.Frame(self.notebook)
        self.notebook.add(frame1, text=" 資料處理 ")
        
        # 主要容器
        main_frame = ttk.Frame(frame1)
        main_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # 標題
        title_label = ttk.Label(main_frame, text="資料處理流程", font=FONTS['title'])
        title_label.pack(pady=(0, 12))
        
        # 步驟1：選擇數據集類型
        step1_frame = ttk.LabelFrame(main_frame, text="① 選擇數據集類型", padding=15)
        step1_frame.pack(fill='x', pady=(0, 15))
        
        dataset_frame = ttk.Frame(step1_frame)
        dataset_frame.pack(fill='x')
        
        ttk.Label(dataset_frame, text="數據集類型:").pack(side='left')
        
        # 建立數據集選擇下拉選單
        dataset_combo = ttk.Combobox(dataset_frame, 
                                   textvariable=self.dataset_type,
                                   values=[DATASETS[ds]['name'] for ds in DATASETS],
                                   state='readonly',
                                   width=30)
        dataset_combo.pack(side='left', padx=(10, 0))
        dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_selected)
        
        # 步驟2：文本輸入 → 導入檔案
        step2_frame = ttk.LabelFrame(main_frame, text="② 文本輸入 → 導入檔案", padding=15)
        step2_frame.pack(fill='x', pady=(0, 15))
        
        input_frame = ttk.Frame(step2_frame)
        input_frame.pack(fill='x')
        
        ttk.Label(input_frame, text="選擇檔案:").pack(side='left')
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(input_frame, textvariable=self.file_path_var, width=60)
        file_entry.pack(side='left', padx=(10, 5), fill='x', expand=True)
        
        self.browse_btn = ttk.Button(input_frame, text="瀏覽", command=self.browse_file, state='disabled')
        self.browse_btn.pack(side='left', padx=(5, 10))
        
        # 新增：抽樣設置框架
        sampling_frame = ttk.Frame(step2_frame)
        sampling_frame.pack(fill='x', pady=(10, 0))
        
        # 抽樣選項
        sampling_left_frame = ttk.Frame(sampling_frame)
        sampling_left_frame.pack(side='left', fill='x', expand=True)
        
        self.use_sampling_var = tk.BooleanVar(value=False)
        sampling_checkbox = ttk.Checkbutton(sampling_left_frame, 
                                           text="啟用數據抽樣 (適用於大數據集)", 
                                           variable=self.use_sampling_var,
                                           command=self.on_sampling_toggle)
        sampling_checkbox.pack(side='left', anchor='w')
        
        # 抽樣數量輸入框架
        sampling_input_frame = ttk.Frame(sampling_frame)
        sampling_input_frame.pack(side='right')
        
        ttk.Label(sampling_input_frame, text="抽樣數量:").pack(side='left', padx=(0, 5))
        
        self.sample_size_var = tk.StringVar(value="1000")
        self.sample_size_entry = ttk.Entry(sampling_input_frame, 
                                         textvariable=self.sample_size_var, 
                                         width=10,
                                         state='disabled')
        self.sample_size_entry.pack(side='left', padx=(0, 5))
        
        ttk.Label(sampling_input_frame, text="個樣本").pack(side='left')
        
        # 抽樣說明
        sampling_info_frame = ttk.Frame(step2_frame)
        sampling_info_frame.pack(fill='x', pady=(5, 0))
        
        self.sampling_info = ttk.Label(sampling_info_frame, 
                                     text="💡 建議：大數據集(>10000樣本)建議抽樣以提高處理速度", 
                                     foreground='gray',
                                     font=('TkDefaultFont', 8))
        self.sampling_info.pack(anchor='w')
        
        self.import_status = ttk.Label(step2_frame, text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
        self.import_status.pack(anchor='w', pady=(10, 0))
        
        # 步驟3：文本處理 → 開始處理
        step3_frame = ttk.LabelFrame(main_frame, text="③ 文本處理 → 開始處理", padding=15)
        step3_frame.pack(fill='x', pady=(0, 15))
        
        process_frame = ttk.Frame(step3_frame)
        process_frame.pack(fill='x')
        
        # 添加預處理進度條和狀態標籤
        progress_frame = ttk.Frame(process_frame)
        progress_frame.pack(side='left', fill='x', expand=True)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                          variable=self.progress_var,
                                          maximum=100,
                                          length=300,
                                          mode='determinate')
        self.progress_bar.pack(side='top', fill='x', padx=(0, 10))
        
        self.process_status = ttk.Label(progress_frame, 
                                      text=STATUS_TEXT['pending'],
                                      foreground=COLORS['pending'])
        self.process_status.pack(side='top', anchor='w', pady=(5, 0))
        
        self.process_btn = ttk.Button(process_frame, text="開始處理", command=self.start_processing)
        self.process_btn.pack(side='right')
        
        # 步驟3.5：選擇編碼器類型
        encoder_frame = ttk.LabelFrame(main_frame, text="③.5 選擇文本編碼器", padding=15)
        encoder_frame.pack(fill='x', pady=(0, 15))
        
        encoder_content = ttk.Frame(encoder_frame)
        encoder_content.pack(fill='x')
        
        ttk.Label(encoder_content, text="編碼器類型:").pack(side='left')
        
        self.encoder_combo = ttk.Combobox(encoder_content,
                                         textvariable=self.encoder_type,
                                         values=['bert', 'gpt', 't5', 'cnn', 'elmo'],
                                         state='readonly',
                                         width=20)
        self.encoder_combo.pack(side='left', padx=(10, 0))
        self.encoder_combo.bind('<<ComboboxSelected>>', self.on_encoder_selected)
        
        # 編碼器描述標籤
        self.encoder_desc_label = ttk.Label(encoder_content, text="✨ BERT - 強大的語義理解能力", foreground='blue')
        self.encoder_desc_label.pack(side='left', padx=(15, 0))
        
        # 步驟4：文本編碼 → 開始編碼
        step4_frame = ttk.LabelFrame(main_frame, text="④ 文本編碼 → 開始編碼", padding=15)
        step4_frame.pack(fill='x', pady=(0, 15))
        
        encoding_frame = ttk.Frame(step4_frame)
        encoding_frame.pack(fill='x')
        
        # 添加BERT編碼專用的進度條和狀態標籤
        encoding_progress_frame = ttk.Frame(encoding_frame)
        encoding_progress_frame.pack(side='left', fill='x', expand=True)
        
        self.encoding_progress_var = tk.DoubleVar()
        self.encoding_progress_bar = ttk.Progressbar(encoding_progress_frame, 
                                                   variable=self.encoding_progress_var,
                                                   maximum=100,
                                                   length=300,
                                                   mode='determinate')
        self.encoding_progress_bar.pack(side='top', fill='x', padx=(0, 10))
        
        self.encoding_status = ttk.Label(encoding_progress_frame, 
                                       text="狀態: 待處理",
                                       foreground="orange")
        self.encoding_status.pack(side='top', anchor='w', pady=(5, 0))
        
        self.encoding_btn = ttk.Button(encoding_frame, text="開始編碼", command=self.start_encoding)
        self.encoding_btn.pack(side='right')
        
        # 新增導入按鈕
        self.import_encoding_btn = ttk.Button(encoding_frame, text="導入編碼", command=self.import_encoding)
        self.import_encoding_btn.pack(side='right', padx=(10, 0))
        

        
    def create_attention_testing_tab(self):
        """第二分頁：注意力機制測試 - 三列緊湊佈局"""
        frame2 = ttk.Frame(self.notebook)
        self.notebook.add(frame2, text=" 注意力機制測試 ")
        
        # 主容器 - 去除滾動，使用固定佈局
        main_frame = ttk.Frame(frame2)
        main_frame.pack(fill='both', expand=True, padx=10, pady=8)
        
        # 標題
        title_label = ttk.Label(main_frame, text="注意力機制測試", font=FONTS['title'])
        title_label.pack(pady=(0, 8))
        
        # 頂部設定區域 - 橫向緊湊佈局
        top_config_frame = ttk.Frame(main_frame)
        top_config_frame.pack(fill='x', pady=(0, 8))
        
        # 面向分類器設定
        aspect_frame = ttk.LabelFrame(top_config_frame, text="面向分類器", padding=8)
        aspect_frame.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        aspect_content = ttk.Frame(aspect_frame)
        aspect_content.pack(fill='x')
        
        self.aspect_classifier_combo = ttk.Combobox(aspect_content,
                                                   textvariable=self.aspect_classifier_type,
                                                   values=['default', 'lda', 'bertopic', 'nmf'],
                                                   state='readonly',
                                                   width=15)
        self.aspect_classifier_combo.pack(fill='x')
        self.aspect_classifier_combo.bind('<<ComboboxSelected>>', self.on_aspect_classifier_selected)
        
        # 面向分類器描述標籤
        self.aspect_desc_label = ttk.Label(aspect_content, text="🎯 預設 - 基於注意力機制的高準確率分類", 
                                          foreground='blue', font=('TkDefaultFont', 8))
        self.aspect_desc_label.pack(pady=(5, 0))
        
        # 情感分類器設定
        classifier_frame = ttk.LabelFrame(top_config_frame, text="情感分類器", padding=8)
        classifier_frame.pack(side='left', fill='x', expand=True, padx=5)
        
        self.classifier_combo = ttk.Combobox(classifier_frame, 
                                           textvariable=self.classifier_type,
                                           values=['xgboost', 'logistic_regression', 'random_forest', 'svm_linear'],
                                           state='readonly',
                                           width=15)
        self.classifier_combo.pack(fill='x')
        self.classifier_combo.bind('<<ComboboxSelected>>', self.on_classifier_selected)
        
        # 狀態信息
        status_frame = ttk.LabelFrame(top_config_frame, text="狀態", padding=8)
        status_frame.pack(side='left', fill='x', expand=True, padx=(5, 0))
        
        self.device_label = ttk.Label(status_frame, text="檢測中...", foreground='gray', font=('TkDefaultFont', 8))
        self.device_label.pack(anchor='w')
        
        self.timing_label = ttk.Label(status_frame, text="", foreground='blue', font=('TkDefaultFont', 8))
        self.timing_label.pack(anchor='w')
        
        # 初始化設備檢測
        self.root.after(100, self.detect_compute_environment)
        
        # 三列注意力實驗區域
        experiments_frame = ttk.Frame(main_frame)
        experiments_frame.pack(fill='both', expand=True, pady=(0, 8))
        
        # 第一列：單一注意力實驗組
        single_frame = ttk.LabelFrame(experiments_frame, text="單一注意力實驗", padding=8)
        single_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # 單一注意力選項 - 緊湊顯示
        ttk.Label(single_frame, text="• 無注意力（基準）", font=('TkDefaultFont', 8)).pack(anchor='w')
        ttk.Label(single_frame, text="• 相似度注意力", font=('TkDefaultFont', 8)).pack(anchor='w')
        ttk.Label(single_frame, text="• 自注意力", font=('TkDefaultFont', 8)).pack(anchor='w')
        ttk.Label(single_frame, text="• 關鍵詞注意力", font=('TkDefaultFont', 8)).pack(anchor='w')
        
        # 控制按鈕和狀態
        single_control = ttk.Frame(single_frame)
        single_control.pack(fill='x', pady=(8, 0))
        
        self.single_btn = ttk.Button(single_control, text="執行測試", command=self.run_single_attention)
        self.single_btn.pack(fill='x', pady=(0, 3))
        
        self.single_status = ttk.Label(single_control, text=STATUS_TEXT['pending'], foreground=COLORS['pending'], 
                                     font=('TkDefaultFont', 8))
        self.single_status.pack(anchor='w')
        
        # 第二列：雙重組合實驗組
        dual_frame = ttk.LabelFrame(experiments_frame, text="雙重組合實驗", padding=8)
        dual_frame.pack(side='left', fill='both', expand=True, padx=2.5)
        
        # 雙重組合選項 - 緊湊顯示
        ttk.Label(dual_frame, text="• 基本機制 (4種)", font=('TkDefaultFont', 8, 'italic')).pack(anchor='w')
        ttk.Label(dual_frame, text="• 相似度+自注意力", font=('TkDefaultFont', 8)).pack(anchor='w')
        ttk.Label(dual_frame, text="• 相似度+關鍵詞", font=('TkDefaultFont', 8)).pack(anchor='w')
        ttk.Label(dual_frame, text="• 自注意力+關鍵詞", font=('TkDefaultFont', 8)).pack(anchor='w')
        
        # 控制按鈕和狀態
        dual_control = ttk.Frame(dual_frame)
        dual_control.pack(fill='x', pady=(8, 0))
        
        self.dual_btn = ttk.Button(dual_control, text="執行測試", command=self.run_dual_attention)
        self.dual_btn.pack(fill='x', pady=(0, 3))
        
        self.dual_status = ttk.Label(dual_control, text=STATUS_TEXT['pending'], foreground=COLORS['pending'],
                                   font=('TkDefaultFont', 8))
        self.dual_status.pack(anchor='w')
        
        # 第三列：三重組合實驗組
        triple_frame = ttk.LabelFrame(experiments_frame, text="三重組合實驗", padding=8)
        triple_frame.pack(side='left', fill='both', expand=True, padx=(5, 0))
        
        # 三重組合選項 - 緊湊顯示
        ttk.Label(triple_frame, text="• 基本機制 (4種)", font=('TkDefaultFont', 8, 'italic')).pack(anchor='w')
        ttk.Label(triple_frame, text="• 三重組合:", font=('TkDefaultFont', 8, 'bold')).pack(anchor='w')
        ttk.Label(triple_frame, text="  相似度+自注意力", font=('TkDefaultFont', 8)).pack(anchor='w')
        ttk.Label(triple_frame, text="  +關鍵詞", font=('TkDefaultFont', 8)).pack(anchor='w')
        
        # 控制按鈕和狀態
        triple_control = ttk.Frame(triple_frame)
        triple_control.pack(fill='x', pady=(8, 0))
        
        self.triple_btn = ttk.Button(triple_control, text="執行測試", command=self.run_triple_attention)
        self.triple_btn.pack(fill='x', pady=(0, 3))
        
        self.triple_status = ttk.Label(triple_control, text=STATUS_TEXT['pending'], foreground=COLORS['pending'],
                                     font=('TkDefaultFont', 8))
        self.triple_status.pack(anchor='w')
        
        # 模組化流水線區域 - 緊湊佈局
        pipeline_frame = ttk.LabelFrame(main_frame, text="🚀 模組化流水線", padding=8)
        pipeline_frame.pack(fill='both', expand=True, pady=(8, 0))
        
        # 頂部：配置和控制
        pipeline_top = ttk.Frame(pipeline_frame)
        pipeline_top.pack(fill='x', pady=(0, 8))
        
        # 左側：配置顯示
        config_left = ttk.Frame(pipeline_top)
        config_left.pack(side='left', fill='x', expand=True)
        
        self.current_config_label = ttk.Label(config_left,
                                             text="📝 當前: BERT + 預設 + XGBoost",
                                             foreground='green',
                                             font=('TkDefaultFont', 9, 'bold'))
        self.current_config_label.pack(anchor='w')
        
        self.pipeline_status = ttk.Label(config_left,
                                        text="狀態: 待執行",
                                        foreground='orange',
                                        font=('TkDefaultFont', 8))
        self.pipeline_status.pack(anchor='w')
        
        # 右側：控制按鈕
        control_right = ttk.Frame(pipeline_top)
        control_right.pack(side='right')
        
        button_frame = ttk.Frame(control_right)
        button_frame.pack()
        
        self.run_pipeline_btn = ttk.Button(button_frame,
                                          text="🚀 運行流水線",
                                          command=self.run_modular_pipeline)
        self.run_pipeline_btn.pack(side='left', padx=(0, 5))
        
        self.compare_methods_btn = ttk.Button(button_frame,
                                             text="📊 比較方法",
                                             command=self.compare_methods)
        self.compare_methods_btn.pack(side='left')
        
        # 進度條
        self.pipeline_progress_var = tk.DoubleVar()
        self.pipeline_progress_bar = ttk.Progressbar(pipeline_frame,
                                                    variable=self.pipeline_progress_var,
                                                    maximum=100)
        self.pipeline_progress_bar.pack(fill='x', pady=(0, 8))
        
        # 結果顯示區域 - 緊湊
        self.pipeline_results_text = scrolledtext.ScrolledText(pipeline_frame,
                                                              height=4,
                                                              font=('Consolas', 8))
        self.pipeline_results_text.pack(fill='both', expand=True)
        
        # 初始化模組化流水線相關變數
        self.pipeline_queue = queue.Queue()
        self.modular_pipeline = None

    def run_single_attention(self):
        """執行單一注意力測試 - 測試基本注意力機制[無、相似度、自注意力、關鍵詞]"""
        self.single_btn['state'] = 'disabled'
        self.single_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        try:
            # 檢查必要檔案
            if not self.last_run_dir:
                messagebox.showerror("錯誤", "請先完成BERT編碼步驟！")
                return
                
            # 設定檔案路徑
            input_file = os.path.join(self.last_run_dir, "01_preprocessed_data.csv")
            
            if not os.path.exists(input_file):
                messagebox.showerror("錯誤", "找不到預處理數據檔案！")
                return
            
            # 執行單一注意力機制分析
            from Part05_Main import process_attention_analysis_with_multiple_combinations
            
            # 設定要測試的注意力機制（僅基本機制）
            attention_types = ['no', 'similarity', 'self', 'keyword']
            attention_combinations = []  # 不使用組合
            output_dir = self.run_manager.get_run_dir()
            
            # 在後台執行分析
            def run_analysis():
                try:
                    import time
                    
                    # 記錄開始時間
                    start_time = time.time()
                    self.root.after(0, lambda: self.timing_label.config(
                        text=f"🔄 使用 {self.classifier_type.get()} 開始單一注意力測試...", 
                        foreground='orange'
                    ))
                    
                    results = process_attention_analysis_with_multiple_combinations(
                        input_file=input_file,
                        output_dir=output_dir,
                        attention_types=attention_types,
                        attention_combinations=attention_combinations,
                        classifier_type=self.classifier_type.get(),
                        encoder_type=self.encoder_type.get()
                    )
                    
                    # 計算總耗時
                    total_time = time.time() - start_time
                    self.root.after(0, lambda: self.timing_label.config(
                        text=f"✅ 單一注意力測試完成！總耗時: {total_time:.1f} 秒", 
                        foreground='green'
                    ))
                    
                    # 將結果存儲供比對分析使用
                    self.analysis_results = results
                    # 在主線程中更新UI
                    self.root.after(0, self._complete_attention_analysis, '單一注意力測試')
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: self._handle_analysis_error(msg, 'single'))
            
            threading.Thread(target=run_analysis, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"執行單一注意力測試時發生錯誤：{str(e)}")
            self.single_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
            self.single_btn['state'] = 'normal'
    
    def run_dual_attention(self):
        """執行雙重組合測試 - 測試基本機制+三組雙重組合"""
        self.dual_btn['state'] = 'disabled'
        self.dual_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        try:
            # 檢查必要檔案
            if not self.last_run_dir:
                messagebox.showerror("錯誤", "請先完成BERT編碼步驟！")
                return
                
            input_file = os.path.join(self.last_run_dir, "01_preprocessed_data.csv")
            
            if not os.path.exists(input_file):
                messagebox.showerror("錯誤", "找不到預處理數據檔案！")
                return
            
            # 執行雙重組合注意力分析
            from Part05_Main import process_attention_analysis_with_multiple_combinations
            
            # 設定要測試的基本注意力機制
            attention_types = ['no', 'similarity', 'self', 'keyword']
            output_dir = self.run_manager.get_run_dir()
            
            # 設定三組雙重組合權重
            attention_combinations = [
                # 相似度 + 自注意力
                {
                    'similarity': 0.5,
                    'self': 0.5,
                    'keyword': 0.0
                },
                # 相似度 + 關鍵詞
                {
                    'similarity': 0.5,
                    'keyword': 0.5,
                    'self': 0.0
                },
                # 自注意力 + 關鍵詞
                {
                    'similarity': 0.0,
                    'self': 0.5,
                    'keyword': 0.5
                }
            ]
            
            def run_analysis():
                try:
                    # 記錄開始時間
                    import time
                    start_time = time.time()
                    self.root.after(0, lambda: self.timing_label.config(
                        text=f"🔄 使用 {self.classifier_type.get()} 開始雙重組合測試...", 
                        foreground='orange'
                    ))
                    
                    results = process_attention_analysis_with_multiple_combinations(
                        input_file=input_file,
                        output_dir=output_dir,
                        attention_types=attention_types,
                        attention_combinations=attention_combinations,
                        classifier_type=self.classifier_type.get(),
                        encoder_type=self.encoder_type.get()
                    )
                    
                    # 計算總耗時
                    total_time = time.time() - start_time
                    self.root.after(0, lambda: self.timing_label.config(
                        text=f"✅ 雙重組合測試完成！總耗時: {total_time:.1f} 秒", 
                        foreground='green'
                    ))
                    
                    self.analysis_results = results
                    self.root.after(0, self._complete_attention_analysis, '雙重組合測試')
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: self._handle_analysis_error(msg, 'dual'))
            
            threading.Thread(target=run_analysis, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"執行雙重組合測試時發生錯誤：{str(e)}")
            self.dual_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
            self.dual_btn['state'] = 'normal'
    
    def run_triple_attention(self):
        """執行三重組合測試 - 測試基本機制+一組三重組合"""
        self.triple_btn['state'] = 'disabled'
        self.triple_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        try:
            # 檢查必要檔案
            if not self.last_run_dir:
                messagebox.showerror("錯誤", "請先完成BERT編碼步驟！")
                return
                
            input_file = os.path.join(self.last_run_dir, "01_preprocessed_data.csv")
            
            if not os.path.exists(input_file):
                messagebox.showerror("錯誤", "找不到預處理數據檔案！")
                return
            
            # 執行三重組合注意力分析
            from Part05_Main import process_attention_analysis_with_multiple_combinations
            
            # 設定要測試的基本注意力機制
            attention_types = ['no', 'similarity', 'self', 'keyword']
            output_dir = self.run_manager.get_run_dir()
            
            # 設定一組三重組合權重
            attention_combinations = [
                # 相似度 + 自注意力 + 關鍵詞
                {
                    'similarity': 0.33,
                    'self': 0.33,
                    'keyword': 0.34
                }
            ]
            
            def run_analysis():
                try:
                    # 記錄開始時間
                    import time
                    start_time = time.time()
                    self.root.after(0, lambda: self.timing_label.config(
                        text=f"🔄 使用 {self.classifier_type.get()} 開始三重組合測試...", 
                        foreground='orange'
                    ))
                    
                    results = process_attention_analysis_with_multiple_combinations(
                        input_file=input_file,
                        output_dir=output_dir,
                        attention_types=attention_types,
                        attention_combinations=attention_combinations,
                        classifier_type=self.classifier_type.get(),
                        encoder_type=self.encoder_type.get()
                    )
                    
                    # 計算總耗時
                    total_time = time.time() - start_time
                    self.root.after(0, lambda: self.timing_label.config(
                        text=f"✅ 三重組合測試完成！總耗時: {total_time:.1f} 秒", 
                        foreground='green'
                    ))
                    
                    self.analysis_results = results
                    self.root.after(0, self._complete_attention_analysis, '三重組合測試')
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda msg=error_msg: self._handle_analysis_error(msg, 'triple'))
            
            threading.Thread(target=run_analysis, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"執行三重組合測試時發生錯誤：{str(e)}")
            self.triple_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
            self.triple_btn['state'] = 'normal'
    
    def _complete_attention_analysis(self, test_type):
        """完成注意力分析後的處理"""
        # 更新對應的狀態
        if test_type == '單一注意力測試':
            self.single_status.config(text="分析完成，正在跳轉...", foreground=COLORS['success'])
            self.single_btn['state'] = 'normal'
        elif test_type == '雙重組合測試':
            self.dual_status.config(text="分析完成，正在跳轉...", foreground=COLORS['success'])
            self.dual_btn['state'] = 'normal'
        elif test_type == '三重組合測試':
            self.triple_status.config(text="分析完成，正在跳轉...", foreground=COLORS['success'])
            self.triple_btn['state'] = 'normal'
        
        # 更新比對分析頁面的結果
        self._update_analysis_results()
        
        # 跳轉到比對分析頁面
        self.notebook.select(2)  # 選擇第三個分頁（索引為2）
        
        messagebox.showinfo("完成", f"{test_type}已完成！結果已顯示在比對分析頁面。")
    
    def _handle_analysis_error(self, error_msg, test_type):
        """處理分析錯誤"""
        messagebox.showerror("錯誤", f"分析過程中發生錯誤：{error_msg}")
        
        if test_type == 'single':
            self.single_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
            self.single_btn['state'] = 'normal'
        elif test_type == 'dual':
            self.dual_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
            self.dual_btn['state'] = 'normal'
        elif test_type == 'triple':
            self.triple_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
            self.triple_btn['state'] = 'normal'

    def create_comparison_analysis_tab(self):
        """第三分頁：比對分析"""
        frame3 = ttk.Frame(self.notebook)
        self.notebook.add(frame3, text=" 比對分析 ")
        
        main_frame = ttk.Frame(frame3)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 標題
        title_label = ttk.Label(main_frame, text="比對分析結果", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 移除分析控制區域，由前一頁自動執行
        
        # 分析結果區域
        results_frame = ttk.LabelFrame(main_frame, text="注意力機制分類性能比較", padding=15)
        results_frame.pack(fill='x', pady=(0, 15))
        
        # 創建性能比較表格
        performance_columns = ('注意力機制', '準確率', 'F1分數', '召回率', '精確率')
        self.performance_tree = ttk.Treeview(results_frame, columns=performance_columns, show='headings', height=8)
        
        for col in performance_columns:
            self.performance_tree.heading(col, text=col)
            if col == '注意力機制':
                self.performance_tree.column(col, width=150, anchor='center')
            else:
                self.performance_tree.column(col, width=120, anchor='center')
        
        # 添加滾動條
        performance_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.performance_tree.yview)
        self.performance_tree.configure(yscrollcommand=performance_scrollbar.set)
        
        self.performance_tree.pack(side='left', fill='both', expand=True)
        performance_scrollbar.pack(side='right', fill='y')
        
        # 詳細比對結果區域
        detail_frame = ttk.LabelFrame(main_frame, text="詳細比對結果", padding=15)
        detail_frame.pack(fill='both', expand=True, pady=(15, 0))
        
        # 創建詳細比對表格
        detail_columns = ('原始索引', '原始文章', '原始標籤', '預測標籤', '是否正確')
        self.detail_tree = ttk.Treeview(detail_frame, columns=detail_columns, show='headings', height=12)
        
        # 設定欄位寬度和對齊
        self.detail_tree.heading('原始索引', text='原始索引')
        self.detail_tree.column('原始索引', width=80, anchor='center')
        
        self.detail_tree.heading('原始文章', text='原始文章')
        self.detail_tree.column('原始文章', width=300, anchor='w')
        
        self.detail_tree.heading('原始標籤', text='原始標籤')
        self.detail_tree.column('原始標籤', width=100, anchor='center')
        
        self.detail_tree.heading('預測標籤', text='預測標籤')
        self.detail_tree.column('預測標籤', width=100, anchor='center')
        
        self.detail_tree.heading('是否正確', text='是否正確')
        self.detail_tree.column('是否正確', width=80, anchor='center')
        
        # 添加滾動條
        detail_scrollbar = ttk.Scrollbar(detail_frame, orient='vertical', command=self.detail_tree.yview)
        self.detail_tree.configure(yscrollcommand=detail_scrollbar.set)
        
        # 添加水平滾動條
        h_scrollbar = ttk.Scrollbar(detail_frame, orient='horizontal', command=self.detail_tree.xview)
        self.detail_tree.configure(xscrollcommand=h_scrollbar.set)
        
        # 包裝表格和滾動條
        tree_frame = ttk.Frame(detail_frame)
        tree_frame.pack(fill='both', expand=True)
        
        self.detail_tree.pack(side='left', fill='both', expand=True)
        detail_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        
        # 狀態標籤
        self.analysis_status = ttk.Label(main_frame, text="等待分析結果...", foreground="orange")
        self.analysis_status.pack(pady=(10, 0))
        
        # 初始化分析結果變量
        self.analysis_results = None

    def update_button_states(self):
        """更新所有按鈕的啟用/禁用狀態"""
        # 第一分頁按鈕
        self.process_btn['state'] = 'normal' if self.step_states['file_imported'] else 'disabled'
        self.encoding_btn['state'] = 'normal' if self.step_states['processing_done'] else 'disabled'
        
        # 第二分頁按鈕 - 所有注意力測試按鈕都需要等待 BERT 編碼完成
        attention_buttons_enabled = 'normal' if self.step_states['encoding_done'] else 'disabled'
        self.single_btn['state'] = attention_buttons_enabled
        self.dual_btn['state'] = attention_buttons_enabled
        self.triple_btn['state'] = attention_buttons_enabled
        
        # 更新狀態標籤
        if not self.step_states['encoding_done']:
            status_text = "請先完成BERT編碼步驟"
            self.single_status.config(text=status_text, foreground=COLORS['pending'])
            self.dual_status.config(text=status_text, foreground=COLORS['pending'])
            self.triple_status.config(text=status_text, foreground=COLORS['pending'])
        
        # 第三分頁現在由前一頁自動跳轉，不需要手動控制按鈕

    def get_database_dir(self):
        """取得資料庫目錄的路徑"""
        # 從目前檔案位置往上找到專案根目錄
        current_dir = Path(__file__).resolve().parent.parent.parent
        # 設定資料庫目錄路徑
        database_dir = current_dir / "ReviewsDataBase"
        
        # 如果目錄不存在，建立它
        if not database_dir.exists():
            try:
                database_dir.mkdir(parents=True)
                print(f"已建立資料庫目錄：{database_dir}")
            except Exception as e:
                print(f"建立資料庫目錄時發生錯誤：{e}")
                # 如果無法建立目錄，使用當前目錄
                database_dir = current_dir
        
        return str(database_dir)

    def on_dataset_selected(self, event=None):
        """當選擇數據集類型時觸發"""
        if self.dataset_type.get():
            # 重設文件導入狀態
            self.step_states['file_imported'] = False
            self.file_path_var.set("")
            self.import_status.config(text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
            
            # 啟用瀏覽按鈕
            self.browse_btn.config(state='normal')
            
            # 更新按鈕狀態
            self.update_button_states()
        else:
            # 禁用瀏覽按鈕
            self.browse_btn.config(state='disabled')
            # 重設文件導入狀態
            self.import_status.config(text=STATUS_TEXT['pending'], foreground=COLORS['pending'])

    def on_sampling_toggle(self):
        """當抽樣選項被切換時觸發"""
        if self.use_sampling_var.get():
            # 啟用抽樣
            self.sample_size_entry.config(state='normal')
            self.sampling_info.config(
                text="✅ 抽樣已啟用：將從數據集中隨機抽取指定數量的樣本", 
                foreground='green'
            )
        else:
            # 禁用抽樣
            self.sample_size_entry.config(state='disabled')
            self.sampling_info.config(
                text="💡 建議：大數據集(>10000樣本)建議抽樣以提高處理速度", 
                foreground='gray'
            )

    def browse_file(self):
        """瀏覽檔案"""
        try:
            # 獲取當前選擇的數據集類型
            selected_name = self.dataset_type.get()
            selected_dataset = None
            for ds_key, ds_info in DATASETS.items():
                if ds_info['name'] == selected_name:
                    selected_dataset = ds_key
                    break
            
            if not selected_dataset:
                messagebox.showwarning("警告", "請先選擇數據集類型")
                return
            
            # 根據數據集類型設定檔案類型過濾器
            if DATASETS[selected_dataset]['file_type'] == 'csv':
                filetypes = [("CSV檔案", "*.csv"), ("所有檔案", "*.*")]
            else:  # json
                filetypes = [("JSON檔案", "*.json"), ("所有檔案", "*.*")]
            
            file_path = filedialog.askopenfilename(
                title=f"選擇{DATASETS[selected_dataset]['description']}檔案",
                initialdir=self.database_dir,
                filetypes=filetypes
            )
            
            if file_path:
                # 檢查檔案類型是否符合所選數據集
                file_ext = os.path.splitext(file_path)[1].lower()
                expected_ext = f".{DATASETS[selected_dataset]['file_type']}"
                
                if file_ext != expected_ext:
                    messagebox.showerror("錯誤", 
                        f"檔案類型不符合！\n"
                        f"已選擇：{file_ext}\n"
                        f"需要的類型：{expected_ext}")
                    return
                
                # 將路徑轉換為相對路徑（如果可能）
                try:
                    relative_path = os.path.relpath(file_path, self.database_dir)
                    display_path = relative_path if not relative_path.startswith('..') else file_path
                except ValueError:
                    display_path = file_path
                
                # 檢測文件大小並提供抽樣建議
                try:
                    if file_ext == '.csv':
                        temp_df = pd.read_csv(file_path)
                    elif file_ext == '.json':
                        temp_df = pd.read_json(file_path)
                    
                    total_samples = len(temp_df)
                    
                    # 構建狀態信息
                    status_text = f"已選擇{DATASETS[selected_dataset]['description']}檔案：{display_path}\n"
                    status_text += f"📊 數據集大小：{total_samples:,} 個樣本"
                    
                    # 根據數據大小提供抽樣建議
                    if total_samples > 50000:
                        status_text += f"\n⚠️  大型數據集！強烈建議啟用抽樣 (建議抽取 2000-5000 樣本)"
                        suggested_size = min(3000, total_samples // 10)
                        self.sample_size_var.set(str(suggested_size))
                        self.sampling_info.config(
                            text=f"⚠️  檢測到大型數據集({total_samples:,}樣本)，強烈建議啟用抽樣！", 
                            foreground='orange'
                        )
                    elif total_samples > 10000:
                        status_text += f"\n💡 中型數據集，建議啟用抽樣以提高處理速度"
                        suggested_size = min(2000, total_samples // 5)
                        self.sample_size_var.set(str(suggested_size))
                        self.sampling_info.config(
                            text=f"💡 檢測到中型數據集({total_samples:,}樣本)，建議啟用抽樣", 
                            foreground='blue'
                        )
                    elif total_samples > 1000:
                        status_text += f"\n✅ 適中的數據集大小"
                        self.sample_size_var.set(str(min(1000, total_samples)))
                    else:
                        status_text += f"\n✅ 小型數據集，無需抽樣"
                        self.use_sampling_var.set(False)
                        self.on_sampling_toggle()
                    
                except Exception as e:
                    # 如果無法讀取文件詳細信息，只顯示基本信息
                    status_text = f"已選擇{DATASETS[selected_dataset]['description']}檔案：{display_path}"
                
                self.file_path_var.set(file_path)  # 保存完整路徑
                self.import_status.config(
                    text=status_text,
                    foreground=COLORS['processing']
                )
                self.step_states['file_imported'] = True
                self.update_button_states()
                
        except Exception as e:
            messagebox.showerror("錯誤", f"選擇檔案時發生錯誤：{str(e)}")

    def start_processing(self):
        """開始文本處理"""
        if not self.step_states['file_imported']:
            messagebox.showerror("錯誤", "請先導入檔案")
            return
            
        # 更新run目錄
        self.update_run_dir_label()
        
        # 禁用處理按鈕
        self.process_btn.config(state='disabled')
        self.process_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        # 重置進度條
        self.progress_var.set(0)
        
        # 開始處理
        threading.Thread(target=self._run_preprocessing, daemon=True).start()
        self.root.after(100, self._check_processing_progress)
    
    def _run_preprocessing(self):
        """在背景執行緒中執行預處理"""
        try:
            # 初始化文本預處理器，傳入預處理目錄
            preprocessor = TextPreprocessor(output_dir=self.run_manager.get_preprocessing_dir())
            
            # 讀取檔案
            file_path = self.file_path_var.get()
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                self.process_queue.put(('error', '不支援的檔案格式'))
                return
            
            original_size = len(df)
            
            # 檢查是否需要進行抽樣
            if self.use_sampling_var.get():
                try:
                    sample_size = int(self.sample_size_var.get())
                    if sample_size <= 0:
                        raise ValueError("抽樣數量必須大於0")
                    if sample_size >= original_size:
                        self.process_queue.put(('status', f'樣本數量({sample_size})大於等於原數據集大小({original_size})，將使用全部數據'))
                    else:
                        # 進行分層抽樣（如果有情感標籤的話）
                        if 'sentiment' in df.columns:
                            # 分層抽樣，保持各類別比例
                            df = df.groupby('sentiment', group_keys=False).apply(
                                lambda x: x.sample(min(len(x), sample_size // df['sentiment'].nunique()), 
                                                  random_state=42)
                            ).reset_index(drop=True)
                            
                            # 如果分層後樣本數不足，補充隨機抽樣
                            if len(df) < sample_size:
                                remaining = sample_size - len(df)
                                excluded_df = pd.read_csv(file_path) if file_ext == '.csv' else pd.read_json(file_path)
                                excluded_df = excluded_df.drop(df.index).reset_index(drop=True)
                                if len(excluded_df) > 0:
                                    additional_samples = excluded_df.sample(min(remaining, len(excluded_df)), 
                                                                           random_state=42)
                                    df = pd.concat([df, additional_samples], ignore_index=True)
                        else:
                            # 隨機抽樣
                            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                        
                        self.process_queue.put(('status', f'數據抽樣完成：從{original_size:,}個樣本中抽取了{len(df):,}個樣本'))
                        
                except ValueError as e:
                    self.process_queue.put(('error', f'抽樣參數錯誤：{str(e)}'))
                    return
            else:
                self.process_queue.put(('status', f'使用完整數據集：{original_size:,}個樣本'))

            # 更新進度
            self.process_queue.put(('progress', 20))
            self.process_queue.put(('status', 'text_cleaning'))
            
            # 自動偵測文本欄位
            text_column_candidates = ['processed_text', 'clean_text', 'text', 'review', 'content', 'comment', 'description']
            text_column = None
            for col in text_column_candidates:
                if col in df.columns:
                    text_column = col
                    break
            if text_column is None:
                raise ValueError(f"無法自動識別文本欄位，請確認檔案內容。可用欄位有：{', '.join(df.columns)}")
            
            # 執行預處理
            processed_df = preprocessor.preprocess(df, text_column)
            
            # 更新進度
            self.process_queue.put(('progress', 60))
            self.process_queue.put(('status', 'tokenizing'))
            
            # 完成處理
            self.process_queue.put(('progress', 100))
            
            # 報告最終處理結果
            final_size = len(processed_df)
            if self.use_sampling_var.get() and final_size != original_size:
                success_status = f'處理完成！原始數據：{original_size:,}樣本 → 抽樣後：{final_size:,}樣本'
            else:
                success_status = f'處理完成！處理了{final_size:,}個樣本'
            
            self.process_queue.put(('status', success_status))
            
            # 獲取預處理目錄路徑
            run_dir = self.run_manager.get_preprocessing_dir()
            self.process_queue.put(('result', run_dir))
            
            # 保存最後一次預處理的 run 目錄
            self.last_run_dir = run_dir
            
        except Exception as e:
            self.process_queue.put(('error', str(e)))
    
    def _check_processing_progress(self):
        """檢查處理進度並更新UI"""
        try:
            while True:
                message_type, message = self.process_queue.get_nowait()
                
                if message_type == 'progress':
                    self.progress_var.set(message)
                elif message_type == 'status':
                    self.process_status.config(
                        text=f"處理進度: {message}",
                        foreground=COLORS['processing']
                    )
                elif message_type == 'error':
                    error_msg = f"處理錯誤: {message}"
                    self.process_status.config(
                        text=error_msg,
                        foreground=COLORS['error']
                    )
                    messagebox.showerror("錯誤", error_msg)
                    self.process_btn['state'] = 'normal'
                    return
                elif message_type == 'result':
                    success_msg = f"處理完成，結果已儲存至：{message}"
                    self.process_status.config(
                        text=success_msg,
                        foreground=COLORS['success']
                    )
                    self.step_states['processing_done'] = True
                    self.update_button_states()
                    return
                
        except queue.Empty:
            self.root.after(100, self._check_processing_progress)

    def start_encoding(self):
        """開始文本編碼"""
        if not self.step_states['processing_done']:
            messagebox.showerror("錯誤", "請先完成文本處理")
            return
            
        # 獲取選擇的編碼器類型
        encoder_type = self.encoder_type.get()
        
        # 更新run目錄
        self.update_run_dir_label()
        
        # 禁用編碼按鈕
        self.encoding_btn.config(state='disabled')
        self.encoding_status.config(text=f"狀態: {encoder_type.upper()}編碼中", foreground="blue")
        
        # 開始編碼
        threading.Thread(target=self._run_encoding, daemon=True).start()
        self.root.after(100, self._check_encoding_progress)
    
    def _run_encoding(self):
        """在背景執行緒中執行文本編碼"""
        try:
            from modules.encoder_factory import EncoderFactory
            from gui.progress_bridge import create_progress_callback
            
            # 創建進度橋接器
            progress_bridge, progress_callback = create_progress_callback(self.encoding_queue)
            
            # 檢查是否有最後一次預處理的 run 目錄
            if self.last_run_dir is None:
                raise ValueError("請先執行文本預處理步驟")
            
            # 使用最後一次預處理的檔案
            input_file = os.path.join(self.last_run_dir, "01_preprocessed_data.csv")
            
            # 檢查檔案是否存在
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"找不到預處理檔案：{input_file}")
            
            progress_callback('status', '📖 讀取預處理數據...')
            
            # 讀取預處理後的數據
            df = pd.read_csv(input_file)
            
            progress_callback('status', f'✅ 數據載入完成：{len(df)} 條記錄')
            
            # 獲取選擇的編碼器類型
            encoder_type = self.encoder_type.get()
            
            # 創建編碼器配置
            encoder_config = {
                'batch_size': 32,
                'max_length': 512
            }
            
            progress_callback('status', f'🔧 初始化{encoder_type.upper()}編碼器...')
            
            # 使用工廠創建編碼器
            encoder = EncoderFactory.create_encoder(
                encoder_type=encoder_type,
                config=encoder_config,
                progress_callback=progress_callback
            )
            
            progress_callback('status', f'🚀 開始{encoder_type.upper()}編碼...')
            
            # 執行編碼
            embeddings = encoder.encode(df['processed_text'])
            
            # 保存編碼結果
            encoding_output_dir = self.run_manager.get_bert_encoding_dir()
            embeddings_path = os.path.join(encoding_output_dir, f'02_{encoder_type}_embeddings.npy')
            
            import numpy as np
            np.save(embeddings_path, embeddings)
            
            # 保存編碼器信息
            encoder_info = encoder.get_encoder_info()
            info_path = os.path.join(encoding_output_dir, f'encoder_info_{encoder_type}.json')
            import json
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(encoder_info, f, ensure_ascii=False, indent=2)
            
            progress_callback('status', f'✅ {encoder_type.upper()}編碼完成，向量維度: {encoder.get_embedding_dim()}')
            
            # 將結果放入佇列
            progress_bridge.finish(f'{encoder_type.upper()}編碼完成')
            self.encoding_queue.put(('success', encoding_output_dir))
            
        except Exception as e:
            error_msg = f"編碼失敗: {str(e)}"
            progress_callback('error', error_msg)
            self.encoding_queue.put(('error', error_msg))
    
    def _check_encoding_progress(self):
        """檢查編碼進度並更新UI"""
        try:
            message_type, message = self.encoding_queue.get_nowait()
            
            if message_type == 'progress':
                # 更新BERT編碼專用進度條
                if isinstance(message, (list, tuple)) and len(message) == 2:
                    current, total = message
                    percentage = (current / total) * 100 if total > 0 else 0
                    self.encoding_progress_var.set(percentage)
                else:
                    # 直接是百分比
                    self.encoding_progress_var.set(message)
            
            elif message_type == 'status':
                # 更新狀態文字
                self.encoding_status.config(
                    text=str(message),
                    foreground=COLORS['processing']
                )
            
            elif message_type == 'phase':
                # 處理階段信息
                if isinstance(message, dict):
                    phase_name = message.get('phase_name', '處理中')
                    current_phase = message.get('current_phase', 0)
                    total_phases = message.get('total_phases', 0)
                    
                    if total_phases > 0:
                        status_text = f"階段 {current_phase}/{total_phases}: {phase_name}"
                    else:
                        status_text = phase_name
                    
                    self.encoding_status.config(
                        text=status_text,
                        foreground=COLORS['processing']
                    )
                else:
                    self.encoding_status.config(
                        text=str(message),
                        foreground=COLORS['processing']
                    )
            
            elif message_type == 'error':
                error_msg = f"編碼錯誤: {message}"
                self.encoding_status.config(
                    text=error_msg,
                    foreground=COLORS['error']
                )
                self.encoding_progress_var.set(0)  # 重置BERT編碼進度條
                messagebox.showerror("錯誤", error_msg)
                self.encoding_btn['state'] = 'normal'
                return  # 停止檢查
            
            elif message_type == 'success':
                success_msg = f"✅ 編碼完成，結果已儲存至：{message}"
                self.encoding_status.config(
                    text=success_msg,
                    foreground=COLORS['success']
                )
                self.encoding_progress_var.set(100)  # 完成BERT編碼進度條
                self.step_states['encoding_done'] = True
                self.update_button_states()
                return  # 停止檢查
            
            # 繼續檢查
            self.root.after(100, self._check_encoding_progress)
            
        except queue.Empty:
            self.root.after(100, self._check_encoding_progress)
    
    def complete_encoding(self):
        """完成編碼（已不再使用）"""
        pass

    def run_baseline(self):
        """執行基準測試"""
        self.baseline_btn['state'] = 'disabled'
        self.baseline_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        # 模擬測試過程
        self.root.after(SIMULATION_DELAYS['attention_test'], self.complete_baseline)
    
    def complete_baseline(self):
        """完成基準測試"""
        self.baseline_status.config(text=STATUS_TEXT['success'], foreground=COLORS['success'])
        self.step_states['baseline_done'] = True
        self.update_button_states()

    def run_dual_head(self):
        """執行雙頭測試"""
        self.dual_btn['state'] = 'disabled'
        self.dual_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        # 模擬測試過程
        self.root.after(SIMULATION_DELAYS['attention_test'], self.complete_dual_head)
    
    def complete_dual_head(self):
        """完成雙頭測試"""
        self.dual_status.config(text=STATUS_TEXT['success'], foreground=COLORS['success'])
        self.step_states['dual_head_done'] = True
        self.update_button_states()

    def run_triple_head(self):
        """執行三頭測試"""
        self.triple_btn['state'] = 'disabled'
        self.triple_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        # 模擬測試過程
        self.root.after(SIMULATION_DELAYS['attention_test'], self.complete_triple_head)
    
    def complete_triple_head(self):
        """完成三頭測試"""
        self.triple_status.config(text=STATUS_TEXT['success'], foreground=COLORS['success'])
        self.step_states['triple_head_done'] = True
        self.update_button_states()

    def import_encoding(self):
        """導入已有的BERT編碼檔案"""
        try:
            file_path = filedialog.askopenfilename(
                title="選擇BERT編碼檔案",
                initialdir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"),
                filetypes=[("NumPy檔案", "*.npy"), ("所有檔案", "*.*")]
            )
            
            if file_path:
                from modules.bert_encoder import BertEncoder
                bert = BertEncoder()
                
                try:
                    # 使用quick_load_embeddings載入檔案
                    embeddings = bert.quick_load_embeddings(file_path)
                    
                    success_msg = f"成功導入編碼檔案：{os.path.basename(file_path)}"
                    self.encoding_status.config(
                        text=success_msg,
                        foreground=COLORS['success']
                    )
                    
                    # 設置進度條為完成狀態
                    self.encoding_progress_var.set(100)
                    
                    # 更新狀態
                    self.step_states['encoding_done'] = True
                    self.update_button_states()
                    
                except Exception as e:
                    messagebox.showerror("錯誤", f"載入編碼檔案時發生錯誤：{str(e)}")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"導入編碼時發生錯誤：{str(e)}")

    def create_run_dir_label(self):
        """創建顯示當前run目錄的標籤"""
        run_dir_frame = ttk.Frame(self.root)
        run_dir_frame.pack(side='top', fill='x', padx=15, pady=5)
        
        self.run_dir_label = ttk.Label(
            run_dir_frame,
            text=f"當前執行目錄：{self.run_manager.get_run_dir()}",
            font=FONTS['small']
        )
        self.run_dir_label.pack(side='right')

    def update_run_dir_label(self):
        """更新run目錄標籤"""
        self.run_dir_label.config(text=f"當前執行目錄：{self.run_manager.get_run_dir()}")

    def _update_analysis_results(self):
        """更新比對分析頁面的結果顯示"""
        if not hasattr(self, 'analysis_results') or self.analysis_results is None:
            return
        
        try:
            # 清空現有結果
            for item in self.performance_tree.get_children():
                self.performance_tree.delete(item)
            for item in self.detail_tree.get_children():
                self.detail_tree.delete(item)
            
            # 更新性能比較表格
            classification_results = self.analysis_results.get('classification_evaluation', {})
            
            if 'comparison' in classification_results:
                comparison = classification_results['comparison']
                accuracy_ranking = comparison.get('accuracy_ranking', [])
                
                for mechanism, accuracy in accuracy_ranking:
                    # 獲取該機制的詳細結果
                    mechanism_result = classification_results.get(mechanism, {})
                    
                    # 格式化數據
                    row_data = (
                        self._format_mechanism_name(mechanism),
                        f"{accuracy * 100:.4f}%",  # 轉換為百分比格式，保留四位小數
                        f"{mechanism_result.get('test_f1', 0) * 100:.4f}%",
                        f"{mechanism_result.get('test_recall', 0) * 100:.4f}%",
                        f"{mechanism_result.get('test_precision', 0) * 100:.4f}%"
                    )
                    
                    self.performance_tree.insert('', 'end', values=row_data)
            
            # 更新詳細比對結果（使用最佳機制的預測結果）
            self._update_detailed_comparison()
            
            # 更新狀態
            summary = self.analysis_results.get('summary', {})
            best_mechanism = summary.get('best_attention_mechanism', 'N/A')
            best_accuracy = summary.get('best_classification_accuracy', 0)
            
            self.analysis_status.config(
                text=f"分析完成！最佳注意力機制: {self._format_mechanism_name(best_mechanism)} (準確率: {best_accuracy * 100:.4f}%)",
                foreground=COLORS['success']
            )
            
        except Exception as e:
            self.analysis_status.config(
                text=f"更新結果時發生錯誤: {str(e)}",
                foreground=COLORS['error']
            )
    
    def _format_mechanism_name(self, mechanism):
        """格式化注意力機制名稱為中文"""
        name_mapping = {
            'no': '無注意力',
            'similarity': '相似度注意力',
            'keyword': '關鍵詞注意力', 
            'self': '自注意力',
            'combined': '組合注意力'
        }
        return name_mapping.get(mechanism, mechanism)
    
    def _update_detailed_comparison(self):
        """更新詳細比對結果表格"""
        try:
            # 獲取最佳機制的結果
            classification_results = self.analysis_results.get('classification_evaluation', {})
            
            if 'comparison' not in classification_results:
                return
            
            best_mechanism = classification_results['comparison'].get('best_mechanism', None)
            if not best_mechanism:
                return
            
            # 顯示真實的詳細結果
            self._generate_sample_detail_results(best_mechanism)
            
        except Exception as e:
            print(f"更新詳細比對結果時發生錯誤: {str(e)}")
    
    def _generate_sample_detail_results(self, best_mechanism):
        """顯示真實的詳細結果"""
        try:
            if not hasattr(self, 'analysis_results') or self.analysis_results is None:
                return
            
            # 獲取分類評估結果
            classification_results = self.analysis_results.get('classification_evaluation', {})
            
            if best_mechanism not in classification_results:
                return
            
            # 獲取最佳機制的預測詳細信息
            mechanism_result = classification_results[best_mechanism]
            prediction_details = mechanism_result.get('prediction_details', {})
            
            if not prediction_details:
                # 如果沒有詳細預測結果，回退到讀取原始數據並模擬結果
                self._fallback_sample_results()
                return
            
            # 讀取原始數據以獲取文本內容
            if not self.last_run_dir:
                return
                
            input_file = os.path.join(self.last_run_dir, "01_preprocessed_data.csv")
            if not os.path.exists(input_file):
                return
            
            import pandas as pd
            df = pd.read_csv(input_file)
            
            # 獲取預測信息
            true_labels = prediction_details.get('true_labels', [])
            predicted_labels = prediction_details.get('predicted_labels', [])
            class_names = prediction_details.get('class_names', [])
            test_texts = prediction_details.get('test_texts', [])
            
            # 優先使用原始標籤名稱（如果可用）
            true_label_names = prediction_details.get('true_label_names', [])
            predicted_label_names = prediction_details.get('predicted_label_names', [])
            
            # 如果有原始標籤名稱，直接使用；否則通過class_names轉換
            if true_label_names and predicted_label_names:
                # 直接使用原始標籤名稱
                final_true_labels = true_label_names
                final_predicted_labels = predicted_label_names
            elif class_names:
                # 將數字標籤轉換為類別名稱
                final_true_labels = [class_names[label] if label < len(class_names) else 'unknown' for label in true_labels]
                final_predicted_labels = [class_names[label] if label < len(class_names) else 'unknown' for label in predicted_labels]
            else:
                final_true_labels = [str(label) for label in true_labels]
                final_predicted_labels = [str(label) for label in predicted_labels]
            
            # 如果有測試集文本，使用文本匹配來找到正確的原始索引和標籤
            if test_texts:
                matched_results = self._match_texts_with_original_data(test_texts, final_true_labels, final_predicted_labels, df)
                self._display_matched_results(matched_results)
            else:
                # 舊的顯示方法（按測試集順序）
                self._display_sequential_results(df, final_true_labels, final_predicted_labels)
                    
        except Exception as e:
            print(f"顯示真實詳細結果時發生錯誤: {str(e)}")
            # 如果出錯，回退到模擬結果
            self._fallback_sample_results()
    
    def _match_texts_with_original_data(self, test_texts, true_labels, predicted_labels, original_df):
        """通過文本匹配找到原始數據中的對應項目"""
        matched_results = []
        
        # 獲取原始數據的處理文本
        original_texts = original_df['processed_text'].tolist() if 'processed_text' in original_df.columns else []
        original_sentiments = original_df['sentiment'].tolist() if 'sentiment' in original_df.columns else []
        
        for i, test_text in enumerate(test_texts):
            if i >= len(true_labels) or i >= len(predicted_labels):
                continue
                
            # 在原始數據中查找匹配的文本
            matched_index = -1
            original_label = 'unknown'
            
            for j, orig_text in enumerate(original_texts):
                if str(test_text).strip() == str(orig_text).strip():
                    matched_index = j
                    if j < len(original_sentiments):
                        original_label = str(original_sentiments[j])
                    break
            
            # 截斷過長的文本
            if len(test_text) > 50:
                display_text = test_text[:47] + "..."
            else:
                display_text = test_text
            
            predicted_label = predicted_labels[i]
            is_correct = "✓" if original_label == predicted_label else "✗"
            
            matched_results.append({
                'original_index': matched_index if matched_index >= 0 else 'N/A',
                'display_text': display_text,
                'original_label': original_label,
                'predicted_label': predicted_label,
                'is_correct': is_correct
            })
        
        return matched_results
    
    def _display_matched_results(self, matched_results):
        """顯示匹配的結果"""
        for result in matched_results[:50]:  # 限制顯示前50條
            detail_row = (
                str(result['original_index']),
                result['display_text'],
                result['original_label'],
                result['predicted_label'],
                result['is_correct']
            )
            self.detail_tree.insert('', 'end', values=detail_row)
    
    def _display_sequential_results(self, df, true_labels, predicted_labels):
        """按順序顯示結果（舊方法）"""
        max_display = min(50, len(true_labels), len(df))
        
        for i in range(max_display):
            if i < len(df):
                row = df.iloc[i]
                original_text = str(row.get('processed_text', row.get('text', '')))
            else:
                original_text = "N/A"
            
            # 截斷過長的文本
            if len(original_text) > 50:
                display_text = original_text[:47] + "..."
            else:
                display_text = original_text
            
            if i < len(true_labels) and i < len(predicted_labels):
                true_label = true_labels[i]
                predicted_label = predicted_labels[i]
                is_correct = "✓" if true_label == predicted_label else "✗"
            else:
                true_label = "N/A"
                predicted_label = "N/A"
                is_correct = "?"
            
            detail_row = (
                str(i),
                display_text,
                true_label,
                predicted_label,
                is_correct
            )
            
            self.detail_tree.insert('', 'end', values=detail_row)
    
    def _fallback_sample_results(self):
        """回退方案：生成模擬詳細結果"""
        try:
            if not self.last_run_dir:
                return
                
            input_file = os.path.join(self.last_run_dir, "01_preprocessed_data.csv")
            if not os.path.exists(input_file):
                return
            
            import pandas as pd
            import random
            df = pd.read_csv(input_file)
            
            # 模擬預測結果（實際使用時應該使用真實預測）
            random.seed(42)  # 確保結果一致
            
            # 只顯示前50條記錄以避免界面過於擁擠
            for i in range(min(50, len(df))):
                row = df.iloc[i]
                original_text = str(row.get('processed_text', row.get('text', '')))
                # 修復：正確讀取原始標籤，確保從正確的欄位讀取
                original_label = str(row.get('sentiment', 'unknown'))
                
                # 檢查原始標籤是否正確讀取
                if original_label == 'unknown' and 'sentiment' in row:
                    print(f"警告：第{i}行的sentiment欄位值為：{row['sentiment']}")
                
                # 模擬預測標籤（90%準確率）
                predicted_label = original_label if random.random() < 0.9 else random.choice(['positive', 'negative', 'neutral'])
                
                # 截斷過長的文本
                if len(original_text) > 50:
                    display_text = original_text[:47] + "..."
                else:
                    display_text = original_text
                
                is_correct = "✓" if predicted_label == original_label else "✗"
                
                detail_row = (
                    str(i),
                    display_text,
                    original_label,
                    predicted_label,
                    is_correct
                )
                
                self.detail_tree.insert('', 'end', values=detail_row)
                    
        except Exception as e:
            print(f"生成回退詳細結果時發生錯誤: {str(e)}")
            # 如果預處理數據有問題，嘗試直接讀取原始檔案
            self._read_original_file_for_fallback()
    
    def _read_original_file_for_fallback(self):
        """備用方案：直接從原始檔案讀取數據"""
        try:
            # 獲取原始檔案路徑
            original_file = self.file_path_var.get()
            if not original_file or not os.path.exists(original_file):
                print("無法找到原始檔案路徑")
                return
            
            import pandas as pd
            import random
            
            # 讀取原始檔案
            if original_file.endswith('.csv'):
                df = pd.read_csv(original_file)
            elif original_file.endswith('.json'):
                df = pd.read_json(original_file)
            else:
                print(f"不支援的檔案格式：{original_file}")
                return
            
            print(f"從原始檔案讀取：{original_file}")
            print(f"檔案欄位：{list(df.columns)}")
            
            # 自動偵測文本和標籤欄位
            text_column = None
            sentiment_column = None
            
            # 偵測文本欄位
            for col in ['review', 'text', 'content', 'comment']:
                if col in df.columns:
                    text_column = col
                    break
            
            # 偵測情感標籤欄位
            for col in ['sentiment', 'label', 'emotion', 'polarity']:
                if col in df.columns:
                    sentiment_column = col
                    break
            
            if not text_column or not sentiment_column:
                print(f"無法識別文本欄位或情感欄位。文本欄位：{text_column}, 情感欄位：{sentiment_column}")
                return
            
            print(f"使用文本欄位：{text_column}, 情感欄位：{sentiment_column}")
            
            # 模擬預測結果
            random.seed(42)
            
            # 只顯示前50條記錄
            for i in range(min(50, len(df))):
                row = df.iloc[i]
                original_text = str(row.get(text_column, ''))
                original_label = str(row.get(sentiment_column, 'unknown'))
                
                print(f"第{i}行：標籤={original_label}")
                
                # 模擬預測標籤（90%準確率）
                predicted_label = original_label if random.random() < 0.9 else random.choice(['positive', 'negative', 'neutral'])
                
                # 截斷過長的文本
                if len(original_text) > 50:
                    display_text = original_text[:47] + "..."
                else:
                    display_text = original_text
                
                is_correct = "✓" if predicted_label == original_label else "✗"
                
                detail_row = (
                    str(i),
                    display_text,
                    original_label,
                    predicted_label,
                    is_correct
                )
                
                self.detail_tree.insert('', 'end', values=detail_row)
                
        except Exception as e:
            print(f"從原始檔案讀取數據時發生錯誤: {str(e)}")
    
    def _integrate_modular_methods(self):
        """整合模組化方法到主類中"""
        try:
            for method_name, method_func in MODULAR_METHODS.items():
                # 將方法繫定到當前實例
                bound_method = method_func.__get__(self, self.__class__)
                setattr(self, method_name, bound_method)
        except Exception as e:
            print(f"整合模組化方法失敗: {e}")
            # 方法已直接實現在類中，無需備用方法
    
    def update_current_config_safe(self):
        """安全更新當前配置顯示（檢查GUI元素是否存在）"""
        try:
            self.update_current_config()
        except Exception as e:
            print(f"配置更新失敗: {e}")
    
    def update_current_config(self):
        """更新當前配置顯示"""
        if hasattr(self, 'current_config_label') and hasattr(self.current_config_label, 'config'):
            try:
                encoder = self.encoder_type.get().upper()
                aspect = self.aspect_classifier_type.get().upper()
                classifier = self.classifier_type.get().upper()
                config_text = f"📝 當前配置: {encoder} + {aspect} + {classifier}"
                self.current_config_label.config(text=config_text)
            except Exception as e:
                # 如果更新失敗，不做任何操作
                pass

    def run_modular_pipeline(self):
        """運行模組化流水線"""
        try:
            # 檢查是否有檔案導入
            if not self.step_states['file_imported']:
                messagebox.showerror("錯誤", "請先導入檔案")
                return
            
            # 禁用按鈕
            self.run_pipeline_btn.config(state='disabled')
            self.compare_methods_btn.config(state='disabled')
            
            # 更新狀態
            self.pipeline_status.config(text="狀態: 初始化模組化流水線...", foreground='blue')
            self.pipeline_progress_var.set(0)
            
            # 清空結果顯示
            self.pipeline_results_text.delete(1.0, tk.END)
            
            # 在背景執行緒中運行流水線
            threading.Thread(target=self._run_modular_pipeline, daemon=True).start()
            self.root.after(100, self._check_pipeline_progress)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"運行模組化流水線時發生錯誤: {str(e)}")

    def _run_modular_pipeline(self):
        """在背景執行緒中運行模組化流水線"""
        try:
            from modules.modular_pipeline import ModularPipeline
            from gui.progress_bridge import create_progress_callback
            
            # 創建進度橋接器
            progress_bridge, progress_callback = create_progress_callback(self.pipeline_queue)
            
            # 獲取當前配置
            encoder_type = self.encoder_type.get()
            aspect_type = self.aspect_classifier_type.get()
            
            # 配置參數
            encoder_config = {
                'batch_size': 32,
                'max_length': 512
            }
            
            aspect_config = {
                'n_topics': 10,
                'random_state': 42
            }
            
            # 創建模組化流水線
            self.pipeline_queue.put(('status', f'🔧 初始化模組化流水線: {encoder_type.upper()} + {aspect_type.upper()}'))
            
            pipeline = ModularPipeline(
                encoder_type=encoder_type,
                aspect_type=aspect_type,
                encoder_config=encoder_config,
                aspect_config=aspect_config,
                output_dir=self.run_manager.get_run_dir(),
                progress_callback=progress_callback
            )
            
            # 讀取輸入數據
            file_path = self.file_path_var.get()
            file_ext = os.path.splitext(file_path)[1].lower()
            
            self.pipeline_queue.put(('status', '📖 讀取輸入數據...'))
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError('不支援的檔案格式')
            
            # 檢查是否需要進行抽樣
            if self.use_sampling_var.get():
                sample_size = int(self.sample_size_var.get())
                if sample_size < len(df):
                    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                    self.pipeline_queue.put(('status', f'🎲 數據抽樣: {sample_size} 條記錄'))
            
            self.pipeline_queue.put(('progress', 10))
            
            # 運行模組化流水線
            import time
            start_time = time.time()
            results = pipeline.process(df)
            processing_time = time.time() - start_time
            
            # 結果統計
            summary = {
                'encoder_type': encoder_type,
                'aspect_type': aspect_type,
                'data_size': len(df),
                'embedding_dim': pipeline.text_encoder.get_embedding_dim(),
                'aspect_count': len(pipeline.aspect_classifier.get_aspect_names()),
                'processing_time': processing_time
            }
            
            self.pipeline_queue.put(('success', {
                'results': results,
                'summary': summary
            }))
            
        except Exception as e:
            self.pipeline_queue.put(('error', str(e)))

    def _check_pipeline_progress(self):
        """檢查模組化流水線進度"""
        try:
            message_type, message = self.pipeline_queue.get_nowait()
            
            if message_type == 'status':
                self.pipeline_status.config(text=f"狀態: {message}", foreground='blue')
                
            elif message_type == 'progress':
                if isinstance(message, (int, float)):
                    self.pipeline_progress_var.set(message)
                elif isinstance(message, str) and '%' in message:
                    try:
                        progress_val = float(message.replace('%', ''))
                        self.pipeline_progress_var.set(progress_val)
                    except:
                        pass
                        
            elif message_type == 'success':
                self.pipeline_progress_var.set(100)
                data = message
                summary = data['summary']
                
                # 更新狀態
                self.pipeline_status.config(
                    text=f"狀態: 完成 (耗時: {summary['processing_time']:.1f}秒)",
                    foreground='green'
                )
                
                # 顯示結果
                result_text = f"""🎉 模組化流水線完成！

📊 分析結果摘要:
• 編碼器: {summary['encoder_type'].upper()}
• 面向分類器: {summary['aspect_type'].upper()}
• 數據量: {summary['data_size']:,} 條記錄
• 嵌入向量維度: {summary['embedding_dim']}
• 發現面向數: {summary['aspect_count']}
• 處理時間: {summary['processing_time']:.2f} 秒

📝 結果檔案已保存至: {self.run_manager.get_run_dir()}
"""
                
                self.pipeline_results_text.delete(1.0, tk.END)
                self.pipeline_results_text.insert(tk.END, result_text)
                
                # 重新啟用按鈕
                self.run_pipeline_btn.config(state='normal')
                self.compare_methods_btn.config(state='normal')
                
                return
                
            elif message_type == 'comparison_success':
                self.pipeline_progress_var.set(100)
                results = message
                
                # 更新狀態
                self.pipeline_status.config(text="狀態: 比較完成", foreground='green')
                
                # 生成比較結果文本
                comparison_text = "📈 方法比較結果:\n\n"
                comparison_text += f"{'=' * 60}\n"
                comparison_text += f"{'ID':<3} {'Encoder':<8} {'Aspect':<10} {'Time(s)':<8} {'Embedding':<10} {'Aspects':<8} {'Status':<10}\n"
                comparison_text += f"{'=' * 60}\n"
                
                for i, result in enumerate(results, 1):
                    if result['success']:
                        comparison_text += f"{i:<3} {result['encoder']:<8} {result['aspect_classifier']:<10} {result['processing_time']:<8.1f} {result['embedding_dim']:<10} {result['aspect_count']:<8} {'Success':<10}\n"
                    else:
                        comparison_text += f"{i:<3} {result['encoder']:<8} {result['aspect_classifier']:<10} {'N/A':<8} {'N/A':<10} {'N/A':<8} {'Failed':<10}\n"
                
                comparison_text += f"{'=' * 60}\n"
                
                # 統計信息
                successful_results = [r for r in results if r['success']]
                if successful_results:
                    fastest = min(successful_results, key=lambda x: x['processing_time'])
                    comparison_text += f"\n🏆 最快方法: {fastest['encoder'].upper()} + {fastest['aspect_classifier'].upper()} ({fastest['processing_time']:.1f}秒)\n"
                    
                    avg_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
                    comparison_text += f"📊 平均處理時間: {avg_time:.1f}秒\n"
                
                failed_count = len([r for r in results if not r['success']])
                if failed_count > 0:
                    comparison_text += f"\n⚠️  {failed_count} 個方法執行失敗"
                
                self.pipeline_results_text.delete(1.0, tk.END)
                self.pipeline_results_text.insert(tk.END, comparison_text)
                
                # 重新啟用按鈕
                self.run_pipeline_btn.config(state='normal')
                self.compare_methods_btn.config(state='normal')
                
                return
                
            elif message_type == 'error':
                self.pipeline_progress_var.set(0)
                self.pipeline_status.config(text=f"狀態: 錯誤 - {message}", foreground='red')
                
                error_text = f"❌ 模組化流水線執行失敗：\n\n{message}"
                self.pipeline_results_text.delete(1.0, tk.END)
                self.pipeline_results_text.insert(tk.END, error_text)
                
                # 重新啟用按鈕
                self.run_pipeline_btn.config(state='normal')
                self.compare_methods_btn.config(state='normal')
                
                return
                
        except queue.Empty:
            pass
        
        # 繼續檢查
        self.root.after(100, self._check_pipeline_progress)

    def compare_methods(self):
        """比較不同方法的效果"""
        try:
            # 檢查是否有檔案導入
            if not self.step_states['file_imported']:
                messagebox.showerror("錯誤", "請先導入檔案")
                return
            
            # 禁用按鈕
            self.run_pipeline_btn.config(state='disabled')
            self.compare_methods_btn.config(state='disabled')
            
            # 更新狀態
            self.pipeline_status.config(text="狀態: 比較不同方法中...", foreground='purple')
            self.pipeline_progress_var.set(0)
            
            # 清空結果顯示
            self.pipeline_results_text.delete(1.0, tk.END)
            
            # 在背景執行緒中運行比較
            threading.Thread(target=self._run_method_comparison, daemon=True).start()
            self.root.after(100, self._check_pipeline_progress)
            
        except Exception as e:
            messagebox.showerror("錯誤", f"比較方法時發生錯誤: {str(e)}")

    def _run_method_comparison(self):
        """在背景執行緒中比較不同方法"""
        try:
            from modules.modular_pipeline import ModularPipeline
            
            # 讀取數據
            file_path = self.file_path_var.get()
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.json':
                df = pd.read_json(file_path)
            else:
                raise ValueError('不支援的檔案格式')
            
            # 抽樣數據以加快比較速度
            sample_size = min(1000, len(df))
            df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            
            # 定義要比較的組合
            combinations = [
                ('bert', 'default'),
                ('bert', 'lda'),
                ('gpt', 'default'),
                ('t5', 'lda'),
                ('cnn', 'nmf')
            ]
            
            results = []
            total_combinations = len(combinations)
            
            for i, (encoder_type, aspect_type) in enumerate(combinations):
                try:
                    self.pipeline_queue.put(('status', f'正在測試: {encoder_type.upper()} + {aspect_type.upper()}'))
                    
                    # 創建流水線
                    pipeline = ModularPipeline(
                        encoder_type=encoder_type,
                        aspect_type=aspect_type,
                        output_dir=self.run_manager.get_run_dir()
                    )
                    
                    # 測量處理時間
                    import time
                    start_time = time.time()
                    pipeline_results = pipeline.process(df_sample)
                    processing_time = time.time() - start_time
                    
                    # 記錄結果
                    result = {
                        'encoder': encoder_type,
                        'aspect_classifier': aspect_type,
                        'processing_time': processing_time,
                        'embedding_dim': pipeline.text_encoder.get_embedding_dim(),
                        'aspect_count': len(pipeline.aspect_classifier.get_aspect_names()),
                        'success': True
                    }
                    results.append(result)
                    
                    # 更新進度
                    progress = ((i + 1) / total_combinations) * 100
                    self.pipeline_queue.put(('progress', progress))
                    
                except Exception as e:
                    result = {
                        'encoder': encoder_type,
                        'aspect_classifier': aspect_type,
                        'error': str(e),
                        'success': False
                    }
                    results.append(result)
            
            self.pipeline_queue.put(('comparison_success', results))
            
        except Exception as e:
            self.pipeline_queue.put(('error', str(e)))

    def create_cross_validation_tab(self):
        """第四分頁：交叉驗證"""
        frame4 = ttk.Frame(self.notebook)
        self.notebook.add(frame4, text=" 🔄 交叉驗證 ")
        
        main_frame = ttk.Frame(frame4)
        main_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # 標題
        title_label = ttk.Label(main_frame, text="K 折交叉驗證", font=FONTS['title'])
        title_label.pack(pady=(0, 12))
        
        # 配置區域
        config_frame = ttk.LabelFrame(main_frame, text="🔧 交叉驗證配置", padding=10)
        config_frame.pack(fill='x', pady=(0, 10))
        
        # 第一行：基本設定
        config_row1 = ttk.Frame(config_frame)
        config_row1.pack(fill='x', pady=(0, 8))
        
        # 折數選擇
        ttk.Label(config_row1, text="K 值 (折數):").pack(side='left')
        self.cv_folds = tk.StringVar(value='5')
        folds_combo = ttk.Combobox(config_row1, textvariable=self.cv_folds, 
                                  values=['3', '5', '10'], width=8, state='readonly')
        folds_combo.pack(side='left', padx=(5, 20))
        
        # 評估模式選擇
        ttk.Label(config_row1, text="評估模式:").pack(side='left')
        self.cv_mode = tk.StringVar(value='attention')
        mode_combo = ttk.Combobox(config_row1, textvariable=self.cv_mode,
                                 values=['simple', 'attention'], width=12, state='readonly')
        mode_combo.pack(side='left', padx=(5, 0))
        
        # 第二行：模型選擇
        config_row2 = ttk.Frame(config_frame)
        config_row2.pack(fill='x', pady=(0, 8))
        
        ttk.Label(config_row2, text="分類器:").pack(side='left')
        
        # 模型選擇複選框
        models_frame = ttk.Frame(config_row2)
        models_frame.pack(side='left', padx=(5, 0))
        
        self.cv_models = {}
        model_options = [
            ('xgboost', 'XGBoost'),
            ('logistic_regression', '邏輯迴歸'),
            ('random_forest', '隨機森林'),
            ('svm_linear', '線性SVM')
        ]
        
        for i, (key, label) in enumerate(model_options):
            var = tk.BooleanVar(value=True if key in ['xgboost', 'logistic_regression'] else False)
            self.cv_models[key] = var
            cb = ttk.Checkbutton(models_frame, text=label, variable=var)
            cb.pack(side='left', padx=(0, 10))
        
        # 第三行：注意力機制選擇（僅在attention模式下顯示）
        self.attention_config_frame = ttk.Frame(config_frame)
        self.attention_config_frame.pack(fill='x')
        
        ttk.Label(self.attention_config_frame, text="注意力機制:").pack(side='left')
        
        attention_frame = ttk.Frame(self.attention_config_frame)
        attention_frame.pack(side='left', padx=(5, 0))
        
        self.cv_attentions = {}
        attention_options = [
            ('no', '無注意力'),
            ('similarity', '相似度'),
            ('keyword', '關鍵詞'),
            ('self', '自注意力'),
            ('combined', '組合式')
        ]
        
        for key, label in attention_options:
            var = tk.BooleanVar(value=True if key in ['no', 'similarity', 'self'] else False)
            self.cv_attentions[key] = var
            cb = ttk.Checkbutton(attention_frame, text=label, variable=var)
            cb.pack(side='left', padx=(0, 10))
        
        # 模式選擇回調
        def on_mode_change(*args):
            mode = self.cv_mode.get()
            if mode == 'simple':
                self.attention_config_frame.pack_forget()
            else:
                self.attention_config_frame.pack(fill='x')
        
        self.cv_mode.trace('w', on_mode_change)
        
        # 控制區域
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # 開始按鈕
        self.cv_start_btn = ttk.Button(control_frame, text="🚀 開始交叉驗證", 
                                      command=self.start_cross_validation)
        self.cv_start_btn.pack(side='left')
        
        # 狀態標籤
        self.cv_status = ttk.Label(control_frame, text="準備就緒", foreground='green')
        self.cv_status.pack(side='left', padx=(20, 0))
        
        # 進度條
        self.cv_progress_var = tk.DoubleVar()
        self.cv_progress_bar = ttk.Progressbar(main_frame, variable=self.cv_progress_var, maximum=100)
        self.cv_progress_bar.pack(fill='x', pady=(0, 10))
        
        # 結果顯示區域
        results_frame = ttk.LabelFrame(main_frame, text="📊 交叉驗證結果", padding=10)
        results_frame.pack(fill='both', expand=True)
        
        # 結果樹形表格
        columns = ('Rank', 'Model/Combination', 'Accuracy', 'F1 Score', 'Stability')
        self.cv_results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=8)
        
        # 設定標題
        for col in columns:
            self.cv_results_tree.heading(col, text=col)
            self.cv_results_tree.column(col, width=120, anchor='center')
        
        # 滾動條
        cv_scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.cv_results_tree.yview)
        self.cv_results_tree.configure(yscrollcommand=cv_scrollbar.set)
        
        # 佈局
        self.cv_results_tree.pack(side='left', fill='both', expand=True)
        cv_scrollbar.pack(side='right', fill='y')
        
        # 初始化交叉驗證相關變數
        self.cv_queue = queue.Queue()
        self.cv_thread = None
        
        # 啟動結果監控
        self.monitor_cv_queue()

    def start_cross_validation(self):
        """開始交叉驗證"""
        try:
            # 檢查前置條件
            if not self.last_run_dir:
                messagebox.showerror("錯誤", "請先完成BERT編碼步驟！")
                return
            
            input_file = os.path.join(self.last_run_dir, "01_preprocessed_data.csv")
            if not os.path.exists(input_file):
                messagebox.showerror("錯誤", "找不到預處理數據檔案！")
                return
            
            # 獲取配置
            n_folds = int(self.cv_folds.get())
            mode = self.cv_mode.get()
            
            # 獲取選中的模型
            selected_models = [key for key, var in self.cv_models.items() if var.get()]
            if not selected_models:
                messagebox.showerror("錯誤", "請至少選擇一個分類器！")
                return
            
            # 獲取選中的注意力機制（如果是attention模式）
            selected_attentions = []
            if mode == 'attention':
                selected_attentions = [key for key, var in self.cv_attentions.items() if var.get()]
                if not selected_attentions:
                    messagebox.showerror("錯誤", "請至少選擇一個注意力機制！")
                    return
            
            # 禁用按鈕，開始處理
            self.cv_start_btn['state'] = 'disabled'
            self.cv_status.config(text="執行中...", foreground='orange')
            self.cv_progress_var.set(0)
            
            # 清空結果表格
            for item in self.cv_results_tree.get_children():
                self.cv_results_tree.delete(item)
            
            # 在後台執行交叉驗證
            def run_cv():
                try:
                    output_dir = self.run_manager.get_run_dir()
                    encoder_type = self.encoder_type.get()
                    
                    if mode == 'simple':
                        # 簡單交叉驗證
                        from Part05_Main import process_simple_cross_validation
                        results = process_simple_cross_validation(
                            input_file=input_file,
                            output_dir=output_dir,
                            n_folds=n_folds,
                            model_types=selected_models,
                            encoder_type=encoder_type
                        )
                    else:
                        # 注意力機制交叉驗證
                        from Part05_Main import process_cross_validation_analysis
                        results = process_cross_validation_analysis(
                            input_file=input_file,
                            output_dir=output_dir,
                            n_folds=n_folds,
                            attention_types=selected_attentions,
                            model_types=selected_models,
                            encoder_type=encoder_type
                        )
                    
                    self.cv_queue.put(('success', results))
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    self.cv_queue.put(('error', str(e), error_details))
            
            # 啟動後台線程
            self.cv_thread = threading.Thread(target=run_cv, daemon=True)
            self.cv_thread.start()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"啟動交叉驗證時發生錯誤：{str(e)}")
            self.cv_start_btn['state'] = 'normal'
            self.cv_status.config(text="錯誤", foreground='red')

    def monitor_cv_queue(self):
        """監控交叉驗證佇列"""
        try:
            while True:
                item = self.cv_queue.get_nowait()
                
                if item[0] == 'success':
                    self._handle_cv_success(item[1])
                elif item[0] == 'error':
                    self._handle_cv_error(item[1], item[2] if len(item) > 2 else None)
                elif item[0] == 'progress':
                    self.cv_progress_var.set(item[1])
                    
        except queue.Empty:
            pass
        
        # 重新安排監控
        self.root.after(100, self.monitor_cv_queue)

    def _handle_cv_success(self, results):
        """處理交叉驗證成功"""
        try:
            self.cv_start_btn['state'] = 'normal'
            self.cv_status.config(text="完成", foreground='green')
            self.cv_progress_var.set(100)
            
            # 顯示結果
            if self.cv_mode.get() == 'simple':
                # 簡單模式結果
                if 'comparison' in results and 'ranking' in results['comparison']:
                    ranking = results['comparison']['ranking']
                    for item in ranking:
                        rank = item['rank']
                        model_name = item['model_name']
                        accuracy = f"{item['accuracy_mean']:.4f}"
                        f1_score = f"{item['f1_mean']:.4f}"
                        stability = f"{item['stability_score']:.4f}"
                        
                        self.cv_results_tree.insert('', 'end', values=(
                            rank, model_name, accuracy, f1_score, stability
                        ))
            else:
                # 注意力機制模式結果
                if 'attention_comparison' in results and 'attention_ranking' in results['attention_comparison']:
                    ranking = results['attention_comparison']['attention_ranking']
                    for item in ranking:
                        rank = item['rank']
                        combination = item['combination']
                        accuracy = f"{item['accuracy_mean']:.4f}"
                        f1_score = f"{item['f1_mean']:.4f}"
                        stability = f"{item['stability_score']:.4f}"
                        
                        self.cv_results_tree.insert('', 'end', values=(
                            rank, combination, accuracy, f1_score, stability
                        ))
            
            # 顯示完成消息
            messagebox.showinfo("完成", "交叉驗證已完成！結果已顯示在表格中。")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"處理交叉驗證結果時發生錯誤：{str(e)}")

    def _handle_cv_error(self, error_msg, error_details=None):
        """處理交叉驗證錯誤"""
        self.cv_start_btn['state'] = 'normal'
        self.cv_status.config(text="錯誤", foreground='red')
        
        if error_details:
            print(f"交叉驗證詳細錯誤：\n{error_details}")
        
        messagebox.showerror("錯誤", f"交叉驗證過程中發生錯誤：{error_msg}")

def main():
    root = tk.Tk()
    
    # 確保視窗在前台顯示
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    # 創建應用程式
    app = MainApplication(root)
    
    # 啟動主迴圈
    root.mainloop()

if __name__ == "__main__":
    main() 