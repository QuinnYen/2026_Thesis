import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
from pathlib import Path
import pandas as pd
from gui.config import WINDOW_TITLE, WINDOW_SIZE, WINDOW_MIN_SIZE, COLORS, STATUS_TEXT, SUPPORTED_FILE_TYPES, FONTS, SIMULATION_DELAYS, DATASETS, PREPROCESSING_STEPS
from modules.text_preprocessor import TextPreprocessor
import threading
import queue
import torch
from modules.run_manager import RunManager

# 添加父目錄到路徑以導入config模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import get_base_output_dir, get_path_config

# 匯入錯誤處理工具
from utils.error_handler import handle_error, handle_warning, handle_info

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*WINDOW_MIN_SIZE)
        
        # 設定資料庫目錄路徑
        self.database_dir = self.get_database_dir()
        
        # 初始化RunManager - 使用配置的輸出目錄
        self.run_manager = RunManager(get_base_output_dir())
        
        # 初始化變數
        self.dataset_type = tk.StringVar()
        self.classifier_type = tk.StringVar(value='xgboost')
        self.encoder_type = tk.StringVar(value='bert')
        self.aspect_classifier_type = tk.StringVar(value='lda')
        
        # 分析結果存儲
        self.analysis_results = None
        
        # 初始化比對報告相關變數
        self.selected_mechanism = None
        self.mechanism_combo = None
        self.update_comparison_btn = None
        self.comparison_tree = None
        
        # 分步驟數據文件追蹤
        self.step1_data_file = None
        self.step2_data_file = None
        self.step3_data_file = None
        self.step3_embeddings_file = None
        
        # 創建筆記本控件（分頁）
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=15, pady=15)
        
        # 創建三個分頁
        self.create_attention_analysis_tab()  # 新的第一頁：注意力分析
        self.create_comparison_analysis_tab()  # 結果分析
        self.create_cross_validation_tab()      # 交叉驗證
        
        # 添加當前run目錄標籤
        self.create_run_dir_label()
        
        # 最大化視窗
        self.root.after(100, self.maximize_window)
    
    def get_database_dir(self):
        """取得資料庫目錄路徑"""
        try:
            config = get_path_config()
            return config.get('database_dir', './data')
        except:
            return './data'
    
    def maximize_window(self):
        """最大化視窗"""
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                pass
        
        self.root.lift()
        self.root.focus_force()
    
    def create_attention_analysis_tab(self):
        """第一分頁：分步驟數據處理與分析"""
        frame1 = ttk.Frame(self.notebook)
        self.notebook.add(frame1, text=" 數據處理分析 ")
        
        # 主要容器
        main_frame = ttk.Frame(frame1)
        main_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # 標題
        title_label = ttk.Label(main_frame, text="情感分析 - 分步驟數據處理", font=FONTS['title'])
        title_label.pack(pady=(0, 15))
        
        # 建立分步驟處理區域
        self.create_step_sections(main_frame)
    
    def create_step_sections(self, parent):
        """建立分步驟處理區域"""
        # 建立步驟1: 數據導入
        self.create_step1_data_import(parent)
        
        # 建立步驟2: 數據預處理
        self.create_step2_preprocessing(parent)
        
        # 建立步驟3: 數據向量處理
        self.create_step3_vectorization(parent)
        
        # 建立步驟4: 注意力機制+面向+分類器
        self.create_step4_analysis(parent)
        
        # 總體執行控制區域
        self.create_execution_control(parent)
    
    def create_step1_data_import(self, parent):
        """步驟1: 數據導入"""
        step1_frame = ttk.LabelFrame(parent, text="步驟 1: 數據導入", padding=10)
        step1_frame.pack(fill='x', pady=(0, 10))
        
        # 配置區域
        config_row = ttk.Frame(step1_frame)
        config_row.pack(fill='x', pady=(0, 10))
        
        # 數據集類型
        ttk.Label(config_row, text="數據集類型:").pack(side='left')
        self.dataset_type = tk.StringVar()
        dataset_combo = ttk.Combobox(config_row, 
                                   textvariable=self.dataset_type,
                                   values=[DATASETS[ds]['name'] for ds in DATASETS],
                                   state='readonly',
                                   width=20)
        dataset_combo.pack(side='left', padx=(5, 20))
        dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_selected)
        
        # 檔案選擇
        ttk.Label(config_row, text="數據檔案:").pack(side='left')
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(config_row, textvariable=self.file_path_var, width=35)
        file_entry.pack(side='left', padx=(5, 5), fill='x', expand=True)
        
        self.browse_btn = ttk.Button(config_row, text="瀏覽", command=self.browse_file, state='disabled')
        self.browse_btn.pack(side='left', padx=(5, 0))
        
        # 抽樣設定
        sampling_row = ttk.Frame(step1_frame)
        sampling_row.pack(fill='x', pady=(0, 10))
        
        self.enable_sampling = tk.BooleanVar(value=False)
        sampling_check = ttk.Checkbutton(sampling_row, 
                                       text="啟用數據抽樣 (推薦大數據集)",
                                       variable=self.enable_sampling)
        sampling_check.pack(side='left')
        
        ttk.Label(sampling_row, text="抽樣數量:").pack(side='left', padx=(20, 5))
        self.sample_size = tk.IntVar(value=1000)
        sample_spin = ttk.Spinbox(sampling_row, 
                                from_=100, to=10000, increment=100,
                                textvariable=self.sample_size,
                                width=8)
        sample_spin.pack(side='left')
        
        # 執行按鈕和進度條
        control_row = ttk.Frame(step1_frame)
        control_row.pack(fill='x')
        
        self.step1_btn = ttk.Button(control_row, text="執行數據導入", 
                                  command=self.run_step1_data_import)
        self.step1_btn.pack(side='left')
        
        self.step1_progress = ttk.Progressbar(control_row, length=200, mode='determinate')
        self.step1_progress.pack(side='left', padx=(10, 10))
        
        self.step1_status = ttk.Label(control_row, text="等待執行", foreground=COLORS['info'])
        self.step1_status.pack(side='left')
    
    def create_step2_preprocessing(self, parent):
        """步驟2: 數據預處理"""
        step2_frame = ttk.LabelFrame(parent, text="步驟 2: 數據預處理", padding=10)
        step2_frame.pack(fill='x', pady=(0, 10))
        
        # 預處理選項
        options_row = ttk.Frame(step2_frame)
        options_row.pack(fill='x', pady=(0, 10))
        
        ttk.Label(options_row, text="預處理選項:").pack(side='left')
        
        self.preprocess_options = {}
        options = [
            ('clean_text', '文本清理'),
            ('remove_stopwords', '移除停用詞'),
            ('lemmatization', '詞形還原'),
            ('handle_negation', '否定處理')
        ]
        
        for key, label in options:
            var = tk.BooleanVar(value=True)
            self.preprocess_options[key] = var
            check = ttk.Checkbutton(options_row, text=label, variable=var)
            check.pack(side='left', padx=(5, 10))
        
        # 執行按鈕和進度條
        control_row = ttk.Frame(step2_frame)
        control_row.pack(fill='x')
        
        self.step2_btn = ttk.Button(control_row, text="執行數據預處理", 
                                  command=self.run_step2_preprocessing, state='disabled')
        self.step2_btn.pack(side='left')
        
        self.step2_progress = ttk.Progressbar(control_row, length=200, mode='determinate')
        self.step2_progress.pack(side='left', padx=(10, 10))
        
        self.step2_status = ttk.Label(control_row, text="等待上一步完成", foreground=COLORS['info'])
        self.step2_status.pack(side='left')
    
    def create_step3_vectorization(self, parent):
        """步驟3: 數據向量處理"""
        step3_frame = ttk.LabelFrame(parent, text="步驟 3: 數據向量處理", padding=10)
        step3_frame.pack(fill='x', pady=(0, 10))
        
        # 編碼器選擇
        encoder_row = ttk.Frame(step3_frame)
        encoder_row.pack(fill='x', pady=(0, 10))
        
        ttk.Label(encoder_row, text="編碼器類型:").pack(side='left')
        # ✅ 修復：動態獲取編碼器工廠中支援的編碼器類型
        try:
            from modules.encoder_factory import EncoderFactory
            encoder_options = EncoderFactory.get_available_encoders()
        except ImportError:
            # 如果編碼器工廠不可用，使用預設選項
            encoder_options = ['bert', 'gpt', 't5', 'cnn']
        
        encoder_combo = ttk.Combobox(encoder_row,
                                   textvariable=self.encoder_type,
                                   values=encoder_options,
                                   state='readonly',
                                   width=15)
        encoder_combo.pack(side='left', padx=(5, 10))
        
        # 編碼器說明標籤
        encoder_info_btn = ttk.Button(encoder_row, text="?", width=3,
                                     command=self.show_encoder_info)
        encoder_info_btn.pack(side='left', padx=(2, 8))
        
        ttk.Label(encoder_row, text="最大序列長度:").pack(side='left')
        self.max_length = tk.IntVar(value=512)
        length_spin = ttk.Spinbox(encoder_row, from_=128, to=512, increment=64,
                                textvariable=self.max_length, width=8)
        length_spin.pack(side='left', padx=(5, 0))
        
        # 執行按鈕和進度條
        control_row = ttk.Frame(step3_frame)
        control_row.pack(fill='x')
        
        self.step3_btn = ttk.Button(control_row, text="執行向量處理", 
                                  command=self.run_step3_vectorization, state='disabled')
        self.step3_btn.pack(side='left')
        
        self.step3_progress = ttk.Progressbar(control_row, length=200, mode='determinate')
        self.step3_progress.pack(side='left', padx=(10, 10))
        
        self.step3_status = ttk.Label(control_row, text="等待上一步完成", foreground=COLORS['info'])
        self.step3_status.pack(side='left')
    
    def create_step4_analysis(self, parent):
        """步驟4: 注意力機制+面向+分類器"""
        step4_frame = ttk.LabelFrame(parent, text="步驟 4: 注意力機制分析", padding=10)
        step4_frame.pack(fill='x', pady=(0, 10))
        
        # 分類器和面向選擇
        classifier_row = ttk.Frame(step4_frame)
        classifier_row.pack(fill='x', pady=(0, 10))
        
        ttk.Label(classifier_row, text="分類器:").pack(side='left')
        self.classifier_type = tk.StringVar(value='xgboost')
        # 支援的分類器類型
        classifier_options = ['xgboost', 'logistic_regression', 'random_forest', 'svm_linear', 'naive_bayes']
        classifier_combo = ttk.Combobox(classifier_row,
                                      textvariable=self.classifier_type,
                                      values=classifier_options,
                                      state='readonly',
                                      width=18)
        classifier_combo.pack(side='left', padx=(5, 10))
        
        # 分類器說明標籤
        classifier_info_btn = ttk.Button(classifier_row, text="?", width=3,
                                        command=self.show_classifier_info)
        classifier_info_btn.pack(side='left', padx=(2, 8))
        
        ttk.Label(classifier_row, text="面向分類:").pack(side='left')
        self.aspect_classifier_type = tk.StringVar(value='lda')
        # 支援的面向分類方法
        aspect_options = ['default', 'lda', 'nmf', 'bertopic', 'clustering']
        aspect_combo = ttk.Combobox(classifier_row,
                                  textvariable=self.aspect_classifier_type,
                                  values=aspect_options,
                                  state='readonly',
                                  width=15)
        aspect_combo.pack(side='left', padx=(5, 0))
        
        # 注意力機制選擇
        attention_row = ttk.Frame(step4_frame)
        attention_row.pack(fill='x', pady=(0, 10))
        
        ttk.Label(attention_row, text="注意力機制:").pack(anchor='w')
        
        attention_options_frame = ttk.Frame(step4_frame)
        attention_options_frame.pack(fill='x', pady=(0, 10))
        
        self.attention_options = {}
        attention_types = [
            ('no', '無注意力'),
            ('similarity', '相似度注意力'),
            ('keyword', '關鍵詞注意力'),
            ('self', '自注意力')
        ]
        
        for key, label in attention_types:
            var = tk.BooleanVar(value=True)
            self.attention_options[key] = var
            check = ttk.Checkbutton(attention_options_frame, text=label, variable=var)
            check.pack(side='left', padx=(0, 15))
        
        # 組合注意力選項
        combo_row = ttk.Frame(step4_frame)
        combo_row.pack(fill='x', pady=(0, 10))
        
        self.enable_combinations = tk.BooleanVar(value=True)
        combo_check = ttk.Checkbutton(combo_row, text="啟用注意力機制組合", 
                                    variable=self.enable_combinations)
        combo_check.pack(side='left')
        
        # 智能權重學習選項
        self.use_adaptive_weights = tk.BooleanVar(value=False)
        adaptive_check = ttk.Checkbutton(combo_row, text="使用智能權重學習", 
                                       variable=self.use_adaptive_weights,
                                       command=self.on_adaptive_weights_changed)
        adaptive_check.pack(side='left', padx=(20, 0))
        
        # 權重配置按鈕
        self.weight_config_btn = ttk.Button(combo_row, text="權重配置", 
                                          command=self.show_weight_config, 
                                          state='disabled')
        self.weight_config_btn.pack(side='left', padx=(10, 0))
        
        # 儲存學習到的權重
        self.learned_weights = None
        
        # 執行按鈕和進度條
        control_row = ttk.Frame(step4_frame)
        control_row.pack(fill='x')
        
        self.step4_btn = ttk.Button(control_row, text="執行注意力分析", 
                                  command=self.run_step4_analysis, state='disabled')
        self.step4_btn.pack(side='left')
        
        self.step4_progress = ttk.Progressbar(control_row, length=200, mode='determinate')
        self.step4_progress.pack(side='left', padx=(10, 10))
        
        self.step4_status = ttk.Label(control_row, text="等待上一步完成", foreground=COLORS['info'])
        self.step4_status.pack(side='left')
    
    def create_execution_control(self, parent):
        """總體執行控制區域"""
        control_frame = ttk.LabelFrame(parent, text="總體進度", padding=10)
        control_frame.pack(fill='x', pady=(10, 0))
        
        # 總體進度條
        progress_row = ttk.Frame(control_frame)
        progress_row.pack(fill='x')
        
        ttk.Label(progress_row, text="總體進度:").pack(side='left')
        self.overall_progress = ttk.Progressbar(progress_row, length=400, mode='determinate')
        self.overall_progress.pack(side='left', padx=(10, 10), fill='x', expand=True)
        
        self.overall_status = ttk.Label(progress_row, text="準備就緒", foreground=COLORS['info'])
        self.overall_status.pack(side='left')
        
        # 重製按鈕
        reset_row = ttk.Frame(control_frame)
        reset_row.pack(fill='x', pady=(10, 0))
        
        self.reset_btn = ttk.Button(reset_row, text="🔄 重製程式", 
                                   command=self.restart_application,
                                   style='Accent.TButton')
        self.reset_btn.pack(side='right')
        
        ttk.Label(reset_row, text="重新開始所有步驟，清除所有數據和結果", 
                 font=('TkDefaultFont', 8), foreground='gray').pack(side='right', padx=(0, 10))
    
    def create_results_preview_table(self, parent):
        """創建結果預覽表格"""
        # 表格框架
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill='both', expand=True)
        
        # 創建表格
        columns = ('機制名稱', '準確率', 'F1分數', '訓練時間')
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        # 設定列標題和寬度
        self.results_tree.heading('機制名稱', text='注意力機制')
        self.results_tree.heading('準確率', text='準確率 (%)')
        self.results_tree.heading('F1分數', text='F1分數 (%)')
        self.results_tree.heading('訓練時間', text='訓練時間 (秒)')
        
        self.results_tree.column('機制名稱', width=200, anchor='w')
        self.results_tree.column('準確率', width=120, anchor='center')
        self.results_tree.column('F1分數', width=120, anchor='center')
        self.results_tree.column('訓練時間', width=120, anchor='center')
        
        # 添加滾動條
        scrollbar_y = ttk.Scrollbar(table_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar_y.set)
        
        # 布局
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar_y.pack(side='right', fill='y')
        
        # 初始提示
        self.results_tree.insert('', 'end', values=('等待分析...', '-', '-', '-'))
    
    def on_dataset_selected(self, event=None):
        """當選擇數據集時啟用檔案瀏覽按鈕"""
        if self.dataset_type.get():
            self.browse_btn['state'] = 'normal'
    
    def browse_file(self):
        """瀏覽檔案"""
        filetypes = [
            ('CSV files', '*.csv'),
            ('Text files', '*.txt'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="選擇數據檔案",
            filetypes=filetypes
        )
        
        if filename:
            self.file_path_var.set(filename)
    
    # 分步驟執行方法
    def run_step1_data_import(self):
        """執行步驟1：數據導入"""
        if not self._validate_step1_config():
            return
            
        self.step1_btn['state'] = 'disabled'
        self.step1_progress.config(value=0)
        self.step1_status.config(text="正在導入數據...", foreground=COLORS['processing'])
        
        def run_import():
            try:
                import time
                
                # 更新進度：開始讀取
                self.root.after(0, lambda: self.step1_progress.config(value=20))
                
                # 讀取原始數據
                df = pd.read_csv(self.file_path_var.get())
                self.root.after(0, lambda: self.step1_status.config(text=f"數據載入完成 ({len(df)} 條記錄)"))
                self.root.after(0, lambda: self.step1_progress.config(value=50))
                
                # 創建輸入參考（不儲存原始數據）
                from utils.storage_manager import StorageManager
                storage_manager = StorageManager(self.run_manager.get_run_dir(self.encoder_type.get()))
                
                self.root.after(0, lambda: self.step1_status.config(text="正在創建輸入參考..."))
                self.root.after(0, lambda: self.step1_progress.config(value=70))
                
                # 創建輸入文件參考而不複製原始數據
                file_path = self.file_path_var.get()
                reference_file = storage_manager.create_input_reference(file_path)
                
                # 如果啟用抽樣，只在記憶體中處理，不儲存
                if self.enable_sampling.get():
                    sample_size = min(self.sample_size.get(), len(df))
                    df_working = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                    status_text = f"✅ 數據導入完成 ({len(df_working)} 條記錄，已抽樣)"
                else:
                    df_working = df
                    status_text = f"✅ 數據導入完成 ({len(df_working)} 條記錄)"
                
                # 將工作數據存儲在記憶體中，供後續步驟使用
                self.working_data = df_working
                self.original_file_path = file_path
                
                self.root.after(0, lambda: self.step1_progress.config(value=100))
                self.root.after(0, lambda: self.step1_status.config(
                    text=status_text,
                    foreground=COLORS['success']
                ))
                
                # 啟用下一步
                self.root.after(0, lambda: self.step2_btn.config(state='normal'))
                self.root.after(0, lambda: self.step2_status.config(text="準備就緒", foreground=COLORS['info']))
                self.root.after(0, lambda: self.overall_progress.config(value=25))
                self.root.after(0, lambda: self.overall_status.config(text="步驟1完成 - 數據已導入"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._handle_step_error(1, error_msg))
        
        # 啟動後台線程
        threading.Thread(target=run_import, daemon=True).start()
    
    def run_step2_preprocessing(self):
        """執行步驟2：數據預處理"""
        self.step2_btn['state'] = 'disabled'
        self.step2_progress.config(value=0)
        self.step2_status.config(text="正在預處理數據...", foreground=COLORS['processing'])
        
        def run_preprocess():
            try:
                import time
                from modules.text_preprocessor import TextPreprocessor
                
                # 更新進度：開始處理
                self.root.after(0, lambda: self.step2_progress.config(value=10))
                
                # 使用記憶體中的工作數據
                df = self.working_data.copy()
                self.root.after(0, lambda: self.step2_progress.config(value=20))
                
                # 初始化預處理器
                preprocessor = TextPreprocessor()
                self.root.after(0, lambda: self.step2_progress.config(value=30))
                
                # 獲取預處理選項
                options = {}
                for key, var in self.preprocess_options.items():
                    options[key] = var.get()
                
                self.root.after(0, lambda: self.step2_status.config(text="正在執行文本預處理..."))
                self.root.after(0, lambda: self.step2_progress.config(value=40))
                
                # 執行預處理（假設有text列）
                if 'text' in df.columns:
                    df['processed_text'] = df['text'].apply(
                        lambda x: preprocessor.preprocess_text(x, **options)
                    )
                elif 'review' in df.columns:
                    df['processed_text'] = df['review'].apply(
                        lambda x: preprocessor.preprocess_text(x, **options)
                    )
                else:
                    # 尋找可能的文本列
                    text_cols = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower() or 'comment' in col.lower()]
                    if text_cols:
                        df['processed_text'] = df[text_cols[0]].apply(
                            lambda x: preprocessor.preprocess_text(x, **options)
                        )
                    else:
                        raise Exception("找不到文本列")
                
                self.root.after(0, lambda: self.step2_progress.config(value=80))
                
                # 使用儲存管理器保存預處理數據
                from utils.storage_manager import StorageManager
                storage_manager = StorageManager(self.run_manager.get_run_dir(self.encoder_type.get()))
                processed_file = storage_manager.save_processed_data(
                    df, 'preprocessing', '01_preprocessed_data.csv',
                    metadata={'preprocessing_options': options}
                )
                
                # 更新工作數據
                self.working_data = df
                self.root.after(0, lambda: self.step2_progress.config(value=100))
                self.root.after(0, lambda: self.step2_status.config(
                    text="✅ 數據預處理完成",
                    foreground=COLORS['success']
                ))
                
                # 啟用下一步
                self.root.after(0, lambda: self.step3_btn.config(state='normal'))
                self.root.after(0, lambda: self.step3_status.config(text="準備就緒", foreground=COLORS['info']))
                self.root.after(0, lambda: self.overall_progress.config(value=50))
                self.root.after(0, lambda: self.overall_status.config(text="步驟2完成 - 數據已預處理"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._handle_step_error(2, error_msg))
        
        threading.Thread(target=run_preprocess, daemon=True).start()
    
    def run_step3_vectorization(self):
        """執行步驟3：數據向量處理"""
        self.step3_btn['state'] = 'disabled'
        self.step3_progress.config(value=0)
        self.step3_status.config(text="正在進行向量處理...", foreground=COLORS['processing'])
        
        def run_vectorize():
            try:
                import time
                # ✅ 修復：使用編碼器工廠而不是硬編碼BERT
                from modules.encoder_factory import EncoderFactory
                
                # 更新進度：開始處理
                self.root.after(0, lambda: self.step3_progress.config(value=10))
                
                # 使用記憶體中的工作數據
                df = self.working_data.copy()
                self.root.after(0, lambda: self.step3_progress.config(value=20))
                
                # ✅ 修復：獲取選定的編碼器類型
                selected_encoder_type = self.encoder_type.get()
                self.root.after(0, lambda: self.step3_status.config(text=f"正在初始化{selected_encoder_type.upper()}編碼器..."))
                
                # ✅ 修復：使用工廠創建選定的編碼器
                def progress_callback(callback_type, message):
                    if callback_type in ['status', 'progress']:
                        self.root.after(0, lambda msg=message: self.step3_status.config(text=msg))
                
                encoder_config = {
                    'max_length': self.max_length.get(),
                    'batch_size': 32
                }
                
                try:
                    encoder = EncoderFactory.create_encoder(
                        encoder_type=selected_encoder_type,
                        config=encoder_config,
                        progress_callback=progress_callback
                    )
                except Exception as encoder_error:
                    # 如果選定的編碼器不可用，回退到BERT
                    self.root.after(0, lambda: self.step3_status.config(text=f"⚠️ {selected_encoder_type.upper()}編碼器不可用，回退使用BERT..."))
                    from modules.bert_encoder import BertEncoder
                    encoder = BertEncoder(progress_callback=progress_callback)
                    selected_encoder_type = 'bert'
                
                self.root.after(0, lambda: self.step3_progress.config(value=30))
                
                self.root.after(0, lambda: self.step3_status.config(text=f"正在進行{selected_encoder_type.upper()}文本編碼..."))
                
                # 進行向量化
                texts = df['processed_text']
                embeddings = encoder.encode(texts)
                self.root.after(0, lambda: self.step3_progress.config(value=70))
                
                # 使用儲存管理器保存向量化結果
                from utils.storage_manager import StorageManager
                storage_manager = StorageManager(self.run_manager.get_run_dir(selected_encoder_type))
                
                # ✅ 修復：保存嵌入向量時使用正確的編碼器類型
                embeddings_file = storage_manager.save_embeddings(
                    embeddings, selected_encoder_type,
                    metadata={
                        'encoder_type': selected_encoder_type, 
                        'embedding_dim': encoder.get_embedding_dim(),
                        'text_count': len(texts),
                        'max_length': self.max_length.get()
                    }
                )
                self.root.after(0, lambda: self.step3_progress.config(value=85))
                
                # 更新工作數據和嵌入向量
                self.working_data = df
                self.working_embeddings = embeddings
                
                # 保存數據檔案路徑供步驟4使用
                data_file = storage_manager.save_processed_data(
                    df, 'encoding', 'step3_vectorized_data.csv',
                    metadata={
                        'step': 'vectorization', 
                        'encoder_type': selected_encoder_type,
                        'embeddings_shape': embeddings.shape
                    }
                )
                self.step3_data_file = data_file
                self.step3_embeddings_file = embeddings_file
                
                self.root.after(0, lambda: self.step3_progress.config(value=100))
                self.root.after(0, lambda: self.step3_status.config(
                    text=f"✅ {selected_encoder_type.upper()}向量處理完成 ({embeddings.shape[0]} 個向量, 維度: {embeddings.shape[1]})",
                    foreground=COLORS['success']
                ))
                
                # 啟用下一步
                self.root.after(0, lambda: self.step4_btn.config(state='normal'))
                self.root.after(0, lambda: self.step4_status.config(text="準備就緒", foreground=COLORS['info']))
                self.root.after(0, lambda: self.overall_progress.config(value=75))
                self.root.after(0, lambda: self.overall_status.config(text=f"步驟3完成 - {selected_encoder_type.upper()}向量處理完成"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._handle_step_error(3, error_msg))
        
        threading.Thread(target=run_vectorize, daemon=True).start()
    
    def run_step4_analysis(self):
        """執行步驟4：注意力機制分析"""
        self.step4_btn['state'] = 'disabled'
        self.step4_progress.config(value=0)
        self.step4_status.config(text="正在執行注意力分析...", foreground=COLORS['processing'])
        
        def run_analysis():
            try:
                import time
                import numpy as np
                start_time = time.time()
                
                # 更新進度：開始處理
                self.root.after(0, lambda: self.step4_progress.config(value=10))
                
                # 使用記憶體中的工作數據
                df = self.working_data.copy()
                embeddings = self.working_embeddings
                self.root.after(0, lambda: self.step4_progress.config(value=20))
                
                self.root.after(0, lambda: self.step4_status.config(text="正在執行注意力機制分析..."))
                
                # 導入主處理函數
                from Part05_Main import process_attention_analysis_with_multiple_combinations
                self.root.after(0, lambda: self.step4_progress.config(value=30))
                
                # 準備注意力機制列表
                attention_types = []
                for key, var in self.attention_options.items():
                    if var.get():
                        attention_types.append(key)
                
                # 準備組合機制列表
                attention_combinations = []
                if self.enable_combinations.get():
                    # 檢查是否使用智能權重學習
                    if hasattr(self, 'use_adaptive_weights') and self.use_adaptive_weights.get():
                        # 如果已有學習到的最佳權重，使用它們
                        if hasattr(self, 'learned_weights') and self.learned_weights:
                            learned_combo = self.learned_weights.copy()
                            learned_combo['_is_learned'] = True  # 標記為智能學習權重
                            attention_combinations = [learned_combo]
                        else:
                            # 使用預設權重，稍後會被智能學習替代
                            attention_combinations = [
                                {'similarity': 0.33, 'self': 0.33, 'keyword': 0.34}
                            ]
                    else:
                        # 使用固定權重組合
                        attention_combinations = [
                            {'similarity': 0.5, 'self': 0.5},
                            {'similarity': 0.5, 'keyword': 0.5},
                            {'self': 0.5, 'keyword': 0.5},
                            {'similarity': 0.33, 'self': 0.33, 'keyword': 0.34}
                        ]
                
                output_dir = self.run_manager.get_run_dir(self.encoder_type.get())
                self.root.after(0, lambda: self.step4_progress.config(value=40))
                
                # 執行分析
                results = process_attention_analysis_with_multiple_combinations(
                    input_file=self.step3_data_file,
                    output_dir=output_dir,
                    attention_types=attention_types,
                    attention_combinations=attention_combinations,
                    classifier_type=self.classifier_type.get(),
                    encoder_type=self.encoder_type.get()
                )
                
                self.root.after(0, lambda: self.step4_progress.config(value=90))
                
                total_time = time.time() - start_time
                
                self.root.after(0, lambda: self.step4_progress.config(value=100))
                self.root.after(0, lambda: self.step4_status.config(
                    text=f"✅ 注意力分析完成 (耗時: {total_time:.4f}秒)",
                    foreground=COLORS['success']
                ))
                
                # 更新總體進度
                self.root.after(0, lambda: self.overall_progress.config(value=100))
                self.root.after(0, lambda: self.overall_status.config(
                    text="✅ 所有步驟完成！", foreground=COLORS['success']
                ))
                
                # 保存結果並切換到第二頁
                self.analysis_results = results
                self.root.after(0, lambda: self.notebook.select(1))
                self.root.after(0, lambda: self._update_analysis_results(results, total_time))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._handle_step_error(4, error_msg))
        
        threading.Thread(target=run_analysis, daemon=True).start()
    

    
    def _validate_step1_config(self):
        """驗證步驟1配置"""
        if not self.dataset_type.get():
            messagebox.showerror("錯誤", "請選擇數據集類型！")
            return False
            
        if not self.file_path_var.get():
            messagebox.showerror("錯誤", "請選擇數據檔案！")
            return False
            
        if not os.path.exists(self.file_path_var.get()):
            messagebox.showerror("錯誤", "數據檔案不存在！")
            return False
            
        return True
    
    def _handle_step_error(self, step_num, error_msg):
        """處理步驟錯誤"""
        # 使用統一錯誤處理機制輸出到終端機
        import traceback
        try:
            # 創建一個包含完整信息的錯誤
            full_error = Exception(f"步驟{step_num}執行失敗: {error_msg}")
            handle_error(full_error, f"GUI步驟{step_num}", show_traceback=True)
        except Exception as e:
            # 如果錯誤處理器本身有問題，使用基本輸出
            print(f"🚨 GUI步驟{step_num}錯誤: {error_msg}")
            print(f"錯誤追蹤:")
            traceback.print_exc()
        
        # 重設相關進度條
        if step_num == 1:
            self.step1_progress.config(value=0)
            self.step1_status.config(text=f"❌ 錯誤: {error_msg}", foreground=COLORS['error'])
        elif step_num == 2:
            self.step2_progress.config(value=0)
            self.step2_status.config(text=f"❌ 錯誤: {error_msg}", foreground=COLORS['error'])
        elif step_num == 3:
            self.step3_progress.config(value=0)
            self.step3_status.config(text=f"❌ 錯誤: {error_msg}", foreground=COLORS['error'])
        elif step_num == 4:
            self.step4_progress.config(value=0)
            self.step4_status.config(text=f"❌ 錯誤: {error_msg}", foreground=COLORS['error'])
        
        # 重新啟用對應步驟按鈕
        if step_num == 1:
            self.step1_btn['state'] = 'normal'
        elif step_num == 2:
            self.step2_btn['state'] = 'normal'
        elif step_num == 3:
            self.step3_btn['state'] = 'normal'
        elif step_num == 4:
            self.step4_btn['state'] = 'normal'
        
        # 顯示錯誤對話框
        messagebox.showerror(f"步驟{step_num}錯誤", f"步驟{step_num}執行失敗：\n{error_msg}\n\n詳細錯誤信息請查看終端機輸出。")
    
    def _validate_analysis_config(self):
        """驗證分析配置（保留向後兼容）"""
        return self._validate_step1_config()
    
    def _prepare_sampled_data(self):
        """準備抽樣數據（已廢棄，使用記憶體中的工作數據）"""
        try:
            # 使用記憶體中的工作數據，不需要重新讀取和儲存
            if hasattr(self, 'working_data'):
                return self.working_data
            else:
                # 如果沒有工作數據，讀取原始數據但不儲存
                df = pd.read_csv(self.file_path_var.get())
                sample_size = min(self.sample_size.get(), len(df))
                df_sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                return df_sampled
            
        except Exception as e:
            raise Exception(f"數據準備失敗：{str(e)}")
    
    def _update_analysis_results(self, results, total_time):
        """更新分析結果到表格"""
        try:
            print(f"🔍 GUI除錯：開始更新分析結果...")
            print(f"🔍 GUI除錯：results的鍵: {list(results.keys())}")
            
            # 清空表格
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # 獲取分類結果
            classification_evaluation = results.get('classification_evaluation', {})
            print(f"🔍 GUI除錯：classification_evaluation的鍵: {list(classification_evaluation.keys())}")
            
            # 從 classification_evaluation 中過濾出機制結果（排除 'comparison' 鍵）
            classification_results = {}
            for key, value in classification_evaluation.items():
                if key != 'comparison' and isinstance(value, dict):
                    classification_results[key] = value
                    print(f"🔍 GUI除錯：找到機制結果: {key}")
            
            # 如果沒有找到，嘗試舊格式
            if not classification_results:
                classification_results = results.get('classification_results', {})
                print(f"🔍 GUI除錯：使用舊格式，classification_results的鍵: {list(classification_results.keys())}")
            
            print(f"🔍 GUI除錯：最終classification_results的鍵: {list(classification_results.keys())}")
            
            # 顯示結果
            for mechanism, result in classification_results.items():
                accuracy = result.get('test_accuracy', 0) * 100
                f1_score = result.get('test_f1', 0) * 100
                train_time = result.get('training_time', 0)
                
                # 格式化機制名稱
                display_name = self._format_mechanism_name(mechanism)
                
                self.results_tree.insert('', 'end', values=(
                    display_name,
                    f"{accuracy:.4f}%",
                    f"{f1_score:.4f}%",
                    f"{train_time:.4f}s"
                ))
            
            # 獲取摘要信息
            summary = results.get('summary', {})
            best_mechanism = summary.get('best_attention_mechanism', 'N/A')
            best_accuracy = summary.get('best_classification_accuracy', 0) * 100
            
            # 保存結果供其他頁面使用
            self.analysis_results = results
            
            # 更新機制選擇下拉菜單
            self._update_mechanism_combo(classification_results)
            
            # 自動切換到第二頁顯示結果
            self.notebook.select(1)  # 切換到第二頁（索引為1）
            
            # 更新第二頁的詳細結果
            self._update_detailed_results(results)
            
            # 顯示完成訊息到終端
            print(f"✅ 分析完成！最佳機制: {self._format_mechanism_name(best_mechanism)} ({best_accuracy:.4f}%) | 總耗時: {total_time:.4f}秒")
            
        except Exception as e:
            error_msg = f"結果更新失敗: {str(e)}"
            print(f"❌ {error_msg}")
            # 使用錯誤處理器
            from utils.error_handler import TerminalErrorHandler
            error_handler = TerminalErrorHandler()
            error_handler.handle_error(e, "GUI結果更新時發生錯誤")
    
    
    def _format_mechanism_name(self, mechanism):
        """格式化注意力機制名稱為中文"""
        # 基本機制名稱映射
        name_mapping = {
            'no': '無注意力',
            'similarity': '相似度注意力',
            'keyword': '關鍵詞注意力', 
            'self': '自注意力',
            'combined': '組合注意力'
        }
        
        # 如果是基本機制名稱，直接映射
        if mechanism in name_mapping:
            return name_mapping[mechanism]
        
        # 如果已經是中文組合名稱（如 "相似度+自注意力組合"），直接返回
        if any(chinese in mechanism for chinese in ['相似度', '關鍵詞', '自注意力', '組合']):
            return mechanism
            
        # 如果是舊的 combination_X 格式，轉換為組合注意力
        if mechanism.startswith('combination_'):
            return f"組合注意力{mechanism.split('_')[1]}"
            
        # 其他情況直接返回原名稱
        return mechanism
    
    def _update_detailed_results(self, results):
        """更新第二頁的詳細結果顯示"""
        try:
            print(f"🔍 GUI除錯：開始更新詳細結果...")
            
            # 清空文字區域
            self.analysis_text.delete('1.0', tk.END)
            
            # 獲取結果數據
            classification_evaluation = results.get('classification_evaluation', {})
            
            # 從 classification_evaluation 中過濾出機制結果（排除 'comparison' 鍵）
            classification_results = {}
            for key, value in classification_evaluation.items():
                if key != 'comparison' and isinstance(value, dict):
                    classification_results[key] = value
            
            # 如果沒有找到，嘗試舊格式
            if not classification_results:
                classification_results = results.get('classification_results', {})
            summary = results.get('summary', {})
            
            # 構建詳細報告
            report = []
            report.append("=" * 60)
            report.append("情感分析 - 注意力機制比較分析報告")
            report.append("=" * 60)
            report.append("")
            
            # 摘要資訊
            if summary:
                report.append("📊 分析摘要")
                report.append("-" * 30)
                best_mechanism = summary.get('best_attention_mechanism', 'N/A')
                best_accuracy = summary.get('best_classification_accuracy', 0) * 100
                report.append(f"最佳注意力機制: {self._format_mechanism_name(best_mechanism)}")
                report.append(f"最佳準確率: {best_accuracy:.4f}%")
                report.append("")
            
            # 詳細結果
            report.append("📈 各機制詳細表現")
            report.append("-" * 30)
            
            for mechanism, result in classification_results.items():
                display_name = self._format_mechanism_name(mechanism)
                accuracy = result.get('test_accuracy', 0) * 100
                f1_score = result.get('test_f1', 0) * 100
                precision = result.get('test_precision', 0) * 100
                recall = result.get('test_recall', 0) * 100
                train_time = result.get('training_time', 0)
                
                # 獲取宏平均指標
                precision_macro = result.get('test_precision_macro', 0) * 100
                recall_macro = result.get('test_recall_macro', 0) * 100
                f1_macro = result.get('test_f1_macro', 0) * 100
                
                report.append(f"🔹 {display_name}")
                report.append(f"   準確率: {accuracy:.4f}%")
                report.append(f"   F1分數 (加權): {f1_score:.4f}%")
                report.append(f"   F1分數 (宏平均): {f1_macro:.4f}%")
                report.append(f"   精確率 (加權): {precision:.4f}%")
                report.append(f"   精確率 (宏平均): {precision_macro:.4f}%")
                report.append(f"   召回率 (加權): {recall:.4f}%")
                report.append(f"   召回率 (宏平均): {recall_macro:.4f}%")
                report.append(f"   訓練時間: {train_time:.4f} 秒")
                
                # 如果是組合注意力，顯示權重配置
                if 'combined' in mechanism.lower() or 'combination' in mechanism.lower():
                    weights_info = result.get('attention_weights', {})
                    if weights_info:
                        report.append(f"   🎯 注意力權重配置:")
                        for weight_name, weight_value in weights_info.items():
                            if isinstance(weight_value, (int, float)):
                                report.append(f"      {weight_name}: {weight_value:.4f}")
                    
                    # 檢查是否有智能學習的權重
                    learned_weights = result.get('learned_weights')
                    if learned_weights:
                        report.append(f"   🧠 智能學習權重:")
                        for weight_name, weight_value in learned_weights.items():
                            if isinstance(weight_value, (int, float)):
                                report.append(f"      {weight_name}: {weight_value:.4f}")
                
                report.append("")
            
            # 顯示報告
            self.analysis_text.insert('1.0', '\n'.join(report))
            
        except Exception as e:
            self.analysis_text.delete('1.0', tk.END)
            self.analysis_text.insert('1.0', f"詳細結果顯示錯誤: {str(e)}")
    
    def create_comparison_analysis_tab(self):
        """第二分頁：比對分析（含結果預覽）"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 結果分析 ")
        
        # 主要容器 - 使用 Paned Window 來分割上下兩部分
        main_paned = ttk.PanedWindow(frame, orient='vertical')
        main_paned.pack(fill='both', expand=True, padx=15, pady=10)
        
        # 上半部分：分析結果摘要
        top_frame = ttk.Frame(main_paned)
        main_paned.add(top_frame, weight=1)
        
        # 標題
        title_label = ttk.Label(top_frame, text="注意力機制比較分析", font=FONTS['title'])
        title_label.pack(pady=(0, 15))
        
        # 說明
        info_label = ttk.Label(top_frame, 
                             text="在第一頁完成分析後，詳細結果將顯示在這裡",
                             foreground='gray')
        info_label.pack(pady=(0, 10))
        
        # 結果預覽區域
        results_frame = ttk.LabelFrame(top_frame, text="分析結果預覽", padding=10)
        results_frame.pack(fill='x', pady=(0, 15))
        
        # 結果表格  
        self.create_results_preview_table(results_frame)
        
        # 詳細結果顯示區域
        details_frame = ttk.LabelFrame(top_frame, text="詳細分析結果", padding=10)
        details_frame.pack(fill='both', expand=True)
        
        self.analysis_text = scrolledtext.ScrolledText(details_frame, height=8, width=80)
        self.analysis_text.pack(fill='both', expand=True)
        self.analysis_text.insert('1.0', "等待分析結果...")
        
        # 下半部分：原始數據與預測比對
        bottom_frame = ttk.Frame(main_paned)
        main_paned.add(bottom_frame, weight=1)
        
        # 比對報告區域
        comparison_frame = ttk.LabelFrame(bottom_frame, text="原始數據與模型預測比對報告", padding=10)
        comparison_frame.pack(fill='both', expand=True)
        
        # 機制選擇和控制按鈕
        control_frame = ttk.Frame(comparison_frame)
        control_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(control_frame, text="選擇注意力機制:").pack(side='left')
        self.selected_mechanism = tk.StringVar()
        self.mechanism_combo = ttk.Combobox(control_frame, 
                                          textvariable=self.selected_mechanism,
                                          state='readonly',
                                          width=20)
        self.mechanism_combo.pack(side='left', padx=(5, 10))
        self.mechanism_combo.bind('<<ComboboxSelected>>', self.on_mechanism_selected)
        
        # 顯示數量控制
        ttk.Label(control_frame, text="顯示筆數:").pack(side='left')
        self.display_count = tk.IntVar(value=50)
        count_spin = ttk.Spinbox(control_frame, from_=10, to=200, increment=10,
                               textvariable=self.display_count, width=8)
        count_spin.pack(side='left', padx=(5, 10))
        
        # 更新按鈕
        self.update_comparison_btn = ttk.Button(control_frame, text="更新比對報告", 
                                              command=self.update_comparison_report,
                                              state='disabled')
        self.update_comparison_btn.pack(side='left', padx=(10, 0))
        
        # 快速更新按鈕（使用最佳機制）
        self.quick_update_btn = ttk.Button(control_frame, text="使用最佳機制", 
                                         command=self.quick_update_best_mechanism,
                                         state='disabled')
        self.quick_update_btn.pack(side='left', padx=(5, 0))
        
        # 比對表格
        self.create_comparison_table(comparison_frame)
    
    def create_comparison_table(self, parent):
        """創建原始數據與預測比對表格"""
        # 表格框架
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill='both', expand=True)
        
        # 創建表格
        columns = ('原始索引', '原始句子', '原始評分', '模型預測', '比對結果')
        self.comparison_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=12)
        
        # 設置標籤樣式
        self.comparison_tree.tag_configure('correct', background='lightgreen')
        self.comparison_tree.tag_configure('incorrect', background='lightcoral')
        
        # 設定列標題和寬度
        self.comparison_tree.heading('原始索引', text='索引')
        self.comparison_tree.heading('原始句子', text='原始句子(縮減版)')
        self.comparison_tree.heading('原始評分', text='原始評分')
        self.comparison_tree.heading('模型預測', text='模型預測')
        self.comparison_tree.heading('比對結果', text='比對結果')
        
        self.comparison_tree.column('原始索引', width=80, anchor='center')
        self.comparison_tree.column('原始句子', width=300, anchor='w')
        self.comparison_tree.column('原始評分', width=100, anchor='center')
        self.comparison_tree.column('模型預測', width=100, anchor='center')
        self.comparison_tree.column('比對結果', width=100, anchor='center')
        
        # 添加滾動條
        scrollbar_y = ttk.Scrollbar(table_frame, orient='vertical', command=self.comparison_tree.yview)
        scrollbar_x = ttk.Scrollbar(table_frame, orient='horizontal', command=self.comparison_tree.xview)
        self.comparison_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 布局
        self.comparison_tree.grid(row=0, column=0, sticky='nsew')
        scrollbar_y.grid(row=0, column=1, sticky='ns')
        scrollbar_x.grid(row=1, column=0, sticky='ew')
        
        # 配置grid權重
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # 初始提示
        self.comparison_tree.insert('', 'end', values=('--', '等待分析完成...', '--', '--', '--'))
    
    def on_mechanism_selected(self, event=None):
        """當選擇注意力機制時啟用更新按鈕"""
        if self.selected_mechanism.get():
            self.update_comparison_btn['state'] = 'normal'
    
    def update_comparison_report(self):
        """更新比對報告"""
        try:
            print(f"🔍 GUI除錯：開始更新比對報告...")
            
            if not hasattr(self, 'analysis_results') or not self.analysis_results:
                print(f"🔍 GUI除錯：沒有分析結果")
                messagebox.showwarning("警告", "尚無分析結果，請先完成分析")
                return
            
            selected_mechanism = self.selected_mechanism.get()
            print(f"🔍 GUI除錯：選擇的機制: {selected_mechanism}")
            if not selected_mechanism:
                messagebox.showwarning("警告", "請選擇要比對的注意力機制")
                return
            
            # 獲取對應機制的分析結果
            classification_evaluation = self.analysis_results.get('classification_evaluation', {})
            print(f"🔍 GUI除錯：classification_evaluation的鍵: {list(classification_evaluation.keys())}")
            
            # 從 classification_evaluation 中過濾出機制結果（排除 'comparison' 鍵）
            classification_results = {}
            for key, value in classification_evaluation.items():
                if key != 'comparison' and isinstance(value, dict):
                    classification_results[key] = value
                    print(f"🔍 GUI除錯：找到機制結果: {key}")
            
            # 如果沒有找到，嘗試舊格式
            if not classification_results:
                classification_results = self.analysis_results.get('classification_results', {})
                print(f"🔍 GUI除錯：使用舊格式，classification_results的鍵: {list(classification_results.keys())}")
            
            print(f"🔍 GUI除錯：比對報告中的classification_results鍵: {list(classification_results.keys())}")
            
            # 找到選擇的機制結果
            mechanism_result = None
            for mechanism, result in classification_results.items():
                formatted_name = self._format_mechanism_name(mechanism)
                print(f"🔍 GUI除錯：檢查機制 {mechanism} -> {formatted_name}")
                if formatted_name == selected_mechanism:
                    mechanism_result = result
                    print(f"🔍 GUI除錯：找到匹配的機制結果")
                    break
            
            if not mechanism_result:
                messagebox.showerror("錯誤", f"找不到機制 '{selected_mechanism}' 的分析結果")
                return
            
            # 獲取預測結果 - 修正：從prediction_details中獲取
            prediction_details = mechanism_result.get('prediction_details', {})
            predicted_labels = prediction_details.get('predicted_label_names', [])
            true_labels = prediction_details.get('true_label_names', [])
            test_texts = prediction_details.get('test_texts', [])
            
            if not predicted_labels:
                messagebox.showwarning("警告", f"機制 '{selected_mechanism}' 沒有預測結果")
                return
            
            # 獲取原始數據
            if not hasattr(self, 'working_data') or self.working_data is None:
                messagebox.showwarning("警告", "找不到原始數據")
                return
            
            # 清空表格
            for item in self.comparison_tree.get_children():
                self.comparison_tree.delete(item)
            
            # 獲取顯示數量
            display_count = min(self.display_count.get(), len(predicted_labels))
            
            # 填充比對數據
            sentiment_mapping = {'positive': '正面', 'negative': '負面', 'neutral': '中性'}
            
            # 使用測試集數據或回退到工作數據
            if test_texts and len(test_texts) > 0:
                # 使用模型測試集的文本
                for i in range(display_count):
                    if i >= len(predicted_labels) or i >= len(true_labels):
                        break
                    
                    # 原始索引（測試集中的索引）
                    original_index = i
                    
                    # 原始句子(縮減版) - 來自測試集
                    if i < len(test_texts):
                        original_text = str(test_texts[i])
                        short_text = original_text[:50] + "..." if len(original_text) > 50 else original_text
                    else:
                        short_text = "無文本數據"
                    
                    # 原始評分 - 來自測試集真實標籤
                    original_sentiment = true_labels[i] if i < len(true_labels) else "未知"
                    original_sentiment_cn = sentiment_mapping.get(original_sentiment, original_sentiment)
                    
                    # 模型預測
                    predicted_sentiment = predicted_labels[i]
                    predicted_sentiment_cn = sentiment_mapping.get(predicted_sentiment, predicted_sentiment)
                    
                    # 比對結果
                    is_correct = (original_sentiment == predicted_sentiment)
                    comparison_result = "✓ 正確" if is_correct else "✗ 錯誤"
                    
                    # 添加到表格（帶顏色標籤）
                    tag = 'correct' if is_correct else 'incorrect'
                    self.comparison_tree.insert('', 'end', values=(
                        original_index,
                        short_text,
                        original_sentiment_cn,
                        predicted_sentiment_cn,
                        comparison_result
                    ), tags=(tag,))
            else:
                # 回退方案：使用工作數據（如果可用）
                if hasattr(self, 'working_data') and self.working_data is not None:
                    df = self.working_data
                    for i in range(min(display_count, len(df))):
                        if i >= len(predicted_labels):
                            break
                        
                        # 原始索引
                        original_index = i
                        
                        # 原始句子(縮減版)
                        text_column = None
                        for col in ['processed_text', 'clean_text', 'text', 'review']:
                            if col in df.columns:
                                text_column = col
                                break
                        
                        if text_column:
                            original_text = str(df.iloc[i][text_column])
                            short_text = original_text[:50] + "..." if len(original_text) > 50 else original_text
                        else:
                            short_text = "無文本數據"
                        
                        # 原始評分
                        original_sentiment = "未知"
                        for col in ['sentiment', 'label', 'category']:
                            if col in df.columns:
                                original_sentiment = str(df.iloc[i][col])
                                break
                        
                        original_sentiment_cn = sentiment_mapping.get(original_sentiment, original_sentiment)
                        
                        # 模型預測
                        predicted_sentiment = predicted_labels[i] if i < len(predicted_labels) else "未知"
                        predicted_sentiment_cn = sentiment_mapping.get(predicted_sentiment, predicted_sentiment)
                        
                        # 比對結果
                        is_correct = (original_sentiment == predicted_sentiment)
                        comparison_result = "✓ 正確" if is_correct else "✗ 錯誤"
                        
                        # 添加到表格（帶顏色標籤）
                        tag = 'correct' if is_correct else 'incorrect'
                        self.comparison_tree.insert('', 'end', values=(
                            original_index,
                            short_text,
                            original_sentiment_cn,
                            predicted_sentiment_cn,
                            comparison_result
                        ), tags=(tag,))
                else:
                    messagebox.showwarning("警告", "無法獲取原始文本數據")
                    return
            
            # 更新狀態 - 使用實際的標籤數據計算準確率
            total_samples = len(predicted_labels)
            if len(true_labels) == len(predicted_labels):
                correct_count = sum(1 for i in range(len(predicted_labels)) 
                                  if true_labels[i] == predicted_labels[i])
            else:
                correct_count = 0
            accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
            
            messagebox.showinfo("更新完成", 
                              f"已更新前 {display_count} 筆比對結果\n"
                              f"總樣本數: {total_samples}\n"
                              f"正確預測: {correct_count}\n"
                              f"準確率: {accuracy:.4f}%")
            
        except Exception as e:
            messagebox.showerror("錯誤", f"更新比對報告時發生錯誤：{str(e)}")
    
    def _get_original_sentiment(self, df, index):
        """獲取原始情感標籤"""
        for col in ['sentiment', 'label', 'category']:
            if col in df.columns:
                return str(df.iloc[index][col])
        return "unknown"
    
    def _update_mechanism_combo(self, classification_results):
        """更新機制選擇下拉菜單"""
        try:
            # 獲取所有可用的機制名稱並轉換為中文
            mechanism_names = []
            for mechanism in classification_results.keys():
                display_name = self._format_mechanism_name(mechanism)
                mechanism_names.append(display_name)
            
            # 更新下拉菜單選項
            if hasattr(self, 'mechanism_combo') and self.mechanism_combo is not None:
                self.mechanism_combo['values'] = mechanism_names
                if mechanism_names:
                    # 預設選擇第一個機制
                    self.mechanism_combo.set(mechanism_names[0])
                    self.update_comparison_btn['state'] = 'normal'
                    self.quick_update_btn['state'] = 'normal'
                    
        except Exception as e:
            print(f"更新機制下拉菜單時發生錯誤: {str(e)}")
    
    def quick_update_best_mechanism(self):
        """快速更新最佳機制的比對報告"""
        try:
            if not hasattr(self, 'analysis_results') or not self.analysis_results:
                messagebox.showwarning("警告", "尚無分析結果")
                return
            
            # 獲取最佳機制
            summary = self.analysis_results.get('summary', {})
            best_mechanism = summary.get('best_attention_mechanism', None)
            
            if not best_mechanism:
                messagebox.showwarning("警告", "無法找到最佳機制信息")
                return
            
            # 格式化機制名稱
            best_mechanism_display = self._format_mechanism_name(best_mechanism)
            
            # 設置到下拉菜單
            if hasattr(self, 'mechanism_combo') and self.mechanism_combo is not None:
                self.mechanism_combo.set(best_mechanism_display)
                
                # 直接調用更新報告
                self.update_comparison_report()
            else:
                messagebox.showerror("錯誤", "機制選擇組件未初始化")
                
        except Exception as e:
            messagebox.showerror("錯誤", f"快速更新失敗：{str(e)}")
    
    def create_cross_validation_tab(self):
        """第三分頁：交叉驗證（保留原有功能）"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text=" 交叉驗證 ")
        
        # 主要容器
        main_frame = ttk.Frame(frame)
        main_frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        # 標題
        title_label = ttk.Label(main_frame, text="交叉驗證分析", font=FONTS['title'])
        title_label.pack(pady=(0, 15))
        
        # 配置區域
        config_frame = ttk.LabelFrame(main_frame, text="交叉驗證配置", padding=15)
        config_frame.pack(fill='x', pady=(0, 15))
        
        # 折數設定
        fold_frame = ttk.Frame(config_frame)
        fold_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(fold_frame, text="交叉驗證折數:").pack(side='left')
        self.cv_folds = tk.IntVar(value=5)
        fold_spin = ttk.Spinbox(fold_frame, from_=3, to=10, textvariable=self.cv_folds, width=10)
        fold_spin.pack(side='left', padx=(5, 0))
        
        # 注意力機制選擇
        attention_frame = ttk.Frame(config_frame)
        attention_frame.pack(fill='x')
        
        ttk.Label(attention_frame, text="注意力機制:").pack(side='left')
        
        attention_options_frame = ttk.Frame(attention_frame)
        attention_options_frame.pack(side='left', padx=(5, 0))
        
        self.cv_attentions = {}
        attention_options = [
            ('no', '無注意力'),
            ('similarity', '相似度'),
            ('keyword', '關鍵詞'),
            ('self', '自注意力'),
            ('combined', '組合注意力')
        ]
        
        for key, label in attention_options:
            var = tk.BooleanVar(value=True if key in ['no', 'similarity', 'self'] else False)
            self.cv_attentions[key] = var
            check = ttk.Checkbutton(attention_options_frame, text=label, variable=var)
            check.pack(side='left', padx=(0, 10))
        
        # 執行按鈕
        self.cv_btn = ttk.Button(config_frame, text="執行交叉驗證", command=self.run_cross_validation)
        self.cv_btn.pack(pady=(15, 0))
        
        # 結果顯示
        results_frame = ttk.LabelFrame(main_frame, text="交叉驗證結果", padding=10)
        results_frame.pack(fill='both', expand=True)
        
        self.cv_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.cv_text.pack(fill='both', expand=True)
        self.cv_text.insert('1.0', "等待交叉驗證結果...")
    
    def run_cross_validation(self):
        """執行交叉驗證"""
        messagebox.showinfo("功能提示", "交叉驗證功能將在後續版本中實現")
    
    def show_encoder_info(self):
        """顯示編碼器資訊"""
        info = """編碼器類型說明：

🔤 BERT: 雙向編碼器，適合理解上下文
• 特點：預訓練效果佳，準確率高
• 適用：一般文本分析任務

🤖 GPT: 生成式預訓練模型
• 特點：強大的語言建模能力
• 適用：文本生成和理解任務

🔄 T5: Text-to-Text 轉換模型
• 特點：統一的文本處理框架
• 適用：多種NLP任務

📊 CNN: 卷積神經網路
• 特點：快速、輕量級
• 適用：大規模文本分類

🧠 ELMo: 深度雙向語言模型
• 特點：上下文敏感的詞向量
• 適用：需要詳細語言理解的任務

🔄 RoBERTa: 強化版BERT
• 特點：更好的預訓練策略
• 適用：需要高準確率的任務

⚡ DistilBERT: 輕量版BERT
• 特點：速度快、資源消耗少
• 適用：實時應用或資源受限環境"""
        
        messagebox.showinfo("編碼器說明", info)
    
    def show_classifier_info(self):
        """顯示分類器資訊"""
        info = """分類器類型說明：

🚀 XGBoost: 極端梯度提升
• 特點：高準確率、支援GPU加速
• 適用：結構化數據分類

📈 Logistic Regression: 邏輯回歸
• 特點：簡單、可解釋性強
• 適用：線性可分問題

🌳 Random Forest: 隨機森林
• 特點：防止過擬合、穩定性好
• 適用：複雜特徵關係

🎯 SVM Linear: 線性支援向量機
• 特點：在高維空間表現良好
• 適用：文本分類任務

🎲 Naive Bayes: 樸素貝葉斯
• 特點：快速、適合小數據集
• 適用：文本分類的基準模型"""
        
        messagebox.showinfo("分類器說明", info)
    
    def on_adaptive_weights_changed(self):
        """當智能權重學習選項改變時"""
        if self.use_adaptive_weights.get():
            self.weight_config_btn.config(state='normal')
        else:
            self.weight_config_btn.config(state='disabled')
    
    def show_weight_config(self):
        """顯示權重配置窗口"""
        from gui.weight_config_window import WeightConfigWindow
        WeightConfigWindow(self.root, self)
    
    def restart_application(self):
        """重製程式"""
        import sys
        import os
        from tkinter import messagebox
        
        # 確認對話框
        result = messagebox.askyesno(
            "確認重製", 
            "確定要重製程式嗎？\n\n這將會：\n• 關閉當前程式\n• 清除所有處理進度\n• 重新啟動程式\n• 回到初始狀態",
            icon='warning'
        )
        
        if result:
            try:
                # 顯示重啟訊息
                self.overall_status.config(text="正在重製程式...", foreground=COLORS['warning'])
                self.root.update()
                
                # 獲取當前程式路徑
                if getattr(sys, 'frozen', False):
                    # 如果是打包的執行檔
                    program_path = sys.executable
                    args = []
                else:
                    # 如果是Python腳本
                    program_path = sys.executable
                    main_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Part05_Main.py')
                    if os.path.exists(main_script):
                        args = [main_script]
                    else:
                        # 尋找主程式檔案
                        possible_mains = ['Part05_Main.py', 'main.py', 'gui_main.py']
                        main_script = None
                        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        
                        for main_name in possible_mains:
                            test_path = os.path.join(base_dir, main_name)
                            if os.path.exists(test_path):
                                main_script = test_path
                                break
                        
                        if main_script:
                            args = [main_script]
                        else:
                            # 如果找不到主程式，使用當前模組
                            args = ['-m', 'gui.main_window']
                
                # 關閉當前視窗
                self.root.quit()
                self.root.destroy()
                
                # 啟動新程式
                import subprocess
                subprocess.Popen([program_path] + args)
                
                # 結束當前程式
                sys.exit(0)
                
            except Exception as e:
                # 如果重啟失敗，顯示錯誤訊息
                messagebox.showerror(
                    "重製失敗", 
                    f"程式重製失敗：{str(e)}\n\n請手動關閉程式並重新啟動。"
                )
                self.overall_status.config(text="重製失敗", foreground=COLORS['error'])
    
    def create_run_dir_label(self):
        """創建run目錄標籤"""
        self.run_dir_frame = ttk.Frame(self.root)
        self.run_dir_frame.pack(side='bottom', fill='x', padx=15, pady=(0, 10))
        
        self.run_dir_label = ttk.Label(self.run_dir_frame, 
                                     text=f"當前輸出目錄: {self.run_manager.get_run_dir('bert')}",
                                     font=('TkDefaultFont', 8))
        self.run_dir_label.pack(anchor='w')

def main():
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()