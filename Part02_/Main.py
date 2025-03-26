"""
Tkinter單線程跨領域情感分析系統應用程式
採用事件驅動設計，完全避免使用後台線程
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import subprocess
import os
import sys
import logging
import nltk
import json
import traceback
import time
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

# 確保matplotlib使用非互動模式
plt.ioff()

# 自定義模塊導入
from src.console_output import ConsoleOutputManager
from src.data_importer import DataImporter
from src.bert_embedder import BertEmbedder
from src.lda_aspect_extractor import LDATopicExtractor
from src.aspect_vector_calculator import AspectVectorCalculator
from src.result_manager import ResultManager
from src.settings.visualization_config import configure_chinese_fonts, check_chinese_display

class TaskProcessor:
    """
    處理長時間運行任務的類，不使用多線程，而是通過Tkinter的after機制分段執行任務
    """
    def __init__(self, root, status_var, progress_var, on_complete=None):
        self.root = root
        self.status_var = status_var
        self.progress_var = progress_var
        self.on_complete = on_complete
        self.is_running = False
        self.current_step = 0
        self.task_start_time = 0
        self.task_gen = None
        self.task_result = None
        self.task_error = None
        self.task_func = None
        self.task_args = None
        self.task_kwargs = None
        
    def start_task(self, task_func, *args, **kwargs):
        """
        開始一個長時間運行的任務
        
        Args:
            task_func: 一個函數或生成器函數，用於執行任務
            *args, **kwargs: 傳遞給task_func的參數
        """
        if self.is_running:
            return False
            
        self.is_running = True
        self.current_step = 0
        self.task_result = None
        self.task_error = None
        self.task_start_time = time.time()
        
        # 保存任務函數和參數
        self.task_func = task_func
        self.task_args = args
        self.task_kwargs = kwargs
        
        # 安排執行任務
        self.root.after(50, self._execute_task)
        return True
    
    def _execute_task(self):
        """執行任務函數，初始化生成器"""
        if not self.is_running:
            return
            
        try:
            # 執行任務函數
            result = self.task_func(*self.task_args, **self.task_kwargs)
            
            # 檢查結果類型
            if hasattr(result, '__iter__') and hasattr(result, '__next__'):
                # 如果返回的是生成器，則開始逐步處理
                self.task_gen = result
                self.root.after(50, self._process_task_step)
            else:
                # 如果返回的是普通值，則直接完成任務
                self.task_result = result
                self._finish_task()
        except Exception as e:
            # 處理任務啟動時的錯誤
            self.task_error = e
            self.status_var.set(f"錯誤: {str(e)}")
            self.progress_var.set(0)
            self._finish_task()
        
    def _process_task_step(self):
        """處理任務的一個步驟"""
        if not self.is_running or not self.task_gen:
            return
            
        try:
            # 執行生成器的下一步
            result = next(self.task_gen)
            
            # 檢查結果格式
            if isinstance(result, tuple) and len(result) >= 2:
                status_msg, progress = result[0], result[1]
                
                # 更新UI
                self.status_var.set(status_msg)
                if progress >= 0:
                    self.progress_var.set(progress)
                
                # 增加步驟計數
                self.current_step += 1
                
                # 安排下一步執行
                self.root.after(50, self._process_task_step)
            else:
                # 不符合預期格式，當作最終結果處理
                self._handle_final_result(result)
                
        except StopIteration as e:
            # 生成器完成，從異常中獲取返回值
            self._handle_stop_iteration(e)
        except Exception as e:
            # 處理執行過程中的錯誤
            self.task_error = e
            self.status_var.set(f"錯誤: {str(e)}")
            self.progress_var.set(0)
            self._finish_task()
    
    def _handle_stop_iteration(self, stop_iter_exception):
        """處理StopIteration異常，提取返回值"""
        # 從StopIteration中獲取返回值
        if hasattr(stop_iter_exception, "value"):
            result = stop_iter_exception.value
            self._handle_final_result(result)
        else:
            # 沒有返回值
            self.task_result = None
            self._finish_task()
    
    def _handle_final_result(self, result):
        """處理最終結果"""
        # 保存結果
        self.task_result = result
        # 如果是生成器，尝试执行它获取实际结果
        if hasattr(result, '__iter__') and hasattr(result, '__next__'):
            try:
                # 尝试执行生成器，获取所有值，取最后一个作为结果
                final_result = None
                for item in result:
                    final_result = item
                self.task_result = final_result
            except Exception:
                # 如果执行失败，保留原始结果
                pass
                
        # 完成任務
        self._finish_task()
    
    def _finish_task(self):
        """完成任務並清理"""
        execution_time = time.time() - self.task_start_time
        
        # 清理資源
        was_running = self.is_running
        self.is_running = False
        self.task_gen = None
        
        # 只有在曾經運行時才呼叫回調
        if was_running and self.on_complete:
            # 在呼叫回調前記錄和清除結果
            result, error = self.task_result, self.task_error
            self.task_result = None
            self.task_error = None
            
            # 呼叫完成回調
            try:
                self.on_complete(result, error, execution_time)
            except Exception as callback_error:
                print(f"回調錯誤: {callback_error}")

class CrossDomainSentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("跨領域情感分析系統-數據處理 v4.0")
        
        # 設置視窗最大化
        self.root.state('zoomed')
        # 設置最小視窗大小
        self.root.minsize(800, 600)
        
        # 計算視窗置中的位置
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 800
        window_height = 600
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 設置日誌
        log_dir = Path("./Part02_/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger('main_app')
        self.logger.setLevel(logging.INFO)
        
        # 檔案處理器設定
        file_handler = logging.FileHandler(log_dir / 'app.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 創建變數
        self.file_path = tk.StringVar()
        self.topic_count = tk.StringVar(value="10")  # 預設LDA主題數量
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="準備就緒。")
        
        # 設置風格
        self.style = ttk.Style()
        self.style.configure('TNotebook', tabposition='n')
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Status.TLabel', font=('Arial', 10), foreground='green')
        
        # 初始化工作目錄
        self._init_directories()
        
        # 當前數據集ID
        self.current_dataset_id = None
        
        # 初始化結果管理器
        self.result_manager = ResultManager(logger=self.logger)
        
        # 初始化任務處理器
        self.task_processor = TaskProcessor(
            root=self.root,
            status_var=self.status_var,
            progress_var=self.progress_var,
            on_complete=self._on_task_complete
        )
        
        # 為了在方法之間共享數據
        self.processed_data_path = None
        self.bert_embeddings_path = None
        self.bert_metadata_path = None
        self.lda_model_path = None
        self.topics_path = None
        self.topic_metadata_path = None
        self.embedding_dim = None
        self.topic_count = None
        self.data_source = None
        
        # 建立介面
        self._create_widgets()
        
        # 確保中文字體設置正確
        self._setup_chinese_support()
        
        # 從結果管理器加載當前分析數據
        self._load_recent_results()
        
        # 設置窗口關閉處理
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.logger.info("應用程式啟動")
        
    def _init_directories(self):
        """初始化所有工作目錄"""
        # 基本目錄
        self.base_dir = Path("./Part02_")
        self.data_dir = self.base_dir / "data"
        self.results_dir = self.base_dir / "results"
        
        # 子目錄
        self.processed_data_dir = self.results_dir / "01_processed_data"
        self.bert_embeddings_dir = self.results_dir / "02_bert_embeddings"
        self.lda_topics_dir = self.results_dir / "03_lda_topics"
        self.aspect_vectors_dir = self.results_dir / "04_aspect_vectors"
        self.visualizations_dir = self.results_dir / "visualizations"
        self.exports_dir = self.results_dir / "exports"
        
        # 可視化子目錄
        self.topic_vis_dir = self.visualizations_dir / "topics"
        self.vector_vis_dir = self.visualizations_dir / "vectors"
        
        # 確保所有目錄存在
        for directory in [
            self.data_dir, self.results_dir, 
            self.processed_data_dir, self.bert_embeddings_dir, 
            self.lda_topics_dir, self.aspect_vectors_dir,
            self.visualizations_dir, self.exports_dir,
            self.topic_vis_dir, self.vector_vis_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_chinese_support(self):
        """設置中文字體支援"""
        try:
            font_config_result = configure_chinese_fonts()
            self.logger.info(f"中文字體配置: {font_config_result}")
            # 計劃延遲測試，確保UI已完全載入
            self.root.after(1000, self._test_chinese_font)
        except Exception as e:
            self.logger.warning(f"中文字體配置失敗: {str(e)}")
    
    def _test_chinese_font(self):
        """測試中文字體顯示"""
        try:
            display_result = check_chinese_display()
            if not display_result:
                self.logger.warning("中文顯示測試不成功，圖表中的中文可能無法正確顯示")
        except Exception as e:
            self.logger.warning(f"中文顯示測試失敗: {str(e)}")
    
    def _create_widgets(self):
        """建立主要介面元素"""
        # 建立分頁
        self.tab_control = ttk.Notebook(self.root)
        
        # 建立各個分頁
        self.tab_data_processing = ttk.Frame(self.tab_control)
        self.tab_results = ttk.Frame(self.tab_control)
        
        # 添加分頁到控制器
        tab_font = ('Arial', 10, 'bold')
        self.tab_control.add(self.tab_data_processing, text="　資料處理　")
        self.tab_control.add(self.tab_results, text="　結果瀏覽　")

        self.style.configure('TNotebook.Tab', font=tab_font)
        self.tab_control.pack(expand=1, fill="both")
        
        # 設置資料處理分頁內容
        self._setup_data_processing_tab()
        
        # 設置結果瀏覽分頁內容
        self._setup_results_tab()
        
        # 底部狀態欄和進度條
        self._setup_status_bar()
    
    def _setup_data_processing_tab(self):
        """資料處理分頁界面"""
        main_frame = ttk.Frame(self.tab_data_processing, padding=10)
        main_frame.pack(fill="both", expand=True)

        # 創建一個專用資料來源框架
        data_source_frame = ttk.LabelFrame(main_frame, text="資料來源選擇", padding=10)
        data_source_frame.pack(fill="x", pady=5)
        
        # 添加切換標籤頁的按鈕
        source_buttons_frame = ttk.Frame(data_source_frame)
        source_buttons_frame.pack(fill="x", pady=5)
        
        # 創建資料來源標籤頁
        source_notebook = ttk.Notebook(data_source_frame)
        source_notebook.pack(fill="x", pady=5)
        
        # ===IMDB 電影評論頁面===
        imdb_frame = ttk.Frame(source_notebook, padding=5)
        source_notebook.add(imdb_frame, text="IMDB電影評論")
        
        ttk.Label(imdb_frame, text="選擇IMDB電影評論資料集檔案").pack(anchor="w", pady=5)
        imdb_file_frame = ttk.Frame(imdb_frame)
        imdb_file_frame.pack(fill="x", pady=2)
        
        self.imdb_path = tk.StringVar()
        ttk.Label(imdb_file_frame, text="IMDB檔案:").pack(side="left", padx=(0,5))
        ttk.Label(imdb_file_frame, textvariable=self.imdb_path, width=50).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(imdb_file_frame, text="選擇", command=lambda: self._select_file_for_source("imdb")).pack(side="right")
        
        ttk.Button(imdb_frame, text="處理IMDB數據", command=lambda: self._import_data_with_source("imdb")).pack(anchor="e", pady=5)
        
        # ===Amazon 產品評論頁面===
        amazon_frame = ttk.Frame(source_notebook, padding=5)
        source_notebook.add(amazon_frame, text="Amazon產品評論")
        
        ttk.Label(amazon_frame, text="選擇Amazon產品評論資料集檔案").pack(anchor="w", pady=5)
        amazon_file_frame = ttk.Frame(amazon_frame)
        amazon_file_frame.pack(fill="x", pady=2)
        
        self.amazon_path = tk.StringVar()
        ttk.Label(amazon_file_frame, text="Amazon檔案:").pack(side="left", padx=(0,5))
        ttk.Label(amazon_file_frame, textvariable=self.amazon_path, width=50).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(amazon_file_frame, text="選擇", command=lambda: self._select_file_for_source("amazon")).pack(side="right")
        
        ttk.Button(amazon_frame, text="處理Amazon數據", command=lambda: self._import_data_with_source("amazon")).pack(anchor="e", pady=5)
        
        # ===Yelp 餐廳評論頁面===
        yelp_frame = ttk.Frame(source_notebook, padding=5)
        source_notebook.add(yelp_frame, text="Yelp餐廳評論")
        
        ttk.Label(yelp_frame, text="選擇Yelp的business和review文件，並自動合併處理").pack(anchor="w", pady=5)
        
        # Business文件選擇
        business_frame = ttk.Frame(yelp_frame)
        business_frame.pack(fill="x", pady=2)
        
        self.business_path = tk.StringVar()
        ttk.Label(business_frame, text="Business文件:").pack(side="left", padx=(0,5))
        ttk.Label(business_frame, textvariable=self.business_path, width=50).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(business_frame, text="選擇", command=self._select_business_file).pack(side="right")
        
        # Review文件選擇
        review_frame = ttk.Frame(yelp_frame)
        review_frame.pack(fill="x", pady=2)
        
        self.review_path = tk.StringVar()
        ttk.Label(review_frame, text="Review文件:   ").pack(side="left", padx=(0,5))
        ttk.Label(review_frame, textvariable=self.review_path, width=50).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Button(review_frame, text="選擇", command=self._select_review_file).pack(side="right")
        
        # 抽樣設置
        sample_frame = ttk.Frame(yelp_frame)
        sample_frame.pack(fill="x", pady=2)
        
        ttk.Label(sample_frame, text="抽樣數量:     ").pack(side="left", padx=(0,5))
        self.sample_size = tk.StringVar(value="50000")
        ttk.Entry(sample_frame, textvariable=self.sample_size, width=10).pack(side="left")
        
        ttk.Label(sample_frame, text="筆評論").pack(side="left", padx=5)
        
        # 處理按鈕
        ttk.Button(yelp_frame, text="處理Yelp數據", command=self._process_yelp_data).pack(anchor="e", pady=5)
        
        # 通用文件選擇頁面
        generic_frame = ttk.Frame(source_notebook, padding=5)
        source_notebook.add(generic_frame, text="其他資料")
        
        ttk.Label(generic_frame, text="選擇評論資料集導入系統 (CSV, JSON, TXT格式)").pack(anchor="w", pady=5)
        
        file_frame = ttk.Frame(generic_frame)
        file_frame.pack(fill="x", pady=5)
        
        ttk.Button(file_frame, text="選擇文件", command=self._select_file).pack(side="left")
        ttk.Label(file_frame, textvariable=self.file_path).pack(side="left", padx=10)
        ttk.Button(file_frame, text="開始導入數據", command=self._import_data).pack(side="right")

        # ===步驟2: BERT提取語義表示===
        step2_frame = ttk.LabelFrame(main_frame, text="步驟 2: BERT提取語義表示", padding=10)
        step2_frame.pack(fill="x", pady=5)
        
        ttk.Label(step2_frame, text="使用BERT模型為評論文本生成向量表示").pack(anchor="w", pady=5)
        ttk.Button(step2_frame, text="執行BERT語義提取", command=self._extract_bert_embeddings).pack(anchor="e", pady=5)
        
        # ===步驟3: LDA面向切割===
        step3_frame = ttk.LabelFrame(main_frame, text="步驟 3: LDA面向切割", padding=10)
        step3_frame.pack(fill="x", pady=5)

        ttk.Label(step3_frame, text="使用LDA進行主題建模，識別不同面向").pack(anchor="w", pady=5)

        # 添加配置區域
        config_frame = ttk.Frame(step3_frame)
        config_frame.pack(fill="x", pady=5)

        # 添加自動偵測選項
        self.auto_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config_frame, text="自動偵測資料來源和主題數量", 
                    variable=self.auto_detect_var, 
                    command=self._toggle_topic_config).pack(anchor="w", pady=2)

        # 資料來源選擇區域 - 將框架保存為類屬性
        self.source_frame = ttk.Frame(config_frame)
        self.source_frame.pack(fill="x", pady=2)
        ttk.Label(self.source_frame, text="資料來源:").pack(side="left", padx=(20, 5))
        self.data_source_var = tk.StringVar(value="imdb")
        self.data_source_combo = ttk.Combobox(self.source_frame, textvariable=self.data_source_var, 
                                        state="readonly", width=15)
        self.data_source_combo['values'] = ["imdb", "amazon", "yelp"]
        self.data_source_combo.pack(side="left", padx=5)
        self.data_source_combo['state'] = 'disabled'  # 初始設為禁用

        # 主題數量設定區域 - 將框架保存為類屬性
        self.topic_frame = ttk.Frame(config_frame)
        self.topic_frame.pack(fill="x", pady=2)
        ttk.Label(self.topic_frame, text="主題數量:").pack(side="left", padx=(20, 5))
        self.topic_count_var = tk.StringVar(value="10")
        self.topic_count_entry = ttk.Entry(self.topic_frame, textvariable=self.topic_count_var, width=8)
        self.topic_count_entry.pack(side="left", padx=5)
        self.topic_count_entry['state'] = 'disabled'  # 初始設為禁用

        # 執行按鈕
        ttk.Button(step3_frame, text="執行LDA面向切割", command=self._perform_lda).pack(anchor="e", pady=5)
        
        # ===步驟4: 計算面向相關句子的平均向量===
        step4_frame = ttk.LabelFrame(main_frame, text="步驟 4: 計算面向相關句子的平均向量", padding=10)
        step4_frame.pack(fill="x", pady=5)
        
        ttk.Label(step4_frame, text="為每個識別出的面向計算代表性向量").pack(anchor="w", pady=5)
        
        # 按鈕布局
        vector_frame = ttk.Frame(step4_frame)
        vector_frame.pack(fill="x", pady=5)
        
        # 含按鈕的容器框架
        buttons_container = ttk.Frame(vector_frame)
        buttons_container.pack(side="right")
        
        # 面向計算按鈕
        ttk.Button(buttons_container, text="執行計算", command=self._calculate_aspect_vectors).pack(pady=(0, 5))
        # 匯出按鈕
        ttk.Button(buttons_container, text="匯出平均向量", command=self._export_vectors).pack()
    
    def _setup_results_tab(self):
        """結果瀏覽分頁界面"""
        # 實作結果瀏覽分頁
        main_frame = ttk.Frame(self.tab_results, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # 上方控制區
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=5)
        
        # 刷新按鈕
        ttk.Button(control_frame, text="刷新結果", command=self._refresh_results).pack(side="left", padx=5)
        
        # 生成概覽報告按鈕
        ttk.Button(control_frame, text="生成概覽報告", command=self._generate_overview_report).pack(side="left", padx=5)
        
        # 打開結果目錄按鈕
        ttk.Button(control_frame, text="打開結果目錄", command=lambda: self._open_file(self.results_dir)).pack(side="right", padx=5)
        
        # 數據集下拉選單
        dataset_frame = ttk.Frame(main_frame)
        dataset_frame.pack(fill="x", pady=5)
        
        ttk.Label(dataset_frame, text="選擇數據集:").pack(side="left", padx=5)
        
        self.dataset_combo = ttk.Combobox(dataset_frame, state="readonly", width=40)
        self.dataset_combo.pack(side="left", padx=5, fill="x", expand=True)
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_selected)
        
        # 查看報告按鈕
        ttk.Button(dataset_frame, text="查看處理報告", command=self._view_dataset_report).pack(side="right", padx=5)
        
        # 結果分類顯示區
        result_notebook = ttk.Notebook(main_frame)
        result_notebook.pack(fill="both", expand=True, pady=10)
        
        # 處理資料頁面
        self.data_tab = ttk.Frame(result_notebook)
        result_notebook.add(self.data_tab, text="處理資料")
        
        # 可視化頁面
        self.vis_tab = ttk.Frame(result_notebook)
        result_notebook.add(self.vis_tab, text="可視化")
        
        # 配置每個標籤頁
        self._setup_data_tab(self.data_tab)
        self._setup_vis_tab(self.vis_tab)
    
    def _select_file_for_source(self, source_type):
        """為特定資料來源選擇文件"""
        file_path = filedialog.askopenfilename(
            title=f"選擇{source_type}評論資料文件",
            filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            if source_type == "imdb":
                self.imdb_path.set(file_path)
            elif source_type == "amazon":
                self.amazon_path.set(file_path)
            self.status_var.set(f"已選擇{source_type}文件: {os.path.basename(file_path)}")
            self.logger.info(f"已選擇{source_type}文件: {file_path}")

    def _import_data_with_source(self, source_type):
        """導入特定來源的數據"""
        # 獲取對應來源的文件路徑
        if source_type == "imdb":
            file_path = self.imdb_path.get()
        elif source_type == "amazon":
            file_path = self.amazon_path.get()
        else:
            file_path = ""
            
        if not file_path:
            messagebox.showerror("錯誤", f"請先選擇{source_type}文件!")
            return
            
        # 設置數據來源
        self.data_source = source_type
        
        # 設置通用文件路徑，以便後續處理
        self.file_path.set(file_path)
        
        # 創建數據集ID
        dataset_name = os.path.basename(file_path).split('.')[0]
        self.current_dataset_id = self.result_manager.register_dataset(dataset_name, file_path)
        
        # 開始導入任務
        self.task_processor.start_task(self._import_data_task, file_path)

    def _setup_data_tab(self, parent):
        """配置數據標籤頁"""
        # 創建Treeview
        columns = ("filename", "step", "path", "metadata")
        self.data_tree = ttk.Treeview(parent, columns=columns, show="headings")
        
        # 設置列標題
        self.data_tree.heading("filename", text="檔案名稱")
        self.data_tree.heading("step", text="處理步驟")
        self.data_tree.heading("path", text="路徑")
        self.data_tree.heading("metadata", text="元數據")
        
        # 設置列寬
        self.data_tree.column("filename", width=200)
        self.data_tree.column("step", width=100)
        self.data_tree.column("path", width=300)
        self.data_tree.column("metadata", width=200)
        
        # 添加滾動條
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # 佈局
        self.data_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 雙擊打開文件
        self.data_tree.bind("<Double-1>", self._on_data_double_click)
    
    def _toggle_topic_config(self):
        """根據自動偵測選項切換配置區域的啟用狀態"""
        state = "disabled" if self.auto_detect_var.get() else "normal"
        # 設置資料來源下拉選單狀態
        self.data_source_combo['state'] = state
        # 設置主題數量輸入框狀態
        self.topic_count_entry['state'] = state

    def _setup_vis_tab(self, parent):
        """配置可視化標籤頁 - 網格布局版"""
        # 上方控制區
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=5)
        
        # 左側控制選項
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(side="left", fill="x", expand=True)
        
        # 選擇可視化類型下拉選單
        ttk.Label(options_frame, text="可視化類型:").pack(side="left", padx=5)
        self.vis_type_combo = ttk.Combobox(options_frame, state="readonly", values=["全部", "主題可視化", "向量可視化"])
        self.vis_type_combo.pack(side="left", padx=5)
        self.vis_type_combo.current(0)
        self.vis_type_combo.bind("<<ComboboxSelected>>", self._on_vis_type_changed)
        
        # 顯示模式選擇
        ttk.Label(options_frame, text="顯示模式:").pack(side="left", padx=(15, 5))
        self.vis_mode_combo = ttk.Combobox(options_frame, state="readonly", values=["縮略圖", "詳細信息"])
        self.vis_mode_combo.pack(side="left", padx=5)
        self.vis_mode_combo.current(0)
        self.vis_mode_combo.bind("<<ComboboxSelected>>", self._refresh_vis_display)
        
        # 右側按鈕
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(side="right")
        
        # 刷新按鈕
        ttk.Button(buttons_frame, text="刷新", command=self._refresh_vis_display).pack(side="right", padx=5)
        
        # 創建主顯示區
        self.vis_main_frame = ttk.Frame(parent)
        self.vis_main_frame.pack(fill="both", expand=True, pady=5)
        
        # 創建滾動視圖
        self.vis_canvas = tk.Canvas(self.vis_main_frame, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(self.vis_main_frame, orient="vertical", command=self.vis_canvas.yview)
        
        # 內容框架
        self.vis_frame = ttk.Frame(self.vis_canvas)
        
        # 配置滾動
        self.vis_frame.bind(
            "<Configure>",
            lambda e: self.vis_canvas.configure(scrollregion=self.vis_canvas.bbox("all"))
        )
        
        self.vis_canvas.create_window((0, 0), window=self.vis_frame, anchor="nw")
        self.vis_canvas.configure(yscrollcommand=scrollbar.set)
        
        # 鼠標滾輪捲動
        def _on_mousewheel(event):
            # Windows和macOS的滾輪事件有所不同
            if sys.platform == "win32":
                self.vis_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                self.vis_canvas.yview_scroll(int(-1*event.delta), "units")
        
        self.vis_canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
        self.vis_canvas.bind_all("<Button-4>", lambda e: self.vis_canvas.yview_scroll(-1, "units"))  # Linux
        self.vis_canvas.bind_all("<Button-5>", lambda e: self.vis_canvas.yview_scroll(1, "units"))  # Linux
        
        # 放置Canvas和滾動條
        self.vis_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 保存所有可視化項目的引用
        self.vis_items = []
        
        # 標記尚未顯示
        self.vis_initialized = False
    
    def _refresh_vis_display(self, event=None):
        """根據選擇的模式刷新可視化顯示"""
        mode = self.vis_mode_combo.get()
        vis_type = self.vis_type_combo.get()
        
        # 清空現有顯示
        for widget in self.vis_frame.winfo_children():
            widget.destroy()
            
        # 過濾可視化項目
        filtered_items = []
        for item in self.vis_items:
            if vis_type == "全部" or item.get("vis_type", "") == vis_type:
                filtered_items.append(item)
        
        if mode == "縮略圖":
            self._display_thumbnail_grid(filtered_items)
        else:
            self._display_detailed_list(filtered_items)

    def _display_thumbnail_grid(self, items):
        """以網格布局顯示縮略圖"""
        if not items:
            ttk.Label(self.vis_frame, text="沒有可用的可視化項目", font=("Arial", 12)).pack(padx=20, pady=20)
            return
        
        # 創建網格框架
        grid_frame = ttk.Frame(self.vis_frame)
        grid_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 配置網格 - 每行顯示3個項目
        COLUMNS = 3
        row = 0
        col = 0
        
        # 計算合適的縮略圖大小
        canvas_width = self.vis_canvas.winfo_width() or 800  # 預設值
        thumb_width = min(200, (canvas_width - 80) // COLUMNS)  # 考慮邊距
        thumb_height = thumb_width * 3 // 4  # 保持4:3比例
        
        # 放置每個項目
        for i, item in enumerate(items):
            # 創建項目框架
            cell_frame = ttk.Frame(grid_frame, borderwidth=2, relief="groove")
            cell_frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            try:
                # 標題
                title = item.get("filename", f"項目 {i+1}")
                title_label = ttk.Label(cell_frame, text=title, font=("Arial", 9, "bold"), wraplength=thumb_width)
                title_label.pack(pady=(5, 2))
                
                # 縮略圖
                image_path = item.get("path", "")
                if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    try:
                        # 讀取原始圖片
                        original_img = Image.open(image_path)
                        
                        # 計算縮略圖尺寸 (保持原始比例)
                        width, height = original_img.size
                        ratio = min(thumb_width/width, thumb_height/height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)
                        
                        # 創建縮略圖
                        thumb_img = original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        # 創建白色背景
                        bg_img = Image.new('RGB', (thumb_width, thumb_height), (240, 240, 240))
                        # 將縮略圖貼在中央
                        offset = ((thumb_width - new_width) // 2, (thumb_height - new_height) // 2)
                        bg_img.paste(thumb_img, offset)
                        
                        # 轉換為Tkinter可用的格式
                        photo = ImageTk.PhotoImage(bg_img)
                        
                        # 創建圖像標籤
                        img_label = ttk.Label(cell_frame, image=photo)
                        img_label.image = photo  # 保持引用
                        img_label.pack(pady=5)
                        
                        # 添加點擊事件
                        img_label.bind("<Button-1>", lambda e, path=image_path: self._open_file(path))
                        
                        # 關閉圖片以釋放資源
                        original_img.close()
                        thumb_img.close()
                        bg_img.close()
                    except Exception as e:
                        self.logger.error(f"載入圖片時出錯: {str(e)}")
                        ttk.Label(cell_frame, text="圖片載入失敗").pack(pady=10)
                else:
                    # 沒有圖片或圖片不存在
                    ttk.Label(cell_frame, text="[無法顯示圖片]", font=("Arial", 9)).pack(pady=10)
                
                # 類型標籤
                vis_type = item.get("vis_type", "未知類型")
                type_label = ttk.Label(cell_frame, text=vis_type, font=("Arial", 8), foreground="gray")
                type_label.pack(pady=2)
                
                # 操作按鈕
                btn_frame = ttk.Frame(cell_frame)
                btn_frame.pack(fill="x", pady=5, padx=5)
                
                ttk.Button(
                    btn_frame, text="查看", width=8,
                    command=lambda path=image_path: self._open_file(path)
                ).pack(side="left", padx=2)
                
                # 添加額外信息按鈕
                ttk.Button(
                    btn_frame, text="詳情", width=8, 
                    command=lambda item=item: self._show_visualization_details(item)
                ).pack(side="right", padx=2)
                
            except Exception as e:
                self.logger.error(f"處理可視化項目時出錯: {str(e)}")
                traceback.print_exc()
                ttk.Label(cell_frame, text=f"錯誤: {str(e)}", foreground="red").pack(pady=10)
            
            # 更新網格位置
            col += 1
            if col >= COLUMNS:
                col = 0
                row += 1
        
        # 確保每行的列寬相等
        for c in range(COLUMNS):
            grid_frame.columnconfigure(c, weight=1)

    def _display_detailed_list(self, items):
        """以詳細列表形式顯示可視化項目"""
        if not items:
            ttk.Label(self.vis_frame, text="沒有可用的可視化項目", font=("Arial", 12)).pack(padx=20, pady=20)
            return
        
        for i, item in enumerate(items):
            # 創建項目卡片
            card = ttk.Frame(self.vis_frame, borderwidth=1, relief="solid")
            card.pack(padx=10, pady=10, fill="x")
            
            try:
                # 標題行
                title_frame = ttk.Frame(card)
                title_frame.pack(fill="x", padx=5, pady=5)
                
                filename = item.get("filename", f"項目 {i+1}")
                vis_type = item.get("vis_type", "未知類型")
                
                ttk.Label(title_frame, text=filename, font=("Arial", 10, "bold")).pack(side="left")
                ttk.Label(title_frame, text=f"[{vis_type}]", foreground="gray").pack(side="right")
                
                # 內容區 - 左側信息，右側縮略圖（如果有）
                content_frame = ttk.Frame(card)
                content_frame.pack(fill="x", padx=5, pady=5)
                
                # 左側信息
                info_frame = ttk.Frame(content_frame)
                info_frame.pack(side="left", fill="both", expand=True)
                
                path = item.get("path", "")
                ttk.Label(info_frame, text=f"路徑: {path}", wraplength=400).pack(anchor="w", pady=2)
                
                step = item.get("step_display", "")
                if step:
                    ttk.Label(info_frame, text=f"處理步驟: {step}").pack(anchor="w", pady=2)
                
                # 右側縮略圖（如果是圖片）
                if path and os.path.exists(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    try:
                        # 讀取圖片並創建縮略圖
                        original_img = Image.open(path)
                        
                        # 縮略圖尺寸
                        thumb_size = (120, 90)
                        thumb_img = original_img.copy()
                        thumb_img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                        
                        # 轉換為Tkinter可用的格式
                        photo = ImageTk.PhotoImage(thumb_img)
                        
                        # 創建圖像框架
                        img_frame = ttk.Frame(content_frame)
                        img_frame.pack(side="right", padx=10)
                        
                        # 顯示縮略圖
                        img_label = ttk.Label(img_frame, image=photo)
                        img_label.image = photo  # 保持引用
                        img_label.pack()
                        
                        # 添加點擊事件
                        img_label.bind("<Button-1>", lambda e, path=path: self._open_file(path))
                        
                        # 關閉圖片以釋放資源
                        original_img.close()
                        thumb_img.close()
                    except Exception as e:
                        self.logger.error(f"載入圖片時出錯: {str(e)}")
                
                # 操作按鈕
                btn_frame = ttk.Frame(card)
                btn_frame.pack(fill="x", padx=5, pady=5)
                
                ttk.Button(
                    btn_frame, text="查看圖片", 
                    command=lambda path=path: self._open_file(path)
                ).pack(side="left", padx=5)
                
            except Exception as e:
                self.logger.error(f"處理可視化項目時出錯: {str(e)}")
                ttk.Label(card, text=f"錯誤: {str(e)}", foreground="red").pack(pady=5)
    
    def _show_visualization_details(self, item):
        """顯示可視化項目的詳細信息"""
        # 創建一個頂層窗口
        details_window = tk.Toplevel(self.root)
        details_window.title("可視化詳細信息")
        details_window.geometry("500x400")
        details_window.transient(self.root)  # 設為主窗口的子窗口
        
        # 設置窗口樣式
        style = ttk.Style(details_window)
        style.configure('Details.TLabel', font=('Arial', 10))
        style.configure('DetailsTitle.TLabel', font=('Arial', 12, 'bold'))
        
        # 主框架
        main_frame = ttk.Frame(details_window, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # 標題
        title = item.get("filename", "可視化項目")
        ttk.Label(main_frame, text=title, style='DetailsTitle.TLabel').pack(pady=(0, 10))
        
        # 創建內容框架
        content = ttk.Frame(main_frame)
        content.pack(fill="both", expand=True)
        
        # 顯示所有項目詳情
        row = 0
        for key, value in item.items():
            if key not in ['image', 'photo', 'widget']:  # 排除非文本屬性
                ttk.Label(content, text=f"{key}:", style='Details.TLabel', width=15, anchor="e").grid(row=row, column=0, sticky="e", padx=5, pady=2)
                
                # 處理不同類型的值
                if isinstance(value, str) and len(value) > 50:
                    # 長文本使用文本框顯示
                    text_widget = tk.Text(content, height=3, width=40, wrap="word")
                    text_widget.grid(row=row, column=1, sticky="w", padx=5, pady=2)
                    text_widget.insert("1.0", value)
                    text_widget.config(state="disabled")  # 設為只讀
                else:
                    # 普通值使用標籤顯示
                    ttk.Label(content, text=str(value), style='Details.TLabel', wraplength=350).grid(row=row, column=1, sticky="w", padx=5, pady=2)
                
                row += 1
        
        # 如果是圖片，添加在下方顯示縮略圖
        path = item.get("path", "")
        if path and os.path.exists(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            try:
                # 打開圖片
                img = Image.open(path)
                
                # 調整大小
                max_size = (400, 300)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 轉換為Tkinter可用的格式
                photo = ImageTk.PhotoImage(img)
                
                # 創建標籤顯示圖片
                img_frame = ttk.Frame(main_frame)
                img_frame.pack(pady=10)
                
                img_label = ttk.Label(img_frame, image=photo)
                img_label.image = photo  # 保持引用
                img_label.pack()
                
                # 添加點擊事件
                img_label.bind("<Button-1>", lambda e, path=path: self._open_file(path))
                
            except Exception as e:
                ttk.Label(main_frame, text=f"無法載入圖片: {str(e)}", foreground="red").pack(pady=5)
        
        # 底部按鈕
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Button(btn_frame, text="打開文件", command=lambda: self._open_file(item.get("path", ""))).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="關閉", command=details_window.destroy).pack(side="right", padx=5)
        
        # 確保窗口置中
        details_window.update_idletasks()
        width = details_window.winfo_width()
        height = details_window.winfo_height()
        x = (details_window.winfo_screenwidth() // 2) - (width // 2)
        y = (details_window.winfo_screenheight() // 2) - (height // 2)
        details_window.geometry(f"{width}x{height}+{x}+{y}")
        
        # 設為模態窗口
        details_window.grab_set()
        details_window.focus_set()
    
    def _setup_status_bar(self):
        """設置底部狀態欄 - 僅使用文本提醒"""
        status_frame = ttk.Frame(self.root, padding=10)
        status_frame.pack(side="bottom", fill="x")
        
        # 使用更明顯的字體和顏色來顯示狀態
        self.status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            font=('Arial', 11),
            foreground='#0066cc'  # 使用藍色顯示狀態文本
        )
        self.status_label.pack(anchor="w", pady=5)
        
        # 移除進度條，但保留進度變數以維持程式邏輯
        # self.progress_bar = ttk.Progressbar(...)
        # self.progress_bar.pack(...)
        
        # 添加一個階段指示器標籤
        self.stage_var = tk.StringVar(value="準備就緒")
        self.stage_label = ttk.Label(
            status_frame, 
            textvariable=self.stage_var,
            font=('Arial', 10, 'italic')
        )
        self.stage_label.pack(anchor="w")
    
    # ================================================
    # 功能方法：不使用多線程，採用事件驅動方式
    # ================================================
    def _select_file(self):
        """選擇數據文件"""
        file_path = filedialog.askopenfilename(
            title="選擇評論資料文件",
            filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)
            self.status_var.set(f"已選擇文件: {os.path.basename(file_path)}")
            self.logger.info(f"已選擇文件: {file_path}")
    
    def _import_data(self):
        """導入數據 - 啟動任務處理器"""
        file_path = self.file_path.get()
        if not file_path:
            messagebox.showerror("錯誤", "請先選擇文件!")
            return
            
        # 先創建數據集ID
        dataset_name = os.path.basename(file_path).split('.')[0]
        self.current_dataset_id = self.result_manager.register_dataset(dataset_name, file_path)
        
        # 開始導入任務
        self.task_processor.start_task(self._import_data_task, file_path)
        
    def _import_data_task(self, file_path):
        """數據導入任務 - 函數版本 (含控制台輸出)"""
        # 打開控制台窗口並獲取日誌文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("數據導入處理", auto_close=True)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        logger = ConsoleOutputManager.setup_console_logger('data_import', log_file)
        logger.info(f"開始數據導入，處理文件: {file_path}")
        
        try:
            # 初始化導入器
            importer = DataImporter(output_dir=str(self.processed_data_dir))
            
            # 定義進度回調
            progress_updates = []
            def progress_callback(message, percentage):
                progress_updates.append((message, percentage))
                self.status_var.set(message)
                
                # 根據百分比更新階段指示文本
                if percentage < 0:
                    self.stage_var.set("處理錯誤")
                elif percentage < 20:
                    self.stage_var.set("初始化階段")
                elif percentage < 40:
                    self.stage_var.set("資料載入中")
                elif percentage < 60:
                    self.stage_var.set("處理進行中")
                elif percentage < 80:
                    self.stage_var.set("分析階段")
                elif percentage < 100:
                    self.stage_var.set("最終處理中")
                else:
                    self.stage_var.set("處理完成")
                
                # 仍然更新進度變數以維持程式邏輯
                if percentage >= 0:
                    self.progress_var.set(percentage)
                
                # 記錄到控制台
                if percentage >= 0:
                    logger.info(f"{message} ({percentage}%)")
                else:
                    logger.error(message)
                
                return message, percentage
            
            # 執行導入操作
            processed_file_path = importer.import_data(file_path, callback=progress_callback)
            
            # 保存結果路徑
            self.processed_data_path = processed_file_path
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"數據導入完成!")
            logger.info(f"處理後的數據保存為: {processed_file_path}")
            logger.info("====================================")
            logger.info(f"處理已完成。窗口將在幾秒後自動關閉。")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 返回結果字典
            return {
                "processed_file_path": processed_file_path,
                "step": "data_import",
                "progress_updates": progress_updates
            }
        except Exception as e:
            # 記錄錯誤
            logger.error(f"數據導入失敗: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 標記處理完成
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 重新拋出異常
            raise
    
    def _extract_bert_embeddings(self):
        """執行BERT語義提取"""
        if not self.processed_data_path:
            messagebox.showerror("錯誤", "請先導入並處理數據!")
            return
            
        # 開始BERT提取任務
        self.task_processor.start_task(self._extract_bert_task, self.processed_data_path)
    
    def _extract_bert_task(self, data_path):
        """BERT語義提取任務 - 函數版本 (含控制台輸出)"""
        # 打開控制台窗口並獲取日誌文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("BERT語義提取", auto_close=True)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        logger = ConsoleOutputManager.setup_console_logger('bert_extraction', log_file)
        logger.info(f"開始BERT語義提取，處理文件: {data_path}")
        
        try:
            # 初始化BERT編碼器 - 自動檢測裝置
            embedder = BertEmbedder(
                model_name='bert-base-uncased',
                output_dir=str(self.bert_embeddings_dir),
                logger=logger,
                force_cpu=False  # 不強制使用CPU，自動檢測裝置
            )
            
            # 定義進度回調
            progress_updates = []
            def progress_callback(message, percentage):
                progress_updates.append((message, percentage))
                self.status_var.set(message)
                if percentage >= 0:
                    self.progress_var.set(percentage)
                # 記錄到控制台
                if percentage >= 0:
                    logger.info(f"{message} ({percentage}%)")
                else:
                    logger.error(message)
                return message, percentage
            
            # 如果檔案不存在則自動處理
            if not os.path.exists(data_path):
                logger.warning(f"找不到指定檔案: {data_path}")
                
                # 嘗試查找正確的檔案名
                data_dir = os.path.dirname(data_path)
                expected_file_prefix = os.path.basename(data_path).split('_')[0:2]
                expected_file_prefix = '_'.join(expected_file_prefix)
                
                if data_dir and os.path.exists(data_dir):
                    logger.info(f"嘗試在 {data_dir} 查找匹配 {expected_file_prefix}* 的檔案")
                    matching_files = [f for f in os.listdir(data_dir) if f.startswith(expected_file_prefix) and f.endswith('.csv')]
                    
                    if matching_files:
                        # 使用最新的匹配檔案
                        newest_file = max(matching_files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)))
                        correct_path = os.path.join(data_dir, newest_file)
                        logger.info(f"找到替代檔案: {correct_path}")
                        data_path = correct_path
                        
                        # 更新訊息
                        self.status_var.set(f"使用找到的檔案: {os.path.basename(data_path)}")
                        logger.info(f"將使用檔案: {data_path}")
                    else:
                        logger.error(f"在 {data_dir} 中找不到匹配的檔案")
            
            # 執行BERT嵌入提取
            result = embedder.extract_embeddings(
                data_path,
                text_column="clean_text",
                batch_size=16,
                callback=progress_callback
            )
            
            # 保存結果到實例變量
            self.bert_embeddings_path = result['embeddings_path']
            self.bert_metadata_path = result['metadata_path']
            self.embedding_dim = result['embedding_dim']
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"BERT語義提取完成!")
            logger.info(f"嵌入向量保存至: {self.bert_embeddings_path}")
            logger.info(f"元數據保存至: {self.bert_metadata_path}")
            logger.info(f"嵌入維度: {self.embedding_dim}")
            logger.info("====================================")
            logger.info(f"處理已完成。窗口將在幾秒後自動關閉。")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 返回結果字典
            return {
                "embeddings_path": result['embeddings_path'],
                "metadata_path": result['metadata_path'],
                "embedding_dim": result['embedding_dim'],
                "step": "bert_embedding",
                "progress_updates": progress_updates
            }
        except Exception as e:
            # 記錄錯誤
            logger.error(f"BERT語義提取失敗: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 標記處理完成
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 重新拋出異常
            raise
    
    def _perform_lda(self):
        """執行LDA面向切割"""
        if not self.bert_metadata_path:
            messagebox.showerror("錯誤", "請先執行BERT語義提取!")
            return
            
        # 根據自動偵測選項決定資料來源和主題數量
        use_custom_topic_count = False
        
        if self.auto_detect_var.get():
            # 自動確定數據來源
            self._determine_data_source()
            # 使用預設主題數量（由資料來源決定）
            topic_count = None
        else:
            # 使用使用者設定的資料來源和主題數量
            self.data_source = self.data_source_var.get()
            try:
                topic_count = int(self.topic_count_var.get())
                if topic_count <= 0:
                    raise ValueError("主題數量必須大於0")
                use_custom_topic_count = True
            except ValueError as e:
                messagebox.showerror("錯誤", f"主題數量設定無效: {str(e)}")
                return
        
        # 開始LDA任務
        self.task_processor.start_task(
            self._perform_lda_task, 
            self.bert_metadata_path, 
            self.data_source,
            topic_count if use_custom_topic_count else None
        )
    
    def _determine_data_source(self):
        """確定數據來源"""
        file_path = self.file_path.get() or ""
        file_name = os.path.basename(file_path).lower()
        
        # 預設為未知數據源
        self.data_source = "unknown"
        
        # 檢查文件名中的關鍵詞
        if "imdb" in file_name or "movie" in file_name or "film" in file_name:
            self.data_source = "imdb"
        elif "amazon" in file_name or "product" in file_name:
            self.data_source = "amazon"
        elif "yelp" in file_name or "restaurant" in file_name:
            self.data_source = "yelp"
        else:
            # 通過對話框讓用戶選擇
            self._ask_data_source()
    
    def _ask_data_source(self):
        """詢問用戶數據來源"""
        sources = {
            "1": "IMDB",
            "2": "Amazon", 
            "3": "Yelp"
        }
        
        source = simpledialog.askstring(
            "選擇數據來源",
            "請選擇評論數據來源:\n1. IMDB電影評論\n2. Amazon產品評論\n3. Yelp餐廳評論",
            initialvalue="1"
        )
        
        if source in sources:
            self.data_source = sources[source]
        else:
            self.data_source = "unknown"
    
    def _perform_lda_task(self, metadata_path, data_source, custom_topic_count=None):
        """LDA面向切割任務 - 函數版本 (含控制台輸出)"""
        # 打開控制台窗口並獲取日誌文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("LDA面向切割", auto_close=True)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        logger = ConsoleOutputManager.setup_console_logger('lda_topic', log_file)
        logger.info(f"開始LDA面向切割，處理文件: {metadata_path}")
        logger.info(f"數據來源: {data_source}")
        
        try:
            # 初始化LDA提取器
            extractor = LDATopicExtractor(output_dir=str(self.lda_topics_dir), logger=logger)
            
            # 獲取主題標籤
            from src.settings.topic_labels import TOPIC_LABELS_zh
            topic_labels = None
            
            if data_source in TOPIC_LABELS_zh:
                topic_labels = TOPIC_LABELS_zh[data_source]
                # 確定主題數量
                if custom_topic_count is not None:
                    # 使用自定義主題數量
                    topic_count = custom_topic_count
                    logger.info(f"使用自定義主題數量: {topic_count}")
                else:
                    # 使用預設主題數量
                    topic_count = len(topic_labels)
                    logger.info(f"使用 {data_source} 的預設主題數量: {topic_count}")
                
                logger.info(f"使用 {data_source} 的自定義主題標籤")
            else:
                # 未知資料來源，使用自定義或預設主題數量
                topic_count = custom_topic_count if custom_topic_count is not None else 10
                logger.info(f"未找到匹配的主題標籤，使用一般主題")
                logger.info(f"主題數量設為: {topic_count}")
            
            # 定義進度回調
            progress_updates = []
            def progress_callback(message, percentage):
                progress_updates.append((message, percentage))
                self.status_var.set(message)
                if percentage >= 0:
                    self.progress_var.set(percentage)
                # 記錄到控制台
                if percentage >= 0:
                    logger.info(f"{message} ({percentage}%)")
                else:
                    logger.error(message)
                return message, percentage
                    
            # 執行LDA主題建模
            results = extractor.run_lda(
                metadata_path,
                n_topics=topic_count,
                topic_labels=topic_labels,
                callback=progress_callback
            )
            
            # 保存結果到實例變量
            self.lda_model_path = results['lda_model_path']
            self.topics_path = results['topics_path']
            self.topic_metadata_path = results['topic_metadata_path']
            self.topic_count = topic_count
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"LDA面向切割完成!")
            logger.info(f"LDA模型保存至: {self.lda_model_path}")
            logger.info(f"主題詞保存至: {self.topics_path}")
            logger.info(f"帶主題標籤的元數據保存至: {self.topic_metadata_path}")
            logger.info("====================================")
            
            # 顯示每個主題的頂部詞語
            logger.info("各個主題的頂部詞語:")
            if 'topic_words' in results:
                for topic, words in results['topic_words'].items():
                    logger.info(f"{topic}: {', '.join(words[:10])}")
            
            logger.info("====================================")
            logger.info(f"處理已完成。窗口將在幾秒後自動關閉。")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 返回結果字典
            return {
                "lda_model_path": results['lda_model_path'],
                "topics_path": results['topics_path'],
                "topic_metadata_path": results['topic_metadata_path'],
                "n_topics": topic_count,
                "step": "lda_topic",
                "visualizations": results.get('visualizations', []),
                "progress_updates": progress_updates
            }
        except Exception as e:
            # 記錄錯誤
            logger.error(f"LDA面向切割失敗: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 標記處理完成
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 重新拋出異常
            raise
    
    def _calculate_aspect_vectors(self):
        """計算面向相關句子的平均向量"""
        if not self.bert_embeddings_path or not self.topic_metadata_path:
            messagebox.showerror("錯誤", "請先完成BERT語義提取和LDA面向切割步驟!")
            return
            
        # 開始向量計算任務
        self.task_processor.start_task(
            self._calculate_aspect_vectors_task, 
            self.bert_embeddings_path, 
            self.topic_metadata_path
        )
    
    def _calculate_aspect_vectors_task(self, embeddings_path, topic_metadata_path):
        """面向向量計算任務 - 函數版本 (含控制台輸出)"""
        # 打開控制台窗口並獲取日誌文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("面向向量計算", auto_close=True)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        logger = ConsoleOutputManager.setup_console_logger('aspect_vector', log_file)
        logger.info(f"開始面向向量計算")
        logger.info(f"嵌入文件: {embeddings_path}")
        logger.info(f"主題元數據: {topic_metadata_path}")
        
        try:
            # 初始化向量計算器
            calculator = AspectVectorCalculator(output_dir=str(self.aspect_vectors_dir), logger=logger)
            
            # 定義進度回調
            progress_updates = []
            def progress_callback(message, percentage):
                progress_updates.append((message, percentage))
                self.status_var.set(message)
                if percentage >= 0:
                    self.progress_var.set(percentage)
                # 記錄到控制台
                if percentage >= 0:
                    logger.info(f"{message} ({percentage}%)")
                else:
                    logger.error(message)
                return message, percentage
                
            # 計算面向向量
            results = calculator.calculate_aspect_vectors(
                embeddings_path,
                topic_metadata_path,
                callback=progress_callback
            )
            
            # 獲取結果路徑
            aspect_vectors_path = results['aspect_vectors_path']
            aspect_metadata_path = results['aspect_metadata_path']
            tsne_plot_path = results.get('tsne_plot_path')
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"面向向量計算完成!")
            logger.info(f"面向向量保存至: {aspect_vectors_path}")
            logger.info(f"元數據保存至: {aspect_metadata_path}")
            if tsne_plot_path:
                logger.info(f"t-SNE可視化保存至: {tsne_plot_path}")
            logger.info("====================================")
            
            # 記錄每個面向的文檔數量
            logger.info("各面向文檔數量:")
            if 'topic_doc_counts' in results:
                for topic, count in results['topic_doc_counts'].items():
                    logger.info(f"{topic}: {count}個文檔")
                    
            logger.info("====================================")
            logger.info(f"處理已完成。窗口將在幾秒後自動關閉。")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 返回結果字典
            return {
                "aspect_vectors_path": aspect_vectors_path,
                "aspect_metadata_path": aspect_metadata_path,
                "tsne_plot_path": tsne_plot_path,
                "topic_count": len(results.get('topics', [])),
                "step": "aspect_vector",
                "progress_updates": progress_updates
            }
        except Exception as e:
            # 記錄錯誤
            logger.error(f"面向向量計算失敗: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 標記處理完成
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 重新拋出異常
            raise
    
    def _export_vectors(self):
        """匯出計算好的平均向量"""
        if not hasattr(self, 'aspect_vectors_path') or not self.aspect_vectors_path:
            messagebox.showerror("錯誤", "請先計算面向向量!")
            return
            
        # 讓用戶選擇導出格式
        formats = ["csv", "json", "pickle"]
        format_choice = simpledialog.askstring(
            "選擇導出格式", 
            "請選擇導出格式:\n1. CSV檔案\n2. JSON檔案\n3. Pickle檔案",
            initialvalue="1"
        )
        
        if not format_choice:
            return
            
        try:
            format_idx = int(format_choice)
            if format_idx < 1 or format_idx > 3:
                raise ValueError()
            selected_format = formats[format_idx - 1]
        except (ValueError, IndexError):
            messagebox.showerror("錯誤", "無效的選擇，請輸入1、2或3")
            return
            
        # 讓用戶選擇保存路徑
        file_types = [("CSV檔案", "*.csv"), ("JSON檔案", "*.json"), ("Pickle檔案", "*.pkl"), ("所有檔案", "*.*")]
        default_extension = f".{selected_format}" if selected_format != "pickle" else ".pkl"
        
        file_path = filedialog.asksaveasfilename(
            title="保存面向向量",
            defaultextension=default_extension,
            filetypes=file_types
        )
        
        if not file_path:
            return
            
        # 開始導出任務
        self.task_processor.start_task(
            self._export_vectors_task, 
            self.aspect_vectors_path, 
            file_path,
            selected_format
        )
    
    def _export_vectors_task(self, aspect_vectors_path, output_path, output_format):
        """向量導出任務 - 函數版本 (含控制台輸出)"""
        # 打開控制台窗口並獲取日誌文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("向量導出", auto_close=True)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        logger = ConsoleOutputManager.setup_console_logger('vector_export', log_file)
        logger.info(f"開始導出面向向量")
        logger.info(f"面向向量文件: {aspect_vectors_path}")
        logger.info(f"輸出格式: {output_format}")
        logger.info(f"輸出路徑: {output_path}")
        
        try:
            # 初始化向量計算器
            calculator = AspectVectorCalculator(output_dir=os.path.dirname(output_path), logger=logger)
            
            # 定義進度回調
            progress_updates = []
            def progress_callback(message, percentage):
                progress_updates.append((message, percentage))
                self.status_var.set(message)
                if percentage >= 0:
                    self.progress_var.set(percentage)
                # 記錄到控制台
                if percentage >= 0:
                    logger.info(f"{message} ({percentage}%)")
                else:
                    logger.error(message)
                return message, percentage
                
            # 導出向量
            result_path = calculator.export_aspect_vectors(
                aspect_vectors_path,
                output_format=output_format,
                callback=progress_callback
            )
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"面向向量導出完成!")
            logger.info(f"導出結果保存至: {result_path}")
            logger.info("====================================")
            logger.info(f"處理已完成。窗口將在幾秒後自動關閉。")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 返回結果字典
            return {
                "result_path": result_path,
                "format": output_format,
                "step": "export",
                "progress_updates": progress_updates
            }
        except Exception as e:
            # 記錄錯誤
            logger.error(f"向量導出失敗: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 標記處理完成
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 重新拋出異常
            raise
    
    def _select_business_file(self):
        """選擇Yelp的business文件"""
        file_path = filedialog.askopenfilename(
            title="選擇Yelp的business.json文件",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.business_path.set(file_path)
            self.status_var.set(f"已選擇business文件: {os.path.basename(file_path)}")
            self.logger.info(f"已選擇business文件: {file_path}")

    def _select_review_file(self):
        """選擇Yelp的review文件"""
        file_path = filedialog.askopenfilename(
            title="選擇Yelp的review.json文件",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            self.review_path.set(file_path)
            self.status_var.set(f"已選擇review文件: {os.path.basename(file_path)}")
            self.logger.info(f"已選擇review文件: {file_path}")

    def _process_yelp_data(self):
        """處理Yelp數據（合併business和review）"""
        business_path = self.business_path.get()
        review_path = self.review_path.get()
        
        if not business_path:
            messagebox.showerror("錯誤", "請先選擇business文件!")
            return
            
        if not review_path:
            messagebox.showerror("錯誤", "請先選擇review文件!")
            return
        
        # 獲取抽樣數量
        try:
            sample_size = int(self.sample_size.get())
            if sample_size <= 0:
                raise ValueError("抽樣數量必須大於0")
        except ValueError as e:
            messagebox.showerror("錯誤", f"抽樣數量設置無效: {str(e)}")
            return
        
        # 打開控制台窗口並獲取日誌文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("Yelp數據處理", auto_close=True)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        logger = ConsoleOutputManager.setup_console_logger('yelp_processing', log_file)
        
        # 更新狀態
        self.status_var.set("正在處理Yelp數據...")
        self.progress_var.set(10)
        
        # 設置輸出文件路徑
        output_dir = str(self.processed_data_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, f"processed_yelp_{sample_size}.csv")
        
        # 在任務處理器中運行Yelp數據處理
        self.task_processor.start_task(
            self._yelp_processing_task,
            business_path,
            review_path,
            output_file,
            sample_size,
            logger,
            status_file
        )

    def _yelp_processing_task(self, business_path, review_path, output_path, sample_size, logger, status_file):
        """Yelp數據處理任務 - 函數版本"""
        try:
            import json
            import pandas as pd
            import random
            import os
            import time
            
            # 記錄開始時間
            start_time = time.time()
            
            # 記錄處理參數
            logger.info(f"=== Yelp數據處理開始 ===")
            logger.info(f"Business文件: {business_path}")
            logger.info(f"Review文件: {review_path}")
            logger.info(f"輸出文件: {output_path}")
            logger.info(f"抽樣數量: {sample_size}")
            logger.info(f"============================")
            
            # 設定隨機種子確保結果可重現
            random.seed(42)
            
            # 更新UI進度
            self.status_var.set("正在讀取餐廳信息...")
            self.progress_var.set(15)
            
            # 第一步：讀取商家數據，只保留餐廳類別
            logger.info("正在讀取餐廳信息...")
            restaurants = {}
            restaurant_count = 0
            
            with open(business_path, 'r', encoding='utf-8') as f:
                for line in f:
                    business = json.loads(line)
                    # 只保留餐廳類別的商家
                    if business.get('categories') and 'Restaurants' in business['categories']:
                        # 提取需要的欄位
                        restaurants[business['business_id']] = {
                            'name': business.get('name', ''),
                            'city': business.get('city', ''),
                            'state': business.get('state', ''),
                            'stars': business.get('stars', 0),
                            'categories': business.get('categories', '')
                        }
                        restaurant_count += 1
                    
                    # 每處理1000條記錄更新一次進度
                    if restaurant_count % 1000 == 0:
                        self.status_var.set(f"已讀取 {restaurant_count} 家餐廳...")
            
            logger.info(f"已識別 {restaurant_count} 家餐廳")
            self.status_var.set(f"已識別 {restaurant_count} 家餐廳")
            self.progress_var.set(30)
            
            # 更新UI進度
            self.status_var.set("正在處理評論數據...")
            self.progress_var.set(35)
            
            # 第二步：讀取評論數據並抽樣
            logger.info("開始處理評論數據...")
            sampled_reviews = []
            processed_count = 0
            eligible_count = 0
            
            with open(review_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        review = json.loads(line)
                        business_id = review['business_id']
                        
                        # 檢查評論是否屬於餐廳
                        if business_id in restaurants:
                            eligible_count += 1
                            
                            # 使用水塘抽樣算法確保均勻抽樣
                            if len(sampled_reviews) < sample_size:
                                # 直接添加，直到達到樣本大小
                                business = restaurants[business_id]
                                merged_review = {
                                    'business_id': business_id,
                                    'business_name': business['name'],
                                    'text': review.get('text', ''),
                                    'review_stars': review.get('stars', 0),
                                    'business_stars': business['stars'],
                                    'city': business['city'],
                                    'state': business['state'],
                                    'categories': business['categories'],
                                    'date': review.get('date', '')
                                }
                                sampled_reviews.append(merged_review)
                            else:
                                # 水塘抽樣：以 sample_size/已處理數量 的概率替換現有樣本
                                j = random.randint(0, eligible_count - 1)
                                if j < sample_size:
                                    business = restaurants[business_id]
                                    merged_review = {
                                        'business_id': business_id,
                                        'business_name': business['name'],
                                        'text': review.get('text', ''),
                                        'review_stars': review.get('stars', 0),
                                        'business_stars': business['stars'],
                                        'city': business['city'],
                                        'state': business['state'],
                                        'categories': business['categories'],
                                        'date': review.get('date', '')
                                    }
                                    sampled_reviews[j] = merged_review
                            
                        # 更新處理計數和進度
                        processed_count += 1
                        if processed_count % 10000 == 0:
                            progress = 35 + min(55, (eligible_count / sample_size) * 55)
                            self.status_var.set(f"已處理 {processed_count} 條評論，找到 {eligible_count} 條餐廳評論")
                            self.progress_var.set(progress)
                            logger.info(f"已處理 {processed_count} 條評論，找到 {eligible_count} 條餐廳評論")
                            
                            # 如果已經找到足夠的樣本且處理了至少sample_size*10的評論，提前結束
                            if eligible_count >= sample_size * 10 and len(sampled_reviews) == sample_size:
                                logger.info(f"已收集足夠樣本，提前結束處理")
                                break
                            
                    except Exception as e:
                        logger.warning(f"處理評論時出錯: {str(e)}")
                        continue
            
            # 確保我們不超過所需的樣本數
            if len(sampled_reviews) > sample_size:
                sampled_reviews = sampled_reviews[:sample_size]
                
            logger.info(f"共處理了 {processed_count} 條評論")
            logger.info(f"其中有 {eligible_count} 條餐廳評論")
            logger.info(f"成功抽樣 {len(sampled_reviews)} 條評論")
            
            # 更新UI進度
            self.status_var.set("正在保存處理結果...")
            self.progress_var.set(90)
            
            # 第三步：將抽樣的評論轉換為DataFrame
            df = pd.DataFrame(sampled_reviews)
            
            # 添加清洗後的文本列
            logger.info("正在進行文本清洗...")
            
            # 定義文本清洗函數（簡化版）
            def clean_text(text):
                if pd.isna(text) or not text:
                    return ""
                # 轉換為字符串
                text = str(text)
                # 移除多餘的空格
                text = ' '.join(text.split())
                return text
            
            # 應用文本清洗
            df['clean_text'] = df['text'].apply(clean_text)
            
            # 保存為CSV
            logger.info(f"正在保存處理結果到 {output_path}")
            df.to_csv(output_path, index=False)
            
            # 計算處理時間
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            
            # 記錄處理完成信息
            logger.info(f"============================")
            logger.info(f"處理完成! 用時: {int(minutes)}分{int(seconds)}秒")
            logger.info(f"總共處理了 {restaurant_count} 家餐廳")
            logger.info(f"從 {eligible_count} 條餐廳評論中抽樣了 {len(sampled_reviews)} 條")
            logger.info(f"結果已保存至: {output_path}")
            logger.info(f"============================")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 創建數據集ID
            dataset_name = f"Yelp_Restaurants_{len(sampled_reviews)}"
            self.current_dataset_id = self.result_manager.register_dataset(dataset_name, output_path)
            
            # 註冊處理結果
            self.result_manager.register_result(
                self.current_dataset_id,
                "data_import",
                "data",
                output_path,
                metadata={
                    "source_files": [business_path, review_path],
                    "restaurants": restaurant_count,
                    "sample_size": len(sampled_reviews)
                }
            )
            
            # 保存處理後的文件路徑以供後續步驟使用
            self.processed_data_path = output_path
            
            # 返回結果
            return {
                "processed_file_path": output_path,
                "step": "data_import",
                "sample_size": len(sampled_reviews),
                "restaurant_count": restaurant_count
            }
            
        except Exception as e:
            # 記錄錯誤
            logger.error(f"Yelp數據處理失敗: {str(e)}")
            logger.error(traceback.format_exc())
            
            # 標記處理完成
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 重新拋出異常
            raise

    def _refresh_results(self):
        """刷新結果顯示"""
        self._refresh_datasets()
        if self.current_dataset_id:
            self._load_dataset_results(self.current_dataset_id)
        self.status_var.set("結果已刷新")
    
    def _refresh_datasets(self):
        """刷新數據集列表"""
        try:
            # 獲取所有數據集
            datasets = self.result_manager.index.get("datasets", {})
            
            # 構建顯示項
            dataset_items = []
            for dataset_id, dataset_info in datasets.items():
                status = dataset_info["summary"]["status"].upper()
                name = dataset_info["name"]
                dataset_items.append(f"{name} ({dataset_id}) - {status}")
            
            # 更新下拉選單
            self.dataset_combo["values"] = dataset_items
            
            # 如果有當前選中的數據集，設置為選中狀態
            if self.current_dataset_id:
                for i, item in enumerate(dataset_items):
                    if self.current_dataset_id in item:
                        self.dataset_combo.current(i)
                        break
            elif dataset_items:
                # 默認選擇第一個
                self.dataset_combo.current(0)
                # 載入選中的數據集
                self._on_dataset_selected(None)
        
        except Exception as e:
            self.logger.error(f"刷新數據集列表時出錯: {str(e)}")
    
    def _on_dataset_selected(self, event):
        """處理數據集選擇事件"""
        try:
            # 獲取選中的數據集ID
            selected = self.dataset_combo.get()
            if not selected:
                return
            
            # 從選中項中提取數據集ID
            import re
            match = re.search(r'\((.*?)\)', selected)
            if not match:
                return
            
            dataset_id = match.group(1)
            
            # 設置當前數據集ID
            self.current_dataset_id = dataset_id
            
            # 加載該數據集的處理結果
            self._load_dataset_results(dataset_id)
        
        except Exception as e:
            self.logger.error(f"選擇數據集時出錯: {str(e)}")
    
    def _load_dataset_results(self, dataset_id):
        """加載數據集的處理結果 - 添加可視化處理"""
        try:
            # 獲取數據集信息
            dataset_info = self.result_manager.get_dataset_summary(dataset_id)
            if not dataset_info:
                return
            
            # 清空結果顯示
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # 清空可視化項目列表
            self.vis_items = []
            
            # 加載處理結果
            for step_name, step_info in dataset_info["steps"].items():
                step_display = {
                    "data_import": "數據導入",
                    "bert_embedding": "BERT語義提取",
                    "lda_topic": "LDA面向切割",
                    "aspect_vector": "面向向量計算",
                    "export": "結果導出"
                }.get(step_name, step_name)
                
                for result in step_info["results"]:
                    result_type = result["type"]
                    
                    if result_type == "data":
                        # 添加到處理資料頁面
                        metadata_str = json.dumps(result.get("metadata", {}), ensure_ascii=False)
                        self.data_tree.insert(
                            "", "end", 
                            values=(result["filename"], step_display, result["path"], metadata_str)
                        )
                    
                    elif result_type == "visualization":
                        # 收集可視化項目
                        vis_item = {
                            "filename": result["filename"],
                            "path": result["path"],
                            "step": step_name,
                            "step_display": step_display,
                            "completed_at": step_info.get("completed_at", ""),
                            "metadata": result.get("metadata", {})
                        }
                        
                        # 確定可視化類型
                        if "topic" in result["filename"].lower():
                            vis_item["vis_type"] = "主題可視化"
                        elif "vector" in result["filename"].lower() or "tsne" in result["filename"].lower():
                            vis_item["vis_type"] = "向量可視化"
                        else:
                            vis_item["vis_type"] = "其他可視化"
                        
                        self.vis_items.append(vis_item)
                
            # 刷新可視化顯示
            self._refresh_vis_display()
            
            # 更新狀態
            self.status_var.set(f"已載入數據集 '{dataset_info['name']}' 的處理結果")
        
        except Exception as e:
            self.logger.error(f"加載數據集結果時出錯: {str(e)}")
            traceback.print_exc()
    
    def _add_visualization(self, result, step_display):
        """添加可視化結果到UI - 不使用真實圖片，改為顯示圖片描述和打開按鈕"""
        try:
            # 確定可視化類型
            vis_type = "未知"
            if "topic" in result["filename"].lower():
                vis_type = "主題可視化"
            elif "vector" in result["filename"].lower() or "tsne" in result["filename"].lower():
                vis_type = "向量可視化"
            
            # 創建可視化卡片
            card = ttk.Frame(self.vis_frame, borderwidth=1, relief="solid")
            card.pack(padx=10, pady=10, fill="x")
            
            # 標題
            title_frame = ttk.Frame(card)
            title_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Label(title_frame, text=result["filename"], font=("Arial", 10, "bold")).pack(side="left")
            ttk.Label(title_frame, text=f"[{vis_type}]", foreground="gray").pack(side="right")
            
            # 簡單描述而非真實圖片
            desc_text = f"圖片類型: {vis_type}\n路徑: {result['path']}\n處理步驟: {step_display}"
            ttk.Label(card, text=desc_text, wraplength=400).pack(padx=5, pady=5)
            
            # 按鈕
            btn_frame = ttk.Frame(card)
            btn_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Button(
                btn_frame, text="查看圖片", 
                command=lambda path=result["path"]: self._open_file(path)
            ).pack(side="left", padx=5)
            
            # 設置可視化類型作為標籤（用於過濾）
            card.vis_type = vis_type
        
        except Exception as e:
            self.logger.error(f"添加可視化結果時出錯: {str(e)}")
    
    def _on_vis_type_changed(self, event):
        """處理可視化類型選擇變更"""
        try:
            selected_type = self.vis_type_combo.get()
            
            # 顯示/隱藏不同類型的可視化
            for card in self.vis_frame.winfo_children():
                if selected_type == "全部" or getattr(card, "vis_type", "") == selected_type:
                    card.pack(padx=10, pady=10, fill="x")
                else:
                    card.pack_forget()
        
        except Exception as e:
            self.logger.error(f"切換可視化類型時出錯: {str(e)}")
    
    def _on_data_double_click(self, event):
        """處理數據結果雙擊事件"""
        # 獲取選中的項目
        selection = self.data_tree.selection()
        if not selection:
            return
            
        item = selection[0]
        file_path = self.data_tree.item(item, "values")[2]  # 路徑在第三列
        
        # 打開文件
        self._open_file(file_path)
    
    def _generate_overview_report(self):
        """生成概覽報告"""
        try:
            report_path = self.result_manager.create_overview_report()
            if report_path:
                response = messagebox.askyesno("成功", f"概覽報告已生成: {os.path.basename(report_path)}\n是否現在查看?")
                if response:
                    self._open_file(report_path)
            else:
                messagebox.showerror("錯誤", "生成概覽報告失敗")
        except Exception as e:
            self.logger.error(f"生成概覽報告時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"生成概覽報告時出錯: {str(e)}")
    
    def _view_dataset_report(self):
        """查看數據集處理報告"""
        if not self.current_dataset_id:
            messagebox.showinfo("提示", "請先選擇一個數據集")
            return
        
        try:
            report_path = self.result_manager.generate_summary_report(
                self.current_dataset_id,
                output_format="html"
            )
            
            if report_path:
                self._open_file(report_path)
            else:
                messagebox.showerror("錯誤", "生成數據集報告失敗")
        except Exception as e:
            self.logger.error(f"查看數據集報告時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"查看數據集報告時出錯: {str(e)}")
    
    def _open_file(self, file_path):
        """打開文件（使用系統默認應用程序）"""
        try:
            file_path = str(file_path)  # 確保Path對象轉為字符串
            if sys.platform == 'win32':
                os.startfile(file_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', file_path])
            else:  # Linux
                subprocess.Popen(['xdg-open', file_path])
        except Exception as e:
            self.logger.error(f"打開文件時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"無法打開文件: {str(e)}")
    
    def _load_recent_results(self):
        """從結果管理器加載最近的分析結果"""
        try:
            # 獲取最新的處理結果
            latest_results = self.result_manager.get_latest_results(limit=1)
            
            if latest_results:
                latest = latest_results[0]
                dataset_id = latest['dataset_id']
                dataset_info = self.result_manager.get_dataset_summary(dataset_id)
                
                if dataset_info:
                    self.current_dataset_id = dataset_id
                    
                    # 更新狀態
                    self.status_var.set(f"已加載最近的分析結果: {dataset_info['name']}")
                    self.logger.info(f"最近的分析結果已加載: {dataset_id} ({dataset_info['name']})")
                    
                    # 更新文件路徑顯示
                    if dataset_info.get('source_path'):
                        self.file_path.set(dataset_info['source_path'])
                    
                    # 刷新數據集列表
                    self._refresh_datasets()
        
        except Exception as e:
            self.logger.error(f"加載最近的分析結果時出錯: {str(e)}")
    
    def _on_task_complete(self, result, error, execution_time):
        """處理任務完成的回調"""
        if error:
            self.logger.error(f"任務執行出錯: {str(error)}")
            messagebox.showerror("錯誤", f"任務執行出錯: {str(error)}")
            return
            
        if not result or not isinstance(result, dict):
            self.logger.warning(f"任務完成但沒有有效結果: {type(result).__name__}")
            return
            
        try:
            # 根據任務類型處理結果
            step = result.get("step", "")
            
            if step == "data_import":
                # 處理數據導入結果
                processed_file_path = result["processed_file_path"]
                self.processed_data_path = processed_file_path
                
                # 註冊處理結果
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "data_import",
                    "data", 
                    processed_file_path,
                    metadata={
                        "original_file": self.file_path.get()
                    }
                )
                
                messagebox.showinfo("成功", "數據已成功導入和初步處理!")
                
            elif step == "bert_embedding":
                # 處理BERT提取結果
                self.bert_embeddings_path = result["embeddings_path"]
                self.bert_metadata_path = result["metadata_path"]
                self.embedding_dim = result["embedding_dim"]
                
                # 註冊處理結果
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "bert_embedding",
                    "data",
                    result["embeddings_path"],
                    metadata={
                        "embedding_dim": result["embedding_dim"],
                        "source_file": self.processed_data_path
                    }
                )
                
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "bert_embedding",
                    "data",
                    result["metadata_path"],
                    metadata={
                        "contains": "text and metadata"
                    }
                )
                
                messagebox.showinfo(
                    "成功", 
                    f"BERT語義提取完成！\n嵌入向量已保存\n嵌入維度: {result['embedding_dim']}"
                )
                
            elif step == "lda_topic":
                # 處理LDA結果
                self.lda_model_path = result["lda_model_path"]
                self.topics_path = result["topics_path"]
                self.topic_metadata_path = result["topic_metadata_path"]
                self.topic_count = result["n_topics"]
                
                # 註冊處理結果 - 模型
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "lda_topic",
                    "model",
                    result["lda_model_path"],
                    metadata={
                        "n_topics": result["n_topics"],
                        "model_type": "LDA",
                        "data_source": self.data_source
                    }
                )
                
                # 註冊處理結果 - 主題詞
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "lda_topic",
                    "data",
                    result["topics_path"],
                    metadata={
                        "n_topics": result["n_topics"],
                        "content": "topic keywords",
                        "data_source": self.data_source
                    }
                )
                
                # 註冊處理結果 - 帶主題標籤的元數據
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "lda_topic",
                    "data",
                    result["topic_metadata_path"],
                    metadata={
                        "n_topics": result["n_topics"],
                        "content": "documents with topic labels",
                        "data_source": self.data_source
                    }
                )
                
                # 註冊處理結果 - 可視化
                for vis_path in result.get("visualizations", []):
                    vis_type = "topic_words" if "topic_words" in vis_path else "doc_topics"
                    self.result_manager.register_result(
                        self.current_dataset_id,
                        "lda_topic",
                        "visualization",
                        vis_path,
                        metadata={
                            "visualization_type": vis_type,
                            "data_source": self.data_source
                        }
                    )
                
                messagebox.showinfo(
                    "成功", 
                    f"LDA面向切割完成！\n已識別出 {result['n_topics']} 個主題"
                )
                
            elif step == "aspect_vector":
                # 處理面向向量結果
                aspect_vectors_path = result["aspect_vectors_path"]
                aspect_metadata_path = result["aspect_metadata_path"]
                tsne_plot_path = result.get("tsne_plot_path")
                
                # 註冊處理結果 - 面向向量
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "aspect_vector",
                    "data",
                    aspect_vectors_path,
                    metadata={
                        "n_aspects": result["topic_count"],
                        "embedding_dim": self.embedding_dim
                    }
                )
                
                # 註冊處理結果 - 面向元數據
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "aspect_vector",
                    "data",
                    aspect_metadata_path,
                    metadata={
                        "content": "aspect metadata"
                    }
                )
                
                # 註冊處理結果 - 可視化
                if tsne_plot_path:
                    self.result_manager.register_result(
                        self.current_dataset_id,
                        "aspect_vector",
                        "visualization",
                        tsne_plot_path,
                        metadata={
                            "visualization_type": "t-SNE"
                        }
                    )
                
                # 標記數據集處理完成
                self.result_manager.complete_dataset(
                    self.current_dataset_id,
                    status="completed",
                    message="所有處理步驟已完成"
                )
                
                # 生成處理報告
                report_path = self.result_manager.generate_summary_report(
                    self.current_dataset_id,
                    output_format="html"
                )
                
                success_msg = f"面向向量計算完成！\n已計算 {result['topic_count']} 個面向的向量"
                if report_path:
                    view_report = messagebox.askyesno(
                        "成功", 
                        f"{success_msg}\n\n完整處理報告已生成，是否現在查看?"
                    )
                    if view_report:
                        self._open_file(report_path)
                else:
                    messagebox.showinfo("成功", success_msg)
                    
            elif step == "export":
                # 處理導出結果
                messagebox.showinfo(
                    "成功", 
                    f"面向向量已成功匯出！\n格式: {result['format']}\n保存路徑: {result['result_path']}"
                )
                
            # 刷新結果顯示
            self._refresh_results()
            
        except Exception as e:
            self.logger.error(f"處理任務結果時出錯: {str(e)}")
            traceback.print_exc()
            messagebox.showerror("錯誤", f"處理任務結果時出錯: {str(e)}")
    
    def _on_closing(self):
        """處理窗口關閉事件"""
        # 顯示正在清理的信息
        self.status_var.set("正在清理資源，請稍候...")
        self.root.update_idletasks()
        
        # 關閉所有matplotlib圖表
        plt.close('all')
        
        # 清除各種參考
        self.data_tree = None
        self.vis_frame = None
        
        # 刪除tkinter變量
        self.file_path = None
        self.topic_count = None
        self.progress_var = None
        self.status_var = None
        
        # 確保結果管理器保存所有變更
        if hasattr(self, 'result_manager'):
            try:
                # 保存結果索引
                if hasattr(self.result_manager, '_save_index'):
                    self.result_manager._save_index()
            except:
                pass
        
        # 最後銷毀窗口
        self.root.destroy()

# 主程序
if __name__ == "__main__":
    # 設置更好的錯誤處理
    def show_error(exctype, value, tb):
        error_msg = ''.join(traceback.format_exception(exctype, value, tb))
        messagebox.showerror('程式錯誤', f'發生未處理的錯誤:\n{error_msg}')
        
    # 設置未處理異常處理器
    sys.excepthook = show_error
    
    # 創建主窗口
    root = tk.Tk()
    app = CrossDomainSentimentAnalysisApp(root)
    root.mainloop()