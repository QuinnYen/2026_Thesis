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
        
        # 創建三個分頁
        self.create_data_processing_tab()
        self.create_attention_testing_tab()
        self.create_comparison_analysis_tab()
        
        # 添加當前run目錄標籤
        self.create_run_dir_label()
        
        # 初始化按鈕狀態
        self.update_button_states()
        
        # 最後將視窗置中於螢幕（在所有UI元素創建完成後）
        self.root.after(100, self.center_window)
    
    def center_window(self):
        """將視窗置中於螢幕"""
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
    
    def create_data_processing_tab(self):
        """第一分頁：資料處理"""
        frame1 = ttk.Frame(self.notebook)
        self.notebook.add(frame1, text=" 資料處理 ")
        
        # 主要容器
        main_frame = ttk.Frame(frame1)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 標題
        title_label = ttk.Label(main_frame, text="資料處理流程", font=FONTS['title'])
        title_label.pack(pady=(0, 20))
        
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
        
        # 步驟4：Bert編碼 → 開始編碼
        step4_frame = ttk.LabelFrame(main_frame, text="④ Bert編碼 → 開始編碼", padding=15)
        step4_frame.pack(fill='x', pady=(0, 15))
        
        encoding_frame = ttk.Frame(step4_frame)
        encoding_frame.pack(fill='x')
        
        self.encoding_btn = ttk.Button(encoding_frame, text="開始編碼", command=self.start_encoding)
        self.encoding_btn.pack(side='left')
        
        # 新增導入按鈕
        self.import_encoding_btn = ttk.Button(encoding_frame, text="導入編碼", command=self.import_encoding)
        self.import_encoding_btn.pack(side='left', padx=(10, 0))
        
        self.encoding_status = ttk.Label(step4_frame, text="狀態: 待處理", foreground="orange")
        self.encoding_status.pack(anchor='w', pady=(10, 0))
        

        
    def create_attention_testing_tab(self):
        """第二分頁：注意力機制測試"""
        frame2 = ttk.Frame(self.notebook)
        self.notebook.add(frame2, text=" 注意力機制測試 ")
        
        main_frame = ttk.Frame(frame2)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 標題
        title_label = ttk.Label(main_frame, text="注意力機制測試", font=FONTS['title'])
        title_label.pack(pady=(0, 20))
        
        # 單一注意力實驗組
        single_frame = ttk.LabelFrame(main_frame, text="單一注意力實驗組", padding=15)
        single_frame.pack(fill='x', pady=(0, 15))
        
        single_content = ttk.Frame(single_frame)
        single_content.pack(fill='x')
        
        # 單一注意力選項
        ttk.Label(single_content, text="↳ 相似度注意力").pack(anchor='w')
        ttk.Label(single_content, text="↳ 自注意力").pack(anchor='w')
        ttk.Label(single_content, text="↳ 關鍵詞注意力").pack(anchor='w')
        
        single_btn_frame = ttk.Frame(single_frame)
        single_btn_frame.pack(fill='x', pady=(10, 0))
        
        self.single_btn = ttk.Button(single_btn_frame, text="執行單一注意力測試", command=self.run_single_attention)
        self.single_btn.pack(side='left')
        
        self.single_status = ttk.Label(single_btn_frame, text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
        self.single_status.pack(side='left', padx=(10, 0))
        
        # 雙重組合實驗組
        dual_frame = ttk.LabelFrame(main_frame, text="雙重組合實驗組", padding=15)
        dual_frame.pack(fill='x', pady=(0, 15))
        
        dual_content = ttk.Frame(dual_frame)
        dual_content.pack(fill='x')
        
        # 雙重組合選項
        ttk.Label(dual_content, text="↳ 相似度 + 自注意力").pack(anchor='w')
        ttk.Label(dual_content, text="↳ 相似度 + 關鍵詞").pack(anchor='w')
        ttk.Label(dual_content, text="↳ 自注意力 + 關鍵詞").pack(anchor='w')
        
        dual_btn_frame = ttk.Frame(dual_frame)
        dual_btn_frame.pack(fill='x', pady=(10, 0))
        
        self.dual_btn = ttk.Button(dual_btn_frame, text="執行雙重組合測試", command=self.run_dual_attention)
        self.dual_btn.pack(side='left')
        
        self.dual_status = ttk.Label(dual_btn_frame, text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
        self.dual_status.pack(side='left', padx=(10, 0))
        
        # 三重組合實驗組
        triple_frame = ttk.LabelFrame(main_frame, text="三重組合實驗組", padding=15)
        triple_frame.pack(fill='x', pady=(0, 15))
        
        triple_content = ttk.Frame(triple_frame)
        triple_content.pack(fill='x')
        
        # 三重組合選項
        ttk.Label(triple_content, text="↳ 相似度 + 自注意力 + 關鍵詞").pack(anchor='w')
        
        triple_btn_frame = ttk.Frame(triple_frame)
        triple_btn_frame.pack(fill='x', pady=(10, 0))
        
        self.triple_btn = ttk.Button(triple_btn_frame, text="執行三重組合測試", command=self.run_triple_attention)
        self.triple_btn.pack(side='left')
        
        self.triple_status = ttk.Label(triple_btn_frame, text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
        self.triple_status.pack(side='left', padx=(10, 0))

    def run_single_attention(self):
        """執行單一注意力測試"""
        self.single_btn['state'] = 'disabled'
        self.single_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        try:
            from modules.attention_analyzer import AttentionAnalyzer
            
            # 檢查必要檔案
            if not self.last_run_dir:
                messagebox.showerror("錯誤", "請先完成BERT編碼步驟！")
                return
                
            # 設定檔案路徑
            bert_attention_path = os.path.join(self.last_run_dir, "bert_attention.npy")
            topic_labels_path = os.path.join(current_dir, "utils", "topic_labels.json")
            
            if not os.path.exists(bert_attention_path):
                messagebox.showerror("錯誤", "找不到BERT注意力權重檔案！")
                return
            
            # 執行單一注意力分析
            # TODO: 實作單一注意力分析邏輯
            
            self.single_status.config(text=STATUS_TEXT['success'], foreground=COLORS['success'])
            self.step_states['single_done'] = True
            self.update_button_states()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"執行單一注意力測試時發生錯誤：{str(e)}")
            self.single_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
        finally:
            self.single_btn['state'] = 'normal'
    
    def run_dual_attention(self):
        """執行雙重組合測試"""
        self.dual_btn['state'] = 'disabled'
        self.dual_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        try:
            # TODO: 實作雙重組合分析邏輯
            
            self.dual_status.config(text=STATUS_TEXT['success'], foreground=COLORS['success'])
            self.step_states['dual_done'] = True
            self.update_button_states()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"執行雙重組合測試時發生錯誤：{str(e)}")
            self.dual_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
        finally:
            self.dual_btn['state'] = 'normal'
    
    def run_triple_attention(self):
        """執行三重組合測試"""
        self.triple_btn['state'] = 'disabled'
        self.triple_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        try:
            # TODO: 實作三重組合分析邏輯
            
            self.triple_status.config(text=STATUS_TEXT['success'], foreground=COLORS['success'])
            self.step_states['triple_done'] = True
            self.update_button_states()
            
        except Exception as e:
            messagebox.showerror("錯誤", f"執行三重組合測試時發生錯誤：{str(e)}")
            self.triple_status.config(text=STATUS_TEXT['error'], foreground=COLORS['error'])
        finally:
            self.triple_btn['state'] = 'normal'

    def create_comparison_analysis_tab(self):
        """第三分頁：比對分析"""
        frame3 = ttk.Frame(self.notebook)
        self.notebook.add(frame3, text=" 比對分析 ")
        
        main_frame = ttk.Frame(frame3)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 標題
        title_label = ttk.Label(main_frame, text="比對分析", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 分析控制區域
        control_frame = ttk.LabelFrame(main_frame, text="分析控制", padding=15)
        control_frame.pack(fill='x', pady=(0, 15))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill='x')
        
        self.analysis_btn = ttk.Button(btn_frame, text="開始比對分析", command=self.start_analysis)
        self.analysis_btn.pack(side='left', padx=5)
        
        self.analysis_status = ttk.Label(btn_frame, text="狀態: 待分析", foreground="orange")
        self.analysis_status.pack(side='left', padx=(20, 0))
        
        # 結果顯示區域
        results_frame = ttk.LabelFrame(main_frame, text="分析結果", padding=15)
        results_frame.pack(fill='both', expand=True, pady=(15, 0))
        
        # 創建表格顯示結果
        columns = ('模型', '準確率', 'F1分數', '召回率', '精確率')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=10)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            if col == '模型':
                self.results_tree.column(col, width=150, anchor='center')
            else:
                self.results_tree.column(col, width=120, anchor='center')
        
        # 添加滾動條
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        

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
        
        # 第三分頁按鈕
        self.analysis_btn['state'] = 'normal' if self.step_states['encoding_done'] else 'disabled'

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
        """當選擇數據集類型時"""
        selected_name = self.dataset_type.get()
        # 找到對應的數據集類型
        selected_dataset = None
        for ds_key, ds_info in DATASETS.items():
            if ds_info['name'] == selected_name:
                selected_dataset = ds_key
                break
        
        if selected_dataset:
            # 啟用瀏覽按鈕
            self.browse_btn['state'] = 'normal'
            # 更新狀態
            self.import_status.config(
                text=f"請選擇 {DATASETS[selected_dataset]['file_type'].upper()} 格式的{DATASETS[selected_dataset]['description']}檔案",
                foreground=COLORS['info']
            )
        else:
            self.browse_btn['state'] = 'disabled'
            self.import_status.config(text=STATUS_TEXT['pending'], foreground=COLORS['pending'])

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
                
                self.file_path_var.set(file_path)  # 保存完整路徑
                self.import_status.config(
                    text=f"已選擇{DATASETS[selected_dataset]['description']}檔案：{display_path}",
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
            self.process_queue.put(('status', 'success'))
            
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
        """開始BERT編碼"""
        if not self.step_states['processing_done']:
            messagebox.showerror("錯誤", "請先完成文本處理")
            return
            
        # 更新run目錄
        self.update_run_dir_label()
        
        # 禁用編碼按鈕
        self.encoding_btn.config(state='disabled')
        self.encoding_status.config(text="狀態: 處理中", foreground="blue")
        
        # 開始編碼
        threading.Thread(target=self._run_encoding, daemon=True).start()
        self.root.after(100, self._check_encoding_progress)
    
    def _run_encoding(self):
        """在背景執行緒中執行BERT編碼"""
        try:
            from modules.bert_encoder import BertEncoder
            
            # 檢查是否有最後一次預處理的 run 目錄
            if self.last_run_dir is None:
                raise ValueError("請先執行文本預處理步驟")
            
            # 使用最後一次預處理的檔案
            input_file = os.path.join(self.last_run_dir, "01_preprocessed_data.csv")
            
            # 檢查檔案是否存在
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"找不到預處理檔案：{input_file}")
            
            # 讀取預處理後的數據
            df = pd.read_csv(input_file)
            
            # 初始化BERT編碼器，傳入BERT編碼目錄
            encoder = BertEncoder(output_dir=self.run_manager.get_bert_encoding_dir())
            
            # 執行BERT編碼
            embeddings = encoder.encode(df['processed_text'])
            
            # 將結果放入佇列
            output_dir = self.run_manager.get_bert_encoding_dir()
            self.encoding_queue.put(('success', output_dir))
            
        except Exception as e:
            self.encoding_queue.put(('error', str(e)))
    
    def _check_encoding_progress(self):
        """檢查編碼進度並更新UI"""
        try:
            message_type, message = self.encoding_queue.get_nowait()
            
            if message_type == 'error':
                error_msg = f"編碼錯誤: {message}"
                self.encoding_status.config(
                    text=error_msg,
                    foreground=COLORS['error']
                )
                messagebox.showerror("錯誤", error_msg)
                self.encoding_btn['state'] = 'normal'
            elif message_type == 'success':
                success_msg = f"編碼完成，結果已儲存至：{message}"
                self.encoding_status.config(
                    text=success_msg,
                    foreground=COLORS['success']
                )
                self.step_states['encoding_done'] = True
                self.update_button_states()
            
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

    def start_analysis(self):
        """開始比對分析"""
        self.analysis_btn['state'] = 'disabled'
        self.analysis_status.config(text=STATUS_TEXT['analysis_processing'], foreground=COLORS['processing'])
        
        # 清空現有結果
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # 模擬分析過程
        self.root.after(SIMULATION_DELAYS['analysis'], self.complete_analysis)
    
    def complete_analysis(self):
        """完成比對分析"""
        # 模擬添加結果到表格
        sample_results = [
            ("基準模型", "85.2%", "0.83", "0.82", "0.84"),
            ("雙頭注意力", "87.5%", "0.86", "0.85", "0.87"),
            ("三頭注意力", "89.1%", "0.88", "0.87", "0.89")
        ]
        
        for result in sample_results:
            self.results_tree.insert('', 'end', values=result)
        
        self.analysis_status.config(text=STATUS_TEXT['analysis_complete'], foreground=COLORS['success'])
        self.step_states['analysis_done'] = True

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