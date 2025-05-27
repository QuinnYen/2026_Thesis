import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from .config import WINDOW_TITLE, WINDOW_SIZE, WINDOW_MIN_SIZE, COLORS, STATUS_TEXT, SUPPORTED_FILE_TYPES, FONTS, SIMULATION_DELAYS

class MainApplication:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(*WINDOW_MIN_SIZE)
        
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
        
        # 將視窗置中於螢幕
        self.center_window()
        
        # 創建筆記本控件（分頁）
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=15, pady=15)
        
        # 創建三個分頁
        self.create_data_processing_tab()
        self.create_attention_testing_tab()
        self.create_comparison_analysis_tab()
        
        # 初始化按鈕狀態
        self.update_button_states()
    
    def center_window(self):
        """將視窗置中於螢幕"""
        # 更新視窗以獲取實際尺寸
        self.root.update_idletasks()
        
        # 獲取視窗尺寸
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        
        # 如果視窗尺寸為1（還未完全初始化），使用配置中的預設值
        if window_width <= 1 or window_height <= 1:
            # 從WINDOW_SIZE解析寬度和高度
            size_parts = WINDOW_SIZE.split('x')
            window_width = int(size_parts[0])
            window_height = int(size_parts[1])
        
        # 獲取螢幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 計算置中位置
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # 設定視窗位置
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
    def create_data_processing_tab(self):
        """第一分頁：資料處理"""
        frame1 = ttk.Frame(self.notebook)
        self.notebook.add(frame1, text="第一分頁 - 資料處理")
        
        # 主要容器
        main_frame = ttk.Frame(frame1)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 標題
        title_label = ttk.Label(main_frame, text="資料處理流程", font=FONTS['title'])
        title_label.pack(pady=(0, 20))
        
        # 步驟1：文本輸入 → 導入檔案
        step1_frame = ttk.LabelFrame(main_frame, text="① 文本輸入 → 導入檔案", padding=15)
        step1_frame.pack(fill='x', pady=(0, 15))
        
        input_frame = ttk.Frame(step1_frame)
        input_frame.pack(fill='x')
        
        ttk.Label(input_frame, text="選擇檔案:").pack(side='left')
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(input_frame, textvariable=self.file_path_var, width=60)
        file_entry.pack(side='left', padx=(10, 5), fill='x', expand=True)
        
        browse_btn = ttk.Button(input_frame, text="瀏覽", command=self.browse_file)
        browse_btn.pack(side='left', padx=(5, 10))
        
        self.import_status = ttk.Label(step1_frame, text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
        self.import_status.pack(anchor='w', pady=(10, 0))
        
        # 步驟2：文本處理 → 開始處理
        step2_frame = ttk.LabelFrame(main_frame, text="② 文本處理 → 開始處理", padding=15)
        step2_frame.pack(fill='x', pady=(0, 15))
        
        process_frame = ttk.Frame(step2_frame)
        process_frame.pack(fill='x')
        
        self.process_btn = ttk.Button(process_frame, text="開始處理", command=self.start_processing)
        self.process_btn.pack(side='left')
        
        self.process_status = ttk.Label(step2_frame, text="狀態: 待處理", foreground="orange")
        self.process_status.pack(anchor='w', pady=(10, 0))
        
        # 步驟3：Bert編碼 → 開始編碼
        step3_frame = ttk.LabelFrame(main_frame, text="③ Bert編碼 → 開始編碼", padding=15)
        step3_frame.pack(fill='x', pady=(0, 15))
        
        encoding_frame = ttk.Frame(step3_frame)
        encoding_frame.pack(fill='x')
        
        self.encoding_btn = ttk.Button(encoding_frame, text="開始編碼", command=self.start_encoding)
        self.encoding_btn.pack(side='left')
        
        self.encoding_status = ttk.Label(step3_frame, text="狀態: 待處理", foreground="orange")
        self.encoding_status.pack(anchor='w', pady=(10, 0))
        

        
    def create_attention_testing_tab(self):
        """第二分頁：注意力機制測試"""
        frame2 = ttk.Frame(self.notebook)
        self.notebook.add(frame2, text="第二分頁 - 注意力機制測試")
        
        main_frame = ttk.Frame(frame2)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 標題
        title_label = ttk.Label(main_frame, text="注意力機制測試", font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 基準（單一）
        baseline_frame = ttk.LabelFrame(main_frame, text="基準（單一注意力機制）", padding=15)
        baseline_frame.pack(fill='x', pady=(0, 15))
        
        baseline_content = ttk.Frame(baseline_frame)
        baseline_content.pack(fill='x')
        
        ttk.Label(baseline_content, text="↳ 相似度").pack(anchor='w')
        ttk.Label(baseline_content, text="↳ 關鍵詞").pack(anchor='w')
        ttk.Label(baseline_content, text="↳ 自").pack(anchor='w')
        
        baseline_btn_frame = ttk.Frame(baseline_frame)
        baseline_btn_frame.pack(fill='x', pady=(10, 0))
        
        self.baseline_btn = ttk.Button(baseline_btn_frame, text="執行", command=self.run_baseline)
        self.baseline_btn.pack(side='left')
        
        self.baseline_status = ttk.Label(baseline_btn_frame, text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
        self.baseline_status.pack(side='left', padx=(10, 0))
        
        # 組合（雙頭）
        dual_frame = ttk.LabelFrame(main_frame, text="組合（雙頭注意力機制）", padding=15)
        dual_frame.pack(fill='x', pady=(0, 15))
        
        dual_content = ttk.Frame(dual_frame)
        dual_content.pack(fill='x')
        
        ttk.Label(dual_content, text="↳ 相似度 + 關鍵詞").pack(anchor='w')
        ttk.Label(dual_content, text="↳ 相似度 + 自").pack(anchor='w')
        ttk.Label(dual_content, text="↳ 關鍵詞 + 自").pack(anchor='w')
        
        dual_btn_frame = ttk.Frame(dual_frame)
        dual_btn_frame.pack(fill='x', pady=(10, 0))
        
        self.dual_btn = ttk.Button(dual_btn_frame, text="執行", command=self.run_dual_head)
        self.dual_btn.pack(side='left')
        
        self.dual_status = ttk.Label(dual_btn_frame, text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
        self.dual_status.pack(side='left', padx=(10, 0))
        
        # 組合（三頭）
        triple_frame = ttk.LabelFrame(main_frame, text="組合（三頭注意力機制）", padding=15)
        triple_frame.pack(fill='x', pady=(0, 15))
        
        triple_content = ttk.Frame(triple_frame)
        triple_content.pack(fill='x')
        
        ttk.Label(triple_content, text="↳ 相似度 + 關鍵詞 + 自").pack(anchor='w')
        
        triple_btn_frame = ttk.Frame(triple_frame)
        triple_btn_frame.pack(fill='x', pady=(10, 0))
        
        self.triple_btn = ttk.Button(triple_btn_frame, text="執行", command=self.run_triple_head)
        self.triple_btn.pack(side='left')
        
        self.triple_status = ttk.Label(triple_btn_frame, text=STATUS_TEXT['pending'], foreground=COLORS['pending'])
        self.triple_status.pack(side='left', padx=(10, 0))
        

        
    def create_comparison_analysis_tab(self):
        """第三分頁：比對分析"""
        frame3 = ttk.Frame(self.notebook)
        self.notebook.add(frame3, text="第三分頁 - 比對分析")
        
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
        self.analysis_btn.pack(side='left')
        
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
        
        # 第二分頁按鈕
        self.baseline_btn['state'] = 'normal' if self.step_states['encoding_done'] else 'disabled'
        self.dual_btn['state'] = 'normal' if self.step_states['baseline_done'] else 'disabled'
        self.triple_btn['state'] = 'normal' if self.step_states['dual_head_done'] else 'disabled'
        
        # 第三分頁按鈕
        self.analysis_btn['state'] = 'normal' if self.step_states['triple_head_done'] else 'disabled'

    def browse_file(self):
        """瀏覽檔案"""
        file_path = filedialog.askopenfilename(
            title="選擇文本檔案",
            filetypes=SUPPORTED_FILE_TYPES
        )
        if file_path:
            self.file_path_var.set(file_path)
            self.import_status.config(text=STATUS_TEXT['file_selected'], foreground=COLORS['processing'])
            self.step_states['file_imported'] = True
            self.update_button_states()

    def start_processing(self):
        """開始處理"""
        if not self.file_path_var.get():
            messagebox.showwarning("警告", "請先選擇檔案")
            return
        
        self.process_btn['state'] = 'disabled'
        self.process_status.config(text=STATUS_TEXT['processing'], foreground=COLORS['processing'])
        
        # 模擬處理過程
        self.root.after(SIMULATION_DELAYS['file_processing'], self.complete_processing)
    
    def complete_processing(self):
        """完成處理"""
        self.process_status.config(text=STATUS_TEXT['success'], foreground=COLORS['success'])
        self.step_states['processing_done'] = True
        self.update_button_states()

    def start_encoding(self):
        """開始編碼"""
        self.encoding_btn['state'] = 'disabled'
        self.encoding_status.config(text=STATUS_TEXT['encoding_processing'], foreground=COLORS['processing'])
        
        # 模擬編碼過程
        self.root.after(SIMULATION_DELAYS['bert_encoding'], self.complete_encoding)
    
    def complete_encoding(self):
        """完成編碼"""
        self.encoding_status.config(text=STATUS_TEXT['encoding_complete'], foreground=COLORS['success'])
        self.step_states['encoding_done'] = True
        self.update_button_states()

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

def main():
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main() 