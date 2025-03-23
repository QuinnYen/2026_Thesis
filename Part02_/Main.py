import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import subprocess
import threading
import os
import sys
import logging
import nltk
import json
import traceback
from src.console_output import ConsoleOutputManager
from src.settings.visualization_config import configure_chinese_fonts, check_chinese_display

class ModuleFilter(logging.Filter):
    def __init__(self, module_name):
        self.module_name = module_name
    
    def filter(self, record):
        if record.name.startswith(self.module_name) and record.levelno < logging.WARNING:
            return False
        return True

def configure_root_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加transformers庫的過濾器
    root_logger.addFilter(ModuleFilter('transformers'))

# 配置根日誌器
configure_root_logger()

class CrossDomainSentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("跨領域情感分析系統 v3.0")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # 計算視窗置中的位置
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 800
        window_height = 600
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 設置應用程式圖標和風格
        self.style = ttk.Style()
        self.style.configure('TNotebook', tabposition='n')
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Status.TLabel', font=('Arial', 10), foreground='green')
        
        # 設置日誌
        log_dir = "./Part02_/logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.logger = logging.getLogger('main_app')
        self.logger.setLevel(logging.INFO)
        
        # 檔案處理器設定
        file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 創建變數 - 確保這些在主線程中創建
        self.file_path = tk.StringVar()
        self.topic_count = tk.StringVar(value="10")  # 預設LDA主題數量
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="準備就緒。")
        
        # 當前數據集ID
        self.current_dataset_id = None
        
        # 初始化中文字體支援
        try:
            from src.settings.visualization_config import configure_chinese_fonts
            font_config_result = configure_chinese_fonts()
            self.logger.info(f"中文字體配置: {font_config_result}")
        except Exception as e:
            self.logger.warning(f"中文字體配置失敗: {str(e)}")
        
        # 確保NLTK資源已下載
        try:
            self._ensure_nltk_resources()
        except Exception as e:
            messagebox.showerror("錯誤", f"無法下載NLTK資源: {str(e)}")
            self.logger.error(f"NLTK資源下載失敗: {str(e)}")
            root.destroy()
            return
        
        # 初始化結果管理器
        from src.result_manager import ResultManager
        self.result_manager = ResultManager(logger=self.logger)
        
        # 初始化輸出目錄結構
        self._init_output_directories()
        
        # 建立介面
        self._create_widgets()
        
        # 從結果管理器加載當前分析數據
        self._load_recent_results()
        
        # 在 UI 就緒後才執行中文字體測試
        self.root.after(1000, self._check_chinese_font_display)
        
        self.logger.info("Application started")

    def _check_chinese_font_display(self):
        """在主線程中執行中文字體顯示測試"""
        try:
            from src.settings.visualization_config import check_chinese_display
            display_result = check_chinese_display()
            if not display_result:
                self.logger.warning("中文顯示測試不成功，圖表中的中文可能無法正確顯示")
        except Exception as e:
            self.logger.warning(f"中文顯示測試失敗: {str(e)}")
        
    def _create_widgets(self):
        # 建立分頁
        self.tab_control = ttk.Notebook(self.root)
        
        # 建立各個分頁
        self.tab_data_processing = ttk.Frame(self.tab_control)
        self.tab_model_training = ttk.Frame(self.tab_control)
        self.tab_evaluation = ttk.Frame(self.tab_control)
        self.tab_visualization = ttk.Frame(self.tab_control)
        self.tab_results = ttk.Frame(self.tab_control)
        
        # 添加分頁到控制器
        tab_font = ('Arial', 10, 'bold')
        self.tab_control.add(self.tab_data_processing, text="　資料處理　")
        self.tab_control.add(self.tab_model_training, text="　模型訓練　")
        self.tab_control.add(self.tab_evaluation, text="　評估分析　")
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
        # 主框架
        main_frame = ttk.Frame(self.tab_data_processing, padding=10)
        main_frame.pack(fill="both", expand=True)
        
        # 步驟1: 導入數據
        step1_frame = ttk.LabelFrame(main_frame, text="步驟 1: 導入數據", padding=10)
        step1_frame.pack(fill="x", pady=5)
        
        ttk.Label(step1_frame, text="選擇評論資料集導入系統 (Amazon、Yelp、IMDB)").pack(anchor="w", pady=5)
        
        file_frame = ttk.Frame(step1_frame)
        file_frame.pack(fill="x", pady=5)
        
        ttk.Button(file_frame, text="選擇文件", command=self.select_file).pack(side="left")
        ttk.Label(file_frame, textvariable=self.file_path).pack(side="left", padx=10)
        ttk.Button(file_frame, text="開始導入數據", command=self.import_data).pack(side="right")
        
        # 步驟2: BERT提取語義表示
        step2_frame = ttk.LabelFrame(main_frame, text="步驟 2: BERT提取語義表示", padding=10)
        step2_frame.pack(fill="x", pady=5)
        
        ttk.Label(step2_frame, text="使用BERT模型為評論文本生成向量表示").pack(anchor="w", pady=5)
        ttk.Button(step2_frame, text="執行BERT語義提取", command=self.extract_bert_embeddings).pack(anchor="e", pady=5)
        
        # 步驟3: LDA面向切割
        step3_frame = ttk.LabelFrame(main_frame, text="步驟 3: LDA面向切割", padding=10)
        step3_frame.pack(fill="x", pady=5)
        
        ttk.Label(step3_frame, text="使用LDA進行主題建模，識別不同面向 (主題數量將根據數據源自動設定)").pack(anchor="w", pady=5)
        
        lda_frame = ttk.Frame(step3_frame)
        lda_frame.pack(fill="x", pady=5)
        
        ttk.Button(lda_frame, text="執行LDA面向切割", command=self.perform_lda).pack(anchor="e", pady=5)
        
        # 步驟4: 計算面向相關句子的平均向量
        step4_frame = ttk.LabelFrame(main_frame, text="步驟 4: 計算面向相關句子的平均向量", padding=10)
        step4_frame.pack(fill="x", pady=5)
        
        ttk.Label(step4_frame, text="為每個識別出的面向計算代表性向量").pack(anchor="w", pady=5)
        
        vector_frame = ttk.Frame(step4_frame)
        vector_frame.pack(fill="x", pady=5)
        
        ttk.Button(vector_frame, text="執行計算", command=self.calculate_aspect_vectors).pack(side="left")
        ttk.Button(vector_frame, text="匯出平均向量", command=self.export_vectors).pack(side="right")

    
    def _setup_status_bar(self):
        status_frame = ttk.Frame(self.root, padding=10)
        status_frame.pack(side="bottom", fill="x")
        
        ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w")
        
        progress_frame = ttk.Frame(status_frame)
        progress_frame.pack(fill="x", pady=5)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            orient="horizontal", 
            length=100, 
            mode="determinate", 
            variable=self.progress_var
        )
        self.progress_bar.pack(fill="x")
    
    def _setup_results_tab(self):
        """設置結果瀏覽分頁內容"""
        # 主框架
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
        # 使用Notebook來分類顯示不同類型的結果
        result_notebook = ttk.Notebook(main_frame)
        result_notebook.pack(fill="both", expand=True, pady=10)
        
        # 處理資料頁面
        self.data_tab = ttk.Frame(result_notebook)
        result_notebook.add(self.data_tab, text="處理資料")
        
        # 模型頁面
        self.models_tab = ttk.Frame(result_notebook)
        result_notebook.add(self.models_tab, text="模型")
        
        # 可視化頁面
        self.vis_tab = ttk.Frame(result_notebook)
        result_notebook.add(self.vis_tab, text="可視化")
        
        # 導出結果頁面
        self.exports_tab = ttk.Frame(result_notebook)
        result_notebook.add(self.exports_tab, text="導出結果")
        
        # 初始化各分頁的內容
        self._setup_result_data_tab(self.data_tab)
        self._setup_result_models_tab(self.models_tab)
        self._setup_result_vis_tab(self.vis_tab)
        self._setup_result_exports_tab(self.exports_tab)
        
        # 加載數據集
        self._refresh_datasets()

    def _setup_result_data_tab(self, parent):
        """設置處理資料結果分頁"""
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

    def _setup_result_models_tab(self, parent):
        """設置模型結果分頁"""
        # 創建Treeview
        columns = ("filename", "step", "model_type", "path", "metadata")
        self.models_tree = ttk.Treeview(parent, columns=columns, show="headings")
        
        # 設置列標題
        self.models_tree.heading("filename", text="檔案名稱")
        self.models_tree.heading("step", text="處理步驟")
        self.models_tree.heading("model_type", text="模型類型")
        self.models_tree.heading("path", text="路徑")
        self.models_tree.heading("metadata", text="元數據")
        
        # 設置列寬
        self.models_tree.column("filename", width=200)
        self.models_tree.column("step", width=100)
        self.models_tree.column("model_type", width=100)
        self.models_tree.column("path", width=300)
        self.models_tree.column("metadata", width=200)
        
        # 添加滾動條
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.models_tree.yview)
        self.models_tree.configure(yscrollcommand=scrollbar.set)
        
        # 佈局
        self.models_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 雙擊打開文件
        self.models_tree.bind("<Double-1>", self._on_model_double_click)

    def _setup_result_vis_tab(self, parent):
        """設置可視化結果分頁"""
        # 上方控制區
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=5)
        
        # 選擇可視化類型下拉選單
        ttk.Label(control_frame, text="可視化類型:").pack(side="left", padx=5)
        self.vis_type_combo = ttk.Combobox(control_frame, state="readonly", values=["全部", "主題可視化", "向量可視化"])
        self.vis_type_combo.pack(side="left", padx=5)
        self.vis_type_combo.current(0)
        self.vis_type_combo.bind("<<ComboboxSelected>>", self._on_vis_type_changed)
        
        # 顯示區域 - 使用Canvas和Frame組合來實現可滾動的網格佈局
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True, pady=5)
        
        self.vis_canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.vis_canvas.yview)
        
        self.vis_frame = ttk.Frame(self.vis_canvas)
        self.vis_frame.bind(
            "<Configure>",
            lambda e: self.vis_canvas.configure(scrollregion=self.vis_canvas.bbox("all"))
        )
        
        self.vis_canvas.create_window((0, 0), window=self.vis_frame, anchor="nw")
        self.vis_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.vis_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _setup_result_exports_tab(self, parent):
        """設置導出結果分頁"""
        # 創建Treeview
        columns = ("filename", "format", "path", "created_at")
        self.exports_tree = ttk.Treeview(parent, columns=columns, show="headings")
        
        # 設置列標題
        self.exports_tree.heading("filename", text="檔案名稱")
        self.exports_tree.heading("format", text="格式")
        self.exports_tree.heading("path", text="路徑")
        self.exports_tree.heading("created_at", text="創建時間")
        
        # 設置列寬
        self.exports_tree.column("filename", width=200)
        self.exports_tree.column("format", width=100)
        self.exports_tree.column("path", width=300)
        self.exports_tree.column("created_at", width=150)
        
        # 添加滾動條
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.exports_tree.yview)
        self.exports_tree.configure(yscrollcommand=scrollbar.set)
        
        # 佈局
        self.exports_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 雙擊打開文件
        self.exports_tree.bind("<Double-1>", self._on_export_double_click)

    def _refresh_datasets(self):
        """刷新數據集選擇下拉選單"""
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
            self.logger.error(f"Error refreshing dataset list: {str(e)}")

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
            self.logger.error(f"Error selecting dataset: {str(e)}")

    def _load_dataset_results(self, dataset_id):
        """加載數據集的處理結果"""
        try:
            # 獲取數據集信息
            dataset_info = self.result_manager.get_dataset_summary(dataset_id)
            if not dataset_info:
                return
            
            # 清空結果顯示
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            for item in self.models_tree.get_children():
                self.models_tree.delete(item)
            
            # 清空可視化顯示區
            for widget in self.vis_frame.winfo_children():
                widget.destroy()
            
            for item in self.exports_tree.get_children():
                self.exports_tree.delete(item)
            
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
                    
                    elif result_type == "model":
                        # 添加到模型頁面
                        metadata = result.get("metadata", {})
                        model_type = metadata.get("model_type", "未知")
                        metadata_str = json.dumps(metadata, ensure_ascii=False)
                        self.models_tree.insert(
                            "", "end", 
                            values=(result["filename"], step_display, model_type, result["path"], metadata_str)
                        )
                    
                    elif result_type == "visualization":
                        # 添加到可視化頁面
                        self._add_visualization(result, step_display)
                    
                    elif result_type == "export":
                        # 添加到導出結果頁面
                        import os
                        filename = result["filename"]
                        format_ext = os.path.splitext(filename)[1].lstrip('.')
                        created_at = step_info.get("completed_at", "")
                        self.exports_tree.insert(
                            "", "end", 
                            values=(filename, format_ext, result["path"], created_at)
                        )
            
            # 更新狀態
            self.status_var.set(f"已載入數據集 '{dataset_info['name']}' 的處理結果")
        
        except Exception as e:
            self.logger.error(f"Error loading dataset results: {str(e)}")
            traceback.print_exc()

    def _add_visualization(self, result, step_display):
        """添加可視化結果到UI"""
        try:
            import os
            from PIL import Image, ImageTk
            
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
            
            # 圖片（如果是圖片文件）
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif')
            if result["path"].lower().endswith(image_extensions):
                try:
                    # 讀取圖片
                    img = Image.open(result["path"])
                    
                    # 調整大小
                    width, height = img.size
                    max_width = 400
                    if width > max_width:
                        ratio = max_width / width
                        width = max_width
                        height = int(height * ratio)
                    
                    img = img.resize((width, height), Image.Resampling.LANCZOS)
                    
                    # 轉換為Tkinter可用的格式
                    photo = ImageTk.PhotoImage(img)
                    
                    # 保持對圖片的引用（否則會被垃圾回收）
                    label = ttk.Label(card, image=photo)
                    label.image = photo
                    label.pack(padx=5, pady=5)
                except Exception as e:
                    self.logger.error(f"Error loading image: {str(e)}")
                    ttk.Label(card, text=f"無法顯示圖片: {str(e)}").pack(padx=5, pady=5)
            
            # 信息
            info_frame = ttk.Frame(card)
            info_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Label(info_frame, text=f"步驟: {step_display}").pack(side="left", padx=5)
            
            # 按鈕
            btn_frame = ttk.Frame(card)
            btn_frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Button(
                btn_frame, text="查看", 
                command=lambda path=result["path"]: self._open_file(path)
            ).pack(side="left", padx=5)
            
            # 設置可視化類型作為標籤（用於過濾）
            card.vis_type = vis_type
        
        except Exception as e:
            self.logger.error(f"Error adding visualization result: {str(e)}")

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
            self.logger.error(f"Error switching visualization type: {str(e)}")

    def _on_data_double_click(self, event):
        """處理數據結果雙擊事件"""
        # 獲取選中的項目
        item = self.data_tree.selection()[0]
        file_path = self.data_tree.item(item, "values")[2]  # 路徑在第三列
        
        # 打開文件
        self._open_file(file_path)

    def _on_model_double_click(self, event):
        """處理模型結果雙擊事件"""
        # 獲取選中的項目
        item = self.models_tree.selection()[0]
        file_path = self.models_tree.item(item, "values")[3]  # 路徑在第四列
        
        # 打開文件
        self._open_file(file_path)

    def _on_export_double_click(self, event):
        """處理導出結果雙擊事件"""
        # 獲取選中的項目
        item = self.exports_tree.selection()[0]
        file_path = self.exports_tree.item(item, "values")[2]  # 路徑在第三列
        
        # 打開文件
        self._open_file(file_path)

    def _refresh_results(self):
        """刷新結果顯示"""
        self._refresh_datasets()
        if self.current_dataset_id:
            self._load_dataset_results(self.current_dataset_id)
        self.status_var.set("Results refreshed")

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
            self.logger.error(f"Error generating overview report: {str(e)}")
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
            self.logger.error(f"Error viewing dataset report: {str(e)}")
            messagebox.showerror("錯誤", f"查看數據集報告時出錯: {str(e)}")

    def _ensure_nltk_resources(self):
        """確保必要的NLTK資源已下載"""
        resources = [
            'punkt',                    # 用於分詞
            'stopwords',                # 停用詞
            'wordnet',                  # 詞形還原
            'omw-1.4',                  # Open Multilingual WordNet
            'averaged_perceptron_tagger' # 詞性標註
        ]
        
        # 設置NLTK下載目錄 - 確保使用專案目錄下的nltk_data
        nltk_data_path = "./Part02_/nltk_data"
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path)
        
        # 將此路徑添加到NLTK搜索路徑中
        nltk.data.path.insert(0, nltk_data_path)
        
        # 下載缺失的資源
        for resource in resources:
            try:
                # 按資源類型嘗試正確的路徑
                if resource in ['punkt', 'averaged_perceptron_tagger']:
                    path = f'tokenizers/{resource}'
                elif resource in ['stopwords', 'wordnet', 'omw-1.4']:
                    path = f'corpora/{resource}'
                else:
                    path = resource
                    
                nltk.data.find(path)
                self.logger.info(f"NLTK resource '{resource}' already exists")
            except LookupError:
                self.logger.info(f"Downloading NLTK resource '{resource}'...")
                try:
                    nltk.download(resource, download_dir=nltk_data_path, quiet=False)
                    self.logger.info(f"NLTK resource '{resource}' download complete")
                except Exception as e:
                    self.logger.error(f"Error downloading NLTK resource '{resource}': {str(e)}")
                    raise
    
    def _init_output_directories(self):
        """初始化輸出目錄結構，創建有組織的子目錄"""
        # 基本目錄
        self.data_dir = "./Part02_/data"
        self.results_dir = "./Part02_/results"
        
        # 子目錄
        self.processed_data_dir = os.path.join(self.results_dir, "01_processed_data")
        self.bert_embeddings_dir = os.path.join(self.results_dir, "02_bert_embeddings")
        self.lda_topics_dir = os.path.join(self.results_dir, "03_lda_topics")
        self.aspect_vectors_dir = os.path.join(self.results_dir, "04_aspect_vectors")
        self.visualizations_dir = os.path.join(self.results_dir, "visualizations")
        self.exports_dir = os.path.join(self.results_dir, "exports")
        
        # 可視化子目錄
        self.topic_vis_dir = os.path.join(self.visualizations_dir, "topics")
        self.vector_vis_dir = os.path.join(self.visualizations_dir, "vectors")
        
        # 創建所有必要的目錄
        for directory in [
            self.data_dir, self.results_dir, 
            self.processed_data_dir, self.bert_embeddings_dir, 
            self.lda_topics_dir, self.aspect_vectors_dir,
            self.visualizations_dir, self.exports_dir,
            self.topic_vis_dir, self.vector_vis_dir
        ]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        self.logger.info("Output directory structure initialized")
    
    # 功能方法
    def select_file(self):
        """選擇數據文件"""
        file_path = filedialog.askopenfilename(
            title="選擇評論資料文件",
            filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path.set(file_path)
            self.status_var.set(f"已選擇文件: {os.path.basename(file_path)}")
            self.logger.info(f"File selected: {file_path}")
    
    def import_data(self):
        """導入數據"""
        if not self.file_path.get():
            messagebox.showerror("錯誤", "請先選擇文件!")
            return
        
        self.status_var.set("正在導入數據...")
        self.progress_var.set(10)
        
        # 在另一個線程中執行數據導入
        threading.Thread(target=self._run_data_import, daemon=True).start()
    
    def _run_data_import(self):
        """實際執行數據導入的方法"""
        try:
            from src.data_importer import DataImporter
            
            file_path = self.file_path.get()
            file_name = os.path.basename(file_path)
            
            # 建立一個數據導入器
            importer = DataImporter(output_dir=self.processed_data_dir)
            
            # 定義進度回調函數
            def update_progress(message, percentage):
                if percentage < 0:
                    self.root.after_idle(lambda: self.status_var.set(f"錯誤: {message}"))
                    return
                    
                def update_ui():
                    try:
                        progress = 20 + (percentage * 0.8 / 100)
                        self.progress_var.set(progress)
                        self.status_var.set(message)
                    except Exception:
                        pass
                
                self.root.after_idle(update_ui)
            
            # 執行數據導入
            processed_file_path = importer.import_data(file_path, callback=update_progress)
            
            # 保存處理後的文件路徑以供後續步驟使用
            self.processed_data_path = processed_file_path
            
            # 註冊處理結果
            self.result_manager.register_result(
                self.current_dataset_id,
                "data_import",
                "data",
                processed_file_path,
                metadata={
                    "original_file": file_path,
                    "processed_rows": None  # 這裡可以添加處理的行數等信息
                }
            )
            
            # 更新UI
            def show_completion():
                try:
                    self.status_var.set(f"數據導入完成: {os.path.basename(processed_file_path)}")
                    self.progress_var.set(25)
                    messagebox.showinfo("成功", "數據已成功導入和初步處理!")
                except Exception:
                    pass
            
            self.root.after_idle(show_completion)
            
        except Exception as e:
            import traceback
            error_msg = f"導入數據時發生錯誤: {str(e)}"
            traceback.print_exc()
            
            def show_error():
                try:
                    messagebox.showerror("錯誤", error_msg)
                    self.status_var.set("導入數據失敗")
                    self.progress_var.set(0)
                except Exception:
                    pass
            
            self.root.after_idle(show_error)

    def extract_bert_embeddings(self):
        """執行BERT語義提取"""
        if not self.processed_data_path:
            messagebox.showerror("錯誤", "請先導入並處理數據!")
            return
        
        # 打開控制台窗口並獲取日誌文件路徑與狀態文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("BERT語義提取", auto_close=True)
        
        self.status_var.set("正在進行BERT語義提取...")
        self.progress_var.set(30)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        bert_logger = ConsoleOutputManager.setup_console_logger('bert_extraction', log_file)
        bert_logger.info(f"Starting BERT semantic extraction, processing document: {self.processed_data_path}")
        
        # 記錄處理參數
        bert_logger.info(f"Starting BERT embedding extraction with model: bert-base-uncased")
        bert_logger.info(f"Output directory: {self.results_dir}")
        bert_logger.info("====================================")
        
        # 在另一個線程中執行BERT提取
        threading.Thread(target=self._run_bert_extraction, args=(bert_logger, log_file, status_file), daemon=True).start()
    
    def _run_bert_extraction(self, logger, log_file, status_file):
        """實際執行BERT提取的方法"""
        try:
            from src.bert_embedder import BertEmbedder
            
            # 獲取處理後的數據路徑
            data_path = self.processed_data_path
            
            # 記錄信息到日誌
            logger.info(f"Processing file: {data_path}")
            
            # 創建BERT編碼器
            embedder = BertEmbedder(
                model_name='bert-base-uncased',
                output_dir=self.bert_embeddings_dir,  # 使用專門的BERT嵌入目錄
                logger=logger
            )
            
            # 定義進度回調函數
            def update_progress(message, percentage):
                if percentage < 0:
                    self.root.after_idle(lambda: self.status_var.set(f"錯誤: {message}"))
                    return
                    
                def update_ui():
                    try:
                        progress = 30 + (percentage * 20 / 100)
                        self.progress_var.set(progress)
                        self.status_var.set(message)
                    except Exception:
                        pass
                
                self.root.after_idle(update_ui)
                logger.info(f"{message} ({percentage}%)")
            
            # 執行BERT嵌入提取
            result = embedder.extract_embeddings(
                data_path,
                text_column="clean_text",
                batch_size=16,
                callback=update_progress
            )
            
            # 保存結果路徑以供後續步驟使用
            self.bert_embeddings_path = result['embeddings_path']
            self.bert_metadata_path = result['metadata_path']
            self.embedding_dim = result['embedding_dim']
            
            # 註冊處理結果
            self.result_manager.register_result(
                self.current_dataset_id,
                "bert_embedding",
                "data",
                self.bert_embeddings_path,
                metadata={
                    "embedding_dim": self.embedding_dim,
                    "source_file": data_path
                }
            )
            
            self.result_manager.register_result(
                self.current_dataset_id,
                "bert_embedding",
                "data",
                self.bert_metadata_path,
                metadata={
                    "contains": "text and metadata"
                }
            )
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"BERT semantic extraction complete!")
            logger.info(f"Embedding vectors saved to: {self.bert_embeddings_path}")
            logger.info(f"Metadata saved to: {self.bert_metadata_path}")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            logger.info("====================================")
            logger.info(f"處理已完成。窗口將在幾秒後自動關閉。")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 更新UI
            def show_completion():
                try:
                    self.status_var.set(f"BERT語義提取完成！嵌入維度: {self.embedding_dim}")
                    self.progress_var.set(50)
                    messagebox.showinfo(
                        "成功", 
                        f"BERT語義提取完成！\n嵌入向量已保存至: {os.path.basename(self.bert_embeddings_path)}\n嵌入維度: {self.embedding_dim}"
                    )
                except Exception:
                    pass
            
            self.root.after_idle(show_completion)
            
        except Exception as e:
            import traceback
            error_msg = f"BERT語義提取時發生錯誤: {str(e)}"
            traceback.print_exc()
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            def show_error():
                try:
                    messagebox.showerror("錯誤", error_msg)
                    self.status_var.set("BERT語義提取失敗")
                    self.progress_var.set(30)
                except Exception:
                    pass
            
            self.root.after_idle(show_error)

    def perform_lda(self):
        """執行LDA面向切割"""
        if not self.bert_metadata_path:
            messagebox.showerror("錯誤", "請先執行BERT語義提取!")
            return
        
        # 打開控制台窗口並獲取日誌文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("LDA面向切割", auto_close=True)
        
        self.status_var.set("正在進行LDA面向切割...")
        self.progress_var.set(60)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        lda_logger = ConsoleOutputManager.setup_console_logger('lda_extraction', log_file)
        lda_logger.info(f"開始LDA面向切割，處理文件: {self.bert_metadata_path}")
        
        # 確定數據來源，以便選擇對應的主題標籤
        self.determine_data_source()
        
        # 在另一個線程中執行LDA面向切割
        threading.Thread(target=self._run_lda, args=(lda_logger, log_file, status_file), daemon=True).start()

    def determine_data_source(self):
        """根據檔案名稱或內容確定數據來源"""
        # 從文件名或路徑判斷數據源
        file_path = self.file_path.get() or ""
        file_name = os.path.basename(file_path).lower()
        
        # 預設為未知數據源
        self.data_source = "unknown"
        
        # 檢查文件名中的關鍵詞
        if "imdb" in file_name or "movie" in file_name or "film" in file_name:
            self.data_source = "imdb"
            self.logger.info(f"檢測到IMDB電影評論數據: {file_name}")
        elif "amazon" in file_name or "product" in file_name:
            self.data_source = "amazon"
            self.logger.info(f"檢測到Amazon產品評論數據: {file_name}")
        elif "yelp" in file_name or "restaurant" in file_name:
            self.data_source = "yelp"
            self.logger.info(f"檢測到Yelp餐廳評論數據: {file_name}")
        else:
            # 也可以通過對話框讓用戶選擇
            self.ask_data_source()

    def ask_data_source(self):
        """詢問用戶數據來源"""
        from tkinter import simpledialog
        
        sources = {
            "1": "imdb",
            "2": "amazon", 
            "3": "yelp"
        }
        
        source = simpledialog.askstring(
            "選擇數據來源",
            "請選擇評論數據來源:\n1. IMDB電影評論\n2. Amazon產品評論\n3. Yelp餐廳評論",
            initialvalue="1"
        )
        
        if source in sources:
            self.data_source = sources[source]
            self.logger.info(f"用戶選擇了數據來源: {self.data_source}")
        else:
            self.data_source = "unknown"
            self.logger.info("未指定數據來源或取消選擇")
    
    def _run_lda(self, logger, log_file, status_file):
        """實際執行LDA的方法"""
        try:
            from src.lda_aspect_extractor import LDATopicExtractor
            from src.settings.topic_labels import TOPIC_LABELS_en, TOPIC_LABELS_zh
            
            # 獲取BERT元數據路徑
            metadata_path = self.bert_metadata_path
            
            # 記錄信息到日誌
            logger.info(f"處理文件: {metadata_path}")
            logger.info(f"數據來源: {self.data_source}")
            
            # 獲取適合的主題標籤
            topic_labels = None
            if self.data_source in TOPIC_LABELS_zh:
                topic_labels = TOPIC_LABELS_zh[self.data_source]
                logger.info(f"已載入 {self.data_source} 的自定義主題標籤")
                
                # 輸出所有標籤
                for idx, label in topic_labels.items():
                    logger.info(f"主題 {idx+1}: {label}")
                
                # 自動設定主題數量為標籤數量
                topic_count = len(topic_labels)
                logger.info(f"根據主題標籤自動設定主題數量: {topic_count}")
            else:
                # 如果沒有匹配的標籤，則使用默認數量
                topic_count = 10
                logger.info(f"未找到匹配的主題標籤，使用默認主題數量: {topic_count}")
            
            # 記錄處理參數
            logger.info(f"主題數量: {topic_count}")
            logger.info(f"輸出目錄: {self.results_dir}")
            logger.info("====================================")
            
            # 創建LDA面向提取器
            extractor = LDATopicExtractor(
                output_dir=self.lda_topics_dir,  # 使用專門的LDA主題目錄
                logger=logger
            )
            
            # 定義進度回調函數
            def update_progress(message, percentage):
                if percentage < 0:
                    self.root.after_idle(lambda: self.status_var.set(f"錯誤: {message}"))
                    return
                    
                def update_ui():
                    try:
                        progress = 60 + (percentage * 15 / 100)
                        self.progress_var.set(progress)
                        self.status_var.set(message)
                    except Exception:
                        pass
                
                self.root.after_idle(update_ui)
                logger.info(f"{message} ({percentage}%)")
            
            # 執行LDA面向切割
            results = extractor.run_lda(
                metadata_path,
                n_topics=topic_count,
                topic_labels=topic_labels,
                callback=update_progress
            )
            
            # 保存結果路徑以供後續步驟使用
            self.lda_model_path = results['lda_model_path']
            self.topics_path = results['topics_path']
            self.topic_metadata_path = results['topic_metadata_path']
            self.topic_count = topic_count
            
            # 註冊處理結果 - 模型
            self.result_manager.register_result(
                self.current_dataset_id,
                "lda_topic",
                "model",
                self.lda_model_path,
                metadata={
                    "n_topics": topic_count,
                    "model_type": "LDA",
                    "data_source": self.data_source
                }
            )
            
            # 註冊處理結果 - 主題詞
            self.result_manager.register_result(
                self.current_dataset_id,
                "lda_topic",
                "data",
                self.topics_path,
                metadata={
                    "n_topics": topic_count,
                    "content": "topic keywords",
                    "data_source": self.data_source
                }
            )
            
            # 註冊處理結果 - 帶主題標籤的元數據
            self.result_manager.register_result(
                self.current_dataset_id,
                "lda_topic",
                "data",
                self.topic_metadata_path,
                metadata={
                    "n_topics": topic_count,
                    "content": "documents with topic labels",
                    "data_source": self.data_source
                }
            )
            
            # 註冊處理結果 - 可視化結果
            for vis_path in results.get('visualizations', []):
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
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"LDA面向切割完成!")
            logger.info(f"LDA模型已保存至: {self.lda_model_path}")
            logger.info(f"主題詞已保存至: {self.topics_path}")
            logger.info(f"帶主題標籤的元數據已保存至: {self.topic_metadata_path}")
            logger.info("====================================")
            
            # 顯示每個主題的頂部詞語
            logger.info("各個主題的頂部詞語:")
            for topic, words in results['topic_words'].items():
                logger.info(f"{topic}: {', '.join(words[:10])}")
            
            logger.info("====================================")
            logger.info(f"處理已完成。窗口將在幾秒後自動關閉。")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 更新UI
            def show_completion():
                try:
                    self.status_var.set(f"LDA面向切割完成! 識別出 {topic_count} 個主題")
                    self.progress_var.set(75)
                    messagebox.showinfo(
                        "成功", 
                        f"LDA面向切割完成！\n已識別出 {topic_count} 個主題\nLDA模型已保存至: {os.path.basename(self.lda_model_path)}"
                    )
                except Exception:
                    pass
            
            self.root.after_idle(show_completion)
            
        except Exception as e:
            import traceback
            error_msg = f"LDA面向切割時發生錯誤: {str(e)}"
            traceback.print_exc()
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            def show_error():
                try:
                    messagebox.showerror("錯誤", error_msg)
                    self.status_var.set("LDA面向切割失敗")
                    self.progress_var.set(60)
                except Exception:
                    pass
            
            self.root.after_idle(show_error)
    
    def calculate_aspect_vectors(self):
        """計算面向相關句子的平均向量"""
        if not self.bert_embeddings_path or not self.topic_metadata_path:
            messagebox.showerror("錯誤", "請先完成BERT語義提取和LDA面向切割步驟!")
            return
        
        # 打開控制台窗口並獲取日誌文件路徑
        log_file, status_file = ConsoleOutputManager.open_console("面向向量計算", auto_close=True)
        
        self.status_var.set("正在計算面向相關句子的平均向量...")
        self.progress_var.set(80)
        
        # 設置日誌器，同時輸出到控制台和日誌文件
        aspect_logger = ConsoleOutputManager.setup_console_logger('aspect_vector_calculation', log_file)
        aspect_logger.info(f"Starting aspect vector calculation, processing embedding file: {self.bert_embeddings_path}")
        aspect_logger.info(f"Topic metadata file: {self.topic_metadata_path}")
        aspect_logger.info("====================================")
        
        # 在另一個線程中執行向量計算
        threading.Thread(target=self._run_vector_calculation, args=(aspect_logger, log_file, status_file), daemon=True).start()
    
    def _run_vector_calculation(self, logger, log_file, status_file):
        """實際執行向量計算的方法"""
        try:
            from src.aspect_vector_calculator import AspectVectorCalculator
            
            # 獲取BERT嵌入路徑和主題元數據路徑
            embeddings_path = self.bert_embeddings_path
            topic_metadata_path = self.topic_metadata_path
            
            # 記錄信息到日誌
            logger.info(f"Processing embedding file: {embeddings_path}")
            logger.info(f"Processing topic metadata: {topic_metadata_path}")
            
            # 創建面向向量計算器
            calculator = AspectVectorCalculator(
                output_dir=self.aspect_vectors_dir,
                logger=logger
            )
            
            # 定義進度回調函數，使用 after 方法確保在主線程中更新 UI
            def update_progress(message, percentage):
                if percentage < 0:
                    self.root.after_idle(lambda: self.status_var.set(f"錯誤: {message}"))
                    return
                    
                def update_ui():
                    try:
                        # 更新進度條和狀態信息
                        self.progress_var.set(80 + (percentage * 20 / 100))
                        self.status_var.set(message)
                    except Exception:
                        pass  # 忽略可能的 Tkinter 錯誤
                
                self.root.after_idle(update_ui)
                logger.info(f"{message} ({percentage}%)")
            
            # 執行面向向量計算
            results = calculator.calculate_aspect_vectors(
                embeddings_path,
                topic_metadata_path,
                callback=update_progress
            )
            
            # 保存結果路徑以供後續步驟使用
            self.aspect_vectors_path = results['aspect_vectors_path']
            self.aspect_metadata_path = results['aspect_metadata_path']
            self.tsne_plot_path = results.get('tsne_plot_path')
            
            # 註冊處理結果 - 面向向量
            self.result_manager.register_result(
                self.current_dataset_id,
                "aspect_vector",
                "data",
                self.aspect_vectors_path,
                metadata={
                    "n_aspects": len(results.get('topics', [])),
                    "embedding_dim": results.get('embedding_dim')
                }
            )
            
            # 註冊處理結果 - 面向元數據
            self.result_manager.register_result(
                self.current_dataset_id,
                "aspect_vector",
                "data",
                self.aspect_metadata_path,
                metadata={
                    "content": "aspect metadata"
                }
            )
            
            # 註冊處理結果 - 可視化結果
            if self.tsne_plot_path:
                self.result_manager.register_result(
                    self.current_dataset_id,
                    "aspect_vector",
                    "visualization",
                    self.tsne_plot_path,
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
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"Aspect vector calculation complete!")
            logger.info(f"Aspect vectors saved to: {self.aspect_vectors_path}")
            logger.info(f"Aspect metadata saved to: {self.aspect_metadata_path}")
            if self.tsne_plot_path:
                logger.info(f"t-SNE visualization saved to: {self.tsne_plot_path}")
            logger.info("====================================")
            
            # 記錄每個面向的文檔數量
            logger.info("Document count for each aspect:")
            for topic, count in results['topic_doc_counts'].items():
                logger.info(f"{topic}: {count} documents")
            
            logger.info("====================================")
            logger.info(f"Processing complete. Window will close automatically in a few seconds.")
            
            # 標記處理完成，觸發控制台自動關閉
            ConsoleOutputManager.mark_process_complete(status_file)
            
            # 更新UI使用 after_idle
            def show_completion():
                try:
                    self.status_var.set(f"面向向量計算完成! 共處理 {len(results['topics'])} 個面向")
                    self.progress_var.set(100)
                    
                    success_msg = f"面向向量計算完成！\n已計算 {len(results['topics'])} 個面向的向量\n面向向量已保存至: {os.path.basename(self.aspect_vectors_path)}"
                    if report_path:
                        success_msg += f"\n\n完整處理報告已生成: {os.path.basename(report_path)}\n是否現在查看報告?"
                        view_report = messagebox.askyesno("成功", success_msg)
                        if view_report:
                            self._open_file(report_path)
                    else:
                        messagebox.showinfo("成功", success_msg)
                except Exception:
                    pass  # 忽略可能的 Tkinter 錯誤
            
            self.root.after_idle(show_completion)
            
        except Exception as e:
            import traceback
            error_msg = f"Error calculating aspect vectors: {str(e)}"
            traceback.print_exc()
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            def show_error():
                try:
                    messagebox.showerror("錯誤", error_msg)
                    self.status_var.set("面向向量計算失敗")
                    self.progress_var.set(80)
                except Exception:
                    pass  # 忽略可能的 Tkinter 錯誤
            
            self.root.after_idle(show_error)

    def export_vectors(self):
        """匯出計算好的平均向量"""
        if not hasattr(self, 'aspect_vectors_path') or not self.aspect_vectors_path:
            messagebox.showerror("錯誤", "請先計算面向向量!")
            return
        
        # 讓用戶選擇導出格式
        formats = {
            "CSV檔案 (*.csv)": "csv",
            "JSON檔案 (*.json)": "json",
            "Pickle檔案 (*.pkl)": "pickle"
        }
        
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
        except ValueError:
            messagebox.showerror("錯誤", "無效的選擇，請輸入1、2或3")
            return
        
        format_list = list(formats.values())
        selected_format = format_list[format_idx - 1]
        
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
        
        # 在另一個線程中執行導出
        self.status_var.set("正在匯出面向向量...")
        threading.Thread(target=self._run_vector_export, args=(file_path, selected_format), daemon=True).start()
    
    def _run_vector_export(self, output_path, output_format):
        """實際執行向量導出的方法"""
        try:
            from src.aspect_vector_calculator import AspectVectorCalculator
            
            calculator = AspectVectorCalculator(output_dir=os.path.dirname(output_path))
            
            def update_progress(message, percentage):
                if percentage < 0:
                    self.root.after_idle(lambda: self.status_var.set(f"錯誤: {message}"))
                    return
                
                self.root.after_idle(lambda: self.status_var.set(message))
            
            result_path = calculator.export_aspect_vectors(
                self.aspect_vectors_path,
                output_format=output_format,
                callback=update_progress
            )
            
            def show_success():
                try:
                    self.status_var.set(f"面向向量已成功匯出至 {os.path.basename(result_path)}")
                    messagebox.showinfo("成功", f"面向向量已成功匯出！\n保存路徑：{result_path}")
                except Exception:
                    pass  # 忽略可能的 Tkinter 錯誤
            
            self.root.after_idle(show_success)
            
        except Exception as e:
            import traceback
            error_msg = f"Error exporting aspect vectors: {str(e)}"
            traceback.print_exc()
            
            def show_error():
                try:
                    messagebox.showerror("錯誤", error_msg)
                    self.status_var.set("匯出面向向量失敗")
                except Exception:
                    pass  # 忽略可能的 Tkinter 錯誤
            
            self.root.after_idle(show_error)
    
    def _open_file(self, file_path):
        """打開文件（使用系統默認應用程序）"""
        try:
            if sys.platform == 'win32':
                os.startfile(file_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', file_path])
            else:  # Linux
                subprocess.Popen(['xdg-open', file_path])
        except Exception as e:
            self.logger.error(f"Error opening file: {str(e)}")
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
                    
                    # 加載已處理的數據路徑
                    for step_name, step_info in dataset_info['steps'].items():
                        if step_name == 'data_import':
                            for result in step_info['results']:
                                if result['type'] == 'data':
                                    self.processed_data_path = result['path']
                                    break
                        
                        elif step_name == 'bert_embedding':
                            for result in step_info['results']:
                                if result['type'] == 'data' and 'embeddings' in result['filename']:
                                    self.bert_embeddings_path = result['path']
                                elif result['type'] == 'data' and 'metadata' in result['filename']:
                                    self.bert_metadata_path = result['path']
                        
                        elif step_name == 'lda_topic':
                            for result in step_info['results']:
                                if result['type'] == 'model':
                                    self.lda_model_path = result['path']
                                elif result['type'] == 'data' and 'topic' in result['filename'] and 'with_topics' in result['filename']:
                                    self.topic_metadata_path = result['path']
                                elif result['type'] == 'data' and 'topics.json' in result['filename']:
                                    self.topics_path = result['path']
                        
                        elif step_name == 'aspect_vector':
                            for result in step_info['results']:
                                if result['type'] == 'data' and 'aspect_vectors' in result['filename']:
                                    self.aspect_vectors_path = result['path']
                                elif result['type'] == 'data' and 'aspect_metadata' in result['filename']:
                                    self.aspect_metadata_path = result['path']
                                elif result['type'] == 'visualization' and 'tsne' in result['filename']:
                                    self.tsne_plot_path = result['path']
                    
                    # 更新狀態
                    self.status_var.set(f"已加載最近的分析結果: {dataset_info['name']}")
                    self.logger.info(f"Recent analysis results loaded: {dataset_id} ({dataset_info['name']})")
                    
                    # 更新文件路徑顯示
                    if dataset_info.get('source_path'):
                        self.file_path.set(dataset_info['source_path'])
        
        except Exception as e:
            self.logger.error(f"Error loading recent analysis results: {str(e)}")

# 啟動應用程式
if __name__ == "__main__":
    root = tk.Tk()
    app = CrossDomainSentimentAnalysisApp(root)
    root.update()
    # 根據操作系統設置窗口最大化
    import platform
    system = platform.system()
    if system == "Windows":
        root.state('zoomed')
    root.mainloop()