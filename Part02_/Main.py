import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import os
import sys
import logging
import nltk
from console_output import ConsoleOutputManager

class CrossDomainSentimentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("跨領域情感分析系統 v2.0")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
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
        
        # 新增控制台輸出處理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # 控制台只顯示警告及以上級別
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # 添加處理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 確保NLTK資源已下載
        try:
            self._ensure_nltk_resources()
        except Exception as e:
            messagebox.showerror("錯誤", f"無法下載NLTK資源: {str(e)}")
            self.logger.error(f"NLTK資源下載失敗: {str(e)}")
            root.destroy()
            return
        
        # 創建變數
        self.file_path = tk.StringVar()
        self.topic_count = tk.StringVar(value="10")  # 預設LDA主題數量
        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="準備就緒。")
        
        # 數據路徑
        self.processed_data_path = None
        self.bert_embeddings_path = None
        self.lda_model_path = None
        self.aspect_vectors_path = None
        
        # 確保數據目錄存在
        self.data_dir = "./Part02_/data"
        self.results_dir = "./Part02_/results"
        
        for directory in [self.data_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # 建立介面
        self._create_widgets()
        
        self.logger.info("應用程序已啟動")
        
    def _create_widgets(self):
        # 建立分頁
        self.tab_control = ttk.Notebook(self.root)
        
        # 建立各個分頁
        self.tab_data_processing = ttk.Frame(self.tab_control)
        self.tab_model_training = ttk.Frame(self.tab_control)
        self.tab_evaluation = ttk.Frame(self.tab_control)
        self.tab_visualization = ttk.Frame(self.tab_control)
        
        # 添加分頁到控制器
        self.tab_control.add(self.tab_data_processing, text="資料處理")
        self.tab_control.add(self.tab_model_training, text="模型訓練")
        self.tab_control.add(self.tab_evaluation, text="評估分析")
        self.tab_control.add(self.tab_visualization, text="結果可視化")
        
        self.tab_control.pack(expand=1, fill="both")
        
        # 設置資料處理分頁內容
        self._setup_data_processing_tab()
        
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
        
        ttk.Label(step3_frame, text="使用LDA進行主題建模，識別不同面向").pack(anchor="w", pady=5)
        
        lda_frame = ttk.Frame(step3_frame)
        lda_frame.pack(fill="x", pady=5)
        
        ttk.Label(lda_frame, text="主題數量:").pack(side="left")
        ttk.Entry(lda_frame, textvariable=self.topic_count, width=5).pack(side="left", padx=5)
        ttk.Button(lda_frame, text="執行LDA面向切割", command=self.perform_lda).pack(side="right")
        
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
            self.logger.info(f"已選擇文件: {file_path}")
    
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
            from data_importer import DataImporter
            
            file_path = self.file_path.get()
            
            # 建立一個數據導入器
            importer = DataImporter()
            
            # 定義進度回調函數
            def update_progress(message, percentage):
                if percentage < 0:  # 錯誤情況
                    self.root.after(0, lambda: self.status_var.set(f"錯誤: {message}"))
                    return
                    
                # 更新進度條和狀態信息
                progress = 20 + (percentage * 0.8 / 100)  # 將百分比轉換到20%-100%範圍
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.status_var.set(message))
                
                # 在控制台輸出日誌
                print(f"{message} ({percentage}%)")
            
            # 執行數據導入
            processed_file_path = importer.import_data(file_path, callback=update_progress)
            
            # 保存處理後的文件路徑以供後續步驟使用
            self.processed_data_path = processed_file_path
            
            # 更新UI
            self.root.after(0, lambda: self.status_var.set(f"數據導入完成: {os.path.basename(processed_file_path)}"))
            self.root.after(0, lambda: self.progress_var.set(25))
            
            # 顯示成功信息
            self.root.after(0, lambda: messagebox.showinfo("成功", "數據已成功導入和初步處理!"))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"導入數據時發生錯誤: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("導入數據失敗"))
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def extract_bert_embeddings(self):
        """執行BERT語義提取"""
        if not self.processed_data_path:
            messagebox.showerror("錯誤", "請先導入並處理數據!")
            return
        
        # 打開控制台窗口並獲取日誌文件路徑
        log_file = ConsoleOutputManager.open_console("BERT語義提取")
        
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
        threading.Thread(target=self._run_bert_extraction, args=(bert_logger, log_file), daemon=True).start()
    
    def _run_bert_extraction(self, logger, log_file):
        """實際執行BERT提取的方法"""
        try:
            from bert_embedder import BertEmbedder
            
            # 獲取處理後的數據路徑
            data_path = self.processed_data_path
            
            # 記錄信息到日誌
            logger.info(f"Processing file: {data_path}")
            
            # 創建BERT編碼器
            embedder = BertEmbedder(
                model_name='bert-base-uncased',
                output_dir=self.results_dir
            )
            
            # 定義進度回調函數
            def update_progress(message, percentage):
                if percentage < 0:  # 錯誤情況
                    self.root.after(0, lambda: self.status_var.set(f"錯誤: {message}"))
                    return
                    
                # 更新進度條和狀態信息
                progress = 30 + (percentage * 20 / 100)  # 將百分比轉換到30%-50%範圍
                self.root.after(0, lambda: self.progress_var.set(progress))
                self.root.after(0, lambda: self.status_var.set(message))
                
                # 記錄日誌
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
            
            # 記錄完成信息
            logger.info("====================================")
            logger.info(f"BERT語義提取完成!")
            logger.info(f"嵌入向量已保存至: {self.bert_embeddings_path}")
            logger.info(f"元數據已保存至: {self.bert_metadata_path}")
            logger.info(f"嵌入維度: {self.embedding_dim}")
            
            # 更新UI
            self.root.after(0, lambda: self.status_var.set(f"BERT語義提取完成！嵌入維度: {self.embedding_dim}"))
            self.root.after(0, lambda: self.progress_var.set(50))
            
            # 顯示成功信息
            self.root.after(0, lambda: messagebox.showinfo("成功", f"BERT語義提取完成！\n嵌入向量已保存至: {os.path.basename(self.bert_embeddings_path)}\n嵌入維度: {self.embedding_dim}"))
            
        except Exception as e:
            import traceback
            error_msg = f"BERT語義提取時發生錯誤: {str(e)}"
            traceback.print_exc()
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
            self.root.after(0, lambda: self.status_var.set("BERT語義提取失敗"))
            self.root.after(0, lambda: self.progress_var.set(30))
    
    def perform_lda(self):
        """執行LDA面向切割"""
        try:
            topic_count = int(self.topic_count.get())
        except ValueError:
            messagebox.showerror("錯誤", "主題數量必須是有效的整數!")
            return
        
        self.status_var.set("正在進行LDA面向切割...")
        self.progress_var.set(60)
        
        # 啟動LDA處理
        threading.Thread(target=self._run_lda, args=(topic_count,), daemon=True).start()
    
    def _run_lda(self, topic_count):
        """實際執行LDA的方法"""
        try:
            print(f"開始進行LDA面向切割，主題數量: {topic_count}")
            
            # 模擬處理過程
            import time
            time.sleep(3)
            
            # 更新UI
            self.root.after(0, lambda: self.status_var.set("LDA面向切割完成"))
            self.root.after(0, lambda: self.progress_var.set(75))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"LDA面向切割時發生錯誤: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("LDA面向切割失敗"))
    
    def calculate_aspect_vectors(self):
        """計算面向相關句子的平均向量"""
        self.status_var.set("正在計算面向相關句子的平均向量...")
        self.progress_var.set(90)
        
        # 啟動向量計算
        threading.Thread(target=self._run_vector_calculation, daemon=True).start()
    
    def _run_vector_calculation(self):
        """實際執行向量計算的方法"""
        try:
            print("開始計算面向相關句子的平均向量")
            
            # 模擬處理過程
            import time
            time.sleep(2)
            
            # 更新UI
            self.root.after(0, lambda: self.status_var.set("面向向量計算完成"))
            self.root.after(0, lambda: self.progress_var.set(100))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"面向向量計算時發生錯誤: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("面向向量計算失敗"))
    
    def export_vectors(self):
        """匯出計算好的平均向量"""
        file_path = filedialog.asksaveasfilename(
            title="保存平均向量",
            defaultextension=".npy",
            filetypes=[("NumPy Files", "*.npy"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set("正在匯出平均向量...")
                
                # 在實際應用中，這裡應該實現將向量數據保存到文件
                print(f"匯出平均向量到: {file_path}")
                
                # 模擬處理過程
                import time
                time.sleep(1)
                
                self.status_var.set(f"平均向量已成功匯出至 {os.path.basename(file_path)}")
                messagebox.showinfo("成功", "平均向量已成功匯出!")
            except Exception as e:
                messagebox.showerror("錯誤", f"匯出平均向量時發生錯誤: {str(e)}")
                self.status_var.set("匯出平均向量失敗")
    
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
                self.logger.info(f"NLTK資源 '{resource}' 已存在")
            except LookupError:
                self.logger.info(f"正在下載NLTK資源 '{resource}'...")
                try:
                    nltk.download(resource, download_dir=nltk_data_path, quiet=False)
                    self.logger.info(f"NLTK資源 '{resource}' 下載完成")
                except Exception as e:
                    self.logger.error(f"下載NLTK資源 '{resource}' 時出錯: {str(e)}")
                    raise

# 啟動應用程式
if __name__ == "__main__":
    root = tk.Tk()
    app = CrossDomainSentimentAnalysisApp(root)
    root.mainloop()