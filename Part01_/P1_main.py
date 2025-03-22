import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import re
import jieba
import aspect_extraction as ae
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# 確保必要的NLTK資源已下載
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class DataPreprocessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Part-01_跨領域情感分析資料前處理工具 v1.5")
        self.root.geometry("1200x800")
        
        # 最大化應用程序窗口 Windows適用
        self.root.state('zoomed')

        # 設置資料存儲變數
        self.amazon_data = None
        self.yelp_data = None
        self.imdb_data = None
        
        # 建立頁籤控制
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 創建三個數據集的頁籤
        self.amazon_frame = ttk.Frame(self.notebook)
        self.yelp_frame = ttk.Frame(self.notebook)
        self.imdb_frame = ttk.Frame(self.notebook)
        self.stats_frame = ttk.Frame(self.notebook)
        self.preprocess_frame = ttk.Frame(self.notebook)
        self.aspect_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.amazon_frame, text="Amazon數據")
        self.notebook.add(self.yelp_frame, text="Yelp數據")
        self.notebook.add(self.imdb_frame, text="IMDB數據")
        self.notebook.add(self.stats_frame, text="數據統計")
        self.notebook.add(self.preprocess_frame, text="數據預處理")
        self.notebook.add(self.aspect_frame, text="面相切割")
        
        # 設置各頁籤的內容
        self._setup_amazon_tab()
        self._setup_yelp_tab()
        self._setup_imdb_tab()
        self._setup_stats_tab()
        self._setup_preprocess_tab()
        self._setup_aspect_tab()
        
        # 顯示狀態欄
        self.status_var = tk.StringVar()
        self.status_var.set("就緒")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_amazon_tab(self):
        # 頂部框架
        top_frame = ttk.Frame(self.amazon_frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        # 添加載入數據按鈕
        load_btn = ttk.Button(top_frame, text="載入Amazon數據", command=self.load_amazon_data)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加下拉選單選擇數據類別
        ttk.Label(top_frame, text="數據類別:").pack(side=tk.LEFT, padx=5)
        self.amazon_category = ttk.Combobox(top_frame, values=["Books", "Electronics", "Movies_and_TV", "Clothing", "Home_and_Kitchen"])
        self.amazon_category.pack(side=tk.LEFT, padx=5)
        self.amazon_category.current(0)
        
        # 中間框架 - 顯示數據預覽
        mid_frame = ttk.LabelFrame(self.amazon_frame, text="數據預覽")
        mid_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 創建Treeview來顯示數據
        columns = ('index', 'review', 'rating', 'category')
        self.amazon_tree = ttk.Treeview(mid_frame, columns=columns, show='headings')
        
        # 設置列標題
        for col in columns:
            self.amazon_tree.heading(col, text=col.capitalize())
            self.amazon_tree.column(col, width=100)
        
        # 特別設置review列寬度
        self.amazon_tree.column('review', width=500)
        
        # 添加滾動條
        scrollbar_y = ttk.Scrollbar(mid_frame, orient=tk.VERTICAL, command=self.amazon_tree.yview)
        scrollbar_x = ttk.Scrollbar(mid_frame, orient=tk.HORIZONTAL, command=self.amazon_tree.xview)
        self.amazon_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 放置元件
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.amazon_tree.pack(fill='both', expand=True)
        
        # 底部框架 - 顯示數據統計
        bottom_frame = ttk.LabelFrame(self.amazon_frame, text="數據統計")
        bottom_frame.pack(fill='x', padx=10, pady=10)
        
        # 添加統計信息標籤
        self.amazon_stats_label = ttk.Label(bottom_frame, text="未載入數據")
        self.amazon_stats_label.pack(padx=10, pady=10)
    
    def _setup_yelp_tab(self):
        # 類似Amazon標籤的設置
        top_frame = ttk.Frame(self.yelp_frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        load_btn = ttk.Button(top_frame, text="載入Yelp數據", command=self.load_yelp_data)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_frame, text="數據類別:").pack(side=tk.LEFT, padx=5)
        self.yelp_category = ttk.Combobox(top_frame, values=["Restaurant", "Hotel", "Shopping", "Beauty", "Nightlife"])
        self.yelp_category.pack(side=tk.LEFT, padx=5)
        self.yelp_category.current(0)
        
        mid_frame = ttk.LabelFrame(self.yelp_frame, text="數據預覽")
        mid_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('index', 'review', 'rating', 'business_id')
        self.yelp_tree = ttk.Treeview(mid_frame, columns=columns, show='headings')
        
        for col in columns:
            self.yelp_tree.heading(col, text=col.capitalize())
            self.yelp_tree.column(col, width=100)
        
        self.yelp_tree.column('review', width=500)
        
        scrollbar_y = ttk.Scrollbar(mid_frame, orient=tk.VERTICAL, command=self.yelp_tree.yview)
        scrollbar_x = ttk.Scrollbar(mid_frame, orient=tk.HORIZONTAL, command=self.yelp_tree.xview)
        self.yelp_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.yelp_tree.pack(fill='both', expand=True)
        
        bottom_frame = ttk.LabelFrame(self.yelp_frame, text="數據統計")
        bottom_frame.pack(fill='x', padx=10, pady=10)
        
        self.yelp_stats_label = ttk.Label(bottom_frame, text="未載入數據")
        self.yelp_stats_label.pack(padx=10, pady=10)
    
    def _setup_imdb_tab(self):
        # 類似Amazon標籤的設置
        top_frame = ttk.Frame(self.imdb_frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        load_btn = ttk.Button(top_frame, text="載入IMDB數據", command=self.load_imdb_data)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_frame, text="數據類型:").pack(side=tk.LEFT, padx=5)
        self.imdb_type = ttk.Combobox(top_frame, values=["Positive", "Negative", "All"])
        self.imdb_type.pack(side=tk.LEFT, padx=5)
        self.imdb_type.current(2)
        
        mid_frame = ttk.LabelFrame(self.imdb_frame, text="數據預覽")
        mid_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        columns = ('index', 'review', 'sentiment')
        self.imdb_tree = ttk.Treeview(mid_frame, columns=columns, show='headings')
        
        for col in columns:
            self.imdb_tree.heading(col, text=col.capitalize())
            self.imdb_tree.column(col, width=100)
        
        self.imdb_tree.column('review', width=600)
        
        scrollbar_y = ttk.Scrollbar(mid_frame, orient=tk.VERTICAL, command=self.imdb_tree.yview)
        scrollbar_x = ttk.Scrollbar(mid_frame, orient=tk.HORIZONTAL, command=self.imdb_tree.xview)
        self.imdb_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.imdb_tree.pack(fill='both', expand=True)
        
        bottom_frame = ttk.LabelFrame(self.imdb_frame, text="數據統計")
        bottom_frame.pack(fill='x', padx=10, pady=10)
        
        self.imdb_stats_label = ttk.Label(bottom_frame, text="未載入數據")
        self.imdb_stats_label.pack(padx=10, pady=10)
    
    def _setup_stats_tab(self):
        # 設置統計分析標籤
        controls_frame = ttk.Frame(self.stats_frame)
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(controls_frame, text="選擇分析類型:").pack(side=tk.LEFT, padx=5)
        
        # 定義分析類型及其說明
        self.analysis_descriptions = {
            "情感分布": "比較不同數據集的情感評分分布",
            "評論長度分布": "分析各數據集的文本長度分布",
            "詞頻分析": "顯示各數據集中最常見的詞語",
            "跨領域詞彙重疊": "計算不同領域間詞彙的重疊度，以Jaccard相似度展示",
            "情感詞分布": "利用VADER情感分析工具分析不同數據集中的情感詞分布"
        }
        
        self.analysis_type = ttk.Combobox(controls_frame, 
                                         values=list(self.analysis_descriptions.keys()))
        self.analysis_type.pack(side=tk.LEFT, padx=5)
        self.analysis_type.current(0)
        
        # 添加說明標籤
        self.description_label = ttk.Label(controls_frame, 
                                         text=self.analysis_descriptions[self.analysis_type.get()],
                                         wraplength=400)
        self.description_label.pack(side=tk.LEFT, padx=20)
        
        # 綁定選擇變更事件
        self.analysis_type.bind('<<ComboboxSelected>>', self._update_analysis_description)
        
        analyze_btn = ttk.Button(controls_frame, text="執行分析", command=self.run_analysis)
        analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # 圖表顯示區域
        self.chart_frame = ttk.LabelFrame(self.stats_frame, text="分析結果")
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 統計信息顯示區域
        self.stats_info_frame = ttk.LabelFrame(self.stats_frame, text="統計信息")
        self.stats_info_frame.pack(fill='x', padx=10, pady=10)
        
        self.stats_info_label = ttk.Label(self.stats_info_frame, text="請選擇分析類型並點擊「執行分析」")
        self.stats_info_label.pack(padx=10, pady=10)
    
    def _update_analysis_description(self, event):
        """更新分析類型的說明文字"""
        selected_type = self.analysis_type.get()
        self.description_label.config(text=self.analysis_descriptions[selected_type])
    
    def _setup_preprocess_tab(self):
        # 設置預處理標籤
        controls_frame = ttk.Frame(self.preprocess_frame)
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # 選擇數據源
        ttk.Label(controls_frame, text="選擇數據源:").pack(side=tk.LEFT, padx=5)
        self.data_source = ttk.Combobox(controls_frame, values=["Amazon", "Yelp", "IMDB", "全部"])
        self.data_source.pack(side=tk.LEFT, padx=5)
        self.data_source.current(3)
        
        # 預處理選項框架
        options_frame = ttk.LabelFrame(self.preprocess_frame, text="預處理選項")
        options_frame.pack(fill='x', padx=10, pady=10)
        
        # 建立預處理選項的變數
        self.clean_html_var = tk.BooleanVar(value=True)
        self.remove_punctuation_var = tk.BooleanVar(value=True)
        self.lowercase_var = tk.BooleanVar(value=True)
        self.remove_stopwords_var = tk.BooleanVar(value=True)
        self.stemming_var = tk.BooleanVar(value=False)
        self.lemmatization_var = tk.BooleanVar(value=True)
        self.balance_data_var = tk.BooleanVar(value=True)
        
        # 添加選項到框架
        ttk.Checkbutton(options_frame, text="清除HTML標籤", variable=self.clean_html_var).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="移除標點符號", variable=self.remove_punctuation_var).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="轉為小寫", variable=self.lowercase_var).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="移除停用詞", variable=self.remove_stopwords_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="詞幹提取", variable=self.stemming_var).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="詞形還原", variable=self.lemmatization_var).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(options_frame, text="平衡數據（正負樣本）", variable=self.balance_data_var).grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        
        # 添加「執行預處理」和「儲存預處理結果」按鈕
        button_frame = ttk.Frame(self.preprocess_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="執行預處理", command=self.run_preprocessing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="批次預處理（全量資料）", command=self.run_batch_preprocessing).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="儲存預處理結果", command=self.save_preprocessed_data).pack(side=tk.LEFT, padx=5)
        
        # 預處理結果預覽
        preview_frame = ttk.LabelFrame(self.preprocess_frame, text="預處理結果預覽")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 創建Treeview來顯示預處理結果
        columns = ('index', 'original', 'preprocessed', 'source', 'label')
        self.preprocess_tree = ttk.Treeview(preview_frame, columns=columns, show='headings')
        
        # 設置列標題
        for col in columns:
            self.preprocess_tree.heading(col, text=col.capitalize())
            self.preprocess_tree.column(col, width=100)
        
        # 特別設置文本列寬度
        self.preprocess_tree.column('original', width=300)
        self.preprocess_tree.column('preprocessed', width=300)
        
        # 添加滾動條
        scrollbar_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preprocess_tree.yview)
        scrollbar_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.preprocess_tree.xview)
        self.preprocess_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # 放置元件
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.preprocess_tree.pack(fill='both', expand=True)
        
        # 狀態顯示
        self.preprocess_status_var = tk.StringVar()
        self.preprocess_status_var.set("請選擇預處理選項並點擊「執行預處理」")
        status_label = ttk.Label(self.preprocess_frame, textvariable=self.preprocess_status_var)
        status_label.pack(padx=10, pady=10)
    
    def load_amazon_data(self):
        # 實際應用中，應從文件加載數據
        self.status_var.set("正在載入Amazon數據...")
        try:
            # 選擇檔案
            file_path = filedialog.askopenfilename(
                title="選擇Amazon數據檔案",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            if not file_path:
                self.status_var.set("未選擇檔案")
                return
            
            # 儲存檔案路徑以供批次處理使用
            self.amazon_file_path = file_path
            
            # 載入數據
            self.amazon_data = pd.read_csv(file_path)
            
            # 清空現有樹內容
            for item in self.amazon_tree.get_children():
                self.amazon_tree.delete(item)
            
            # 填充樹視圖
            for idx, row in self.amazon_data.head(100).iterrows():
                try:
                    # 嘗試獲取必要的列，不同數據集可能有不同的列名
                    review = str(row.get('reviewText', row.get('review', "無評論內容")))
                    rating = str(row.get('overall', row.get('rating', "無評分")))
                    category = str(row.get('category', self.amazon_category.get()))
                    
                    self.amazon_tree.insert('', 'end', values=(idx, review[:100] + "...", rating, category))
                except Exception as e:
                    print(f"Error inserting row {idx}: {e}")
            
            # 更新統計信息
            stats_text = f"共載入 {len(self.amazon_data)} 條Amazon評論\n"
            stats_text += f"正面評論: {sum(self.amazon_data.get('overall', self.amazon_data.get('rating', 0)) > 3)}\n"
            stats_text += f"負面評論: {sum(self.amazon_data.get('overall', self.amazon_data.get('rating', 0)) < 3)}"
            self.amazon_stats_label.config(text=stats_text)
            
            self.status_var.set(f"已成功載入 {len(self.amazon_data)} 條Amazon評論數據")
            
        except Exception as e:
            self.status_var.set(f"載入Amazon數據時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"載入Amazon數據時出錯: {str(e)}")
    
    def load_yelp_data(self):
        # 與load_amazon_data類似的實現
        self.status_var.set("正在載入Yelp數據...")
        try:
            file_path = filedialog.askopenfilename(
                title="選擇Yelp數據檔案",
                filetypes=[("JSON Files", "*.json"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            if not file_path:
                self.status_var.set("未選擇檔案")
                return
            
            # 儲存檔案路徑以供批次處理使用
            self.yelp_file_path = file_path
            
            # 根據檔案類型載入
            if file_path.endswith('.json'):
                self.yelp_data = pd.read_json(file_path, lines=True)
            else:
                self.yelp_data = pd.read_csv(file_path)
            
            # 清空現有樹內容
            for item in self.yelp_tree.get_children():
                self.yelp_tree.delete(item)
            
            # 填充樹視圖
            for idx, row in self.yelp_data.head(100).iterrows():
                try:
                    review = str(row.get('text', row.get('review', "無評論內容")))
                    rating = str(row.get('stars', row.get('rating', "無評分")))
                    business_id = str(row.get('business_id', "無商家ID"))
                    
                    self.yelp_tree.insert('', 'end', values=(idx, review[:100] + "...", rating, business_id))
                except Exception as e:
                    print(f"Error inserting row {idx}: {e}")
            
            # 更新統計信息
            stats_text = f"共載入 {len(self.yelp_data)} 條Yelp評論\n"
            stats_text += f"正面評論: {sum(self.yelp_data.get('stars', self.yelp_data.get('rating', 0)) > 3)}\n"
            stats_text += f"負面評論: {sum(self.yelp_data.get('stars', self.yelp_data.get('rating', 0)) < 3)}"
            self.yelp_stats_label.config(text=stats_text)
            
            self.status_var.set(f"已成功載入 {len(self.yelp_data)} 條Yelp評論數據")
            
        except Exception as e:
            self.status_var.set(f"載入Yelp數據時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"載入Yelp數據時出錯: {str(e)}")
    
    def load_imdb_data(self):
        # 與load_amazon_data類似的實現
        self.status_var.set("正在載入IMDB數據...")
        try:
            file_path = filedialog.askopenfilename(
                title="選擇IMDB數據檔案",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            
            if not file_path:
                self.status_var.set("未選擇檔案")
                return
            
            # 儲存檔案路徑以供批次處理使用
            self.imdb_file_path = file_path
            
            self.imdb_data = pd.read_csv(file_path)
            
            # 清空現有樹內容
            for item in self.imdb_tree.get_children():
                self.imdb_tree.delete(item)
            
            # 填充樹視圖
            for idx, row in self.imdb_data.head(100).iterrows():
                try:
                    review = str(row.get('review', row.get('text', "無評論內容")))
                    sentiment = str(row.get('sentiment', row.get('label', "無情感標籤")))
                    
                    self.imdb_tree.insert('', 'end', values=(idx, review[:100] + "...", sentiment))
                except Exception as e:
                    print(f"Error inserting row {idx}: {e}")
            
            # 更新統計信息
            sentiment_col = 'sentiment' if 'sentiment' in self.imdb_data.columns else 'label'
            stats_text = f"共載入 {len(self.imdb_data)} 條IMDB評論\n"
            
            if sentiment_col in self.imdb_data.columns:
                positive_count = sum(self.imdb_data[sentiment_col] == 'positive')
                negative_count = sum(self.imdb_data[sentiment_col] == 'negative')
                stats_text += f"正面評論: {positive_count}\n"
                stats_text += f"負面評論: {negative_count}"
            
            self.imdb_stats_label.config(text=stats_text)
            
            self.status_var.set(f"已成功載入 {len(self.imdb_data)} 條IMDB評論數據")
            
        except Exception as e:
            self.status_var.set(f"載入IMDB數據時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"載入IMDB數據時出錯: {str(e)}")
    
    def run_analysis(self):
        # 執行選定的數據分析
        analysis_type = self.analysis_type.get()
        self.status_var.set(f"正在執行{analysis_type}分析...")
        
        # 清除現有圖表
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        try:
            # 根據選擇的分析類型執行不同的分析
            if analysis_type == "情感分布":
                self._analyze_sentiment_distribution()
            elif analysis_type == "評論長度分布":
                self._analyze_review_length()
            elif analysis_type == "詞頻分析":
                self._analyze_word_frequency()
            elif analysis_type == "跨領域詞彙重疊":
                self._analyze_vocabulary_overlap()
            elif analysis_type == "情感詞分布":
                self._analyze_sentiment_words()
            
            self.status_var.set(f"已完成{analysis_type}分析")
        except Exception as e:
            self.status_var.set(f"執行分析時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"執行分析時出錯: {str(e)}")
    
    def _analyze_sentiment_distribution(self):
        # 分析各數據集的情感分布
        fig, ax = plt.subplots(figsize=(10, 6))
        data_sets = []
        labels = []
        
        # 收集各數據集的情感分布
        if self.amazon_data is not None:
            amazon_ratings = self.amazon_data.get('overall', self.amazon_data.get('rating', None))
            if amazon_ratings is not None:
                data_sets.append(amazon_ratings)
                labels.append('Amazon')
        
        if self.yelp_data is not None:
            yelp_ratings = self.yelp_data.get('stars', self.yelp_data.get('rating', None))
            if yelp_ratings is not None:
                data_sets.append(yelp_ratings)
                labels.append('Yelp')
        
        if self.imdb_data is not None:
            sentiment_col = 'sentiment' if 'sentiment' in self.imdb_data.columns else 'label'
            if sentiment_col in self.imdb_data.columns:
                # 將文本情感標籤轉換為數值
                sentiment_map = {'positive': 5, 'negative': 1}
                imdb_ratings = self.imdb_data[sentiment_col].map(sentiment_map)
                data_sets.append(imdb_ratings)
                labels.append('IMDB')
        
        # 繪製箱型圖
        if data_sets:
            ax.boxplot(data_sets, labels=labels)
            ax.set_title('Sentiment Distribution Across Datasets')
            ax.set_ylabel('Sentiment Score')
            
            # 更新統計信息
            stats_text = "Sentiment Distribution Statistics:\n"
            for i, (data, label) in enumerate(zip(data_sets, labels)):
                stats_text += f"{label}: 平均評分 = {data.mean():.2f}, 標準差 = {data.std():.2f}\n"
            
            self.stats_info_label.config(text=stats_text)
            
            # 將圖表添加到GUI
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.chart_frame, text="未載入足夠的數據進行分析").pack(padx=20, pady=20)
    
    def _analyze_review_length(self):
        # 分析各數據集的評論長度分布
        fig, ax = plt.subplots(figsize=(10, 6))
        data_sets = []
        labels = []
        
        # 獲取各數據集的評論長度
        if self.amazon_data is not None:
            review_col = 'reviewText' if 'reviewText' in self.amazon_data.columns else 'review'
            if review_col in self.amazon_data.columns:
                amazon_lengths = self.amazon_data[review_col].astype(str).apply(len)
                data_sets.append(amazon_lengths)
                labels.append('Amazon')
        
        if self.yelp_data is not None:
            review_col = 'text' if 'text' in self.yelp_data.columns else 'review'
            if review_col in self.yelp_data.columns:
                yelp_lengths = self.yelp_data[review_col].astype(str).apply(len)
                data_sets.append(yelp_lengths)
                labels.append('Yelp')
        
        if self.imdb_data is not None:
            review_col = 'review' if 'review' in self.imdb_data.columns else 'text'
            if review_col in self.imdb_data.columns:
                imdb_lengths = self.imdb_data[review_col].astype(str).apply(len)
                data_sets.append(imdb_lengths)
                labels.append('IMDB')
        
        # 繪製直方圖
        if data_sets:
            for i, (data, label) in enumerate(zip(data_sets, labels)):
                ax.hist(data, bins=30, alpha=0.5, label=label)
            
            ax.set_title('Review Length Distribution')
            ax.set_xlabel('Review Character Length')
            ax.set_ylabel('Frequency')
            ax.legend()
            
            # 限制x軸範圍，避免極端值影響可視化
            ax.set_xlim(0, np.percentile(np.concatenate(data_sets), 95))
            
            # 更新統計信息
            stats_text = "Review Length Statistics:\n"
            for i, (data, label) in enumerate(zip(data_sets, labels)):
                stats_text += f"{label}: 平均長度 = {data.mean():.2f}, 中位數 = {data.median():.2f}\n"
                stats_text += f"     最短評論 = {data.min()}, 最長評論 = {data.max()}\n"
            
            self.stats_info_label.config(text=stats_text)
            
            # 將圖表添加到GUI
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.chart_frame, text="未載入足夠的數據進行分析").pack(padx=20, pady=20)
    
    def _analyze_word_frequency(self):
        # 分析各數據集中的詞頻
        n_datasets = sum(1 for data in [self.amazon_data, self.yelp_data, self.imdb_data] if data is not None)
        if n_datasets == 0:
            ttk.Label(self.chart_frame, text="未載入任何數據進行分析").pack(padx=20, pady=20)
            return
            
        fig, axs = plt.subplots(1, n_datasets, figsize=(15, 6))
        if n_datasets == 1:
            axs = [axs]  # 確保axs是列表
            
        all_texts = {}
        
        # 獲取各數據集的文本
        if self.amazon_data is not None:
            review_col = 'reviewText' if 'reviewText' in self.amazon_data.columns else 'review'
            if review_col in self.amazon_data.columns:
                all_texts['Amazon'] = ' '.join(self.amazon_data[review_col].astype(str))
        
        if self.yelp_data is not None:
            review_col = 'text' if 'text' in self.yelp_data.columns else 'review'
            if review_col in self.yelp_data.columns:
                all_texts['Yelp'] = ' '.join(self.yelp_data[review_col].astype(str))
        
        if self.imdb_data is not None:
            review_col = 'review' if 'review' in self.imdb_data.columns else 'text'
            if review_col in self.imdb_data.columns:
                all_texts['IMDB'] = ' '.join(self.imdb_data[review_col].astype(str))
        
        # 分析詞頻
        if all_texts:
            # 獲取停用詞
            stop_words = set(stopwords.words('english'))
            
            stats_text = "Word Frequency Statistics:\n"
            
            for i, (dataset, text) in enumerate(all_texts.items()):
                # 簡單的文本清理
                text = text.lower()
                text = re.sub(r'[^\w\s]', '', text)
                words = text.split()
                words = [word for word in words if word not in stop_words and len(word) > 1]
                
                # 計算詞頻
                word_freq = {}
                for word in words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
                
                # 獲取前20個高頻詞
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                
                # 繪製詞頻條形圖
                if top_words:  # 確保有數據
                    words, freqs = zip(*top_words)
                    axs[i].barh(words, freqs)
                    axs[i].set_title(f'{dataset} Top Words')
                    axs[i].set_xlabel('Frequency')
                
                    # 添加統計信息
                    stats_text += f"{dataset}: 總詞數 = {len(words)}, 唯一詞數 = {len(word_freq)}\n"
                    stats_text += f"     前5高頻詞: {', '.join([w for w, f in top_words[:5]])}\n"
            
            # 調整佈局
            plt.tight_layout()
            
            # 更新統計信息
            self.stats_info_label.config(text=stats_text)
            
            # 將圖表添加到GUI
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.chart_frame, text="未載入足夠的數據進行分析").pack(padx=20, pady=20)
            
    def _analyze_vocabulary_overlap(self):
        # 分析數據集之間的詞彙重疊情況
        datasets_vocab = {}
        
        # 載入各數據集的詞彙
        if self.amazon_data is not None:
            review_col = 'reviewText' if 'reviewText' in self.amazon_data.columns else 'review'
            if review_col in self.amazon_data.columns:
                text = ' '.join(self.amazon_data[review_col].astype(str))
                words = set(re.findall(r'\w+', text.lower()))
                datasets_vocab['Amazon'] = words
        
        if self.yelp_data is not None:
            review_col = 'text' if 'text' in self.yelp_data.columns else 'review'
            if review_col in self.yelp_data.columns:
                text = ' '.join(self.yelp_data[review_col].astype(str))
                words = set(re.findall(r'\w+', text.lower()))
                datasets_vocab['Yelp'] = words
        
        if self.imdb_data is not None:
            review_col = 'review' if 'review' in self.imdb_data.columns else 'text'
            if review_col in self.imdb_data.columns:
                text = ' '.join(self.imdb_data[review_col].astype(str))
                words = set(re.findall(r'\w+', text.lower()))
                datasets_vocab['IMDB'] = words
        
        # 至少需要兩個數據集才能進行比較
        if len(datasets_vocab) >= 2:
            datasets = list(datasets_vocab.keys())
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 創建重疊矩陣
            overlap_matrix = np.zeros((len(datasets), len(datasets)))
            
            for i, dataset1 in enumerate(datasets):
                for j, dataset2 in enumerate(datasets):
                    if i == j:
                        overlap_matrix[i, j] = 1.0
                    else:
                        intersection = len(datasets_vocab[dataset1].intersection(datasets_vocab[dataset2]))
                        union = len(datasets_vocab[dataset1].union(datasets_vocab[dataset2]))
                        overlap_matrix[i, j] = intersection / union if union > 0 else 0
            
            # 繪製熱力圖
            im = ax.imshow(overlap_matrix, cmap='Blues')
            ax.set_xticks(np.arange(len(datasets)))
            ax.set_yticks(np.arange(len(datasets)))
            ax.set_xticklabels(datasets)
            ax.set_yticklabels(datasets)
            
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Jaccard Similarity')
            
            for i in range(len(datasets)):
                for j in range(len(datasets)):
                    text = ax.text(j, i, f"{overlap_matrix[i, j]:.2f}",
                                 ha="center", va="center",
                                 color="black" if overlap_matrix[i, j] < 0.7 else "white")
            
            ax.set_title("Vocabulary Overlap Between Datasets")
            fig.tight_layout()
            
            # 更新統計信息
            stats_text = "Vocabulary Overlap Statistics:\n"
            for dataset, vocab in datasets_vocab.items():
                stats_text += f"{dataset}: Unique Words = {len(vocab)}\n"
            
            for i, dataset1 in enumerate(datasets):
                for j, dataset2 in enumerate(datasets):
                    if i < j:
                        intersection = len(datasets_vocab[dataset1].intersection(datasets_vocab[dataset2]))
                        stats_text += f"{dataset1} & {dataset2} Common Words: {intersection}\n"
            
            self.stats_info_label.config(text=stats_text)
            
            # 將圖表添加到GUI
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.chart_frame, text="請至少載入兩個數據集進行詞彙重疊分析").pack(padx=20, pady=20)
            
    def _analyze_sentiment_words(self):
        # 確保VADER資源已下載
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            self.status_var.set("下載VADER情感詞典...")
            nltk.download('vader_lexicon')
            self.status_var.set("VADER情感詞典已下載")
        
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        
        # 準備各數據集的文本
        datasets_text = {}
        
        if self.amazon_data is not None:
            review_col = 'reviewText' if 'reviewText' in self.amazon_data.columns else 'review'
            if review_col in self.amazon_data.columns:
                datasets_text['Amazon'] = self.amazon_data[review_col].astype(str).tolist()
        
        if self.yelp_data is not None:
            review_col = 'text' if 'text' in self.yelp_data.columns else 'review'
            if review_col in self.yelp_data.columns:
                datasets_text['Yelp'] = self.yelp_data[review_col].astype(str).tolist()
        
        if self.imdb_data is not None:
            review_col = 'review' if 'review' in self.imdb_data.columns else 'text'
            if review_col in self.imdb_data.columns:
                datasets_text['IMDB'] = self.imdb_data[review_col].astype(str).tolist()
        
        # 分析情感詞分布
        if datasets_text:
            # 設置圖表
            n_datasets = len(datasets_text)
            fig, axs = plt.subplots(1, n_datasets, figsize=(15, 6))
            if n_datasets == 1:
                axs = [axs]  # 確保axs是列表
            
            sentiment_stats = {}
            stats_text = "Sentiment Words Distribution Statistics:\n"
            
            for i, (dataset, texts) in enumerate(datasets_text.items()):
                # 計算每個評論的情感分數
                pos_scores = []
                neg_scores = []
                neu_scores = []
                compound_scores = []
                
                for text in texts[:500]:  # 限制樣本數以提高性能
                    scores = sia.polarity_scores(text)
                    pos_scores.append(scores['pos'])
                    neg_scores.append(scores['neg'])
                    neu_scores.append(scores['neu'])
                    compound_scores.append(scores['compound'])
                
                # 存儲統計數據
                sentiment_stats[dataset] = {
                    'positive': np.mean(pos_scores),
                    'negative': np.mean(neg_scores),
                    'neutral': np.mean(neu_scores),
                    'compound': np.mean(compound_scores)
                }
                
                # 繪製條形圖
                categories = ['Positive', 'Negative', 'Neutral']
                values = [np.mean(pos_scores), np.mean(neg_scores), np.mean(neu_scores)]
                colors = ['green', 'red', 'blue']
                
                axs[i].bar(categories, values, color=colors)
                axs[i].set_title(f'{dataset} Sentiment Words Distribution')
                axs[i].set_ylim(0, 1)
                
                # 添加統計信息
                stats_text += f"{dataset}:\n"
                stats_text += f"  Average Positive Intensity: {np.mean(pos_scores):.4f}\n"
                stats_text += f"  Average Negative Intensity: {np.mean(neg_scores):.4f}\n"
                stats_text += f"  Average Neutral Intensity: {np.mean(neu_scores):.4f}\n"
                stats_text += f"  Average Compound Score: {np.mean(compound_scores):.4f}\n"
            
            # 調整佈局
            plt.tight_layout()
            
            # 更新統計信息
            self.stats_info_label.config(text=stats_text)
            
            # 將圖表添加到GUI
            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.chart_frame, text="未載入任何數據進行情感詞分析").pack(padx=20, pady=20)

    # ===批次處理區塊===================================================
    def run_batch_preprocessing(self):
        """執行批次處理腳本進行全量資料預處理"""
        try:
            # 檢查是否有至少一個資料集已載入
            has_data = False
            
            if hasattr(self, 'amazon_file_path') and self.amazon_file_path:
                has_data = True
            
            if hasattr(self, 'yelp_file_path') and self.yelp_file_path:
                has_data = True
            
            if hasattr(self, 'imdb_file_path') and self.imdb_file_path:
                has_data = True
            
            if not has_data:
                messagebox.showerror("錯誤", "請先載入至少一個資料集")
                return
            
            # 定義批次檔案的路徑
            batch_file_path = "preprocess_full_data.bat"
            
            # 檢查是否需要產生批次檔
            self._generate_batch_file(batch_file_path)

            # 驗證檔案是否存在
            python_script_path = "preprocess_full_data.py"  
            if not os.path.exists(python_script_path):
                messagebox.showerror("錯誤", f"Python腳本檔案不存在: {python_script_path}")
                return
            
            # 顯示確認對話框
            if messagebox.askyesno("確認", "這將啟動全量資料預處理，可能需要較長時間執行。是否繼續？"):
                # 更新狀態
                self.status_var.set("正在啟動批次預處理...")
                
                # 非同步執行批次檔
                import subprocess
                process = subprocess.Popen(batch_file_path, shell=True)
                
                # 通知使用者
                messagebox.showinfo("批次處理已啟動", 
                                "批次預處理程序已在背景啟動，處理結果將自動儲存。\n" +
                                "您可以繼續使用GUI或關閉GUI，批次處理將繼續執行。")
                
                self.status_var.set("批次預處理已在背景啟動")
        except Exception as e:
            self.status_var.set(f"啟動批次預處理時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"啟動批次預處理時出錯: {str(e)}")
    
    def _generate_batch_file(self, batch_file_path):
        """產生批次預處理的批次檔案"""
        # 取得預處理選項設定
        clean_html = self.clean_html_var.get()
        remove_punctuation = self.remove_punctuation_var.get()
        lowercase = self.lowercase_var.get()
        remove_stopwords = self.remove_stopwords_var.get()
        stemming = self.stemming_var.get()
        lemmatization = self.lemmatization_var.get()
        
        # 獲取已載入的檔案路徑
        amazon_path = getattr(self, 'amazon_file_path', None)
        yelp_path = getattr(self, 'yelp_file_path', None)
        imdb_path = getattr(self, 'imdb_file_path', None)
        
        # 獲取當前工作目錄並組合路徑
        current_dir = os.getcwd()
        output_dir = os.path.join(current_dir, "ReviewsDataBase", "preprocessedFile")
        output_dir_escaped = output_dir.replace('\\', '\\\\')

        # 使用 textwrap.dedent 自動移除多行字串中的共同縮排
        from textwrap import dedent
        
        # 準備Python腳本內容 - 注意每行開頭不要有縮排
        python_script = dedent('''
        import pandas as pd
        import re
        import os
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        import datetime

        print("Starting full data preprocessing...")
        
        # 設置輸出目錄
        output_dir = r"{6}"
        print("Output directory: " + output_dir)
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print("Created output directory: " + output_dir)
        else:
            print("Output directory exists: " + output_dir)

        # 確保必要的NLTK資源已下載
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')

        # 預處理選項
        CLEAN_HTML = {0}
        REMOVE_PUNCTUATION = {1}
        LOWERCASE = {2}
        REMOVE_STOPWORDS = {3}
        STEMMING = {4}
        LEMMATIZATION = {5}

        # 準備預處理工具
        stop_words = set(stopwords.words('english')) if REMOVE_STOPWORDS else set()
        stemmer = PorterStemmer() if STEMMING else None
        lemmatizer = WordNetLemmatizer() if LEMMATIZATION else None

        def preprocess_text(text):
            processed_text = text
            
            # 執行各預處理步驟
            if CLEAN_HTML:
                processed_text = re.sub(r'<.*?>', '', processed_text)
            
            if LOWERCASE:
                processed_text = processed_text.lower()
            
            if REMOVE_PUNCTUATION:
                processed_text = re.sub(r'[^\\w\\s]', '', processed_text)
            
            # 分詞
            tokens = processed_text.split()
            
            if REMOVE_STOPWORDS:
                tokens = [token for token in tokens if token not in stop_words]
            
            if STEMMING:
                tokens = [stemmer.stem(token) for token in tokens]
            
            if LEMMATIZATION:
                tokens = [lemmatizer.lemmatize(token) for token in tokens]
            
            # 重新組合文本
            return ' '.join(tokens)
        ''').format(clean_html, remove_punctuation, lowercase, remove_stopwords, stemming, lemmatization, output_dir_escaped)
        
        # 添加各數據集的處理邏輯，使用條件判斷確保檔案路徑存在
        # [ Amazon資料集 ]
        if amazon_path:
            amazon_path_escaped = amazon_path.replace('\\', '\\\\')
            amazon_script = '''
            # 處理Amazon資料
            try:
                amazon_path = r"{amazon_path}"
                if os.path.exists(amazon_path):
                    print("Processing Amazon data: " + amazon_path)
                    amazon_data = pd.read_csv(amazon_path)
                    
                    review_col = 'reviewText' if 'reviewText' in amazon_data.columns else 'review'
                    if review_col in amazon_data.columns:
                        print("Found review column: " + review_col + ", total " + str(len(amazon_data)) + " records")
                        
                        # 處理評論文本
                        amazon_data['preprocessed_review'] = amazon_data[review_col].astype(str).apply(preprocess_text)
                        
                        # 儲存處理結果
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_file = os.path.join(output_dir, "amazon_preprocessed_" + timestamp + ".csv")
                        amazon_data.to_csv(output_file, index=False)
                        print("Amazon data processing completed, saved to: " + output_file)
                else:
                    print("Amazon data file not found: " + amazon_path)
            except Exception as e:
                print("Error processing Amazon data: " + str(e))
            '''
            # 使用 format 方法進行替換，避免 f-string 問題
            amazon_script = amazon_script.format(amazon_path=amazon_path_escaped)
            python_script += dedent(amazon_script)
        
        # Yelp資料集
        if yelp_path:
            yelp_path_escaped = yelp_path.replace('\\', '\\\\')
            yelp_script = '''
            # 處理Yelp資料
            try:
                yelp_path = r"{yelp_path}"
                if os.path.exists(yelp_path):
                    print("Processing Yelp data: " + yelp_path)
                    # 讀取檔案（可能很大，使用分批讀取）
                    chunk_size = 100000
                    chunk_num = 0
                    
                    # 判斷檔案類型並使用適當的讀取方法
                    if yelp_path.endswith('.json'):
                        for chunk in pd.read_json(yelp_path, lines=True, chunksize=chunk_size):
                            chunk_num += 1
                            print("Processing Yelp data chunk " + str(chunk_num) + ", " + str(len(chunk)) + " records")
                            
                            review_col = 'text' if 'text' in chunk.columns else 'review'
                            if review_col in chunk.columns:
                                # 處理評論文本
                                chunk['preprocessed_review'] = chunk[review_col].astype(str).apply(preprocess_text)
                                
                                # 儲存此批次結果
                                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                output_file = os.path.join(output_dir, "yelp_preprocessed_chunk" + str(chunk_num) + "_" + timestamp + ".csv")
                                chunk.to_csv(output_file, index=False)
                                print("Yelp data chunk " + str(chunk_num) + " processing completed, saved to: " + output_file)
                    else:
                        for chunk in pd.read_csv(yelp_path, chunksize=chunk_size):
                            chunk_num += 1
                            print("Processing Yelp data chunk " + str(chunk_num) + ", " + str(len(chunk)) + " records")
                            
                            review_col = 'text' if 'text' in chunk.columns else 'review'
                            if review_col in chunk.columns:
                                # 處理評論文本
                                chunk['preprocessed_review'] = chunk[review_col].astype(str).apply(preprocess_text)
                                
                                # 儲存此批次結果
                                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                                output_file = os.path.join(output_dir, "yelp_preprocessed_chunk" + str(chunk_num) + "_" + timestamp + ".csv")
                                chunk.to_csv(output_file, index=False)
                                print("Yelp data chunk " + str(chunk_num) + " processing completed, saved to: " + output_file)
                else:
                    print("Yelp data file not found: " + yelp_path)
            except Exception as e:
                print("Error processing Yelp data: " + str(e))
            '''
            # 使用 format 方法進行替換，避免 f-string 問題
            yelp_script = yelp_script.format(yelp_path=yelp_path_escaped)
            python_script += dedent(yelp_script)

        # IMDB 資料集
        if imdb_path:
            imdb_path_escaped = imdb_path.replace('\\', '\\\\')
            imdb_script = '''
            # 處理IMDB資料
            try:
                imdb_path = r"{imdb_path}"
                if os.path.exists(imdb_path):
                    print("Processing IMDB data: " + imdb_path)
                    imdb_data = pd.read_csv(imdb_path)
                    
                    review_col = 'review' if 'review' in imdb_data.columns else 'text'
                    if review_col in imdb_data.columns:
                        print("Found review column: " + review_col + ", total " + str(len(imdb_data)) + " records")
                        
                        # 處理評論文本
                        imdb_data['preprocessed_review'] = imdb_data[review_col].astype(str).apply(preprocess_text)
                        
                        # 儲存處理結果
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_file = os.path.join(output_dir, "imdb_preprocessed_" + timestamp + ".csv")
                        imdb_data.to_csv(output_file, index=False)
                        print("IMDB data processing completed, saved to: " + output_file)
                else:
                    print("IMDB data file not found: " + imdb_path)
            except Exception as e:
                print("Error processing IMDB data: " + str(e))
            '''
            # 使用 format 方法進行替換，避免 f-string 問題
            imdb_script = imdb_script.format(imdb_path=imdb_path_escaped)
            python_script += dedent(imdb_script)
        
        # 將批次檔案與Python腳本放在相同目錄
        python_script_path = "preprocess_full_data.py"
        
        # 寫入Python腳本
        with open(python_script_path, "w", encoding="utf-8") as f:
            f.write(python_script)
        
        # 生成批次檔
        with open(batch_file_path, "w", encoding="utf-8") as f:
            f.write('@echo off\n')
            f.write('echo Starting full data preprocessing\n')
            f.write(f'python "{python_script_path}"\n')
            f.write('echo Processing completed\n')

        print(f"Batch file generated: {batch_file_path}")
    # ==================================================================

    def run_preprocessing(self):
        """執行所選的預處理步驟"""
        self.status_var.set("正在執行預處理...")
        self.preprocess_status_var.set("正在處理數據...")
        
        try:
            # 獲取選定的數據源
            source = self.data_source.get()
            data_to_process = {}
            
            if (source == "Amazon" or source == "全部") and self.amazon_data is not None:
                review_col = 'reviewText' if 'reviewText' in self.amazon_data.columns else 'review'
                if review_col in self.amazon_data.columns:
                    # 獲取情感標籤
                    rating_col = 'overall' if 'overall' in self.amazon_data.columns else 'rating'
                    if rating_col in self.amazon_data.columns:
                        # 轉換評分為二元情感標籤
                        sentiment = self.amazon_data[rating_col].apply(lambda x: "positive" if x > 3 else ("negative" if x < 3 else "neutral"))
                        # 取樣本數據以提高性能
                        sample_size = min(1000, len(self.amazon_data))
                        reviews = self.amazon_data[review_col].astype(str).head(sample_size).tolist()
                        data_to_process['Amazon'] = (reviews, sentiment.head(sample_size).tolist())
            
            if (source == "Yelp" or source == "全部") and self.yelp_data is not None:
                review_col = 'text' if 'text' in self.yelp_data.columns else 'review'
                if review_col in self.yelp_data.columns:
                    # 獲取情感標籤
                    rating_col = 'stars' if 'stars' in self.yelp_data.columns else 'rating'
                    if rating_col in self.yelp_data.columns:
                        # 轉換評分為二元情感標籤
                        sentiment = self.yelp_data[rating_col].apply(lambda x: "positive" if x > 3 else ("negative" if x < 3 else "neutral"))
                        # 取樣本數據以提高性能
                        sample_size = min(1000, len(self.yelp_data))
                        reviews = self.yelp_data[review_col].astype(str).head(sample_size).tolist()
                        data_to_process['Yelp'] = (reviews, sentiment.head(sample_size).tolist())
            
            if (source == "IMDB" or source == "全部") and self.imdb_data is not None:
                review_col = 'review' if 'review' in self.imdb_data.columns else 'text'
                if review_col in self.imdb_data.columns:
                    # 獲取情感標籤
                    sentiment_col = 'sentiment' if 'sentiment' in self.imdb_data.columns else 'label'
                    if sentiment_col in self.imdb_data.columns:
                        # 取樣本數據以提高性能
                        sample_size = min(1000, len(self.imdb_data))
                        reviews = self.imdb_data[review_col].astype(str).head(sample_size).tolist()
                        sentiments = self.imdb_data[sentiment_col].head(sample_size).tolist()
                        data_to_process['IMDB'] = (reviews, sentiments)
            
            if not data_to_process:
                self.preprocess_status_var.set("沒有數據可供處理，請先載入數據")
                self.status_var.set("預處理失敗：沒有數據可供處理")
                return
            
            # 執行預處理
            self.preprocessed_data = []
            
            # 獲取預處理選項
            clean_html = self.clean_html_var.get()
            remove_punctuation = self.remove_punctuation_var.get()
            lowercase = self.lowercase_var.get()
            remove_stopwords = self.remove_stopwords_var.get()
            stemming = self.stemming_var.get()
            lemmatization = self.lemmatization_var.get()
            
            # 準備預處理工具
            stop_words = set(stopwords.words('english')) if remove_stopwords else set()
            stemmer = PorterStemmer() if stemming else None
            lemmatizer = WordNetLemmatizer() if lemmatization else None
            
            # 清空預處理結果樹視圖
            for item in self.preprocess_tree.get_children():
                self.preprocess_tree.delete(item)
            
            # 處理每個數據源
            for source_name, (texts, labels) in data_to_process.items():
                for i, (text, label) in enumerate(zip(texts, labels)):
                    original_text = text
                    processed_text = text
                    
                    # 執行各預處理步驟
                    if clean_html:
                        processed_text = re.sub(r'<.*?>', '', processed_text)
                    
                    if lowercase:
                        processed_text = processed_text.lower()
                    
                    if remove_punctuation:
                        processed_text = re.sub(r'[^\w\s]', '', processed_text)
                    
                    # 分詞
                    tokens = processed_text.split()
                    
                    if remove_stopwords:
                        tokens = [token for token in tokens if token not in stop_words]
                    
                    if stemming:
                        tokens = [stemmer.stem(token) for token in tokens]
                    
                    if lemmatization:
                        tokens = [lemmatizer.lemmatize(token) for token in tokens]
                    
                    # 重新組合文本
                    processed_text = ' '.join(tokens)
                    
                    # 添加到結果列表
                    self.preprocessed_data.append({
                        'original': original_text,
                        'preprocessed': processed_text,
                        'source': source_name,
                        'label': label
                    })
                    
                    # 添加到樹視圖（只顯示前100個結果）
                    if len(self.preprocessed_data) <= 100:
                        self.preprocess_tree.insert('', 'end', values=(
                            len(self.preprocessed_data),
                            original_text[:50] + "..." if len(original_text) > 50 else original_text,
                            processed_text[:50] + "..." if len(processed_text) > 50 else processed_text,
                            source_name,
                            label
                        ))
            
            # 更新狀態
            self.preprocess_status_var.set(f"預處理完成，共處理 {len(self.preprocessed_data)} 條數據")
            self.status_var.set("預處理完成")
            
        except Exception as e:
            self.preprocess_status_var.set(f"預處理時出錯: {str(e)}")
            self.status_var.set(f"預處理失敗: {str(e)}")
            messagebox.showerror("錯誤", f"預處理時出錯: {str(e)}")
    
    def save_preprocessed_data(self):
        """儲存預處理後的數據"""
        if not hasattr(self, 'preprocessed_data') or not self.preprocessed_data:
            messagebox.showwarning("警告", "沒有可儲存的預處理數據，請先執行預處理")
            return
        
        try:
            # 創建DataFrame
            df = pd.DataFrame(self.preprocessed_data)
            
            # 選擇保存路徑
            file_path = filedialog.asksaveasfilename(
                title="儲存預處理數據",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                defaultextension=".csv"
            )
            
            if not file_path:
                return  # 用戶取消
            
            # 保存到CSV
            df.to_csv(file_path, index=False)
            
            self.status_var.set(f"已成功儲存預處理數據到 {file_path}")
            messagebox.showinfo("成功", f"已成功儲存 {len(self.preprocessed_data)} 條預處理數據")
            
        except Exception as e:
            self.status_var.set(f"儲存預處理數據時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"儲存預處理數據時出錯: {str(e)}")

    def _setup_aspect_tab(self):
        """設置面相切割標籤頁"""
        # 控制框架
        controls_frame = ttk.Frame(self.aspect_frame)
        controls_frame.pack(fill='x', padx=10, pady=10)
        
        # 選擇數據源
        ttk.Label(controls_frame, text="選擇數據源:").pack(side=tk.LEFT, padx=5)
        self.aspect_data_source = ttk.Combobox(controls_frame, values=["Amazon", "Yelp", "IMDB", "預處理結果"])
        self.aspect_data_source.pack(side=tk.LEFT, padx=5)
        self.aspect_data_source.current(3)
        
        # 選擇領域
        ttk.Label(controls_frame, text="選擇領域:").pack(side=tk.LEFT, padx=5)
        self.aspect_domain = ttk.Combobox(controls_frame, values=["general", "restaurant", "electronics", "movies"])
        self.aspect_domain.pack(side=tk.LEFT, padx=5)
        self.aspect_domain.current(0)
        
        # 面相提取方法
        self.aspect_method_frame = ttk.LabelFrame(self.aspect_frame, text="面相提取方法")
        self.aspect_method_frame.pack(fill='x', padx=10, pady=5)
        
        # 創建變數
        self.use_rule_method = tk.BooleanVar(value=True)
        self.use_dictionary_method = tk.BooleanVar(value=True)
        self.use_spacy_method = tk.BooleanVar(value=False)
        self.use_topic_modeling = tk.BooleanVar(value=True)
        
        # 添加選項
        ttk.Checkbutton(self.aspect_method_frame, text="規則基礎方法", variable=self.use_rule_method).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(self.aspect_method_frame, text="詞典匹配方法", variable=self.use_dictionary_method).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(self.aspect_method_frame, text="spaCy依存分析(需安裝)", variable=self.use_spacy_method).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(self.aspect_method_frame, text="主題模型(大量數據)", variable=self.use_topic_modeling).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 執行面相切割和保存結果的按鈕
        button_frame = ttk.Frame(self.aspect_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="執行面相切割", command=self.run_aspect_extraction).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="儲存面相切割結果", command=self.save_aspect_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="生成面相特定數據集", command=self.create_aspect_specific_dataset).pack(side=tk.LEFT, padx=5)
        
        # 結果顯示區域
        result_pane = ttk.PanedWindow(self.aspect_frame, orient=tk.VERTICAL)
        result_pane.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 面相切割結果預覽
        preview_frame = ttk.LabelFrame(result_pane, text="面相切割結果預覽")
        result_pane.add(preview_frame, weight=3)
        
        columns = ('review_id', 'original_review', 'aspects_found', 'aspect_terms', 'aspect_categories', 'aspects_count')
        self.aspect_tree = ttk.Treeview(preview_frame, columns=columns, show='headings')
        
        # 設置列標題
        column_widths = {
            'review_id': 50,
            'original_review': 300,
            'aspects_found': 80,
            'aspect_terms': 200,
            'aspect_categories': 150,
            'aspects_count': 80
        }
        
        for col in columns:
            self.aspect_tree.heading(col, text=col.replace('_', ' ').title())
            self.aspect_tree.column(col, width=column_widths.get(col, 100))
        
        # 添加滾動條
        scrollbar_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.aspect_tree.yview)
        scrollbar_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.aspect_tree.xview)
        self.aspect_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.aspect_tree.pack(fill='both', expand=True)
        
        # 面相統計信息
        stats_frame = ttk.LabelFrame(result_pane, text="面相統計信息")
        result_pane.add(stats_frame, weight=1)
        
        # 左側面相統計
        stats_left_frame = ttk.Frame(stats_frame)
        stats_left_frame.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)
        
        self.aspect_stats_text = tk.Text(stats_left_frame, height=10, width=40)
        self.aspect_stats_text.pack(fill='both', expand=True)
        stats_scrollbar = ttk.Scrollbar(stats_left_frame, orient=tk.VERTICAL, command=self.aspect_stats_text.yview)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.aspect_stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        # 右側面相圖表框架
        stats_right_frame = ttk.Frame(stats_frame)
        stats_right_frame.pack(side=tk.RIGHT, fill='both', expand=True, padx=5, pady=5)
        
        # 圖表容器
        self.aspect_chart_container = ttk.Frame(stats_right_frame)
        self.aspect_chart_container.pack(fill='both', expand=True)
        
        # 狀態顯示
        self.aspect_status_var = tk.StringVar()
        self.aspect_status_var.set("請選擇數據源和面相提取方法，然後點擊「執行面相切割」")
        status_label = ttk.Label(self.aspect_frame, textvariable=self.aspect_status_var)
        status_label.pack(padx=10, pady=10)

    # 4. 添加面相切割功能的實現方法
    def run_aspect_extraction(self):
        """執行面相切割操作"""
        self.status_var.set("正在執行面相切割...")
        self.aspect_status_var.set("正在處理數據...")
        
        try:
            # 獲取數據源
            source = self.aspect_data_source.get()
            data_to_process = None
            text_column = None
            label_column = None
            
            # 獲取選定領域
            domain = self.aspect_domain.get()
            
            # 獲取提取方法
            methods = []
            if self.use_rule_method.get():
                methods.append('rule')
            if self.use_dictionary_method.get():
                methods.append('dictionary')
            if self.use_spacy_method.get():
                methods.append('spacy')
            
            if not methods:
                messagebox.showwarning("警告", "請至少選擇一種面相提取方法")
                return
            
            # 準備數據
            if source == "Amazon" and self.amazon_data is not None:
                data_to_process = self.amazon_data
                text_column = 'reviewText' if 'reviewText' in data_to_process.columns else 'review'
                label_column = 'overall' if 'overall' in data_to_process.columns else 'rating'
            
            elif source == "Yelp" and self.yelp_data is not None:
                data_to_process = self.yelp_data
                text_column = 'text' if 'text' in data_to_process.columns else 'review'
                label_column = 'stars' if 'stars' in data_to_process.columns else 'rating'
            
            elif source == "IMDB" and self.imdb_data is not None:
                data_to_process = self.imdb_data
                text_column = 'review' if 'review' in data_to_process.columns else 'text'
                label_column = 'sentiment' if 'sentiment' in data_to_process.columns else 'label'
            
            elif source == "預處理結果" and hasattr(self, 'preprocessed_data') and self.preprocessed_data:
                # 將預處理結果轉換為DataFrame
                data_to_process = pd.DataFrame(self.preprocessed_data)
                text_column = 'preprocessed'  # 使用預處理後的文本
                label_column = 'label'
            
            if data_to_process is None or text_column not in data_to_process.columns:
                self.aspect_status_var.set("無法處理所選數據源，請確保數據已正確載入")
                messagebox.showerror("錯誤", "無法處理所選數據源，請確保數據已正確載入")
                return
            
            # 限制處理的數據量，以提高性能
            if len(data_to_process) > 1000:
                self.aspect_status_var.set(f"數據量較大 ({len(data_to_process)} 條)，只處理前 1000 條...")
                processing_data = data_to_process.head(1000).copy()
            else:
                processing_data = data_to_process.copy()
            
            # 執行面相切割
            self.aspect_status_var.set(f"正在對 {len(processing_data)} 條評論進行面相切割...")
            
            # 使用面相切割模組處理數據
            segmented_df, aspect_stats, topic_results = ae.segment_reviews_by_aspects(
                processing_data, 
                domain=domain, 
                text_column=text_column, 
                label_column=label_column, 
                methods=methods
            )
            
            # 保存結果
            self.aspect_segmented_df = segmented_df
            self.aspect_stats_results = aspect_stats
            self.aspect_topic_results = topic_results
            
            # 清空現有樹視圖
            for item in self.aspect_tree.get_children():
                self.aspect_tree.delete(item)
            
            # 填充樹視圖
            for idx, row in segmented_df.iterrows():
                try:
                    self.aspect_tree.insert('', 'end', values=(
                        row['review_id'],
                        row['original_review'][:50] + "..." if len(row['original_review']) > 50 else row['original_review'],
                        "是" if row['aspects_found'] else "否",
                        row['aspect_terms'][:50] + "..." if len(row['aspect_terms']) > 50 else row['aspect_terms'],
                        row['aspect_categories'],
                        row['aspects_count']
                    ))
                except Exception as e:
                    print(f"Error inserting row {idx}: {e}")
            
            # 更新統計信息
            self._update_aspect_stats_display(aspect_stats, topic_results)
            
            # 更新狀態
            self.aspect_status_var.set(f"面相切割完成，找到 {sum(segmented_df['aspects_found'])} 條含面相的評論")
            self.status_var.set("面相切割完成")
            
        except Exception as e:
            self.aspect_status_var.set(f"面相切割時出錯: {str(e)}")
            self.status_var.set(f"面相切割失敗: {str(e)}")
            messagebox.showerror("錯誤", f"面相切割時出錯: {str(e)}")

    def _update_aspect_stats_display(self, aspect_stats, topic_results):
        """更新面相統計顯示"""
        # 更新文字統計信息
        self.aspect_stats_text.delete(1.0, tk.END)
        
        stats_text = "面相統計信息:\n\n"
        for category, stats in aspect_stats.items():
            stats_text += f"類別: {category}\n"
            stats_text += f"  出現次數: {stats['count']}\n"
            stats_text += f"  佔比: {stats['percentage']:.2f}%\n"
            
            if 'avg_sentiment' in stats:
                sentiment_text = "正面" if stats['avg_sentiment'] > 0 else ("負面" if stats['avg_sentiment'] < 0 else "中性")
                stats_text += f"  平均情感: {stats['avg_sentiment']:.2f} ({sentiment_text})\n"
                stats_text += f"  正面評論: {stats.get('positive_count', 0)}\n"
                stats_text += f"  負面評論: {stats.get('negative_count', 0)}\n"
                stats_text += f"  中性評論: {stats.get('neutral_count', 0)}\n"
            
            stats_text += "\n"
        
        if topic_results:
            stats_text += "\n主題模型發現的潛在面相主題:\n\n"
            for i, topic in enumerate(topic_results):
                stats_text += f"主題 {i+1}: {', '.join(topic[:5])}...\n"
        
        self.aspect_stats_text.insert(tk.END, stats_text)
        
        # 清除現有圖表
        for widget in self.aspect_chart_container.winfo_children():
            widget.destroy()
        
        # 如果有足夠數據，繪製面相分布圖表
        if aspect_stats:
            # 創建圖表
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # 準備數據
            categories = list(aspect_stats.keys())
            counts = [stats['count'] for stats in aspect_stats.values()]
            
            # 繪製柱狀圖
            bars = ax.bar(categories, counts)
            
            # 添加標籤
            ax.set_title('面相類別分布')
            ax.set_xlabel('面相類別')
            ax.set_ylabel('出現次數')
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(categories, rotation=45, ha='right')
            
            # 為每個柱子添加計數標籤
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}',
                        ha='center', va='bottom')
            
            # 調整佈局
            plt.tight_layout()
            
            # 將圖表添加到GUI
            canvas = FigureCanvasTkAgg(fig, master=self.aspect_chart_container)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_aspect_results(self):
        """保存面相切割結果"""
        if not hasattr(self, 'aspect_segmented_df') or self.aspect_segmented_df is None:
            messagebox.showwarning("警告", "沒有可保存的面相切割結果，請先執行面相切割")
            return
        
        try:
            # 選擇保存路徑
            file_path = filedialog.asksaveasfilename(
                title="保存面相切割結果",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                defaultextension=".csv"
            )
            
            if not file_path:
                return  # 用戶取消
            
            # 保存到CSV
            self.aspect_segmented_df.to_csv(file_path, index=False)
            
            self.status_var.set(f"已成功保存面相切割結果到 {file_path}")
            self.aspect_status_var.set(f"已成功保存 {len(self.aspect_segmented_df)} 條面相切割結果")
            messagebox.showinfo("成功", f"已成功保存面相切割結果")
            
        except Exception as e:
            self.status_var.set(f"保存面相切割結果時出錯: {str(e)}")
            self.aspect_status_var.set(f"保存面相切割結果時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"保存面相切割結果時出錯: {str(e)}")

    def create_aspect_specific_dataset(self):
        """生成面相特定數據集"""
        if not hasattr(self, 'aspect_segmented_df') or self.aspect_segmented_df is None:
            messagebox.showwarning("警告", "沒有可用的面相切割結果，請先執行面相切割")
            return
        
        try:
            # 獲取所有可用的面相類別
            all_categories = set()
            for cats in self.aspect_segmented_df['aspect_categories']:
                if cats:
                    all_categories.update(cats.split('|'))
            
            if not all_categories:
                messagebox.showwarning("警告", "未找到任何面相類別")
                return
            
            # 創建選擇對話框
            aspect_dialog = tk.Toplevel(self.root)
            aspect_dialog.title("選擇要包含的面相類別")
            aspect_dialog.geometry("400x300")
            aspect_dialog.transient(self.root)
            aspect_dialog.grab_set()
            
            # 添加說明
            ttk.Label(aspect_dialog, text="請選擇要包含在面相特定數據集中的類別:").pack(padx=10, pady=10)
            
            # 創建複選框
            category_vars = {}
            for category in sorted(all_categories):
                var = tk.BooleanVar(value=True)
                category_vars[category] = var
                ttk.Checkbutton(aspect_dialog, text=category, variable=var).pack(anchor=tk.W, padx=20, pady=2)
            
            # 確認按鈕的回調函數
            def on_confirm():
                selected_categories = [cat for cat, var in category_vars.items() if var.get()]
                
                if not selected_categories:
                    messagebox.showwarning("警告", "請至少選擇一個面相類別")
                    return
                
                # 關閉對話框
                aspect_dialog.destroy()
                
                # 獲取數據源
                source = self.aspect_data_source.get()
                original_df = None
                
                if source == "Amazon" and self.amazon_data is not None:
                    original_df = self.amazon_data
                elif source == "Yelp" and self.yelp_data is not None:
                    original_df = self.yelp_data
                elif source == "IMDB" and self.imdb_data is not None:
                    original_df = self.imdb_data
                elif source == "預處理結果" and hasattr(self, 'preprocessed_data'):
                    original_df = pd.DataFrame(self.preprocessed_data)
                
                if original_df is None:
                    messagebox.showerror("錯誤", "無法獲取原始數據集")
                    return
                
                # 創建面相特定數據集
                self.aspect_status_var.set("正在生成面相特定數據集...")
                
                # 初始化面相提取器（用於獲取詞典）
                domain = self.aspect_domain.get()
                extractor = ae.AspectExtractor(domain=domain)
                
                aspect_df = ae.create_aspect_specific_dataset(
                    original_df, self.aspect_segmented_df, selected_categories
                )
                
                # 保存結果
                file_path = filedialog.asksaveasfilename(
                    title="保存面相特定數據集",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                    defaultextension=".csv"
                )
                
                if not file_path:
                    return  # 用戶取消
                
                # 保存到CSV
                aspect_df.to_csv(file_path, index=False)
                
                self.status_var.set(f"已成功保存面相特定數據集到 {file_path}")
                self.aspect_status_var.set(f"已成功生成 {len(aspect_df)} 條面相特定數據")
                messagebox.showinfo("成功", f"已成功生成面相特定數據集，共 {len(aspect_df)} 條記錄")
            
            # 添加按鈕
            button_frame = ttk.Frame(aspect_dialog)
            button_frame.pack(fill='x', padx=10, pady=10)
            
            ttk.Button(button_frame, text="確認", command=on_confirm).pack(side=tk.RIGHT, padx=5)
            ttk.Button(button_frame, text="取消", command=aspect_dialog.destroy).pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            self.status_var.set(f"生成面相特定數據集時出錯: {str(e)}")
            self.aspect_status_var.set(f"生成面相特定數據集時出錯: {str(e)}")
            messagebox.showerror("錯誤", f"生成面相特定數據集時出錯: {str(e)}")

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = DataPreprocessingApp(root)
    root.mainloop()