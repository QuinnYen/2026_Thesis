#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
權重配置窗口 - GUI中的智能權重學習與配置介面
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import json
import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging

# 設定 matplotlib 字體，避免中文亂碼
try:
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 項目內部模組
try:
    from modules.adaptive_weight_learner import AdaptiveWeightLearner, OPTUNA_AVAILABLE
    ADAPTIVE_LEARNER_AVAILABLE = True
except ImportError:
    AdaptiveWeightLearner = None
    OPTUNA_AVAILABLE = False
    ADAPTIVE_LEARNER_AVAILABLE = False

logger = logging.getLogger(__name__)

class WeightConfigWindow:
    """權重配置窗口類"""
    
    def __init__(self, parent, main_app):
        self.parent = parent
        self.main_app = main_app
        
        # 創建頂級窗口
        self.window = tk.Toplevel(parent)
        self.window.title("智能權重學習與配置")
        self.window.geometry("800x700")
        self.window.resizable(True, True)
        
        # 設定窗口關閉事件
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 初始化變數
        self.weight_learner = None
        self.learning_results = None
        self.current_weights = {
            'similarity': tk.DoubleVar(value=0.33),
            'keyword': tk.DoubleVar(value=0.33),
            'self': tk.DoubleVar(value=0.34)
        }
        
        # 創建GUI元件
        self.create_widgets()
        
        # 載入已保存的權重（如果有）
        self.load_saved_weights()
    
    def create_widgets(self):
        """創建窗口元件"""
        # 創建筆記本控件（分頁）
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 創建三個分頁
        self.create_manual_config_tab()
        self.create_auto_learning_tab()
        self.create_results_tab()
    
    def create_manual_config_tab(self):
        """手動權重配置分頁"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="手動配置")
        
        # 標題
        title_label = ttk.Label(frame, text="手動權重配置", 
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=10)
        
        # 權重配置框架
        config_frame = ttk.LabelFrame(frame, text="注意力機制權重", padding=15)
        config_frame.pack(fill='x', padx=20, pady=10)
        
        # 相似度注意力權重
        self.create_weight_slider(config_frame, "相似度注意力 (Similarity):", 
                                 self.current_weights['similarity'], 0)
        
        # 關鍵詞注意力權重  
        self.create_weight_slider(config_frame, "關鍵詞注意力 (Keyword):", 
                                 self.current_weights['keyword'], 1)
        
        # 自注意力權重
        self.create_weight_slider(config_frame, "自注意力 (Self-Attention):", 
                                 self.current_weights['self'], 2)
        
        # 權重總和顯示
        self.weight_sum_label = ttk.Label(config_frame, text="權重總和: 1.00", 
                                         font=('TkDefaultFont', 10, 'bold'))
        self.weight_sum_label.grid(row=3, column=0, columnspan=3, pady=10)
        
        # 綁定權重變化事件
        for var in self.current_weights.values():
            var.trace('w', self.on_weight_changed)
        
        # 預設權重配置按鈕
        preset_frame = ttk.LabelFrame(frame, text="預設配置", padding=15)
        preset_frame.pack(fill='x', padx=20, pady=10)
        
        preset_buttons = [
            ("均等權重", {"similarity": 0.33, "keyword": 0.33, "self": 0.34}),
            ("重視相似度", {"similarity": 0.6, "keyword": 0.2, "self": 0.2}),
            ("重視關鍵詞", {"similarity": 0.2, "keyword": 0.6, "self": 0.2}),
            ("重視自注意力", {"similarity": 0.2, "keyword": 0.2, "self": 0.6}),
            ("雙重組合(相似+自)", {"similarity": 0.5, "keyword": 0.0, "self": 0.5}),
            ("雙重組合(相似+關鍵)", {"similarity": 0.5, "keyword": 0.5, "self": 0.0})
        ]
        
        for i, (name, weights) in enumerate(preset_buttons):
            btn = ttk.Button(preset_frame, text=name, 
                           command=lambda w=weights: self.set_preset_weights(w))
            btn.grid(row=i//2, column=i%2, padx=5, pady=2, sticky='ew')
        
        preset_frame.columnconfigure(0, weight=1)
        preset_frame.columnconfigure(1, weight=1)
        
        # 操作按鈕
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill='x', padx=20, pady=20)
        
        ttk.Button(button_frame, text="套用權重", 
                  command=self.apply_manual_weights).pack(side='left', padx=5)
        ttk.Button(button_frame, text="重置為預設", 
                  command=self.reset_to_default).pack(side='left', padx=5)
        ttk.Button(button_frame, text="保存配置", 
                  command=self.save_weights_config).pack(side='left', padx=5)
    
    def create_auto_learning_tab(self):
        """自動學習分頁"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="智能學習")
        
        # 標題
        title_label = ttk.Label(frame, text="智能權重學習", 
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=10)
        
        # 學習器選擇
        learner_frame = ttk.LabelFrame(frame, text="學習器配置", padding=15)
        learner_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(learner_frame, text="學習器類型:").grid(row=0, column=0, sticky='w')
        self.learner_type = tk.StringVar(value='auto')
        
        # 根據可用性決定學習器選項
        learner_options = ['auto', 'grid', 'genetic']
        if OPTUNA_AVAILABLE:
            learner_options.append('bayesian')
        
        learner_combo = ttk.Combobox(learner_frame, textvariable=self.learner_type,
                                   values=learner_options,
                                   state='readonly', width=15)
        learner_combo.grid(row=0, column=1, padx=(5, 0), sticky='w')
        
        # 學習器說明
        learner_info_base = """學習器說明：
• auto: 根據數據大小自動選擇
• grid: 網格搜索（適合小數據集）
• genetic: 遺傳算法（適合中等數據集）"""
        
        if OPTUNA_AVAILABLE:
            learner_info = learner_info_base + "\n• bayesian: 貝葉斯優化（適合大數據集）"
        else:
            learner_info = learner_info_base + "\n⚠️ bayesian: 需要安裝optuna庫"
        
        info_label = ttk.Label(learner_frame, text=learner_info, 
                              font=('TkDefaultFont', 8))
        info_label.grid(row=1, column=0, columnspan=2, pady=5, sticky='w')
        
        # 學習參數配置
        params_frame = ttk.LabelFrame(frame, text="學習參數", padding=15)
        params_frame.pack(fill='x', padx=20, pady=10)
        
        # 驗證比例
        ttk.Label(params_frame, text="驗證比例:").grid(row=0, column=0, sticky='w')
        self.validation_ratio = tk.DoubleVar(value=0.2)
        val_scale = ttk.Scale(params_frame, from_=0.1, to=0.4, 
                            variable=self.validation_ratio, orient='horizontal')
        val_scale.grid(row=0, column=1, padx=(5, 5), sticky='ew')
        self.val_label = ttk.Label(params_frame, text="20%")
        self.val_label.grid(row=0, column=2, sticky='w')
        self.validation_ratio.trace('w', self.update_val_label)
        
        params_frame.columnconfigure(1, weight=1)
        
        # 學習控制
        control_frame = ttk.Frame(frame)
        control_frame.pack(fill='x', padx=20, pady=10)
        
        self.learn_btn = ttk.Button(control_frame, text="開始學習", 
                                   command=self.start_learning)
        self.learn_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="停止學習", 
                                  command=self.stop_learning, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # 進度顯示
        progress_frame = ttk.LabelFrame(frame, text="學習進度", padding=15)
        progress_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.learning_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.learning_progress.pack(fill='x', pady=5)
        
        self.learning_status = ttk.Label(progress_frame, text="準備就緒")
        self.learning_status.pack(pady=5)
        
        # 學習日誌
        self.learning_log = scrolledtext.ScrolledText(progress_frame, height=8)
        self.learning_log.pack(fill='both', expand=True, pady=5)
        self.learning_log.insert('1.0', "智能權重學習日誌：\n\n")
    
    def create_results_tab(self):
        """結果展示分頁"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="學習結果")
        
        # 標題
        title_label = ttk.Label(frame, text="學習結果與分析", 
                               font=('TkDefaultFont', 12, 'bold'))
        title_label.pack(pady=10)
        
        # 結果摘要
        summary_frame = ttk.LabelFrame(frame, text="最佳結果摘要", padding=15)
        summary_frame.pack(fill='x', padx=20, pady=10)
        
        self.result_text = scrolledtext.ScrolledText(summary_frame, height=6)
        self.result_text.pack(fill='both', expand=True)
        self.result_text.insert('1.0', "等待學習結果...")
        
        # 權重可視化
        viz_frame = ttk.LabelFrame(frame, text="權重分佈視覺化", padding=15)
        viz_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 創建matplotlib圖表
        self.create_weight_visualization(viz_frame)
        
        # 操作按鈕
        result_buttons = ttk.Frame(frame)
        result_buttons.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(result_buttons, text="套用最佳權重", 
                  command=self.apply_learned_weights).pack(side='left', padx=5)
        ttk.Button(result_buttons, text="匯出結果", 
                  command=self.export_results).pack(side='left', padx=5)
        ttk.Button(result_buttons, text="重新學習", 
                  command=self.restart_learning).pack(side='left', padx=5)
    
    def create_weight_slider(self, parent, label, variable, row):
        """創建權重滑桿"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', pady=5)
        
        scale = ttk.Scale(parent, from_=0.0, to=1.0, variable=variable, 
                         orient='horizontal')
        scale.grid(row=row, column=1, sticky='ew', padx=(10, 10), pady=5)
        
        value_label = ttk.Label(parent, text=f"{variable.get():.2f}")
        value_label.grid(row=row, column=2, sticky='w', pady=5)
        
        # 更新數值標籤
        def update_label(*args):
            value_label.config(text=f"{variable.get():.2f}")
        variable.trace('w', update_label)
        
        parent.columnconfigure(1, weight=1)
    
    def create_weight_visualization(self, parent):
        """創建權重視覺化圖表"""
        try:
            # 設定matplotlib使用英文字體，避免中文亂碼
            plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
            self.fig.suptitle('Weight Distribution Analysis', fontsize=12)
            
            # 初始化空圖表
            self.ax1.set_title('Current Weight Distribution')
            self.ax2.set_title('Learning History')
            
            self.canvas = FigureCanvasTkAgg(self.fig, parent)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            # 如果matplotlib不可用，顯示文字說明
            placeholder = ttk.Label(parent, text="Weight visualization requires matplotlib library\nInstall with: pip install matplotlib")
            placeholder.pack(fill='both', expand=True)
    
    def on_weight_changed(self, *args):
        """權重改變時的回調"""
        total = sum(var.get() for var in self.current_weights.values())
        self.weight_sum_label.config(
            text=f"權重總和: {total:.2f}",
            foreground='red' if abs(total - 1.0) > 0.01 else 'black'
        )
        
        # 更新視覺化
        self.update_weight_visualization()
    
    def update_weight_visualization(self):
        """更新權重視覺化"""
        try:
            if hasattr(self, 'ax1'):
                self.ax1.clear()
                
                weights = [var.get() for var in self.current_weights.values()]
                labels = ['Similarity', 'Keyword', 'Self-Attention']
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                self.ax1.pie(weights, labels=labels, colors=colors, autopct='%1.1f%%')
                self.ax1.set_title('Current Weight Distribution')
                
                self.canvas.draw()
        except:
            pass  # 忽略視覺化錯誤
    
    def update_val_label(self, *args):
        """更新驗證比例標籤"""
        ratio = self.validation_ratio.get()
        self.val_label.config(text=f"{ratio:.0%}")
    
    def set_preset_weights(self, weights):
        """設定預設權重"""
        for key, value in weights.items():
            if key in self.current_weights:
                self.current_weights[key].set(value)
    
    def apply_manual_weights(self):
        """套用手動設定的權重"""
        weights = {key: var.get() for key, var in self.current_weights.items()}
        
        # 檢查權重總和
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            if messagebox.askyesno("權重警告", 
                                 f"權重總和為 {total:.2f}，不等於1.0\n是否自動歸一化？"):
                weights = {k: v/total for k, v in weights.items()}
                # 更新滑桿值
                for key, value in weights.items():
                    self.current_weights[key].set(value)
            else:
                return
        
        # 套用到主應用程序
        self.main_app.learned_weights = weights
        
        messagebox.showinfo("成功", f"已套用權重配置：\n{self.format_weights(weights)}")
        
        self.log_message(f"套用手動權重：{self.format_weights(weights)}")
    
    def reset_to_default(self):
        """重置為預設權重"""
        default_weights = {"similarity": 0.33, "keyword": 0.33, "self": 0.34}
        for key, value in default_weights.items():
            self.current_weights[key].set(value)
    
    def save_weights_config(self):
        """保存權重配置到文件"""
        try:
            config = {
                'manual_weights': {key: var.get() for key, var in self.current_weights.items()},
                'learner_type': self.learner_type.get(),
                'validation_ratio': self.validation_ratio.get(),
                'learned_weights': self.main_app.learned_weights,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            output_dir = getattr(self.main_app.run_manager, 'get_run_dir', lambda: './output')()
            config_file = os.path.join(output_dir, 'weight_config.json')
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("保存成功", f"權重配置已保存到：\n{config_file}")
            self.log_message(f"權重配置已保存到：{config_file}")
            
        except Exception as e:
            messagebox.showerror("保存失敗", f"保存權重配置時發生錯誤：\n{str(e)}")
    
    def load_saved_weights(self):
        """載入已保存的權重配置"""
        try:
            output_dir = getattr(self.main_app.run_manager, 'get_run_dir', lambda: './output')()
            config_file = os.path.join(output_dir, 'weight_config.json')
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 載入手動權重
                if 'manual_weights' in config:
                    for key, value in config['manual_weights'].items():
                        if key in self.current_weights:
                            self.current_weights[key].set(value)
                
                # 載入其他配置
                if 'learner_type' in config:
                    self.learner_type.set(config['learner_type'])
                if 'validation_ratio' in config:
                    self.validation_ratio.set(config['validation_ratio'])
                if 'learned_weights' in config and config['learned_weights']:
                    self.main_app.learned_weights = config['learned_weights']
                
                self.log_message(f"已載入保存的權重配置")
                
        except Exception as e:
            self.log_message(f"載入權重配置時發生錯誤：{str(e)}")
    
    def start_learning(self):
        """開始智能權重學習"""
        if not ADAPTIVE_LEARNER_AVAILABLE:
            messagebox.showerror("錯誤", "智能權重學習模組未正確安裝\n\n請安裝必要的依賴套件：\npip install optuna matplotlib")
            return
        
        # 檢查是否有可用數據
        if not hasattr(self.main_app, 'working_data') or self.main_app.working_data is None:
            # 提供測試模式
            result = messagebox.askyesno("數據缺失", 
                "沒有找到處理好的數據。\n\n是否使用測試模式？\n測試模式將使用虛擬數據進行演示。")
            if not result:
                return
            self._use_test_mode = True
        else:
            self._use_test_mode = False
        
        # 禁用學習按鈕，啟用停止按鈕
        self.learn_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.learning_progress.start()
        
        # 在新線程中執行學習
        self.learning_thread = threading.Thread(target=self._run_learning, daemon=True)
        self.learning_thread.start()
    
    def _run_learning(self):
        """執行學習的線程函數"""
        try:
            self.log_message("開始智能權重學習...")
            
            # 準備數據
            if self._use_test_mode:
                # 測試模式：使用虛擬數據
                self.log_message("使用測試模式，生成虛擬數據...")
                n_samples = 100
                n_features = 50
                
                # 生成虛擬特徵矩陣
                embeddings = np.random.randn(n_samples, n_features)
                
                # 生成虛擬標籤
                labels = np.random.randint(0, 3, n_samples)
                
                # 生成虛擬metadata
                df = pd.DataFrame({
                    'text': [f'test_text_{i}' for i in range(n_samples)],
                    'sentiment': labels
                })
                
                self.log_message(f"生成了 {n_samples} 條虛擬數據")
            else:
                # 正常模式：使用實際數據
                df = self.main_app.working_data.copy()
                embeddings = self.main_app.working_embeddings
            
            # 準備標籤（這裡假設使用sentiment列作為標籤）
            if 'sentiment' in df.columns:
                labels = df['sentiment'].values
            else:
                # 如果沒有標籤，創建虛擬標籤用於演示
                labels = np.random.randint(0, 3, len(df))
                self.log_message("警告：未找到標籤列，使用隨機標籤進行演示")
            
            # 劃分訓練和驗證集
            from sklearn.model_selection import train_test_split
            
            val_ratio = self.validation_ratio.get()
            X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
                embeddings, labels, range(len(embeddings)), test_size=val_ratio, random_state=42)
            
            # 正確地根據索引劃分metadata
            metadata_train = df.iloc[idx_train].reset_index(drop=True)
            metadata_val = df.iloc[idx_val].reset_index(drop=True)
            
            # 確保metadata包含必要的列
            if 'sentiment' not in metadata_train.columns and 'main_topic' not in metadata_train.columns:
                # 添加標籤列到metadata中
                metadata_train['sentiment'] = y_train
                metadata_val['sentiment'] = y_val
                self.log_message("已將標籤添加到metadata中")
            
            self.log_message(f"數據劃分完成：訓練集 {len(X_train)}, 驗證集 {len(X_val)}")
            
            # 創建權重學習器
            output_dir = getattr(self.main_app.run_manager, 'get_run_dir', lambda: './output')()
            self.weight_learner = AdaptiveWeightLearner(output_dir=output_dir)
            
            # 執行學習
            learner_type = self.learner_type.get()
            self.log_message(f"使用學習器：{learner_type}")
            
            results = self.weight_learner.learn_optimal_weights(
                X_train, y_train, X_val, y_val,
                metadata_train, metadata_val, learner_type)
            
            self.learning_results = results
            
            # 更新UI（在主線程中）
            self.window.after(0, self._update_learning_results, results)
            
        except Exception as e:
            import traceback
            error_msg = f"學習過程中發生錯誤：{str(e)}"
            logger.error(f"智能權重學習錯誤: {error_msg}")
            logger.error(f"錯誤追蹤: {traceback.format_exc()}")
            self.window.after(0, self._handle_learning_error, error_msg)
    
    def _update_learning_results(self, results):
        """更新學習結果（主線程）"""
        self.learning_progress.stop()
        self.learn_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        # 更新結果摘要
        summary = f"""智能權重學習完成！

最佳權重組合：
{self.format_weights(results['best_weights'])}

最佳性能分數：{results['best_score']:.4f}

學習器：{results['learner_name']}
學習時間：{results['learning_time']:.2f} 秒

學習完成時間：{results['timestamp']}
"""
        
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', summary)
        
        # 自動切換到結果頁
        self.notebook.select(2)
        
        self.log_message("智能權重學習完成！")
        self.log_message(f"最佳權重：{self.format_weights(results['best_weights'])}")
        self.log_message(f"最佳分數：{results['best_score']:.4f}")
        
        # 更新視覺化
        self.update_results_visualization(results)
    
    def _handle_learning_error(self, error_msg):
        """處理學習錯誤（主線程）"""
        self.learning_progress.stop()
        self.learn_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        self.log_message(f"錯誤：{error_msg}")
        messagebox.showerror("學習失敗", error_msg)
    
    def stop_learning(self):
        """停止學習"""
        # 這裡可以實現停止學習的邏輯
        self.learning_progress.stop()
        self.learn_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.log_message("學習已停止")
    
    def apply_learned_weights(self):
        """套用學習到的最佳權重"""
        if self.learning_results and 'best_weights' in self.learning_results:
            weights = self.learning_results['best_weights']
            self.main_app.learned_weights = weights
            
            # 更新手動配置頁的滑桿
            for key, value in weights.items():
                if key in self.current_weights:
                    self.current_weights[key].set(value)
            
            messagebox.showinfo("成功", f"已套用學習到的最佳權重：\n{self.format_weights(weights)}")
            self.log_message(f"已套用學習權重：{self.format_weights(weights)}")
        else:
            messagebox.showwarning("警告", "沒有可用的學習結果")
    
    def export_results(self):
        """匯出學習結果"""
        if self.learning_results:
            try:
                output_dir = getattr(self.main_app.run_manager, 'get_run_dir', lambda: './output')()
                export_file = os.path.join(output_dir, f'weight_learning_export_{int(time.time())}.json')
                
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(self.learning_results, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("匯出成功", f"學習結果已匯出到：\n{export_file}")
                self.log_message(f"結果已匯出到：{export_file}")
                
            except Exception as e:
                messagebox.showerror("匯出失敗", f"匯出時發生錯誤：\n{str(e)}")
        else:
            messagebox.showwarning("警告", "沒有可匯出的結果")
    
    def restart_learning(self):
        """重新開始學習"""
        if messagebox.askyesno("確認", "確定要重新開始學習嗎？這將清除當前結果。"):
            self.learning_results = None
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', "等待學習結果...")
            self.notebook.select(1)  # 切換到學習頁
            self.log_message("準備重新開始學習")
    
    def update_results_visualization(self, results):
        """更新結果視覺化"""
        try:
            if hasattr(self, 'ax1') and hasattr(self, 'ax2'):
                # 清除之前的圖表
                self.ax1.clear()
                self.ax2.clear()
                
                # 繪製最佳權重分佈
                weights = list(results['best_weights'].values())
                labels = ['Similarity', 'Keyword', 'Self-Attention']
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                
                self.ax1.pie(weights, labels=labels, colors=colors, autopct='%1.1f%%')
                self.ax1.set_title('Optimal Weight Distribution')
                
                # 繪製學習歷史（如果有）
                if 'learning_history' in results and results['learning_history']:
                    history = results['learning_history']
                    scores = [h['score'] for h in history]
                    iterations = [h['iteration'] for h in history]
                    
                    self.ax2.plot(iterations, scores, 'b-', alpha=0.7)
                    self.ax2.scatter(iterations, scores, c='red', s=20, alpha=0.7)
                    self.ax2.set_xlabel('Iterations')
                    self.ax2.set_ylabel('Performance Score')
                    self.ax2.set_title('Learning Convergence')
                    self.ax2.grid(True, alpha=0.3)
                else:
                    self.ax2.text(0.5, 0.5, 'No learning history available', 
                                ha='center', va='center', transform=self.ax2.transAxes)
                    self.ax2.set_title('Learning History')
                
                self.fig.tight_layout()
                self.canvas.draw()
        except:
            pass  # 忽略視覺化錯誤
    
    def format_weights(self, weights):
        """格式化權重顯示"""
        if weights:
            return "\n".join([f"  {key}: {value:.3f}" for key, value in weights.items()])
        return "無"
    
    def log_message(self, message):
        """記錄日誌訊息"""
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        
        self.learning_log.insert(tk.END, log_entry)
        self.learning_log.see(tk.END)
        
        # 同時輸出到控制台
        logger.info(message)
    
    def on_closing(self):
        """窗口關閉事件"""
        # 停止任何正在進行的學習
        if hasattr(self, 'learning_thread') and self.learning_thread.is_alive():
            self.stop_learning()
        
        self.window.destroy()

# 便利函數
def show_weight_config_window(parent, main_app):
    """顯示權重配置窗口"""
    return WeightConfigWindow(parent, main_app)