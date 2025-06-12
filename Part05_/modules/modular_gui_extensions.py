"""
GUI擴展模組 - 模組化流水線相關的GUI方法
這些方法將被整合到主GUI類中
"""

import tkinter as tk
from tkinter import messagebox
import threading
import queue
import time
import pandas as pd
import os

def update_current_config(self):
    """更新當前配置顯示"""
    if hasattr(self, 'current_config_label'):
        encoder = self.encoder_type.get().upper()
        aspect = self.aspect_classifier_type.get().upper()
        classifier = self.classifier_type.get().upper()
        config_text = f"📝 當前配置: {encoder} + {aspect} + {classifier}"
        self.current_config_label.config(text=config_text)

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


# 將這些方法添加到MainApplication類的方法字典中
MODULAR_METHODS = {
    'update_current_config': update_current_config,
    'run_modular_pipeline': run_modular_pipeline,
    '_run_modular_pipeline': _run_modular_pipeline,
    '_check_pipeline_progress': _check_pipeline_progress,
    'compare_methods': compare_methods,
    '_run_method_comparison': _run_method_comparison
}