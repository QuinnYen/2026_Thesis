"""
報告生成模組
此模組負責生成面向分析的報告
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import time
import datetime
from pathlib import Path
import base64
import io
import logging

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 導入相關模組
from Part03_.utils.config_manager import ConfigManager
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.visualizers import Visualizer

class ReportGenerator:
    """
    報告生成器類
    用於生成面向分析結果的HTML報告
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化報告生成器
        
        Args:
            config: 配置管理器，如果沒有提供則創建新的
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 設置日誌
        log_dir = self.config.get('logging.log_dir', './Part03_/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'report_generator.log')
        
        self.logger = logging.getLogger('report_generator')
        self.logger.setLevel(logging.INFO)
        
        # 移除所有處理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 添加文件處理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # 添加控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        # 設置輸出目錄
        self.output_dir = self.config.get('data_settings.output_directory', './Part03_/results/')
        self.templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        
        # 初始化視覺化工具
        self.visualizer = Visualizer()
    
    def generate_report(self, result_id: str, report_data: Dict[str, Any], 
                      console_output: bool = True) -> str:
        """
        生成HTML報告
        
        Args:
            result_id: 結果ID
            report_data: 報告數據
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            報告文件路徑
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("生成報告")
            logger = ConsoleOutputManager.setup_console_logger("report_generation", log_file)
        else:
            logger = self.logger
        
        try:
            logger.info(f"開始為結果 {result_id} 生成報告...")
            
            # 確保報告包含所需數據
            required_fields = ['dataset', 'description', 'aspect_results']
            for field in required_fields:
                if field not in report_data:
                    error_msg = f"報告數據缺少必要欄位: {field}"
                    logger.error(error_msg)
                    if console_output:
                        ConsoleOutputManager.mark_process_complete(status_file)
                    raise ValueError(error_msg)
            
            # 提取基本信息
            dataset = report_data['dataset']
            description = report_data['description']
            sample_size = report_data.get('sample_size', 'Unknown')
            aspect_results = report_data['aspect_results']
            evaluation = report_data.get('evaluation', {})
            
            # 生成報告文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.output_dir, f"report_{result_id}_{timestamp}.html")
            
            # 生成報告內容
            logger.info("生成報告內容...")
            html_content = self._generate_html_report(
                result_id, dataset, description, sample_size, aspect_results, 
                evaluation, report_data.get('lda_results', {})
            )
            
            # 寫入報告文件
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"報告已生成並保存到: {report_file}")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return report_file
            
        except Exception as e:
            logger.error(f"生成報告時發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def generate_comparison_report(self, result_ids: List[str], 
                                comparison_data: Dict[str, Any],
                                console_output: bool = True) -> str:
        """
        生成模型比較的HTML報告
        
        Args:
            result_ids: 結果ID列表
            comparison_data: 比較數據
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            報告文件路徑
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("生成比較報告")
            logger = ConsoleOutputManager.setup_console_logger("comparison_report", log_file)
        else:
            logger = self.logger
        
        try:
            logger.info(f"開始生成比較報告，比較 {len(result_ids)} 個結果...")
            
            # 生成報告文件名
            result_id_str = "_".join(result_ids)
            if len(result_id_str) > 50:  # 避免文件名過長
                result_id_str = result_id_str[:50]
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(self.output_dir, f"comparison_report_{timestamp}.html")
            
            # 提取比較數據
            results = comparison_data.get('results', [])
            metrics_comparison = comparison_data.get('metrics_comparison', {})
            
            # 生成報告內容
            logger.info("生成比較報告內容...")
            html_content = self._generate_comparison_html(
                result_ids, results, metrics_comparison
            )
            
            # 寫入報告文件
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"比較報告已生成並保存到: {report_file}")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return report_file
            
        except Exception as e:
            logger.error(f"生成比較報告時發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def _generate_html_report(self, result_id: str, dataset: str, description: str, 
                           sample_size: int, aspect_results: Dict[str, Any],
                           evaluation: Dict[str, Any], lda_results: Dict[str, Any]) -> str:
        """
        生成HTML報告內容
        
        Args:
            result_id: 結果ID
            dataset: 數據集名稱
            description: 描述
            sample_size: 樣本大小
            aspect_results: 面向分析結果
            evaluation: 評估結果
            lda_results: LDA主題模型結果
            
        Returns:
            HTML內容
        """
        # 提取面向信息
        aspect_vectors = aspect_results.get('aspect_vectors', {})
        aspect_labels = aspect_results.get('aspect_labels', {})
        
        # 提取評估指標
        metrics = evaluation.get('metrics', {})
        overall_score = metrics.get('overall_score', 0.0)
        
        # 生成圖表
        charts_html = self._generate_chart_sections(aspect_results, evaluation, lda_results)
        
        # 生成詳細分析表格
        aspects_table = self._generate_aspects_table(aspect_vectors, aspect_labels)
        metrics_table = self._generate_metrics_table(metrics)
        
        # 報告生成時間
        generated_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 構建HTML內容
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-tw">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>面向分析報告 - {dataset}</title>
            <style>
                body {{
                    font-family: "Microsoft JhengHei", Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    background-color: #f9f9f9;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    margin-bottom: 30px;
                }}
                h1, h2, h3 {{
                    margin-top: 0;
                }}
                .report-info {{
                    background-color: #f5f5f5;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .score-container {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .score-circle {{
                    width: 150px;
                    height: 150px;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #3498db, #2ecc71);
                    display: inline-flex;
                    justify-content: center;
                    align-items: center;
                    color: white;
                    font-size: 36px;
                    font-weight: bold;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }}
                footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding: 15px;
                    background-color: #2c3e50;
                    color: white;
                }}
                .keyword-badge {{
                    display: inline-block;
                    background-color: #e0f7fa;
                    border-radius: 20px;
                    padding: 4px 10px;
                    margin: 2px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <header>
                <div class="container">
                    <h1>面向分析報告</h1>
                    <p>數據集: {dataset} | 樣本大小: {sample_size}</p>
                </div>
            </header>
            
            <div class="container">
                <div class="report-info">
                    <p><strong>結果ID:</strong> {result_id}</p>
                    <p><strong>描述:</strong> {description}</p>
                    <p><strong>生成時間:</strong> {generated_time}</p>
                </div>
                
                <div class="section">
                    <h2>分析概覽</h2>
                    <div class="score-container">
                        <div class="score-circle">
                            {overall_score:.1f}
                        </div>
                        <p>總體評分 (0-10分)</p>
                    </div>
                    
                    <div class="row">
                        <div class="col">
                            <h3>面向分析指標</h3>
                            {metrics_table}
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>識別到的面向</h2>
                    <p>共識別出 <strong>{len(aspect_labels)}</strong> 個面向:</p>
                    {aspects_table}
                </div>
                
                <!-- 圖表區域 -->
                {charts_html}
                
                <footer>
                    <p>面向分析報告系統 © {datetime.datetime.now().year}</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_chart_sections(self, aspect_results: Dict[str, Any], 
                               evaluation: Dict[str, Any],
                               lda_results: Dict[str, Any]) -> str:
        """
        生成各種圖表的HTML部分
        
        Args:
            aspect_results: 面向分析結果
            evaluation: 評估結果
            lda_results: LDA主題模型結果
            
        Returns:
            包含圖表的HTML
        """
        # 結果圖表
        chart_html = ""
        
        # 使用評估中的視覺化圖表（如果有）
        visualizations = evaluation.get('visualizations', {})
        
        # 添加熱力圖
        if 'heatmap' in visualizations:
            img_data = self._encode_image_file(visualizations['heatmap'])
            if img_data:
                chart_html += f"""
                <div class="section">
                    <h2>文本-面向關聯熱力圖</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{img_data}" alt="文本-面向關聯熱力圖">
                    </div>
                    <p>此熱力圖顯示了文本與各個面向之間的關聯強度。</p>
                </div>
                """
        
        # 添加面向權重圖
        if 'weights_chart' in visualizations:
            img_data = self._encode_image_file(visualizations['weights_chart'])
            if img_data:
                chart_html += f"""
                <div class="section">
                    <h2>面向權重分布</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{img_data}" alt="面向權重分布">
                    </div>
                    <p>此圖顯示了各個面向的平均重要性權重。</p>
                </div>
                """
        
        # 添加情感-面向對應圖（如果有）
        if 'sentiment_chart' in visualizations:
            img_data = self._encode_image_file(visualizations['sentiment_chart'])
            if img_data:
                chart_html += f"""
                <div class="section">
                    <h2>面向與情感的關聯</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{img_data}" alt="面向與情感的關聯">
                    </div>
                    <p>此圖顯示了各個面向與情感極性的相關程度。正值表示與正面情感相關，負值表示與負面情感相關。</p>
                </div>
                """
        
        # 如果沒有圖表，生成一些基本圖表
        if not chart_html and 'aspect_vectors' in aspect_results and 'aspect_labels' in aspect_results:
            # 嘗試生成面向權重分布圖
            try:
                aspect_vectors = aspect_results['aspect_vectors']
                aspect_labels = aspect_results['aspect_labels']
                
                # 準備數據
                labels = [label for _, label in aspect_labels.items()]
                
                # 生成簡單的條形圖
                plt.figure(figsize=(10, 6))
                plt.bar(labels, [1] * len(labels))
                plt.title('面向分布')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # 保存到內存
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                plt.close()
                
                # 編碼為base64
                buf.seek(0)
                img_data = base64.b64encode(buf.read()).decode('utf-8')
                
                chart_html += f"""
                <div class="section">
                    <h2>面向分布</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{img_data}" alt="面向分布">
                    </div>
                    <p>識別出的面向分布概況。</p>
                </div>
                """
                
            except Exception as e:
                self.logger.warning(f"生成面向分布圖時發生錯誤: {str(e)}")
        
        return chart_html
    
    def _generate_aspects_table(self, aspect_vectors: Dict[int, Dict[str, Any]], 
                             aspect_labels: Dict[int, str]) -> str:
        """
        生成面向表格的HTML
        
        Args:
            aspect_vectors: 面向向量字典
            aspect_labels: 面向標籤字典
            
        Returns:
            HTML表格
        """
        table_rows = ""
        
        # 按面向ID排序
        sorted_aspect_ids = sorted([int(k) for k in aspect_labels.keys()])
        
        for aspect_id in sorted_aspect_ids:
            str_aspect_id = str(aspect_id)
            
            # 獲取面向標籤和關鍵詞
            label = aspect_labels.get(str_aspect_id, f"面向{aspect_id}")
            
            keywords = []
            if str_aspect_id in aspect_vectors:
                keywords = aspect_vectors[str_aspect_id].get('keywords', [])
            
            # 處理關鍵詞為徽章樣式
            keyword_badges = ""
            for keyword in keywords:
                keyword_badges += f'<span class="keyword-badge">{keyword}</span> '
            
            # 添加表格行
            table_rows += f"""
            <tr>
                <td>{aspect_id + 1}</td>
                <td>{label}</td>
                <td>{keyword_badges}</td>
            </tr>
            """
        
        # 構建表格
        table_html = f"""
        <table>
            <thead>
                <tr>
                    <th>面向ID</th>
                    <th>標籤</th>
                    <th>關鍵詞</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
        
        return table_html
    
    def _generate_metrics_table(self, metrics: Dict[str, float]) -> str:
        """
        生成指標表格的HTML
        
        Args:
            metrics: 指標字典
            
        Returns:
            HTML表格
        """
        # 定義指標顯示名稱和描述
        metric_display = {
            'overall_score': {'name': '總體評分', 'desc': '面向分析的綜合得分 (0-10分)'},
            'silhouette_score': {'name': '輪廓係數', 'desc': '聚類效果評估指標 (-1到1，越高越好)'},
            'davies_bouldin_score': {'name': 'Davies-Bouldin指數', 'desc': '聚類評估指標 (值越小越好)'},
            'calinski_harabasz_score': {'name': 'Calinski-Harabasz指數', 'desc': '聚類評估指標 (值越大越好)'},
            'aspect_uniqueness': {'name': '面向獨特性', 'desc': '面向間的差異度量 (0-1，越高越好)'},
            'max_aspect_similarity': {'name': '最大面向相似度', 'desc': '最相似的兩個面向之間的相似度'},
            'avg_aspect_similarity': {'name': '平均面向相似度', 'desc': '所有面向對之間的平均相似度'},
            'coverage_ratio': {'name': '覆蓋率', 'desc': '至少有一個顯著面向的文本比例'},
            'avg_significant_aspects': {'name': '平均顯著面向數', 'desc': '每個文本中顯著面向的平均數量'},
            'avg_sentiment_correlation': {'name': '情感相關性', 'desc': '面向與情感的平均相關度'}
        }
        
        table_rows = ""
        
        # 過濾和排序指標，優先顯示重要指標
        important_metrics = ['overall_score', 'aspect_uniqueness', 'coverage_ratio', 
                          'avg_sentiment_correlation', 'silhouette_score']
        
        # 先添加重要指標
        for metric in important_metrics:
            if metric in metrics:
                value = metrics[metric]
                display_info = metric_display.get(metric, {'name': metric, 'desc': ''})
                
                # 格式化數值
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                
                table_rows += f"""
                <tr>
                    <td>{display_info['name']}</td>
                    <td>{formatted_value}</td>
                    <td>{display_info['desc']}</td>
                </tr>
                """
        
        # 再添加其他指標
        for metric, value in metrics.items():
            if metric not in important_metrics:
                display_info = metric_display.get(metric, {'name': metric, 'desc': ''})
                
                # 格式化數值
                formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                
                table_rows += f"""
                <tr>
                    <td>{display_info['name']}</td>
                    <td>{formatted_value}</td>
                    <td>{display_info['desc']}</td>
                </tr>
                """
        
        # 構建表格
        table_html = f"""
        <table>
            <thead>
                <tr>
                    <th>指標</th>
                    <th>數值</th>
                    <th>說明</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
        
        return table_html
    
    def _generate_comparison_html(self, result_ids: List[str], results: List[Dict[str, Any]],
                               metrics_comparison: Dict[str, List]) -> str:
        """
        生成比較報告的HTML
        
        Args:
            result_ids: 結果ID列表
            results: 結果數據列表
            metrics_comparison: 指標比較數據
            
        Returns:
            HTML內容
        """
        # 報告生成時間
        generated_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 提取模型名稱
        model_names = [result.get('dataset_name', f"模型 {i+1}") for i, result in enumerate(results)]
        
        # 生成指標比較表
        metrics_table = self._generate_comparison_metrics_table(metrics_comparison, model_names)
        
        # 生成比較圖表
        charts_html = self._generate_comparison_charts_html(results)
        
        # 構建HTML內容
        html = f"""
        <!DOCTYPE html>
        <html lang="zh-tw">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>面向分析模型比較報告</title>
            <style>
                body {{
                    font-family: "Microsoft JhengHei", Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 0;
                    background-color: #f9f9f9;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    margin-bottom: 30px;
                }}
                h1, h2, h3 {{
                    margin-top: 0;
                }}
                .report-info {{
                    background-color: #f5f5f5;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }}
                footer {{
                    text-align: center;
                    margin-top: 30px;
                    padding: 15px;
                    background-color: #2c3e50;
                    color: white;
                }}
                .best-value {{
                    font-weight: bold;
                    color: #27ae60;
                }}
            </style>
        </head>
        <body>
            <header>
                <div class="container">
                    <h1>面向分析模型比較報告</h1>
                    <p>共比較 {len(results)} 個模型</p>
                </div>
            </header>
            
            <div class="container">
                <div class="report-info">
                    <p><strong>比較模型:</strong> {', '.join(model_names)}</p>
                    <p><strong>生成時間:</strong> {generated_time}</p>
                </div>
                
                <div class="section">
                    <h2>模型評估指標比較</h2>
                    {metrics_table}
                </div>
                
                <!-- 圖表區域 -->
                {charts_html}
                
                <footer>
                    <p>面向分析報告系統 © {datetime.datetime.now().year}</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_comparison_metrics_table(self, metrics_comparison: Dict[str, List], 
                                        model_names: List[str]) -> str:
        """
        生成比較指標表格的HTML
        
        Args:
            metrics_comparison: 指標比較數據
            model_names: 模型名稱列表
            
        Returns:
            HTML表格
        """
        # 定義指標顯示名稱
        metric_display = {
            'overall_score': '總體評分',
            'silhouette_score': '輪廓係數',
            'davies_bouldin_score': 'Davies-Bouldin指數',
            'calinski_harabasz_score': 'Calinski-Harabasz指數',
            'aspect_uniqueness': '面向獨特性',
            'coverage_ratio': '覆蓋率',
            'avg_sentiment_correlation': '情感相關性'
        }
        
        # 定義如何比較指標值（越大越好？越小越好？）
        # True表示越大越好，False表示越小越好
        metric_comparison = {
            'overall_score': True,
            'silhouette_score': True,
            'davies_bouldin_score': False,
            'calinski_harabasz_score': True,
            'aspect_uniqueness': True,
            'coverage_ratio': True,
            'avg_sentiment_correlation': True
        }
        
        # 構建表頭
        header = "<tr><th>指標</th>"
        for name in model_names:
            header += f"<th>{name}</th>"
        header += "</tr>"
        
        # 構建表體
        table_rows = ""
        
        # 重要指標優先
        important_metrics = ['overall_score', 'aspect_uniqueness', 'coverage_ratio', 
                          'avg_sentiment_correlation', 'silhouette_score']
        
        # 處理所有指標
        all_metrics = list(metrics_comparison.keys())
        # 先處理重要指標
        for metric in important_metrics:
            if metric in all_metrics:
                all_metrics.remove(metric)
                
                # 獲取顯示名稱
                display_name = metric_display.get(metric, metric)
                
                # 創建行
                row = f"<tr><td>{display_name}</td>"
                
                # 獲取該指標的所有值
                values = []
                for model_data in metrics_comparison[metric]:
                    value = model_data.get('value')
                    values.append(value if value is not None else None)
                
                # 確定最佳值
                best_value = None
                if values and any(v is not None for v in values):
                    is_greater_better = metric_comparison.get(metric, True)
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        best_value = max(valid_values) if is_greater_better else min(valid_values)
                
                # 添加每個模型的值
                for value in values:
                    if value is None:
                        row += "<td>-</td>"
                    elif value == best_value:
                        row += f"<td class='best-value'>{value:.4f}</td>"
                    else:
                        row += f"<td>{value:.4f}</td>"
                
                row += "</tr>"
                table_rows += row
        
        # 處理剩餘指標
        for metric in all_metrics:
            # 獲取顯示名稱
            display_name = metric_display.get(metric, metric)
            
            # 創建行
            row = f"<tr><td>{display_name}</td>"
            
            # 獲取該指標的所有值
            values = []
            for model_data in metrics_comparison[metric]:
                value = model_data.get('value')
                values.append(value if value is not None else None)
            
            # 確定最佳值
            best_value = None
            if values and any(v is not None for v in values):
                is_greater_better = metric_comparison.get(metric, True)
                valid_values = [v for v in values if v is not None]
                if valid_values:
                    best_value = max(valid_values) if is_greater_better else min(valid_values)
            
            # 添加每個模型的值
            for value in values:
                if value is None:
                    row += "<td>-</td>"
                elif value == best_value:
                    row += f"<td class='best-value'>{value:.4f}</td>"
                else:
                    row += f"<td>{value:.4f}</td>"
            
            row += "</tr>"
            table_rows += row
        
        # 構建表格
        table_html = f"""
        <table>
            <thead>
                {header}
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """
        
        return table_html
    
    def _generate_comparison_charts_html(self, results: List[Dict[str, Any]]) -> str:
        """
        生成比較圖表的HTML
        
        Args:
            results: 結果數據列表
            
        Returns:
            HTML內容
        """
        charts_html = ""
        
        # 檢查每個結果中的視覺化
        radar_charts = []
        score_charts = []
        
        for result in results:
            evaluation = result.get('evaluation', {})
            visualizations = evaluation.get('visualizations', {})
            
            # 收集雷達圖
            if 'radar_chart' in visualizations:
                radar_charts.append(visualizations['radar_chart'])
            
            # 收集得分圖
            if 'score_chart' in visualizations:
                score_charts.append(visualizations['score_chart'])
        
        # 添加雷達圖
        if radar_charts:
            img_data = self._encode_image_file(radar_charts[0])
            if img_data:
                charts_html += f"""
                <div class="section">
                    <h2>模型評估指標雷達圖</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{img_data}" alt="模型評估雷達圖">
                    </div>
                    <p>此圖顯示了各個模型在不同評估指標上的表現比較。</p>
                </div>
                """
        
        # 添加得分圖
        if score_charts:
            img_data = self._encode_image_file(score_charts[0])
            if img_data:
                charts_html += f"""
                <div class="section">
                    <h2>模型總體得分比較</h2>
                    <div class="chart-container">
                        <img src="data:image/png;base64,{img_data}" alt="模型得分比較">
                    </div>
                    <p>此圖顯示了各個模型的總體評估得分比較。</p>
                </div>
                """
        
        # 如果沒有現成的圖表，嘗試生成一個簡單的比較圖表
        if not charts_html:
            try:
                # 提取模型名稱和總體評分
                model_names = []
                scores = []
                
                for result in results:
                    name = result.get('dataset_name', 'Unknown')
                    model_names.append(name)
                    
                    evaluation = result.get('evaluation', {})
                    metrics = evaluation.get('metrics', {})
                    score = metrics.get('overall_score', 0.0)
                    scores.append(score)
                
                if model_names and scores:
                    # 生成簡單的條形圖
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(model_names, scores)
                    plt.title('模型總體得分比較')
                    plt.ylabel('得分')
                    plt.ylim(0, 10)
                    
                    # 添加數值標籤
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                               f'{height:.2f}', ha='center', va='bottom')
                    
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # 保存到內存
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=300)
                    plt.close()
                    
                    # 編碼為base64
                    buf.seek(0)
                    img_data = base64.b64encode(buf.read()).decode('utf-8')
                    
                    charts_html += f"""
                    <div class="section">
                        <h2>模型總體得分比較</h2>
                        <div class="chart-container">
                            <img src="data:image/png;base64,{img_data}" alt="模型得分比較">
                        </div>
                        <p>此圖顯示了各個模型的總體評估得分比較。</p>
                    </div>
                    """
            except Exception as e:
                self.logger.warning(f"生成簡單比較圖表時發生錯誤: {str(e)}")
        
        return charts_html
    
    def _encode_image_file(self, image_path: str) -> Optional[str]:
        """
        將圖像文件編碼為Base64
        
        Args:
            image_path: 圖像文件路徑
            
        Returns:
            Base64編碼字符串，如果出錯則返回None
        """
        try:
            if not os.path.exists(image_path):
                self.logger.warning(f"圖像文件不存在: {image_path}")
                return None
            
            with open(image_path, 'rb') as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded
        except Exception as e:
            self.logger.warning(f"編碼圖像時發生錯誤: {str(e)}")
            return None

# 使用示例
if __name__ == "__main__":
    # 初始化報告生成器
    report_gen = ReportGenerator()
    
    # 模擬數據
    import numpy as np
    
    # 創建模擬數據框
    data = {
        'id': [f"text_{i}" for i in range(10)],
        'text': [f"Sample text {i}" for i in range(10)],
        'processed_text': [f"sample text {i}" for i in range(10)],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative',
                    'neutral', 'positive', 'negative', 'positive', 'neutral']
    }
    df = pd.DataFrame(data)
    
    # 創建模擬面向結果
    aspect_vectors = {
        0: {'vector': np.random.random(10), 'keywords': ['price', 'cost'], 'label': '價格'},
        1: {'vector': np.random.random(10), 'keywords': ['quality', 'good'], 'label': '質量'},
        2: {'vector': np.random.random(10), 'keywords': ['service', 'staff'], 'label': '服務'}
    }
    
    aspect_labels = {0: '價格', 1: '質量', 2: '服務'}
    
    text_aspect_matrix = np.random.random((10, 3))
    text_aspect_matrix = text_aspect_matrix / text_aspect_matrix.sum(axis=1, keepdims=True)
    
    aspect_results = {
        'aspect_vectors': aspect_vectors,
        'aspect_labels': aspect_labels,
        'text_aspect_matrix': text_aspect_matrix
    }
    
    # 創建模擬評估結果
    evaluation = {
        'metrics': {
            'overall_score': 8.5,
            'aspect_uniqueness': 0.85,
            'coverage_ratio': 0.92,
            'silhouette_score': 0.73
        }
    }
    
    # 生成報告
    report_data = {
        'dataset': 'IMDB_Test',
        'description': '測試報告生成',
        'sample_size': 10,
        'processed_data': df,
        'aspect_results': aspect_results,
        'evaluation': evaluation
    }
    
    # 生成並保存報告
    report_file = report_gen.generate_report('test_result', report_data)
    print(f"報告已生成: {report_file}")
