"""
評估模組 - 負責評估不同注意力機制的效能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import json
import pickle
import logging
import time
import io
from datetime import datetime

# 導入系統模組
from utils.logger import get_logger

# 獲取logger
logger = get_logger("evaluator")

class AttentionEvaluator:
    """評估不同注意力機制的效能"""
    
    def __init__(self, config=None):
        """初始化評估器
        
        Args:
            config: 配置參數字典，可包含以下鍵:
                - output_dir: 輸出目錄
                - visualizations: 是否生成可視化
                - report_format: 報告格式('html', 'json', 'csv')
        """
        self.config = config or {}
        self.logger = logger
        
        # 設置默認配置
        self.output_dir = self.config.get('output_dir', './output/evaluation')
        self.enable_visualizations = self.config.get('visualizations', True)
        self.report_format = self.config.get('report_format', 'html')
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 創建可視化目錄
        self.vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def evaluate_attention_mechanisms(self, result_paths, progress_callback=None):
        """評估多種注意力機制的效能
        
        Args:
            result_paths: 包含不同注意力機制結果的字典，格式為:
                {attention_type: {vectors_path: path, metadata_path: path}}
            progress_callback: 進度回調函數
            
        Returns:
            dict: 包含評估結果的字典
        """
        try:
            start_time = time.time()
            self.logger.info(f"開始評估注意力機制效能")
            
            # 加載所有結果
            self.logger.info(f"加載注意力機制結果...")
            if progress_callback:
                progress_callback("加載注意力機制結果...", 10)
                
            attention_results = {}
            
            for attention_type, paths in result_paths.items():
                # 檢查所需路徑是否都存在
                if 'metadata_path' not in paths or not os.path.exists(paths['metadata_path']):
                    self.logger.warning(f"{attention_type} 的元數據文件不存在，跳過")
                    continue
                
                # 加載結果元數據
                try:
                    with open(paths['metadata_path'], 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        
                    attention_results[attention_type] = metadata
                except Exception as e:
                    self.logger.error(f"加載 {attention_type} 元數據時出錯: {str(e)}")
                    continue
            
            if not attention_results:
                self.logger.error("沒有找到有效的注意力機制結果")
                return None
                
            self.logger.info(f"成功加載 {len(attention_results)} 種注意力機制的結果")
            
            # 提取評估指標
            self.logger.info(f"提取評估指標...")
            if progress_callback:
                progress_callback("提取評估指標...", 30)
                
            metrics_df = self._create_metrics_dataframe(attention_results)
            
            # 找到最佳注意力機制
            self.logger.info(f"識別最佳注意力機制...")
            if progress_callback:
                progress_callback("識別最佳注意力機制...", 40)
                
            best_attention = self._identify_best_attention(metrics_df)
            
            self.logger.info(f"最佳注意力機制: {best_attention['type']}")
            self.logger.info(f"內聚度: {best_attention['coherence']:.4f}")
            self.logger.info(f"分離度: {best_attention['separation']:.4f}")
            self.logger.info(f"綜合得分: {best_attention['combined_score']:.4f}")
            
            # 生成可視化
            visualizations = []
            
            if self.enable_visualizations:
                self.logger.info(f"生成評估可視化...")
                if progress_callback:
                    progress_callback("生成評估可視化...", 50)
                
                # 產生文件名前綴
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                prefix = f"attention_eval_{timestamp}"
                
                # 條形圖比較
                bar_chart_path = self._create_bar_comparison(metrics_df, prefix)
                if bar_chart_path:
                    visualizations.append(bar_chart_path)
                
                # 雷達圖比較
                radar_chart_path = self._create_radar_chart(metrics_df, prefix)
                if radar_chart_path:
                    visualizations.append(radar_chart_path)
                
                # 熱力圖比較
                heatmap_path = self._create_heatmap(metrics_df, prefix)
                if heatmap_path:
                    visualizations.append(heatmap_path)
                
                # 3D比較圖
                threed_chart_path = self._create_3d_comparison(metrics_df, prefix)
                if threed_chart_path:
                    visualizations.append(threed_chart_path)
            
            # 生成評估報告
            self.logger.info(f"生成評估報告...")
            if progress_callback:
                progress_callback("生成評估報告...", 80)
                
            report_path = self._generate_report(metrics_df, best_attention, visualizations, prefix)
            
            # 計算處理時間
            elapsed_time = time.time() - start_time
            self.logger.info(f"評估完成，耗時: {elapsed_time:.2f} 秒")
            
            # 完成並返回結果
            if progress_callback:
                progress_callback("評估完成", 100)
                
            return {
                'metrics_df': metrics_df,
                'best_attention': best_attention,
                'visualizations': visualizations,
                'report_path': report_path
            }
            
        except Exception as e:
            self.logger.error(f"評估注意力機制時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _create_metrics_dataframe(self, attention_results):
        """從注意力機制結果中提取評估指標並創建DataFrame
        
        Args:
            attention_results: 注意力機制結果字典
            
        Returns:
            pd.DataFrame: 評估指標DataFrame
        """
        # 創建數據列表
        data = []
        
        for attention_type, result in attention_results.items():
            metrics = result.get('metrics', {})
            
            data.append({
                'attention_type': attention_type,
                'coherence': metrics.get('coherence', 0.0),
                'separation': metrics.get('separation', 0.0),
                'combined_score': metrics.get('combined_score', 0.0)
            })
        
        # 創建DataFrame
        metrics_df = pd.DataFrame(data)
        
        # 排序以便後續使用
        metrics_df = metrics_df.sort_values('combined_score', ascending=False)
        
        return metrics_df

    def _identify_best_attention(self, metrics_df):
        """識別最佳注意力機制
        
        Args:
            metrics_df: 評估指標DataFrame
            
        Returns:
            dict: 最佳注意力機制信息
        """
        # 找到綜合得分最高的機制
        best_row = metrics_df.iloc[0]
        
        return {
            'type': best_row['attention_type'],
            'coherence': best_row['coherence'],
            'separation': best_row['separation'],
            'combined_score': best_row['combined_score']
        }
    
    def _create_bar_comparison(self, metrics_df, prefix):
        """創建條形圖比較
        
        Args:
            metrics_df: 評估指標DataFrame
            prefix: 文件名前綴
            
        Returns:
            str: 圖片路徑
        """
        try:
            # 創建圖表
            plt.figure(figsize=(15, 10))
            
            # 獲取注意力類型和指標
            attention_types = metrics_df['attention_type'].tolist()
            coherence_scores = metrics_df['coherence'].tolist()
            separation_scores = metrics_df['separation'].tolist()
            combined_scores = metrics_df['combined_score'].tolist()
            
            # 設置柱的位置
            x = np.arange(len(attention_types))
            width = 0.25
            
            # 繪製三組柱狀圖
            ax = plt.subplot(111)
            
            # 內聚度
            bars1 = ax.bar(x - width, coherence_scores, width, label='內聚度')
            
            # 分離度
            bars2 = ax.bar(x, separation_scores, width, label='分離度')
            
            # 綜合得分
            bars3 = ax.bar(x + width, combined_scores, width, label='綜合得分')
            
            # 添加標籤、標題等
            ax.set_xlabel('注意力機制類型', fontsize=14)
            ax.set_ylabel('評估指標分數', fontsize=14)
            ax.set_title('不同注意力機制的評估指標比較', fontsize=16)
            ax.set_xticks(x)
            ax.set_xticklabels(attention_types)
            ax.legend()
            
            # 添加數值標籤
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.4f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom', rotation=45, fontsize=9)
            
            add_labels(bars1)
            add_labels(bars2)
            add_labels(bars3)
            
            plt.tight_layout()
            
            # 保存圖片
            output_path = os.path.join(self.vis_dir, f"{prefix}_bar_comparison.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"創建條形圖比較時出錯: {str(e)}")
            return None
    
    def _create_radar_chart(self, metrics_df, prefix):
        """創建雷達圖
        
        Args:
            metrics_df: 評估指標DataFrame
            prefix: 文件名前綴
            
        Returns:
            str: 圖片路徑
        """
        try:
            # 創建圖表
            plt.figure(figsize=(10, 8))
            
            # 獲取注意力類型和指標
            attention_types = metrics_df['attention_type'].tolist()
            
            # 準備雷達圖數據
            metrics = ['內聚度', '分離度', '綜合得分']
            num_metrics = len(metrics)
            
            # 計算角度
            angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
            angles += angles[:1]  # 閉合雷達圖
            
            # 初始化雷達圖
            ax = plt.subplot(111, polar=True)
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=12)
            
            # 繪製每種注意力機制的雷達圖
            for i, attention_type in enumerate(attention_types):
                row = metrics_df[metrics_df['attention_type'] == attention_type].iloc[0]
                values = [
                    row['coherence'],
                    row['separation'],
                    row['combined_score']
                ]
                values += values[:1]  # 閉合雷達圖
                
                ax.plot(angles, values, linewidth=2, label=attention_type)
                ax.fill(angles, values, alpha=0.1)
            
            # 添加標題和圖例
            plt.title('注意力機制評估指標雷達圖', size=15)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            # 調整佈局
            plt.tight_layout()
            
            # 保存圖片
            output_path = os.path.join(self.vis_dir, f"{prefix}_radar_chart.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"創建雷達圖時出錯: {str(e)}")
            return None
    
    def _create_heatmap(self, metrics_df, prefix):
        """創建熱力圖
        
        Args:
            metrics_df: 評估指標DataFrame
            prefix: 文件名前綴
            
        Returns:
            str: 圖片路徑
        """
        try:
            # 創建圖表
            plt.figure(figsize=(12, 8))
            
            # 準備熱力圖數據
            pivot_df = metrics_df.set_index('attention_type')
            pivot_df = pivot_df[['coherence', 'separation', 'combined_score']]
            
            # 創建自定義顏色映射（從藍色到紅色）
            cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#4575B4', '#FFFFBF', '#D73027'])
            
            # 繪製熱力圖
            sns.heatmap(pivot_df, annot=True, cmap=cmap, fmt='.4f',
                      linewidths=0.5, cbar_kws={'label': '指標分數'})
            
            plt.title('注意力機制評估指標熱力圖', fontsize=16)
            plt.xlabel('評估指標', fontsize=14)
            plt.ylabel('注意力機制', fontsize=14)
            
            plt.tight_layout()
            
            # 保存圖片
            output_path = os.path.join(self.vis_dir, f"{prefix}_heatmap.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"創建熱力圖時出錯: {str(e)}")
            return None
    
    def _create_3d_comparison(self, metrics_df, prefix):
        """創建3D比較圖
        
        Args:
            metrics_df: 評估指標DataFrame
            prefix: 文件名前綴
            
        Returns:
            str: 圖片路徑
        """
        try:
            # 創建圖表
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # 獲取數據
            attention_types = metrics_df['attention_type'].tolist()
            coherence = metrics_df['coherence'].values
            separation = metrics_df['separation'].values
            combined = metrics_df['combined_score'].values
            
            # 創建顏色映射
            colors = plt.cm.viridis(np.linspace(0, 1, len(attention_types)))
            
            # 繪製3D散點圖
            scatter = ax.scatter(coherence, separation, combined, c=colors, s=100, alpha=0.6)
            
            # 添加標籤
            for i, txt in enumerate(attention_types):
                ax.text(coherence[i], separation[i], combined[i], txt,
                      size=10, zorder=1, color='k')
            
            # 設置軸標籤和標題
            ax.set_xlabel('內聚度', fontsize=12)
            ax.set_ylabel('分離度', fontsize=12)
            ax.set_zlabel('綜合得分', fontsize=12)
            ax.set_title('注意力機制評估指標3D比較', fontsize=14)
            
            # 添加網格線
            ax.grid(True)
            
            # 保存圖片
            output_path = os.path.join(self.vis_dir, f"{prefix}_3d_comparison.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"創建3D比較圖時出錯: {str(e)}")
            return None
    
    def _generate_report(self, metrics_df, best_attention, visualizations, prefix):
        """生成評估報告
        
        Args:
            metrics_df: 評估指標DataFrame
            best_attention: 最佳注意力機制信息
            visualizations: 可視化圖片路徑列表
            prefix: 文件名前綴
            
        Returns:
            str: 報告文件路徑
        """
        if self.report_format == 'html':
            return self._generate_html_report(metrics_df, best_attention, visualizations, prefix)
        elif self.report_format == 'json':
            return self._generate_json_report(metrics_df, best_attention, visualizations, prefix)
        elif self.report_format == 'csv':
            return self._generate_csv_report(metrics_df, best_attention, visualizations, prefix)
        else:
            self.logger.warning(f"不支持的報告格式: {self.report_format}，使用HTML")
            return self._generate_html_report(metrics_df, best_attention, visualizations, prefix)
    
    def _generate_html_report(self, metrics_df, best_attention, visualizations, prefix):
        """生成HTML格式的評估報告
        
        Args:
            metrics_df: 評估指標DataFrame
            best_attention: 最佳注意力機制信息
            visualizations: 可視化圖片路徑列表
            prefix: 文件名前綴
            
        Returns:
            str: 報告文件路徑
        """
        try:
            # 創建HTML報告
            html = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>注意力機制評估報告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .highlight {{ background-color: #e6ffe6; }}
                    .container {{ display: flex; flex-wrap: wrap; justify-content: space-around; }}
                    .image-container {{ margin: 10px; max-width: 45%; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .footer {{ margin-top: 30px; text-align: center; font-size: 0.8em; color: #777; }}
                </style>
            </head>
            <body>
                <h1>注意力機制評估報告</h1>
                <p>生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h2>評估摘要</h2>
                    <p><strong>最佳注意力機制:</strong> {best_attention['type']}</p>
                    <p><strong>內聚度:</strong> {best_attention['coherence']:.4f}</p>
                    <p><strong>分離度:</strong> {best_attention['separation']:.4f}</p>
                    <p><strong>綜合得分:</strong> {best_attention['combined_score']:.4f}</p>
                </div>
                
                <h2>評估指標詳情</h2>
                <table>
                    <tr>
                        <th>注意力機制</th>
                        <th>內聚度</th>
                        <th>分離度</th>
                        <th>綜合得分</th>
                    </tr>
            '''
            
            # 添加每個注意力機制的結果
            for _, row in metrics_df.iterrows():
                highlight = 'highlight' if row['attention_type'] == best_attention['type'] else ''
                html += f'''
                    <tr class="{highlight}">
                        <td>{row['attention_type']}</td>
                        <td>{row['coherence']:.4f}</td>
                        <td>{row['separation']:.4f}</td>
                        <td>{row['combined_score']:.4f}</td>
                    </tr>
                '''
            
            html += '''
                </table>
                
                <h2>評估可視化</h2>
                <div class="container">
            '''
            
            # 添加所有可視化圖片
            for i, vis_path in enumerate(visualizations):
                vis_name = os.path.basename(vis_path).replace(f"{prefix}_", "").replace(".png", "")
                vis_title = vis_name.replace("_", " ").title()
                
                html += f'''
                    <div class="image-container">
                        <h3>{vis_title}</h3>
                        <img src="file://{os.path.abspath(vis_path)}" alt="{vis_title}">
                    </div>
                '''
            
            html += '''
                </div>
                
                <h2>分析與建議</h2>
                <p>基於上述評估指標和可視化結果，我們可以得出以下結論：</p>
                <ul>
            '''
            
            # 添加關於最佳注意力機制的分析
            best_type = best_attention['type']
            
            if best_type == 'no' or best_type == 'none':
                html += '''
                    <li>無注意力機制（簡單平均）在本次評估中表現最好，這可能意味著數據集中的主題區分較為明確，不需要複雜的注意力機制。</li>
                    <li>對於這種情況，可以考慮使用更簡單的模型或算法來降低計算成本。</li>
                '''
            elif best_type == 'similarity':
                html += '''
                    <li>基於相似度的注意力機制在本次評估中表現最好，這表明文檔與主題中心的相似性是一個很好的權重指標。</li>
                    <li>這種機制在處理語義相似度較高的文檔群組時特別有效。</li>
                '''
            elif best_type == 'keyword':
                html += '''
                    <li>基於關鍵詞的注意力機制在本次評估中表現最好，這表明主題關鍵詞在區分不同面向時起著重要作用。</li>
                    <li>這種機制在領域特定的文本和有明確關鍵詞的場景中效果較好。</li>
                '''
            elif best_type == 'self':
                html += '''
                    <li>自注意力機制在本次評估中表現最好，這表明文檔間的相互關係對確定其重要性有幫助。</li>
                    <li>這種機制能夠捕捉文檔集合中的複雜關係，特別適合處理結構化或上下文相關的文本。</li>
                '''
            elif best_type == 'combined':
                html += '''
                    <li>組合注意力機制在本次評估中表現最好，這表明多種注意力機制的結合能夠更全面地捕捉文檔特徵。</li>
                    <li>這種機制通過結合不同類型的注意力，能夠兼顧文檔與主題中心的相似性、關鍵詞匹配度和文檔間的相互關係。</li>
                '''
            
            # 添加一般建議
            html += '''
                    <li>內聚度和分離度是評估面向向量質量的重要指標，應同時考慮這兩個方面。</li>
                    <li>在實際應用中，可以根據具體任務的需求調整不同注意力機制的權重。</li>
                </ul>
                
                <div class="footer">
                    <p>此報告由跨領域情感分析系統自動生成</p>
                </div>
            </body>
            </html>
            '''
            
            # 保存HTML報告
            output_path = os.path.join(self.output_dir, f"{prefix}_report.html")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            return output_path
        except Exception as e:
            self.logger.error(f"生成HTML報告時出錯: {str(e)}")
            return None
    
    def _generate_json_report(self, metrics_df, best_attention, visualizations, prefix):
        """生成JSON格式的評估報告
        
        Args:
            metrics_df: 評估指標DataFrame
            best_attention: 最佳注意力機制信息
            visualizations: 可視化圖片路徑列表
            prefix: 文件名前綴
            
        Returns:
            str: 報告文件路徑
        """
        try:
            # 準備JSON數據
            report_data = {
                'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'best_attention': best_attention,
                'metrics': metrics_df.to_dict('records'),
                'visualizations': [os.path.abspath(path) for path in visualizations]
            }
            
            # 保存JSON報告
            output_path = os.path.join(self.output_dir, f"{prefix}_report.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            return output_path
        except Exception as e:
            self.logger.error(f"生成JSON報告時出錯: {str(e)}")
            return None
    
    def _generate_csv_report(self, metrics_df, best_attention, visualizations, prefix):
        """生成CSV格式的評估報告
        
        Args:
            metrics_df: 評估指標DataFrame
            best_attention: 最佳注意力機制信息
            visualizations: 可視化圖片路徑列表
            prefix: 文件名前綴
            
        Returns:
            str: 報告文件路徑
        """
        try:
            # 直接保存指標DataFrame
            output_path = os.path.join(self.output_dir, f"{prefix}_report.csv")
            metrics_df.to_csv(output_path, index=False)
            
            # 同時保存一個元數據文件
            meta_path = os.path.join(self.output_dir, f"{prefix}_meta.json")
            meta_data = {
                'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'best_attention': best_attention,
                'visualizations': [os.path.abspath(path) for path in visualizations]
            }
            
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
            
            return output_path
        except Exception as e:
            self.logger.error(f"生成CSV報告時出錯: {str(e)}")
            return None

# 添加 Evaluator 作為 AttentionEvaluator 的別名，以保持向後相容性
Evaluator = AttentionEvaluator

def evaluate_attention_results(result_paths, output_dir='./output/evaluation', 
                             report_format='html', progress_callback=None):
    """評估注意力機制結果的便捷函數
    
    Args:
        result_paths: 包含不同注意力機制結果的字典
        output_dir: 輸出目錄
        report_format: 報告格式
        progress_callback: 進度回調函數
        
    Returns:
        dict: 包含評估結果的字典
    """
    # 創建配置
    config = {
        'output_dir': output_dir,
        'visualizations': True,
        'report_format': report_format
    }
    
    # 創建評估器
    evaluator = AttentionEvaluator(config)
    
    # 評估注意力機制
    return evaluator.evaluate_attention_mechanisms(result_paths, progress_callback)


# 測試代碼
if __name__ == "__main__":
    # 配置日誌級別
    logger.setLevel(logging.INFO)
    
    # 模擬結果路徑
    result_paths = {
        'no': {
            'vectors_path': 'path/to/no_aspect_vectors.npz',
            'metadata_path': 'path/to/no_aspect_metadata.json'
        },
        'similarity': {
            'vectors_path': 'path/to/similarity_aspect_vectors.npz',
            'metadata_path': 'path/to/similarity_aspect_metadata.json'
        },
        # 其他注意力機制...
    }
    
    # 檢查文件是否存在
    missing_files = False
    for attn_type, paths in result_paths.items():
        if 'metadata_path' in paths and not os.path.exists(paths['metadata_path']):
            logger.warning(f"{paths['metadata_path']} 不存在")
            missing_files = True
    
    if missing_files:
        logger.warning("測試文件不存在，此處僅為示例代碼")
    else:
        # 定義進度回調函數
        def progress_callback(message, percentage):
            print(f"{message} - {percentage}%")
        
        # 評估注意力機制
        result = evaluate_attention_results(
            result_paths,
            progress_callback=progress_callback
        )
        
        if result:
            print(f"最佳注意力機制: {result['best_attention']['type']}")
            print(f"綜合得分: {result['best_attention']['combined_score']:.4f}")
            print(f"報告已保存至: {result['report_path']}")
        else:
            print("評估失敗")