import os
import json
import shutil
import logging
import pandas as pd
from datetime import datetime

class ResultManager:
    """管理和組織分析結果的工具類"""
    
    def __init__(self, base_dir='./Part02_/results', logger=None):
        """
        初始化結果管理器
        
        Args:
            base_dir: 結果基礎目錄
            logger: 日誌記錄器
        """
        self.base_dir = base_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化子目錄
        self.processed_data_dir = os.path.join(base_dir, "01_processed_data")
        self.bert_embeddings_dir = os.path.join(base_dir, "02_bert_embeddings")
        self.lda_topics_dir = os.path.join(base_dir, "03_lda_topics")
        self.aspect_vectors_dir = os.path.join(base_dir, "04_aspect_vectors")
        self.visualizations_dir = os.path.join(base_dir, "visualizations")
        self.exports_dir = os.path.join(base_dir, "exports")
        self.models_dir = os.path.join(base_dir, "models")
        
        # 可視化子目錄
        self.topic_vis_dir = os.path.join(self.visualizations_dir, "topics")
        self.vector_vis_dir = os.path.join(self.visualizations_dir, "vectors")
        
        # 創建所有必要的目錄
        self._ensure_directories()
        
        # 初始化結果索引文件
        self.index_path = os.path.join(base_dir, "results_index.json")
        self._init_index()
    
    def _ensure_directories(self):
        """確保所有必要的目錄都存在"""
        for directory in [
            self.base_dir, 
            self.processed_data_dir, self.bert_embeddings_dir, 
            self.lda_topics_dir, self.aspect_vectors_dir,
            self.visualizations_dir, self.exports_dir, self.models_dir,
            self.topic_vis_dir, self.vector_vis_dir
        ]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.info(f"創建目錄: {directory}")
    
    def _init_index(self):
        """初始化或加載結果索引"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                self.logger.info(f"已加載結果索引: {self.index_path}")
            except Exception as e:
                self.logger.error(f"讀取結果索引時出錯: {str(e)}")
                self.index = {"datasets": {}}
        else:
            self.index = {"datasets": {}}
    
    def _save_index(self):
        """保存結果索引"""
        try:
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已保存結果索引: {self.index_path}")
        except Exception as e:
            self.logger.error(f"保存結果索引時出錯: {str(e)}")
    
    def register_dataset(self, dataset_name, source_path):
        """
        註冊一個新的數據集
        
        Args:
            dataset_name: 數據集名稱
            source_path: 原始數據文件路徑
            
        Returns:
            dataset_id: 數據集ID（用於後續引用）
        """
        # 生成唯一ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_id = f"{dataset_name}_{timestamp}"
        
        # 更新索引
        self.index["datasets"][dataset_id] = {
            "name": dataset_name,
            "source_path": source_path,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "steps": {},
            "summary": {
                "status": "initialized",
                "message": "數據集已導入"
            }
        }
        self._save_index()
        
        return dataset_id
    
    def register_result(self, dataset_id, step_name, result_type, file_path, metadata=None):
        """
        註冊一個處理結果
        
        Args:
            dataset_id: 數據集ID
            step_name: 處理步驟名稱（如'bert_embedding', 'lda_topic'）
            result_type: 結果類型（如'data', 'model', 'visualization'）
            file_path: 結果文件路徑
            metadata: 結果相關的元數據
            
        Returns:
            bool: 是否成功註冊
        """
        if dataset_id not in self.index["datasets"]:
            self.logger.error(f"數據集ID不存在: {dataset_id}")
            return False
        
        # 將文件移動到適當的目錄（如果需要）
        dest_path = self._organize_file(file_path, step_name, result_type)
        
        # 更新索引
        if step_name not in self.index["datasets"][dataset_id]["steps"]:
            self.index["datasets"][dataset_id]["steps"][step_name] = {
                "status": "completed",
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": []
            }
        
        # 添加結果記錄
        result_record = {
            "type": result_type,
            "path": dest_path,
            "filename": os.path.basename(dest_path)
        }
        
        # 添加元數據（如果提供）
        if metadata:
            result_record["metadata"] = metadata
        
        self.index["datasets"][dataset_id]["steps"][step_name]["results"].append(result_record)
        
        # 更新數據集摘要
        self.index["datasets"][dataset_id]["summary"]["status"] = "processing"
        self.index["datasets"][dataset_id]["summary"]["message"] = f"已完成步驟: {step_name}"
        self.index["datasets"][dataset_id]["summary"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存索引
        self._save_index()
        
        return True
    
    def _organize_file(self, file_path, step_name, result_type):
        """
        將結果文件移動到合適的目錄
        
        Args:
            file_path: 原始文件路徑
            step_name: 處理步驟名稱
            result_type: 結果類型
            
        Returns:
            str: 移動後的文件路徑
        """
        # 確定目標目錄
        if step_name == "data_import" and result_type == "data":
            target_dir = self.processed_data_dir
        elif step_name == "bert_embedding":
            if result_type == "data":
                target_dir = self.bert_embeddings_dir
            elif result_type == "visualization":
                target_dir = self.vector_vis_dir
        elif step_name == "lda_topic":
            if result_type == "data":
                target_dir = self.lda_topics_dir
            elif result_type == "model":
                target_dir = self.models_dir
            elif result_type == "visualization":
                target_dir = self.topic_vis_dir
        elif step_name == "aspect_vector":
            if result_type == "data":
                target_dir = self.aspect_vectors_dir
            elif result_type == "visualization":
                target_dir = self.vector_vis_dir
        elif step_name == "export":
            target_dir = self.exports_dir
        else:
            # 默認保留在原位置
            return file_path
        
        # 如果文件已經在目標目錄中，不需要移動
        if os.path.dirname(file_path) == target_dir:
            return file_path
        
        # 複製文件到目標目錄
        filename = os.path.basename(file_path)
        dest_path = os.path.join(target_dir, filename)
        
        try:
            # 如果目標文件已存在，添加時間戳
            if os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = os.path.join(target_dir, f"{name}_{timestamp}{ext}")
            
            shutil.copy2(file_path, dest_path)
            self.logger.info(f"文件已複製到: {dest_path}")
            return dest_path
        except Exception as e:
            self.logger.error(f"複製文件時出錯: {str(e)}")
            return file_path
    
    def complete_dataset(self, dataset_id, status="completed", message="所有處理完成"):
        """
        標記數據集處理完成
        
        Args:
            dataset_id: 數據集ID
            status: 狀態標識
            message: 狀態消息
            
        Returns:
            bool: 是否成功更新
        """
        if dataset_id not in self.index["datasets"]:
            self.logger.error(f"數據集ID不存在: {dataset_id}")
            return False
        
        # 更新狀態
        self.index["datasets"][dataset_id]["summary"]["status"] = status
        self.index["datasets"][dataset_id]["summary"]["message"] = message
        self.index["datasets"][dataset_id]["summary"]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存索引
        self._save_index()
        
        return True
    
    def get_dataset_summary(self, dataset_id):
        """
        獲取數據集處理摘要
        
        Args:
            dataset_id: 數據集ID
            
        Returns:
            dict: 數據集摘要
        """
        if dataset_id not in self.index["datasets"]:
            self.logger.error(f"數據集ID不存在: {dataset_id}")
            return None
        
        return self.index["datasets"][dataset_id]
    
    def generate_summary_report(self, dataset_id, output_format="html"):
        """
        生成數據集處理摘要報告
        
        Args:
            dataset_id: 數據集ID
            output_format: 輸出格式（'html'或'text'）
            
        Returns:
            str: 報告文件路徑
        """
        if dataset_id not in self.index["datasets"]:
            self.logger.error(f"數據集ID不存在: {dataset_id}")
            return None
        
        dataset_info = self.index["datasets"][dataset_id]
        
        # 構建報告內容
        if output_format == "html":
            report_content = self._generate_html_report(dataset_id, dataset_info)
            report_ext = "html"
        else:
            report_content = self._generate_text_report(dataset_id, dataset_info)
            report_ext = "txt"
        
        # 保存報告
        report_path = os.path.join(self.base_dir, f"report_{dataset_id}.{report_ext}")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"摘要報告已保存至: {report_path}")
            return report_path
        except Exception as e:
            self.logger.error(f"保存摘要報告時出錯: {str(e)}")
            return None
    
    def _generate_html_report(self, dataset_id, dataset_info):
        """生成HTML格式的摘要報告"""
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>跨領域情感分析 - 處理報告</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .step {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .result {{ margin-left: 20px; margin-bottom: 10px; }}
                .result-file {{ font-family: monospace; background-color: #f0f0f0; padding: 3px 6px; border-radius: 3px; }}
                .status {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 0.9em; color: white; }}
                .status-completed {{ background-color: #4CAF50; }}
                .status-processing {{ background-color: #2196F3; }}
                .status-error {{ background-color: #F44336; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .visualization {{ max-width: 300px; margin: 10px 0; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>跨領域情感分析處理報告</h1>
            
            <div class="summary">
                <h2>數據集摘要</h2>
                <p><strong>數據集名稱:</strong> {dataset_info['name']}</p>
                <p><strong>數據集ID:</strong> {dataset_id}</p>
                <p><strong>創建時間:</strong> {dataset_info['created_at']}</p>
                <p><strong>原始數據:</strong> {dataset_info['source_path']}</p>
                <p><strong>狀態:</strong> <span class="status status-{dataset_info['summary']['status']}">
                    {dataset_info['summary']['status'].upper()}</span></p>
                <p><strong>狀態信息:</strong> {dataset_info['summary']['message']}</p>
                {f"<p><strong>完成時間:</strong> {dataset_info['summary'].get('completed_at', 'N/A')}</p>" 
                    if 'completed_at' in dataset_info['summary'] else ""}
            </div>
            
            <h2>處理步驟</h2>
        '''
        
        # 添加每個處理步驟的詳細信息
        step_titles = {
            "data_import": "數據導入與預處理",
            "bert_embedding": "BERT語義提取",
            "lda_topic": "LDA面向切割",
            "aspect_vector": "面向向量計算"
        }
        
        for step_name, step_info in dataset_info["steps"].items():
            step_title = step_titles.get(step_name, step_name)
            html += f'''
            <div class="step">
                <h3>{step_title}</h3>
                <p><strong>狀態:</strong> <span class="status status-{step_info['status']}">
                    {step_info['status'].upper()}</span></p>
                <p><strong>完成時間:</strong> {step_info['completed_at']}</p>
                
                <h4>處理結果:</h4>
            '''
            
            # 分類顯示不同類型的結果
            result_types = {}
            for result in step_info["results"]:
                result_type = result["type"]
                if result_type not in result_types:
                    result_types[result_type] = []
                result_types[result_type].append(result)
            
            for result_type, results in result_types.items():
                html += f'<h5>{result_type.capitalize()}:</h5>'
                
                # 表格顯示數據結果
                if result_type == "data":
                    html += '''
                    <table>
                        <tr>
                            <th>文件名</th>
                            <th>路徑</th>
                            <th>元數據</th>
                        </tr>
                    '''
                    for result in results:
                        metadata_str = json.dumps(result.get("metadata", {}), ensure_ascii=False)
                        html += f'''
                        <tr>
                            <td class="result-file">{result["filename"]}</td>
                            <td>{result["path"]}</td>
                            <td><pre>{metadata_str}</pre></td>
                        </tr>
                        '''
                    html += '</table>'
                
                # 图片形式显示可视化结果
                elif result_type == "visualization":
                    html += '<div style="display: flex; flex-wrap: wrap; gap: 20px;">'
                    for result in results:
                        html += f'''
                        <div style="text-align: center; margin-bottom: 20px;">
                            <img src="file:///{result["path"]}" class="visualization" style="max-width: 300px;">
                            <div>{result["filename"]}</div>
                        </div>
                        '''
                    html += '</div>'
                
                # 列表显示其他类型结果
                else:
                    html += '<ul>'
                    for result in results:
                        html += f'<li class="result"><span class="result-file">{result["filename"]}</span> - {result["path"]}</li>'
                    html += '</ul>'
            
            html += '</div>'
        
        html += '''
        </body>
        </html>
        '''
        
        return html
    
    def _generate_text_report(self, dataset_id, dataset_info):
        """生成文本格式的摘要報告"""
        lines = [
            "=== 跨領域情感分析處理報告 ===",
            "",
            "--- 數據集摘要 ---",
            f"數據集名稱: {dataset_info['name']}",
            f"數據集ID: {dataset_id}",
            f"創建時間: {dataset_info['created_at']}",
            f"原始數據: {dataset_info['source_path']}",
            f"狀態: {dataset_info['summary']['status'].upper()}",
            f"狀態信息: {dataset_info['summary']['message']}",
        ]
        
        if 'completed_at' in dataset_info['summary']:
            lines.append(f"完成時間: {dataset_info['summary']['completed_at']}")
        
        lines.append("")
        lines.append("--- 處理步驟 ---")
        
        # 添加每個處理步驟的詳細信息
        step_titles = {
            "data_import": "數據導入與預處理",
            "bert_embedding": "BERT語義提取",
            "lda_topic": "LDA面向切割",
            "aspect_vector": "面向向量計算"
        }
        
        for step_name, step_info in dataset_info["steps"].items():
            step_title = step_titles.get(step_name, step_name)
            lines.append("")
            lines.append(f">> {step_title}")
            lines.append(f"狀態: {step_info['status'].upper()}")
            lines.append(f"完成時間: {step_info['completed_at']}")
            lines.append("處理結果:")
            
            # 分類顯示不同類型的結果
            result_types = {}
            for result in step_info["results"]:
                result_type = result["type"]
                if result_type not in result_types:
                    result_types[result_type] = []
                result_types[result_type].append(result)
            
            for result_type, results in result_types.items():
                lines.append(f"  - {result_type.capitalize()}:")
                for result in results:
                    lines.append(f"    * {result['filename']} - {result['path']}")
        
        return "\n".join(lines)
    
    def get_latest_results(self, dataset_id=None, limit=5):
        """
        獲取最新的處理結果
        
        Args:
            dataset_id: 數據集ID（可選）
            limit: 返回的結果數量
            
        Returns:
            list: 最新的處理結果
        """
        latest_results = []
        
        # 收集所有結果
        for d_id, dataset in self.index["datasets"].items():
            if dataset_id and d_id != dataset_id:
                continue
                
            for step_name, step_info in dataset["steps"].items():
                for result in step_info["results"]:
                    latest_results.append({
                        "dataset_id": d_id,
                        "dataset_name": dataset["name"],
                        "step": step_name,
                        "type": result["type"],
                        "filename": result["filename"],
                        "path": result["path"],
                        "timestamp": step_info["completed_at"]
                    })
        
        # 按時間排序並限制數量
        latest_results.sort(key=lambda x: x["timestamp"], reverse=True)
        return latest_results[:limit]
    
    def create_overview_report(self):
        """
        創建所有數據集的概覽報告
        
        Returns:
            str: 報告文件路徑
        """
        # 收集所有數據集的摘要信息
        datasets = []
        for dataset_id, dataset_info in self.index["datasets"].items():
            ds = {
                "id": dataset_id,
                "name": dataset_info["name"],
                "created_at": dataset_info["created_at"],
                "status": dataset_info["summary"]["status"],
                "message": dataset_info["summary"]["message"],
                "completed_steps": len(dataset_info["steps"])
            }
            
            # 統計不同類型的結果數量
            result_counts = {"data": 0, "model": 0, "visualization": 0}
            for step_info in dataset_info["steps"].values():
                for result in step_info["results"]:
                    result_type = result["type"]
                    if result_type in result_counts:
                        result_counts[result_type] += 1
            
            ds.update(result_counts)
            datasets.append(ds)
        
        # 生成HTML報告
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>跨領域情感分析 - 概覽報告</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .status {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 0.9em; color: white; }}
                .status-completed {{ background-color: #4CAF50; }}
                .status-processing {{ background-color: #2196F3; }}
                .status-error {{ background-color: #F44336; }}
            </style>
        </head>
        <body>
            <h1>跨領域情感分析概覽報告</h1>
            <p>生成時間: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <h2>數據集概覽</h2>
            <table>
                <tr>
                    <th>數據集名稱</th>
                    <th>ID</th>
                    <th>創建時間</th>
                    <th>狀態</th>
                    <th>完成步驟</th>
                    <th>數據文件</th>
                    <th>模型文件</th>
                    <th>可視化</th>
                </tr>
        '''
        
        for ds in sorted(datasets, key=lambda x: x["created_at"], reverse=True):
            html += f'''
                <tr>
                    <td>{ds["name"]}</td>
                    <td>{ds["id"]}</td>
                    <td>{ds["created_at"]}</td>
                    <td><span class="status status-{ds["status"]}">{ds["status"].upper()}</span></td>
                    <td>{ds["completed_steps"]}</td>
                    <td>{ds["data"]}</td>
                    <td>{ds["model"]}</td>
                    <td>{ds["visualization"]}</td>
                </tr>
            '''
        
        html += '''
            </table>
            
            <h2>目錄結構</h2>
            <pre>
results/
├── 01_processed_data/  # 處理後的原始數據
├── 02_bert_embeddings/ # BERT嵌入向量
├── 03_lda_topics/      # LDA主題結果
├── 04_aspect_vectors/  # 面向向量結果
├── models/             # 保存的模型
├── visualizations/     # 可視化結果
│   ├── topics/         # 主題可視化
│   └── vectors/        # 向量可視化
└── exports/            # 匯出的數據
            </pre>
        </body>
        </html>
        '''
        
        # 保存報告
        report_path = os.path.join(self.base_dir, "overview_report.html")
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html)
            self.logger.info(f"概覽報告已保存至: {report_path}")
            return report_path
        except Exception as e:
            self.logger.error(f"保存概覽報告時出錯: {str(e)}")
            return None