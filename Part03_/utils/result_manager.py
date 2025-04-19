"""
結果管理模組
此模組負責管理和保存面向分析結果
"""

import os
import json
import csv
import pickle
import datetime
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import logging

class ResultManager:
    """
    結果管理類，處理分析結果的儲存和載入
    """
    
    def __init__(self, base_dir: str = './Part03_/results', create_dirs: bool = True):
        """
        初始化結果管理器
        
        Args:
            base_dir: 結果儲存的基本目錄
            create_dirs: 是否創建需要的目錄
        """
        self.base_dir = base_dir
        self.index_file = os.path.join(base_dir, 'results_index.json')
        self.results_index = {}
        
        # 設置各類結果資料夾路徑
        self.processed_data_dir = os.path.join(base_dir, '01_processed_data')
        self.bert_embeddings_dir = os.path.join(base_dir, '02_bert_embeddings')
        self.lda_topics_dir = os.path.join(base_dir, '03_lda_topics')
        self.aspect_vectors_dir = os.path.join(base_dir, '04_aspect_vectors')
        self.exports_dir = os.path.join(base_dir, 'exports')
        self.models_dir = os.path.join(base_dir, 'models')
        self.visualizations_dir = os.path.join(base_dir, 'visualizations')
        
        # 設置日誌記錄
        self.logger = logging.getLogger('result_manager')
        self.logger.setLevel(logging.INFO)
        
        # 如果還沒有處理器，添加一個
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # 確保所有需要的目錄存在
        if create_dirs:
            self._ensure_dirs_exist()
        
        # 載入現有的結果索引（如果存在）
        self._load_results_index()
    
    def _ensure_dirs_exist(self) -> None:
        """確保所有需要的目錄存在"""
        for dir_path in [
            self.base_dir,
            self.processed_data_dir,
            self.bert_embeddings_dir,
            self.lda_topics_dir,
            self.aspect_vectors_dir,
            self.exports_dir,
            self.models_dir,
            self.visualizations_dir
        ]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                self.logger.info(f"創建目錄: {dir_path}")
    
    def _load_results_index(self) -> None:
        """載入現有的結果索引文件"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.results_index = json.load(f)
                self.logger.info(f"已載入結果索引，含有 {len(self.results_index)} 個結果組")
            except Exception as e:
                self.logger.error(f"載入結果索引時發生錯誤: {str(e)}")
                self.results_index = {}
        else:
            self.results_index = {}
    
    def _save_results_index(self) -> None:
        """保存結果索引到文件"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_index, f, indent=4, ensure_ascii=False)
            self.logger.info("已保存結果索引")
        except Exception as e:
            self.logger.error(f"保存結果索引時發生錯誤: {str(e)}")
    
    def create_result_set(self, dataset_name: str, description: str = "") -> str:
        """
        創建一個新的結果集
        
        Args:
            dataset_name: 數據集名稱
            description: 結果集描述
            
        Returns:
            result_id: 生成的結果集ID
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_id = f"{dataset_name}_{timestamp}"
        
        result_info = {
            'id': result_id,
            'dataset_name': dataset_name,
            'description': description,
            'created_at': timestamp,
            'updated_at': timestamp,
            'files': {},
            'metrics': {},
            'status': 'created'
        }
        
        self.results_index[result_id] = result_info
        self._save_results_index()
        
        self.logger.info(f"創建新結果集: {result_id}")
        
        return result_id
    
    def get_result_sets(self) -> List[Dict[str, Any]]:
        """
        獲取所有結果集資訊
        
        Returns:
            結果集資訊列表
        """
        return list(self.results_index.values())
    
    def get_result_set(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        獲取指定ID的結果集資訊
        
        Args:
            result_id: 結果集ID
            
        Returns:
            結果集資訊，如果未找到則返回None
        """
        return self.results_index.get(result_id)
    
    def update_result_set(self, result_id: str, **kwargs) -> bool:
        """
        更新結果集資訊
        
        Args:
            result_id: 結果集ID
            **kwargs: 要更新的結果集屬性
            
        Returns:
            更新是否成功
        """
        if result_id not in self.results_index:
            self.logger.error(f"結果集不存在: {result_id}")
            return False
        
        # 更新屬性
        for key, value in kwargs.items():
            if key != 'id':  # 不允許修改ID
                self.results_index[result_id][key] = value
        
        # 更新時間戳
        self.results_index[result_id]['updated_at'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self._save_results_index()
        self.logger.info(f"更新結果集: {result_id}")
        
        return True
    
    def save_data(self, result_id: str, data: Any, data_type: str, file_name: str = None) -> str:
        """
        保存數據到結果集
        
        Args:
            result_id: 結果集ID
            data: 要保存的數據
            data_type: 數據類型（如 'processed_data', 'bert_embeddings', 'lda_topics', 'aspect_vectors'）
            file_name: 可選的文件名稱，如果未提供將自動生成
            
        Returns:
            保存的文件路徑
        """
        if result_id not in self.results_index:
            self.logger.error(f"結果集不存在: {result_id}")
            raise ValueError(f"結果集不存在: {result_id}")
        
        # 根據數據類型選擇目錄
        if data_type == 'processed_data':
            dir_path = self.processed_data_dir
            suffix = '.pkl'
        elif data_type == 'bert_embeddings':
            dir_path = self.bert_embeddings_dir
            suffix = '.pkl'
        elif data_type == 'lda_topics':
            dir_path = self.lda_topics_dir
            suffix = '.pkl'
        elif data_type == 'aspect_vectors':
            dir_path = self.aspect_vectors_dir
            suffix = '.pkl'
        elif data_type == 'model':
            dir_path = self.models_dir
            suffix = '.pkl'
        elif data_type == 'visualization':
            dir_path = self.visualizations_dir
            suffix = '.png'
        elif data_type == 'export':
            dir_path = self.exports_dir
            suffix = '.csv'
        else:
            dir_path = os.path.join(self.base_dir, data_type)
            suffix = ''
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # 生成文件名
        if not file_name:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{result_id}_{data_type}_{timestamp}{suffix}"
        elif not file_name.endswith(suffix) and suffix:
            file_name += suffix
        
        # 完整文件路徑
        file_path = os.path.join(dir_path, file_name)
        
        # 根據數據類型保存
        try:
            if isinstance(data, pd.DataFrame):
                if file_path.endswith('.csv'):
                    data.to_csv(file_path, index=False, encoding='utf-8')
                else:
                    data.to_pickle(file_path)
            elif isinstance(data, plt.Figure):
                data.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close(data)
            elif file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
            elif file_path.endswith('.csv') and isinstance(data, (list, dict)):
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    else:
                        writer = csv.writer(f)
                        writer.writerows(data)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            # 更新結果集的文件記錄
            if 'files' not in self.results_index[result_id]:
                self.results_index[result_id]['files'] = {}
                
            if data_type not in self.results_index[result_id]['files']:
                self.results_index[result_id]['files'][data_type] = []
                
            # 添加文件資訊
            file_info = {
                'name': file_name,
                'path': file_path,
                'created_at': datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            }
            
            self.results_index[result_id]['files'][data_type].append(file_info)
            self._save_results_index()
            
            self.logger.info(f"已保存 {data_type} 數據到 {file_path}")
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"保存數據到 {file_path} 時發生錯誤: {str(e)}")
            raise
    
    def load_data(self, result_id: str, data_type: str, file_name: str = None) -> Any:
        """
        從結果集載入數據
        
        Args:
            result_id: 結果集ID
            data_type: 數據類型
            file_name: 文件名稱，如果未提供則載入最新的
            
        Returns:
            載入的數據
        """
        if result_id not in self.results_index:
            self.logger.error(f"結果集不存在: {result_id}")
            raise ValueError(f"結果集不存在: {result_id}")
        
        # 獲取檔案列表
        if 'files' not in self.results_index[result_id] or data_type not in self.results_index[result_id]['files'] or not self.results_index[result_id]['files'][data_type]:
            self.logger.error(f"結果集 {result_id} 中不存在 {data_type} 類型的數據")
            raise ValueError(f"結果集 {result_id} 中不存在 {data_type} 類型的數據")
        
        files = self.results_index[result_id]['files'][data_type]
        
        # 確定要載入的文件
        if file_name:
            file_info = next((f for f in files if f['name'] == file_name), None)
            if not file_info:
                self.logger.error(f"未找到文件名 {file_name}")
                raise ValueError(f"未找到文件名 {file_name}")
        else:
            # 使用最新的文件
            file_info = sorted(files, key=lambda x: x['created_at'], reverse=True)[0]
        
        file_path = file_info['path']
        
        # 檢查文件是否存在
        if not os.path.exists(file_path):
            self.logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根據文件類型載入
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.endswith('.png'):
                return plt.imread(file_path)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                
        except Exception as e:
            self.logger.error(f"從 {file_path} 載入數據時發生錯誤: {str(e)}")
            raise
    
    def delete_result_set(self, result_id: str, delete_files: bool = True) -> bool:
        """
        刪除結果集
        
        Args:
            result_id: 結果集ID
            delete_files: 是否同時刪除相關文件
            
        Returns:
            刪除是否成功
        """
        if result_id not in self.results_index:
            self.logger.error(f"結果集不存在: {result_id}")
            return False
        
        # 如果要刪除文件
        if delete_files and 'files' in self.results_index[result_id]:
            for data_type, files in self.results_index[result_id]['files'].items():
                for file_info in files:
                    file_path = file_info['path']
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                            self.logger.info(f"已刪除文件: {file_path}")
                        except Exception as e:
                            self.logger.error(f"刪除文件 {file_path} 時發生錯誤: {str(e)}")
        
        # 從索引中刪除
        del self.results_index[result_id]
        self._save_results_index()
        
        self.logger.info(f"已刪除結果集: {result_id}")
        return True
    
    def save_metrics(self, result_id: str, metrics: Dict[str, Any]) -> None:
        """
        保存評估指標到結果集
        
        Args:
            result_id: 結果集ID
            metrics: 指標數據字典
        """
        if result_id not in self.results_index:
            self.logger.error(f"結果集不存在: {result_id}")
            raise ValueError(f"結果集不存在: {result_id}")
        
        if 'metrics' not in self.results_index[result_id]:
            self.results_index[result_id]['metrics'] = {}
        
        # 更新指標
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_index[result_id]['metrics'].update(metrics)
        self.results_index[result_id]['updated_at'] = timestamp
        
        self._save_results_index()
        self.logger.info(f"已更新結果集 {result_id} 的指標")
    
    def export_report(self, result_id: str, template_file: str = None) -> str:
        """
        匯出結果報告
        
        Args:
            result_id: 結果集ID
            template_file: 報告模板文件路徑
            
        Returns:
            報告文件路徑
        """
        if result_id not in self.results_index:
            self.logger.error(f"結果集不存在: {result_id}")
            raise ValueError(f"結果集不存在: {result_id}")
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(self.base_dir, f"report_{result_id}_{timestamp}.html")
        
        # 後續可以實現具體的報告生成邏輯
        # 例如使用模板引擎（如Jinja2）生成報告
        
        # 更新報告文件信息
        if 'reports' not in self.results_index[result_id]:
            self.results_index[result_id]['reports'] = []
            
        report_info = {
            'path': report_file,
            'created_at': timestamp
        }
        
        self.results_index[result_id]['reports'].append(report_info)
        self._save_results_index()
        
        self.logger.info(f"已為結果集 {result_id} 生成報告: {report_file}")
        return report_file
    
    def export_result_set(self, result_id: str, export_dir: str) -> List[str]:
        """
        導出結果集到指定目錄
        
        Args:
            result_id: 結果集ID
            export_dir: 導出目錄路徑
            
        Returns:
            導出的文件路徑列表
        """
        if result_id not in self.results_index:
            self.logger.error(f"結果集不存在: {result_id}")
            raise ValueError(f"結果集不存在: {result_id}")
        
        # 確保導出目錄存在
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        
        result_info = self.results_index[result_id]
        exported_files = []
        
        # 導出索引信息
        index_file = os.path.join(export_dir, f"{result_id}_info.json")
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(result_info, f, indent=4, ensure_ascii=False)
        exported_files.append(index_file)
        
        # 導出相關文件
        if 'files' in result_info:
            for data_type, files in result_info['files'].items():
                for file_info in files:
                    source_path = file_info['path']
                    if os.path.exists(source_path):
                        # 創建目標子目錄
                        dest_subdir = os.path.join(export_dir, data_type)
                        if not os.path.exists(dest_subdir):
                            os.makedirs(dest_subdir)
                        
                        # 複製文件
                        file_name = os.path.basename(source_path)
                        dest_path = os.path.join(dest_subdir, file_name)
                        shutil.copy2(source_path, dest_path)
                        exported_files.append(dest_path)
                        self.logger.info(f"已導出文件: {dest_path}")
        
        self.logger.info(f"已完成結果集 {result_id} 的導出，共 {len(exported_files)} 個文件")
        return exported_files
    
    def get_results_index(self) -> Dict[str, Any]:
        """
        獲取結果索引
        
        Returns:
            結果索引字典
        """
        return self.results_index.copy()
    
    def search_result_sets(self, query: str) -> List[Dict[str, Any]]:
        """
        搜索結果集
        
        Args:
            query: 搜索查詢字符串
            
        Returns:
            匹配的結果集列表
        """
        query = query.lower()
        results = []
        
        for result_id, info in self.results_index.items():
            # 檢查ID、數據集名稱和描述
            if (query in result_id.lower() or
                query in info.get('dataset_name', '').lower() or
                query in info.get('description', '').lower()):
                results.append(info)
        
        return results

# 使用範例
if __name__ == "__main__":
    # 初始化結果管理器
    rm = ResultManager()
    
    # 創建新的結果集
    result_id = rm.create_result_set("IMDB_Test", "IMDB資料集測試結果")
    
    # 保存數據
    import numpy as np
    test_data = {"vectors": np.random.rand(10, 5), "labels": ["pos"] * 5 + ["neg"] * 5}
    rm.save_data(result_id, test_data, "processed_data", "test_data.pkl")
    
    # 更新結果集狀態
    rm.update_result_set(result_id, status="completed")
    
    # 保存指標
    test_metrics = {"accuracy": 0.85, "precision": 0.87, "recall": 0.84}
    rm.save_metrics(result_id, test_metrics)
    
    print(f"已創建結果集: {result_id}")
    print(f"結果集信息: {rm.get_result_set(result_id)}")
