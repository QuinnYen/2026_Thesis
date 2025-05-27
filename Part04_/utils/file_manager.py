"""
文件管理工具模組 - 提供檔案讀寫和管理相關功能
"""

import os
import shutil
import json
import csv
import pickle
import pandas as pd
import numpy as np
import logging
import glob
from datetime import datetime
from pathlib import Path
import time
import zipfile
import io
import re
import fnmatch

# 導入系統模組
from utils.logger import get_logger

# 獲取logger
logger = get_logger("file_manager")

class FileManager:
    """提供檔案管理功能的類"""
    
    def __init__(self, config=None):
        """初始化檔案管理器
        
        Args:
            config: 配置字典或配置管理器對象
        """
        self.config = config or {}
        
        # 設置logger
        self.logger = logger
        
        # 從配置中提取路徑設置
        self._init_paths()
        
        # 確保所需目錄存在
        self._ensure_directories()
    
    def _init_paths(self):
        """初始化檔案路徑"""
        # 安全地從不同類型的配置對象中獲取路徑
        paths_config = None
        
        if isinstance(self.config, dict) and "paths" in self.config:
            paths_config = self.config["paths"]
        elif hasattr(self.config, "get"):
            try:
                paths_config = self.config.get("paths")
            except:
                pass
            
        # 如果無法獲取配置，使用默認值
        if not paths_config:
            paths_config = {}
            
        # 尋找 Part04_ 目錄的絕對路徑
        part04_dir = self._find_part04_directory()
        if part04_dir:
            base_dir = os.path.abspath(part04_dir)
        else:
            base_dir = os.path.abspath("./Part04_")

        # 生成帶有時間戳的輸出目錄名稱
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 設置輸出根目錄為1_output/run_timestamp
        output_root = os.path.join(base_dir, "1_output", f"run_{timestamp}")
            
        # 設置各個目錄路徑 - 只定義路徑，不創建目錄
        self.data_dir = os.path.abspath(paths_config.get("data_dir", os.path.join(base_dir, "..", "ReviewsDataBase")))
        self.output_dir = os.path.abspath(paths_config.get("output_dir", output_root))
        
        # 日誌目錄路徑（固定在1_output/logs，不隨run目錄變動）
        # 如果配置中有log_dir，確保它是絕對路徑；否則使用默認的絕對路徑
        config_log_dir = paths_config.get("log_dir")
        if config_log_dir:
            # 如果配置中的路徑是相對路徑，則相對於base_dir解析
            if not os.path.isabs(config_log_dir):
                self.log_dir = os.path.abspath(os.path.join(base_dir, "..", config_log_dir.lstrip("./")))
            else:
                self.log_dir = os.path.abspath(config_log_dir)
        else:
            self.log_dir = os.path.abspath(os.path.join(base_dir, "1_output", "logs"))
        
        # 其他目錄路徑定義（不創建）
        self.model_dir = paths_config.get("model_dir", os.path.join(output_root, "models"))
        self.export_dir = paths_config.get("export_dir", os.path.join(output_root, "exports"))
        self.results_dir = paths_config.get("results_dir", os.path.join(output_root, "results"))
        self.embeddings_dir = paths_config.get("embeddings_dir", os.path.join(output_root, "embeddings"))
        self.topics_dir = paths_config.get("topics_dir", os.path.join(output_root, "topics"))
        self.vectors_dir = paths_config.get("vectors_dir", os.path.join(output_root, "vectors"))
        self.evaluation_dir = paths_config.get("evaluation_dir", os.path.join(output_root, "evaluation"))
        self.visualizations_dir = paths_config.get("visualizations_dir", os.path.join(output_root, "visualizations"))
        self.temp_dir = paths_config.get("temp_dir", os.path.join(output_root, "temp"))
        self.resources_dir = paths_config.get("resources_dir", os.path.join(base_dir, "resources"))
        
        # 保存時間戳和輸出根目錄，以便其他方法使用
        self.timestamp = timestamp
        self.output_root = output_root
        
        # 只創建日誌目錄
        self._ensure_log_directory()

    def _find_part04_directory(self):
        """尋找 Part04_ 目錄的絕對路徑
        
        Returns:
            str: Part04_ 目錄的絕對路徑，如果找不到則返回 None
        """
        try:
            # 方法1：檢查當前工作目錄
            current_dir = os.path.abspath(os.curdir)
            if os.path.basename(current_dir) == "Part04_":
                return current_dir
                
            # 方法2：檢查當前工作目錄的父目錄
            parent_dir = os.path.dirname(current_dir)
            if os.path.basename(parent_dir) == "Part04_":
                return parent_dir
                
            # 方法3：檢查當前工作目錄下是否有 Part04_ 子目錄
            potential_path = os.path.join(current_dir, "Part04_")
            if os.path.exists(potential_path) and os.path.isdir(potential_path):
                return potential_path
                
            # 方法4：檢查當前檔案所在目錄
            if __file__:
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                # 如果是在 utils 目錄中
                if os.path.basename(current_file_dir) == "utils":
                    potential_path = os.path.dirname(current_file_dir)
                    if os.path.basename(potential_path) == "Part04_":
                        return potential_path
            
            # 方法5：搜索 D:\Project 目錄
            project_dir = "D:\\Project"
            if os.path.exists(project_dir):
                for item in os.listdir(project_dir):
                    if item == "2026_Thesis":
                        thesis_dir = os.path.join(project_dir, item)
                        part04_dir = os.path.join(thesis_dir, "Part04_")
                        if os.path.exists(part04_dir) and os.path.isdir(part04_dir):
                            return part04_dir
            
            # 如果以上方法都失敗，返回 None
            return None
            
        except Exception as e:
            self.logger.error(f"尋找 Part04_ 目錄時出錯: {str(e)}")
            return None

    def _ensure_log_directory(self):
        """確保日誌目錄存在"""
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                self.logger.info(f"已創建日誌目錄: {self.log_dir}")
        except Exception as e:
            self.logger.warning(f"無法創建日誌目錄 {self.log_dir}: {str(e)}")

    def _ensure_directories(self):
        """確保指定目錄存在"""
        # 此方法保留但不再自動創建所有目錄
        # 其他目錄將在需要時通過 ensure_dir 方法創建
        pass

    def ensure_dir(self, directory):
        """確保目錄存在，不存在則創建
        
        Args:
            directory: 目錄路徑
            
        Returns:
            bool: 目錄是否存在或創建成功
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                self.logger.debug(f"已創建目錄: {directory}")
            return True
        except Exception as e:
            self.logger.error(f"確保目錄 {directory} 存在失敗: {str(e)}")
            return False

    def _create_backup(self, file_path):
        """為指定文件創建備份
        
        Args:
            file_path: 要備份的文件路徑
            
        Returns:
            str: 備份文件的路徑，若備份失敗則返回None
        """
        if not os.path.exists(file_path):
            return None
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{file_path}.{timestamp}.bak"
            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"已備份文件: {file_path} -> {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.warning(f"備份文件 {file_path} 失敗: {str(e)}")
            return None

    def migrate_output_directories(self):
        """遷移輸出目錄結構，將根目錄下的output檔案移動到Part04_/1_output目錄
        
        將舊的輸出目錄結構遷移到新的統一目錄結構中，確保所有資料都在正確位置。
        此方法會執行以下操作：
        1. 找到根目錄下的output資料夾
        2. 找到Part04_下的logs資料夾
        3. 將上述資料夾中的檔案移動到Part04_/1_output的對應子目錄
        
        Returns:
            dict: 遷移統計信息
        """
        try:
            migration_stats = {
                "migrated_files": 0, 
                "migrated_folders": 0, 
                "errors": 0,
                "details": []
            }
            
            # 尋找根目錄下的output資料夾
            root_output_dir = None
            thesis_dir = os.path.abspath(os.path.join(self._find_part04_directory() or ".", ".."))
            
            if os.path.exists(thesis_dir):
                potential_output = os.path.join(thesis_dir, "output")
                if os.path.exists(potential_output) and os.path.isdir(potential_output):
                    root_output_dir = potential_output
                    migration_stats["details"].append(f"找到根目錄下的output資料夾: {root_output_dir}")
                    
            # 尋找Part04_下的獨立logs資料夾
            part04_dir = self._find_part04_directory()
            if part04_dir:
                part04_logs = os.path.join(part04_dir, "logs")
                if os.path.exists(part04_logs) and os.path.isdir(part04_logs):
                    migration_stats["details"].append(f"找到Part04_下的logs資料夾: {part04_logs}")
            else:
                part04_logs = None
                
            # 如果找不到任何需要遷移的資料夾，跳過
            if not root_output_dir and not part04_logs:
                migration_stats["details"].append("未找到需要遷移的資料夾")
                return migration_stats
                
            # 1. 遷移根目錄下的output資料夾
            if root_output_dir:
                # 遷移logs檔案
                root_logs = os.path.join(root_output_dir, "logs")
                if os.path.exists(root_logs) and os.path.isdir(root_logs):
                    self._migrate_directory_contents(root_logs, self.log_dir, migration_stats)
                    
                # 遷移models檔案
                root_models = os.path.join(root_output_dir, "models")
                if os.path.exists(root_models) and os.path.isdir(root_models):
                    self._migrate_directory_contents(root_models, self.model_dir, migration_stats)
                    
                # 遷移embeddings檔案
                root_embeddings = os.path.join(root_output_dir, "embeddings")
                if os.path.exists(root_embeddings) and os.path.isdir(root_embeddings):
                    self._migrate_directory_contents(root_embeddings, self.embeddings_dir, migration_stats)
                    
                # 遷移vectors檔案
                root_vectors = os.path.join(root_output_dir, "vectors")
                if os.path.exists(root_vectors) and os.path.isdir(root_vectors):
                    self._migrate_directory_contents(root_vectors, self.vectors_dir, migration_stats)
                    
                # 遷移topics檔案
                root_topics = os.path.join(root_output_dir, "topics")
                if os.path.exists(root_topics) and os.path.isdir(root_topics):
                    self._migrate_directory_contents(root_topics, self.topics_dir, migration_stats)
                    
                # 遷移visualizations檔案
                root_vis = os.path.join(root_output_dir, "visualizations")
                if os.path.exists(root_vis) and os.path.isdir(root_vis):
                    self._migrate_directory_contents(root_vis, self.visualizations_dir, migration_stats)

                # 遷移其他檔案
                for item in os.listdir(root_output_dir):
                    item_path = os.path.join(root_output_dir, item)
                    if os.path.isfile(item_path):
                        try:
                            # 其他檔案直接移至1_output根目錄
                            target_path = os.path.join(self.output_dir, item)
                            # 避免覆蓋已存在的檔案
                            if os.path.exists(target_path):
                                base, ext = os.path.splitext(item)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                target_path = os.path.join(self.output_dir, f"{base}_{timestamp}{ext}")
                                
                            shutil.move(item_path, target_path)
                            migration_stats["migrated_files"] += 1
                            migration_stats["details"].append(f"已遷移檔案: {item_path} → {target_path}")
                        except Exception as e:
                            migration_stats["errors"] += 1
                            migration_stats["details"].append(f"遷移檔案失敗 {item_path}: {str(e)}")
            
            # 2. 遷移Part04_下的logs資料夾
            if part04_logs:
                self._migrate_directory_contents(part04_logs, self.log_dir, migration_stats)
                
            # 記錄遷移結果
            self.logger.info(f"目錄遷移完成: 成功遷移 {migration_stats['migrated_files']} 個檔案, "
                           f"{migration_stats['migrated_folders']} 個資料夾, "
                           f"遇到 {migration_stats['errors']} 個錯誤")
                
            return migration_stats
            
        except Exception as e:
            self.logger.error(f"遷移輸出目錄時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            migration_stats["errors"] += 1
            migration_stats["details"].append(f"遷移過程出錯: {str(e)}")
            return migration_stats
    
    def _migrate_directory_contents(self, source_dir, target_dir, stats):
        """遷移目錄中的所有內容到目標目錄
        
        Args:
            source_dir: 來源目錄
            target_dir: 目標目錄
            stats: 統計字典
        """
        try:
            # 確保目標目錄存在
            os.makedirs(target_dir, exist_ok=True)
            
            for item in os.listdir(source_dir):
                source_item = os.path.join(source_dir, item)
                target_item = os.path.join(target_dir, item)
                
                try:
                    # 處理資料夾
                    if os.path.isdir(source_item):
                        if not os.path.exists(target_item):
                            shutil.copytree(source_item, target_item)
                        else:
                            # 如果目標資料夾已存在，遞迴地遷移其內容
                            self._migrate_directory_contents(source_item, target_item, stats)
                        stats["migrated_folders"] += 1
                    # 處理檔案
                    else:
                        # 避免覆蓋已存在的檔案
                        if os.path.exists(target_item):
                            base, ext = os.path.splitext(item)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            target_item = os.path.join(target_dir, f"{base}_{timestamp}{ext}")
                            
                        shutil.copy2(source_item, target_item)
                        stats["migrated_files"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    stats["details"].append(f"遷移 {source_item} 失敗: {str(e)}")
                    
            # 記錄成功遷移的目錄
            stats["details"].append(f"已遷移目錄: {source_dir} → {target_dir}")
            
        except Exception as e:
            stats["errors"] += 1
            stats["details"].append(f"遷移目錄 {source_dir} 失敗: {str(e)}")

    def read_file(self, file_path, encoding='utf-8'):
        """讀取文本文件
        
        Args:
            file_path: 文件路徑
            encoding: 編碼格式
            
        Returns:
            str: 文件內容，讀取失敗則返回None
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            self.logger.debug(f"已讀取文件: {file_path}")
            return content
        except Exception as e:
            self.logger.error(f"讀取文件 {file_path} 失敗: {str(e)}")
            return None

    def write_file(self, file_path, content, encoding='utf-8', mode='w'):
        """寫入文本文件
        
        Args:
            file_path: 文件路徑
            content: 要寫入的內容
            encoding: 編碼格式
            mode: 寫入模式，'w'為覆寫，'a'為追加
            
        Returns:
            bool: 寫入是否成功
        """
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 如果是覆寫模式且啟用了備份功能，則進行備份
            if mode == 'w' and os.path.exists(file_path):
                self._create_backup(file_path)
                
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)
                
            self.logger.debug(f"已寫入文件: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"寫入文件 {file_path} 失敗: {str(e)}")
            return False

    def read_json(self, file_path, encoding='utf-8'):
        """讀取JSON文件
        
        Args:
            file_path: 文件路徑
            encoding: 編碼格式
            
        Returns:
            dict: 解析後的JSON數據，讀取失敗則返回None
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            self.logger.debug(f"已讀取JSON文件: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"讀取JSON文件 {file_path} 失敗: {str(e)}")
            return None

    def write_json(self, file_path, data, encoding='utf-8', indent=4):
        """寫入JSON文件
        
        Args:
            file_path: 文件路徑
            data: 要寫入的數據(字典或列表)
            encoding: 編碼格式
            indent: 縮進空格數
            
        Returns:
            bool: 寫入是否成功
        """
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 如果啟用了備份功能且文件已存在，則進行備份
            if os.path.exists(file_path):
                self._create_backup(file_path)
                
            with open(file_path, 'w', encoding=encoding) as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
                
            self.logger.debug(f"已寫入JSON文件: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"寫入JSON文件 {file_path} 失敗: {str(e)}")
            return False

    def read_csv(self, file_path, encoding='utf-8', **kwargs):
        """讀取CSV文件為DataFrame
        
        Args:
            file_path: 文件路徑
            encoding: 編碼格式
            **kwargs: 傳遞給pandas.read_csv的其他參數
            
        Returns:
            pandas.DataFrame: 讀取的數據框，失敗則返回None
        """
        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            self.logger.debug(f"已讀取CSV文件: {file_path}，共{len(df)}行")
            return df
        except Exception as e:
            self.logger.error(f"讀取CSV文件 {file_path} 失敗: {str(e)}")
            return None

    def write_csv(self, file_path, df, encoding='utf-8', index=False, **kwargs):
        """將DataFrame寫入CSV文件
        
        Args:
            file_path: 文件路徑
            df: 要寫入的DataFrame
            encoding: 編碼格式
            index: 是否包含索引
            **kwargs: 傳遞給df.to_csv的其他參數
            
        Returns:
            bool: 寫入是否成功
        """
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 如果啟用了備份功能且文件已存在，則進行備份
            if os.path.exists(file_path):
                self._create_backup(file_path)
                
            df.to_csv(file_path, encoding=encoding, index=index, **kwargs)
            
            self.logger.debug(f"已寫入CSV文件: {file_path}，共{len(df)}行")
            return True
        except Exception as e:
            self.logger.error(f"寫入CSV文件 {file_path} 失敗: {str(e)}")
            return False

    def read_excel(self, file_path, sheet_name=0, **kwargs):
        """讀取Excel文件為DataFrame
        
        Args:
            file_path: 文件路徑
            sheet_name: 工作表名稱或索引
            **kwargs: 傳遞給pandas.read_excel的其他參數
            
        Returns:
            pandas.DataFrame: 讀取的數據框，失敗則返回None
        """
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            self.logger.debug(f"已讀取Excel文件: {file_path}，表格: {sheet_name}，共{len(df)}行")
            return df
        except Exception as e:
            self.logger.error(f"讀取Excel文件 {file_path} 失敗: {str(e)}")
            return None

    def write_excel(self, file_path, df, sheet_name='Sheet1', index=False, **kwargs):
        """將DataFrame寫入Excel文件
        
        Args:
            file_path: 文件路徑
            df: 要寫入的DataFrame
            sheet_name: 工作表名稱
            index: 是否包含索引
            **kwargs: 傳遞給df.to_excel的其他參數
            
        Returns:
            bool: 寫入是否成功
        """
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 如果啟用了備份功能且文件已存在，則進行備份
            if os.path.exists(file_path):
                self._create_backup(file_path)
                
            df.to_excel(file_path, sheet_name=sheet_name, index=index, **kwargs)
            
            self.logger.debug(f"已寫入Excel文件: {file_path}，表格: {sheet_name}，共{len(df)}行")
            return True
        except Exception as e:
            self.logger.error(f"寫入Excel文件 {file_path} 失敗: {str(e)}")
            return False

    def save_numpy(self, file_path, arr):
        """保存NumPy數組
        
        Args:
            file_path: 文件路徑
            arr: NumPy數組
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 如果啟用了備份功能且文件已存在，則進行備份
            if os.path.exists(file_path):
                self._create_backup(file_path)
                
            np.save(file_path, arr)
            
            self.logger.debug(f"已保存NumPy數組: {file_path}，形狀: {arr.shape}")
            return True
        except Exception as e:
            self.logger.error(f"保存NumPy數組 {file_path} 失敗: {str(e)}")
            return False

    def load_numpy(self, file_path):
        """加載NumPy數組
        
        Args:
            file_path: 文件路徑
            
        Returns:
            numpy.ndarray: 加載的數組，失敗則返回None
        """
        try:
            # 如果文件沒有.npy後綴，自動添加
            if not file_path.endswith('.npy'):
                file_path += '.npy'
                
            arr = np.load(file_path)
            
            self.logger.debug(f"已加載NumPy數組: {file_path}，形狀: {arr.shape}")
            return arr
        except Exception as e:
            self.logger.error(f"加載NumPy數組 {file_path} 失敗: {str(e)}")
            return None

    def save_npz(self, file_path, **arrays):
        """保存多個NumPy數組到單個文件
        
        Args:
            file_path: 文件路徑
            **arrays: 以關鍵字參數形式提供的多個NumPy數組
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 如果啟用了備份功能且文件已存在，則進行備份
            if os.path.exists(file_path):
                self._create_backup(file_path)
                
            np.savez_compressed(file_path, **arrays)
            
            shapes_info = ', '.join([f"{name}: {arr.shape}" for name, arr in arrays.items()])
            self.logger.debug(f"已保存NPZ數組: {file_path}，包含: {shapes_info}")
            return True
        except Exception as e:
            self.logger.error(f"保存NPZ數組 {file_path} 失敗: {str(e)}")
            return False

    def load_npz(self, file_path):
        """加載NPZ文件中的多個NumPy數組
        
        Args:
            file_path: 文件路徑
            
        Returns:
            dict: 數組字典，失敗則返回None
        """
        try:
            # 如果文件沒有.npz後綴，自動添加
            if not file_path.endswith('.npz'):
                file_path += '.npz'
                
            npz_file = np.load(file_path)
            
            # 將NpzFile對象轉換為字典
            arrays = {key: npz_file[key] for key in npz_file.files}
            
            shapes_info = ', '.join([f"{name}: {arr.shape}" for name, arr in arrays.items()])
            self.logger.debug(f"已加載NPZ數組: {file_path}，包含: {shapes_info}")
            return arrays
        except Exception as e:
            self.logger.error(f"加載NPZ數組 {file_path} 失敗: {str(e)}")
            return None

    def save_pickle(self, file_path, obj):
        """使用pickle保存Python對象
        
        Args:
            file_path: 文件路徑
            obj: 要保存的Python對象
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 如果啟用了備份功能且文件已存在，則進行備份
            if os.path.exists(file_path):
                self._create_backup(file_path)
                
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
                
            self.logger.debug(f"已保存Pickle對象: {file_path}，類型: {type(obj)}")
            return True
        except Exception as e:
            self.logger.error(f"保存Pickle對象 {file_path} 失敗: {str(e)}")
            return False

    def load_pickle(self, file_path):
        """使用pickle加載Python對象
        
        Args:
            file_path: 文件路徑
            
        Returns:
            object: 加載的Python對象，失敗則返回None
        """
        try:
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
                
            self.logger.debug(f"已加載Pickle對象: {file_path}，類型: {type(obj)}")
            return obj
        except Exception as e:
            self.logger.error(f"加載Pickle對象 {file_path} 失敗: {str(e)}")
            return None

    def list_files(self, directory, pattern=None, recursive=False):
        """列出目錄中的文件
        
        Args:
            directory: 目錄路徑
            pattern: 文件匹配模式，如'*.csv'
            recursive: 是否遞歸搜索子目錄
            
        Returns:
            list: 文件路徑列表
        """
        try:
            if not os.path.exists(directory):
                self.logger.warning(f"目錄不存在: {directory}")
                return []
                
            if recursive:
                search_path = os.path.join(directory, '**', pattern or '*')
                files = glob.glob(search_path, recursive=True)
            else:
                search_path = os.path.join(directory, pattern or '*')
                files = glob.glob(search_path)
                
            # 只返回文件，不包含目錄
            files = [f for f in files if os.path.isfile(f)]
            
            self.logger.debug(f"在 {directory} 中找到 {len(files)} 個文件")
            return files
        except Exception as e:
            self.logger.error(f"列出目錄 {directory} 中的文件失敗: {str(e)}")
            return []

    def copy_file(self, source, destination):
        """複製文件
        
        Args:
            source: 源文件路徑
            destination: 目標文件路徑
            
        Returns:
            bool: 複製是否成功
        """
        try:
            # 確保目標目錄存在
            dest_dir = os.path.dirname(destination)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                
            # 如果目標文件已存在且啟用了備份功能，則進行備份
            if os.path.exists(destination):
                self._create_backup(destination)
                
            shutil.copy2(source, destination)
            
            self.logger.debug(f"已複製文件: {source} -> {destination}")
            return True
        except Exception as e:
            self.logger.error(f"複製文件 {source} 到 {destination} 失敗: {str(e)}")
            return False

    def move_file(self, source, destination):
        """移動文件
        
        Args:
            source: 源文件路徑
            destination: 目標文件路徑
            
        Returns:
            bool: 移動是否成功
        """
        try:
            # 確保目標目錄存在
            dest_dir = os.path.dirname(destination)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                
            # 如果目標文件已存在且啟用了備份功能，則進行備份
            if os.path.exists(destination):
                self._create_backup(destination)
                
            shutil.move(source, destination)
            
            self.logger.debug(f"已移動文件: {source} -> {destination}")
            return True
        except Exception as e:
            self.logger.error(f"移動文件 {source} 到 {destination} 失敗: {str(e)}")
            return False

    def delete_file(self, file_path, backup=True):
        """刪除文件
        
        Args:
            file_path: 文件路徑
            backup: 是否在刪除前創建備份
            
        Returns:
            bool: 刪除是否成功
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"要刪除的文件不存在: {file_path}")
                return False
                
            # 如果啟用了備份功能，則進行備份
            if backup:
                self._create_backup(file_path)
                
            os.remove(file_path)
            
            self.logger.debug(f"已刪除文件: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"刪除文件 {file_path} 失敗: {str(e)}")
            return False

    def get_file_info(self, file_path):
        """獲取文件信息
        
        Args:
            file_path: 文件路徑
            
        Returns:
            dict: 包含文件信息的字典，失敗則返回None
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"文件不存在: {file_path}")
                return None
                
            stat = os.stat(file_path)
            
            info = {
                'path': file_path,
                'size': stat.st_size,
                'size_human': self._format_size(stat.st_size),
                'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'accessed': datetime.fromtimestamp(stat.st_atime).strftime('%Y-%m-%d %H:%M:%S'),
                'extension': os.path.splitext(file_path)[1],
                'filename': os.path.basename(file_path),
                'directory': os.path.dirname(file_path),
                'is_file': os.path.isfile(file_path),
                'is_dir': os.path.isdir(file_path)
            }
            
            return info
        except Exception as e:
            self.logger.error(f"獲取文件 {file_path} 信息失敗: {str(e)}")
            return None

    def _format_size(self, size_bytes):
        """將字節大小格式化為人類可讀形式
        
        Args:
            size_bytes: 字節大小
            
        Returns:
            str: 格式化後的大小字符串
        """
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

    def create_zip(self, output_path, file_paths, compression=zipfile.ZIP_DEFLATED):
        """創建ZIP壓縮文件
        
        Args:
            output_path: 輸出的ZIP文件路徑
            file_paths: 要包含的文件路徑列表
            compression: 壓縮方式
            
        Returns:
            bool: 創建是否成功
        """
        try:
            # 確保輸出目錄存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 如果文件已存在且啟用了備份功能，則進行備份
            if os.path.exists(output_path):
                self._create_backup(output_path)
                
            with zipfile.ZipFile(output_path, 'w', compression) as zipf:
                for file in file_paths:
                    if os.path.exists(file):
                        # 使用相對路徑添加到ZIP中
                        arc_name = os.path.basename(file)
                        zipf.write(file, arc_name)
                    else:
                        self.logger.warning(f"要添加到ZIP的文件不存在: {file}")
            
            self.logger.debug(f"已創建ZIP文件: {output_path}，包含 {len(file_paths)} 個文件")
            return True
        except Exception as e:
            self.logger.error(f"創建ZIP文件 {output_path} 失敗: {str(e)}")
            return False

    def extract_zip(self, zip_path, extract_to=None):
        """解壓ZIP文件
        
        Args:
            zip_path: ZIP文件路徑
            extract_to: 解壓目標目錄，如果為None則解壓到ZIP文件所在目錄
            
        Returns:
            bool: 解壓是否成功
        """
        try:
            if extract_to is None:
                extract_to = os.path.dirname(zip_path)
                
            # 確保解壓目錄存在
            os.makedirs(extract_to, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(extract_to)
                
            self.logger.debug(f"已解壓ZIP文件: {zip_path} -> {extract_to}")
            return True
        except Exception as e:
            self.logger.error(f"解壓ZIP文件 {zip_path} 失敗: {str(e)}")
            return False

    def generate_filename(self, prefix=None, suffix=None, extension=None, with_timestamp=True, timestamp_format="%Y%m%d_%H%M%S"):
        """生成文件名
        
        Args:
            prefix: 文件名前綴
            suffix: 文件名後綴
            extension: 文件擴展名（不包含點號）
            with_timestamp: 是否包含時間戳
            timestamp_format: 時間戳格式
            
        Returns:
            str: 生成的文件名
        """
        parts = []
        
        if prefix:
            parts.append(prefix)
            
        if with_timestamp:
            timestamp = datetime.now().strftime(timestamp_format)
            parts.append(timestamp)
            
        if suffix:
            parts.append(suffix)
            
        filename = "_".join(parts)
        
        if extension:
            if not extension.startswith('.'):
                extension = '.' + extension
            filename += extension
            
        return filename

    def get_project_stats(self):
        """獲取項目統計信息
        
        Returns:
            dict: 項目統計信息
        """
        try:
            stats = {
                'data_files': len(self.list_files(self.data_dir, recursive=True)),
                'result_files': len(self.list_files(self.results_dir, recursive=True)),
                'log_files': len(self.list_files(self.log_dir, recursive=True)),
                'model_files': len(self.list_files(self.model_dir, recursive=True)),
                'export_files': len(self.list_files(self.export_dir, recursive=True)),
                'visualization_files': len(self.list_files(self.visualizations_dir, recursive=True)),
                'data_size': self._get_dir_size(self.data_dir),
                'results_size': self._get_dir_size(self.results_dir),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 格式化大小
            stats['data_size_human'] = self._format_size(stats['data_size'])
            stats['results_size_human'] = self._format_size(stats['results_size'])
            
            return stats
        except Exception as e:
            self.logger.error(f"獲取項目統計信息失敗: {str(e)}")
            return {}

    def _get_dir_size(self, directory):
        """獲取目錄大小（以字節為單位）
        
        Args:
            directory: 目錄路徑
            
        Returns:
            int: 目錄大小（字節）
        """
        total_size = 0
        
        try:
            if not os.path.exists(directory):
                return 0
                
            for path, dirs, files in os.walk(directory):
                for f in files:
                    file_path = os.path.join(path, f)
                    total_size += os.path.getsize(file_path)
                    
            return total_size
        except Exception as e:
            self.logger.error(f"計算目錄 {directory} 大小失敗: {str(e)}")
            return 0

    def monitor_file(self, file_path, timeout=30, check_interval=1, initial_delay=0):
        """監控文件變化
        
        Args:
            file_path: 要監控的文件路徑
            timeout: 最大監控時間（秒）
            check_interval: 檢查間隔（秒）
            initial_delay: 初始延遲（秒）
            
        Returns:
            bool: 文件是否有變化
        """
        try:
            # 初始延遲
            if initial_delay > 0:
                time.sleep(initial_delay)
                
            if not os.path.exists(file_path):
                self.logger.warning(f"要監控的文件不存在: {file_path}")
                return False
                
            # 獲取初始修改時間
            initial_mtime = os.path.getmtime(file_path)
            initial_size = os.path.getsize(file_path)
            
            start_time = time.time()
            while time.time() - start_time < timeout:
                time.sleep(check_interval)
                
                # 檢查文件是否存在
                if not os.path.exists(file_path):
                    self.logger.warning(f"監控期間文件被刪除: {file_path}")
                    return True  # 文件被刪除也算是變化
                    
                # 檢查修改時間和大小是否變化
                current_mtime = os.path.getmtime(file_path)
                current_size = os.path.getsize(file_path)
                
                if current_mtime != initial_mtime or current_size != initial_size:
                    self.logger.debug(f"檢測到文件變化: {file_path}")
                    return True
                    
            self.logger.debug(f"在 {timeout} 秒內未檢測到文件變化: {file_path}")
            return False
        except Exception as e:
            self.logger.error(f"監控文件 {file_path} 失敗: {str(e)}")
            return False

    def watch_directory(self, directory, callback, patterns=None, recursive=False, timeout=None):
        """監控目錄中的文件變化
        注意: 此函數會阻塞當前線程，適合在單獨的線程中運行
        
        Args:
            directory: 要監控的目錄
            callback: 當文件變化時調用的回調函數，接收文件路徑參數
            patterns: 要監控的文件模式列表，如["*.txt", "*.csv"]
            recursive: 是否遞歸監控子目錄
            timeout: 最大監控時間（秒），如果為None則無限監控
            
        Returns:
            bool: 監控是否成功完成
        """
        try:
            if not os.path.exists(directory):
                self.logger.warning(f"要監控的目錄不存在: {directory}")
                return False
                
            # 獲取初始文件列表及其修改時間
            file_mtimes = {}
            
            def scan_directory():
                nonlocal file_mtimes
                new_file_mtimes = {}
                
                if recursive:
                    for root, _, files in os.walk(directory):
                        for file in files:
                            file_path = os.path.join(root, file)
                            
                            # 如果指定了模式，則檢查文件是否匹配
                            if patterns:
                                if not any(fnmatch.fnmatch(file, pattern) for pattern in patterns):
                                    continue
                                    
                            new_file_mtimes[file_path] = os.path.getmtime(file_path)
                else:
                    for file in os.listdir(directory):
                        file_path = os.path.join(directory, file)
                        if os.path.isfile(file_path):
                            # 如果指定了模式，則檢查文件是否匹配
                            if patterns:
                                if not any(fnmatch.fnmatch(file, pattern) for pattern in patterns):
                                    continue
                                    
                            new_file_mtimes[file_path] = os.path.getmtime(file_path)
                            
                return new_file_mtimes
            
            # 初始掃描
            file_mtimes = scan_directory()
            self.logger.debug(f"開始監控目錄: {directory}，初始文件數: {len(file_mtimes)}")
            
            start_time = time.time()
            while True:
                # 檢查超時
                if timeout is not None and time.time() - start_time > timeout:
                    self.logger.debug(f"目錄監控超時: {directory}")
                    break
                    
                time.sleep(1)  # 檢查間隔
                
                # 重新掃描目錄
                new_file_mtimes = scan_directory()
                
                # 檢查新增的文件
                for file_path, mtime in new_file_mtimes.items():
                    if file_path not in file_mtimes:
                        self.logger.debug(f"檢測到新文件: {file_path}")
                        callback(file_path)
                    elif mtime != file_mtimes[file_path]:
                        self.logger.debug(f"檢測到文件修改: {file_path}")
                        callback(file_path)
                
                # 檢查刪除的文件
                for file_path in file_mtimes:
                    if file_path not in new_file_mtimes:
                        self.logger.debug(f"檢測到文件刪除: {file_path}")
                        callback(file_path)
                
                # 更新文件列表
                file_mtimes = new_file_mtimes
                
            return True
        except Exception as e:
            self.logger.error(f"監控目錄 {directory} 失敗: {str(e)}")
            return False

    def search_files(self, directory, keyword, file_patterns=None, search_content=False, recursive=True):
        """搜索文件名或文件內容
        
        Args:
            directory: 搜索目錄
            keyword: 搜索關鍵字
            file_patterns: 文件模式列表，如["*.txt", "*.csv"]
            search_content: 是否搜索文件內容
            recursive: 是否遞歸搜索子目錄
            
        Returns:
            list: 匹配的文件路徑列表
        """
        try:
            import fnmatch
            
            if not os.path.exists(directory):
                self.logger.warning(f"搜索目錄不存在: {directory}")
                return []
                
            matching_files = []
            
            # 遍歷目錄
            if recursive:
                for root, _, files in os.walk(directory):
                    for file in files:
                        # 檢查文件名模式
                        if file_patterns and not any(fnmatch.fnmatch(file, pattern) for pattern in file_patterns):
                            continue
                            
                        file_path = os.path.join(root, file)
                        
                        # 檢查文件名是否包含關鍵字
                        if keyword.lower() in file.lower():
                            matching_files.append(file_path)
                            continue
                            
                        # 如果需要搜索文件內容
                        if search_content:
                            try:
                                # 嘗試讀取文件內容（僅處理文本文件）
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    if keyword.lower() in content.lower():
                                        matching_files.append(file_path)
                            except:
                                # 忽略無法讀取的文件
                                pass
            else:
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        # 檢查文件名模式
                        if file_patterns and not any(fnmatch.fnmatch(file, pattern) for pattern in file_patterns):
                            continue
                            
                        # 檢查文件名是否包含關鍵字
                        if keyword.lower() in file.lower():
                            matching_files.append(file_path)
                            continue
                            
                        # 如果需要搜索文件內容
                        if search_content:
                            try:
                                # 嘗試讀取文件內容（僅處理文本文件）
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    if keyword.lower() in content.lower():
                                        matching_files.append(file_path)
                            except:
                                # 忽略無法讀取的文件
                                pass
                                
            self.logger.debug(f"搜索 '{keyword}' 找到 {len(matching_files)} 個匹配文件")
            return matching_files
        except Exception as e:
            self.logger.error(f"搜索文件失敗: {str(e)}")
            return []

    def clean_directory(self, directory, patterns=None, older_than=None, max_files=None, keep_latest=0):
        """清理目錄中的文件
        
        Args:
            directory: 要清理的目錄
            patterns: 要清理的文件模式列表，如["*.tmp", "*.log"]
            older_than: 清理早於指定天數的文件
            max_files: 如果目錄中文件數超過此值，則刪除最舊的文件
            keep_latest: 保留最新的N個文件
            
        Returns:
            dict: 清理結果統計
        """
        try:
            import fnmatch
            
            if not os.path.exists(directory):
                self.logger.warning(f"要清理的目錄不存在: {directory}")
                return {'status': 'failed', 'reason': '目錄不存在', 'deleted': 0}
                
            # 獲取目錄中的所有文件
            files = []
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    # 檢查文件是否匹配模式
                    if patterns and not any(fnmatch.fnmatch(file, pattern) for pattern in patterns):
                        continue
                        
                    # 獲取文件修改時間
                    mtime = os.path.getmtime(file_path)
                    files.append((file_path, mtime))
            
            # 按修改時間排序（最舊的在前面）
            files.sort(key=lambda x: x[1])
            
            # 計算要刪除的文件
            to_delete = []
            
            # 如果有older_than參數，刪除早於指定天數的文件
            if older_than is not None:
                cutoff_time = time.time() - older_than * 24 * 3600
                to_delete.extend(f for f, mtime in files if mtime < cutoff_time)
            
            # 如果有max_files參數，刪除超過最大數量的最舊文件
            if max_files is not None and len(files) > max_files:
                # 計算要刪除的文件數量（考慮keep_latest參數）
                delete_count = len(files) - max(max_files, keep_latest)
                if delete_count > 0:
                    to_delete.extend(f for f, _ in files[:delete_count])
            
            # 如果只有keep_latest參數，保留最新的N個文件
            elif keep_latest > 0 and len(files) > keep_latest:
                to_delete.extend(f for f, _ in files[:-keep_latest])
            
            # 去重
            to_delete = list(set(to_delete))
            
            # 刪除文件
            deleted_count = 0
            for file_path in to_delete:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    self.logger.debug(f"已刪除文件: {file_path}")
                except Exception as e:
                    self.logger.warning(f"無法刪除文件 {file_path}: {str(e)}")
            
            self.logger.info(f"清理目錄 {directory} 完成，刪除了 {deleted_count} 個文件")
            return {'status': 'success', 'deleted': deleted_count, 'total_files': len(files)}
        except Exception as e:
            self.logger.error(f"清理目錄 {directory} 失敗: {str(e)}")
            return {'status': 'failed', 'reason': str(e), 'deleted': 0}
            
            
# 測試代碼
if __name__ == "__main__":
    # 配置日誌級別
    logger.setLevel(logging.INFO)
    
    # 創建文件管理器
    file_mgr = FileManager()
    
    # 測試基本功能
    print("Base directory:", file_mgr.output_dir)
    
    # 測試寫入文本文件
    test_content = "This is a test file.\n測試中文內容。"
    result = file_mgr.write_file(f"{file_mgr.output_dir}/test_output.txt", test_content)
    print(f"Write file result: {result}")
    
    # 測試讀取文本文件
    content = file_mgr.read_file(f"{file_mgr.output_dir}/test_output.txt")
    print(f"Read file content: {content}")
    
    # 測試寫入JSON文件
    test_data = {"name": "測試", "value": 123, "items": [1, 2, 3]}
    result = file_mgr.write_json(f"{file_mgr.output_dir}/test_output.json", test_data)
    print(f"Write JSON result: {result}")
    
    # 測試讀取JSON文件
    json_data = file_mgr.read_json(f"{file_mgr.output_dir}/test_output.json")
    print(f"Read JSON data: {json_data}")
    
    # 測試獲取文件信息
    file_info = file_mgr.get_file_info(f"{file_mgr.output_dir}/test_output.txt")
    print(f"File info: {file_info}")