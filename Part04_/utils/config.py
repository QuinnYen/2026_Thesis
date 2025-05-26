"""
配置管理模組 - 負責管理應用程式的配置與設定
"""

import os
import json
import logging
from pathlib import Path
import copy

# 導入系統日誌模組
from utils.logger import get_logger

# 獲取logger
logger = get_logger("config")

class Config:
    """配置管理類，處理應用程式的配置讀取、保存和訪問"""
    
    def __init__(self, config_path=None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路徑，若為None則使用預設路徑
        """
        self.logger = logger
        self.config = {}
        self.config_path = config_path
        
        # 如果未指定配置文件路徑，使用預設路徑
        if self.config_path is None:
            self.config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         "utils", "settings", "config.json")
        
        # 載入配置
        self.load()
        
    def load(self, path=None):
        """從指定路徑載入配置
        
        Args:
            path: 配置文件路徑，若為None則使用初始化時設定的路徑
            
        Returns:
            bool: 是否成功載入配置
        """
        if path is not None:
            self.config_path = path
            
        try:
            # 檢查配置文件是否存在
            if not os.path.exists(self.config_path):
                self.logger.warning(f"配置文件不存在: {self.config_path}，將創建默認配置")
                self._create_default_config()
                return True
                
            # 讀取配置文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                
            self.logger.info(f"成功載入配置: {self.config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"載入配置出錯: {str(e)}")
            self._create_default_config()
            return False
            
    def save(self, path=None):
        """保存配置到指定路徑
        
        Args:
            path: 配置文件路徑，若為None則使用當前配置路徑
            
        Returns:
            bool: 是否成功保存配置
        """
        try:
            save_path = path if path is not None else self.config_path
            
            # 確保目錄存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存配置
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"成功保存配置: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存配置出錯: {str(e)}")
            return False
    
    def _create_default_config(self):
        """創建默認配置"""
        try:
            app_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            
            self.config = {
                "app": {
                    "name": "跨領域情感分析系統",
                    "version": "0.1.0",
                    "language": "zh_TW"
                },
                "paths": {
                    "data_dir": os.path.join(app_dir, "data"),
                    "output_dir": os.path.join(app_dir, "1_output"),
                    "logs_dir": os.path.join(app_dir, "1_output", "logs"),
                    "resources_dir": os.path.join(app_dir, "resources"),
                    "models_dir": os.path.join(app_dir, "1_output", "models"),
                    "visualizations_dir": os.path.join(app_dir, "1_output", "visualizations")
                },
                "data_processing": {
                    "encoding": "utf-8",
                    "max_sample_size": 5000,
                    "random_seed": 42,
                    "clean_html": True,
                    "remove_stopwords": True,
                    "lemmatize": True
                },
                "bert": {
                    "model_name": "bert-base-uncased",
                    "max_length": 128,
                    "batch_size": 16,
                    "use_gpu": True
                },
                "lda": {
                    "n_topics": 10,
                    "max_iter": 50,
                    "random_state": 42,
                    "n_top_words": 15
                },
                "attention": {
                    "enabled_mechanisms": ["no", "similarity", "keyword", "self", "combined"],
                    "similarity_temperature": 0.1,
                    "keyword_weight": 0.5
                },
                "evaluation": {
                    "coherence_weight": 0.5,
                    "separation_weight": 0.5,
                    "visualizations": True,
                    "report_format": "html"
                },
                "visualization": {
                    "output_dir": os.path.join(app_dir, "1_output", "visualizations"),
                    "dpi": 300,
                    "figsize": [12, 8],
                    "cmap": "viridis",
                    "show_values": True
                }
            }
            
            # 延遲創建目錄，直到實際需要時才創建
            # 註釋掉自動目錄創建，避免在應用程式啟動時生成空資料夾
            # for path_key, path_value in self.config["paths"].items():
            #     os.makedirs(path_value, exist_ok=True)
                
            # 保存默認配置
            self.save()
            self.logger.info("已創建默認配置")
            
        except Exception as e:
            self.logger.error(f"創建默認配置出錯: {str(e)}")
    
    def get(self, section, key=None, default=None):
        """獲取配置項
        
        Args:
            section: 配置區段或巢狀路徑列表，例如：'app' 或 ['app', 'name']
            key: 配置鍵名，若為None則返回整個區段
            default: 默認值，若配置項不存在則返回此值
            
        Returns:
            配置值或默認值
        """
        try:
            # 支援使用列表表示巢狀路徑，例如 ['app', 'name']
            if isinstance(section, list):
                return self._get_by_path(section, key, default)
                
            # 向下兼容：支援使用字典表示巢狀路徑，但不發出警告
            if isinstance(section, dict):
                # 嘗試轉換字典為字符串鍵
                try:
                    section_key = next(iter(section.keys()))
                    if isinstance(section_key, str):
                        return self.get_nested(section, key, default)
                    else:
                        return default
                except:
                    return default
                
            # 確保 section 是字符串型別
            if not isinstance(section, str) and not isinstance(section, int):
                self.logger.debug(f"配置區段名稱必須是字符串或整數，收到: {type(section)}")
                return default
                
            if section not in self.config:
                return default
                
            if key is None:
                return copy.deepcopy(self.config[section])
            
            # 確保 key 是可哈希的類型
            if not isinstance(key, str) and not isinstance(key, int):
                self.logger.debug(f"配置鍵名必須是字符串或整數，收到: {type(key)}")
                return default
                
            # 檢查 section 是否為字典
            if isinstance(self.config[section], dict):
                if key in self.config[section]:
                    return self.config[section][key]
                else:
                    return default
            
            # 如果 section 不是字典，返回默認值
            return default
        except Exception as e:
            self.logger.error(f"獲取配置項出錯: {str(e)}")
            return default
    
    def _get_by_path(self, path, key=None, default=None):
        """根據路徑列表獲取巢狀配置
        
        Args:
            path: 配置路徑列表，例如 ['app', 'name']
            key: 最終鍵名，若為None則返回巢狀字典
            default: 默認值，若配置項不存在則返回此值
            
        Returns:
            配置值或默認值
        """
        try:
            if not isinstance(path, list):
                self.logger.warning(f"路徑必須是列表，而不是 {type(path)}")
                return default
                
            # 從配置中依序查找
            config_ptr = self.config
            for item in path:
                if not isinstance(item, (str, int)):
                    self.logger.warning(f"路徑項必須是字符串或整數，而不是 {type(item)}")
                    return default
                    
                if isinstance(config_ptr, dict) and item in config_ptr:
                    config_ptr = config_ptr[item]
                else:
                    return default
            
            # 已找到巢狀字典
            if key is None:
                return copy.deepcopy(config_ptr)
            
            # 檢查最終鍵名
            if isinstance(config_ptr, dict) and key in config_ptr:
                return config_ptr[key]
            
            return default
        except Exception as e:
            self.logger.error(f"獲取巢狀配置項出錯: {str(e)}")
            return default
    
    def get_nested(self, path_dict, key=None, default=None):
        """獲取巢狀配置項 (舊版方法，建議使用列表路徑)
        
        Args:
            path_dict: 配置路徑字典，例如 {'section': 'subsection'}
            key: 最終鍵名，若為None則返回巢狀字典
            default: 默認值，若配置項不存在則返回此值
            
        Returns:
            配置值或默認值
        """
        try:
            # 將路徑字典轉換成路徑列表
            path_items = []
            for k, v in path_dict.items():
                path_items.append(k)
                if v is not None:
                    path_items.append(v)
            
            # 使用新方法處理
            return self._get_by_path(path_items, key, default)
        except Exception as e:
            self.logger.error(f"獲取巢狀配置項出錯: {str(e)}")
            return default
    
    def set(self, section, key, value):
        """設置配置項
        
        Args:
            section: 配置區段
            key: 配置鍵名
            value: 配置值
            
        Returns:
            bool: 是否成功設置配置項
        """
        try:
            # 確保區段存在
            if section not in self.config:
                self.config[section] = {}
                
            # 設置配置項
            self.config[section][key] = value
            
            return True
        except Exception as e:
            self.logger.error(f"設置配置項出錯: {str(e)}")
            return False
    
    def update_section(self, section, values):
        """更新配置區段
        
        Args:
            section: 配置區段
            values: 要更新的鍵值對字典
            
        Returns:
            bool: 是否成功更新配置區段
        """
        try:
            # 確保區段存在
            if section not in self.config:
                self.config[section] = {}
                
            # 更新區段
            self.config[section].update(values)
            
            return True
        except Exception as e:
            self.logger.error(f"更新配置區段出錯: {str(e)}")
            return False
    
    def get_all(self):
        """獲取所有配置
        
        Returns:
            dict: 所有配置的深複製
        """
        return copy.deepcopy(self.config)
    
    def __getitem__(self, key):
        """支持字典訪問語法獲取配置區段
        
        Args:
            key: 配置區段名
            
        Returns:
            dict: 配置區段
        """
        # 確保 key 是可哈希的類型
        if not isinstance(key, str) and not isinstance(key, int):
            self.logger.warning(f"配置區段名稱必須是字符串或整數，而不是 {type(key)}")
            return {}
            
        return self.config.get(key, {})