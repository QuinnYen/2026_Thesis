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
                    "output_dir": os.path.join("Part04_", "1_output"),
                    "logs_dir": os.path.join("Part04_", "1_output", "logs"),
                    "resources_dir": os.path.join(app_dir, "resources"),
                    "models_dir": os.path.join("Part04_", "1_output", "models"),
                    "visualizations_dir": os.path.join("Part04_", "1_output", "visualizations")
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
                    "output_dir": os.path.join("Part04_", "1_output", "visualizations"),
                    "dpi": 300,
                    "figsize": [12, 8],
                    "cmap": "viridis",
                    "show_values": True
                }
            }
            
            # 確保目錄存在
            for path_key, path_value in self.config["paths"].items():
                os.makedirs(path_value, exist_ok=True)
                
            # 保存默認配置
            self.save()
            self.logger.info("已創建默認配置")
            
        except Exception as e:
            self.logger.error(f"創建默認配置出錯: {str(e)}")
    
    def get(self, section, key=None, default=None):
        """獲取配置項
        
        Args:
            section: 配置區段
            key: 配置鍵名，若為None則返回整個區段
            default: 默認值，若配置項不存在則返回此值
            
        Returns:
            配置值或默認值
        """
        try:
            if section not in self.config:
                return default
                
            if key is None:
                return self.config[section]
            
            # 修正: 先檢查 section 是否為字典
            if isinstance(self.config[section], dict):
                return self.config[section].get(key, default)
            
            # 如果 section 不是字典，返回默認值
            return default
        except Exception as e:
            self.logger.error(f"獲取配置項出錯: {str(e)}")
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
        return self.config.get(key, {})