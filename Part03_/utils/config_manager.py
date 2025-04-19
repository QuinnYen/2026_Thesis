"""
設定管理模組
此模組提供讀取和管理配置文件的功能
"""

import os
import json
import yaml
from typing import Dict, Any, Optional

class ConfigManager:
    """
    配置管理器類
    用於讀取、修改和保存項目配置
    """
    
    def __init__(self, config_dir: str = './Part03_/config', default_config_file: str = 'default_config.yaml'):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件所在目錄
            default_config_file: 預設配置文件名稱
        """
        self.config_dir = config_dir
        self.default_config_file = default_config_file
        self.current_config = {}
        self.user_config_file = os.path.join(config_dir, 'user_config.json')
        
        # 確保配置目錄存在
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        
        # 載入默認配置
        self._load_default_config()
        
        # 載入用戶配置（如果存在）
        self._load_user_config()
    
    def _load_default_config(self) -> None:
        """載入默認配置"""
        default_config_path = os.path.join(self.config_dir, self.default_config_file)
        
        # 如果默認配置不存在，創建一個基本配置
        if not os.path.exists(default_config_path):
            self._create_default_config()
        
        # 根據文件擴展名選擇讀取方式
        if default_config_path.endswith('.yaml') or default_config_path.endswith('.yml'):
            try:
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    self.current_config = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"無法讀取默認配置文件: {str(e)}")
                self.current_config = {}
        else:
            try:
                with open(default_config_path, 'r', encoding='utf-8') as f:
                    self.current_config = json.load(f)
            except Exception as e:
                print(f"無法讀取默認配置文件: {str(e)}")
                self.current_config = {}
    
    def _create_default_config(self) -> None:
        """創建默認配置文件"""
        default_config = {
            "data_settings": {
                "input_directories": {
                    "amazon": "./ReviewsDataBase/Amazon/",
                    "yelp": "./ReviewsDataBase/Yelp/",
                    "imdb": "./ReviewsDataBase/IMDB/"
                },
                "output_directory": "./Part03_/results/",
                "temp_directory": "./Part03_/temp/"
            },
            "model_settings": {
                "bert": {
                    "model_name": "bert-base-chinese",
                    "cache_dir": "./Part03_/models/bert",
                    "max_length": 128,
                    "batch_size": 16
                },
                "lda": {
                    "num_topics": 10,
                    "passes": 15,
                    "alpha": "auto",
                    "workers": 4
                }
            },
            "processing": {
                "use_cuda": True,
                "num_workers": 4,
                "cache_embeddings": True
            },
            "logging": {
                "log_level": "INFO",
                "log_dir": "./Part03_/logs/",
                "console_output": True
            },
            "gui": {
                "theme": "light",
                "language": "zh_TW",
                "auto_save": True,
                "chart_style": "seaborn"
            }
        }
        
        # 根據擴展名保存為不同格式
        default_config_path = os.path.join(self.config_dir, self.default_config_file)
        if default_config_path.endswith('.yaml') or default_config_path.endswith('.yml'):
            with open(default_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(default_config, f, allow_unicode=True, sort_keys=False)
        else:
            with open(default_config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        
        self.current_config = default_config
    
    def _load_user_config(self) -> None:
        """載入用戶配置檔案並與默認配置合併"""
        if not os.path.exists(self.user_config_file):
            return
        
        try:
            with open(self.user_config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # 遞迴合併配置
            self._merge_configs(self.current_config, user_config)
        except Exception as e:
            print(f"無法讀取用戶配置: {str(e)}")
    
    def _merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> None:
        """
        遞迴地將用戶配置合併到默認配置中
        
        Args:
            default_config: 默認配置字典
            user_config: 用戶配置字典
        """
        for key, value in user_config.items():
            if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                self._merge_configs(default_config[key], value)
            else:
                default_config[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        獲取配置項的值
        
        Args:
            key: 配置項的鍵，使用點號分隔來訪問嵌套項（例如：'model_settings.bert.model_name'）
            default: 如果配置項不存在時返回的默認值
            
        Returns:
            配置項的值或默認值
        """
        if not key:
            return self.current_config
        
        # 處理嵌套鍵
        keys = key.split('.')
        value = self.current_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        設置配置項的值
        
        Args:
            key: 配置項的鍵，使用點號分隔來訪問嵌套項
            value: 要設置的值
        """
        if not key:
            return
        
        # 處理嵌套鍵
        keys = key.split('.')
        config = self.current_config
        
        # 找到最後一個鍵的父級
        for i, k in enumerate(keys[:-1]):
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # 設置最後一個鍵的值
        config[keys[-1]] = value
    
    def save_user_config(self) -> None:
        """保存當前配置到用戶配置文件"""
        try:
            with open(self.user_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"無法保存用戶配置: {str(e)}")
    
    def reset_to_default(self) -> None:
        """重置所有配置為默認值"""
        self._load_default_config()
        
        # 如果存在用戶配置文件，則刪除
        if os.path.exists(self.user_config_file):
            try:
                os.remove(self.user_config_file)
            except:
                print("無法刪除用戶配置文件")

# 使用範例
if __name__ == "__main__":
    # 創建配置管理器
    config_manager = ConfigManager()
    
    # 讀取配置
    model_name = config_manager.get('model_settings.bert.model_name')
    print(f"當前 BERT 模型: {model_name}")
    
    # 修改配置
    config_manager.set('model_settings.bert.batch_size', 32)
    print(f"更新後的 batch_size: {config_manager.get('model_settings.bert.batch_size')}")
    
    # 保存用戶配置
    config_manager.save_user_config()
