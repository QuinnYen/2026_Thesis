"""
配置管理模組 - 負責系統配置的讀取、保存和管理
"""
import os
import json
import logging

class Config:
    """配置管理類"""
    
    # 默認配置
    DEFAULT_CONFIG = {
        # 路徑配置
        "paths": {
            "data_dir": "./data",
            "output_dir": "./output",
            "model_dir": "./models",
            "log_dir": "./logs",
        },
        
        # BERT模型配置
        "bert": {
            "model_name": "bert-base-uncased",
            "max_length": 128,
            "batch_size": 16,
            "use_gpu": True,
        },
        
        # LDA模型配置
        "lda": {
            "n_topics": 10,
            "alpha": 0.1,
            "beta": 0.01,
            "max_iter": 50,
            "random_state": 42,
        },
        
        # 注意力機制配置
        "attention": {
            "enabled_mechanisms": ["similarity", "keyword", "self", "combined"],
            "weights": {
                "similarity": 0.33,
                "keyword": 0.33,
                "self": 0.34,
            }
        },
        
        # 評估指標配置
        "evaluation": {
            "metrics": ["coherence", "separation", "combined"],
            "coherence_weight": 0.5,
            "separation_weight": 0.5,
        },
        
        # 可視化配置
        "visualization": {
            "dpi": 300,
            "format": "png",
            "color_scheme": "viridis",
        },
        
        # 數據集配置
        "datasets": {
            "imdb": {
                "name": "IMDB電影評論",
                "type": "movie",
                "language": "english",
            },
            "amazon": {
                "name": "Amazon產品評論",
                "type": "product",
                "language": "english",
            },
            "yelp": {
                "name": "Yelp餐廳評論",
                "type": "restaurant",
                "language": "english",
            }
        },
        
        # 系統配置
        "system": {
            "log_level": "INFO",
            "multi_processing": True,
            "num_workers": 4,
        }
    }
    
    def __init__(self, config_file="config.json"):
        """初始化配置管理器
        
        Args:
            config_file: 配置文件路徑
        """
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 嘗試載入配置文件
        if os.path.exists(config_file):
            self.load()
        else:
            # 如果配置文件不存在，則創建目錄並保存默認配置
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)
            self.save()
    
    def get(self, section, key=None):
        """獲取配置項
        
        Args:
            section: 配置段名稱
            key: 配置項名稱，如果為None則返回整個段
            
        Returns:
            配置值
        """
        if key is None:
            return self.config.get(section, {})
        
        # 檢查 section 是否存在於配置中
        section_data = self.config.get(section, {})
        if isinstance(section_data, dict):
            # 如果 section_data 是字典，則直接獲取 key 對應的值
            return section_data.get(key)
        else:
            # 如果 section_data 不是字典，則返回 None 或者默認值
            return None
    
    def set(self, section, key, value):
        """設置配置項
        
        Args:
            section: 配置段名稱
            key: 配置項名稱
            value: 配置值
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def load(self):
        """從文件載入配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # 使用遞迴更新，保留默認配置中存在但加載的配置中不存在的項
                self._recursive_update(self.config, loaded_config)
            logging.info(f"配置已從 {self.config_file} 載入")
        except Exception as e:
            logging.error(f"載入配置文件時出錯: {str(e)}")
    
    def save(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=4)
            logging.info(f"配置已保存到 {self.config_file}")
        except Exception as e:
            logging.error(f"保存配置文件時出錯: {str(e)}")
    
    def _recursive_update(self, d, u):
        """遞迴更新字典
        
        Args:
            d: 目標字典
            u: 源字典
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._recursive_update(d[k], v)
            else:
                d[k] = v
    
    def reset_to_default(self):
        """重置為默認配置"""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()
    
    def get_all(self):
        """獲取所有配置"""
        return self.config