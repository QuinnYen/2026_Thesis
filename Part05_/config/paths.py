"""
路徑配置模組
統一管理所有輸出路徑設定，支援跨裝置開發
"""
import os
import json
from pathlib import Path
from typing import Dict, Optional


class PathConfig:
    """路徑配置管理器"""
    
    # 預設配置
    DEFAULT_CONFIG = {
        "base_output_dir": "",  # 將由程式動態設定
        "subdirectories": {
            "preprocessing": "01_preprocessing",
            "bert_encoding": "02_bert_encoding", 
            "attention_testing": "03_attention_testing",
            "analysis": "04_analysis"
        },
        "file_patterns": {
            "preprocessed_data": "01_preprocessed_data.csv",
            "bert_embeddings": "02_bert_embeddings.npy",
            "gpt_embeddings": "02_gpt_embeddings.npy", 
            "t5_embeddings": "02_t5_embeddings.npy",
            "cnn_embeddings": "02_cnn_embeddings.npy",
            "elmo_embeddings": "02_elmo_embeddings.npy",
            "complete_analysis": "complete_analysis_results.json",
            "multiple_analysis": "multiple_combinations_analysis_results.json"
        }
    }
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化路徑配置
        
        Args:
            config_file: 配置檔案路徑，如果為None則使用預設配置
        """
        self.config_file = config_file
        self.config = self.DEFAULT_CONFIG.copy()
        
        # 嘗試從配置檔案載入設定
        if config_file and os.path.exists(config_file):
            self._load_config()
        
        # 如果沒有設定基礎輸出目錄，則使用環境變數或預設值  
        if not self.config["base_output_dir"]:
            self._set_default_output_dir()
    
    def _load_config(self):
        """從JSON檔案載入配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        except Exception as e:
            print(f"警告：無法載入配置檔案 {self.config_file}: {e}")
            print("使用預設配置")
    
    def _set_default_output_dir(self):
        """設定預設輸出目錄"""
        # 優先使用環境變數
        env_output = os.getenv('BERT_OUTPUT_DIR')
        if env_output:
            self.config["base_output_dir"] = env_output
            return
        
        # 獲取當前文件的路徑，並向上尋找專案根目錄
        current_file = Path(__file__).absolute()
        
        # 從當前文件位置開始向上搜尋，找到包含Part05_的目錄的父目錄
        search_path = current_file.parent
        while search_path.parent != search_path:  # 避免無限循環到根目錄
            # 檢查當前目錄是否包含Part05_目錄
            part05_dir = search_path / "Part05_"
            if part05_dir.exists() and part05_dir.is_dir():
                # 找到專案根目錄，使用其下的output目錄
                output_dir = search_path / "output"
                self.config["base_output_dir"] = str(output_dir)
                return
            search_path = search_path.parent
        
        # 如果沒有找到標準結構，使用當前目錄的相對位置
        # 假設當前文件在 Part05_/config/paths.py
        fallback_output = current_file.parent.parent.parent / "output"
        self.config["base_output_dir"] = str(fallback_output)
    
    def get_base_output_dir(self) -> str:
        """獲取基礎輸出目錄"""
        return self.config["base_output_dir"]
    
    def get_subdirectory_name(self, subdir_key: str, encoder_type: str = 'bert') -> str:
        """獲取子目錄名稱，支援編碼器類型動態替換"""
        pattern = self.config["subdirectories"].get(subdir_key, subdir_key)
        if "{encoder_type}" in pattern:
            return pattern.format(encoder_type=encoder_type)
        return pattern
    
    def get_file_pattern(self, file_key: str, encoder_type: str = 'bert') -> str:
        """獲取檔案命名模式，支援編碼器類型動態替換"""
        # 優先使用通用模式
        if file_key == "embeddings" or (file_key.endswith("_embeddings") and file_key != f"{encoder_type}_embeddings"):
            pattern = self.config["file_patterns"].get("embeddings", f"02_{encoder_type}_embeddings.npy")
        else:
            pattern = self.config["file_patterns"].get(file_key, f"{file_key}.txt")
        
        if "{encoder_type}" in pattern:
            return pattern.format(encoder_type=encoder_type)
        return pattern
    
    def get_full_subdirectory_path(self, run_dir: str, subdir_key: str, encoder_type: str = 'bert') -> str:
        """獲取完整子目錄路徑"""
        subdir_name = self.get_subdirectory_name(subdir_key, encoder_type)
        return os.path.join(run_dir, subdir_name)
    
    def get_full_file_path(self, directory: str, file_key: str, encoder_type: str = 'bert') -> str:
        """獲取完整檔案路徑"""
        file_pattern = self.get_file_pattern(file_key, encoder_type)
        return os.path.join(directory, file_pattern)
    
    def save_config(self, config_file: Optional[str] = None):
        """儲存配置到檔案"""
        target_file = config_file or self.config_file
        if not target_file:
            return
        
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"警告：無法儲存配置檔案 {target_file}: {e}")
    
    def set_base_output_dir(self, output_dir: str):
        """設定基礎輸出目錄"""
        self.config["base_output_dir"] = output_dir
        
        # 如果有配置檔案，儲存更新
        if self.config_file:
            self.save_config()
    
    def create_config_template(self, config_file: str):
        """創建配置檔案範本"""
        template_config = self.DEFAULT_CONFIG.copy()
        template_config["base_output_dir"] = "D:\\Project\\2026_Thesis\\output"
        
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(template_config, f, ensure_ascii=False, indent=2)
            print(f"已創建配置檔案範本：{config_file}")
        except Exception as e:
            print(f"錯誤：無法創建配置檔案 {config_file}: {e}")


# 全域配置實例
# 優先嘗試載入自定義配置檔案
config_file_path = os.path.join(os.path.dirname(__file__), "paths_config.json")
path_config = PathConfig(config_file_path if os.path.exists(config_file_path) else None)


def get_path_config() -> PathConfig:
    """獲取全域路徑配置實例"""
    return path_config


def setup_custom_output_dir(output_dir: str):
    """設定自定義輸出目錄"""
    path_config.set_base_output_dir(output_dir)
    print(f"輸出目錄已設定為：{output_dir}")


# 便利函數
def get_base_output_dir() -> str:
    """獲取基礎輸出目錄"""
    return path_config.get_base_output_dir()


def get_preprocessing_dir_name() -> str:
    """獲取預處理目錄名稱"""
    return path_config.get_subdirectory_name("preprocessing")


def get_bert_encoding_dir_name() -> str:
    """獲取BERT編碼目錄名稱"""
    return path_config.get_subdirectory_name("bert_encoding")

def get_encoding_dir_name(encoder_type: str = 'bert') -> str:
    """獲取編碼器編碼目錄名稱（動態）"""
    return path_config.get_subdirectory_name("encoding", encoder_type)

def get_embeddings_file_pattern(encoder_type: str = 'bert') -> str:
    """獲取嵌入向量檔案名稱模式（動態）"""
    return path_config.get_file_pattern("embeddings", encoder_type)


def get_attention_testing_dir_name() -> str:
    """獲取注意力測試目錄名稱"""
    return path_config.get_subdirectory_name("attention_testing")


def get_analysis_dir_name() -> str:
    """獲取分析目錄名稱"""
    return path_config.get_subdirectory_name("analysis")