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
        
        # 檢查是否為指定的專案目錄結構
        current_dir = Path(__file__).parent.parent.absolute()
        project_root = current_dir.parent
        
        # 如果是在 2026_Thesis 專案中，使用指定的輸出路徑
        if project_root.name == "2026_Thesis":
            output_dir = project_root / "output"
        else:
            # 否則使用當前目錄下的 output 資料夾
            output_dir = current_dir / "output"
        
        self.config["base_output_dir"] = str(output_dir)
    
    def get_base_output_dir(self) -> str:
        """獲取基礎輸出目錄"""
        return self.config["base_output_dir"]
    
    def get_subdirectory_name(self, subdir_key: str) -> str:
        """獲取子目錄名稱"""
        return self.config["subdirectories"].get(subdir_key, subdir_key)
    
    def get_file_pattern(self, file_key: str) -> str:
        """獲取檔案命名模式"""
        return self.config["file_patterns"].get(file_key, f"{file_key}.txt")
    
    def get_full_subdirectory_path(self, run_dir: str, subdir_key: str) -> str:
        """獲取完整子目錄路徑"""
        subdir_name = self.get_subdirectory_name(subdir_key)
        return os.path.join(run_dir, subdir_name)
    
    def get_full_file_path(self, directory: str, file_key: str) -> str:
        """獲取完整檔案路徑"""
        file_pattern = self.get_file_pattern(file_key)
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


def get_attention_testing_dir_name() -> str:
    """獲取注意力測試目錄名稱"""
    return path_config.get_subdirectory_name("attention_testing")


def get_analysis_dir_name() -> str:
    """獲取分析目錄名稱"""
    return path_config.get_subdirectory_name("analysis")