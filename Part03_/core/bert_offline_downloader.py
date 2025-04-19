"""
BERT模型離線下載模組
用於下載並管理BERT模型供離線使用
"""

import os
import sys
import shutil
import requests
import zipfile
import tarfile
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from tqdm import tqdm
import torch

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 從utils模組導入工具類
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.config_manager import ConfigManager

class BertOfflineDownloader:
    """
    BERT模型離線下載管理器
    負責下載、保存和管理預訓練BERT模型供離線使用
    """
    # 模型來源類型
    SOURCE_HUGGINGFACE = 'huggingface'
    SOURCE_LOCAL = 'local'
    
    # 預設使用的中文BERT模型
    DEFAULT_CHINESE_MODEL = 'bert-base-chinese'
    DEFAULT_ENGLISH_MODEL = 'bert-base-uncased'
    
    # 模型資源目錄
    DEFAULT_MODEL_DIR = './Part03_/results/models'
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化BERT模型下載器
        
        Args:
            config: 配置管理器，如不提供則使用預設設定
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 設置模型存儲目錄
        self.model_dir = self.config.get('models.bert_models_dir', self.DEFAULT_MODEL_DIR)
        
        # 創建目錄(如果不存在)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 設置日誌
        log_dir = self.config.get('logging.log_dir', './Part03_/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'bert_download.log')
        
        # 初始化日誌器
        self.log_file, self.status_file = ConsoleOutputManager.open_console(
            "BERT模型下載", log_to_file=True, clear_previous=False)
        
        self.logger = ConsoleOutputManager.setup_console_logger('bert_downloader', self.log_file)
        
        # 已下載模型的索引文件
        self.models_index_file = os.path.join(self.model_dir, 'models_index.json')
        
        # 讀取或創建模型索引
        self.models_index = self._load_models_index()
        
        self.logger.info(f"BERT模型下載器初始化完成，模型將存儲在: {self.model_dir}")

    def _load_models_index(self) -> Dict[str, Dict[str, Any]]:
        """
        載入模型索引文件，如不存在則創建
        
        Returns:
            models_index: 模型索引字典
        """
        if os.path.exists(self.models_index_file):
            try:
                with open(self.models_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"讀取模型索引文件時出錯: {str(e)}，將創建新索引")
        
        # 創建新的索引
        models_index = {}
        self._save_models_index(models_index)
        return models_index
    
    def _save_models_index(self, models_index: Dict[str, Dict[str, Any]]) -> None:
        """
        保存模型索引到文件
        
        Args:
            models_index: 模型索引字典
        """
        try:
            with open(self.models_index_file, 'w', encoding='utf-8') as f:
                json.dump(models_index, f, ensure_ascii=False, indent=2)
        except IOError as e:
            self.logger.error(f"保存模型索引文件時出錯: {str(e)}")
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有已下載的可用模型
        
        Returns:
            models_info: 已下載模型的信息字典
        """
        self.logger.info("列出已下載的模型...")
        
        # 重新掃描模型目錄，確保索引是最新的
        self._scan_local_models()
        
        if not self.models_index:
            self.logger.info("目前沒有已下載的模型")
            return {}
        
        # 打印模型信息
        for model_name, model_info in self.models_index.items():
            self.logger.info(f"模型名稱: {model_name}")
            self.logger.info(f"  - 路徑: {model_info.get('local_path', 'N/A')}")
            self.logger.info(f"  - 語言: {model_info.get('language', 'N/A')}")
            self.logger.info(f"  - 下載時間: {model_info.get('download_time', 'N/A')}")
            self.logger.info(f"  - 大小: {model_info.get('size', 'N/A')} MB")
            self.logger.info("  " + "-" * 30)
        
        return self.models_index
    
    def _scan_local_models(self) -> None:
        """
        掃描模型目錄，更新本地模型索引
        """
        models_found = set()
        
        # 遍歷模型目錄
        for item in os.listdir(self.model_dir):
            item_path = os.path.join(self.model_dir, item)
            
            # 判斷是否為目錄
            if os.path.isdir(item_path):
                # 檢查是否為BERT模型目錄
                if self._is_valid_model_dir(item_path):
                    model_name = item
                    models_found.add(model_name)
                    
                    # 如果模型不在索引中，則添加
                    if model_name not in self.models_index:
                        # 獲取目錄大小（MB）
                        size = self._get_dir_size(item_path) / (1024 * 1024)
                        
                        # 從config.json中獲取語言信息
                        language = self._detect_model_language(item_path)
                        
                        # 添加到索引
                        self.models_index[model_name] = {
                            'local_path': item_path,
                            'language': language,
                            'source': self.SOURCE_LOCAL,
                            'download_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'size': round(size, 2)
                        }
        
        # 從索引中移除不再存在的模型
        for model_name in list(self.models_index.keys()):
            if model_name not in models_found:
                self.logger.warning(f"模型 {model_name} 不再存在，從索引中移除")
                del self.models_index[model_name]
        
        # 保存更新後的索引
        self._save_models_index(self.models_index)
    
    def _is_valid_model_dir(self, model_dir: str) -> bool:
        """
        檢查目錄是否是有效的BERT模型目錄
        
        Args:
            model_dir: 模型目錄路徑
            
        Returns:
            is_valid: 是否為有效模型目錄
        """
        # 檢查必要的文件是否存在
        required_files = ['config.json', 'pytorch_model.bin']
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                return False
        
        return True
    
    def _get_dir_size(self, path: str) -> float:
        """
        計算目錄總大小（以字節為單位）
        
        Args:
            path: 目錄路徑
            
        Returns:
            size: 目錄總大小（字節）
        """
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        
        return total_size
    
    def _detect_model_language(self, model_dir: str) -> str:
        """
        從模型配置中檢測模型語言
        
        Args:
            model_dir: 模型目錄路徑
            
        Returns:
            language: 檢測到的語言
        """
        config_path = os.path.join(model_dir, 'config.json')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 從配置中獲取語言
                vocab_size = config.get('vocab_size', 0)
                
                # 簡單規則：vocab_size > 20000 通常為中文模型
                if vocab_size > 20000:
                    return 'Chinese'
                else:
                    return 'English'
            
            except (json.JSONDecodeError, IOError):
                pass
        
        # 默認返回未知
        return 'Unknown'

    def download_model(self, model_name: str = None, language: str = 'Chinese') -> str:
        """
        下載BERT模型
        
        Args:
            model_name: 要下載的模型名稱(默認根據語言選擇)
            language: 模型語言('Chinese'或'English')
            
        Returns:
            local_path: 下載后的本地路徑
        """
        # 根據語言選擇默認模型
        if model_name is None:
            if language.lower() == 'chinese':
                model_name = self.DEFAULT_CHINESE_MODEL
                self.logger.info(f"未指定模型名稱，使用默認中文BERT模型: {model_name}")
            else:
                model_name = self.DEFAULT_ENGLISH_MODEL
                self.logger.info(f"未指定模型名稱，使用默認英文BERT模型: {model_name}")
        
        # 檢查模型是否已下載
        if model_name in self.models_index:
            local_path = self.models_index[model_name]['local_path']
            self.logger.info(f"模型 {model_name} 已經下載，位於: {local_path}")
            return local_path
        
        # 下載前檢查PyTorch和transformers是否可用
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.logger.info("已確認PyTorch和transformers可用")
        except ImportError as e:
            error_msg = f"缺少必要的庫: {str(e)}"
            self.logger.error(error_msg)
            raise ImportError(error_msg + "\n請先安裝必要的庫: pip install torch transformers")
        
        # 開始下載
        self.logger.info(f"開始從Hugging Face下載模型: {model_name}")
        self.logger.info("這可能需要一些時間，請耐心等待...")
        
        try:
            # 設置本地保存路徑
            local_path = os.path.join(self.model_dir, model_name)
            
            # 使用transformers下載模型
            from transformers import AutoModel, AutoTokenizer
            
            # 顯示進度的回調函數
            download_progress = []
            
            def progress_callback(downloaded, total):
                if len(download_progress) == 0:
                    download_progress.append(tqdm(total=total, unit='iB', unit_scale=True))
                
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                self.logger.info(f"下載進度: {downloaded_mb:.2f}MB / {total_mb:.2f}MB ({(downloaded / total) * 100:.1f}%)")
                
                download_progress[0].update(downloaded - download_progress[0].n)
            
            # 下載模型，設置緩存路徑
            self.logger.info(f"正在下載模型參數...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.model_dir)
            model = AutoModel.from_pretrained(model_name, cache_dir=self.model_dir)
            
            # 保存模型到指定路徑
            self.logger.info(f"正在保存模型到本地: {local_path}")
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            
            # 計算模型大小
            size = self._get_dir_size(local_path) / (1024 * 1024)  # 轉換為MB
            
            # 更新模型索引
            self.models_index[model_name] = {
                'local_path': local_path,
                'language': language,
                'source': self.SOURCE_HUGGINGFACE,
                'download_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'size': round(size, 2)
            }
            
            self._save_models_index(self.models_index)
            
            self.logger.info(f"模型 {model_name} 下載完成，保存到: {local_path}")
            
            if len(download_progress) > 0:
                download_progress[0].close()
            
            return local_path
        
        except Exception as e:
            self.logger.error(f"下載模型時發生錯誤: {str(e)}")
            raise Exception(f"下載模型失敗: {str(e)}")
    
    def import_local_model(self, model_path: str, model_name: str = None, language: str = 'Unknown') -> str:
        """
        導入本地模型到模型管理
        
        Args:
            model_path: 本地模型路徑
            model_name: 模型名稱(默認使用目錄名)
            language: 模型語言
            
        Returns:
            target_path: 導入后的目標路徑
        """
        # 檢查源路徑是否存在
        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            error_msg = f"找不到模型目錄: {model_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 檢查是否是有效的模型目錄
        if not self._is_valid_model_dir(model_path):
            error_msg = f"無效的模型目錄: {model_path}，缺少必要的模型文件"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 如果未指定模型名稱，使用目錄名
        if model_name is None:
            model_name = os.path.basename(model_path)
        
        # 檢測模型語言
        if language == 'Unknown':
            language = self._detect_model_language(model_path)
        
        # 設置目標路徑
        target_path = os.path.join(self.model_dir, model_name)
        
        # 檢查目標路徑是否存在
        if os.path.exists(target_path):
            if model_name in self.models_index:
                self.logger.warning(f"模型 {model_name} 已存在，將被覆蓋")
            else:
                self.logger.warning(f"目標路徑已存在，將被覆蓋: {target_path}")
            
            # 刪除目標路徑
            shutil.rmtree(target_path, ignore_errors=True)
        
        # 複製模型文件
        self.logger.info(f"正在導入模型 {model_name} 從 {model_path} 到 {target_path}")
        
        try:
            # 創建目標目錄
            os.makedirs(target_path, exist_ok=True)
            
            # 複製所有文件
            for item in os.listdir(model_path):
                s = os.path.join(model_path, item)
                d = os.path.join(target_path, item)
                
                if os.path.isdir(s):
                    shutil.copytree(s, d)
                else:
                    shutil.copy2(s, d)
            
            # 計算模型大小
            size = self._get_dir_size(target_path) / (1024 * 1024)  # 轉換為MB
            
            # 更新模型索引
            self.models_index[model_name] = {
                'local_path': target_path,
                'language': language,
                'source': self.SOURCE_LOCAL,
                'import_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'size': round(size, 2)
            }
            
            self._save_models_index(self.models_index)
            
            self.logger.info(f"模型 {model_name} 導入完成")
            
            return target_path
        
        except Exception as e:
            self.logger.error(f"導入模型時發生錯誤: {str(e)}")
            raise Exception(f"導入模型失敗: {str(e)}")
    
    def delete_model(self, model_name: str) -> bool:
        """
        刪除已下載的模型
        
        Args:
            model_name: 要刪除的模型名稱
            
        Returns:
            success: 是否成功刪除
        """
        # 檢查模型是否在索引中
        if model_name not in self.models_index:
            self.logger.warning(f"模型 {model_name} 不存在，無法刪除")
            return False
        
        # 獲取模型路徑
        model_path = self.models_index[model_name]['local_path']
        
        # 檢查路徑是否存在
        if not os.path.exists(model_path):
            self.logger.warning(f"模型路徑不存在: {model_path}")
            # 從索引中刪除
            del self.models_index[model_name]
            self._save_models_index(self.models_index)
            return True
        
        # 刪除模型目錄
        try:
            self.logger.info(f"正在刪除模型 {model_name}，路徑: {model_path}")
            shutil.rmtree(model_path)
            
            # 從索引中刪除
            del self.models_index[model_name]
            self._save_models_index(self.models_index)
            
            self.logger.info(f"模型 {model_name} 已成功刪除")
            
            return True
            
        except Exception as e:
            self.logger.error(f"刪除模型時發生錯誤: {str(e)}")
            return False
    
    def get_model_path(self, model_name: str = None, language: str = 'Chinese') -> str:
        """
        獲取模型路徑，如果模型不存在則下載
        
        Args:
            model_name: 模型名稱(默認根據語言選擇)
            language: 模型語言('Chinese'或'English')
            
        Returns:
            model_path: 模型路徑
        """
        # 根據語言選擇默認模型
        if model_name is None:
            if language.lower() == 'chinese':
                model_name = self.DEFAULT_CHINESE_MODEL
            else:
                model_name = self.DEFAULT_ENGLISH_MODEL
        
        # 檢查模型是否已下載
        if model_name in self.models_index:
            return self.models_index[model_name]['local_path']
        
        # 下載模型
        self.logger.info(f"模型 {model_name} 不存在，正在下載...")
        return self.download_model(model_name, language)
    
    def cleanup(self) -> None:
        """
        清理和標記進程完成
        """
        ConsoleOutputManager.mark_process_complete(self.status_file)
        self.logger.info("BERT下載器清理完成")

# 當腳本直接執行時的示例用法
if __name__ == "__main__":
    # 初始化下載器
    downloader = BertOfflineDownloader()
    
    # 列出已下載的模型
    models = downloader.list_available_models()
    
    # 下載一個中文BERT模型
    try:
        model_path = downloader.download_model('bert-base-chinese')
        print(f"模型已下載到: {model_path}")
    except Exception as e:
        print(f"下載模型時發生錯誤: {str(e)}")
    
    # 清理並標記完成
    downloader.cleanup()