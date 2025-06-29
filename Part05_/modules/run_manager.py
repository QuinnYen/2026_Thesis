#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管理執行目錄的模組
"""

import os
import logging
from datetime import datetime
from typing import Optional
import sys

# 添加父目錄到路徑以導入config模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import get_path_config

logger = logging.getLogger(__name__)

class RunManager:
    def __init__(self, base_dir: str):
        """
        初始化RunManager
        
        Args:
            base_dir: 基礎目錄路徑
        """
        self.base_dir = base_dir
        self._current_run_dir = None
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """確保輸出目錄存在"""
        # 直接使用傳入的base_dir作為輸出目錄，不再自動添加output子目錄
        self.output_base = self.base_dir
        os.makedirs(self.output_base, exist_ok=True)
    
    def get_run_dir(self, encoder_type: str = 'bert') -> str:
        """
        獲取當前執行目錄，如果不存在則創建新的
        
        Args:
            encoder_type: 編碼器類型，用於創建編碼器特定目錄
        
        Returns:
            str: 當前執行目錄的路徑
        """
        # 如果已經有當前執行目錄，直接返回
        if self._current_run_dir is not None:
            return self._current_run_dir
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(self.output_base, f"run_{timestamp}")
        
        try:
            # 創建run目錄及其子目錄 - 使用配置的目錄名稱
            path_config = get_path_config()
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, path_config.get_subdirectory_name("preprocessing")), exist_ok=True)
            
            # 創建統一的編碼目錄
            encoding_dir_name = path_config.get_subdirectory_name("encoding")
            os.makedirs(os.path.join(run_dir, encoding_dir_name), exist_ok=True)
            
            os.makedirs(os.path.join(run_dir, path_config.get_subdirectory_name("attention_testing")), exist_ok=True)
            os.makedirs(os.path.join(run_dir, path_config.get_subdirectory_name("analysis")), exist_ok=True)
            
            logger.info(f"已創建新的執行目錄：{run_dir}")
            logger.info(f"已創建統一編碼目錄：{encoding_dir_name}")
            
            # 保存當前執行目錄
            self._current_run_dir = run_dir
            return run_dir
            
        except Exception as e:
            logger.error(f"創建執行目錄時發生錯誤：{str(e)}")
            raise
    
    def get_preprocessing_dir(self) -> str:
        """獲取預處理目錄"""
        path_config = get_path_config()
        return os.path.join(self.get_run_dir(), path_config.get_subdirectory_name("preprocessing"))
    
    def get_bert_encoding_dir(self) -> str:
        """獲取BERT編碼目錄"""
        path_config = get_path_config()
        return os.path.join(self.get_run_dir(), path_config.get_subdirectory_name("bert_encoding"))
    
    def get_encoding_dir(self, encoder_type: str = 'bert') -> str:
        """獲取統一的編碼目錄"""
        path_config = get_path_config()
        return os.path.join(self.get_run_dir(encoder_type), path_config.get_subdirectory_name("encoding"))
    
    def get_attention_testing_dir(self) -> str:
        """獲取注意力測試目錄"""
        path_config = get_path_config()
        return os.path.join(self.get_run_dir(), path_config.get_subdirectory_name("attention_testing"))
    
    def get_analysis_dir(self) -> str:
        """獲取分析目錄"""
        path_config = get_path_config()
        return os.path.join(self.get_run_dir(), path_config.get_subdirectory_name("analysis"))
    
    def clear_current_run(self):
        """清除當前執行目錄"""
        if self._current_run_dir and os.path.exists(self._current_run_dir):
            try:
                import shutil
                shutil.rmtree(self._current_run_dir)
                logger.info(f"已清除執行目錄：{self._current_run_dir}")
            except Exception as e:
                logger.error(f"清除執行目錄時發生錯誤：{str(e)}")
                raise
        self._current_run_dir = None 