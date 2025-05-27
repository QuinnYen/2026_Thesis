#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管理執行目錄的模組
"""

import os
import logging
from datetime import datetime
from typing import Optional

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
        if os.path.basename(self.base_dir) == "output":
            self.output_base = self.base_dir
        else:
            self.output_base = os.path.join(self.base_dir, "output")
        
        os.makedirs(self.output_base, exist_ok=True)
    
    def get_run_dir(self) -> str:
        """
        獲取當前執行目錄，如果不存在則創建新的
        
        Returns:
            str: 當前執行目錄的路徑
        """
        # 如果已經有當前執行目錄，直接返回
        if self._current_run_dir is not None:
            return self._current_run_dir
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(self.output_base, f"run_{timestamp}")
        
        try:
            # 創建run目錄及其子目錄
            os.makedirs(run_dir, exist_ok=True)
            os.makedirs(os.path.join(run_dir, "01_preprocessing"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "02_bert_encoding"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "03_attention_testing"), exist_ok=True)
            os.makedirs(os.path.join(run_dir, "04_analysis"), exist_ok=True)
            
            logger.info(f"已創建新的執行目錄：{run_dir}")
            
            # 保存當前執行目錄
            self._current_run_dir = run_dir
            return run_dir
            
        except Exception as e:
            logger.error(f"創建執行目錄時發生錯誤：{str(e)}")
            raise
    
    def get_preprocessing_dir(self) -> str:
        """獲取預處理目錄"""
        return os.path.join(self.get_run_dir(), "01_preprocessing")
    
    def get_bert_encoding_dir(self) -> str:
        """獲取BERT編碼目錄"""
        return os.path.join(self.get_run_dir(), "02_bert_encoding")
    
    def get_attention_testing_dir(self) -> str:
        """獲取注意力測試目錄"""
        return os.path.join(self.get_run_dir(), "03_attention_testing")
    
    def get_analysis_dir(self) -> str:
        """獲取分析目錄"""
        return os.path.join(self.get_run_dir(), "04_analysis")
    
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