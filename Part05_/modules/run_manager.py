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
    
    def get_run_dir(self) -> str:
        """
        獲取當前執行目錄，如果不存在則創建新的
        
        Returns:
            str: 當前執行目錄的路徑
        """
        # 如果已經有當前執行目錄，直接返回
        if self._current_run_dir is not None:
            return self._current_run_dir
        
        # 如果 base_dir 已經是 output，就不要再加一層
        if os.path.basename(self.base_dir) == "output":
            output_base = self.base_dir
        else:
            output_base = os.path.join(self.base_dir, "output")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(output_base, f"run_{timestamp}")
        
        try:
            # 確保目錄存在
            os.makedirs(run_dir, exist_ok=True)
            logger.info(f"已創建新的執行目錄：{run_dir}")
            
            # 保存當前執行目錄
            self._current_run_dir = run_dir
            return run_dir
            
        except Exception as e:
            logger.error(f"創建執行目錄時發生錯誤：{str(e)}")
            raise 