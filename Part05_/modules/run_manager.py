#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管理執行目錄的模組
"""

import os
import json
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
        self.run_info_file = os.path.join(base_dir, "output", "last_run_info.json")
        
    def get_run_dir(self) -> str:
        """
        獲取當前執行目錄，如果不存在則創建新的
        """
        current_run_dir = self._load_last_run_dir()
        if current_run_dir is None:
            current_run_dir = self._create_new_run_dir()
            self._save_run_dir(current_run_dir)
        return current_run_dir
    
    def _load_last_run_dir(self) -> Optional[str]:
        """
        讀取上次執行的目錄
        """
        if os.path.exists(self.run_info_file):
            try:
                with open(self.run_info_file, 'r') as f:
                    info = json.load(f)
                    return info.get('last_run_dir')
            except Exception as e:
                logger.warning(f"讀取last_run_info.json失敗: {e}")
        return None
    
    def _save_run_dir(self, run_dir: str) -> None:
        """
        保存執行目錄
        """
        os.makedirs(os.path.dirname(self.run_info_file), exist_ok=True)
        try:
            with open(self.run_info_file, 'w') as f:
                json.dump({'last_run_dir': run_dir}, f)
        except Exception as e:
            logger.warning(f"保存last_run_info.json失敗: {e}")
    
    def _create_new_run_dir(self) -> str:
        """
        創建新的執行目錄
        """
        run_dir = os.path.join(self.base_dir, "output", 
                              f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def clear_last_run(self) -> None:
        """
        清除上次執行的記錄
        """
        if os.path.exists(self.run_info_file):
            try:
                os.remove(self.run_info_file)
            except Exception as e:
                logger.warning(f"清除last_run_info.json失敗: {e}") 