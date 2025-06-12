#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
進度橋接器 - 將後台處理進度橋接到GUI
"""

import queue
import threading
from tqdm import tqdm
import sys
import time
import logging
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)

class ProgressBridge:
    """將後台處理進度橋接到GUI的處理器"""
    
    def __init__(self, gui_queue: queue.Queue):
        """
        初始化進度橋接器
        
        Args:
            gui_queue: GUI用於接收進度更新的隊列
        """
        self.gui_queue = gui_queue
        self.active = True
        self.last_update_time = 0
        self.update_interval = 0.1  # 最小更新間隔（秒）
    
    def update(self, current: int, total: int, description: str = ""):
        """
        更新進度
        
        Args:
            current: 當前進度
            total: 總進度
            description: 進度描述
        """
        if not self.active:
            return
        
        # 限制更新頻率以避免GUI阻塞
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval and current < total:
            return
        
        self.last_update_time = current_time
        
        try:
            percentage = (current / total) * 100 if total > 0 else 0
            self.gui_queue.put(('progress', percentage))
            
            if description:
                status_msg = f"{description}: {current}/{total} ({percentage:.1f}%)"
            else:
                status_msg = f"進度: {current}/{total} ({percentage:.1f}%)"
            
            self.gui_queue.put(('status', status_msg))
            
        except Exception as e:
            logger.error(f"更新進度時發生錯誤: {str(e)}")
    
    def set_status(self, message: str):
        """
        設置狀態訊息
        
        Args:
            message: 狀態訊息
        """
        if self.active:
            try:
                self.gui_queue.put(('status', message))
            except Exception as e:
                logger.error(f"設置狀態時發生錯誤: {str(e)}")
    
    def set_phase(self, phase_name: str, current_phase: int = 0, total_phases: int = 0):
        """
        設置處理階段
        
        Args:
            phase_name: 階段名稱
            current_phase: 當前階段號
            total_phases: 總階段數
        """
        if self.active:
            try:
                if total_phases > 0:
                    phase_progress = (current_phase / total_phases) * 100
                    self.gui_queue.put(('phase_progress', phase_progress))
                    message = f"階段 {current_phase}/{total_phases}: {phase_name}"
                else:
                    message = f"🔄 {phase_name}"
                
                self.gui_queue.put(('status', message))
                
            except Exception as e:
                logger.error(f"設置階段時發生錯誤: {str(e)}")
    
    def finish(self, message: str = "處理完成"):
        """
        完成處理
        
        Args:
            message: 完成訊息
        """
        if self.active:
            try:
                self.gui_queue.put(('progress', 100))
                self.gui_queue.put(('status', f"✅ {message}"))
            except Exception as e:
                logger.error(f"設置完成狀態時發生錯誤: {str(e)}")
    
    def error(self, message: str):
        """
        報告錯誤
        
        Args:
            message: 錯誤訊息
        """
        if self.active:
            try:
                self.gui_queue.put(('error', f"❌ {message}"))
            except Exception as e:
                logger.error(f"報告錯誤時發生錯誤: {str(e)}")
    
    def deactivate(self):
        """停用進度橋接器"""
        self.active = False

class TkinterTqdm(tqdm):
    """將tqdm進度橋接到GUI的自定義類"""
    
    def __init__(self, *args, progress_bridge: Optional[ProgressBridge] = None, **kwargs):
        """
        初始化GUI版本的tqdm
        
        Args:
            progress_bridge: 進度橋接器實例
            *args, **kwargs: tqdm的原始參數
        """
        self.progress_bridge = progress_bridge
        # 禁用終端機輸出，避免與GUI衝突
        kwargs['disable'] = kwargs.get('disable', progress_bridge is not None)
        super().__init__(*args, **kwargs)
    
    def update(self, n=1):
        """更新進度"""
        super().update(n)
        
        if self.progress_bridge and self.progress_bridge.active:
            try:
                description = getattr(self, 'desc', '') or "處理中"
                self.progress_bridge.update(self.n, self.total, description)
            except Exception as e:
                logger.error(f"TkinterTqdm更新進度時發生錯誤: {str(e)}")
    
    def close(self):
        """關閉進度條"""
        super().close()
        if self.progress_bridge and self.progress_bridge.active:
            description = getattr(self, 'desc', '') or "處理"
            self.progress_bridge.finish(f"{description}完成")

class ProgressCallback:
    """簡單的進度回調包裝器"""
    
    def __init__(self, progress_bridge: ProgressBridge):
        self.progress_bridge = progress_bridge
    
    def __call__(self, message_type: str, data: Any):
        """
        處理進度回調
        
        Args:
            message_type: 訊息類型 ('progress', 'status', 'error' 等)
            data: 訊息資料
        """
        if not self.progress_bridge.active:
            return
        
        try:
            if message_type == 'progress':
                # data 可能是百分比或 (current, total) 元組
                if isinstance(data, (list, tuple)) and len(data) == 2:
                    current, total = data
                    self.progress_bridge.update(current, total)
                else:
                    # 直接是百分比
                    self.progress_bridge.gui_queue.put(('progress', data))
            
            elif message_type == 'status':
                self.progress_bridge.set_status(str(data))
            
            elif message_type == 'error':
                self.progress_bridge.error(str(data))
            
            elif message_type == 'phase':
                if isinstance(data, dict):
                    self.progress_bridge.set_phase(**data)
                else:
                    self.progress_bridge.set_phase(str(data))
            
            else:
                # 其他類型的訊息直接傳遞
                self.progress_bridge.gui_queue.put((message_type, data))
                
        except Exception as e:
            logger.error(f"處理進度回調時發生錯誤: {str(e)}")

def create_progress_callback(gui_queue: queue.Queue) -> tuple[ProgressBridge, ProgressCallback]:
    """
    創建進度橋接器和回調函數
    
    Args:
        gui_queue: GUI隊列
        
    Returns:
        tuple: (ProgressBridge實例, ProgressCallback實例)
    """
    bridge = ProgressBridge(gui_queue)
    callback = ProgressCallback(bridge)
    return bridge, callback

# 便利函數
def replace_tqdm_in_module(module, progress_bridge: ProgressBridge):
    """
    在指定模組中將tqdm替換為TkinterTqdm
    
    Args:
        module: 要替換的模組
        progress_bridge: 進度橋接器
    """
    if hasattr(module, 'tqdm'):
        def tkinter_tqdm(*args, **kwargs):
            kwargs['progress_bridge'] = progress_bridge
            return TkinterTqdm(*args, **kwargs)
        
        module.tqdm = tkinter_tqdm
        logger.info(f"已在模組 {module.__name__} 中替換tqdm為TkinterTqdm") 