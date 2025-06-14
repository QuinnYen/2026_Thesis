#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
錯誤處理工具模組
提供統一的錯誤訊息輸出到終端機的功能
"""

import sys
import traceback
import logging
from datetime import datetime
from typing import Optional, Any
import os

class TerminalErrorHandler:
    """
    終端機錯誤處理器
    確保所有錯誤訊息都能清楚地輸出到終端機
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        初始化錯誤處理器
        
        Args:
            log_file: 錯誤日誌檔案路徑（可選）
        """
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """設置日誌配置"""
        # 創建logger
        self.logger = logging.getLogger('TerminalErrorHandler')
        self.logger.setLevel(logging.ERROR)
        
        # 清除現有的handlers
        self.logger.handlers.clear()
        
        # 創建終端機handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR)
        
        # 設置格式
        formatter = logging.Formatter(
            '🚨 [%(asctime)s] 錯誤 - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        # 如果指定了日誌檔案，也添加檔案handler
        if self.log_file:
            try:
                # 確保日誌目錄存在
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                
                file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
                file_handler.setLevel(logging.ERROR)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️ 無法創建日誌檔案 {self.log_file}: {e}")
    
    def handle_error(self, error: Exception, 
                    context: str = "", 
                    show_traceback: bool = True,
                    exit_on_error: bool = False):
        """
        處理錯誤並輸出到終端機
        
        Args:
            error: 錯誤物件
            context: 錯誤發生的上下文訊息
            show_traceback: 是否顯示詳細追蹤信息
            exit_on_error: 是否在錯誤後退出程式
        """
        # 基本錯誤訊息
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 構建錯誤訊息
        if context:
            full_message = f"{context}: {error_type} - {error_msg}"
        else:
            full_message = f"{error_type} - {error_msg}"
        
        # 輸出到終端機
        print("\n" + "="*80)
        print(f"🚨 系統錯誤發生！")
        print("="*80)
        print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"錯誤類型: {error_type}")
        print(f"錯誤訊息: {error_msg}")
        if context:
            print(f"發生位置: {context}")
        
        # 如果需要顯示詳細追蹤信息
        if show_traceback:
            print("\n📋 詳細錯誤追蹤:")
            print("-" * 60)
            traceback.print_exc()
            print("-" * 60)
        
        print("="*80)
        
        # 記錄到日誌
        self.logger.error(full_message)
        if show_traceback:
            self.logger.error(f"詳細錯誤追蹤:\n{traceback.format_exc()}")
        
        # 如果需要退出程式
        if exit_on_error:
            print("\n💥 程式因嚴重錯誤而終止")
            sys.exit(1)
    
    def handle_warning(self, message: str, context: str = ""):
        """
        處理警告訊息
        
        Args:
            message: 警告訊息
            context: 上下文訊息
        """
        if context:
            full_message = f"{context}: {message}"
        else:
            full_message = message
        
        print(f"⚠️ 警告: {full_message}")
        
        # 記錄到日誌
        warning_logger = logging.getLogger('Warning')
        warning_logger.warning(full_message)
    
    def handle_info(self, message: str, context: str = ""):
        """
        處理信息訊息
        
        Args:
            message: 信息訊息
            context: 上下文訊息
        """
        if context:
            full_message = f"{context}: {message}"
        else:
            full_message = message
        
        print(f"ℹ️ 信息: {full_message}")

# 創建全域錯誤處理器實例
_global_error_handler = None

def get_error_handler(log_file: Optional[str] = None) -> TerminalErrorHandler:
    """
    獲取全域錯誤處理器實例
    
    Args:
        log_file: 日誌檔案路徑
        
    Returns:
        TerminalErrorHandler: 錯誤處理器實例
    """
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = TerminalErrorHandler(log_file)
    return _global_error_handler

def handle_error(error: Exception, 
                context: str = "", 
                show_traceback: bool = True,
                exit_on_error: bool = False,
                log_file: Optional[str] = None):
    """
    便利函數：處理錯誤
    
    Args:
        error: 錯誤物件
        context: 錯誤發生的上下文訊息
        show_traceback: 是否顯示詳細追蹤信息
        exit_on_error: 是否在錯誤後退出程式
        log_file: 日誌檔案路徑
    """
    handler = get_error_handler(log_file)
    handler.handle_error(error, context, show_traceback, exit_on_error)

def handle_warning(message: str, context: str = "", log_file: Optional[str] = None):
    """
    便利函數：處理警告
    
    Args:
        message: 警告訊息
        context: 上下文訊息
        log_file: 日誌檔案路徑
    """
    handler = get_error_handler(log_file)
    handler.handle_warning(message, context)

def handle_info(message: str, context: str = "", log_file: Optional[str] = None):
    """
    便利函數：處理信息
    
    Args:
        message: 信息訊息
        context: 上下文訊息
        log_file: 日誌檔案路徑
    """
    handler = get_error_handler(log_file)
    handler.handle_info(message, context)

def safe_execute(func, *args, context: str = "", 
                show_traceback: bool = True, 
                return_on_error: Any = None,
                log_file: Optional[str] = None, **kwargs):
    """
    安全執行函數，自動處理錯誤
    
    Args:
        func: 要執行的函數
        *args: 函數參數
        context: 上下文訊息
        show_traceback: 是否顯示詳細追蹤信息
        return_on_error: 錯誤時的返回值
        log_file: 日誌檔案路徑
        **kwargs: 函數關鍵字參數
        
    Returns:
        函數執行結果或錯誤時的返回值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handle_error(e, context, show_traceback, log_file=log_file)
        return return_on_error

# 裝飾器版本
def with_error_handling(context: str = "", 
                       show_traceback: bool = True, 
                       return_on_error: Any = None,
                       log_file: Optional[str] = None):
    """
    錯誤處理裝飾器
    
    Args:
        context: 上下文訊息
        show_traceback: 是否顯示詳細追蹤信息
        return_on_error: 錯誤時的返回值
        log_file: 日誌檔案路徑
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return safe_execute(func, *args, 
                              context=context or f"函數 {func.__name__}",
                              show_traceback=show_traceback,
                              return_on_error=return_on_error,
                              log_file=log_file,
                              **kwargs)
        return wrapper
    return decorator 