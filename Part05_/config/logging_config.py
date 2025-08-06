#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日誌配置模組 - 統一管理系統日誌輸出
"""

import logging

# 日誌等級配置
LOGGING_LEVELS = {
    'SILENT': logging.CRITICAL,  # 幾乎無輸出
    'QUIET': logging.ERROR,      # 只顯示錯誤
    'NORMAL': logging.WARNING,   # 顯示警告和錯誤
    'VERBOSE': logging.INFO,     # 顯示所有信息
    'DEBUG': logging.DEBUG       # 顯示調試信息
}

# 預設使用QUIET模式（最簡化輸出）
DEFAULT_LEVEL = 'QUIET'

def setup_logging(level: str = DEFAULT_LEVEL, include_timestamp: bool = False):
    """
    設置日誌配置
    
    Args:
        level: 日誌等級 ('QUIET', 'NORMAL', 'VERBOSE', 'DEBUG')
        include_timestamp: 是否包含時間戳
    """
    log_level = LOGGING_LEVELS.get(level, LOGGING_LEVELS[DEFAULT_LEVEL])
    
    if include_timestamp:
        format_str = '%(asctime)s - %(levelname)s: %(message)s'
    else:
        format_str = '%(levelname)s: %(message)s'
    
    logging.basicConfig(
        level=log_level,
        format=format_str,
        force=True  # 覆蓋現有配置
    )

def get_logger(name: str) -> logging.Logger:
    """獲取指定名稱的logger"""
    return logging.getLogger(name)

# 進度輸出函數
def print_step(step_msg: str, is_substep: bool = False):
    """簡化的步驟輸出"""
    prefix = "   •" if is_substep else "🔄"
    print(f"{prefix} {step_msg}")

def print_result(result_msg: str, is_error: bool = False):
    """簡化的結果輸出"""
    prefix = "❌" if is_error else "✅"
    print(f"{prefix} {result_msg}")

def print_info(info_msg: str):
    """簡化的信息輸出"""
    print(f"ℹ️  {info_msg}")

def print_warning(warning_msg: str):
    """簡化的警告輸出"""
    print(f"⚠️  {warning_msg}")