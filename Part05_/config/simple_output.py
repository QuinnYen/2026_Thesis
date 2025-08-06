#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超簡化輸出模組 - 提供最精簡的終端機輸出
"""

import contextlib
import io
import sys
import os

# 全域開關
SILENT_MODE = False
MINIMAL_OUTPUT = True

def enable_silent_mode():
    """啟用安靜模式，幾乎無輸出"""
    global SILENT_MODE
    SILENT_MODE = True

def disable_silent_mode():
    """關閉安靜模式"""
    global SILENT_MODE
    SILENT_MODE = False

def enable_minimal_output():
    """啟用最小輸出模式"""
    global MINIMAL_OUTPUT
    MINIMAL_OUTPUT = True

def disable_minimal_output():
    """關閉最小輸出模式"""
    global MINIMAL_OUTPUT
    MINIMAL_OUTPUT = False

def simple_print(msg: str, force: bool = False):
    """簡化的打印函數，可以被控制"""
    if force or (not SILENT_MODE and MINIMAL_OUTPUT):
        print(msg)

def simple_step(msg: str):
    """簡化的步驟輸出"""
    if not SILENT_MODE:
        print(f"🔄 {msg}")

def simple_result(msg: str, is_error: bool = False):
    """簡化的結果輸出"""
    if not SILENT_MODE or is_error:
        prefix = "❌" if is_error else "✅"
        print(f"{prefix} {msg}")

@contextlib.contextmanager
def suppress_output():
    """臨時抑制所有輸出的上下文管理器"""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# 替換內建的print函數（在需要時）
def override_print():
    """覆蓋內建的print函數"""
    import builtins
    original_print = builtins.print
    
    def controlled_print(*args, **kwargs):
        if not SILENT_MODE:
            original_print(*args, **kwargs)
    
    builtins.print = controlled_print
    return original_print

def restore_print(original_print):
    """恢復原始的print函數"""
    import builtins
    builtins.print = original_print