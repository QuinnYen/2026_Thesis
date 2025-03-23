"""
用於數據處理的工具函數集合
"""

import sys
import subprocess
import logging
import os
import tempfile
import threading
import time

class ConsoleOutputManager:
    """
    管理控制台輸出的工具類
    用於將處理過程輸出到獨立的控制台視窗
    """
    
    @staticmethod
    def open_console(title="處理輸出", log_to_file=True, clear_previous=True, auto_close=True, auto_close_delay=3):
        """
        打開一個獨立的控制台視窗並設置日誌檔案
        
        Args:
            title: 控制台視窗標題
            log_to_file: 是否同時將輸出寫入日誌檔案
            clear_previous: 是否清除之前的日誌內容
            auto_close: 是否在完成後自動關閉控制台
            auto_close_delay: 自動關閉前的等待時間（秒）
            
        Returns:
            log_file: 日誌檔案路徑（如果 log_to_file=True）
        """
        # 創建臨時日誌檔案
        log_dir = "./Part02_/logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"{title.replace(' ', '_').lower()}.log")
        
        # 如果需要清除之前的日誌
        if clear_previous and os.path.exists(log_file):
            # 清空日誌檔案內容但保留檔案
            open(log_file, 'w').close()
        
        # 存儲控制台狀態的文件
        status_file = os.path.join(log_dir, f"{title.replace(' ', '_').lower()}_status.txt")
        # 初始化狀態為"運行中"
        with open(status_file, 'w') as f:
            f.write("running")
        
        # 不同平台的處理方式
        if sys.platform == 'win32':
            # Windows平台 - 使用單獨的批處理檔案來顯示日誌
            bat_file = os.path.join(tempfile.gettempdir(), f"{title.replace(' ', '_').lower()}.bat")
            with open(bat_file, 'w') as f:
                f.write(f'@echo off\n')
                f.write(f'title {title}\n')
                f.write(f'echo 正在處理，請等待...\n')
                f.write(f'echo 日誌將同時保存到: {log_file}\n')
                f.write(f'echo.\n')
                
                if auto_close:
                    # 添加檢查狀態文件的循環
                    f.write(f':loop\n')
                    f.write(f'cls\n')
                    f.write(f'type "{log_file}" 2>nul\n')
                    f.write(f'findstr "complete" "{status_file}" >nul\n')
                    f.write(f'if %errorlevel% equ 0 (\n')
                    f.write(f'    echo.\n')
                    f.write(f'    echo 處理完成！窗口將在 {auto_close_delay} 秒後自動關閉...\n')
                    f.write(f'    timeout /t {auto_close_delay} /nobreak >nul\n')
                    f.write(f'    exit\n')
                    f.write(f')\n')
                    f.write(f'timeout /t 2 /nobreak >nul\n')
                    f.write(f'goto loop\n')
                else:
                    # 原來的無限循環
                    f.write(f':loop\n')
                    f.write(f'cls\n')
                    f.write(f'type "{log_file}" 2>nul\n')
                    f.write(f'timeout /t 2 /nobreak >nul\n')
                    f.write(f'goto loop\n')
            
            # 啟動批處理文件
            subprocess.Popen(f'start "" "{bat_file}"', shell=True)
        
        return log_file, status_file

    @staticmethod
    def mark_process_complete(status_file):
        """
        標記處理過程已完成，用於觸發自動關閉
        
        Args:
            status_file: 狀態文件路徑
        """
        try:
            with open(status_file, 'w') as f:
                f.write("complete")
        except Exception as e:
            print(f"無法更新狀態文件: {str(e)}")

    @staticmethod
    def setup_console_logger(name, log_file=None):
        """
        設置將日誌輸出到控制台和文件(可選)的日誌器
        
        Args:
            name: 日誌器名稱
            log_file: 日誌文件路徑(可選)
            
        Returns:
            logger: 配置好的日誌器
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # 清除現有處理器
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # 文件處理器(如果提供了文件路徑)
        if log_file:
            # 確保日誌目錄存在
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger

# 使用示例
if __name__ == "__main__":
    # 打開獨立控制台
    log_file, status_file = ConsoleOutputManager.open_console("數據導入處理", auto_close=True)
    
    # 設置日誌器
    logger = ConsoleOutputManager.setup_console_logger("import_data", log_file)
    
    # 輸出一些日誌
    logger.info("數據導入開始")
    logger.info("正在處理文件...")
    logger.warning("發現一些格式問題，但已自動修正")
    
    # 模擬處理過程
    time.sleep(5)
    
    logger.info("數據導入完成")
    
    # 標記處理完成
    ConsoleOutputManager.mark_process_complete(status_file)