import sys
import subprocess
import logging
import os

class ConsoleOutputManager:
    """
    管理控制台輸出的工具類
    用於將處理過程輸出到獨立的控制台視窗
    """
    
    @staticmethod
    def open_console(title="處理輸出"):
        """
        打開一個獨立的控制台視窗
        
        Args:
            title: 控制台視窗標題
            
        Returns:
            process: 啟動的進程對象
        """
        if sys.platform == 'win32':
            # Windows平台使用cmd
            command = f'start "{title}" cmd /K'
            return subprocess.Popen(command, shell=True)
        elif sys.platform == 'darwin':
            # macOS平台使用Terminal
            command = ['osascript', '-e', f'tell app "Terminal" to do script ""']
            return subprocess.Popen(command)
        else:
            # Linux平台嘗試使用xterm或gnome-terminal
            try:
                return subprocess.Popen(['gnome-terminal', '--title', title, '--'])
            except FileNotFoundError:
                try:
                    return subprocess.Popen(['xterm', '-title', title, '-e', 'bash'])
                except FileNotFoundError:
                    logging.warning("無法打開獨立控制台，將使用標準輸出")
                    return None

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
    console_proc = ConsoleOutputManager.open_console("數據導入處理")
    
    # 設置日誌器
    logger = ConsoleOutputManager.setup_console_logger("import_data", "./Part02_/logs/data_import.log")
    
    # 輸出一些日誌
    logger.info("數據導入開始")
    logger.info("正在處理文件...")
    logger.warning("發現一些格式問題，但已自動修正")
    logger.info("數據導入完成")