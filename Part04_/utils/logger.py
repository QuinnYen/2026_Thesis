"""
日誌管理模組 - 負責系統日誌的配置和管理
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time

class Logger:
    """日誌管理類"""
    
    # 日誌級別映射
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    # 默認日誌格式
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    def __init__(self, name="cross_domain_sentiment", log_dir="./Part04_/0_output/logs", level="INFO"):
        """初始化日誌管理器
        
        Args:
            name: 日誌名稱
            log_dir: 日誌目錄
            level: 日誌級別，可以是'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        self.name = name
        self.log_dir = log_dir
        self.level = self.LEVELS.get(level.upper(), logging.INFO)
        
        # 創建日誌目錄
        os.makedirs(log_dir, exist_ok=True)
        
        # 獲取根日誌器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.propagate = False
        
        # 清除已有的處理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 創建默認的控制台和文件處理器
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """設置默認的日誌處理器"""
        # 控制台處理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_formatter = logging.Formatter(self.DEFAULT_FORMAT)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 普通日誌文件處理器（按日期）
        log_file = os.path.join(self.log_dir, f"{self.name}_{time.strftime('%Y%m%d')}.log")
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        file_formatter = logging.Formatter(self.DEFAULT_FORMAT)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 錯誤日誌文件處理器
        error_log_file = os.path.join(self.log_dir, f"{self.name}_error_{time.strftime('%Y%m%d')}.log")
        error_file_handler = RotatingFileHandler(
            error_log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(self.DEFAULT_FORMAT)
        error_file_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_file_handler)
    
    def set_level(self, level):
        """設置日誌級別
        
        Args:
            level: 日誌級別，可以是'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
        """
        level_value = self.LEVELS.get(level.upper(), logging.INFO)
        self.level = level_value
        self.logger.setLevel(level_value)
        
        # 更新處理器級別
        for handler in self.logger.handlers:
            # 對於錯誤日誌處理器，保持ERROR級別
            if isinstance(handler, RotatingFileHandler) and handler.baseFilename.endswith('_error.log'):
                continue
            handler.setLevel(level_value)
    
    def add_file_handler(self, filename, level="INFO"):
        """添加文件處理器
        
        Args:
            filename: 日誌文件名
            level: 日誌級別
        """
        level_value = self.LEVELS.get(level.upper(), logging.INFO)
        log_file = os.path.join(self.log_dir, filename)
        
        handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setLevel(level_value)
        formatter = logging.Formatter(self.DEFAULT_FORMAT)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        return handler
    
    def get_logger(self):
        """獲取日誌器實例"""
        return self.logger
    
    def create_child_logger(self, name):
        """創建子日誌器
        
        Args:
            name: 子日誌器名稱
            
        Returns:
            logging.Logger: 子日誌器實例
        """
        logger = logging.getLogger(f"{self.name}.{name}")
        logger.setLevel(self.level)
        # 子日誌器不添加處理器，使用父日誌器的處理器
        return logger

# 創建一個默認的全局日誌管理器
default_logger = Logger()

# 便於直接使用的函數
def get_logger(name=None):
    """獲取日誌器
    
    Args:
        name: 日誌器名稱，如果為None則返回根日誌器
        
    Returns:
        logging.Logger: 日誌器實例
    """
    if name is None:
        return default_logger.get_logger()
    return default_logger.create_child_logger(name)

def set_level(level):
    """設置全局日誌級別
    
    Args:
        level: 日誌級別
    """
    default_logger.set_level(level)

def setup_logger(name="cross_domain_sentiment", log_dir="./Part04_/0_output/logs", level="INFO"):
    """設置全局日誌器
    
    Args:
        name: 日誌名稱
        log_dir: 日誌目錄，默認為Part04_資料夾下的0_output/logs目錄
        level: 日誌級別
        
    Returns:
        Logger: 日誌管理器實例
    """
    # 獲取Part04_目錄的絕對路徑
    part04_dir = None
    
    # 嘗試找到Part04_目錄
    current_dir = os.path.abspath(os.curdir)
    if os.path.basename(current_dir) == 'Part04_':
        part04_dir = current_dir
    elif 'Part04_' in current_dir:
        # 如果當前在Part04_的子目錄中
        part04_dir = os.path.abspath(os.path.join(current_dir, '..'))
    else:
        # 嘗試查找腳本的位置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(script_dir) == 'Part04_':
            part04_dir = script_dir
        elif 'Part04_' in script_dir:
            # 假設我們在utils子目錄下
            part04_dir = os.path.abspath(os.path.join(script_dir, '..'))
    
    # 如果找到Part04_目錄，則使用它
    if part04_dir and os.path.isdir(part04_dir):
        # 確保使用絕對路徑
        log_dir = os.path.join(part04_dir, '0_output/logs')
    
    # 創建新的日誌管理器
    global default_logger
    default_logger = Logger(name, log_dir, level)
    
    return default_logger