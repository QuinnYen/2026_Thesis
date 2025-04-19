#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本分析系統 - 主程式入口點
用於啟動文本分析系統應用程式
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# 添加調試訊息
print("應用程式啟動中...")
print(f"Python 版本: {sys.version}")
print(f"當前工作目錄: {os.getcwd()}")

# 確保能夠正確引入模組，無論從哪裡執行腳本
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
print(f"模組搜尋路徑已添加: {current_dir}")

# 導入系統所需函式庫
try:
    print("正在導入必要的模組...")
    
    # NLTK 資源檢查與下載
    try:
        import nltk
        import os
        
        print("檢查 NLTK 資源...")
        nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        
        # 添加自定義 NLTK 數據路徑
        nltk.data.path.append(nltk_data_path)
        
        # 下載必要的 NLTK 資源
        resources = ['punkt', 'wordnet', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
                print(f"NLTK '{resource}' 資源已存在")
            except LookupError:
                print(f"下載 NLTK '{resource}' 資源...")
                nltk.download(resource, download_dir=nltk_data_path)
        
        # 特殊處理 punkt_tab 資源 - 通常不需要直接下載，而是自動作為 punkt 的一部分
        # 創建所需目錄結構
        punkt_tab_dir = os.path.join(nltk_data_path, "tokenizers", "punkt", "english")
        os.makedirs(punkt_tab_dir, exist_ok=True)
        
        # 檢查 punkt_tab 是否可用，如果不可用，使用 punkt 作為替代方案
        try:
            nltk.data.find("tokenizers/punkt_tab/english")
            print("NLTK punkt_tab 資源已存在")
        except LookupError:
            print("配置 punkt_tab 替代方案...")
            # 使用 punkt 作為替代
            punkt_path = os.path.join(nltk_data_path, "tokenizers", "punkt")
            if os.path.exists(punkt_path):
                print("已配置 punkt 作為 punkt_tab 的替代")
        
        print("NLTK 資源檢查完成")
        
    except ImportError as e:
        print(f"NLTK 相關錯誤: {e}")
        print("請確保已安裝 NLTK: pip install nltk")
    
    # PyQt相關引入
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
        print("PyQt5.QtWidgets 導入成功")
        from PyQt5.QtGui import QPixmap, QFont, QIcon
        print("PyQt5.QtGui 導入成功")
        from PyQt5.QtCore import Qt, QTimer
        print("PyQt5.QtCore 導入成功")
    except ImportError as e:
        print(f"PyQt5 導入錯誤: {e}")
        print("這可能是由於以下原因造成:")
        print("1. PyQt5 未正確安裝")
        print("2. 缺少必要的 Visual C++ 運行庫")
        print("3. 系統 PATH 環境變數未包含 PyQt5 DLL 檔案的位置")
        print("\n請嘗試以下解決方案:")
        print("1. 重新安裝 PyQt5: pip uninstall PyQt5 && pip install PyQt5")
        print("2. 安裝 Microsoft Visual C++ Redistributable 2015-2019")
        print("3. 檢查 Python 和 PyQt5 是否為相同架構(32位/64位)")
        sys.exit(1)
    
    # 導入主窗口類
    print("正在導入應用程式模組...")
    from gui.main_window import MainWindow
    print("MainWindow 導入成功")
    from utils.logger import setup_logger, get_logger
    print("Logger 導入成功")
    from utils.config import Config
    print("Config 導入成功")
except ImportError as e:
    # 處理缺少必要套件的情況
    print(f"錯誤：缺少必要的套件：{e}")
    print("請確保已安裝所有必要的套件。可以通過運行以下命令安裝：")
    print("pip install -r requirements.txt")
    sys.exit(1)


def configure_app():
    """配置應用程式環境"""
    # 設置應用程式路徑
    app_dir = os.path.abspath(os.path.dirname(__file__))
    
    # 設置輸出目錄
    output_dir = os.path.join(app_dir, "output")
    logs_dir = os.path.join(output_dir, "logs")
    resources_dir = os.path.join(app_dir, "resources")
    
    # 創建必要的目錄
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 配置日誌系統
    log_file = os.path.join(logs_dir, "app.log")
    setup_logger(log_file)
    
    return app_dir, resources_dir


def show_error_dialog(exception_type, exception_value, exception_traceback):
    """顯示未捕獲異常的錯誤對話框"""
    # 格式化異常信息
    error_text = "\n".join(traceback.format_exception(exception_type, exception_value, exception_traceback))
    
    # 記錄錯誤到日誌
    logger = logging.getLogger(__name__)
    logger.critical("未捕獲的異常：\n%s", error_text)
    
    # 創建錯誤對話框
    error_dialog = QMessageBox()
    error_dialog.setIcon(QMessageBox.Critical)
    error_dialog.setText("應用程式發生錯誤")
    error_dialog.setInformativeText("應用程式遇到了意外錯誤。詳細信息已記錄到日誌檔案。")
    error_dialog.setDetailedText(error_text)
    error_dialog.setWindowTitle("錯誤")
    error_dialog.setStandardButtons(QMessageBox.Ok)
    
    # 顯示錯誤對話框
    error_dialog.exec_()


def show_splash_screen(app, resources_dir):
    """顯示啟動畫面"""
    # 尋找啟動畫面圖像
    splash_image = os.path.join(resources_dir, "splash.png")
    if not os.path.exists(splash_image):
        # 如果找不到指定的啟動畫面，則使用空白的啟動畫面
        pixmap = QPixmap(600, 400)
        pixmap.fill(Qt.white)
    else:
        pixmap = QPixmap(splash_image)
    
    splash_screen = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    
    # 設置啟動畫面的字體和顯示內容
    splash_screen.setFont(QFont('Arial', 12))
    
    # 顯示啟動畫面
    splash_screen.show()
    splash_screen.showMessage("正在初始化應用程式...", 
                            Qt.AlignBottom | Qt.AlignCenter, 
                            Qt.black)
    app.processEvents()
    
    return splash_screen


def main():
    """應用程式主函數"""
    # 配置應用程式環境
    app_dir, resources_dir = configure_app()
    
    # 獲取日誌記錄器
    logger = get_logger("main")
    logger.info("應用程式啟動")
    
    # 創建 Qt 應用程式
    app = QApplication(sys.argv)
    app.setApplicationName("文本分析系統")
    app.setOrganizationName("論文研究項目")
    
    # 設置應用程式圖示
    app_icon = os.path.join(resources_dir, "icon.png")
    if os.path.exists(app_icon):
        app.setWindowIcon(QIcon(app_icon))
    
    # 設置應用程式樣式
    app.setStyle("Fusion")
    
    # 配置全域異常處理
    sys.excepthook = show_error_dialog
    
    # 顯示啟動畫面
    splash = show_splash_screen(app, resources_dir)
    
    try:
        # 載入配置文件
        config_path = os.path.join(app_dir, "utils", "settings", "config.json")
        config = Config(config_path)
        
        # 更新啟動畫面
        splash.showMessage("正在載入模組...", 
                          Qt.AlignBottom | Qt.AlignCenter, 
                          Qt.black)
        app.processEvents()
        
        # 使用延時模擬載入過程（在實際應用中可以根據需要移除）
        QTimer.singleShot(1000, lambda: None)
        
        # 更新啟動畫面
        splash.showMessage("正在初始化界面...", 
                          Qt.AlignBottom | Qt.AlignCenter, 
                          Qt.black)
        app.processEvents()
        
        # 創建主窗口
        main_window = MainWindow(config_file=config_path)
        
        # 窗口準備完畢，隱藏啟動畫面並顯示主窗口
        splash.finish(main_window)
        main_window.show()
        
        # 記錄啟動完成
        logger.info("應用程式界面初始化完成")
        
    except Exception as e:
        # 捕獲初始化過程中的異常
        logger.critical(f"應用程式初始化失敗: {str(e)}", exc_info=True)
        
        # 顯示錯誤對話框
        splash.hide()  # 隱藏啟動畫面
        QMessageBox.critical(
            None,
            "初始化失敗",
            f"應用程式初始化過程中發生錯誤:\n{str(e)}\n\n請檢查日誌文件獲取詳細信息。"
        )
        return 1
    
    # 執行應用程式主迴圈
    exit_code = app.exec_()
    
    # 記錄應用程式結束
    logger.info(f"應用程式結束，退出碼：{exit_code}")
    
    return exit_code


# 如果直接執行此腳本，則啟動應用程式
if __name__ == "__main__":
    sys.exit(main())