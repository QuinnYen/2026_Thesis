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
print(f"Python 版本: {sys.version}", f"｜當前工作目錄: {os.getcwd()}")

# 確保能夠正確引入模組，無論從哪裡執行腳本
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
print(f"模組搜尋路徑已添加: {current_dir}")

# 設定字體目錄
FONTS_DIR = os.path.join(current_dir, "resources", "fonts")
os.environ["QT_QPA_FONTDIR"] = FONTS_DIR

# 設定 Qt 平台插件
os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=freetype"  # Windows 平台使用 freetype 引擎

# 設定固定隨機種子，確保每次執行結果一致
try:
    import random
    import numpy as np
    import torch
    from sklearn.utils import check_random_state
    
    # 設定固定的隨機種子
    RANDOM_SEED = 42
    
    # 設定Python內建的random模組
    random.seed(RANDOM_SEED)
    
    # 設定NumPy的隨機種子
    np.random.seed(RANDOM_SEED)
    
    # 設定PyTorch的隨機種子
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 設定scikit-learn的隨機種子
    check_random_state(RANDOM_SEED)
    
    # 設定環境變數，確保某些庫也使用固定種子
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
    
    print(f"已設定固定隨機種子: {RANDOM_SEED}")
    print("已設定所有相關模組的隨機種子")
except ImportError as e:
    print(f"隨機種子設置錯誤: {e}")
    print("請確保已安裝必要的模組: pip install numpy torch scikit-learn")

# 導入系統所需函式庫
try:
    print("正在導入必要的模組...")
    
    # NLTK 資源檢查與下載
    try:
        import nltk
        import os
        import shutil
        import sys
        from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
        
        print("檢查 NLTK 資源...")
        # 指定一個固定的NLTK數據路徑，避免多個路徑造成混淆
        nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        
        # 優先使用此路徑，將它放在搜索路徑的最前面
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)
        print(f"NLTK資源路徑設置為: {nltk_data_path}")
        
        # 下載必要的NLTK資源
        resources = ['punkt', 'wordnet', 'stopwords']
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}")
                print(f"NLTK '{resource}' 資源已存在")
            except LookupError:
                print(f"下載NLTK '{resource}' 資源...")
                nltk.download(resource, download_dir=nltk_data_path, quiet=False)
        
        # 專門處理punkt_tab問題
        print("正在處理punkt_tab資源...")
        
        # 定義路徑
        punkt_dir = os.path.join(nltk_data_path, "tokenizers", "punkt")
        english_dir = os.path.join(punkt_dir, "english")
        punkt_tab_dir = os.path.join(nltk_data_path, "tokenizers", "punkt_tab", "english")
        
        # 確保目錄存在
        os.makedirs(english_dir, exist_ok=True)
        os.makedirs(punkt_tab_dir, exist_ok=True)
        
        # 複製英文模型文件從punkt到punkt_tab
        if os.path.isdir(english_dir) and os.listdir(english_dir):
            print(f"從{english_dir}複製文件到{punkt_tab_dir}")
            for file in os.listdir(english_dir):
                src = os.path.join(english_dir, file)
                dst = os.path.join(punkt_tab_dir, file)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"已複製文件: {file}")
        else:
            print("無法找到英文模型文件，嘗試創建所需的文件...")
            
            # 如果找不到原始文件，我們可以手動創建一個最小的pickle文件
            try:
                # 建立一個空的PunktTrainer
                trainer = PunktTrainer()
                tokenizer = PunktSentenceTokenizer(trainer.get_params())
                
                # 儲存tokenizer到punkt_tab目錄
                import pickle
                with open(os.path.join(punkt_tab_dir, "english.pickle"), "wb") as f:
                    pickle.dump(tokenizer, f)
                print("成功創建英文分詞模型文件")
            except Exception as e:
                print(f"創建分詞模型失敗: {e}")
        
        # 驗證資源是否可用
        try:
            nltk.data.find("tokenizers/punkt_tab/english")
            print("punkt_tab 資源驗證成功")
        except LookupError as e:
            print(f"警告: punkt_tab 資源驗證失敗 - {e}")
            print("系統將使用替代的分詞方法")
        
        print("NLTK 資源檢查與設置完成")
        
    except ImportError as e:
        print(f"NLTK 相關錯誤: {e}")
        print("請確保已安裝 NLTK: pip install nltk")
    
    # PyQt相關引入
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
        from PyQt5.QtGui import QPixmap, QFont, QIcon
        from PyQt5.QtCore import Qt, QTimer
        print("PyQt5.QtWidgets、QtGui、QtCore 導入成功")
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
    from utils.logger import setup_logger, get_logger
    from utils.config import Config
    print("MainWindow、Logger、Config 導入成功")
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
    
    # 設置輸出目錄 - 修正為 1_output 而非 output
    output_dir = os.path.join(app_dir, "1_output")
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

# d:/Project/2026_Thesis/Part04_/resources/logo.jpg
def show_splash_screen(app, resources_dir, width=600, height=400):
    """顯示啟動畫面
    
    Args:
        app: QApplication 實例
        resources_dir: 資源目錄路徑
        width: 啟動畫面寬度 (預設: 600)
        height: 啟動畫面高度 (預設: 400)
        
    Returns:
        QSplashScreen: 啟動畫面實例
    """
    # 尋找啟動畫面圖像
    splash_image = os.path.join(resources_dir, "logo.jpg")
    if not os.path.exists(splash_image):
        # 如果找不到指定的啟動畫面，則使用空白的啟動畫面
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.white)
    else:
        original_pixmap = QPixmap(splash_image)
        # 調整圖片大小
        pixmap = original_pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    
    splash_screen = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    
    # 設置啟動畫面的字體和顯示內容
    splash_screen.setFont(QFont('Arial', 12))
    
    # 計算螢幕中央位置
    screen_geometry = app.desktop().screenGeometry()
    x = (screen_geometry.width() - pixmap.width()) // 2
    y = (screen_geometry.height() - pixmap.height()) // 2
    
    # 移動到螢幕中央
    splash_screen.move(x, y)
    
    # 顯示啟動畫面
    splash_screen.show()
    splash_screen.showMessage("正在初始化應用程式...", Qt.AlignBottom | Qt.AlignCenter, Qt.black)
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
    
    # 設置應用程式屬性
    app.setAttribute(Qt.AA_EnableHighDpiScaling)  # 啟用高 DPI 縮放
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)     # 使用高 DPI 圖像
    
    # 設置應用程式信息
    app.setApplicationName("文本分析系統")
    app.setOrganizationName("論文研究項目")
    
    # 設置應用程式樣式
    app.setStyle("Fusion")  # 使用 Fusion 樣式
    
    # 顯示啟動畫面
    splash = show_splash_screen(app, resources_dir)
    app.processEvents()  # 確保啟動畫面顯示
    
    try:
        # 載入配置文件
        config_path = os.path.join(app_dir, "utils", "settings", "config.json")
        config = Config(config_path)
        
        # 更新啟動畫面
        splash.showMessage("正在載入模組...", 
                          Qt.AlignBottom | Qt.AlignCenter, 
                          Qt.black)
        app.processEvents()
        
        # 使用延時模擬載入過程
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
        main_window.raise_()  # 確保窗口在最前面
        
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