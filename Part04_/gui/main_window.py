"""
主窗口模組 - 實現應用程式的主要窗口和界面框架
"""
import sys
import os
import logging
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, 
    QHBoxLayout, QSplitter, QStatusBar, QMessageBox, QFileDialog,
    QAction, QToolBar
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QSettings
from PyQt5.QtGui import QIcon, QFont, QDesktopServices

# 導入GUI子模組
from gui.analysis_tab import AnalysisTab
from gui.settings_tab import SettingsTab
from gui.visualization_tab import VisualizationTab
from gui.classifier_tab import ClassifierTab

# 導入系統模組
from utils.logger import get_logger
from utils.config import Config
from utils.file_manager import FileManager

# 獲取logger
logger = get_logger("main_window")

class MainWindow(QMainWindow):
    """應用程式主窗口類"""
    
    # 定義信號
    status_message = pyqtSignal(str, int)  # 狀態欄訊息信號，參數：訊息, 顯示時間(毫秒)
    progress_updated = pyqtSignal(int, int)  # 進度更新信號，參數：當前值, 最大值
    
    def __init__(self, config_file=None):
        """初始化主窗口
        
        Args:
            config_file: 配置文件路徑，如果為None則使用默認路徑
        """
        super().__init__()

        # 初始化記錄器
        from utils.logger import get_logger
        self.logger = get_logger("main_window")
        
        # 載入配置
        self.logger.info("正在載入配置...")
        try:
            from utils.config import Config
            self.config = Config(config_file) if config_file else None
            
            # 如果配置加載失敗，創建一個默認配置
            if self.config is None:
                self.logger.warning("配置加載失敗，使用默認配置")
                self.config = self._create_default_config()
        except Exception as e:
            self.logger.error(f"載入配置文件失敗: {str(e)}")
            self.config = self._create_default_config()
        
        # 初始化文件管理器
        try:
            from utils.file_manager import FileManager
            self.file_manager = FileManager(self.config)
        except Exception as e:
            self.logger.error(f"初始化文件管理器失敗: {str(e)}")
            self.file_manager = None
        
        # 初始化設定
        self.settings = QSettings("ThesisResearch", "TextAnalysisTool")
        
        # 初始化UI
        self._init_ui()
        
        # 設置窗口屬性
        self._set_window_properties()
        
        # 連接信號和槽
        self._connect_signals_slots()
        
        self.logger.info("主窗口初始化完成")
        
    # 設置窗口屬性
    def _set_window_properties(self):
        """設置窗口屬性"""
        # 設置窗口標題
        self.setWindowTitle("文本分析系統")
        
        # 設置窗口大小
        self.resize(1200, 800)
        
        # 設置窗口圖標（如果可用）
        try:
            from PyQt5.QtGui import QIcon
            icon_path = os.path.join(self.config.get("paths", "resources_dir", ""), "icon.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            logger.warning(f"設置窗口圖標失敗: {str(e)}")
        
        # 載入窗口設置
        self._load_settings()

    def _connect_signals_slots(self):
        """連接信號和槽"""
        # 連接自定義信號
        self.status_message.connect(self.show_status_message)
        self.progress_updated.connect(self.update_progress)
        
        # 連接標籤頁的信號
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'status_message'):
            self.analysis_tab.status_message.connect(self.show_status_message)
            
        if hasattr(self, 'analysis_tab') and hasattr(self.analysis_tab, 'progress_updated'):
            self.analysis_tab.progress_updated.connect(self.update_progress)
            
        if hasattr(self, 'visualization_tab') and hasattr(self.visualization_tab, 'status_message'):
            self.visualization_tab.status_message.connect(self.show_status_message)
            
        if hasattr(self, 'classifier_tab') and hasattr(self.classifier_tab, 'status_message'):
            self.classifier_tab.status_message.connect(self.show_status_message)
            
        if hasattr(self, 'classifier_tab') and hasattr(self.classifier_tab, 'progress_updated'):
            self.classifier_tab.progress_updated.connect(self.update_progress)
            
        if hasattr(self, 'settings_tab') and hasattr(self.settings_tab, 'settings_changed'):
            self.settings_tab.settings_changed.connect(self.on_settings_changed)
    
    def _create_default_config(self):
        """創建默認配置"""
        # 獲取應用程式目錄
        app_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        
        # 創建默認配置
        default_config = {
            "paths": {
                "app_dir": app_dir,
                "output_dir": os.path.join(app_dir, "1_output"),  # 修改為 1_output 而不是 output
                "resources_dir": os.path.join(app_dir, "resources"),
                "logs_dir": os.path.join(app_dir, "1_output", "logs"),  # 修改為 1_output/logs
                "data_dir": os.path.join(app_dir, "data")
            },
            "gui": {
                "theme": "default",
                "language": "zh_TW"
            },
            "processing": {
                "num_workers": 1,
                "batch_size": 32
            }
        }
        
        # 確保目錄存在
        for path_key, path_value in default_config["paths"].items():
            if path_key.endswith("_dir"):
                os.makedirs(path_value, exist_ok=True)
        
        # 創建一個類似於Config的對象
        class SimpleConfig:
            def get(self, section, key=None, default=None):
                if section not in default_config:
                    return default
                    
                if key is None:
                    return default_config[section]
                    
                if isinstance(default_config[section], dict):
                    return default_config[section].get(key, default)
                    
                return default
            
            def get_all(self):
                """返回所有配置"""
                return default_config
        
        return SimpleConfig()

    def _init_ui(self):
        """初始化UI組件"""
        # 設置中心窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主佈局
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 創建標籤頁面管理器
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setMovable(True)
        
        # 創建各功能標籤頁
        self.analysis_tab = AnalysisTab(self.config, self.file_manager)
        self.visualization_tab = VisualizationTab(self.config, self.file_manager)
        self.classifier_tab = ClassifierTab(self.config, self.file_manager)
        self.settings_tab = SettingsTab(self.config, self.file_manager)
        
        # 將標籤頁添加到標籤頁管理器
        self.tab_widget.addTab(self.analysis_tab, "分析處理")
        self.tab_widget.addTab(self.visualization_tab, "結果視覺化")
        self.tab_widget.addTab(self.classifier_tab, "分類器")
        self.tab_widget.addTab(self.settings_tab, "設定")
        
        # 將標籤頁管理器添加到主佈局
        self.main_layout.addWidget(self.tab_widget)
        
        # 創建狀態欄
        self._create_status_bar()
        
        # 創建工具欄和選單
        self._create_toolbars()
        self._create_menus()

    def _create_status_bar(self):
        """創建狀態欄"""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 添加進度條到狀態欄
        self.progress_widget = QWidget()
        progress_layout = QHBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        from PyQt5.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMinimumWidth(100)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        progress_layout.addWidget(self.progress_bar)
        self.statusBar.addPermanentWidget(self.progress_widget)

    def _create_toolbars(self):
        """創建工具欄"""
        # 主工具欄
        self.main_toolbar = QToolBar("主工具欄")
        self.main_toolbar.setMovable(False)
        self.main_toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.TopToolBarArea, self.main_toolbar)
        
        # 添加工具欄操作
        open_action = QAction(self.style().standardIcon(QApplication.style().SP_DialogOpenButton), "開啟數據", self)
        open_action.triggered.connect(self.open_data)
        self.main_toolbar.addAction(open_action)
        
        save_action = QAction(self.style().standardIcon(QApplication.style().SP_DialogSaveButton), "保存結果", self)
        save_action.triggered.connect(self.save_results)
        self.main_toolbar.addAction(save_action)
        
        self.main_toolbar.addSeparator()
        
        analyze_action = QAction(QIcon(), "分析數據", self)
        analyze_action.triggered.connect(self.start_analysis)
        self.main_toolbar.addAction(analyze_action)
        
        visualize_action = QAction(QIcon(), "視覺化", self)
        visualize_action.triggered.connect(self.visualize_results)
        self.main_toolbar.addAction(visualize_action)
        
        self.main_toolbar.addSeparator()
        
        settings_action = QAction(self.style().standardIcon(QApplication.style().SP_FileDialogDetailedView), "設定", self)
        settings_action.triggered.connect(self.show_settings)
        self.main_toolbar.addAction(settings_action)
        
        help_action = QAction(self.style().standardIcon(QApplication.style().SP_MessageBoxQuestion), "幫助", self)
        help_action.triggered.connect(self.show_help)
        self.main_toolbar.addAction(help_action)

    def _create_menus(self):
        """創建選單"""
        # 文件選單
        file_menu = self.menuBar().addMenu("檔案")
        
        open_action = QAction("開啟數據集", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_data)
        file_menu.addAction(open_action)
        
        save_results_action = QAction("保存結果", self)
        save_results_action.setShortcut("Ctrl+S")
        save_results_action.triggered.connect(self.save_results)
        file_menu.addAction(save_results_action)
        
        export_action = QAction("導出報告", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_report)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 分析選單
        analysis_menu = self.menuBar().addMenu("分析")
        
        start_analysis_action = QAction("開始分析", self)
        start_analysis_action.triggered.connect(self.start_analysis)
        analysis_menu.addAction(start_analysis_action)
        
        stop_analysis_action = QAction("停止分析", self)
        stop_analysis_action.triggered.connect(self.stop_analysis)
        analysis_menu.addAction(stop_analysis_action)
        
        analysis_menu.addSeparator()
        
        bert_config_action = QAction("BERT設定", self)
        bert_config_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.settings_tab))
        analysis_menu.addAction(bert_config_action)
        
        lda_config_action = QAction("LDA設定", self)
        lda_config_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.settings_tab))
        analysis_menu.addAction(lda_config_action)
        
        # 視覺化選單
        viz_menu = self.menuBar().addMenu("視覺化")
        
        visualize_action = QAction("生成視覺化", self)
        visualize_action.triggered.connect(self.visualize_results)
        viz_menu.addAction(visualize_action)
        
        viz_menu.addSeparator()
        
        topic_viz_action = QAction("主題可視化", self)
        topic_viz_action.triggered.connect(lambda: self.visualization_tab.show_topic_visualization())
        viz_menu.addAction(topic_viz_action)
        
        attention_viz_action = QAction("注意力可視化", self)
        attention_viz_action.triggered.connect(lambda: self.visualization_tab.show_attention_visualization())
        viz_menu.addAction(attention_viz_action)
        
        # 工具選單
        tools_menu = self.menuBar().addMenu("工具")
        
        settings_action = QAction("設定", self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        log_action = QAction("查看日誌", self)
        log_action.triggered.connect(self.show_logs)
        tools_menu.addAction(log_action)
        
        tools_menu.addSeparator()
        
        reset_action = QAction("重置設定", self)
        reset_action.triggered.connect(self.reset_settings)
        tools_menu.addAction(reset_action)
        
        # 幫助選單
        help_menu = self.menuBar().addMenu("幫助")
        
        help_action = QAction("幫助文檔", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        
        about_action = QAction("關於", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _load_settings(self):
        """載入應用程式設置"""
        # 從QSettings載入窗口設置
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
            
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
            
        # 最後使用的標籤頁
        last_tab = self.settings.value("lastTab", 0, int)
        if 0 <= last_tab < self.tab_widget.count():
            self.tab_widget.setCurrentIndex(last_tab)
            
        # 載入最近使用的文件列表
        self._load_recent_files()

    def _save_settings(self):
        """保存應用程式設置"""
        # 保存窗口設置
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("lastTab", self.tab_widget.currentIndex())

    def _load_recent_files(self):
        """載入最近使用的文件列表"""
        # 實現最近文件列表功能
        pass

    def show_status_message(self, message, timeout=0):
        """在狀態欄顯示訊息
        
        Args:
            message: 要顯示的訊息
            timeout: 顯示時間(毫秒)，0表示持續顯示
        """
        self.statusBar.showMessage(message, timeout)

    def update_progress(self, value, maximum=100):
        """更新進度條
        
        Args:
            value: 當前值
            maximum: 最大值
        """
        if value < 0:
            # 負值表示隱藏進度條
            self.progress_bar.setVisible(False)
            return
            
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        
        # 如果達到最大值，3秒後自動隱藏進度條
        if value >= maximum:
            QTimer.singleShot(3000, lambda: self.progress_bar.setVisible(False))

    def show_message(self, title, message):
        """顯示消息對話框"""
        QMessageBox.information(self, title, message)

    def show_error(self, title, message):
        """顯示錯誤對話框"""
        QMessageBox.critical(self, title, message)

    def show_warning(self, title, message):
        """顯示警告對話框"""
        QMessageBox.warning(self, title, message)

    def show_question(self, title, message):
        """顯示問題對話框"""
        return QMessageBox.question(
            self, title, message,
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )

    def open_data(self):
        """開啟數據文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "開啟數據檔案",
            self.file_manager.data_dir,
            "數據文件 (*.csv *.txt *.json);;所有文件 (*.*)"
        )
        
        if file_path:
            self._open_file(file_path)

    def save_results(self):
        """保存分析結果"""
        # 獲取活動標籤頁
        current_tab = self.tab_widget.currentWidget()
        
        if current_tab == self.analysis_tab:
            self.analysis_tab.save_results()
        elif current_tab == self.visualization_tab:
            self.visualization_tab.save_visualization()
        else:
            self.show_message("保存", "當前頁面沒有可保存的結果")

    def export_report(self):
        """導出分析報告"""
        # 獲取分析結果並導出報告
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "導出分析報告",
            os.path.join(self.file_manager.exports_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"),
            "HTML文件 (*.html);;PDF文件 (*.pdf)"
        )
        
        if file_path:
            try:
                # 根據活動標籤頁來決定導出哪種報告
                current_tab = self.tab_widget.currentWidget()
                
                if current_tab == self.analysis_tab:
                    self.analysis_tab.export_report(file_path)
                elif current_tab == self.visualization_tab:
                    self.visualization_tab.export_report(file_path)
                else:
                    self.show_message("導出報告", "當前頁面沒有可導出的報告")
                    
                self.show_status_message(f"報告已導出: {os.path.basename(file_path)}", 5000)
            except Exception as e:
                logger.error(f"導出報告失敗: {str(e)}")
                self.show_error("導出失敗", str(e))

    def start_analysis(self):
        """開始數據分析"""
        # 切換到分析標籤頁
        self.tab_widget.setCurrentWidget(self.analysis_tab)
        # 調用分析標籤頁的分析功能
        self.analysis_tab.start_analysis()

    def stop_analysis(self):
        """停止數據分析"""
        self.analysis_tab.stop_analysis()

    def analyze_file(self, file_path):
        """分析指定文件"""
        # 切換到分析標籤頁
        self.tab_widget.setCurrentWidget(self.analysis_tab)
        # 載入並分析文件
        self.analysis_tab.load_data(file_path)
        self.analysis_tab.start_analysis()

    def visualize_results(self):
        """生成可視化結果"""
        # 切換到視覺化標籤頁
        self.tab_widget.setCurrentWidget(self.visualization_tab)
        # 調用視覺化功能
        self.visualization_tab.generate_visualizations()

    def show_settings(self):
        """顯示設定頁面"""
        self.tab_widget.setCurrentWidget(self.settings_tab)

    def show_logs(self):
        """顯示日誌文件"""
        log_dir = self.file_manager.logs_dir
        
        # 列出日誌目錄中的所有日誌文件
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        if not log_files:
            self.show_message("日誌", "沒有找到日誌文件")
            return
            
        # 如果有多個日誌文件，讓用戶選擇
        if len(log_files) == 1:
            log_file = log_files[0]
        else:
            from PyQt5.QtWidgets import QInputDialog
            log_file, ok = QInputDialog.getItem(
                self, "選擇日誌文件", "選擇要查看的日誌文件:", 
                log_files, 0, False
            )
            
            if not ok:
                return
                
        # 打開日誌文件查看器
        log_path = os.path.join(log_dir, log_file)
        
        try:
            # 嘗試使用系統默認的文本編輯器打開
            QDesktopServices.openUrl(f"file:///{log_path}")
        except:
            # 如果失敗，則使用內置的簡單查看器
            self._show_file_content(log_path, "日誌文件 - " + log_file)

    def _show_file_content(self, file_path, title):
        """顯示文件內容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            from PyQt5.QtWidgets import QDialog, QTextEdit, QVBoxLayout
            
            dialog = QDialog(self)
            dialog.setWindowTitle(title)
            dialog.resize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setFont(QFont("Courier New", 10))
            text_edit.setText(content)
            
            layout.addWidget(text_edit)
            
            dialog.exec_()
        except Exception as e:
            self.show_error("打開文件失敗", str(e))

    def reset_settings(self):
        """重置所有設定"""
        reply = self.show_question(
            "重置設定", 
            "確定要將所有設定重置為默認值嗎？\n這將會失去所有自定義配置。"
        )
        
        if reply == QMessageBox.Yes:
            # 重置配置
            self.config.reset_to_default()
            self.settings_tab.load_settings()
            
            self.show_message("設定已重置", "所有設定已重置為默認值。部分設定可能需要重啟應用程式才能生效。")

    def show_help(self):
        """顯示幫助文檔"""
        help_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "docs", "help.html"
        )
        
        if os.path.exists(help_file):
            QDesktopServices.openUrl(f"file:///{help_file}")
        else:
            self.show_message(
                "幫助文檔", 
                "幫助文檔尚未找到。\n\n"
                "本系統用於分析文本數據，提取主題並進行可視化。\n"
                "1. 在「分析處理」標籤頁可以載入數據並進行分析\n"
                "2. 在「結果視覺化」標籤頁可以檢視和導出分析結果\n"
                "3. 在「設定」標籤頁可以調整系統參數"
            )

    def show_about(self):
        """顯示關於對話框"""
        QMessageBox.about(
            self,
            "關於文本分析系統",
            f"<h3>文本分析系統</h3>"
            f"<p>版本: 1.0.0</p>"
            f"<p>構建日期: 2025年4月20日</p>"
            f"<p>本系統用於分析文本數據，提取主題並進行可視化。</p>"
            f"<p>&copy; 2025 論文研究專案</p>"
        )

    def on_settings_changed(self):
        """處理設定變更"""
        # 當設定改變時重新載入配置
        self.config.load()
        # 通知其他標籤頁設定已變更
        self.analysis_tab.on_settings_changed()
        self.visualization_tab.on_settings_changed()
        
        self.show_status_message("設定已更新", 3000)

    def closeEvent(self, event):
        """處理窗口關閉事件"""
        # 保存設定
        self._save_settings()
        
        # 確認是否有正在進行的分析任務
        if hasattr(self.analysis_tab, "is_processing") and self.analysis_tab.is_processing:
            reply = self.show_question(
                "確認退出",
                "目前有分析任務正在進行中，確定要退出嗎？"
            )
            
            if reply == QMessageBox.No:
                event.ignore()
                return
        
        # 接受關閉事件
        event.accept()


# 如果直接運行此模組
if __name__ == "__main__":
    # 初始化應用程式
    app = QApplication(sys.argv)
    
    # 設置應用程式風格
    app.setStyle("Fusion")
    
    # 創建主窗口
    window = MainWindow()
    window.show()
    
    # 運行應用程式
    sys.exit(app.exec_())