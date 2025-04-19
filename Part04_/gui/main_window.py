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
    QAction, QToolBar, QDockWidget, QTreeView, QMenu
)
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal, QSettings
from PyQt5.QtGui import QIcon, QFont, QDesktopServices

# 導入GUI子模組
from gui.analysis_tab import AnalysisTab
from gui.settings_tab import SettingsTab
from gui.visualization_tab import VisualizationTab

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
        
        # 初始化系統組件
        self.config = Config(config_file or "config.json")
        self.file_manager = FileManager(config=self.config.get("paths"))
        self.settings = QSettings("ThesisResearch", "TextAnalysisTool")
        
        # 設置窗口屬性
        self.setWindowTitle("文本分析系統")
        self.setMinimumSize(1024, 768)
        
        # 實例化UI組件
        self._init_ui()
        self._setup_signals()
        self._load_settings()
        
        # 初始化完畢後顯示歡迎信息
        self.show_status_message("系統已準備就緒", 3000)
        logger.info("主窗口初始化完成")

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
        self.settings_tab = SettingsTab(self.config, self.file_manager)
        
        # 將標籤頁添加到標籤頁管理器
        self.tab_widget.addTab(self.analysis_tab, "分析處理")
        self.tab_widget.addTab(self.visualization_tab, "結果視覺化")
        self.tab_widget.addTab(self.settings_tab, "設定")
        
        # 將標籤頁管理器添加到主佈局
        self.main_layout.addWidget(self.tab_widget)
        
        # 創建狀態欄
        self._create_status_bar()
        
        # 創建工具欄和選單
        self._create_toolbars()
        self._create_menus()
        
        # 創建側邊欄（檔案瀏覽器）
        self._create_sidebar()

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

    def _create_sidebar(self):
        """創建側邊欄"""
        # 文件瀏覽器側邊欄
        self.file_dock = QDockWidget("檔案瀏覽", self)
        self.file_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # 樹狀視圖用於顯示文件
        self.file_tree = QTreeView()
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self._show_file_context_menu)
        
        # 使用QFileSystemModel
        from PyQt5.QtWidgets import QFileSystemModel
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.file_manager.data_dir)
        
        # 設置過濾器以只顯示所關心的文件類型
        self.file_model.setNameFilters(["*.csv", "*.txt", "*.json", "*.html", "*.png", "*.jpg"])
        self.file_model.setNameFilterDisables(False)
        
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(self.file_manager.data_dir))
        
        # 隱藏不需要的列
        self.file_tree.setColumnHidden(1, True)  # 隱藏大小列
        self.file_tree.setColumnHidden(2, True)  # 隱藏類型列
        
        self.file_dock.setWidget(self.file_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.file_dock)
        
        # 連接信號
        self.file_tree.doubleClicked.connect(self._on_file_double_clicked)

    def _show_file_context_menu(self, point):
        """顯示文件右鍵選單"""
        # 獲取選中的文件索引
        index = self.file_tree.indexAt(point)
        if not index.isValid():
            return
            
        # 獲取文件路徑
        file_path = self.file_model.filePath(index)
        
        # 創建上下文選單
        context_menu = QMenu(self)
        
        # 添加選單項
        open_action = context_menu.addAction("開啟")
        open_action.triggered.connect(lambda: self._open_file(file_path))
        
        if file_path.endswith(('.csv', '.txt', '.json')):
            analyze_action = context_menu.addAction("分析")
            analyze_action.triggered.connect(lambda: self.analyze_file(file_path))
            
        if file_path.endswith(('.png', '.jpg', '.html')):
            view_action = context_menu.addAction("查看")
            view_action.triggered.connect(lambda: QDesktopServices.openUrl(f"file:///{file_path}"))
        
        context_menu.addSeparator()
        
        delete_action = context_menu.addAction("刪除")
        delete_action.triggered.connect(lambda: self._delete_file(file_path))
        
        # 顯示選單
        context_menu.exec_(self.file_tree.mapToGlobal(point))

    def _on_file_double_clicked(self, index):
        """處理文件雙擊事件"""
        if not index.isValid():
            return
            
        file_path = self.file_model.filePath(index)
        self._open_file(file_path)

    def _open_file(self, file_path):
        """打開文件"""
        try:
            # 根據文件類型選擇打開方式
            if os.path.isdir(file_path):
                return  # 是目錄則不處理
                
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension in ['.csv', '.txt']:
                self.analysis_tab.load_data(file_path)
                self.tab_widget.setCurrentWidget(self.analysis_tab)
            elif extension == '.json':
                if 'result' in file_path.lower():
                    self.visualization_tab.load_results(file_path)
                    self.tab_widget.setCurrentWidget(self.visualization_tab)
                else:
                    self.analysis_tab.load_data(file_path)
                    self.tab_widget.setCurrentWidget(self.analysis_tab)
            elif extension in ['.png', '.jpg', '.html']:
                # 使用默認瀏覽器或圖片查看器打開
                QDesktopServices.openUrl(f"file:///{file_path}")
            else:
                self.show_message("不支持的文件格式", "無法打開此文件類型：" + extension)
        except Exception as e:
            logger.error(f"打開文件失敗: {str(e)}")
            self.show_error("文件打開失敗", str(e))

    def _delete_file(self, file_path):
        """刪除文件"""
        try:
            reply = QMessageBox.question(
                self, 
                "確認刪除", 
                f"確定要刪除檔案 {os.path.basename(file_path)} 嗎？",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                result = self.file_manager.delete_file(file_path)
                if result:
                    self.show_status_message(f"已刪除: {os.path.basename(file_path)}", 3000)
                else:
                    self.show_error("刪除失敗", f"無法刪除文件: {file_path}")
        except Exception as e:
            logger.error(f"刪除文件失敗: {str(e)}")
            self.show_error("刪除文件失敗", str(e))

    def _setup_signals(self):
        """設置信號連接"""
        # 連接自定義信號
        self.status_message.connect(self.show_status_message)
        self.progress_updated.connect(self.update_progress)
        
        # 連接標籤頁的信號
        self.analysis_tab.status_message.connect(self.show_status_message)
        self.analysis_tab.progress_updated.connect(self.update_progress)
        self.visualization_tab.status_message.connect(self.show_status_message)
        self.settings_tab.settings_changed.connect(self.on_settings_changed)

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