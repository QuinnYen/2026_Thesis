#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI主程式啟動器 - 提供單獨的GUI入口點
"""

import tkinter as tk
import sys
import os

# 添加父目錄到路徑以便導入模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """GUI主程式入口點"""
    try:
        # 導入主應用程序
        from gui.main_window import MainApplication
        
        # 創建主視窗
        root = tk.Tk()
        
        # 設定視窗圖示（如果有的話）
        try:
            # 可以添加程式圖示
            # root.iconbitmap('icon.ico')
            pass
        except:
            pass
        
        # 創建應用程序實例
        app = MainApplication(root)
        
        # 啟動主循環
        root.mainloop()
        
    except ImportError as e:
        print(f"模組導入錯誤: {e}")
        print("請確保已正確安裝所有依賴套件")
        input("按Enter鍵退出...")
    except Exception as e:
        print(f"程式啟動錯誤: {e}")
        input("按Enter鍵退出...")

if __name__ == "__main__":
    main()