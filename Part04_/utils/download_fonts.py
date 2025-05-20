#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下載並安裝 DejaVu 字體
"""

import os
import sys
import requests
import zipfile
from pathlib import Path

def download_dejavu_fonts():
    """
    下載並解壓 DejaVu 字體到 resources/fonts 目錄
    """
    # 設定目錄路徑
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fonts_dir = os.path.join(current_dir, "resources", "fonts")
    
    # 確保字體目錄存在
    os.makedirs(fonts_dir, exist_ok=True)
    
    # DejaVu 字體下載 URL
    dejavu_url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
    
    try:
        print("開始下載 DejaVu 字體...")
        
        # 下載字體文件
        response = requests.get(dejavu_url)
        response.raise_for_status()
        
        # 保存 zip 文件
        zip_path = os.path.join(fonts_dir, "dejavu-fonts.zip")
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        print("解壓字體文件...")
        
        # 解壓文件
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # 只提取 .ttf 文件
            for file in zip_ref.namelist():
                if file.endswith(".ttf"):
                    zip_ref.extract(file, fonts_dir)
        
        # 移動字體文件到正確位置
        ttf_dir = os.path.join(fonts_dir, "dejavu-fonts-ttf-2.37", "ttf")
        if os.path.exists(ttf_dir):
            for font_file in os.listdir(ttf_dir):
                if font_file.endswith(".ttf"):
                    src = os.path.join(ttf_dir, font_file)
                    dst = os.path.join(fonts_dir, font_file)
                    os.rename(src, dst)
        
        # 清理臨時文件
        os.remove(zip_path)
        os.system(f'rmdir /s /q "{os.path.join(fonts_dir, "dejavu-fonts-ttf-2.37")}"')
        
        print("字體安裝完成！")
        return True
        
    except Exception as e:
        print(f"下載字體時發生錯誤：{str(e)}")
        return False

if __name__ == "__main__":
    download_dejavu_fonts() 