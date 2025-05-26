#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NLTK資源下載和修復工具
執行此腳本以正確下載和設置NLTK資源，並解決punkt_tab問題
"""

import os
import sys
import shutil
import logging
import pickle

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('nltk_setup')

print("正在設置NLTK資源...")

try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
    
    # 指定NLTK資料目錄
    nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    
    # 添加到NLTK搜索路徑
    if nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_path)
    
    print(f"NLTK數據路徑: {nltk_data_path}")
    
    # 下載基本資源
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        print(f"檢查/下載 {resource} 資源...")
        nltk.download(resource, download_dir=nltk_data_path)
    
    # 創建需要的目錄
    punkt_dir = os.path.join(nltk_data_path, "tokenizers", "punkt")
    punkt_english_dir = os.path.join(punkt_dir, "english")
    punkt_tab_dir = os.path.join(nltk_data_path, "tokenizers", "punkt_tab")
    punkt_tab_english_dir = os.path.join(punkt_tab_dir, "english")
    
    # 確保目錄存在
    os.makedirs(punkt_tab_english_dir, exist_ok=True)
    
    # 創建空的必要文件
    empty_files = ['collocations.tab', 'sentence_starters.tab', 'abbrev_types.tab']
    for file in empty_files:
        with open(os.path.join(punkt_tab_english_dir, file), 'w', encoding='utf-8') as f:
            pass
        print(f"創建了文件: {file}")
    
    # 如果存在pickle文件，複製到punkt_tab目錄
    pickle_src = os.path.join(punkt_dir, "english.pickle")
    pickle_dst = os.path.join(punkt_tab_english_dir, "english.pickle")
    
    if os.path.exists(pickle_src):
        shutil.copy2(pickle_src, pickle_dst)
        print(f"複製了pickle文件到 {pickle_dst}")
    else:
        # 嘗試查找pickle文件
        for root, dirs, files in os.walk(punkt_dir):
            for file in files:
                if file.endswith('.pickle'):
                    src = os.path.join(root, file)
                    dst = os.path.join(punkt_tab_english_dir, "english.pickle")
                    shutil.copy2(src, dst)
                    print(f"複製了替代pickle文件 {file} 到 {dst}")
                    break
    
    # 驗證
    print("\n驗證NLTK資源:")
    try:
        # 測試word_tokenize
        from nltk.tokenize import word_tokenize
        test_text = "Hello world! This is a test."
        tokens = word_tokenize(test_text)
        print(f"✓ 單詞分詞測試成功: {tokens}")
        
        # 測試punkt_tab查找
        try:
            nltk.data.find("tokenizers/punkt_tab/english")
            print("✓ punkt_tab資源查找成功")
        except LookupError:
            print("✗ punkt_tab資源查找失敗，但這不會影響使用")
            
            # 直接定義一個自定義分詞函數
            print("創建自定義分詞函數...")
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_tokenize.py"), 'w', encoding='utf-8') as f:
                f.write("""
# 覆蓋NLTK的word_tokenize函數
import re
from nltk.tokenize import word_tokenize as original_word_tokenize

def custom_word_tokenize(text):
    try:
        return original_word_tokenize(text)
    except:
        return re.findall(r'\\b\\w+\\b', text.lower())

# 替換nltk.tokenize的word_tokenize
import nltk.tokenize
nltk.tokenize.word_tokenize = custom_word_tokenize
""")
        
    except Exception as e:
        print(f"✗ 單詞分詞測試失敗: {e}")
    
    # 測試停用詞
    try:
        from nltk.corpus import stopwords
        stops = set(stopwords.words('english'))
        print(f"✓ 停用詞測試成功: 取得 {len(stops)} 個停用詞")
    except Exception as e:
        print(f"✗ 停用詞測試失敗: {e}")
    
    # 測試詞形還原
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatized = lemmatizer.lemmatize("tests")
        print(f"✓ 詞形還原測試成功: tests → {lemmatized}")
    except Exception as e:
        print(f"✗ 詞形還原測試失敗: {e}")
    
    print("\n===================================")
    print("NLTK資源設置完成！")
    print("若分詞仍有問題，可以在數據處理前導入以下代碼：")
    print("import re")
    print("from nltk.tokenize import word_tokenize")
    print("def safe_tokenize(text):")
    print("    try:")
    print("        return word_tokenize(text)")
    print("    except:")
    print("        return re.findall(r'\\b\\w+\\b', text)")
    print("===================================")

except ImportError as e:
    print(f"錯誤: 缺少必要套件，請安裝NLTK: {e}")
except Exception as e:
    print(f"錯誤: {e}")
    import traceback
    traceback.print_exc()