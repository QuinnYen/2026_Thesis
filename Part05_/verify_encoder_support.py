#!/usr/bin/env python3
"""
編碼器支援驗證腳本
檢查注意力機制系統對多種編碼器的支援程度
"""

import os
import ast
import sys

def analyze_attention_processor():
    """分析注意力處理器的編碼器支援"""
    print("🔍 分析注意力處理器...")
    
    file_path = 'modules/attention_processor.py'
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查關鍵功能
    checks = [
        ("編碼器類型參數", "encoder_type" in content),
        ("多編碼器檔案搜尋", "02_{encoder_type}_embeddings.npy" in content),
        ("檔案驗證功能", "_validate_embeddings_file" in content),
        ("向後相容性", "02_bert_embeddings.npy" in content),
        ("新目錄結構支援", "02_encoding" in content),
        ("舊目錄結構支援", "02_bert_encoding" in content),
        ("動態編碼器檢測", "encoder_type.upper()" in content),
    ]
    
    passed = 0
    for check_name, condition in checks:
        if condition:
            print(f"  ✅ {check_name}")
            passed += 1
        else:
            print(f"  ❌ {check_name}")
    
    print(f"  📊 注意力處理器檢查: {passed}/{len(checks)} 通過")
    return passed == len(checks)

def analyze_main_functions():
    """分析主程式的編碼器支援"""
    print("\n🔍 分析主程式函數...")
    
    file_path = 'Part05_Main.py'
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查函數簽名是否包含encoder_type參數
    functions_to_check = [
        'process_attention_analysis',
        'process_attention_analysis_with_multiple_combinations'
    ]
    
    passed = 0
    total = 0
    
    for func_name in functions_to_check:
        total += 1
        if f"def {func_name}" in content:
            # 檢查函數是否有encoder_type參數
            func_start = content.find(f"def {func_name}")
            func_sig_end = content.find(":", func_start)
            func_signature = content[func_start:func_sig_end]
            
            if "encoder_type" in func_signature:
                print(f"  ✅ {func_name} 支援編碼器類型參數")
                passed += 1
            else:
                print(f"  ❌ {func_name} 缺少編碼器類型參數")
        else:
            print(f"  ❌ 找不到函數: {func_name}")
    
    # 檢查AttentionProcessor初始化是否包含encoder_type
    processor_inits = content.count("AttentionProcessor(")
    encoder_type_inits = content.count("encoder_type=")
    
    print(f"  📋 AttentionProcessor初始化: {encoder_type_inits}/{processor_inits} 包含編碼器類型")
    
    if processor_inits > 0:
        total += 1
        if encoder_type_inits >= processor_inits:
            print(f"  ✅ 所有AttentionProcessor初始化都包含編碼器類型")
            passed += 1
        else:
            print(f"  ⚠️ 部分AttentionProcessor初始化缺少編碼器類型")
    
    print(f"  📊 主程式函數檢查: {passed}/{total} 通過")
    return passed == total

def analyze_gui_integration():
    """分析GUI整合情況"""
    print("\n🔍 分析GUI整合...")
    
    file_path = 'gui/main_window.py'
    
    if not os.path.exists(file_path):
        print(f"❌ 找不到檔案: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = [
        ("編碼器類型變數", "self.encoder_type" in content),
        ("編碼器選擇元件", "encoder_combo" in content),
        ("注意力分析調用更新", "encoder_type=self.encoder_type.get()" in content),
        ("編碼器類型傳遞", "process_attention_analysis_with_multiple_combinations" in content and "encoder_type=" in content),
    ]
    
    passed = 0
    for check_name, condition in checks:
        if condition:
            print(f"  ✅ {check_name}")
            passed += 1
        else:
            print(f"  ❌ {check_name}")
    
    print(f"  📊 GUI整合檢查: {passed}/{len(checks)} 通過")
    return passed == len(checks)

def analyze_file_structure():
    """分析檔案結構支援"""
    print("\n🔍 分析檔案結構支援...")
    
    # 檢查模組化架構檔案
    modular_files = [
        'modules/text_encoders.py',
        'modules/modular_pipeline.py',
        'modules/encoder_factory.py'
    ]
    
    supported_encoders = []
    
    for file_path in modular_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            encoders = ['bert', 'gpt', 't5', 'cnn', 'elmo']
            for encoder in encoders:
                if f"{encoder}Encoder" in content or f"{encoder.upper()}Encoder" in content:
                    if encoder not in supported_encoders:
                        supported_encoders.append(encoder)
    
    print(f"  📋 支援的編碼器: {', '.join(supported_encoders).upper()}")
    
    # 檢查檔案命名格式
    file_path = 'modules/modular_pipeline.py'
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "02_{encoder_type}_embeddings.npy" in content:
            print(f"  ✅ 模組化檔案命名格式正確")
        else:
            print(f"  ❌ 模組化檔案命名格式缺失")
    
    return len(supported_encoders) >= 3  # 至少支援3種編碼器

def check_syntax_errors():
    """檢查語法錯誤"""
    print("\n🔧 檢查語法錯誤...")
    
    files_to_check = [
        'modules/attention_processor.py',
        'Part05_Main.py',
        'gui/main_window.py'
    ]
    
    syntax_ok = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                ast.parse(code)
                print(f"  ✅ {file_path} 語法正確")
            except SyntaxError as e:
                print(f"  ❌ {file_path} 語法錯誤: {e}")
                syntax_ok = False
            except Exception as e:
                print(f"  ⚠️ {file_path} 檢查時發生錯誤: {e}")
        else:
            print(f"  ⚠️ 找不到檔案: {file_path}")
    
    return syntax_ok

def main():
    """主驗證函數"""
    print("🚀 編碼器支援驗證")
    print("=" * 60)
    
    tests = [
        ("語法檢查", check_syntax_errors),
        ("注意力處理器分析", analyze_attention_processor),
        ("主程式函數分析", analyze_main_functions),
        ("GUI整合分析", analyze_gui_integration),
        ("檔案結構支援", analyze_file_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通過")
            else:
                print(f"❌ {test_name} 失敗")
        except Exception as e:
            print(f"❌ {test_name} 發生錯誤: {e}")
    
    print(f"\n{'='*60}")
    print(f"🏁 驗證完成: {passed}/{total} 檢查通過")
    
    if passed == total:
        print("🎉 所有檢查通過！系統現在支援多種編碼器")
        print("\n📋 支援的功能:")
        print("  • 多種編碼器檔案自動檢測")
        print("  • 新舊檔案格式相容")
        print("  • GUI編碼器選擇整合")
        print("  • 智慧檔案驗證機制")
        print("\n💡 使用方式:")
        print("  1. 在GUI中選擇編碼器類型")
        print("  2. 運行模組化流水線生成編碼檔案")
        print("  3. 注意力機制測試會自動找到對應的編碼檔案")
        return True
    else:
        print("⚠️ 部分檢查失敗，可能需要進一步調整")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)