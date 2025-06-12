#!/usr/bin/env python3
"""
基本導入測試 - 驗證模組化架構的基本結構
"""

import sys
import os

# 添加當前目錄到路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_basic_structure():
    """測試基本文件結構"""
    print("🔍 測試基本文件結構...")
    
    required_files = [
        'modules/base_interfaces.py',
        'modules/encoder_factory.py', 
        'modules/aspect_factory.py',
        'modules/modular_pipeline.py',
        'modules/modular_gui_extensions.py',
        'test_modular_architecture.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(current_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    if not missing_files:
        print("  🎉 所有必要文件都存在！")
        return True
    else:
        print(f"  ⚠️  缺少 {len(missing_files)} 個文件")
        return False

def test_basic_imports():
    """測試基本導入"""
    print("\n🧪 測試基本模組導入...")
    
    try:
        # 測試基礎接口
        from modules.base_interfaces import BaseTextEncoder, BaseAspectClassifier, BasePipeline
        print("  ✅ 基礎接口導入成功")
    except ImportError as e:
        print(f"  ❌ 基礎接口導入失敗: {e}")
        return False
    
    try:
        # 測試工廠類
        from modules.encoder_factory import EncoderFactory
        print("  ✅ 編碼器工廠導入成功")
    except ImportError as e:
        print(f"  ❌ 編碼器工廠導入失敗: {e}")
        return False
    
    try:
        from modules.aspect_factory import AspectFactory
        print("  ✅ 面向分類器工廠導入成功")
    except ImportError as e:
        print(f"  ❌ 面向分類器工廠導入失敗: {e}")
        return False
    
    try:
        from modules.modular_pipeline import ModularPipeline
        print("  ✅ 模組化流水線導入成功")
    except ImportError as e:
        print(f"  ❌ 模組化流水線導入失敗: {e}")
        return False
    
    return True

def test_factory_basic_functionality():
    """測試工廠的基本功能（不需要額外依賴）"""
    print("\n⚙️ 測試工廠基本功能...")
    
    try:
        from modules.encoder_factory import EncoderFactory
        from modules.aspect_factory import AspectFactory
        
        # 測試獲取可用選項
        encoders = EncoderFactory.get_available_encoders()
        classifiers = AspectFactory.get_available_classifiers()
        
        print(f"  📋 可用編碼器: {encoders}")
        print(f"  📋 可用分類器: {classifiers}")
        
        # 測試獲取信息
        for encoder in encoders[:2]:  # 只測試前兩個
            info = EncoderFactory.get_encoder_info(encoder)
            print(f"  📄 {encoder}: {info.get('name', 'Unknown')}")
        
        for classifier in classifiers[:2]:  # 只測試前兩個
            info = AspectFactory.get_classifier_info(classifier) 
            print(f"  📄 {classifier}: {info.get('name', 'Unknown')}")
        
        return True
    except Exception as e:
        print(f"  ❌ 工廠功能測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 開始基本架構驗證")
    print("=" * 50)
    
    tests = [
        ("文件結構", test_basic_structure),
        ("基本導入", test_basic_imports), 
        ("工廠功能", test_factory_basic_functionality)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name} 測試通過")
            else:
                print(f"❌ {test_name} 測試失敗")
        except Exception as e:
            print(f"❌ {test_name} 測試發生錯誤: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 基本驗證完成: {passed}/{len(tests)} 測試通過")
    
    if passed == len(tests):
        print("🎉 模組化架構基本結構正常！")
        print("\n📝 下一步:")
        print("  1. 確保安裝完整依賴: pip install -r requirements.txt")
        print("  2. 運行完整測試: python test_modular_architecture.py")
        print("  3. 啟動GUI: python Part05_Main.py")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查相關組件")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)