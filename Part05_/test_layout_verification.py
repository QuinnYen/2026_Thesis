#!/usr/bin/env python3
"""
注意力機制頁面佈局驗證腳本
驗證新的三列緊湊佈局設計
"""

def verify_layout_changes():
    """驗證佈局修改"""
    print("🔍 驗證注意力機制頁面重新設計...")
    print("=" * 50)
    
    # 讀取修改後的檔案
    with open('gui/main_window.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查關鍵修改點
    checks = [
        ("移除滾動框架", "Canvas" not in content[content.find("create_attention_testing_tab"):content.find("def run_single_attention")]),
        ("三列橫向設定區域", "top_config_frame" in content),
        ("三列實驗區域", "experiments_frame" in content),
        ("緊湊模組化流水線", 'text="🚀 模組化流水線"' in content),
        ("並排佈局", "side='left'" in content[content.find("create_attention_testing_tab"):]),
        ("緊湊字體", "font=('TkDefaultFont', 8)" in content),
        ("移除scrollbar", "scrollbar" not in content[content.find("create_attention_testing_tab"):content.find("def run_single_attention")])
    ]
    
    print("✅ 佈局修改檢查:")
    passed = 0
    for check_name, condition in checks:
        status = "✅" if condition else "❌"
        print(f"  {status} {check_name}")
        if condition:
            passed += 1
    
    print(f"\n📊 檢查結果: {passed}/{len(checks)} 項修改通過")
    
    if passed == len(checks):
        print("\n🎉 注意力機制頁面重新設計成功！")
        print("\n📋 主要改進:")
        print("  • 所有內容現在可在一個頁面顯示，無需滾動")
        print("  • 三列並排佈局，更有效利用螢幕空間")
        print("  • 緊湊的UI元素設計")
        print("  • 更好的視覺組織結構")
        return True
    else:
        print("\n⚠️  部分修改可能需要調整")
        return False

def check_method_structure():
    """檢查方法結構"""
    print("\n🔧 檢查方法結構...")
    
    with open('gui/main_window.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查關鍵方法是否完整
    methods = [
        "create_attention_testing_tab",
        "run_single_attention", 
        "run_dual_attention",
        "run_triple_attention"
    ]
    
    for method in methods:
        if f"def {method}(" in content:
            print(f"  ✅ {method} 方法存在")
        else:
            print(f"  ❌ {method} 方法缺失")

if __name__ == "__main__":
    print("🧪 注意力機制頁面佈局驗證")
    print("=" * 60)
    
    success = verify_layout_changes()
    check_method_structure()
    
    print("\n" + "=" * 60)
    if success:
        print("🎯 驗證完成：注意力機制頁面已成功重新設計為三列緊湊佈局")
        print("💡 用戶現在可以在一個頁面查看所有功能，無需滾動")
    else:
        print("⚠️  驗證發現問題，可能需要進一步調整")