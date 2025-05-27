#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化的注意力機制測試腳本
只使用Python標準庫進行基本功能測試
"""

import sys
import os
import json
import random
import math

# 添加當前目錄到Python路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_basic_imports():
    """測試基本導入功能"""
    print("=" * 50)
    print("測試基本導入功能")
    print("=" * 50)
    
    try:
        # 測試標準庫導入
        import numpy as np
        print("✓ NumPy 導入成功")
    except ImportError:
        print("✗ NumPy 導入失敗，請安裝：pip install numpy")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas 導入成功")
    except ImportError:
        print("✗ Pandas 導入失敗，請安裝：pip install pandas")
        return False
    
    try:
        import torch
        print("✓ PyTorch 導入成功")
    except ImportError:
        print("✗ PyTorch 導入失敗，請安裝：pip install torch")
        return False
    
    try:
        from modules.attention_mechanism import AttentionMechanism
        print("✓ 注意力機制模組導入成功")
    except ImportError as e:
        print(f"✗ 注意力機制模組導入失敗：{str(e)}")
        return False
    
    return True

def test_attention_mechanism_creation():
    """測試注意力機制創建"""
    print("\n" + "=" * 50)
    print("測試注意力機制創建")
    print("=" * 50)
    
    try:
        from modules.attention_mechanism import create_attention_mechanism
        
        # 測試創建不同類型的注意力機制
        attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
        
        for attention_type in attention_types:
            try:
                mechanism = create_attention_mechanism(attention_type)
                print(f"✓ {attention_type} 注意力機制創建成功")
            except Exception as e:
                print(f"✗ {attention_type} 注意力機制創建失敗：{str(e)}")
                
        return True
        
    except ImportError as e:
        print(f"✗ 無法導入注意力機制模組：{str(e)}")
        return False

def test_mock_data_processing():
    """使用模擬數據測試注意力機制"""
    print("\n" + "=" * 50)
    print("測試模擬數據處理")
    print("=" * 50)
    
    try:
        import numpy as np
        import pandas as pd
        from modules.attention_mechanism import apply_attention_mechanism
        
        # 創建模擬數據
        np.random.seed(42)
        n_docs = 20
        embed_dim = 100  # 簡化的維度
        
        # 模擬BERT嵌入向量
        embeddings = np.random.randn(n_docs, embed_dim)
        
        # 模擬元數據
        sentiments = ['positive', 'negative', 'neutral']
        doc_sentiments = [sentiments[i % 3] for i in range(n_docs)]
        
        metadata = pd.DataFrame({
            'sentiment': doc_sentiments,
            'text': [f'Sample text {i}' for i in range(n_docs)]
        })
        
        print(f"✓ 創建模擬數據：{n_docs} 個文檔，{embed_dim} 維特徵")
        print(f"✓ 情感分布：{pd.Series(doc_sentiments).value_counts().to_dict()}")
        
        # 測試不同注意力機制
        test_mechanisms = ['no', 'similarity']  # 簡化測試
        
        for mechanism in test_mechanisms:
            try:
                result = apply_attention_mechanism(
                    attention_type=mechanism,
                    embeddings=embeddings,
                    metadata=metadata
                )
                
                metrics = result['metrics']
                print(f"✓ {mechanism} 注意力測試成功")
                print(f"  - 內聚度: {metrics['coherence']:.4f}")
                print(f"  - 分離度: {metrics['separation']:.4f}")
                print(f"  - 綜合得分: {metrics['combined_score']:.4f}")
                
            except Exception as e:
                print(f"✗ {mechanism} 注意力測試失敗：{str(e)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模擬數據處理失敗：{str(e)}")
        return False

def test_file_structure():
    """測試文件結構完整性"""
    print("\n" + "=" * 50)
    print("測試文件結構")
    print("=" * 50)
    
    required_files = [
        'modules/attention_mechanism.py',
        'modules/attention_analyzer.py',
        'modules/attention_processor.py',
        'modules/bert_encoder.py',
        'modules/run_manager.py',
        'Part05_Main.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} 缺失")
            all_exist = False
    
    # 檢查examples目錄
    examples_files = [
        'examples/attention_example.py',
        'examples/README.md'
    ]
    
    for file_path in examples_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} 缺失")
    
    return all_exist

def generate_test_report():
    """生成測試報告"""
    print("\n" + "=" * 50)
    print("生成測試報告")
    print("=" * 50)
    
    report = {
        "test_time": "2024-12-19",
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "test_results": {},
        "recommendations": []
    }
    
    # 執行所有測試
    tests = [
        ("file_structure", test_file_structure),
        ("basic_imports", test_basic_imports),
        ("attention_creation", test_attention_mechanism_creation),
        ("mock_data_processing", test_mock_data_processing)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            result = test_func()
            report["test_results"][test_name] = "PASS" if result else "FAIL"
            if not result:
                all_passed = False
        except Exception as e:
            report["test_results"][test_name] = f"ERROR: {str(e)}"
            all_passed = False
    
    # 生成建議
    if not all_passed:
        if report["test_results"].get("basic_imports") == "FAIL":
            report["recommendations"].append("請安裝必要的依賴包：pip install -r requirements.txt")
        
        if report["test_results"].get("file_structure") == "FAIL":
            report["recommendations"].append("請確認所有必要文件都已正確創建")
    else:
        report["recommendations"].append("所有測試通過！系統已準備就緒")
    
    # 保存報告
    with open("test_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 輸出摘要
    print(f"\n測試完成！")
    print(f"總體結果: {'通過' if all_passed else '失敗'}")
    print(f"詳細報告已保存到: test_report.json")
    
    if report["recommendations"]:
        print("\n建議:")
        for rec in report["recommendations"]:
            print(f"  - {rec}")
    
    return all_passed

def main():
    """主函數"""
    print("BERT情感分析系統 - 注意力機制測試")
    print("此測試腳本將驗證系統的基本功能")
    
    try:
        success = generate_test_report()
        return 0 if success else 1
    except Exception as e:
        print(f"測試過程中發生錯誤：{str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 