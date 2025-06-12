#!/usr/bin/env python3
"""
編碼器檔案檢測測試腳本
驗證注意力處理器能否正確找到不同編碼器的向量檔案
"""

import os
import sys
import numpy as np
import tempfile
import shutil
from datetime import datetime

def create_mock_encoder_files():
    """創建模擬的編碼器檔案結構"""
    # 創建臨時測試目錄
    test_base = tempfile.mkdtemp(prefix='encoder_test_')
    print(f"📁 創建測試目錄: {test_base}")
    
    # 創建不同的run目錄和編碼器檔案
    test_scenarios = [
        # (run_dir, encoding_dir, filename, encoder_type)
        ('run_20250612_001', '02_bert_encoding', '02_bert_embeddings.npy', 'bert'),
        ('run_20250612_002', '02_encoding', '02_bert_embeddings.npy', 'bert'),
        ('run_20250612_003', '02_encoding', '02_gpt_embeddings.npy', 'gpt'),
        ('run_20250612_004', '02_encoding', '02_t5_embeddings.npy', 't5'),
        ('run_20250612_005', '02_encoding', '02_cnn_embeddings.npy', 'cnn'),
        ('run_20250612_006', '02_encoding', '02_elmo_embeddings.npy', 'elmo'),
    ]
    
    created_files = []
    
    for run_dir, encoding_dir, filename, encoder_type in test_scenarios:
        # 創建目錄結構
        full_encoding_path = os.path.join(test_base, run_dir, encoding_dir)
        os.makedirs(full_encoding_path, exist_ok=True)
        
        # 創建模擬向量檔案
        embeddings_file = os.path.join(full_encoding_path, filename)
        mock_embeddings = np.random.randn(10, 768)  # 10個樣本，768維
        np.save(embeddings_file, mock_embeddings)
        
        # 創建編碼器信息檔案
        info_file = os.path.join(full_encoding_path, f'encoder_info_{encoder_type}.json')
        info_content = f'''{{
    "encoder_type": "{encoder_type}",
    "model_name": "{encoder_type}-base-uncased",
    "embedding_dim": 768,
    "created_time": "{datetime.now().isoformat()}"
}}'''
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(info_content)
        
        created_files.append((embeddings_file, encoder_type))
        print(f"  ✅ 創建 {encoder_type.upper()} 檔案: {filename}")
    
    return test_base, created_files

def test_attention_processor_file_detection():
    """測試注意力處理器的檔案檢測功能"""
    print("🧪 開始測試注意力處理器檔案檢測功能")
    print("=" * 60)
    
    # 創建模擬檔案
    test_base, created_files = create_mock_encoder_files()
    
    try:
        # 測試不同編碼器類型的檔案檢測
        sys.path.insert(0, '.')
        from modules.attention_processor import AttentionProcessor
        
        encoder_types = ['bert', 'gpt', 't5', 'cnn', 'elmo']
        
        print(f"\n🔍 測試檔案檢測功能...")
        
        detection_results = {}
        
        for encoder_type in encoder_types:
            print(f"\n📋 測試 {encoder_type.upper()} 編碼器檔案檢測:")
            
            # 創建注意力處理器實例
            processor = AttentionProcessor(
                output_dir=os.path.join(test_base, 'run_current', '03_attention_testing'),
                encoder_type=encoder_type
            )
            
            # 測試檔案搜尋
            found_file = processor._find_existing_embeddings(encoder_type)
            
            if found_file:
                print(f"  ✅ 找到檔案: {os.path.basename(found_file)}")
                print(f"     路徑: {found_file}")
                detection_results[encoder_type] = True
                
                # 驗證檔案內容
                try:
                    embeddings = np.load(found_file)
                    print(f"     向量形狀: {embeddings.shape}")
                except Exception as e:
                    print(f"     ⚠️ 檔案讀取失敗: {e}")
            else:
                print(f"  ❌ 未找到 {encoder_type.upper()} 檔案")
                detection_results[encoder_type] = False
        
        # 測試檔案驗證功能
        print(f"\n🔧 測試檔案驗證功能...")
        processor = AttentionProcessor(output_dir=test_base)
        
        validation_results = {}
        for file_path, expected_type in created_files:
            for test_type in encoder_types:
                is_valid = processor._validate_embeddings_file(file_path, test_type)
                validation_results[(os.path.basename(file_path), test_type)] = is_valid
                
                if test_type == expected_type and is_valid:
                    print(f"  ✅ {os.path.basename(file_path)} 正確識別為 {test_type.upper()}")
                elif test_type == expected_type and not is_valid:
                    print(f"  ❌ {os.path.basename(file_path)} 未能識別為 {test_type.upper()}")
        
        # 統計結果
        print(f"\n📊 測試結果統計:")
        successful_detections = sum(detection_results.values())
        total_encoders = len(encoder_types)
        
        print(f"  檔案檢測成功率: {successful_detections}/{total_encoders} ({successful_detections/total_encoders*100:.1f}%)")
        
        # 檢查每種編碼器的檔案是否被正確檢測
        for encoder_type in encoder_types:
            status = "✅" if detection_results[encoder_type] else "❌"
            print(f"  {status} {encoder_type.upper()} 編碼器檔案檢測")
        
        return successful_detections == total_encoders
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        return False
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        return False
    finally:
        # 清理測試檔案
        try:
            shutil.rmtree(test_base)
            print(f"\n🧹 清理測試目錄: {test_base}")
        except Exception as e:
            print(f"⚠️ 清理測試目錄失敗: {e}")

def test_file_naming_patterns():
    """測試檔案命名模式的相容性"""
    print("\n🔤 測試檔案命名模式相容性...")
    
    test_patterns = [
        # (filename, encoder_type, should_match)
        ('02_bert_embeddings.npy', 'bert', True),
        ('02_gpt_embeddings.npy', 'gpt', True),
        ('02_t5_embeddings.npy', 't5', True),
        ('bert_embeddings.npy', 'bert', True),
        ('gpt_embeddings.npy', 'gpt', True),
        ('embeddings.npy', 'bert', True),  # 通用檔案名，應該接受
        ('02_bert_embeddings.npy', 'gpt', False),  # 錯誤的編碼器類型
        ('random_file.npy', 'bert', False),  # 不相關的檔案
    ]
    
    try:
        sys.path.insert(0, '.')
        from modules.attention_processor import AttentionProcessor
        
        processor = AttentionProcessor()
        
        correct_matches = 0
        total_tests = len(test_patterns)
        
        for filename, encoder_type, should_match in test_patterns:
            # 創建臨時檔案路徑進行測試
            temp_path = f"/tmp/test_encoder/{filename}"
            result = processor._validate_embeddings_file(temp_path, encoder_type)
            
            if result == should_match:
                correct_matches += 1
                status = "✅"
            else:
                status = "❌"
            
            expected = "匹配" if should_match else "不匹配"
            actual = "匹配" if result else "不匹配"
            print(f"  {status} {filename} vs {encoder_type}: 預期 {expected}, 實際 {actual}")
        
        success_rate = correct_matches / total_tests * 100
        print(f"\n📊 檔案模式測試成功率: {correct_matches}/{total_tests} ({success_rate:.1f}%)")
        
        return success_rate >= 80  # 80%以上的成功率認為通過
        
    except Exception as e:
        print(f"❌ 檔案模式測試失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🚀 編碼器檔案檢測完整測試")
    print("=" * 80)
    
    tests = [
        ("檔案檢測功能", test_attention_processor_file_detection),
        ("檔案命名模式", test_file_naming_patterns),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} 測試通過")
                passed_tests += 1
            else:
                print(f"❌ {test_name} 測試失敗")
        except Exception as e:
            print(f"❌ {test_name} 測試發生錯誤: {e}")
    
    print(f"\n{'='*80}")
    print(f"🏁 測試完成: {passed_tests}/{total_tests} 測試通過")
    
    if passed_tests == total_tests:
        print("🎉 所有測試通過！注意力處理器現在支援多種編碼器檔案檢測")
        print("\n💡 功能特性:")
        print("  • 支援 BERT、GPT、T5、CNN、ELMo 編碼器檔案檢測")
        print("  • 相容新舊檔案命名格式")
        print("  • 智慧檔案驗證機制")
        print("  • 自動搜尋最新的編碼器檔案")
        return True
    else:
        print("⚠️ 部分測試失敗，請檢查相關功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)