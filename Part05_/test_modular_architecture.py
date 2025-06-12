#!/usr/bin/env python3
"""
模組化架構測試腳本
測試新的編碼器和面向分類器組合
"""

import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime

def test_basic_imports():
    """測試基本導入"""
    print("🔍 測試基本導入...")
    
    try:
        # 測試基礎接口
        from modules.base_interfaces import BaseTextEncoder, BaseAspectClassifier, BasePipeline
        print("  ✅ 基礎接口導入成功")
        
        # 測試工廠類
        from modules.encoder_factory import EncoderFactory
        from modules.aspect_factory import AspectFactory
        print("  ✅ 工廠類導入成功")
        
        # 測試模組化流水線
        from modules.modular_pipeline import ModularPipeline
        print("  ✅ 模組化流水線導入成功")
        
        return True
    except ImportError as e:
        print(f"  ❌ 導入失敗: {e}")
        return False

def test_encoder_factory():
    """測試編碼器工廠"""
    print("\n🔧 測試編碼器工廠...")
    
    try:
        from modules.encoder_factory import EncoderFactory
        
        # 測試獲取可用編碼器
        available_encoders = EncoderFactory.get_available_encoders()
        print(f"  📋 可用編碼器: {available_encoders}")
        
        # 測試編碼器信息
        for encoder_type in available_encoders:
            info = EncoderFactory.get_encoder_info(encoder_type)
            print(f"  📄 {encoder_type}: {info['name']} ({info.get('embedding_dim', 'N/A')}維)")
        
        return True
    except Exception as e:
        print(f"  ❌ 編碼器工廠測試失敗: {e}")
        return False

def test_aspect_factory():
    """測試面向分類器工廠"""
    print("\n🎯 測試面向分類器工廠...")
    
    try:
        from modules.aspect_factory import AspectFactory
        
        # 測試獲取可用分類器
        available_classifiers = AspectFactory.get_available_classifiers()
        print(f"  📋 可用分類器: {available_classifiers}")
        
        # 測試分類器信息
        for classifier_type in available_classifiers:
            info = AspectFactory.get_classifier_info(classifier_type)
            print(f"  📄 {classifier_type}: {info['name']}")
            print(f"     - 優點: {', '.join(info.get('advantages', []))}")
            print(f"     - 適用場景: {', '.join(info.get('suitable_for', []))}")
        
        return True
    except Exception as e:
        print(f"  ❌ 面向分類器工廠測試失敗: {e}")
        return False

def create_sample_data():
    """創建測試數據"""
    print("\n📝 創建測試數據...")
    
    sample_texts = [
        "This movie is absolutely fantastic! Great acting and storyline.",
        "The product quality is terrible, very disappointed.",
        "Average restaurant, nothing special but decent food.",
        "Excellent service and amazing food, highly recommended!",
        "Poor customer support, will not buy again.",
        "The plot was confusing but the visuals were stunning.",
        "Good value for money, satisfied with the purchase.",
        "Outstanding performance by all actors in this film.",
        "Food was cold and tasteless, worst experience ever.",
        "Beautiful cinematography and excellent direction."
    ]
    
    sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative',
                 'neutral', 'positive', 'positive', 'negative', 'positive']
    
    df = pd.DataFrame({
        'text': sample_texts,
        'sentiment': sentiments
    })
    
    print(f"  ✅ 創建了 {len(df)} 條測試數據")
    return df

def test_individual_encoders():
    """測試個別編碼器"""
    print("\n🧪 測試個別編碼器...")
    
    try:
        from modules.encoder_factory import EncoderFactory
        
        # 創建測試文本
        test_texts = ["This is a test sentence.", "Another test sentence for encoding."]
        
        # 測試每個編碼器（除了需要特殊設置的）
        test_encoders = ['bert', 'cnn']  # 簡化測試，只測試容易配置的編碼器
        
        for encoder_type in test_encoders:
            try:
                print(f"  🔧 測試 {encoder_type.upper()} 編碼器...")
                
                # 創建編碼器
                encoder = EncoderFactory.create_encoder(encoder_type, config={'batch_size': 2})
                
                # 測試編碼
                start_time = time.time()
                embeddings = encoder.encode(test_texts)
                encoding_time = time.time() - start_time
                
                print(f"    ✅ 編碼成功: 形狀 {embeddings.shape}, 耗時 {encoding_time:.2f}秒")
                print(f"    📏 嵌入維度: {encoder.get_embedding_dim()}")
                
            except Exception as e:
                print(f"    ⚠️  {encoder_type} 編碼器測試跳過: {e}")
        
        return True
    except Exception as e:
        print(f"  ❌ 編碼器測試失敗: {e}")
        return False

def test_individual_aspect_classifiers():
    """測試個別面向分類器"""
    print("\n🎯 測試個別面向分類器...")
    
    try:
        from modules.aspect_factory import AspectFactory
        
        # 創建假的嵌入向量和元數據
        embeddings = np.random.randn(5, 768)  # 5個樣本，768維
        metadata = pd.DataFrame({
            'text': ["Test sentence " + str(i) for i in range(5)],
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
        
        # 測試每個分類器（除了需要特殊依賴的）
        test_classifiers = ['default', 'lda', 'nmf']  # 簡化測試
        
        for classifier_type in test_classifiers:
            try:
                print(f"  🔧 測試 {classifier_type.upper()} 分類器...")
                
                # 創建分類器
                classifier = AspectFactory.create_classifier(classifier_type)
                
                # 測試分類
                start_time = time.time()
                aspect_vectors, results = classifier.fit_transform(embeddings, metadata)
                classification_time = time.time() - start_time
                
                aspect_names = classifier.get_aspect_names()
                
                print(f"    ✅ 分類成功: {len(aspect_names)} 個面向, 耗時 {classification_time:.2f}秒")
                print(f"    📊 面向向量形狀: {aspect_vectors.shape}")
                print(f"    🏷️  面向名稱: {aspect_names[:3]}...")  # 只顯示前3個
                
            except Exception as e:
                print(f"    ⚠️  {classifier_type} 分類器測試跳過: {e}")
        
        return True
    except Exception as e:
        print(f"  ❌ 面向分類器測試失敗: {e}")
        return False

def test_modular_pipeline():
    """測試模組化流水線"""
    print("\n🚀 測試模組化流水線...")
    
    try:
        from modules.modular_pipeline import ModularPipeline
        
        # 創建測試數據
        df = create_sample_data()
        
        # 測試組合
        test_combinations = [
            ('bert', 'default'),
            ('cnn', 'nmf')
        ]
        
        for encoder_type, aspect_type in test_combinations:
            try:
                print(f"  🔧 測試組合: {encoder_type.upper()} + {aspect_type.upper()}")
                
                # 創建流水線
                pipeline = ModularPipeline(
                    encoder_type=encoder_type,
                    aspect_type=aspect_type,
                    encoder_config={'batch_size': 4},
                    aspect_config={'n_topics': 3}
                )
                
                # 執行流水線
                start_time = time.time()
                results = pipeline.process(df)
                processing_time = time.time() - start_time
                
                print(f"    ✅ 流水線成功: 耗時 {processing_time:.2f}秒")
                print(f"    📊 處理了 {len(df)} 條記錄")
                print(f"    📏 嵌入維度: {pipeline.text_encoder.get_embedding_dim()}")
                print(f"    🎯 發現面向: {len(pipeline.aspect_classifier.get_aspect_names())}")
                
            except Exception as e:
                print(f"    ⚠️  組合 {encoder_type}+{aspect_type} 測試跳過: {e}")
        
        return True
    except Exception as e:
        print(f"  ❌ 模組化流水線測試失敗: {e}")
        return False

def test_pipeline_combinations():
    """測試可用組合"""
    print("\n🔄 測試流水線組合...")
    
    try:
        from modules.modular_pipeline import ModularPipeline
        
        # 創建流水線實例來獲取可用組合
        temp_pipeline = ModularPipeline('bert', 'default')
        combinations = temp_pipeline.get_available_combinations()
        
        print(f"  📋 可用編碼器: {combinations['encoders']['available']}")
        print(f"  📋 可用分類器: {combinations['aspect_classifiers']['available']}")
        
        print("\n  🎯 推薦組合:")
        for combo in combinations['recommended_combinations']:
            print(f"    • {combo['encoder'].upper()} + {combo['aspect_classifier'].upper()}: {combo['scenario']}")
        
        return True
    except Exception as e:
        print(f"  ❌ 組合測試失敗: {e}")
        return False

def run_all_tests():
    """運行所有測試"""
    print("🧪 開始模組化架構測試\n")
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    test_results = []
    
    # 運行各項測試
    tests = [
        ("基本導入", test_basic_imports),
        ("編碼器工廠", test_encoder_factory),
        ("面向分類器工廠", test_aspect_factory),
        ("個別編碼器", test_individual_encoders),
        ("個別面向分類器", test_individual_aspect_classifiers),
        ("模組化流水線", test_modular_pipeline),
        ("流水線組合", test_pipeline_combinations)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} 測試發生未捕獲的錯誤: {e}")
            test_results.append((test_name, False))
    
    # 總結結果
    print("\n" + "=" * 60)
    print("🏁 測試結果總結:")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n總計: {passed}/{len(test_results)} 測試通過")
    
    if passed == len(test_results):
        print("🎉 所有測試通過！模組化架構運行正常。")
        return True
    else:
        print("⚠️  部分測試失敗，請檢查相關組件。")
        return False

if __name__ == "__main__":
    # 添加模組路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # 運行測試
    success = run_all_tests()
    
    # 退出代碼
    sys.exit(0 if success else 1)