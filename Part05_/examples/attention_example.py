#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力機制使用範例
展示如何使用不同的注意力機制進行情感分析
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# 添加父目錄到路徑
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from modules.attention_processor import AttentionProcessor
from modules.attention_mechanism import create_attention_mechanism, apply_attention_mechanism

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def create_sample_data():
    """創建示例數據"""
    sample_texts = [
        "這個產品真的很棒！質量非常好，我很滿意",
        "我喜歡這個商品，設計很漂亮，功能也不錯",
        "還可以吧，沒有特別好也沒有特別差",
        "普通的產品，價格合理，但沒什麼亮點",
        "這個產品很糟糕，質量很差，我很失望",
        "完全不推薦，浪費錢，服務態度也很差",
        "Great product! Love the quality and design.",
        "Amazing features, highly recommend to everyone.",
        "It's okay, nothing special but does the job.",
        "Terrible quality, very disappointed with purchase."
    ]
    
    sample_sentiments = [
        'positive', 'positive', 'neutral', 'neutral', 
        'negative', 'negative', 'positive', 'positive', 
        'neutral', 'negative'
    ]
    
    return pd.DataFrame({
        'text': sample_texts,
        'sentiment': sample_sentiments
    })

def demo_individual_attention_mechanisms():
    """演示單個注意力機制的使用"""
    print("=" * 60)
    print("演示單個注意力機制")
    print("=" * 60)
    
    # 創建示例數據
    df = create_sample_data()
    print(f"創建了 {len(df)} 條示例數據")
    
    # 模擬BERT特徵向量（實際使用時會從BERT編碼器獲取）
    np.random.seed(42)
    embeddings = np.random.randn(len(df), 768)  # 768維BERT特徵
    print(f"模擬特徵向量形狀: {embeddings.shape}")
    
    # 測試不同的注意力機制
    attention_types = ['no', 'similarity', 'keyword', 'self']
    
    results = {}
    for attention_type in attention_types:
        print(f"\n--- 測試 {attention_type} 注意力機制 ---")
        
        try:
            # 應用注意力機制
            result = apply_attention_mechanism(
                attention_type=attention_type,
                embeddings=embeddings,
                metadata=df
            )
            
            results[attention_type] = result
            
            # 輸出結果
            metrics = result['metrics']
            print(f"內聚度: {metrics['coherence']:.4f}")
            print(f"分離度: {metrics['separation']:.4f}")
            print(f"綜合得分: {metrics['combined_score']:.4f}")
            
            # 顯示面向向量
            aspect_vectors = result['aspect_vectors']
            print(f"生成的面向向量數量: {len(aspect_vectors)}")
            for aspect, vector in aspect_vectors.items():
                print(f"  {aspect}: 向量維度 {vector.shape}")
                
        except Exception as e:
            print(f"錯誤: {str(e)}")
    
    return results

def demo_combined_attention():
    """演示組合注意力機制"""
    print("\n" + "=" * 60)
    print("演示組合注意力機制")
    print("=" * 60)
    
    # 創建示例數據
    df = create_sample_data()
    
    # 模擬BERT特徵向量
    np.random.seed(42)
    embeddings = np.random.randn(len(df), 768)
    
    # 測試不同的組合權重配置
    weight_configs = [
        {'similarity': 0.33, 'keyword': 0.33, 'self': 0.34},  # 均衡配置
        {'similarity': 0.5, 'keyword': 0.3, 'self': 0.2},     # 偏重相似度
        {'similarity': 0.2, 'keyword': 0.6, 'self': 0.2},     # 偏重關鍵詞
        {'similarity': 0.2, 'keyword': 0.2, 'self': 0.6},     # 偏重自注意力
    ]
    
    config_names = ['均衡配置', '偏重相似度', '偏重關鍵詞', '偏重自注意力']
    
    for i, (weights, name) in enumerate(zip(weight_configs, config_names)):
        print(f"\n--- {name} ---")
        print(f"權重配置: {weights}")
        
        try:
            # 應用組合注意力機制
            result = apply_attention_mechanism(
                attention_type='combined',
                embeddings=embeddings,
                metadata=df,
                weights=weights
            )
            
            # 輸出結果
            metrics = result['metrics']
            print(f"內聚度: {metrics['coherence']:.4f}")
            print(f"分離度: {metrics['separation']:.4f}")
            print(f"綜合得分: {metrics['combined_score']:.4f}")
            
        except Exception as e:
            print(f"錯誤: {str(e)}")

def demo_attention_processor():
    """演示完整的注意力處理器功能"""
    print("\n" + "=" * 60)
    print("演示注意力處理器")
    print("=" * 60)
    
    # 創建示例數據文件
    df = create_sample_data()
    sample_file = "sample_data.csv"
    df.to_csv(sample_file, index=False)
    print(f"創建示例數據文件: {sample_file}")
    
    try:
        # 初始化注意力處理器
        output_dir = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        processor = AttentionProcessor(output_dir=output_dir)
        
        print(f"輸出目錄: {output_dir}")
        
        # 執行完整的注意力分析
        print("\n執行完整的注意力機制分析...")
        results = processor.process_with_attention(
            input_file=sample_file,
            attention_types=['no', 'similarity', 'combined'],
            save_results=True
        )
        
        # 顯示比較結果
        if 'comparison' in results:
            comparison = results['comparison']
            print("\n比較結果:")
            
            if 'summary' in comparison:
                summary = comparison['summary']
                print(f"最佳機制: {summary.get('best_mechanism', 'N/A')}")
                print(f"最佳得分: {summary.get('best_score', 0):.4f}")
            
            if 'combined_ranking' in comparison:
                print("\n綜合得分排名:")
                for i, (mechanism, score) in enumerate(comparison['combined_ranking'], 1):
                    print(f"  {i}. {mechanism}: {score:.4f}")
        
        # 執行專門的比較分析
        print("\n執行專門的注意力機制比較...")
        comparison_results = processor.compare_attention_mechanisms(
            input_file=sample_file,
            attention_types=['no', 'similarity', 'keyword', 'self', 'combined']
        )
        
        # 顯示詳細報告
        if 'detailed_report' in comparison_results:
            report = comparison_results['detailed_report']
            print("\n詳細報告:")
            print(f"推薦使用: {report['summary'].get('best_overall', 'N/A')}")
            
            if 'recommendations' in report:
                print("\n建議:")
                for recommendation in report['recommendations']:
                    print(f"  - {recommendation}")
        
        print(f"\n結果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"錯誤: {str(e)}")
    
    finally:
        # 清理示例文件
        if os.path.exists(sample_file):
            os.remove(sample_file)
            print(f"清理示例文件: {sample_file}")

def demo_attention_with_keywords():
    """演示使用關鍵詞的注意力機制"""
    print("\n" + "=" * 60)
    print("演示關鍵詞注意力機制")
    print("=" * 60)
    
    # 創建示例數據
    df = create_sample_data()
    
    # 模擬BERT特徵向量
    np.random.seed(42)
    embeddings = np.random.randn(len(df), 768)
    
    # 定義關鍵詞
    topic_keywords = {
        'positive': ['好', '棒', '滿意', '喜歡', 'great', 'amazing', 'love', 'quality'],
        'negative': ['糟糕', '差', '失望', '不推薦', 'terrible', 'disappointed', 'worst'],
        'neutral': ['還可以', '普通', '一般', 'okay', 'nothing', 'average']
    }
    
    print("使用的關鍵詞:")
    for sentiment, keywords in topic_keywords.items():
        print(f"  {sentiment}: {', '.join(keywords)}")
    
    try:
        # 應用關鍵詞注意力機制
        result = apply_attention_mechanism(
            attention_type='keyword',
            embeddings=embeddings,
            metadata=df,
            topic_keywords=topic_keywords
        )
        
        # 輸出結果
        metrics = result['metrics']
        print(f"\n關鍵詞注意力機制結果:")
        print(f"內聚度: {metrics['coherence']:.4f}")
        print(f"分離度: {metrics['separation']:.4f}")
        print(f"綜合得分: {metrics['combined_score']:.4f}")
        
        # 比較與無關鍵詞的情況
        result_no_keywords = apply_attention_mechanism(
            attention_type='keyword',
            embeddings=embeddings,
            metadata=df
        )
        
        metrics_no_keywords = result_no_keywords['metrics']
        print(f"\n無關鍵詞情況下的結果:")
        print(f"內聚度: {metrics_no_keywords['coherence']:.4f}")
        print(f"分離度: {metrics_no_keywords['separation']:.4f}")
        print(f"綜合得分: {metrics_no_keywords['combined_score']:.4f}")
        
        print(f"\n關鍵詞的影響:")
        print(f"內聚度提升: {metrics['coherence'] - metrics_no_keywords['coherence']:.4f}")
        print(f"分離度提升: {metrics['separation'] - metrics_no_keywords['separation']:.4f}")
        print(f"綜合得分提升: {metrics['combined_score'] - metrics_no_keywords['combined_score']:.4f}")
        
    except Exception as e:
        print(f"錯誤: {str(e)}")

def main():
    """主函數"""
    print("BERT情感分析系統 - 注意力機制使用範例")
    print("本範例展示如何使用各種注意力機制進行情感分析")
    
    try:
        # 演示1: 單個注意力機制
        demo_individual_attention_mechanisms()
        
        # 演示2: 組合注意力機制
        demo_combined_attention()
        
        # 演示3: 關鍵詞注意力機制
        demo_attention_with_keywords()
        
        # 演示4: 完整的注意力處理器
        demo_attention_processor()
        
        print("\n" + "=" * 60)
        print("所有演示完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 