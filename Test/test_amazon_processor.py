#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
測試Amazon數據處理功能
"""

import os
import sys
import pandas as pd
import time
import json
from datetime import datetime

# 添加上級目錄到模塊搜索路徑，以便能夠導入Part04_中的模塊
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
part04_dir = os.path.join(parent_dir, 'Part04_')

if part04_dir not in sys.path:
    sys.path.append(part04_dir)

# 導入Amazon處理器
from modules.data_processor import AmazonProcessor

# 輸出分隔線函數
def print_separator(title=None):
    separator = "=" * 60
    if title:
        print(f"\n{separator}\n{title}\n{separator}")
    else:
        print(f"\n{separator}\n")

def process_large_jsonl_file(file_path, sample_size=1000):
    """處理大型JSONL文件，採用分塊讀取的方式
    
    Args:
        file_path: JSONL文件路徑
        sample_size: 要抽取的樣本數量
        
    Returns:
        pd.DataFrame: 包含抽樣數據的DataFrame
    """
    import random
    print(f"開始處理大型JSONL文件: {file_path}")
    print(f"目標樣本大小: {sample_size}")
    
    # 設定隨機種子確保結果可重現
    random.seed(42)
    
    # 採用水塘抽樣算法
    samples = []
    total_count = 0
    
    try:
        # 打開文件並逐行讀取
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i % 10000 == 0 and i > 0:
                    print(f"已處理 {i:,} 行，目前樣本數: {len(samples)}")
                
                total_count += 1
                
                try:
                    # 解析JSON行
                    record = json.loads(line.strip())
                    
                    # 水塘抽樣算法
                    if len(samples) < sample_size:
                        samples.append(record)
                    else:
                        # 以樣本大小/當前處理行數的概率替換現有樣本
                        j = random.randint(0, total_count - 1)
                        if j < sample_size:
                            samples[j] = record
                    
                    # 如果已經處理了足夠多的行數，提前結束
                    if total_count >= sample_size * 10 and len(samples) == sample_size:
                        print(f"已收集足夠樣本，提前結束處理。處理了 {total_count:,} 行")
                        break
                        
                except json.JSONDecodeError:
                    # 忽略無法解析的行
                    continue
                    
                except Exception as e:
                    print(f"處理行 {i} 時出錯: {str(e)}")
                    continue
                    
    except Exception as e:
        print(f"讀取文件時出錯: {str(e)}")
        if len(samples) == 0:
            raise ValueError("無法從文件中提取任何有效記錄")
    
    # 創建DataFrame
    print(f"共處理了 {total_count:,} 行，抽取了 {len(samples)} 個樣本")
    df = pd.DataFrame(samples)
    
    return df

def main():
    print_separator("Amazon數據處理測試")
    
    # 顯示時間戳
    print(f"測試開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 初始化處理器
    print("初始化Amazon數據處理器...")
    processor = AmazonProcessor()
    
    # 準備輸入和輸出路徑
    amazon_file = os.path.join(parent_dir, 'ReviewsDataBase', 'Amazon', 'Electronics_5.json')
    
    # 創建輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(part04_dir, 'output', 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'amazon_processed_{timestamp}.csv')
    
    # 顯示文件信息
    print(f"輸入文件: {amazon_file}")
    print(f"輸出文件: {output_file}")
    
    # 檢查輸入文件是否存在
    if not os.path.exists(amazon_file):
        print(f"錯誤: 輸入文件不存在: {amazon_file}")
        return
    
    # 記錄開始時間
    start_time = time.time()
    
    try:
        # 限制樣本大小以加快測試速度
        sample_size = 1000
        print(f"處理文件中，限制樣本大小為: {sample_size}...")
        
        # 直接處理大型JSONL文件
        print("檢測到大型JSONL文件，使用優化的讀取方法...")
        df_raw = process_large_jsonl_file(amazon_file, sample_size)
        
        # 使用處理器的預處理方法進行後續處理
        print(f"原始數據載入完成。開始預處理，資料大小: {len(df_raw)} 行 x {len(df_raw.columns)} 列")
        df = processor.preprocess(df_raw, text_column='reviewText')
        
        # 保存處理後的結果
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(f"保存處理結果到: {output_file}")
            df.to_csv(output_file, index=False)
        
        # 顯示處理結果
        print_separator("處理結果統計")
        print(f"處理後資料列數: {len(df)}")
        print(f"處理後資料欄位: {', '.join(df.columns.tolist())}")
        
        # 顯示評分分佈
        if 'rating' in df.columns:
            rating_counts = df['rating'].value_counts().sort_index()
            print("\n評分分佈:")
            for rating, count in rating_counts.items():
                print(f"  {rating} 星: {count} 條評論 ({count/len(df)*100:.1f}%)")
        
        # 顯示文本長度統計
        if 'clean_text' in df.columns:
            text_lengths = df['clean_text'].str.len()
            print("\n清洗後文本長度統計:")
            print(f"  最短: {text_lengths.min()} 字元")
            print(f"  最長: {text_lengths.max()} 字元")
            print(f"  平均: {text_lengths.mean():.1f} 字元")
            print(f"  中位數: {text_lengths.median()} 字元")
        
        # 顯示洗清後的文本樣本
        print("\n處理後文本樣本:")
        for i, text in enumerate(df['clean_text'].head(3).tolist()):
            if len(text) > 300:
                text = text[:300] + "..."
            print(f"  樣本 {i+1}: {text}")
        
        # 顯示分詞結果樣本
        if 'tokens' in df.columns:
            print("\n分詞後樣本:")
            for i, tokens in enumerate(df['tokens'].head(3).tolist()):
                if len(tokens) > 20:
                    tokens_sample = tokens[:20] + ['...']
                else:
                    tokens_sample = tokens
                print(f"  樣本 {i+1}: {tokens_sample}")
    
        # 顯示結果保存路徑
        print(f"\n處理後的數據已保存至: {output_file}")
        
    except Exception as e:
        import traceback
        print(f"處理過程中發生錯誤:\n{str(e)}")
        print(traceback.format_exc())
    
    # 計算處理時間
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print_separator("測試完成")
    print(f"總處理時間: {int(minutes)}分{int(seconds)}秒")
    print(f"測試結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()