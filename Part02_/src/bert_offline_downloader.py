"""
BERT模型離線下載工具
這個腳本用於預先下載BERT模型，以便在離線環境中使用。
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModel

def download_model(model_name, save_dir):
    """
    下載並保存BERT模型
    
    Args:
        model_name: 要下載的模型名稱
        save_dir: 保存目錄
    """
    print(f"正在下載模型 {model_name}...")
    
    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)
    
    # 下載分詞器
    print("下載分詞器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    
    # 下載模型
    print("下載模型...")
    model = AutoModel.from_pretrained(model_name)
    model_path = os.path.join(save_dir, "model")
    model.save_pretrained(model_path)
    
    print(f"模型已保存至 {save_dir}")
    print("使用方法:")
    print(f"tokenizer = AutoTokenizer.from_pretrained('{tokenizer_path}')")
    print(f"model = AutoModel.from_pretrained('{model_path}')")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下載BERT模型用於離線使用")
    parser.add_argument("--model", type=str, default="bert-base-uncased", 
                       help="要下載的模型名稱，例如'bert-base-uncased'或'bert-base-chinese'")
    parser.add_argument("--output", type=str, default="./bert_models",
                       help="保存模型的目錄")
    
    args = parser.parse_args()
    
    # 下載模型
    download_model(args.model, os.path.join(args.output, args.model))