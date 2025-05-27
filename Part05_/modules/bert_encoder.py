import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import logging
import os
from tqdm import tqdm

class BertEncoder:
    """BERT編碼器 - 使用BERT模型提取文本的768維特徵向量"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', output_dir: str = "D:\\Project\\2026_Thesis\\Part05_\\output"):
        """
        初始化BERT編碼器
        
        Args:
            model_name: 預訓練模型名稱，預設使用 'bert-base-uncased'
            output_dir: 輸出目錄的基礎路徑
        """
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_dir
        
        # 設置設備（GPU/CPU）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用設備: {self.device}")
        
        # 載入BERT模型和分詞器
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # 設置為評估模式
            self.logger.info(f"成功載入BERT模型: {model_name}")
        except Exception as e:
            self.logger.error(f"載入BERT模型時發生錯誤: {str(e)}")
            raise
    
    def encode(self, texts: Union[str, List[str], pd.Series], batch_size: int = 32) -> np.ndarray:
        """
        使用BERT模型編碼文本，提取[CLS]標記的768維特徵向量
        
        Args:
            texts: 輸入文本，可以是單個字符串、字符串列表或pandas Series
            batch_size: 批次大小，用於處理大量數據
            
        Returns:
            np.ndarray: 特徵向量矩陣，shape為(n_samples, 768)
        """
        # 將輸入轉換為列表格式
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # 初始化結果列表
        all_embeddings = []
        
        # 使用tqdm顯示進度
        for i in tqdm(range(0, len(texts), batch_size), desc="編碼進度"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # 對批次文本進行分詞
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # 將輸入移到指定設備
                encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                
                # 使用無梯度計算
                with torch.no_grad():
                    # 獲取模型輸出
                    outputs = self.model(**encoded_input)
                    # 提取[CLS]標記的輸出（最後一層的第一個token）
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(batch_embeddings)
                
            except Exception as e:
                self.logger.error(f"處理批次 {i//batch_size + 1} 時發生錯誤: {str(e)}")
                raise
        
        # 合併所有批次的結果
        embeddings = np.vstack(all_embeddings)
        
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, output_file: str):
        """
        保存特徵向量到文件
        
        Args:
            embeddings: 特徵向量矩陣
            output_file: 輸出文件名
        """
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 構建完整的輸出路徑
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            # 保存為numpy格式
            np.save(output_path, embeddings)
            self.logger.info(f"特徵向量已保存至: {output_path}")
            
            # 同時保存一個CSV版本以便查看
            df = pd.DataFrame(embeddings, columns=[f'feature_{i+1}' for i in range(embeddings.shape[1])])
            csv_path = output_path.replace('.npy', '.csv')
            df.to_csv(csv_path, index=False)
            self.logger.info(f"特徵向量的CSV版本已保存至: {csv_path}")
            
        except Exception as e:
            self.logger.error(f"保存特徵向量時發生錯誤: {str(e)}")
            raise 