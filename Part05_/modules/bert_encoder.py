import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import logging
import os
from tqdm import tqdm
from .run_manager import RunManager

logger = logging.getLogger(__name__)

class BertEncoder:
    """BERT編碼器 - 使用BERT模型提取文本的768維特徵向量"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化BERT編碼器
        
        Args:
            output_dir: 輸出目錄路徑，如果為None則使用預設路徑
        """
        self.output_dir = output_dir
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"使用設備: {self.device}")
    
    def encode(self, texts: pd.Series) -> np.ndarray:
        """
        對文本進行BERT編碼
        
        Args:
            texts: 要編碼的文本序列
            
        Returns:
            np.ndarray: 編碼後的特徵向量
        """
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts.iloc[i:i+batch_size].tolist()
            encoded = self.tokenizer(batch_texts, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=512, 
                                   return_tensors='pt')
            
            with torch.no_grad():
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                # 使用[CLS]標記的輸出作為文本表示
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        
        # 保存特徵向量
        if self.output_dir:
            output_file = os.path.join(self.output_dir, "02_bert_embeddings.npy")
            np.save(output_file, embeddings)
            logger.info(f"已保存BERT特徵向量到：{output_file}")
        
        return embeddings
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """
        載入先前保存的BERT特徵向量
        
        Args:
            file_path: .npy檔案的路徑
            
        Returns:
            np.ndarray: 載入的特徵向量矩陣，shape為(n_samples, 768)
        """
        try:
            # 檢查檔案是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到特徵向量檔案：{file_path}")
            
            # 載入.npy檔案
            embeddings = np.load(file_path)
            self.logger.info(f"成功載入特徵向量，形狀為：{embeddings.shape}")
            
            # 檢查維度是否正確（BERT base模型輸出為768維）
            if embeddings.shape[1] != 768:
                raise ValueError(f"特徵向量維度不正確：預期為768，實際為{embeddings.shape[1]}")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"載入特徵向量時發生錯誤：{str(e)}")
            raise
    
    def quick_load_embeddings(self, file_path: str) -> np.ndarray:
        """
        快速載入.npy特徵向量檔案，不進行額外檢查
        
        Args:
            file_path: .npy檔案的完整路徑
            
        Returns:
            np.ndarray: 載入的特徵向量矩陣
        """
        return np.load(file_path)
    
    def save_embeddings(self, embeddings: np.ndarray, output_file: str):
        """
        保存特徵向量到文件
        
        Args:
            embeddings: 特徵向量矩陣
            output_file: 輸出文件名
        """
        try:
            # 構建完整的輸出路徑
            output_path = os.path.join(self.output_dir, output_file)
            
            # 確保輸出目錄存在且可寫
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 檢查檔案是否已存在，如果存在則先刪除
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception as e:
                    self.logger.warning(f"無法刪除現有檔案 {output_path}: {str(e)}")
            
            # 保存為numpy格式
            try:
                np.save(output_path, embeddings)
                self.logger.info(f"特徵向量已保存至: {output_path}")
            except Exception as e:
                self.logger.error(f"保存特徵向量時發生錯誤: {str(e)}")
                raise
            
        except Exception as e:
            self.logger.error(f"保存特徵向量時發生錯誤: {str(e)}")
            raise 