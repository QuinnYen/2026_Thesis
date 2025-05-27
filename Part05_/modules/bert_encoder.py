import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import logging
import os
from tqdm import tqdm
from .run_manager import RunManager

class BertEncoder:
    """BERT編碼器 - 使用BERT模型提取文本的768維特徵向量"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', output_dir: Optional[str] = None):
        """
        初始化BERT編碼器
        
        Args:
            model_name: 預訓練模型名稱，預設使用 'bert-base-uncased'
            output_dir: 輸出目錄的基礎路徑，如果為None則使用當前目錄
        """
        self.logger = logging.getLogger(__name__)
        
        # 設置輸出目錄
        if output_dir is None:
            output_dir = os.path.dirname(os.path.dirname(__file__))
        
        # 使用RunManager管理輸出目錄
        self.run_manager = RunManager(output_dir)
        self.output_dir = self.run_manager.get_run_dir()
        self.logger.info(f"本次執行的輸出目錄：{self.output_dir}")
        
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