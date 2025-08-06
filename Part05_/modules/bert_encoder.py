import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import logging
import os
import sys
from tqdm import tqdm
from .run_manager import RunManager

# 添加父目錄到路徑以導入config模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import get_path_config
from utils.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class BertEncoder:
    """BERT編碼器 - 使用BERT模型提取文本的768維特徵向量"""
    
    def __init__(self, output_dir: Optional[str] = None, progress_callback=None):
        """
        初始化BERT編碼器
        
        Args:
            output_dir: 輸出目錄路徑，如果為None則使用預設路徑
            progress_callback: 進度回調函數，用於更新GUI進度
        """
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logger.info(f"使用設備: {self.device}")
        
        # 初始化儲存管理器
        self.storage_manager = StorageManager(output_dir) if output_dir else None
        
        # 通知GPU狀態
        if self.progress_callback:
            gpu_info = f"使用設備: {self.device}"
            if torch.cuda.is_available():
                gpu_info += f" ({torch.cuda.get_device_name()})"
            self.progress_callback('status', gpu_info)
    
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
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # 簡化輸出 - 只在批次較多時顯示進度
        if total_batches > 10:
            print(f"   編碼 {len(texts)} 條文本 (共 {total_batches} 批次)")
        
        # 通知開始編碼
        if self.progress_callback:
            self.progress_callback('phase', {
                'phase_name': 'BERT編碼',
                'current_phase': 1,
                'total_phases': 3
            })
            self.progress_callback('status', f"開始BERT編碼：{len(texts)} 條文本，{total_batches} 個批次")
        
        # 使用進度條或回調顯示進度
        if self.progress_callback:
            # 使用進度回調而非tqdm
            for i in range(0, len(texts), batch_size):
                batch_num = i // batch_size + 1
                
                # 更新進度
                self.progress_callback('progress', (batch_num, total_batches))
                
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
        else:
            # 沒有回調時使用原始tqdm
            for i in tqdm(range(0, len(texts), batch_size), 
                         desc="BERT編碼進度", 
                         unit="batch", 
                         total=total_batches):
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
        # 移除詳細日誌，在主程式中已有輸出
        
        # 通知保存階段
        if self.progress_callback:
            self.progress_callback('phase', {
                'phase_name': '保存特徵向量',
                'current_phase': 2,
                'total_phases': 3
            })
        
        # 保存特徵向量到02_bert_encoding目錄
        if self.output_dir:
            if self.storage_manager:
                # 使用儲存管理器保存特徵向量
                metadata = {
                    'model_name': self.model_name,
                    'device': str(self.device),
                    'batch_size': batch_size
                }
                output_file = self.storage_manager.save_embeddings(embeddings, 'bert', metadata)
                logger.info(f"✅ 已通過儲存管理器保存BERT特徵向量：{output_file}")
            else:
                # 原有的保存方式
                path_config = get_path_config()
                bert_encoding_subdir = path_config.get_subdirectory_name("bert_encoding")
                
                if bert_encoding_subdir in self.output_dir:
                    bert_encoding_dir = self.output_dir
                else:
                    bert_encoding_dir = os.path.join(self.output_dir, bert_encoding_subdir)
                
                os.makedirs(bert_encoding_dir, exist_ok=True)
                filename = path_config.get_file_pattern("bert_embeddings")
                output_file = os.path.join(bert_encoding_dir, filename)
                np.save(output_file, embeddings)
                logger.info(f"已保存BERT特徵向量到：{output_file}")
            
            if self.progress_callback:
                self.progress_callback('status', f"特徵向量已保存：{output_file}")
        
        # 完成編碼
        if self.progress_callback:
            self.progress_callback('phase', {
                'phase_name': 'BERT編碼完成',
                'current_phase': 3,
                'total_phases': 3
            })
            self.progress_callback('progress', 100)
        
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