"""
基礎接口定義模組
定義文本編碼器和面向分類器的抽象基類
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np

class BaseTextEncoder(ABC):
    """文本編碼器抽象基類"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        """
        初始化編碼器
        
        Args:
            config: 配置參數字典
            progress_callback: 進度回調函數
        """
        self.config = config or {}
        self.progress_callback = progress_callback
        self.encoder_name = self.__class__.__name__
        
    @abstractmethod
    def encode(self, texts: Union[pd.Series, List[str]]) -> np.ndarray:
        """
        將文本編碼為向量
        
        Args:
            texts: 要編碼的文本
            
        Returns:
            np.ndarray: 編碼後的特徵向量 (N, feature_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        獲取嵌入向量維度
        
        Returns:
            int: 向量維度
        """
        pass
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """
        獲取編碼器信息
        
        Returns:
            Dict: 編碼器詳細信息
        """
        return {
            'name': self.encoder_name,
            'embedding_dim': self.get_embedding_dim(),
            'config': self.config
        }


class BaseAspectClassifier(ABC):
    """面向分類器抽象基類"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        """
        初始化面向分類器
        
        Args:
            config: 配置參數字典
            progress_callback: 進度回調函數
        """
        self.config = config or {}
        self.progress_callback = progress_callback
        self.classifier_name = self.__class__.__name__
        
    @abstractmethod
    def fit_transform(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        訓練模型並轉換為面向向量
        
        Args:
            embeddings: 文本嵌入向量 (N, feature_dim)
            metadata: 文本元數據
            
        Returns:
            Tuple[np.ndarray, Dict]: (面向向量, 分析結果)
        """
        pass
    
    @abstractmethod
    def get_aspect_names(self) -> List[str]:
        """
        獲取面向名稱列表
        
        Returns:
            List[str]: 面向名稱
        """
        pass
    
    def get_classifier_info(self) -> Dict[str, Any]:
        """
        獲取分類器信息
        
        Returns:
            Dict: 分類器詳細信息
        """
        return {
            'name': self.classifier_name,
            'aspect_count': len(self.get_aspect_names()),
            'config': self.config
        }


class BasePipeline(ABC):
    """流水線處理基類"""
    
    def __init__(self, 
                 text_encoder: BaseTextEncoder,
                 aspect_classifier: BaseAspectClassifier,
                 config: Optional[Dict] = None):
        """
        初始化流水線
        
        Args:
            text_encoder: 文本編碼器
            aspect_classifier: 面向分類器
            config: 配置參數
        """
        self.text_encoder = text_encoder
        self.aspect_classifier = aspect_classifier
        self.config = config or {}
        
    @abstractmethod
    def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        執行完整的處理流水線
        
        Args:
            data: 輸入數據
            
        Returns:
            Dict: 處理結果
        """
        pass
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        獲取流水線配置信息
        
        Returns:
            Dict: 流水線信息
        """
        return {
            'text_encoder': self.text_encoder.get_encoder_info(),
            'aspect_classifier': self.aspect_classifier.get_classifier_info(),
            'config': self.config
        }