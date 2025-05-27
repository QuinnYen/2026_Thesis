import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import unicodedata
from typing import Union, List, Optional
import sys
import os
from datetime import datetime
from .run_manager import RunManager

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

class TextPreprocessor:
    """文本預處理類，提供完整的文本清理和預處理功能"""
    
    def __init__(self, language: str = 'english', output_dir: str = "D:\\Project\\2026_Thesis\\Part05_"):
        """
        初始化文本預處理器
        
        Args:
            language: 文本語言，預設為英文
            output_dir: 輸出目錄的基礎路徑
        """
        self.language = language
        self.logger = logging.getLogger(__name__)
        
        # 使用RunManager管理輸出目錄
        self.run_manager = RunManager(output_dir)
        self.run_dir = self.run_manager.get_run_dir()
        self.logger.info(f"本次執行的輸出目錄：{self.run_dir}")
        
        # 初始化NLTK組件
        try:
            # 檢查並下載必要的NLTK資源
            required_resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
            for resource in required_resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    print(f"正在下載 NLTK 資源: {resource}")
                    nltk.download(resource, quiet=True)
            
            self.stop_words = set(stopwords.words(language))
            self.lemmatizer = WordNetLemmatizer()
            print("NLTK 資源初始化完成")
            
        except Exception as e:
            print(f"NLTK 初始化錯誤: {str(e)}")
            print("請確保網路連接正常，並重新執行程式")
            raise
    
    def preprocess(self, data: Union[pd.DataFrame, str], text_column: Optional[str] = None) -> Union[pd.DataFrame, str]:
        """
        預處理文本數據
        
        Args:
            data: 輸入數據，可以是DataFrame或字串
            text_column: 如果輸入是DataFrame，指定要處理的文本列名
            
        Returns:
            處理後的數據，格式與輸入相同
        """
        processed_data = None
        
        if isinstance(data, str):
            processed_data = self._process_single_text(data)
            # 保存處理後的單一文本
            self.save_output(processed_data, "processed_text.txt")
        elif isinstance(data, pd.DataFrame):
            processed_data = self._process_dataframe(data, text_column)
            # 保存處理後的DataFrame
            self.save_output(processed_data, "01_processed_dataframe.csv")
        else:
            raise ValueError("輸入數據必須是DataFrame或字串")
            
        # 顯示處理完成訊息
        self.logger.info(f"處理完成，結果已存至：{os.path.basename(self.run_dir)}")
            
        return processed_data
    
    def _process_single_text(self, text: str) -> str:
        """
        處理單個文本
        
        Args:
            text: 輸入文本
            
        Returns:
            處理後的文本
        """
        # 基礎清理
        text = self._basic_clean(text)
        
        # 標準化
        text = self._normalize_text(text)
        
        # 分詞
        tokens = self._tokenize(text)
        
        # 移除停用詞和詞形還原
        tokens = self._remove_stopwords_and_lemmatize(tokens)
        
        # 合併回文本
        return ' '.join(tokens)
    
    def _process_dataframe(self, df: pd.DataFrame, text_column: Optional[str] = None) -> pd.DataFrame:
        """
        處理DataFrame中的文本
        
        Args:
            df: 輸入DataFrame
            text_column: 要處理的文本列名
            
        Returns:
            處理後的DataFrame
        """
        # 複製DataFrame以避免修改原始數據
        df = df.copy()
        
        # 確定文本列
        if text_column is None:
            text_column = self._identify_text_column(df)
        
        if text_column not in df.columns:
            raise ValueError(f"找不到文本列 '{text_column}'")
        
        # 處理缺失值
        df[text_column] = df[text_column].fillna('')
        
        # 應用文本處理
        df['processed_text'] = df[text_column].apply(self._process_single_text)
        
        return df
    
    def _basic_clean(self, text: str) -> str:
        """
        基礎文本清理
        
        Args:
            text: 輸入文本
            
        Returns:
            清理後的文本
        """
        # 轉換為字符串
        text = str(text)
        
        # 移除HTML標籤
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除特殊字符和數字
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # 移除多餘的空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """
        文本標準化
        
        Args:
            text: 輸入文本
            
        Returns:
            標準化後的文本
        """
        # 轉換為小寫
        text = text.lower()
        
        # Unicode標準化
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分詞
        
        Args:
            text: 輸入文本
            
        Returns:
            分詞後的token列表
        """
        return word_tokenize(text)
    
    def _remove_stopwords_and_lemmatize(self, tokens: List[str]) -> List[str]:
        """
        移除停用詞並進行詞形還原
        
        Args:
            tokens: 輸入token列表
            
        Returns:
            處理後的token列表
        """
        # 移除停用詞並進行詞形還原
        return [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
    
    def _identify_text_column(self, df: pd.DataFrame) -> str:
        """
        識別DataFrame中的文本列
        
        Args:
            df: 輸入DataFrame
            
        Returns:
            文本列名
        """
        # 常見的文本列名
        text_column_candidates = ['text', 'content', 'description', 'review', 'comment']
        
        # 尋找第一個匹配的列名
        for column in text_column_candidates:
            if column in df.columns:
                return column
        
        # 如果找不到，使用第一個object類型的列
        object_columns = df.select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            return object_columns[0]
        
        raise ValueError("無法識別文本列")
    
    def save_output(self, data: Union[pd.DataFrame, str], filename: str) -> str:
        """
        將處理後的數據保存到輸出目錄
        
        Args:
            data: 要保存的數據
            filename: 檔案名稱
            
        Returns:
            str: 保存的檔案完整路徑
        """
        output_path = os.path.join(self.run_dir, filename)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False, encoding='utf-8')
        elif isinstance(data, str):
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(data)
        
        self.logger.info(f"已將輸出保存至：{output_path}")
        return output_path 