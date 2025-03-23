import pandas as pd
import numpy as np
import re
import os
import nltk
import logging
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_import')

class DataImporter:
    def __init__(self, output_dir='./data_processed'):
        """
        初始化數據導入器
        
        Args:
            output_dir: 處理後數據的保存目錄
        """
        self.output_dir = output_dir
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 下載NLTK資源（如果尚未下載）
        self._download_nltk_resources()
        
        # 初始化NLTK工具
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def _download_nltk_resources(self):
        """下載NLTK所需資源"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("正在下載NLTK資源...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
    
    def import_data(self, file_path, callback=None):
        """
        導入數據並進行基礎處理
        
        Args:
            file_path: 要導入的文件路徑
            callback: 回調函數，用於更新進度
            
        Returns:
            processed_data_path: 處理後數據的保存路徑
        """
        logger.info(f"開始導入數據: {file_path}")
        
        # 確定文件類型並讀取
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if callback:
            callback("正在讀取文件...", 10)
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.json':
                df = pd.read_json(file_path, lines=True)
            elif file_extension == '.txt':
                # 假設文本文件每行是一條評論
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                df = pd.DataFrame({'text': lines})
            else:
                raise ValueError(f"不支持的文件類型: {file_extension}")
            
            logger.info(f"成功讀取數據，共 {len(df)} 行")
            
            # 檢查並處理常見的評論數據列名
            if callback:
                callback("識別數據結構...", 20)
            
            # 嘗試識別評論文本列
            text_column = None
            for col in ['text', 'review_text', 'reviewText', 'content', 'comment', 'review']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None and len(df.columns) == 1:
                # 如果只有一列，則假設它是評論文本
                text_column = df.columns[0]
            
            if text_column is None:
                logger.warning("無法自動識別評論文本列，使用第一列作為評論文本")
                text_column = df.columns[0]
            
            logger.info(f"使用 '{text_column}' 作為評論文本列")
            
            # 確保有評論文本列
            if text_column not in df.columns:
                raise ValueError(f"找不到評論文本列: {text_column}")
            
            # 清洗文本
            if callback:
                callback("正在清洗文本...", 40)
            
            df['clean_text'] = df[text_column].apply(self._clean_text)
            logger.info("文本清洗完成")
            
            # 進行分詞和停用詞過濾
            if callback:
                callback("正在進行分詞和詞形還原...", 70)
            
            df['tokens'] = df['clean_text'].apply(self._tokenize_and_lemmatize)
            logger.info("分詞和詞形還原完成")
            
            # 保存處理後的數據
            processed_data_path = os.path.join(
                self.output_dir, 
                f"processed_{os.path.basename(file_path)}"
            )
            
            if callback:
                callback("正在保存處理後的數據...", 90)
            
            df.to_csv(processed_data_path, index=False)
            logger.info(f"處理後的數據已保存至: {processed_data_path}")
            
            if callback:
                callback("數據導入和基礎處理完成", 100)
            
            return processed_data_path
            
        except Exception as e:
            logger.error(f"導入數據時出錯: {str(e)}")
            if callback:
                callback(f"錯誤: {str(e)}", -1)
            raise
    
    def _clean_text(self, text):
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            cleaned_text: 清洗後的文本
        """
        if pd.isna(text):
            return ""
        
        # 轉換為字符串
        text = str(text)
        
        # 移除HTML標籤
        text = BeautifulSoup(text, "html.parser").get_text()
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除特殊字符和數字
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # 移除多餘的空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize_and_lemmatize(self, text):
        """
        對文本進行分詞、移除停用詞和詞形還原
        
        Args:
            text: 清洗後的文本
            
        Returns:
            tokens: 處理後的詞列表
        """
        # 分詞
        tokens = word_tokenize(text.lower())
        
        # 移除停用詞和詞形還原
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word.isalpha() and word not in self.stop_words
        ]
        
        return tokens

# 示例用法
if __name__ == "__main__":
    importer = DataImporter()
    
    def print_progress(message, percentage):
        print(f"{message} ({percentage}%)")
    
    try:
        processed_file = importer.import_data("sample_reviews.csv", callback=print_progress)
        print(f"處理後的文件保存在: {processed_file}")
    except Exception as e:
        print(f"發生錯誤: {str(e)}")