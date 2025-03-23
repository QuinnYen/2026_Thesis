import pandas as pd
import numpy as np
import re
import os
import nltk
import logging
import traceback
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

class SimpleTokenizer:
    """
    簡單的分詞器，不依賴NLTK的word_tokenize
    """
    def __init__(self):
        # 基本的英文縮寫和標點符號模式
        self.patterns = [
            (r'([a-z])\.([a-z])', r'\1\2'),  # 處理如a.b的情況
            (r'([0-9])\.([0-9])', r'\1\2'),  # 處理如1.2的情況
            (r'([A-Za-z])\'([a-z])', r'\1\2'),  # 處理如don't的情況
            (r'[^\w\s]', ' '),  # 替換所有標點為空格
            (r'\s+', ' ')  # 合併多個空格
        ]
    
    def tokenize(self, text):
        """
        將文本分割成單詞
        
        Args:
            text: 輸入文本
            
        Returns:
            tokens: 分詞後的單詞列表
        """
        # 轉為小寫
        text = text.lower()
        
        # 應用模式替換
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)
        
        # 分割為單詞
        return text.strip().split()

class DataImporter:
    def __init__(self, output_dir='./Part02_/data_processed'):
        """
        初始化數據導入器
        
        Args:
            output_dir: 處理後數據的保存目錄
        """
        self.output_dir = output_dir
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 用於控制警告消息的標誌
        self.tokenize_warning_shown = False
        self.lemmatize_warning_shown = False
        
        # 設置NLTK數據路徑 - 使用絕對路徑
        self._setup_nltk_path()
            
        # 下載NLTK資源（如果尚未下載）
        self._download_nltk_resources()
        
        # 初始化自定義分詞器
        self.tokenizer = SimpleTokenizer()
        
        # 初始化NLTK工具
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            logger.info("NLTK工具初始化成功")
        except Exception as e:
            logger.error(f"NLTK工具初始化失敗: {str(e)}")
            # 創建一個備用停用詞列表
            self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 
                          'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 'on', 'is', 
                          'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}
            # 嘗試使用一個備用詞形還原函數
            class SimpleLemmatizer:
                def lemmatize(self, word):
                    return word
            self.lemmatizer = SimpleLemmatizer()
    
    def _setup_nltk_path(self):
        """設置NLTK數據路徑"""
        # 使用絕對路徑設置NLTK數據目錄
        home_dir = os.path.expanduser("~")
        nltk_data_path = os.path.join(home_dir, "nltk_data")
        
        # 確保目錄存在
        if not os.path.exists(nltk_data_path):
            os.makedirs(nltk_data_path)
        
        # 將此路徑添加到NLTK搜索路徑的最前面
        nltk.data.path.insert(0, nltk_data_path)
        
        # 直接設置環境變量 NLTK_DATA (可選)
        os.environ['NLTK_DATA'] = nltk_data_path
        
        logger.info(f"NLTK數據路徑設置為: {nltk_data_path}")
    
    def _download_nltk_resources(self):
        """下載NLTK所需資源"""
        resources = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet'),
            ('omw-1.4', 'corpora/omw-1.4')
        ]
        
        # 確保資源已下載
        for resource_name, resource_path in resources:
            try:
                nltk.data.find(resource_path)
                logger.info(f"NLTK資源 '{resource_name}' 已存在")
            except LookupError:
                logger.info(f"正在下載NLTK資源 '{resource_name}'...")
                try:
                    nltk.download(resource_name)
                    logger.info(f"NLTK資源 '{resource_name}' 下載完成")
                except Exception as e:
                    logger.error(f"下載NLTK資源 '{resource_name}' 時出錯: {str(e)}")
    
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
        
        # 重置警告標誌 - 每次導入新文件時
        self.tokenize_warning_shown = False
        self.lemmatize_warning_shown = False
        
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
            
            # 為了減少日誌輸出，我們先記錄開始處理
            logger.info(f"開始清洗 {len(df)} 條文本...")
            df['clean_text'] = df[text_column].apply(self._clean_text)
            logger.info("文本清洗完成")
            
            # 進行分詞和停用詞過濾
            if callback:
                callback("正在進行分詞和詞形還原...", 70)
            
            # 同樣，只記錄開始和結束的信息
            logger.info(f"開始對 {len(df)} 條清洗後的文本進行分詞和詞形還原...")
            df['tokens'] = df['clean_text'].apply(self._tokenize_and_lemmatize)
            df['tokens_str'] = df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
            logger.info("分詞和詞形還原完成")
            
            # 保存處理後的數據
            processed_data_path = os.path.join(
                self.output_dir, 
                f"processed_{os.path.basename(file_path)}"
            )
            
            if callback:
                callback("正在保存處理後的數據...", 90)
            
            # 確保輸出目錄存在
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            df.to_csv(processed_data_path, index=False)
            logger.info(f"處理後的數據已保存至: {processed_data_path}")
            
            if callback:
                callback("數據導入和基礎處理完成", 100)
            
            return processed_data_path
            
        except Exception as e:
            logger.error(f"導入數據時出錯: {str(e)}")
            logger.error(traceback.format_exc())
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
        try:
            if pd.isna(text):
                return ""
            
            # 轉換為字符串
            text = str(text)
            
            # 移除HTML標籤
            try:
                text = BeautifulSoup(text, "html.parser").get_text()
            except:
                # 如果BeautifulSoup處理失敗，使用正則表達式
                text = re.sub(r'<[^>]+>', ' ', text)
            
            # 移除URL
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # 移除特殊字符和數字
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            
            # 移除多餘的空格
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            # 不要為每個文本記錄錯誤，可能會產生大量日誌
            return ""
    
    def _tokenize_and_lemmatize(self, text):
        """
        對文本進行分詞、移除停用詞和詞形還原
        
        Args:
            text: 清洗後的文本
            
        Returns:
            tokens: 處理後的詞列表
        """
        try:
            # 使用自定義分詞器
            tokens = self.tokenizer.tokenize(text)
            
            # 移除停用詞和詞形還原
            filtered_tokens = []
            
            for word in tokens:
                if word.isalpha() and word not in self.stop_words:
                    lemmatized = self.lemmatizer.lemmatize(word)
                    filtered_tokens.append(lemmatized)
            
            return filtered_tokens
            
        except Exception as e:
            if not self.tokenize_warning_shown:
                logger.error(f"分詞過程出錯: {str(e)}")
                self.tokenize_warning_shown = True
            # 返回一個空列表作為後備
            return []

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
        traceback.print_exc()