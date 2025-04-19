"""
數據處理模組 - 負責數據的加載、清洗和預處理
"""

import os
import re
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 導入系統日誌模組
from utils.logger import get_logger

# 獲取logger
logger = get_logger("data_processor")

class DataProcessor:
    """數據處理基類，定義了通用的數據處理接口"""
    
    def __init__(self, config=None):
        """初始化數據處理器
        
        Args:
            config: 配置參數字典
        """
        self.config = config or {}
        self.logger = logger
        
        # 確保NLTK資源已下載
        self._ensure_nltk_resources()
        
        # 初始化詞形還原器和停用詞
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def update_config(self, new_config):
        """更新處理器配置
        
        Args:
            new_config: 新的配置參數字典
        """
        if not new_config:
            return
            
        self.logger.info("更新數據處理器配置")
        # 更新配置，保留原有的鍵值，只更新新配置中的值
        if isinstance(new_config, dict):
            self.config.update(new_config)
            self.logger.debug(f"配置更新完成: {json.dumps(self.config, ensure_ascii=False)[:100]}...")
        else:
            self.logger.warning(f"無效的配置格式: {type(new_config)}")
    
    def _ensure_nltk_resources(self):
        """確保NLTK所需資源已下載"""
        resources = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet')
        ]
        
        for resource_name, resource_path in resources:
            try:
                # 嘗試查找資源
                nltk.data.find(resource_path)
                self.logger.debug(f"已找到NLTK資源: {resource_name}")
            except LookupError:
                # 如果找不到，則下載
                self.logger.info(f"正在下載NLTK資源: {resource_name}")
                try:
                    nltk.download(resource_name, quiet=True)
                    self.logger.info(f"NLTK資源 {resource_name} 下載完成")
                except Exception as e:
                    self.logger.error(f"下載NLTK資源 {resource_name} 時發生錯誤: {str(e)}")
    
    def load_data(self, file_path):
        """加載數據
        
        Args:
            file_path: 數據文件路徑
            
        Returns:
            pd.DataFrame: 加載的數據
        """
        self.logger.info(f"正在加載數據: {file_path}")
        
        # 根據文件擴展名選擇加載方式
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.json':
                # 嘗試常規JSON加載
                try:
                    df = pd.read_json(file_path)
                except:
                    # 嘗試逐行JSON加載 (用於JSONL格式)
                    df = pd.read_json(file_path, lines=True)
            elif file_ext == '.txt':
                # 假設文本文件每行是一條評論
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                df = pd.DataFrame({'text': lines})
            elif file_ext == '.xlsx' or file_ext == '.xls':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"不支持的文件類型: {file_ext}")
            
            self.logger.info(f"成功加載數據，共 {len(df)} 行")
            return df
            
        except Exception as e:
            self.logger.error(f"加載數據失敗: {str(e)}")
            raise
    
    def preprocess(self, df, text_column=None):
        """預處理數據
        
        Args:
            df: 數據框或字串。如果是字串，則直接處理此文本
            text_column: 文本列名，如果df是DataFrame且text_column為None則自動檢測
            
        Returns:
            pd.DataFrame 或字串: 預處理後的數據
        """
        self.logger.info("開始數據預處理")
        
        # 檢查輸入類型
        if isinstance(df, str):
            # 如果輸入是字串，直接處理文本
            clean_text = self._clean_text(df)
            tokens = self._tokenize_and_lemmatize(clean_text)
            return " ".join(tokens) if isinstance(tokens, list) else ""
        
        # 如果是DataFrame，執行常規處理流程
        # 檢查並處理缺失值
        self._handle_missing_values(df)
        
        # 識別文本列
        text_column = self._identify_text_column(df, text_column)
        self.logger.info(f"使用 '{text_column}' 作為文本列")
        
        # 清洗文本
        self.logger.info("正在清洗文本...")
        df['clean_text'] = df[text_column].apply(self._clean_text)
        
        # 對文本進行標記化和詞形還原
        self.logger.info("正在進行分詞和詞形還原...")
        df['tokens'] = df['clean_text'].apply(self._tokenize_and_lemmatize)
        
        # 將標記轉換為字符串，便於後續處理
        df['tokens_str'] = df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        self.logger.info("數據預處理完成")
        return df
    
    def _identify_text_column(self, df, text_column=None):
        """識別文本列
        
        Args:
            df: 數據框
            text_column: 指定的文本列名，如果為None則自動檢測
            
        Returns:
            str: 文本列名
        """
        if text_column is not None and text_column in df.columns:
            return text_column
        
        # 嘗試常見的文本列名
        common_text_columns = ['text', 'review', 'review_text', 'reviewText', 'content', 'comment', 'description']
        for col in common_text_columns:
            if col in df.columns:
                return col
        
        # 如果找不到常見文本列，選擇字符串類型的列
        for col in df.columns:
            if df[col].dtype == 'object':  # pandas中的字符串類型
                # 進一步檢查是否包含足夠長的文本
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > 20:  # 假設平均長度大於20的是文本
                    return col
        
        # 如果仍然找不到，使用第一列
        self.logger.warning("無法識別文本列，使用第一列")
        return df.columns[0]
    
    def _handle_missing_values(self, df):
        """處理缺失值
        
        Args:
            df: 數據框
        """
        # 計算每列的缺失值比例
        missing_percentages = (df.isnull().sum() / len(df)) * 100
        
        for col, percentage in missing_percentages.items():
            if percentage > 0:
                self.logger.info(f"列 '{col}' 有 {percentage:.2f}% 的缺失值")
                
                # 處理缺失值
                if percentage > 50:
                    # 如果缺失值超過50%，則刪除該列
                    self.logger.warning(f"列 '{col}' 缺失值過多，將被刪除")
                    df.drop(col, axis=1, inplace=True)
                else:
                    # 根據列的數據類型選擇填充方式
                    if pd.api.types.is_numeric_dtype(df[col]):
                        # 數值型列使用中位數填充
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        # 非數值型列使用空字符串填充
                        df[col].fillna('', inplace=True)
    
    def _clean_text(self, text):
        """清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清洗後的文本
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
            self.logger.error(f"清洗文本時出錯: {str(e)}")
            return ""
    
    def _tokenize_and_lemmatize(self, text):
        """對文本進行分詞、詞形還原和停用詞移除
        
        Args:
            text: 清洗後的文本
            
        Returns:
            list: 處理後的詞列表
        """
        try:
            # 分詞
            tokens = word_tokenize(text.lower())
            
            # 移除停用詞和詞形還原
            filtered_tokens = []
            for word in tokens:
                if word.isalpha() and word not in self.stop_words:
                    lemmatized = self.lemmatizer.lemmatize(word)
                    filtered_tokens.append(lemmatized)
            
            return filtered_tokens
        except Exception as e:
            self.logger.error(f"分詞和詞形還原時出錯: {str(e)}")
            return []
    
    def save_processed_data(self, df, output_path):
        """保存處理後的數據
        
        Args:
            df: 預處理後的數據框
            output_path: 輸出路徑
            
        Returns:
            str: 保存的文件路徑
        """
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存為CSV
        df.to_csv(output_path, index=False)
        self.logger.info(f"處理後的數據已保存至: {output_path}")
        
        return output_path


class IMDBProcessor(DataProcessor):
    """IMDB電影評論數據處理器"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
    def preprocess(self, df, text_column=None):
        """IMDB數據預處理
        
        Args:
            df: 數據框
            text_column: 文本列名
            
        Returns:
            pd.DataFrame: 預處理後的數據
        """
        # 基本預處理
        df = super().preprocess(df, text_column)
        
        # IMDB特定處理: 標記電影名稱，識別指導和演員相關評論
        self.logger.info("進行IMDB特定處理...")
        
        # 提取評分（如果有）
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        elif 'score' in df.columns:
            df['rating'] = pd.to_numeric(df['score'], errors='coerce')
        
        # 設置數據來源標記
        df['data_source'] = 'imdb'
        
        return df


class AmazonProcessor(DataProcessor):
    """Amazon產品評論數據處理器"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
    def preprocess(self, df, text_column=None):
        """Amazon數據預處理
        
        Args:
            df: 數據框
            text_column: 文本列名
            
        Returns:
            pd.DataFrame: 預處理後的數據
        """
        # 基本預處理
        df = super().preprocess(df, text_column)
        
        # Amazon特定處理：標準化產品類別、處理評分等
        self.logger.info("進行Amazon特定處理...")
        
        # 處理評分列（常見的有 'rating', 'stars', 'overall'）
        rating_columns = ['rating', 'stars', 'overall']
        for col in rating_columns:
            if col in df.columns:
                df['rating'] = pd.to_numeric(df[col], errors='coerce')
                break
        
        # 標準化產品類別列（如果存在）
        category_columns = ['category', 'categories', 'productCategory']
        for col in category_columns:
            if col in df.columns:
                df['product_category'] = df[col]
                break
        
        # 設置數據來源標記
        df['data_source'] = 'amazon'
        
        return df


class YelpProcessor(DataProcessor):
    """Yelp餐廳評論數據處理器"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
    def preprocess(self, df, text_column=None):
        """Yelp數據預處理
        
        Args:
            df: 數據框
            text_column: 文本列名
            
        Returns:
            pd.DataFrame: 預處理後的數據
        """
        # 基本預處理
        df = super().preprocess(df, text_column)
        
        # Yelp特定處理：標準化餐廳類別、地理位置等
        self.logger.info("進行Yelp特定處理...")
        
        # 處理評分列
        if 'stars' in df.columns:
            df['rating'] = pd.to_numeric(df['stars'], errors='coerce')
        
        # 處理商家ID
        if 'business_id' in df.columns:
            # 保留原始ID
            pass
        
        # 處理類別信息
        if 'categories' in df.columns:
            # 如果categories是字符串，嘗試解析它
            if df['categories'].dtype == 'object':
                df['categories'] = df['categories'].apply(
                    lambda x: x.split(',') if isinstance(x, str) else x
                )
        
        # 設置數據來源標記
        df['data_source'] = 'yelp'
        
        return df


def create_processor(data_source, config=None):
    """創建對應數據源的處理器
    
    Args:
        data_source: 數據源名稱，可以是'imdb', 'amazon', 'yelp'或'auto'
        config: 配置參數
        
    Returns:
        DataProcessor: 數據處理器實例
    """
    data_source = data_source.lower()
    
    if data_source == 'imdb':
        return IMDBProcessor(config)
    elif data_source == 'amazon':
        return AmazonProcessor(config)
    elif data_source == 'yelp':
        return YelpProcessor(config)
    elif data_source == 'auto':
        # 自動選擇處理器會在運行時根據文件內容判斷
        return DataProcessor(config)
    else:
        logger.warning(f"未知的數據源: {data_source}，使用通用處理器")
        return DataProcessor(config)


def process_file(file_path, data_source='auto', output_dir='./output/data', config=None):
    """處理單個數據文件
    
    Args:
        file_path: 數據文件路徑
        data_source: 數據源類型，'imdb', 'amazon', 'yelp'或'auto'
        output_dir: 輸出目錄
        config: 配置參數
        
    Returns:
        str: 處理後的文件路徑
    """
    logger.info(f"開始處理文件: {file_path}, 數據源: {data_source}")
    
    # 創建處理器
    processor = create_processor(data_source, config)
    
    # 加載數據
    df = processor.load_data(file_path)
    
    # 自動檢測數據源（如果設置為'auto'）
    if data_source == 'auto':
        data_source = _detect_data_source(df)
        logger.info(f"自動檢測到數據源: {data_source}")
        processor = create_processor(data_source, config)
    
    # 預處理數據
    processed_df = processor.preprocess(df)
    
    # 構建輸出文件路徑
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, f"processed_{file_name}")
    
    # 如果輸出是非CSV格式，確保改為CSV
    if not output_path.endswith('.csv'):
        output_path = os.path.splitext(output_path)[0] + '.csv'
    
    # 保存處理後的數據
    return processor.save_processed_data(processed_df, output_path)


def _detect_data_source(df):
    """根據數據框內容自動檢測數據源
    
    Args:
        df: 數據框
        
    Returns:
        str: 檢測到的數據源類型
    """
    # 檢查列名
    columns = set(df.columns.str.lower())
    
    # IMDB特有列
    imdb_columns = {'movie', 'director', 'actor', 'title', 'imdb'}
    
    # Amazon特有列
    amazon_columns = {'product', 'asin', 'productid', 'price', 'amazon'}
    
    # Yelp特有列
    yelp_columns = {'business_id', 'restaurant', 'city', 'state', 'yelp'}
    
    # 計算匹配度
    imdb_score = len(columns.intersection(imdb_columns))
    amazon_score = len(columns.intersection(amazon_columns))
    yelp_score = len(columns.intersection(yelp_columns))
    
    # 如果列名匹配不明確，檢查文本內容
    if max(imdb_score, amazon_score, yelp_score) == 0:
        # 提取一些樣本文本進行分析
        if 'text' in df.columns:
            sample_texts = df['text'].dropna().astype(str).head(10).str.lower()
        else:
            # 使用第一個字符串類型的列
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            if text_cols:
                sample_texts = df[text_cols[0]].dropna().astype(str).head(10).str.lower()
            else:
                return 'unknown'
        
        # 檢查關鍵詞
        imdb_keywords = ['movie', 'film', 'actor', 'director', 'scene', 'character']
        amazon_keywords = ['product', 'purchase', 'bought', 'seller', 'shipping', 'amazon']
        yelp_keywords = ['restaurant', 'food', 'service', 'menu', 'waiter', 'dish', 'meal']
        
        imdb_score = sum(any(kw in text for kw in imdb_keywords) for text in sample_texts)
        amazon_score = sum(any(kw in text for kw in amazon_keywords) for text in sample_texts)
        yelp_score = sum(any(kw in text for kw in yelp_keywords) for text in sample_texts)
    
    # 返回最高分的數據源
    max_score = max(imdb_score, amazon_score, yelp_score)
    if max_score == 0:
        return 'unknown'
    
    if imdb_score == max_score:
        return 'imdb'
    elif amazon_score == max_score:
        return 'amazon'
    else:
        return 'yelp'


# 測試代碼
if __name__ == "__main__":
    # 配置日誌級別
    logger.setLevel(logging.INFO)
    
    # 測試處理IMDB數據
    test_file = "path/to/your/imdb_sample.csv"
    
    if os.path.exists(test_file):
        # 處理文件
        processed_file = process_file(test_file, data_source='imdb')
        logger.info(f"處理完成，結果保存為: {processed_file}")
    else:
        logger.warning(f"測試文件不存在: {test_file}")