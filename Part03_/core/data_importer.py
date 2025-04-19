"""
數據導入模組
此模組負責從不同來源導入和預處理文本數據
"""

import os
import pandas as pd
import json
import re
import nltk
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import datetime
import sys

# 添加專案根目錄到模塊搜索路徑
sys.path.append(os.path.abspath('.'))

# 從utils模塊導入工具類
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.config_manager import ConfigManager

class DataImporter:
    """
    數據導入處理類
    用於從各種數據源（如CSV、JSON等）導入數據並進行基礎預處理
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化數據導入器
        
        Args:
            config: 配置管理器，如果沒有提供則創建新的
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 獲取輸入目錄和輸出目錄
        self.input_dirs = {
            'amazon': self.config.get('data_settings.input_directories.amazon'),
            'yelp': self.config.get('data_settings.input_directories.yelp'),
            'imdb': self.config.get('data_settings.input_directories.imdb')
        }
        self.output_dir = self.config.get('data_settings.output_directory')
        
        # 確保輸出目錄存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 初始化日誌
        log_file = os.path.join(self.config.get('logging.log_dir', './Part03_/logs'), 'data_import.log')
        self.logger = logging.getLogger('data_importer')
        self.logger.setLevel(logging.INFO)
        
        # 移除所有處理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 添加新的處理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # 確保nltk數據已下載
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """確保必要的NLTK數據已下載"""
        try:
            nltk_data_dir = './Part03_/nltk_data'
            os.makedirs(nltk_data_dir, exist_ok=True)
            nltk.data.path.append(os.path.abspath(nltk_data_dir))
            
            required_packages = ['punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
            for package in required_packages:
                try:
                    nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else
                                   f'taggers/{package}' if package == 'averaged_perceptron_tagger' else
                                   f'corpora/{package}')
                    self.logger.info(f"已找到NLTK套件: {package}")
                except LookupError:
                    self.logger.info(f"正在下載NLTK套件: {package}")
                    nltk.download(package, download_dir=nltk_data_dir)
        except Exception as e:
            self.logger.error(f"準備NLTK數據時發生錯誤: {str(e)}")
    
    def import_imdb_data(self, file_path: str = None, sample_size: Optional[int] = None, 
                        min_length: int = 50, console_output: bool = True) -> pd.DataFrame:
        """
        導入IMDB數據集
        
        Args:
            file_path: 數據文件路徑，如果不提供則使用配置中的默認路徑
            sample_size: 如果指定，則隨機抽取指定數量的數據
            min_length: 評論的最小長度（字符數）
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            處理後的數據集DataFrame
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("IMDB數據處理")
            logger = ConsoleOutputManager.setup_console_logger("imdb_import", log_file)
        else:
            logger = self.logger
        
        try:
            logger.info("開始導入IMDB數據...")
            
            # 確定文件路徑
            if not file_path:
                default_path = os.path.join(self.input_dirs['imdb'], 'IMDB Dataset.csv')
                test_path = os.path.join(self.input_dirs['imdb'], 'IMDB_Test.csv')
                
                if os.path.exists(test_path):
                    file_path = test_path
                    logger.info(f"使用測試數據集: {test_path}")
                elif os.path.exists(default_path):
                    file_path = default_path
                    logger.info(f"使用完整數據集: {default_path}")
                else:
                    logger.error(f"未找到IMDB數據集文件")
                    raise FileNotFoundError(f"IMDB數據集文件未找到: {default_path}")
            
            # 讀取數據
            logger.info(f"正在讀取文件: {file_path}")
            df = pd.read_csv(file_path)
            
            # 檢查數據結構
            required_cols = ['review', 'sentiment']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"數據缺少必要列: {col}")
                    raise ValueError(f"數據缺少必要列: {col}")
            
            initial_count = len(df)
            logger.info(f"原始數據包含 {initial_count} 條評論")
            
            # 數據清洗
            logger.info("正在進行數據清洗...")
            
            # 移除過短的評論
            df = df[df['review'].str.len() >= min_length]
            logger.info(f"移除過短評論後剩餘 {len(df)} 條評論")
            
            # 數據處理
            df['processed_text'] = df['review'].apply(self._preprocess_text)
            
            # 如果指定了樣本大小，隨機抽取
            if sample_size and sample_size < len(df):
                df = df.sample(sample_size, random_state=42)
                logger.info(f"隨機抽取 {sample_size} 條評論")
            
            # 添加ID列
            df['id'] = [f"imdb_{i}" for i in range(len(df))]
            
            # 標準化情感標籤
            df['sentiment'] = df['sentiment'].map({'positive': 'positive', 'negative': 'negative'})
            
            # 調整列順序
            df = df[['id', 'review', 'processed_text', 'sentiment']]
            
            logger.info(f"IMDB數據處理完成，結果包含 {len(df)} 條評論")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return df
            
        except Exception as e:
            logger.error(f"處理IMDB數據時發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def import_yelp_data(self, review_file: str = None, business_file: str = None, 
                       category_filter: str = 'Restaurants', min_stars: int = 1, max_stars: int = 5,
                       sample_size: Optional[int] = None, console_output: bool = True) -> pd.DataFrame:
        """
        導入Yelp數據集
        
        Args:
            review_file: 評論文件路徑，如果不提供則使用配置中的默認路徑
            business_file: 商家文件路徑，如果不提供則使用配置中的默認路徑
            category_filter: 商家類別過濾條件
            min_stars: 最低星級（1-5）
            max_stars: 最高星級（1-5）
            sample_size: 如果指定，則隨機抽取指定數量的數據
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            處理後的數據集DataFrame
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("Yelp數據處理")
            logger = ConsoleOutputManager.setup_console_logger("yelp_import", log_file)
        else:
            logger = self.logger
        
        try:
            logger.info("開始導入Yelp數據...")
            
            # 確定文件路徑
            if not review_file:
                review_file = os.path.join(self.input_dirs['yelp'], 'yelp_review.json')
            
            if not business_file:
                business_file = os.path.join(self.input_dirs['yelp'], 'yelp_business.json')
            
            # 檢查文件是否存在
            if not os.path.exists(review_file):
                logger.error(f"評論文件不存在: {review_file}")
                raise FileNotFoundError(f"評論文件不存在: {review_file}")
            
            if not os.path.exists(business_file):
                logger.error(f"商家文件不存在: {business_file}")
                raise FileNotFoundError(f"商家文件不存在: {business_file}")
            
            # 讀取商家數據
            logger.info(f"正在讀取商家數據: {business_file}")
            businesses = {}
            with open(business_file, 'r', encoding='utf-8') as f:
                for line in f:
                    business = json.loads(line)
                    # 過濾符合類別的商家
                    if 'categories' in business and business['categories'] and category_filter in business['categories']:
                        businesses[business['business_id']] = business
            
            logger.info(f"找到 {len(businesses)} 個符合類別 '{category_filter}' 的商家")
            
            if not businesses:
                logger.warning(f"未找到符合類別 '{category_filter}' 的商家")
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                return pd.DataFrame()
            
            # 讀取評論數據
            logger.info(f"正在讀取評論數據: {review_file}")
            reviews = []
            with open(review_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    review = json.loads(line)
                    # 過濾符合條件的評論
                    if (review['business_id'] in businesses and 
                        min_stars <= review['stars'] <= max_stars):
                        review['business_name'] = businesses[review['business_id']]['name']
                        reviews.append(review)
                    
                    # 顯示進度
                    if i % 10000 == 0:
                        logger.info(f"已處理 {i} 條評論")
            
            logger.info(f"找到 {len(reviews)} 條符合條件的評論")
            
            if not reviews:
                logger.warning("未找到符合條件的評論")
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                return pd.DataFrame()
            
            # 轉換為DataFrame
            df = pd.DataFrame(reviews)
            
            # 清理數據
            logger.info("正在清理數據...")
            df = df[['review_id', 'business_id', 'business_name', 'text', 'stars', 'date']]
            
            # 處理文本
            df['processed_text'] = df['text'].apply(self._preprocess_text)
            
            # 轉換星級為情感標籤
            df['sentiment'] = df['stars'].apply(lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')
            
            # 提取評論年份
            df['year'] = pd.to_datetime(df['date']).dt.year
            
            # 如果指定了樣本大小，隨機抽取
            if sample_size and sample_size < len(df):
                df = df.sample(sample_size, random_state=42)
                logger.info(f"隨機抽取 {sample_size} 條評論")
            
            # 調整列順序
            df = df[['review_id', 'business_id', 'business_name', 'text', 'processed_text', 'stars', 'sentiment', 'date', 'year']]
            
            logger.info(f"Yelp數據處理完成，結果包含 {len(df)} 條評論")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return df
            
        except Exception as e:
            logger.error(f"處理Yelp數據時發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def import_amazon_data(self, file_path: str = None, product_category: str = None,
                         min_rating: float = 1.0, max_rating: float = 5.0,
                         sample_size: Optional[int] = None, console_output: bool = True) -> pd.DataFrame:
        """
        導入Amazon數據集
        
        Args:
            file_path: 數據文件路徑，如果不提供則使用配置中的默認路徑
            product_category: 產品類別過濾條件
            min_rating: 最低評分
            max_rating: 最高評分
            sample_size: 如果指定，則隨機抽取指定數量的數據
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            處理後的數據集DataFrame
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("Amazon數據處理")
            logger = ConsoleOutputManager.setup_console_logger("amazon_import", log_file)
        else:
            logger = self.logger
        
        try:
            logger.info("開始導入Amazon數據...")
            
            # 確定文件路徑
            if not file_path:
                # 如果指定了產品類別，嘗試找到對應的文件
                if product_category:
                    category_path = os.path.join(self.input_dirs['amazon'], f"{product_category}.json")
                    if os.path.exists(category_path):
                        file_path = category_path
                    else:
                        # 嘗試尋找其他可能的命名格式
                        potential_files = [f for f in os.listdir(self.input_dirs['amazon']) 
                                         if product_category.lower() in f.lower() and f.endswith('.json')]
                        
                        if potential_files:
                            file_path = os.path.join(self.input_dirs['amazon'], potential_files[0])
                            logger.info(f"找到匹配的產品類別文件: {file_path}")
                        else:
                            logger.error(f"未找到產品類別 '{product_category}' 的數據文件")
                            raise FileNotFoundError(f"未找到產品類別數據文件: {product_category}")
                else:
                    # 嘗試找到任何可用的JSON文件
                    json_files = [f for f in os.listdir(self.input_dirs['amazon']) if f.endswith('.json')]
                    if json_files:
                        file_path = os.path.join(self.input_dirs['amazon'], json_files[0])
                        product_category = os.path.splitext(json_files[0])[0]
                        logger.info(f"使用發現的數據文件: {file_path}")
                    else:
                        logger.error(f"在 {self.input_dirs['amazon']} 中未找到任何JSON數據文件")
                        raise FileNotFoundError(f"在Amazon數據目錄中未找到任何JSON數據文件")
            
            # 讀取數據
            logger.info(f"正在讀取文件: {file_path}")
            
            # 如果是JSON格式
            if file_path.endswith('.json'):
                reviews = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        try:
                            review = json.loads(line.strip())
                            # 確保review有所需的字段
                            if 'reviewText' in review and 'overall' in review:
                                if min_rating <= review['overall'] <= max_rating:
                                    reviews.append(review)
                            
                            # 顯示進度
                            if i % 10000 == 0 and i > 0:
                                logger.info(f"已處理 {i} 行數據")
                        except json.JSONDecodeError:
                            logger.warning(f"第 {i+1} 行包含無效的JSON數據")
                
                df = pd.DataFrame(reviews)
                
                # 檢查必要的列
                required_cols = ['reviewText', 'overall']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    logger.error(f"數據缺少必要的列: {missing_cols}")
                    raise ValueError(f"數據缺少必要的列: {missing_cols}")
                
            # 如果是CSV格式
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                
                # 檢查和重命名列
                if 'review_text' in df.columns and 'reviewText' not in df.columns:
                    df.rename(columns={'review_text': 'reviewText'}, inplace=True)
                
                if 'rating' in df.columns and 'overall' not in df.columns:
                    df.rename(columns={'rating': 'overall'}, inplace=True)
                
                # 過濾評分範圍
                df = df[(df['overall'] >= min_rating) & (df['overall'] <= max_rating)]
                
            else:
                logger.error(f"不支持的文件格式: {file_path}")
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            initial_count = len(df)
            logger.info(f"原始數據包含 {initial_count} 條評論")
            
            if initial_count == 0:
                logger.warning("過濾後沒有符合條件的評論")
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                return pd.DataFrame()
            
            # 數據處理
            logger.info("正在進行數據處理...")
            
            # 確保有標準化的列名
            if 'reviewText' in df.columns:
                df.rename(columns={'reviewText': 'text'}, inplace=True)
            
            # 處理文本
            df['processed_text'] = df['text'].apply(self._preprocess_text)
            
            # 標準化情感標籤
            df['sentiment'] = df['overall'].apply(
                lambda x: 'positive' if x >= 4 else 'negative' if x <= 2 else 'neutral')
            
            # 添加產品類別信息
            if product_category:
                df['product_category'] = product_category
            
            # 添加ID列（如果不存在）
            if 'reviewID' not in df.columns and 'review_id' not in df.columns:
                df['review_id'] = [f"amazon_{i}" for i in range(len(df))]
            elif 'reviewID' in df.columns:
                df.rename(columns={'reviewID': 'review_id'}, inplace=True)
            
            # 如果指定了樣本大小，隨機抽取
            if sample_size and sample_size < len(df):
                df = df.sample(sample_size, random_state=42)
                logger.info(f"隨機抽取 {sample_size} 條評論")
            
            # 選擇和重排列
            essential_cols = ['review_id', 'text', 'processed_text', 'overall', 'sentiment']
            
            # 添加可選列（如果存在）
            for col in ['product_id', 'product_category', 'summary', 'reviewTime', 'reviewerID']:
                if col in df.columns:
                    essential_cols.append(col)
            
            # 只保留存在的列
            existing_cols = [col for col in essential_cols if col in df.columns]
            df = df[existing_cols]
            
            logger.info(f"Amazon數據處理完成，結果包含 {len(df)} 條評論")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return df
            
        except Exception as e:
            logger.error(f"處理Amazon數據時發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """
        對文本進行預處理
        
        Args:
            text: 原始文本
            
        Returns:
            處理後的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 移除HTML標籤
        text = re.sub(r'<.*?>', '', text)
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 統一空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)
        
        # 統一大小寫
        text = text.lower()
        
        return text.strip()
    
    def save_processed_data(self, df: pd.DataFrame, dataset_name: str, output_format: str = 'csv') -> str:
        """
        保存處理後的數據
        
        Args:
            df: 處理後的DataFrame
            dataset_name: 數據集名稱
            output_format: 輸出格式 ('csv', 'pickle', 'json')
            
        Returns:
            保存的文件路徑
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(self.output_dir, '01_processed_data')
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 根據格式保存文件
        if output_format == 'csv':
            file_path = os.path.join(output_dir, f"{dataset_name}_{timestamp}.csv")
            df.to_csv(file_path, index=False, encoding='utf-8')
        elif output_format == 'pickle':
            file_path = os.path.join(output_dir, f"{dataset_name}_{timestamp}.pkl")
            df.to_pickle(file_path)
        elif output_format == 'json':
            file_path = os.path.join(output_dir, f"{dataset_name}_{timestamp}.json")
            df.to_json(file_path, orient='records', lines=True)
        else:
            raise ValueError(f"不支持的輸出格式: {output_format}")
        
        self.logger.info(f"已保存處理後的數據到: {file_path}")
        return file_path
    
    @staticmethod
    def combine_datasets(datasets: List[pd.DataFrame], id_prefix: bool = True) -> pd.DataFrame:
        """
        合併多個數據集
        
        Args:
            datasets: 數據集列表
            id_prefix: 是否為每個數據集的ID添加前綴以避免衝突
            
        Returns:
            合併後的DataFrame
        """
        if not datasets:
            return pd.DataFrame()
        
        if len(datasets) == 1:
            return datasets[0].copy()
        
        result = []
        for i, df in enumerate(datasets):
            if len(df) == 0:
                continue
                
            df_copy = df.copy()
            
            # 確保有ID列
            id_col = None
            for col_name in ['id', 'review_id', 'reviewID']:
                if col_name in df_copy.columns:
                    id_col = col_name
                    break
            
            if id_col is None:
                df_copy['id'] = [f"dataset{i+1}_{j}" for j in range(len(df_copy))]
                id_col = 'id'
            
            # 如果需要添加前綴
            if id_prefix:
                df_copy[id_col] = df_copy[id_col].apply(lambda x: f"dataset{i+1}_{x}" if not str(x).startswith(f"dataset{i+1}_") else x)
            
            result.append(df_copy)
        
        # 合併數據集
        combined = pd.concat(result, ignore_index=True)
        
        return combined

# 使用示例
if __name__ == "__main__":
    importer = DataImporter()
    
    # 導入IMDB數據
    imdb_df = importer.import_imdb_data(sample_size=1000)
    print(f"IMDB數據: {len(imdb_df)} 行")
    
    # 保存處理後的數據
    if len(imdb_df) > 0:
        output_path = importer.save_processed_data(imdb_df, 'IMDB_Test')
        print(f"數據已保存到: {output_path}")
