"""
數據處理模組 - 負責數據的載入、清洗和預處理
"""

import os
import re
import sys
import json
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import logging
import traceback
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import time
import random
from sklearn.model_selection import train_test_split

# 導入NLTK相關套件
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    nltk_available = True
except ImportError:
    nltk_available = False

# 導入系統日誌模組
from utils.logger import get_logger

# 獲取logger
logger = get_logger("data_processor")

# 固定所有隨機種子，確保結果可重現
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# 如果存在pandas的random_seed方法也設置
pd.Series([]).sample(frac=1, random_state=RANDOM_SEED)

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
        # 指定NLTK數據目錄路徑
        nltk_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
        os.makedirs(nltk_data_path, exist_ok=True)
        
        # 將此路徑添加到NLTK搜索路徑的最前面
        if nltk_data_path not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_path)
            self.logger.info(f"已將 {nltk_data_path} 添加到NLTK搜索路徑")
        
        # 需要下載的資源
        resources = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet')
        ]
        
        # 確保每個資源都已下載
        for resource_name, resource_path in resources:
            try:
                # 嘗試查找資源
                nltk.data.find(resource_path)
                self.logger.debug(f"已找到NLTK資源: {resource_name}")
            except LookupError:
                # 如果找不到，則下載到指定目錄
                self.logger.info(f"正在下載NLTK資源: {resource_name}")
                try:
                    nltk.download(resource_name, download_dir=nltk_data_path, quiet=False)
                    self.logger.info(f"NLTK資源 {resource_name} 下載完成")
                    
                    # 特殊處理punkt資源，確保punkt_tab目錄存在
                    if resource_name == 'punkt':
                        punkt_dir = os.path.join(nltk_data_path, "tokenizers", "punkt")
                        if os.path.exists(punkt_dir):
                            # 確保english資料夾存在
                            english_dir = os.path.join(punkt_dir, "english")
                            os.makedirs(english_dir, exist_ok=True)
                            
                            # 建立punkt_tab目錄並複製相關檔案
                            punkt_tab_dir = os.path.join(nltk_data_path, "tokenizers", "punkt_tab", "english")
                            os.makedirs(os.path.dirname(punkt_tab_dir), exist_ok=True)
                            
                            if not os.path.exists(punkt_tab_dir):
                                os.makedirs(punkt_tab_dir, exist_ok=True)
                                self.logger.info(f"已建立punkt_tab目錄: {punkt_tab_dir}")
                                
                                # 複製相關檔案
                                punkt_english_files = os.listdir(english_dir) if os.path.exists(english_dir) else []
                                for file in punkt_english_files:
                                    src = os.path.join(english_dir, file)
                                    dst = os.path.join(punkt_tab_dir, file)
                                    if os.path.isfile(src):
                                        import shutil
                                        shutil.copy2(src, dst)
                                        self.logger.info(f"已複製檔案: {file}")
                                        
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
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        
        # 移除中間處理列
        df.drop(['clean_text', 'tokens'], axis=1, inplace=True)
        
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
        
        # 嘗試常見的文本列名（擴展列表以包含更多可能的文本欄位名稱）
        common_text_columns = [
            'text', 'review', 'review_text', 'reviewText', 
            'content', 'comment', 'description', 
            'Review', 'TEXT', 'REVIEW', 'Content',
            'review_body', 'reviews', 'feedback'
        ]
        
        # 先檢查最常見的列名
        for col in common_text_columns:
            if col in df.columns:
                self.logger.info(f"找到文本列: '{col}'")
                return col
        
        # 檢查列名中包含常見文本關鍵字的列
        text_keywords = ['text', 'review', 'comment', 'content', 'description', 'feedback']
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in text_keywords):
                self.logger.info(f"找到可能的文本列: '{col}'")
                return col
        
        # 如果找不到常見文本列，選擇字符串類型的列
        for col in df.columns:
            if df[col].dtype == 'object':  # pandas中的字符串類型
                # 進一步檢查是否包含足夠長的文本
                try:
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > 20:  # 假設平均長度大於20的是文本
                        self.logger.info(f"基於內容長度選擇文本列: '{col}'")
                        return col
                except Exception as e:
                    self.logger.debug(f"計算列 '{col}' 平均長度時出錯: {str(e)}")
        
        # 如果仍然找不到適合的列，檢查資料中是否有可能的評論文本欄位
        # 特別檢查 Amazon 格式
        if 'asin' in df.columns or 'overall' in df.columns:
            # 這可能是 Amazon 數據
            self.logger.info("檢測到可能的 Amazon 格式數據")
            if 'reviewText' in df.columns:
                return 'reviewText'
        
        # 如果仍然找不到，使用第一列或報錯
        if len(df.columns) > 0:
            self.logger.warning(f"無法識別文本列，使用第一列: {df.columns[0]}")
            return df.columns[0]
        else:
            self.logger.error("數據框為空或無法識別文本列")
            raise ValueError("數據中找不到文本列，請確保數據中包含'text'或'review'等列")
    
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
        if not text or not isinstance(text, str):
            return []
            
        try:
            # 轉換為小寫
            text = text.lower()
            
            # 優先使用不依賴punkt_tab的tokenization方法
            try:
                # 直接使用punkt而非punkt_tab
                from nltk.tokenize import word_tokenize as nltk_word_tokenize
                tokens = nltk_word_tokenize(text)
                self.logger.debug("成功使用NLTK基本分詞")
            except Exception as e:
                self.logger.debug(f"基本分詞失敗，嘗試替代方法: {str(e)}")
                
                # 嘗試使用其他NLTK分詞器
                try:
                    from nltk.tokenize import RegexpTokenizer
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(text)
                    self.logger.debug("成功使用NLTK正則表達式分詞器")
                except Exception as e2:
                    self.logger.warning(f"NLTK分詞完全失敗: {str(e2)}，使用基本正則表達式")
                    tokens = re.findall(r'\b\w+\b', text)
            
            # 初始化停用詞集合（如果尚未初始化）
            if not hasattr(self, 'stop_words') or not self.stop_words:
                try:
                    self.stop_words = set(stopwords.words('english'))
                    # 添加自定義停用詞
                    custom_stopwords = {
                        'would', 'could', 'should', 'might', 'must', 'need',
                        'gonna', 'wanna', 'gotta', 'let', 'amp',
                        'im', 'ive', 'id', 'dont', 'doesnt', 'isnt', 'wasnt',
                        'wouldnt', 'couldnt', 'shouldnt', 'cant', 'cannot',
                        'wont', 'hadnt', 'havent', 'hasnt', 'didnt',
                        'year', 'month', 'day', 'time', 'today', 'tomorrow',
                        'yesterday', 'week', 'weekend'
                    }
                    self.stop_words.update(custom_stopwords)
                except:
                    # 如果無法獲取NLTK停用詞，使用基本停用詞列表
                    self.stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 
                                     'at', 'from', 'by', 'for', 'with', 'about', 'to', 'in', 'on'}
                    self.logger.warning("使用基本停用詞列表替代NLTK停用詞")
            
            # 初始化詞形還原器（如果尚未初始化）
            if not hasattr(self, 'lemmatizer'):
                try:
                    self.lemmatizer = WordNetLemmatizer()
                except:
                    # 如果無法創建詞形還原器，設為None
                    self.lemmatizer = None
                    self.logger.warning("無法創建WordNetLemmatizer，將跳過詞形還原")
            
            # 移除停用詞和過短的詞
            filtered_tokens = []
            for token in tokens:
                # 檢查詞長度（至少2個字符）和停用詞
                if len(token) >= 2 and token not in self.stop_words:
                    # 進行詞形還原
                    if self.lemmatizer:
                        # 嘗試不同的詞性進行詞形還原
                        lemmatized = self.lemmatizer.lemmatize(token, pos='v')  # 動詞
                        if lemmatized == token:
                            lemmatized = self.lemmatizer.lemmatize(token, pos='n')  # 名詞
                        if lemmatized == token:
                            lemmatized = self.lemmatizer.lemmatize(token, pos='a')  # 形容詞
                        token = lemmatized
                    filtered_tokens.append(token)
            
            # 如果處理後的詞列表為空，返回原始標記化結果
            if not filtered_tokens and tokens:
                return tokens
            
            return filtered_tokens
            
        except Exception as e:
            self.logger.error(f"標記化和詞形還原時出錯: {str(e)}")
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

    def split_dataset(self, df, test_size=0.2, val_size=0.1, random_state=None, stratify_column=None):
        """將資料集分割為訓練集、驗證集和測試集
        
        Args:
            df: 要分割的數據框
            test_size: 測試集比例 (預設 0.2，即20%)
            val_size: 驗證集比例 (預設 0.1，即10%)，從訓練集中劃分
            random_state: 隨機種子 (預設使用全域種子)
            stratify_column: 用於分層抽樣的列名 (可選)
            
        Returns:
            tuple: (train_df, val_df, test_df) 訓練集、驗證集、測試集
        """
        self.logger.info(f"開始資料集分割，測試集比例: {test_size}, 驗證集比例: {val_size}")
        
        # 使用全域隨機種子
        if random_state is None:
            random_state = RANDOM_SEED
            
        original_size = len(df)
        
        # 處理分層抽樣參數
        stratify_values = None
        if stratify_column and stratify_column in df.columns:
            # 檢查每個類別的樣本數
            class_counts = df[stratify_column].value_counts()
            min_samples = class_counts.min()
            
            if min_samples < 2:
                self.logger.warning(f"類別 {class_counts[class_counts == min_samples].index[0]} 只有 {min_samples} 個樣本，無法進行分層抽樣。將使用隨機抽樣代替。")
                stratify_values = None
            else:
                stratify_values = df[stratify_column]
                self.logger.info(f"使用分層抽樣，基於列: {stratify_column}")
                self.logger.info(f"各類別樣本數: {class_counts.to_dict()}")
        
        try:
            # 第一次分割：將資料分為 訓練+驗證 和 測試
            train_val_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=stratify_values
            )
            
            # 計算驗證集在訓練+驗證集中的比例
            # 如果原始資料100筆，測試集20筆，剩餘80筆
            # 驗證集要10筆，則在80筆中佔12.5% (10/80 = 0.125)
            val_size_adjusted = val_size / (1 - test_size)
            
            # 處理驗證集的分層抽樣
            val_stratify_values = None
            if stratify_column and stratify_column in train_val_df.columns:
                # 再次檢查每個類別的樣本數
                val_class_counts = train_val_df[stratify_column].value_counts()
                val_min_samples = val_class_counts.min()
                
                if val_min_samples < 2:
                    self.logger.warning(f"驗證集分割時，類別 {val_class_counts[val_class_counts == val_min_samples].index[0]} 只有 {val_min_samples} 個樣本，無法進行分層抽樣。將使用隨機抽樣代替。")
                    val_stratify_values = None
                else:
                    val_stratify_values = train_val_df[stratify_column]
            
            # 第二次分割：將訓練+驗證集分為 訓練 和 驗證
            if val_size > 0:
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=val_size_adjusted,
                    random_state=random_state,
                    stratify=val_stratify_values
                )
            else:
                # 如果不需要驗證集
                train_df = train_val_df
                val_df = pd.DataFrame(columns=df.columns)  # 空的驗證集
            
            # 記錄分割結果
            self.logger.info(f"資料集分割完成:")
            self.logger.info(f"  原始資料: {original_size} 筆")
            self.logger.info(f"  訓練集: {len(train_df)} 筆 ({len(train_df)/original_size:.1%})")
            self.logger.info(f"  驗證集: {len(val_df)} 筆 ({len(val_df)/original_size:.1%})")
            self.logger.info(f"  測試集: {len(test_df)} 筆 ({len(test_df)/original_size:.1%})")
            
            # 重置索引
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            
            return train_df, val_df, test_df
            
        except Exception as e:
            self.logger.error(f"資料集分割失敗: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def save_split_datasets(self, train_df, val_df, test_df, output_dir, base_filename):
        """保存分割後的資料集
        
        Args:
            train_df: 訓練集
            val_df: 驗證集  
            test_df: 測試集
            output_dir: 輸出目錄
            base_filename: 基礎文件名
        """
        self.logger.info(f"正在保存分割後的資料集到: {output_dir}")
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成時間戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存各個資料集
        train_path = os.path.join(output_dir, f"{base_filename}_train_{timestamp}.csv")
        val_path = os.path.join(output_dir, f"{base_filename}_val_{timestamp}.csv")
        test_path = os.path.join(output_dir, f"{base_filename}_test_{timestamp}.csv")
        
        try:
            train_df.to_csv(train_path, index=False, encoding='utf-8')
            val_df.to_csv(val_path, index=False, encoding='utf-8')
            test_df.to_csv(test_path, index=False, encoding='utf-8')
            
            self.logger.info(f"訓練集已保存: {train_path}")
            self.logger.info(f"驗證集已保存: {val_path}")
            self.logger.info(f"測試集已保存: {test_path}")
            
            # 創建分割摘要文件
            summary_path = os.path.join(output_dir, f"{base_filename}_split_summary_{timestamp}.json")
            summary = {
                "timestamp": timestamp,
                "original_data": base_filename,
                "total_samples": len(train_df) + len(val_df) + len(test_df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "train_ratio": len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
                "val_ratio": len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
                "test_ratio": len(test_df) / (len(train_df) + len(val_df) + len(test_df)),
                "files": {
                    "train": os.path.basename(train_path),
                    "validation": os.path.basename(val_path),
                    "test": os.path.basename(test_path)
                }
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"分割摘要已保存: {summary_path}")
            
            return {
                "train_path": train_path,
                "val_path": val_path, 
                "test_path": test_path,
                "summary_path": summary_path
            }
            
        except Exception as e:
            self.logger.error(f"保存分割資料集失敗: {str(e)}")
            raise


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
    
    def process_amazon_files(self, file_paths, output_path=None, sample_size=10000):
        """處理多個Amazon數據文件，合併結果並抽樣
        
        Args:
            file_paths: Amazon數據文件路徑列表
            output_path: 輸出文件路徑，如果為None則自動生成
            sample_size: 抽樣數量，用於處理大型數據集
            
        Returns:
            pd.DataFrame: 處理後的數據
        """
        import random
        import time
        
        self.logger.info("===== Amazon數據處理開始 =====")
        for i, path in enumerate(file_paths):
            self.logger.info(f"輸入文件{i+1}: {path}")
        self.logger.info(f"抽樣數量: {sample_size}")
        
        # 設定隨機種子確保結果可重現
        random.seed(42)
        
        # 記錄開始時間
        start_time = time.time()
        
        # 讀取並合併所有文件
        dfs = []
        total_reviews = 0
        
        for file_path in file_paths:
            try:
                # 根據文件擴展名選擇加載方式
                file_ext = os.path.splitext(file_path)[1].lower()
                
                if file_ext == '.csv':
                    df = pd.read_csv(file_path)
                elif file_ext == '.json':
                    try:
                        df = pd.read_json(file_path)
                    except:
                        df = pd.read_json(file_path, lines=True)
                else:
                    self.logger.warning(f"不支持的文件類型: {file_ext}，跳過")
                    continue
                
                # 記錄讀取信息
                self.logger.info(f"成功讀取 {len(df)} 條評論，來自 {file_path}")
                total_reviews += len(df)
                
                # 將數據添加到列表
                dfs.append(df)
                
            except Exception as e:
                self.logger.error(f"讀取文件 {file_path} 時出錯: {str(e)}")
                continue
        
        # 如果沒有成功讀取任何數據，返回錯誤
        if not dfs:
            raise ValueError("沒有成功讀取任何數據")
        
        # 合併所有數據框
        self.logger.info("正在合併數據...")
        merged_df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"合併後共有 {len(merged_df)} 條評論")
        
        # 抽樣（如果數據量超過sample_size）
        if len(merged_df) > sample_size:
            self.logger.info(f"正在從 {len(merged_df)} 條評論中抽樣 {sample_size} 條...")
            merged_df = merged_df.sample(sample_size, random_state=42)
        
        # 預處理數據
        self.logger.info("正在進行數據預處理...")
        processed_df = self.preprocess(merged_df)
        
        # 如果提供了輸出路徑，則保存為CSV
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            self.logger.info(f"正在保存處理結果到 {output_path}")
            processed_df.to_csv(output_path, index=False)
        
        # 計算處理時間
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        
        # 記錄處理完成信息
        self.logger.info(f"===== 處理完成! 用時: {int(minutes)}分{int(seconds)}秒 =====")
        self.logger.info(f"從 {total_reviews} 條評論中處理並抽樣了 {len(processed_df)} 條")
        if output_path:
            self.logger.info(f"結果已保存至: {output_path}")
        
        return processed_df


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
        elif 'review_stars' in df.columns:
            df['rating'] = pd.to_numeric(df['review_stars'], errors='coerce')
        
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
        
    def process_yelp_files(self, business_path, review_path, output_path=None, sample_size=5000):
        """處理Yelp數據文件，合併business和review數據
        
        Args:
            business_path: Yelp business文件路徑
            review_path: Yelp review文件路徑
            output_path: 輸出文件路徑，如果為None則自動生成
            sample_size: 抽樣數量，用於處理大型數據集
            
        Returns:
            pd.DataFrame: 處理後的數據
        """
        import random
        import time
        
        self.logger.info("===== Yelp數據合併處理開始 =====")
        self.logger.info(f"Business文件: {business_path}")
        self.logger.info(f"Review文件: {review_path}")
        self.logger.info(f"抽樣數量: {sample_size}")
        
        # 設定隨機種子確保結果可重現
        random.seed(42)
        
        # 記錄開始時間
        start_time = time.time()
        
        # 第一步：讀取商家數據，只保留餐廳類別
        self.logger.info("正在讀取餐廳信息...")
        restaurants = {}
        restaurant_count = 0
        
        try:
            with open(business_path, 'r', encoding='utf-8') as f:
                for line in f:
                    business = json.loads(line)
                    # 只保留餐廳類別的商家
                    if business.get('categories') and 'Restaurants' in business['categories']:
                        # 提取需要的欄位
                        restaurants[business['business_id']] = {
                            'name': business.get('name', ''),
                            'city': business.get('city', ''),
                            'state': business.get('state', ''),
                            'stars': business.get('stars', 0),
                            'categories': business.get('categories', '')
                        }
                        restaurant_count += 1
            
            self.logger.info(f"已識別 {restaurant_count} 家餐廳")
            
            # 第二步：讀取評論數據並抽樣
            self.logger.info("開始處理評論數據...")
            sampled_reviews = []
            processed_count = 0
            eligible_count = 0
            
            with open(review_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        review = json.loads(line)
                        business_id = review['business_id']
                        
                        # 檢查評論是否屬於餐廳
                        if business_id in restaurants:
                            eligible_count += 1
                            
                            # 使用水塘抽樣算法確保均勻抽樣
                            if len(sampled_reviews) < sample_size:
                                # 直接添加，直到達到樣本大小
                                business = restaurants[business_id]
                                merged_review = {
                                    'business_id': business_id,
                                    'business_name': business['name'],
                                    'text': review.get('text', ''),
                                    'review_stars': review.get('stars', 0),
                                    'business_stars': business['stars'],
                                    'city': business['city'],
                                    'state': business['state'],
                                    'categories': business['categories'],
                                    'date': review.get('date', '')
                                }
                                sampled_reviews.append(merged_review)
                            else:
                                # 水塘抽樣：以 sample_size/已處理數量 的概率替換現有樣本
                                j = random.randint(0, eligible_count - 1)
                                if j < sample_size:
                                    business = restaurants[business_id]
                                    merged_review = {
                                        'business_id': business_id,
                                        'business_name': business['name'],
                                        'text': review.get('text', ''),
                                        'review_stars': review.get('stars', 0),
                                        'business_stars': business['stars'],
                                        'city': business['city'],
                                        'state': business['state'],
                                        'categories': business['categories'],
                                        'date': review.get('date', '')
                                    }
                                    sampled_reviews[j] = merged_review
                            
                        # 更新處理計數
                        processed_count += 1
                        if processed_count % 10000 == 0:
                            self.logger.info(f"已處理 {processed_count} 條評論，找到 {eligible_count} 條餐廳評論")
                            
                            # 如果已經找到足夠的樣本且處理了至少sample_size*10的評論，提前結束
                            if eligible_count >= sample_size * 10 and len(sampled_reviews) == sample_size:
                                self.logger.info(f"已收集足夠樣本，提前結束處理")
                                break
                            
                    except Exception as e:
                        self.logger.warning(f"處理評論時出錯: {str(e)}")
                        continue
            
            # 確保我們不超過所需的樣本數
            if len(sampled_reviews) > sample_size:
                sampled_reviews = sampled_reviews[:sample_size]
                
            self.logger.info(f"共處理了 {processed_count} 條評論")
            self.logger.info(f"其中有 {eligible_count} 條餐廳評論")
            self.logger.info(f"成功抽樣 {len(sampled_reviews)} 條評論")
            
            # 第三步：將抽樣的評論轉換為DataFrame
            df = pd.DataFrame(sampled_reviews)
            
            # 添加清洗後的文本列
            self.logger.info("正在進行文本清洗...")
            df['clean_text'] = df['text'].apply(self._clean_text)
            
            # 如果提供了輸出路徑，則保存為CSV
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                self.logger.info(f"正在保存處理結果到 {output_path}")
                df.to_csv(output_path, index=False)
            
            # 計算處理時間
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            
            # 記錄處理完成信息
            self.logger.info(f"===== 處理完成! 用時: {int(minutes)}分{int(seconds)}秒 =====")
            self.logger.info(f"總共處理了 {restaurant_count} 家餐廳")
            self.logger.info(f"從 {eligible_count} 條餐廳評論中抽樣了 {len(sampled_reviews)} 條")
            if output_path:
                self.logger.info(f"結果已保存至: {output_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Yelp數據處理失敗: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise


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