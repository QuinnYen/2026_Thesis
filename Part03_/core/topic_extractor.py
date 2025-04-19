"""
主題提取模組
此模組負責使用LDA模型進行主題提取
"""

import os
import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
import logging
import time
import sys
from pathlib import Path
import itertools
import jieba
import re
import gensim
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 從utils模組導入工具類
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.config_manager import ConfigManager

# 設置jieba日誌級別
jieba.setLogLevel(logging.INFO)

class TopicExtractor:
    """
    主題提取器類
    使用LDA模型從文本數據中提取主題
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化主題提取器
        
        Args:
            config: 配置管理器，如果沒有提供則創建新的
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 從配置中獲取參數
        self.num_topics = self.config.get('model_settings.lda.num_topics', 10)
        self.passes = self.config.get('model_settings.lda.passes', 15)
        self.alpha = self.config.get('model_settings.lda.alpha', 'auto')
        self.workers = self.config.get('model_settings.lda.workers', 4)
        
        # 設置輸出目錄
        self.output_dir = self.config.get('data_settings.output_directory', './Part03_/results/')
        self.lda_dir = os.path.join(self.output_dir, '03_lda_topics')
        os.makedirs(self.lda_dir, exist_ok=True)
        
        # 初始化模型相關屬性
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.topic_terms = None
        self.id2word = None
        
        # 初始化日誌
        log_dir = self.config.get('logging.log_dir', './Part03_/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'topic_extractor.log')
        
        self.logger = logging.getLogger('topic_extractor')
        self.logger.setLevel(logging.INFO)
        
        # 移除所有處理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 添加文件處理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # 添加控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
    
    def preprocess_text(self, text: str, min_length: int = 2) -> List[str]:
        """
        對文本進行預處理，返回分詞結果
        
        Args:
            text: 輸入文本
            min_length: 最小詞長度
            
        Returns:
            分詞列表
        """
        if not isinstance(text, str) or not text.strip():
            return []
        
        # 將文本轉為小寫並移除多餘空白
        text = text.lower().strip()
        
        # 使用正則表達式檢測語言
        is_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        
        # 根據語言選擇分詞方式
        if is_chinese:
            # 中文分詞
            words = [word for word in jieba.cut(text) if len(word) >= min_length and not word.isdigit()]
            # 過濾停用詞（可以自定義停用詞表）
            stop_words = {'的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '或', '一個', '沒有', '因為', '但是', '所以'}
            words = [word for word in words if word not in stop_words]
        else:
            # 英文分詞
            words = simple_preprocess(text, min_len=min_length)
            # 過濾英文停用詞
            from gensim.parsing.preprocessing import STOPWORDS
            words = [word for word in words if word not in STOPWORDS]
        
        return words
    
    def prepare_corpus(self, texts: List[str], no_below: int = 5, no_above: float = 0.5) -> Tuple[List[List[int]], Dictionary]:
        """
        準備語料庫和字典
        
        Args:
            texts: 文本列表
            no_below: 詞至少在多少文檔中出現
            no_above: 詞最多在多少比例的文檔中出現
            
        Returns:
            (corpus, dictionary): 語料庫和字典
        """
        # 分詞
        self.logger.info(f"開始處理 {len(texts)} 條文本...")
        tokenized_texts = [self.preprocess_text(text) for text in texts]
        
        # 移除空文本
        tokenized_texts = [tokens for tokens in tokenized_texts if tokens]
        
        # 創建字典
        self.logger.info("創建詞彙字典...")
        dictionary = Dictionary(tokenized_texts)
        initial_tokens = len(dictionary)
        
        # 過濾極端詞頻的詞
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        filtered_tokens = len(dictionary)
        
        self.logger.info(f"從 {initial_tokens} 個詞彙中過濾得到 {filtered_tokens} 個有效詞彙")
        
        # 創建語料庫
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        self.logger.info(f"語料庫包含 {len(corpus)} 個文檔")
        
        return corpus, dictionary
    
    def train_lda_model(self, corpus: List[List[int]], dictionary: Dictionary, 
                      num_topics: int = None, passes: int = None,
                      alpha: str = None, iterations: int = 50) -> LdaModel:
        """
        訓練LDA模型
        
        Args:
            corpus: 文檔-詞語語料庫
            dictionary: 詞彙字典
            num_topics: 主題數
            passes: 訓練遍數
            alpha: 主題分布先驗參數
            iterations: 每次遍歷的迭代次數
            
        Returns:
            訓練好的LDA模型
        """
        # 使用參數或默認值
        num_topics = num_topics or self.num_topics
        passes = passes or self.passes
        alpha = alpha or self.alpha
        
        self.logger.info(f"開始訓練LDA模型，主題數: {num_topics}，訓練遍數: {passes}...")
        
        # 訓練LDA模型
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha=alpha,
            iterations=iterations,
            random_state=42
        )
        
        self.logger.info("LDA模型訓練完成")
        
        return lda_model
    
    def extract_topics(self, df: pd.DataFrame, text_column: str = 'processed_text',
                     num_topics: int = None, no_below: int = 5, no_above: float = 0.5,
                     num_words: int = 10, console_output: bool = True) -> Dict[str, Any]:
        """
        從數據集中提取主題
        
        Args:
            df: 輸入數據框
            text_column: 文本列名
            num_topics: 主題數量
            no_below: 詞至少在多少文檔中出現
            no_above: 詞最多在多少比例的文檔中出現
            num_words: 每個主題顯示的關鍵詞數量
            console_output: 是否顯示處理進度
            
        Returns:
            主題模型結果字典
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("LDA面向切割")
            logger = ConsoleOutputManager.setup_console_logger("lda_topics", log_file)
        else:
            logger = self.logger
        
        try:
            # 確保文本列存在
            if text_column not in df.columns:
                error_msg = f"數據集中不存在列 '{text_column}'"
                logger.error(error_msg)
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                raise ValueError(error_msg)
            
            logger.info("開始提取主題...")
            
            # 獲取文本
            texts = df[text_column].tolist()
            
            # 準備語料庫
            logger.info("準備語料庫和字典...")
            self.corpus, self.dictionary = self.prepare_corpus(
                texts, no_below=no_below, no_above=no_above
            )
            self.id2word = self.dictionary
            
            # 訓練模型
            num_topics = num_topics or self.num_topics
            logger.info(f"訓練LDA模型，主題數: {num_topics}")
            self.lda_model = self.train_lda_model(self.corpus, self.dictionary, num_topics=num_topics)
            
            # 提取主題詞
            topics = []
            for i in range(num_topics):
                topic_terms = self.lda_model.get_topic_terms(i, topn=num_words)
                topic_words = [(self.dictionary[id], prob) for id, prob in topic_terms]
                topics.append({
                    'id': i,
                    'words': topic_words,
                    'top_words': [word for word, _ in topic_words]
                })
                
                # 顯示主題詞
                word_prob_str = ", ".join([f"{word} ({prob:.3f})" for word, prob in topic_words])
                logger.info(f"主題 {i+1}: {word_prob_str}")
            
            self.topic_terms = topics
            
            # 計算模型評估指標
            logger.info("計算模型評估指標...")
            coherence_model = CoherenceModel(
                model=self.lda_model, 
                texts=[self.preprocess_text(text) for text in texts], 
                dictionary=self.dictionary, 
                coherence='c_v'
            )
            coherence = coherence_model.get_coherence()
            logger.info(f"主題一致性分數 (CV): {coherence:.4f}")
            
            # 計算每個文檔的主題分布
            logger.info("計算文檔-主題分布...")
            doc_topics = []
            for doc_bow in self.corpus:
                doc_topic_dist = self.lda_model.get_document_topics(doc_bow)
                # 將稀疏表示轉為密集向量
                dense_vec = np.zeros(num_topics)
                for topic_id, prob in doc_topic_dist:
                    dense_vec[topic_id] = prob
                doc_topics.append(dense_vec)
            
            # 準備返回的結果
            result = {
                'lda_model': self.lda_model,
                'dictionary': self.dictionary,
                'corpus': self.corpus,
                'topics': topics,
                'coherence': coherence,
                'doc_topics': np.array(doc_topics),
                'num_topics': num_topics
            }
            
            logger.info("主題提取完成")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return result
            
        except Exception as e:
            logger.error(f"主題提取過程中發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def save_model(self, dataset_name: str) -> str:
        """
        保存LDA模型和相關數據
        
        Args:
            dataset_name: 數據集名稱
            
        Returns:
            保存的文件路徑
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"{dataset_name}_lda_{self.num_topics}_{timestamp}"
        
        # 保存模型
        model_path = os.path.join(self.lda_dir, f"{base_filename}_model")
        self.lda_model.save(model_path)
        
        # 保存字典
        dict_path = os.path.join(self.lda_dir, f"{base_filename}_dict")
        self.dictionary.save(dict_path)
        
        # 保存主題詞
        topics_path = os.path.join(self.lda_dir, f"{base_filename}_topics.pkl")
        with open(topics_path, 'wb') as f:
            pickle.dump(self.topic_terms, f)
        
        # 保存語料庫
        corpus_path = os.path.join(self.lda_dir, f"{base_filename}_corpus.pkl")
        with open(corpus_path, 'wb') as f:
            pickle.dump(self.corpus, f)
        
        self.logger.info(f"模型和相關數據已保存到: {self.lda_dir}")
        
        # 返回模型路徑
        return model_path
    
    def load_model(self, model_path: str, dict_path: str = None) -> None:
        """
        載入LDA模型和字典
        
        Args:
            model_path: 模型文件路徑
            dict_path: 字典文件路徑（如果不提供，則推斷）
        """
        try:
            # 如果沒有提供字典路徑，嘗試推斷
            if not dict_path and model_path.endswith('_model'):
                dict_path = model_path.replace('_model', '_dict')
            
            # 載入模型
            self.lda_model = LdaModel.load(model_path)
            self.logger.info(f"已載入模型: {model_path}")
            
            # 載入字典
            if dict_path and os.path.exists(dict_path):
                self.dictionary = Dictionary.load(dict_path)
                self.id2word = self.dictionary
                self.logger.info(f"已載入字典: {dict_path}")
            else:
                self.logger.warning("未提供字典文件路徑或文件不存在")
            
            # 設置主題數
            self.num_topics = self.lda_model.num_topics
            
        except Exception as e:
            self.logger.error(f"載入模型時發生錯誤: {str(e)}")
            raise
    
    def get_document_topics(self, text: Union[str, List[str]], minimum_probability: float = 0.01) -> Union[np.ndarray, List[np.ndarray]]:
        """
        獲取文本的主題分布
        
        Args:
            text: 輸入文本或文本列表
            minimum_probability: 最小概率閾值
            
        Returns:
            主題概率分布（單個文本）或主題概率分布列表（多個文本）
        """
        # 確保模型已載入
        if not self.lda_model or not self.dictionary:
            raise RuntimeError("模型或字典未載入，無法進行主題分析")
        
        # 處理單個文本
        if isinstance(text, str):
            tokens = self.preprocess_text(text)
            bow = self.dictionary.doc2bow(tokens)
            
            # 獲取主題分布
            topic_dist = self.lda_model.get_document_topics(bow, minimum_probability=minimum_probability)
            
            # 轉換為密集向量
            dense_vec = np.zeros(self.num_topics)
            for topic_id, prob in topic_dist:
                dense_vec[topic_id] = prob
                
            return dense_vec
            
        # 處理文本列表
        elif isinstance(text, list):
            results = []
            for t in text:
                results.append(self.get_document_topics(t, minimum_probability))
            return results
        
        else:
            raise TypeError("輸入必須是字符串或字符串列表")
    
    def get_topic_keywords(self, topic_id: int, num_words: int = 10) -> List[Tuple[str, float]]:
        """
        獲取指定主題的關鍵詞
        
        Args:
            topic_id: 主題ID
            num_words: 返回的關鍵詞數量
            
        Returns:
            關鍵詞和權重的列表
        """
        # 確保模型已載入
        if not self.lda_model or not self.dictionary:
            raise RuntimeError("模型或字典未載入，無法獲取主題關鍵詞")
        
        # 確保主題ID有效
        if topic_id < 0 or topic_id >= self.num_topics:
            raise ValueError(f"主題ID無效，必須在0到{self.num_topics-1}之間")
        
        # 獲取主題關鍵詞
        topic_terms = self.lda_model.get_topic_terms(topic_id, topn=num_words)
        keywords = [(self.dictionary[id], prob) for id, prob in topic_terms]
        
        return keywords
    
    def generate_topic_labels(self, topics_keywords: List[List[Tuple[str, float]]]) -> List[str]:
        """
        根據主題關鍵詞自動生成主題標籤
        
        Args:
            topics_keywords: 每個主題的關鍵詞和權重列表
            
        Returns:
            主題標籤列表
        """
        labels = []
        for i, keywords in enumerate(topics_keywords):
            # 使用前兩個關鍵詞作為標籤
            if len(keywords) > 1:
                top_words = [word for word, _ in keywords[:2]]
                label = f"{top_words[0]}+{top_words[1]}"
            elif len(keywords) == 1:
                label = keywords[0][0]
            else:
                label = f"主題{i+1}"
            
            labels.append(label)
        
        return labels
    
    def find_optimal_topics(self, texts: List[str], start: int = 2, limit: int = 20, 
                         step: int = 2, coherence_method: str = 'c_v',
                         console_output: bool = True) -> Dict[str, Any]:
        """
        找出最佳的主題數量
        
        Args:
            texts: 文本列表
            start: 起始主題數
            limit: 最大主題數
            step: 每次增加的主題數
            coherence_method: 一致性計算方法，可選 'c_v', 'u_mass', 'c_uci', 'c_npmi'
            console_output: 是否顯示處理進度
            
        Returns:
            包含評估結果的字典
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("LDA參數評估")
            logger = ConsoleOutputManager.setup_console_logger("lda_evaluation", log_file)
        else:
            logger = self.logger
        
        try:
            logger.info("開始LDA最優主題數評估...")
            
            # 準備語料庫
            logger.info("準備語料庫和字典...")
            corpus, dictionary = self.prepare_corpus(texts)
            
            # 分詞
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            # 儲存結果
            coherence_values = []
            model_list = []
            
            # 嘗試不同的主題數
            for num_topics in range(start, limit + 1, step):
                logger.info(f"評估主題數: {num_topics}")
                
                # 訓練LDA模型
                lda_model = self.train_lda_model(
                    corpus, dictionary, num_topics=num_topics, 
                    passes=self.passes, alpha=self.alpha
                )
                model_list.append(lda_model)
                
                # 計算一致性評分
                coherence_model = CoherenceModel(
                    model=lda_model, texts=processed_texts, 
                    dictionary=dictionary, coherence=coherence_method
                )
                coherence = coherence_model.get_coherence()
                coherence_values.append(coherence)
                
                logger.info(f"主題數 {num_topics}，一致性分數: {coherence:.4f}")
            
            # 找出最佳的主題數
            optimal_index = np.argmax(coherence_values)
            optimal_topics = start + optimal_index * step
            max_coherence = coherence_values[optimal_index]
            
            logger.info(f"最佳主題數: {optimal_topics}，一致性分數: {max_coherence:.4f}")
            
            # 生成評估圖表
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = list(range(start, limit + 1, step))
            ax.plot(x, coherence_values, 'o-', color='blue')
            ax.set_xlabel('主題數')
            ax.set_ylabel('一致性分數')
            ax.set_title('LDA模型最優主題數評估')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 標記最佳點
            ax.plot(optimal_topics, max_coherence, 'ro', ms=10)
            ax.annotate(f'最佳主題數: {optimal_topics}\n分數: {max_coherence:.4f}',
                       xy=(optimal_topics, max_coherence),
                       xytext=(optimal_topics + 2, max_coherence - 0.05),
                       arrowprops=dict(arrowstyle='->'))
            
            # 保存圖表
            chart_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(chart_dir, exist_ok=True)
            chart_path = os.path.join(chart_dir, f'lda_evaluation_{time.strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"評估圖表已保存到: {chart_path}")
            
            # 創建返回結果
            result = {
                'optimal_num_topics': optimal_topics,
                'coherence_values': coherence_values,
                'topic_numbers': x,
                'optimal_model': model_list[optimal_index],
                'optimal_coherence': max_coherence,
                'chart_path': chart_path
            }
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return result
            
        except Exception as e:
            logger.error(f"LDA參數評估過程中發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def generate_pyLDAvis(self, output_file: str = None) -> str:
        """
        生成pyLDAvis可視化
        
        Args:
            output_file: 輸出HTML文件路徑（可選）
            
        Returns:
            生成的HTML文件路徑
        """
        # 確保模型、字典和語料庫都已載入
        if not self.lda_model or not self.dictionary or not self.corpus:
            raise RuntimeError("模型、字典或語料庫未載入，無法生成可視化")
        
        # 如果未指定輸出路徑，設置默認路徑
        if not output_file:
            vis_dir = os.path.join(self.output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(vis_dir, f'pyldavis_{timestamp}.html')
        
        # 生成可視化
        self.logger.info("正在生成pyLDAvis可視化...")
        
        vis_data = gensimvis.prepare(
            self.lda_model, self.corpus, self.dictionary, sort_topics=False
        )
        
        # 保存到HTML文件
        pyLDAvis.save_html(vis_data, output_file)
        
        self.logger.info(f"pyLDAvis可視化已保存到: {output_file}")
        
        return output_file

# 使用示例
if __name__ == "__main__":
    # 初始化主題提取器
    topic_extractor = TopicExtractor()
    
    # 模擬一些示例文本
    texts = [
        "這家餐廳的食物非常美味，服務也很好",
        "菜品價格合理，但服務態度需要改進",
        "環境很舒適，但食物味道一般",
        "這家店的服務員態度很好，食物也不錯",
        "價格有點貴，但是食物品質很高",
        "餐廳環境優美，適合約會",
        "服務速度慢，但是食物很美味"
    ]
    
    # 將示例文本轉換為DataFrame
    df = pd.DataFrame({'text': texts})
    df['processed_text'] = df['text']
    
    # 提取主題
    topics_result = topic_extractor.extract_topics(
        df, text_column='processed_text', num_topics=3
    )
    
    print(f"主題一致性分數: {topics_result['coherence']:.4f}")
    
    # 查看每個文本的主題分布
    doc_topics = topics_result['doc_topics']
    for i, (text, topics) in enumerate(zip(texts, doc_topics)):
        dominant_topic = np.argmax(topics)
        prob = topics[dominant_topic]
        print(f"文本 {i+1}: {text[:30]}... | 主要主題: {dominant_topic+1}, 概率: {prob:.2f}")
