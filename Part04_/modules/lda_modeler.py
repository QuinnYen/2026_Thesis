"""
LDA主題建模模組 - 負責使用LDA算法進行主題建模和面向切割
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
import logging
import time
import tempfile
import random

# 設置臨時目錄環境變數
os.environ['JOBLIB_TEMP_FOLDER'] = tempfile.gettempdir()
os.environ['JOBLIB_MULTIPROCESSING'] = '0'  # 禁用多處理

# 導入系統日誌模組
from utils.logger import get_logger

# 導入主題標籤設定
from utils.settings.topic_labels import get_topic_labels

# 獲取logger
logger = get_logger("lda_modeler")

# 固定所有隨機種子，確保結果可重現
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class LDAModeler:
    """使用LDA算法進行主題建模和面向切割"""
    
    def __init__(self, config=None):
        """初始化LDA模型
        
        Args:
            config: 配置參數字典，可包含以下鍵:
                - num_topics: 主題數量
                - alpha: 文檔-主題分佈的先驗參數
                - eta: 主題-詞分佈的先驗參數
                - passes: 訓練通過次數
                - iterations: 迭代次數
                - random_state: 隨機種子
                - output_dir: 輸出目錄
                - dataset_name: 資料集名稱（用於獲取主題標籤）
                - language: 語言（'zh'或'en'）
        """
        self.config = config or {}
        self.logger = logger
        
        # 設置默認配置
        self.num_topics = self.config.get('num_topics', 10)
        # 將 'auto' 轉換為 None (sklearn LDA 不接受 'auto' 字符串)
        self.alpha = None if self.config.get('alpha', 'auto') == 'auto' else self.config.get('alpha')
        self.eta = None if self.config.get('eta', 'auto') == 'auto' else self.config.get('eta')
        self.passes = self.config.get('passes', 10)
        self.iterations = self.config.get('iterations', 50)
        
        # 固定隨機種子，確保結果可重現
        self.random_state = self.config.get('random_state', RANDOM_SEED)
        
        # 確保導入的模組也使用固定隨機種子
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        # 獲取當前檔案所在的Part04_目錄
        current_dir = os.path.dirname(os.path.abspath(__file__))
        part04_dir = os.path.dirname(current_dir)
        self.output_dir = self.config.get('output_dir', os.path.join(part04_dir, '1_output', 'topics'))
        
        # 保存資料集名稱和語言設定
        self.dataset_name = self.config.get('dataset_name', 'default')
        self.language = self.config.get('language', 'zh')
        
        # 獲取主題標籤
        self.topic_labels = get_topic_labels(
            dataset_name=self.dataset_name,
            language=self.language,
            num_topics=self.num_topics
        )
        
        # 延遲創建輸出目錄，直到實際需要時才創建
        # os.makedirs(self.output_dir, exist_ok=True)
        
        # 創建可視化目錄路徑
        self.vis_dir = os.path.join(self.output_dir, 'visualizations')
        # os.makedirs(self.vis_dir, exist_ok=True)
    
    def update_config(self, new_config):
        """更新LDA模型配置
        
        Args:
            new_config: 新的配置參數字典
        """
        if not new_config:
            return
            
        self.logger.info("更新LDA模型配置")
        
        # 更新配置字典
        if isinstance(new_config, dict):
            self.config.update(new_config)
            
            # 更新類屬性
            self.num_topics = self.config.get('num_topics', self.num_topics)
            self.alpha = None if self.config.get('alpha', self.alpha) == 'auto' else self.config.get('alpha', self.alpha)
            self.eta = None if self.config.get('eta', self.eta) == 'auto' else self.config.get('eta', self.eta)
            self.passes = self.config.get('passes', self.passes)
            self.iterations = self.config.get('iterations', self.iterations)
            self.random_state = self.config.get('random_state', self.random_state)
            self.output_dir = self.config.get('output_dir', self.output_dir)
            
            # 更新資料集名稱和語言設定
            if 'dataset_name' in new_config:
                self.dataset_name = new_config['dataset_name']
            if 'language' in new_config:
                self.language = new_config['language']
                
            # 更新主題標籤
            self.topic_labels = get_topic_labels(
                dataset_name=self.dataset_name,
                language=self.language,
                num_topics=self.num_topics
            )
            
            # 延遲創建輸出目錄，直到實際需要時才創建
            # os.makedirs(self.output_dir, exist_ok=True)
            # os.makedirs(self.vis_dir, exist_ok=True)
            
            self.logger.debug(f"LDA模型配置更新完成")
        else:
            self.logger.warning(f"無效的配置格式: {type(new_config)}")
            
    def get_topic_label(self, topic_id):
        """獲取指定主題ID的標籤名稱
        
        Args:
            topic_id: 主題ID
            
        Returns:
            str: 主題標籤名稱
        """
        if isinstance(topic_id, (int, np.integer)) and topic_id in self.topic_labels:
            return self.topic_labels[topic_id]
        else:
            # 如果找不到對應標籤，返回默認格式
            try:
                topic_id = int(topic_id)
                if topic_id in self.topic_labels:
                    return self.topic_labels[topic_id]
            except (ValueError, TypeError):
                pass
                
            # 無論當前語言設定如何，始終使用中文標籤格式
            return f"主題 {topic_id+1}"
    
    def train(self, texts, progress_callback=None):
        """訓練LDA模型 (與 build_topic_model 相容的介面方法)
        
        Args:
            texts: 文本列表
            progress_callback: 進度回調函數
        
        Returns:
            tuple: (lda_model, topics_dict) LDA模型和主題詞字典
        """
        try:
            self.logger.info(f"開始訓練LDA模型，使用 {self.num_topics} 個主題")
            
            # 判斷資料集類型 - 檢查電影相關詞彙來偵測是否為IMDB電影評論
            sample_texts = " ".join(texts[:min(100, len(texts))]).lower()
            if 'movie' in sample_texts or 'film' in sample_texts or 'actor' in sample_texts or 'cinema' in sample_texts:
                self.dataset_name = 'imdb'
                self.logger.info(f"自動檢測到電影評論資料集，使用IMDB標籤")
            elif 'restaurant' in sample_texts or 'food' in sample_texts or 'service' in sample_texts or 'menu' in sample_texts:
                self.dataset_name = 'yelp'
                self.logger.info(f"自動檢測到餐廳評論資料集，使用Yelp標籤")
            elif 'product' in sample_texts or 'purchase' in sample_texts or 'bought' in sample_texts or 'shipping' in sample_texts:
                self.dataset_name = 'amazon'
                self.logger.info(f"自動檢測到產品評論資料集，使用Amazon標籤")
            
            # 更新主題標籤
            self.topic_labels = get_topic_labels(
                dataset_name=self.dataset_name,
                language='zh',
                num_topics=self.num_topics
            )
            
            # 使用TF-IDF向量化文本
            self.logger.info(f"正在向量化文本...")
            if progress_callback:
                progress_callback(1, self.iterations, "向量化文本")
                
            # 使用CountVectorizer進行特徵抽取
            vectorizer = CountVectorizer(
                max_df=0.8,      # 過濾出現在超過80%文檔中的詞
                min_df=5,        # 過濾出現次數少於5次的詞
                max_features=10000,  # 最多使用10000個特徵
                ngram_range=(1, 1),  # 使用單詞
                stop_words='english' # 使用英文停用詞
            )
            
            # 轉換文本為詞頻矩陣
            X = vectorizer.fit_transform(texts)
            self.logger.info(f"創建了詞頻矩陣，形狀為: {X.shape}")
            
            # 獲取特徵名稱（詞彙表）
            feature_names = vectorizer.get_feature_names_out()
            
            # 構建LDA模型
            self.logger.info(f"正在訓練LDA模型...")
            if progress_callback:
                progress_callback(2, self.iterations, "初始化模型")
            
            # 設置LDA模型參數
            lda_model = LatentDirichletAllocation(
                n_components=self.num_topics,
                doc_topic_prior=self.alpha,
                topic_word_prior=self.eta,
                max_iter=self.iterations,
                learning_method='online',
                random_state=self.random_state,
                verbose=1,       # 啟用詳細輸出，但不使用callback
                n_jobs=1        # 使用所有可用CPU核心
            )
            
            # 不使用callback，直接訓練模型
            # 如果需要進度更新，可以手動實現
            if progress_callback:
                # 初始通知
                progress_callback(2, self.iterations, f"LDA迭代 0/{self.iterations}")
                
                # 分批訓練，手動更新進度
                for i in range(self.iterations):
                    if i == 0:
                        lda_model.fit_transform(X)
                    else:
                        lda_model.partial_fit(X)
                        
                    # 更新進度
                    progress_callback(i + 3, self.iterations, f"LDA迭代 {i+1}/{self.iterations}")
            else:
                # 正常訓練
                lda_model.fit(X)
                
            self.logger.info(f"LDA模型訓練完成")
            
            # 獲取主題-詞語分布，並正規化
            topic_word_distributions = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
            
            # 獲取每個主題的頂部詞語
            n_top_words = 15
            topic_top_words = {}
            
            for topic_idx, topic in enumerate(topic_word_distributions):
                # 獲取前n_top_words個詞
                top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_word_indices]
                
                # 獲取主題標籤 - 更有描述性的中文標籤
                topic_label = self.get_topic_label(topic_idx)
                
                # 不論語言設定，一律使用中文標籤格式："主題編號_中文標籤"
                topic_key = f"{topic_idx+1}_{topic_label}"
                    
                topic_top_words[topic_key] = top_words
                self.logger.info(f"{topic_key}: {', '.join(top_words[:10])}")
            
            # 完成進度
            if progress_callback:
                progress_callback(self.iterations, self.iterations, "LDA模型訓練完成")
                
            # 保存vectorizer作為實例變數，以便後續使用
            self.vectorizer = vectorizer
            self.feature_names = feature_names
            
            # 返回模型和主題詞
            return lda_model, topic_top_words
            
        except Exception as e:
            self.logger.error(f"訓練LDA模型時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def build_topic_model(self, data_path, text_column='tokens_str', progress_callback=None):
        """構建LDA主題模型
        
        Args:
            data_path: 處理後的數據文件路徑
            text_column: 文本列名
            progress_callback: 進度回調函數
            
        Returns:
            dict: 包含模型和結果的字典
        """
        try:
            start_time = time.time()
            self.logger.info(f"開始構建LDA主題模型，使用 {self.num_topics} 個主題")
            
            # 加載數據
            self.logger.info(f"正在加載數據...")
            if progress_callback:
                progress_callback("加載數據文件...", 10)
                
            df = pd.read_csv(data_path)
            
            # 檢查文本列是否存在
            if text_column not in df.columns:
                # 嘗試找到文本列
                if 'tokens_str' in df.columns:
                    text_column = 'tokens_str'
                elif 'clean_text' in df.columns:
                    text_column = 'clean_text'
                elif 'text' in df.columns:
                    text_column = 'text'
                else:
                    self.logger.error(f"找不到文本列 '{text_column}'，可用列: {', '.join(df.columns)}")
                    return None
                    
                self.logger.warning(f"指定的文本列 '{text_column}' 不存在，使用 '{text_column}' 代替")
            
            # 獲取文本列表
            texts = df[text_column].fillna('').tolist()
            self.logger.info(f"共 {len(texts)} 條文本")
            
            # 使用TF-IDF向量化文本
            self.logger.info(f"正在向量化文本...")
            if progress_callback:
                progress_callback("創建詞頻矩陣...", 20)
                
            # 使用TF-IDF進行特徵抽取，針對大型資料集優化
            vectorizer = TfidfVectorizer(
                max_df=0.8,      # 過濾出現在超過80%文檔中的詞
                min_df=5,        # 過濾出現次數少於5次的詞
                max_features=10000,  # 最多使用10000個特徵
                ngram_range=(1, 2),  # 使用單詞和雙詞組合
                stop_words='english' # 使用英文停用詞
            )
            
            # 轉換文本為TF-IDF矩陣
            X = vectorizer.fit_transform(texts)
            self.logger.info(f"創建了詞頻矩陣，形狀為: {X.shape}")
            
            # 獲取特徵名稱（詞彙表）
            feature_names = vectorizer.get_feature_names_out()
            
            # 構建LDA模型
            self.logger.info(f"正在訓練LDA模型...")
            if progress_callback:
                progress_callback("訓練LDA模型...", 40)
            
            # 設置LDA模型參數
            lda_model = LatentDirichletAllocation(
                n_components=self.num_topics,
                doc_topic_prior=self.alpha,
                topic_word_prior=self.eta,
                max_iter=self.iterations,
                learning_method='online',
                random_state=self.random_state,
                n_jobs=1  # 使用所有可用CPU核心
            )
            
            # 訓練模型
            lda_model.fit(X)
            self.logger.info(f"LDA模型訓練完成")
            
            # 獲取主題-詞語分布和詞頻矩陣
            self.logger.info(f"正在提取主題-詞語分布...")
            if progress_callback:
                progress_callback("提取主題分布...", 60)
                
            # 獲取主題-詞語分布，並正規化
            topic_word_distributions = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
            
            # 獲取每個主題的頂部詞語
            n_top_words = 15
            topic_top_words = {}
            
            for topic_idx, topic in enumerate(topic_word_distributions):
                # 獲取前n_top_words個詞
                top_word_indices = topic.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_word_indices]
                
                # 獲取主題標籤
                topic_label = self.get_topic_label(topic_idx)
                
                # 不論語言設定，一律使用中文標籤格式："主題編號_中文標籤"
                topic_key = f"{topic_idx+1}_{topic_label}"
                    
                topic_top_words[topic_key] = top_words
                self.logger.info(f"{topic_key}: {', '.join(top_words[:10])}")
            
            # 獲取文檔-主題分布
            self.logger.info(f"正在獲取文檔-主題分布...")
            if progress_callback:
                progress_callback("計算文檔-主題分布...", 70)
                
            doc_topic_distributions = lda_model.transform(X)
            
            # 獲取每個文檔的主要主題
            doc_main_topics = np.argmax(doc_topic_distributions, axis=1)
            
            # 為每個文檔分配主題標籤
            df['main_topic'] = [self.get_topic_label(topic_idx) for topic_idx in doc_main_topics]
            
            # 計算每個主題的文檔數量
            topic_counts = df['main_topic'].value_counts().to_dict()
            self.logger.info("主題分布統計:")
            for topic, count in topic_counts.items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"  {topic}: {count} 文檔 ({percentage:.2f}%)")
            
            # 保存模型和結果
            self.logger.info(f"正在保存模型和結果...")
            if progress_callback:
                progress_callback("保存模型和結果...", 80)
                
            # 確保輸出目錄存在
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)
                
            # 生成輸出文件路徑
            base_name = os.path.basename(data_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            
            # 保存LDA模型
            lda_model_path = os.path.join(self.output_dir, f"{base_name_without_ext}_lda_{self.num_topics}_topics.pkl")
            with open(lda_model_path, 'wb') as f:
                pickle.dump(lda_model, f)
            
            # 保存向量化器
            vectorizer_path = os.path.join(self.output_dir, f"{base_name_without_ext}_vectorizer.pkl")
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            # 保存主題詞分布
            topics_path = os.path.join(self.output_dir, f"{base_name_without_ext}_topics.json")
            with open(topics_path, 'w', encoding='utf-8') as f:
                json.dump(topic_top_words, f, ensure_ascii=False, indent=2)
            
            # 保存含主題標籤的元數據
            topic_metadata_path = os.path.join(self.output_dir, f"{base_name_without_ext}_with_topics.csv")
            df.to_csv(topic_metadata_path, index=False)
            
            # 保存主題統計數據
            stats_path = os.path.join(self.output_dir, f"{base_name_without_ext}_topic_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'topic_counts': topic_counts,
                    'n_docs': len(df),
                    'n_topics': self.num_topics,
                    'alpha': self.alpha,
                    'eta': self.eta,
                    'iterations': self.iterations
                }, f, ensure_ascii=False, indent=2)
            
            # 生成可視化
            self.logger.info(f"正在生成可視化...")
            if progress_callback:
                progress_callback("生成可視化...", 90)
                
            # 1. 主題詞雲可視化
            vis_path1 = self._generate_wordclouds(topic_top_words, base_name_without_ext)
            
            # 2. 主題詞條可視化
            vis_path2 = self._plot_top_words(lda_model, feature_names, n_top_words, base_name_without_ext)
            
            # 3. 文檔-主題分布可視化
            vis_path3 = self._plot_doc_topics(doc_topic_distributions, base_name_without_ext)
            
            # 4. 主題相關度可視化
            vis_path4 = self._plot_topic_similarity(lda_model, base_name_without_ext)
            
            visualizations = [p for p in [vis_path1, vis_path2, vis_path3, vis_path4] if p]
            
            # 計算處理時間
            elapsed_time = time.time() - start_time
            self.logger.info(f"LDA主題模型構建完成，耗時: {elapsed_time:.2f} 秒")
            
            if progress_callback:
                progress_callback("LDA主題模型構建完成", 100)
            
            # 返回結果
            return {
                'lda_model_path': lda_model_path,
                'vectorizer_path': vectorizer_path,
                'topics_path': topics_path,
                'topic_metadata_path': topic_metadata_path,
                'topic_stats_path': stats_path,
                'visualizations': visualizations,
                'topic_counts': topic_counts,
                'feature_names': feature_names.tolist(),
                'n_topics': self.num_topics,
                'topic_words': topic_top_words
            }
            
        except Exception as e:
            self.logger.error(f"構建LDA主題模型時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _generate_wordclouds(self, topic_words, base_name):
        """為每個主題生成詞雲
        
        Args:
            topic_words: 主題-詞語字典
            base_name: 基礎文件名
            
        Returns:
            str: 保存的圖像路徑
        """
        try:
            # 創建詞雲圖像
            plt.figure(figsize=(15, int(5 * (len(topic_words) + 1) / 2)))
            
            for i, (topic, words) in enumerate(topic_words.items()):
                # 創建詞頻字典
                word_freq = {word: 1/(j+1) for j, word in enumerate(words)}
                
                # 創建詞雲
                plt.subplot(int((len(topic_words)+1)/2), 2, i+1)
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    background_color='white',
                    max_words=50,
                    colormap='viridis',
                    random_state=42
                ).generate_from_frequencies(word_freq)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(topic, fontsize=14)
                plt.axis("off")
            
            plt.tight_layout()
            
            # 保存圖像
            output_path = os.path.join(self.vis_dir, f"{base_name}_topic_wordclouds.png")
            plt.savefig(output_path, dpi=200)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"生成詞雲可視化時出錯: {str(e)}")
            return None
    
    def _plot_top_words(self, lda_model, feature_names, n_top_words, base_name):
        """繪製每個主題的頂部詞語
        
        Args:
            lda_model: LDA模型
            feature_names: 特徵名稱（詞彙表） 
            n_top_words: 每個主題顯示的詞語數量
            base_name: 基礎文件名
            
        Returns:
            str: 保存的圖像路徑
        """
        try:
            # 設定畫布大小
            plt.figure(figsize=(15, int(3 * self.num_topics / 2)))
            
            for topic_idx, topic in enumerate(lda_model.components_):
                # 獲取頂部詞語
                top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                weights = topic[top_features_ind]
                
                # 繪製條形圖
                ax = plt.subplot(int((self.num_topics+1)/2), 2, topic_idx+1)
                ax.barh(top_features, weights)
                
                # 設置標題
                topic_label = self.get_topic_label(topic_idx)
                ax.set_title(f'主題 {topic_idx+1}: {topic_label}', fontsize=12)
                
                # 設置字體大小
                ax.tick_params(axis='y', labelsize=10)
            
            plt.tight_layout()
            
            # 保存圖像
            output_path = os.path.join(self.vis_dir, f"{base_name}_topic_words.png")
            plt.savefig(output_path, dpi=200)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"繪製主題詞語可視化時出錯: {str(e)}")
            return None
    
    def _plot_doc_topics(self, doc_topic_dist, base_name):
        """繪製文檔-主題分布
        
        Args:
            doc_topic_dist: 文檔-主題分布矩陣
            base_name: 基礎文件名
            
        Returns:
            str: 保存的圖像路徑
        """
        try:
            # 計算每個主題的文檔數量
            topic_counts = np.zeros(self.num_topics)
            doc_main_topics = np.argmax(doc_topic_dist, axis=1)
            
            for topic_idx in range(self.num_topics):
                topic_counts[topic_idx] = np.sum(doc_main_topics == topic_idx)
            
            # 繪製條形圖
            plt.figure(figsize=(12, 7))
            x = np.arange(self.num_topics)
            plt.bar(x, topic_counts)
            
            # 使用自定義標籤
            x_labels = [self.get_topic_label(i) for i in range(self.num_topics)]
            
            plt.xlabel('主題')
            plt.ylabel('文檔數量')
            plt.title('文檔在各主題中的分佈')
            plt.xticks(x, x_labels, rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存圖像
            output_path = os.path.join(self.vis_dir, f"{base_name}_doc_topics.png")
            plt.savefig(output_path, dpi=200)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"繪製文檔-主題分布可視化時出錯: {str(e)}")
            return None
    
    def _plot_topic_similarity(self, lda_model, base_name):
        """繪製主題相似度矩陣
        
        Args:
            lda_model: LDA模型
            base_name: 基礎文件名
            
        Returns:
            str: 保存的圖像路徑
        """
        try:
            # 計算主題之間的相似度矩陣
            n_topics = lda_model.n_components
            similarity_matrix = np.zeros((n_topics, n_topics))
            
            # 使用詞分布的餘弦相似度
            for i in range(n_topics):
                for j in range(n_topics):
                    # 計算兩個主題詞分布的餘弦相似度
                    topic_i = lda_model.components_[i]
                    topic_j = lda_model.components_[j]
                    
                    # 正規化向量
                    topic_i_norm = topic_i / np.sqrt(np.sum(topic_i ** 2))
                    topic_j_norm = topic_j / np.sqrt(np.sum(topic_j ** 2))
                    
                    # 計算餘弦相似度
                    similarity = np.dot(topic_i_norm, topic_j_norm)
                    similarity_matrix[i, j] = similarity
            
            # 繪製熱力圖
            plt.figure(figsize=(10, 8))
            plt.imshow(similarity_matrix, cmap='viridis')
            plt.colorbar(label='餘弦相似度')
            plt.title('主題間相似度矩陣')
            plt.xlabel('主題')
            plt.ylabel('主題')
            
            # 使用自定義標籤
            topic_label_texts = [self.get_topic_label(i) for i in range(n_topics)]
                
            plt.xticks(range(n_topics), topic_label_texts, rotation=45, ha='right')
            plt.yticks(range(n_topics), topic_label_texts)
            
            # 在每個單元格中添加數值
            for i in range(n_topics):
                for j in range(n_topics):
                    plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha='center', va='center', 
                           color='white' if similarity_matrix[i, j] > 0.5 else 'black')
            
            plt.tight_layout()
            
            # 保存圖像
            output_path = os.path.join(self.vis_dir, f"{base_name}_topic_similarity.png")
            plt.savefig(output_path, dpi=200)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"繪製主題相似度可視化時出錯: {str(e)}")
            return None
    
    def evaluate_topics(self, data_path, min_topics=5, max_topics=15, step=1, progress_callback=None):
        """評估不同主題數量的LDA模型性能
        
        Args:
            data_path: 數據文件路徑
            min_topics: 最小主題數量
            max_topics: 最大主題數量
            step: 步長
            progress_callback: 進度回調函數
            
        Returns:
            dict: 包含評估結果的字典
        """
        try:
            self.logger.info(f"開始評估不同主題數量的LDA模型")
            
            # 加載數據
            self.logger.info(f"正在加載數據...")
            if progress_callback:
                progress_callback("加載數據文件...", 10)
                
            df = pd.read_csv(data_path)
            
            # 嘗試找到文本列
            text_column = None
            for col in ['tokens_str', 'clean_text', 'text']:
                if col in df.columns:
                    text_column = col
                    break
                    
            if text_column is None:
                self.logger.error(f"找不到可用的文本列，可用列: {', '.join(df.columns)}")
                return None
                
            self.logger.info(f"使用 '{text_column}' 作為文本列")
            
            # 獲取文本列表
            texts = df[text_column].fillna('').tolist()
            
            # 使用TF-IDF向量化文本
            self.logger.info(f"正在向量化文本...")
            vectorizer = TfidfVectorizer(
                max_df=0.8, 
                min_df=5, 
                max_features=10000, 
                ngram_range=(1, 2),
                stop_words='english'
            )
            X = vectorizer.fit_transform(texts)
            
            # 評估不同主題數量
            self.logger.info(f"正在評估主題數量範圍: {min_topics}~{max_topics}")
            
            perplexity_scores = {}
            coherence_scores = {}
            
            for n_topics in range(min_topics, max_topics + 1, step):
                # 更新進度
                progress = 10 + int((n_topics - min_topics) / (max_topics - min_topics + 1) * 80)
                if progress_callback:
                    progress_callback(f"測試 {n_topics} 個主題...", progress)
                
                # 訓練LDA模型
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    doc_topic_prior=self.alpha,
                    topic_word_prior=self.eta,
                    max_iter=10,  # 使用較小的迭代次數加速評估
                    learning_method='online',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                lda.fit(X)
                
                # 計算困惑度
                perplexity = lda.perplexity(X)
                perplexity_scores[n_topics] = perplexity
                
                # 使用負困惑度作為一致性指標
                coherence_scores[n_topics] = -perplexity
                
                self.logger.info(f"主題數量: {n_topics}, 困惑度: {perplexity:.2f}")
            
            # 找到最佳主題數量
            best_n_topics = min(perplexity_scores, key=perplexity_scores.get)
            self.logger.info(f"最佳主題數量: {best_n_topics}, 困惑度: {perplexity_scores[best_n_topics]:.2f}")
            
            # 繪製困惑度曲線
            self.logger.info(f"正在繪製困惑度曲線...")
            if progress_callback:
                progress_callback("繪製困惑度曲線...", 90)
                
            plt.figure(figsize=(10, 6))
            plt.plot(list(perplexity_scores.keys()), list(perplexity_scores.values()), 'o-')
            plt.xlabel('主題數量')
            plt.ylabel('困惑度 (越低越好)')
            plt.title('不同主題數量的LDA模型困惑度')
            plt.grid(True)
            
            base_name = os.path.basename(data_path).split('.')[0]
            vis_path = os.path.join(self.vis_dir, f"{base_name}_perplexity.png")
            plt.savefig(vis_path, dpi=200)
            plt.close()
            
            # 保存評估結果
            results_path = os.path.join(self.output_dir, f"{base_name}_topic_evaluation.json")
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'perplexity_scores': perplexity_scores,
                    'best_n_topics': best_n_topics,
                    'alpha': self.alpha,
                    'eta': self.eta
                }, f, ensure_ascii=False, indent=2)
            
            if progress_callback:
                progress_callback("主題評估完成", 100)
            
            return {
                'perplexity_scores': perplexity_scores,
                'coherence_scores': coherence_scores,
                'best_n_topics': best_n_topics,
                'visualization_path': vis_path,
                'results_path': results_path
            }
            
        except Exception as e:
            self.logger.error(f"評估主題時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


def build_lda_model(data_path, output_dir='./output/topics', num_topics=10, 
                  alpha='auto', eta='auto', iterations=50, dataset_name='default', language='zh', progress_callback=None):
    """構建LDA主題模型的方便函數
    
    Args:
        data_path: 數據文件路徑
        output_dir: 輸出目錄
        num_topics: 主題數量
        alpha: 文檔-主題分佈的先驗參數
        eta: 主題-詞分佈的先驗參數
        iterations: 最大迭代次數
        dataset_name: 資料集名稱
        language: 語言（'zh'或'en'）
        progress_callback: 進度回調函數
        
    Returns:
        dict: 包含結果的字典
    """
    # 創建配置
    config = {
        'num_topics': num_topics,
        'alpha': alpha,
        'eta': eta,
        'iterations': iterations,
        'output_dir': output_dir,
        'dataset_name': dataset_name,
        'language': language
    }
    
    # 創建LDA建模器
    modeler = LDAModeler(config)
    
    # 構建主題模型
    return modeler.build_topic_model(data_path, progress_callback=progress_callback)


# 測試代碼
if __name__ == "__main__":
    # 配置日誌級別
    logger.setLevel(logging.INFO)
    
    # 測試LDA主題建模
    test_file = "path/to/your/processed_data.csv"
    
    if os.path.exists(test_file):
        # 定義進度回調函數
        def progress_callback(message, percentage):
            print(f"{message} - {percentage}%")
        
        # 構建LDA模型
        result = build_lda_model(
            test_file,
            num_topics=10,
            dataset_name='example_dataset',
            language='zh',
            progress_callback=progress_callback
        )
        
        if result:
            print(f"LDA模型已保存至: {result['lda_model_path']}")
            print(f"主題詞已保存至: {result['topics_path']}")
        else:
            print("構建LDA模型失敗")
    else:
        logger.warning(f"測試文件不存在: {test_file}")