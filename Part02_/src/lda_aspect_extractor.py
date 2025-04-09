"""
LDA主題提取器
"""

import numpy as np
import pandas as pd
import os
import logging
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.settings.visualization_config import apply_chinese_to_plot, format_topic_labels
import pickle
import json
import re
import time
import tempfile

# 設置 joblib 臨時目錄為僅包含 ASCII 字符的路徑
temp_dir = tempfile.gettempdir()
os.environ['JOBLIB_TEMP_FOLDER'] = os.path.join(temp_dir, 'joblib_tmp')

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('lda_aspect_extractor')

class LDATopicExtractor:
    """使用LDA算法進行面向切割"""
    
    def __init__(self, output_dir='./Part02_/results', logger=None):
        """
        初始化LDA面向提取器
        
        Args:
            output_dir: 輸出目錄
            logger: 日誌器
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 創建模型和可視化子目錄
        self.models_dir = os.path.join(output_dir, 'models')
        self.vis_dir = os.path.join(output_dir, 'visualizations')
        
        for directory in [self.models_dir, self.vis_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def log(self, message, level=logging.INFO):
        """統一的日誌處理方法"""
        if self.logger:
            self.logger.log(level, message)
    
    def run_lda(self, metadata_path, n_topics=10, text_column='tokens_str', topic_labels=None, 
            doc_topic_prior=None, topic_word_prior=None, max_iter=None, callback=None):
        """
        執行LDA主題建模來識別面向 - 加入更多參數選項
        
        Args:
            metadata_path: BERT元數據文件路徑（CSV）
            n_topics: 主題數量
            text_column: 文本列名
            topic_labels: 主題標籤字典，鍵為主題索引，值為標籤名稱 (可選)
            doc_topic_prior: 文檔主題先驗分布的濃度參數（alpha值）
            topic_word_prior: 主題詞先驗分布的濃度參數（beta值）
            max_iter: 最大迭代次數
            callback: 進度回調函數
                
        Returns:
            dict: 包含LDA模型和相關結果的字典
        """
        try:
            self.log(f"Starting LDA topic modeling with {n_topics} topics")
            self.log(f"Using metadata from: {metadata_path}")
            self.log(f"Text column: {text_column}")
            
            # 記錄額外參數
            if doc_topic_prior is not None:
                self.log(f"Alpha (doc_topic_prior): {doc_topic_prior}")
            if topic_word_prior is not None:
                self.log(f"Beta (topic_word_prior): {topic_word_prior}")
            if max_iter is not None:
                self.log(f"Max iterations: {max_iter}")
            
            # 讀取數據
            if callback:
                callback("Loading metadata...", 10)
            
            self.log(f"Reading metadata from: {metadata_path}")
            df = pd.read_csv(metadata_path)
            
            if text_column not in df.columns:
                # 嘗試找到文本列
                if 'tokens_str' in df.columns:
                    text_column = 'tokens_str'
                elif 'clean_text' in df.columns:
                    text_column = 'clean_text'
                else:
                    text_column = df.columns[0]
                self.log(f"Specified text column not found, using '{text_column}' instead")
            
            texts = df[text_column].fillna('').tolist()
            self.log(f"Loaded {len(texts)} text entries")
            
            # 創建向量化器
            if callback:
                callback("Creating document-term matrix...", 30)
            
            # 使用TF-IDF進行特徵抽取 - 針對大型資料集優化
            vectorizer = TfidfVectorizer(
                max_df=0.8,      # 降低閾值，過濾出現在超過80%文檔中的詞
                min_df=5,        # 增加最小文檔頻率，忽略極少出現的詞
                max_features=10000,  # 增加特徵數量以捕捉更多詞彙
                ngram_range=(1, 2),  # 使用單詞和雙詞組合作為特徵
                stop_words='english'  # 使用英文停用詞
            )
            
            self.log("Vectorizing documents...")
            X = vectorizer.fit_transform(texts)
            
            self.log(f"Created document-term matrix with shape: {X.shape}")
            feature_names = vectorizer.get_feature_names_out()
            
            # 訓練LDA模型
            if callback:
                callback(f"Training LDA model with {n_topics} topics...", 50)
            
            # 設定 LDA 模型參數
            lda_params = {
                'n_components': n_topics,
                'random_state': 42,
                'learning_method': 'online',
                'n_jobs': 1  # 使用所有可用 CPU 核心
            }
            
            # 添加自訂參數（如果提供）
            if doc_topic_prior is not None:
                lda_params['doc_topic_prior'] = doc_topic_prior
            if topic_word_prior is not None:
                lda_params['topic_word_prior'] = topic_word_prior
            if max_iter is not None:
                lda_params['max_iter'] = max_iter
            else:
                # 根據資料集大小自動調整迭代次數
                if len(texts) < 1000:
                    lda_params['max_iter'] = 20
                elif len(texts) < 10000:
                    lda_params['max_iter'] = 30
                else:
                    lda_params['max_iter'] = 50
                    lda_params['evaluate_every'] = 5
                    lda_params['verbose'] = 1
            
            # 記錄最終模型參數
            self.log(f"LDA model parameters: {lda_params}")
            
            # 初始化 LDA 模型
            lda_model = LatentDirichletAllocation(**lda_params)
            
            self.log("Training LDA model...")
            lda_model.fit(X)
            self.log("LDA model training complete")
            
            # 獲取主題-詞語分布
            if callback:
                callback("Extracting topic-word distributions...", 70)
            
            topic_word_distributions = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
            
            # 獲取每個主題的頂部詞語
            n_top_words = 15
            topic_top_words = {}
            
            # 使用自定義標籤（如果提供）
            use_custom_labels = topic_labels is not None and len(topic_labels) >= n_topics
            
            for topic_idx, topic in enumerate(topic_word_distributions):
                topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
                
                # 獲取主題標籤：如果提供了自定義標籤則使用，否則使用默認的Topic_X格式
                if use_custom_labels and topic_idx in topic_labels:
                    topic_label = topic_labels[topic_idx]
                    topic_key = f"Topic_{topic_idx+1}_{topic_label}"
                else:
                    topic_key = f"Topic_{topic_idx+1}"
                    
                topic_top_words[topic_key] = topic_words
                self.log(f"{topic_key}: {', '.join(topic_words[:10])}")
            
            # 獲取文檔-主題分布
            if callback:
                callback("Extracting document-topic distributions...", 80)
            
            doc_topic_distributions = lda_model.transform(X)
            
            # 獲取每個文檔的主要主題
            doc_main_topics = np.argmax(doc_topic_distributions, axis=1)
            
            # 創建主題標籤映射（用於文檔標籤）
            if use_custom_labels:
                topic_id_to_label = {idx: f"Topic_{idx+1}_{topic_labels[idx]}" for idx in range(n_topics) if idx in topic_labels}
            else:
                topic_id_to_label = {idx: f"Topic_{idx+1}" for idx in range(n_topics)}
                
            # 為每個文檔分配主題標籤
            df['main_topic'] = [topic_id_to_label.get(topic_idx, f"Topic_{topic_idx+1}") for topic_idx in doc_main_topics]
            
            # 新增：計算並記錄每個主題的文檔數量分布
            topic_counts = df['main_topic'].value_counts().to_dict()
            self.log("Topic distribution statistics:")
            for topic, count in topic_counts.items():
                percentage = (count / len(df)) * 100
                self.log(f"  {topic}: {count} documents ({percentage:.2f}%)")
            
            # 保存模型和結果
            if callback:
                callback("Saving model and results...", 90)
            
            base_name = os.path.basename(metadata_path)
            base_name_without_ext = os.path.splitext(base_name)[0].replace('_bert_metadata', '')
            
            # 保存LDA模型
            lda_model_path = os.path.join(self.models_dir, f"{base_name_without_ext}_lda_{n_topics}_topics.pkl")
            with open(lda_model_path, 'wb') as f:
                pickle.dump(lda_model, f)
            
            # 保存向量化器
            vectorizer_path = os.path.join(self.models_dir, f"{base_name_without_ext}_vectorizer.pkl")
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            # 保存主題詞分布
            topics_path = os.path.join(self.output_dir, f"{base_name_without_ext}_topics.json")
            with open(topics_path, 'w', encoding='utf-8') as f:
                json.dump(topic_top_words, f, ensure_ascii=False, indent=2)
            
            # 保存含主題標籤的元數據
            topic_metadata_path = os.path.join(self.output_dir, f"{base_name_without_ext}_with_topics.csv")
            df.to_csv(topic_metadata_path, index=False)
            
            # 保存主題分布統計數據
            stats_path = os.path.join(self.output_dir, f"{base_name_without_ext}_topic_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'topic_counts': topic_counts,
                    'params': lda_params,
                    'n_docs': len(df),
                    'n_topics': n_topics
                }, f, ensure_ascii=False, indent=2)
            
            # 生成可視化
            if callback:
                callback("Generating visualizations...", 95)
            
            # 1. 主題詞雲可視化
            vis_path1 = self._plot_top_words(lda_model, feature_names, n_top_words, base_name_without_ext, topic_labels)
            
            # 2. 文檔-主題分布可視化
            vis_path2 = self._plot_doc_topics(doc_topic_distributions, n_topics, base_name_without_ext, topic_labels)
            
            # 3. 生成主題相似度矩陣可視化
            vis_path3 = self._plot_topic_similarity(lda_model, base_name_without_ext, topic_labels)
            
            if callback:
                callback("LDA topic modeling complete", 100)
            
            # 返回結果
            return {
                'lda_model_path': lda_model_path,
                'vectorizer_path': vectorizer_path,
                'topics_path': topics_path,
                'topic_metadata_path': topic_metadata_path,
                'topic_stats_path': stats_path,
                'n_topics': n_topics,
                'topic_words': topic_top_words,
                'topic_counts': topic_counts,
                'visualizations': [vis_path1, vis_path2, vis_path3]
            }
            
        except Exception as e:
            self.log(f"Error in LDA topic modeling: {str(e)}", level=logging.ERROR)
            self.log(traceback.format_exc(), level=logging.ERROR)
            if callback:
                callback(f"Error: {str(e)}", -1)
            raise
    
    def _plot_topic_similarity(self, lda_model, base_name, topic_labels=None):
        """生成主題相似度矩陣可視化 - 支持自定義主題標籤"""
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
            
            # 創建可視化
            plt.figure(figsize=(10, 8))
            plt.imshow(similarity_matrix, cmap='viridis')
            plt.colorbar(label='餘弦相似度')
            plt.title('主題間相似度矩陣')
            plt.xlabel('主題')
            plt.ylabel('主題')
            
            # 添加主題標籤 - 使用自定義標籤（如果提供）
            if topic_labels is not None:
                # 格式化標籤
                formatted_labels = {}
                for i in range(n_topics):
                    if i in topic_labels:
                        formatted_labels[i] = f'主題 {i+1}\n{topic_labels.get(i, "")}'
                    else:
                        formatted_labels[i] = f'主題 {i+1}'
                topic_label_texts = [formatted_labels[i] for i in range(n_topics)]
            else:
                topic_label_texts = [f'主題 {i+1}' for i in range(n_topics)]
                
            plt.xticks(range(n_topics), topic_label_texts, rotation=45, ha='right')
            plt.yticks(range(n_topics), topic_label_texts)
            
            # 在每個單元格中添加數值
            for i in range(n_topics):
                for j in range(n_topics):
                    plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                            ha='center', va='center', 
                            color='white' if similarity_matrix[i, j] > 0.5 else 'black')
            
            plt.tight_layout()
            
            # 保存圖形
            vis_path = os.path.join(self.vis_dir, f"{base_name}_topic_similarity.png")
            plt.savefig(vis_path, dpi=200, bbox_inches='tight')
            plt.close('all')  # 確保關閉所有圖表
            
            return vis_path
        except Exception as e:
            self.log(f"Warning: Failed to generate topic similarity visualization: {str(e)}", level=logging.WARNING)
            return None

    def _plot_top_words(self, lda_model, feature_names, n_top_words, base_name, topic_labels=None):
        """生成主題-詞語分布可視化（支持自定義主題標籤）"""
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), sharex=True)
        axes = axes.flatten()
        
        for topic_idx, topic in enumerate(lda_model.components_):
            if topic_idx >= 10:  # 限制為前10個主題
                break
                    
            top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]
            
            ax = axes[topic_idx]
            ax.barh(top_features, weights)
            
            # 使用自定義標籤（如果提供）
            if topic_labels is not None and topic_idx in topic_labels:
                # 格式化主題標題，處理中文顯示
                title = f'主題 {topic_idx+1}: {topic_labels[topic_idx]}'
            else:
                title = f'主題 {topic_idx+1}'
                
            # 應用中文標題
            apply_chinese_to_plot(ax, title=title)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_yticklabels(top_features, fontdict={'fontsize': 8})
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('各主題關鍵詞分佈', fontsize=16)
        
        # 保存圖形
        vis_path = os.path.join(self.vis_dir, f"{base_name}_topic_words.png")
        plt.savefig(vis_path, dpi=200, bbox_inches='tight')
        plt.close('all')  # 確保關閉所有圖表
        
        return vis_path
    
    def _plot_doc_topics(self, doc_topic_dist, n_topics, base_name, topic_labels=None):
        """生成文檔-主題分布可視化（支持自定義主題標籤）"""
        # 確保使用非互動模式
        plt.ioff()
        
        # 計算每個主題的文檔數量
        topic_counts = np.zeros(n_topics)
        doc_main_topics = np.argmax(doc_topic_dist, axis=1)
        
        for topic_idx in range(n_topics):
            topic_counts[topic_idx] = np.sum(doc_main_topics == topic_idx)
        
        # 繪製主題分布柱狀圖
        plt.figure(figsize=(12, 7))
        x = np.arange(n_topics)
        plt.bar(x, topic_counts)
        
        # 使用自定義標籤（如果提供）
        if topic_labels is not None:
            # 格式化標籤
            formatted_labels = {}
            for i in range(n_topics):
                if i in topic_labels:
                    formatted_labels[i] = f'主題 {i+1}\n{topic_labels.get(i, "")}'
                else:
                    formatted_labels[i] = f'主題 {i+1}'
            x_labels = [formatted_labels[i] for i in range(n_topics)]
        else:
            x_labels = [f'主題 {i+1}' for i in range(n_topics)]
        
        plt.xlabel('主題')
        plt.ylabel('文檔數量')
        plt.title('文檔在各主題中的分佈')
        plt.xticks(x, x_labels, rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存圖形
        vis_path = os.path.join(self.vis_dir, f"{base_name}_doc_topics.png")
        plt.savefig(vis_path, dpi=200, bbox_inches='tight')
        plt.close('all')  # 確保關閉所有圖表，避免內存洩露
        
        return vis_path

    def analyze_topic_coherence(self, metadata_path, n_topics_range=range(5, 16), callback=None):
        """
        分析不同主題數量下的LDA模型一致性
        
        Args:
            metadata_path: BERT元數據文件路徑（CSV）
            n_topics_range: 要測試的主題數量範圍
            callback: 進度回調函數
            
        Returns:
            dict: 包含不同主題數量的得分
        """
        try:
            self.log(f"Starting LDA topic coherence analysis")
            
            # 讀取數據
            if callback:
                callback("Loading metadata for coherence analysis...", 10)
            
            df = pd.read_csv(metadata_path)
            text_column = 'tokens_str' if 'tokens_str' in df.columns else 'clean_text'
            texts = df[text_column].fillna('').tolist()
            
            # 創建TF-IDF 向量化器
            vectorizer = TfidfVectorizer(
                max_df=0.8,  # 降低閾值，過濾出現在超過 80% 文檔中的詞
                min_df=5,    # 增加最小文檔頻率，忽略極少出現的詞
                max_features=5000,  # 保留特徵數量
                ngram_range=(1, 2),  # 使用單詞和雙詞組合作為特徵
                stop_words='english'  # 使用英文停用詞
            )
            X = vectorizer.fit_transform(texts)
            
            # 對不同的主題數量進行測試
            coherence_scores = {}
            perplexity_scores = {}
            
            total_iterations = len(n_topics_range)
            for i, n_topics in enumerate(n_topics_range):
                progress = 10 + (i / total_iterations * 80)
                if callback:
                    callback(f"Testing model with {n_topics} topics...", progress)
                
                lda_model = LatentDirichletAllocation(
                    n_components=n_topics, 
                    random_state=42,
                    learning_method='online',
                    max_iter=10,
                    verbose=0
                )
                
                lda_model.fit(X)
                
                # 計算困惑度（perplexity）
                perplexity = lda_model.perplexity(X)
                perplexity_scores[n_topics] = perplexity
                
                # 此處可以添加其他一致性指標的計算，如UMass or UCI coherence
                # 暫時使用困惑度作為評估指標
                coherence_scores[n_topics] = -perplexity  # 負困惑度，越高越好
                
                self.log(f"Topics: {n_topics}, Perplexity: {perplexity}")
            
            # 生成一致性曲線
            if callback:
                callback("Generating coherence plot...", 90)
            
            plt.figure(figsize=(10, 6))
            plt.plot(list(perplexity_scores.keys()), list(perplexity_scores.values()), marker='o')
            plt.xlabel('Number of Topics')
            plt.ylabel('Perplexity Score (lower is better)')
            plt.title('LDA Model Perplexity by Number of Topics')
            plt.grid(True)
            
            # 保存圖形
            base_name = os.path.basename(metadata_path)
            base_name_without_ext = os.path.splitext(base_name)[0].replace('_bert_metadata', '')
            vis_path = os.path.join(self.vis_dir, f"{base_name_without_ext}_coherence.png")
            plt.savefig(vis_path, dpi=200, bbox_inches='tight')
            plt.close('all')  # 確保關閉所有圖表
            
            if callback:
                callback("Coherence analysis complete", 100)
            
            # 返回結果
            return {
                'coherence_scores': coherence_scores,
                'perplexity_scores': perplexity_scores,
                'visualization_path': vis_path,
                'optimal_topics': min(perplexity_scores, key=perplexity_scores.get)  # 最低困惑度的主題數
            }
            
        except Exception as e:
            self.log(f"Error in LDA coherence analysis: {str(e)}", level=logging.ERROR)
            self.log(traceback.format_exc(), level=logging.ERROR)
            if callback:
                callback(f"Error: {str(e)}", -1)
            raise

    # 對於不同規模的資料集自動調整參數的函數
    def get_lda_params(data_size, n_topics):
        """
        根據資料集大小自動調整 LDA 參數
        
        Args:
            data_size: 資料集大小
            n_topics: 主題數量
            
        Returns:
            dict: LDA 參數字典
        """
        if data_size < 100:
            # 小型資料集
            return {
                'n_components': n_topics,
                'doc_topic_prior': 0.5,  # 適中的 alpha
                'topic_word_prior': 0.1,  # 適中的 beta
                'max_iter': 20,
                'n_jobs': -1,
                'random_state': 42,
                'learning_method': 'online',
            }
        elif data_size < 10000:
            # 中型資料集
            return {
                'n_components': n_topics,
                'doc_topic_prior': 0.7,  # 較大的 alpha
                'topic_word_prior': 0.05,  # 較小的 beta
                'max_iter': 30,
                'n_jobs': -1,
                'random_state': 42,
                'learning_method': 'online',
            }
        else:
            # 大型資料集
            return {
                'n_components': n_topics,
                'doc_topic_prior': 0.9,  # 更大的 alpha
                'topic_word_prior': 0.01,  # 更小的 beta
                'max_iter': 50,
                'n_jobs': -1,
                'random_state': 42,
                'learning_method': 'online',
                'learning_decay': 0.7,
                'evaluate_every': 5,
                'verbose': 1
            }
    
    def evaluate_topic_models(self, metadata_path, min_topics=5, max_topics=15, step=1, 
                          alpha_values=[0.1, 0.5, 0.9], beta_values=[0.01, 0.05, 0.1],
                          callback=None):
        """
        評估不同主題數量和參數組合的 LDA 模型
        
        Args:
            metadata_path: BERT元數據文件路徑（CSV）
            min_topics: 最小主題數量
            max_topics: 最大主題數量
            step: 主題數量遞增步長
            alpha_values: 要測試的 alpha 值列表
            beta_values: 要測試的 beta 值列表
            callback: 進度回調函數
            
        Returns:
            dict: 不同參數組合的評估結果
        """
        try:
            self.log(f"開始進行 LDA 參數網格搜索")
            
            # 讀取數據
            if callback:
                callback("Loading metadata for grid search...", 5)
            
            df = pd.read_csv(metadata_path)
            text_column = 'tokens_str' if 'tokens_str' in df.columns else 'clean_text'
            texts = df[text_column].fillna('').tolist()
            
            # 創建向量化器並轉換文本
            vectorizer = TfidfVectorizer(
                max_df=0.8, 
                min_df=5, 
                max_features=5000, 
                ngram_range=(1, 2),
                stop_words='english'
            )
            X = vectorizer.fit_transform(texts)
            
            # 儲存評估結果
            results = {}
            total_combinations = len(range(min_topics, max_topics+1, step)) * len(alpha_values) * len(beta_values)
            current_step = 0
            
            # 記錄開始時間
            start_time = time.time()
            
            # 開始網格搜索
            for n_topics in range(min_topics, max_topics+1, step):
                results[n_topics] = {}
                
                for alpha in alpha_values:
                    for beta in beta_values:
                        # 更新進度
                        current_step += 1
                        progress = 5 + int((current_step / total_combinations) * 90)
                        
                        if callback:
                            callback(f"Testing model with {n_topics} topics, alpha={alpha}, beta={beta}", progress)
                            
                        # 初始化和訓練 LDA 模型
                        lda_model = LatentDirichletAllocation(
                            n_components=n_topics,
                            doc_topic_prior=alpha,
                            topic_word_prior=beta,
                            random_state=42,
                            learning_method='online',
                            max_iter=20,
                            n_jobs=-1
                        )
                        
                        # 訓練模型
                        lda_model.fit(X)
                        
                        # 計算困惑度（perplexity）
                        perplexity = lda_model.perplexity(X)
                        
                        # 計算主題一致性（模擬 u_mass 一致性計算法）
                        topic_words = []
                        feature_names = vectorizer.get_feature_names_out()
                        for topic_idx, topic in enumerate(lda_model.components_):
                            top_words_idx = topic.argsort()[:-11:-1]
                            top_words = [feature_names[i] for i in top_words_idx]
                            topic_words.append(top_words)
                        
                        # 計算不同主題間的距離（越大越好）
                        topic_distances = 0
                        n_pairs = 0
                        for i in range(len(topic_words)):
                            for j in range(i+1, len(topic_words)):
                                overlap = len(set(topic_words[i]) & set(topic_words[j]))
                                topic_distances += (10 - overlap) / 10.0  # 歸一化距離
                                n_pairs += 1
                        
                        avg_topic_distance = topic_distances / n_pairs if n_pairs > 0 else 0
                        
                        # 計算文檔-主題分布的熵（越大越好，表示多樣性）
                        doc_topic_dist = lda_model.transform(X)
                        entropy_sum = 0
                        for dist in doc_topic_dist:
                            # 小數值修正，避免 log(0)
                            dist = np.clip(dist, 1e-10, 1.0)
                            # 歸一化
                            dist = dist / dist.sum()
                            # 計算熵
                            entropy = -np.sum(dist * np.log(dist))
                            entropy_sum += entropy
                        
                        avg_entropy = entropy_sum / len(doc_topic_dist)
                        
                        # 綜合評分（越高越好）
                        # 主題距離越大、熵越大、困惑度越小，表現越好
                        combined_score = avg_topic_distance + avg_entropy - (perplexity / 10000)
                        
                        # 保存結果
                        param_key = f"alpha={alpha:.2f},beta={beta:.2f}"
                        results[n_topics][param_key] = {
                            'perplexity': perplexity,
                            'avg_topic_distance': avg_topic_distance,
                            'avg_entropy': avg_entropy,
                            'combined_score': combined_score
                        }
                        
                        self.log(f"Topics: {n_topics}, Alpha: {alpha}, Beta: {beta}, " +
                                f"Perplexity: {perplexity:.2f}, " +
                                f"Avg Topic Distance: {avg_topic_distance:.2f}, " +
                                f"Avg Entropy: {avg_entropy:.2f}, " +
                                f"Combined Score: {combined_score:.2f}")
            
            # 找到最佳參數組合
            best_score = -float('inf')
            best_params = None
            
            for n_topics in results:
                for param_key, metrics in results[n_topics].items():
                    if metrics['combined_score'] > best_score:
                        best_score = metrics['combined_score']
                        alpha, beta = [float(val.split('=')[1]) for val in param_key.split(',')]
                        best_params = {
                            'n_topics': n_topics,
                            'alpha': alpha,
                            'beta': beta,
                            'metrics': metrics
                        }
            
            # 生成評估報告
            if callback:
                callback("Generating evaluation report...", 95)
                
            # 創建報告目錄
            report_dir = os.path.join(self.vis_dir, "param_search")
            if not os.path.exists(report_dir):
                os.makedirs(report_dir)
                
            # 生成可視化圖表
            plt.figure(figsize=(12, 8))
            
            # 繪製不同主題數量的綜合得分
            plt.subplot(2, 2, 1)
            for alpha in alpha_values:
                for beta in beta_values:
                    param_key = f"alpha={alpha:.2f},beta={beta:.2f}"
                    scores = [results[n_topics][param_key]['combined_score'] 
                            if param_key in results[n_topics] else float('nan') 
                            for n_topics in results]
                    plt.plot(list(results.keys()), scores, marker='o', label=param_key)
            
            plt.xlabel('主題數量')
            plt.ylabel('綜合評分 (越高越好)')
            plt.title('不同參數組合的綜合評分')
            plt.legend(loc='best')
            plt.grid(True)
            
            # 繪製不同主題數量的困惑度
            plt.subplot(2, 2, 2)
            for alpha in alpha_values:
                for beta in beta_values:
                    param_key = f"alpha={alpha:.2f},beta={beta:.2f}"
                    perplexities = [results[n_topics][param_key]['perplexity'] 
                                if param_key in results[n_topics] else float('nan') 
                                for n_topics in results]
                    plt.plot(list(results.keys()), perplexities, marker='o', label=param_key)
            
            plt.xlabel('主題數量')
            plt.ylabel('困惑度 (越低越好)')
            plt.title('不同參數組合的困惑度')
            plt.legend(loc='best')
            plt.grid(True)
            
            # 繪製不同主題數量的熵
            plt.subplot(2, 2, 3)
            for alpha in alpha_values:
                for beta in beta_values:
                    param_key = f"alpha={alpha:.2f},beta={beta:.2f}"
                    entropies = [results[n_topics][param_key]['avg_entropy'] 
                            if param_key in results[n_topics] else float('nan') 
                            for n_topics in results]
                    plt.plot(list(results.keys()), entropies, marker='o', label=param_key)
            
            plt.xlabel('主題數量')
            plt.ylabel('平均熵 (越高越好)')
            plt.title('不同參數組合的平均熵')
            plt.legend(loc='best')
            plt.grid(True)
            
            # 繪製不同主題數量的主題距離
            plt.subplot(2, 2, 4)
            for alpha in alpha_values:
                for beta in beta_values:
                    param_key = f"alpha={alpha:.2f},beta={beta:.2f}"
                    distances = [results[n_topics][param_key]['avg_topic_distance'] 
                                if param_key in results[n_topics] else float('nan') 
                                for n_topics in results]
                    plt.plot(list(results.keys()), distances, marker='o', label=param_key)
            
            plt.xlabel('主題數量')
            plt.ylabel('平均主題距離 (越高越好)')
            plt.title('不同參數組合的平均主題距離')
            plt.legend(loc='best')
            plt.grid(True)
            
            plt.tight_layout()
            
            # 保存圖表
            base_name = os.path.basename(metadata_path)
            base_name_without_ext = os.path.splitext(base_name)[0].replace('_bert_metadata', '')
            vis_path = os.path.join(report_dir, f"{base_name_without_ext}_param_search.png")
            plt.savefig(vis_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            # 計算總執行時間
            total_time = time.time() - start_time
            minutes, seconds = divmod(total_time, 60)
            
            if callback:
                callback("Evaluation complete", 100)
            
            # 返回結果
            self.log(f"Parameter grid search completed in {int(minutes)}m {int(seconds)}s")
            self.log(f"Best parameters: Topics={best_params['n_topics']}, " +
                    f"Alpha={best_params['alpha']}, Beta={best_params['beta']}")
            self.log(f"Best score: {best_params['metrics']['combined_score']:.2f}")
            
            return {
                'results': results,
                'best_params': best_params,
                'visualization_path': vis_path,
                'execution_time': f"{int(minutes)}m {int(seconds)}s"
            }
            
        except Exception as e:
            self.log(f"Error in parameter evaluation: {str(e)}", level=logging.ERROR)
            self.log(traceback.format_exc(), level=logging.ERROR)
            if callback:
                callback(f"Error: {str(e)}", -1)
            raise

# 使用示例
if __name__ == "__main__":
    extractor = LDATopicExtractor()
    
    def print_progress(message, percentage):
        print(f"{message} ({percentage}%)")
    
    try:
        # 假設我們已經有了BERT處理後的元數據
        metadata_path = "./Part02_/results/processed_reviews_bert_metadata.csv"
        
        # 分析不同主題數量的一致性
        coherence_results = extractor.analyze_topic_coherence(
            metadata_path, 
            n_topics_range=range(5, 16),
            callback=print_progress
        )
        print(f"Optimal number of topics: {coherence_results['optimal_topics']}")
        
        # 執行最佳數量的主題建模
        optimal_topics = coherence_results['optimal_topics']
        results = extractor.run_lda(
            metadata_path,
            n_topics=optimal_topics,
            callback=print_progress
        )
        
        print(f"LDA model saved to: {results['lda_model_path']}")
        print(f"Topics saved to: {results['topics_path']}")
        
        # 打印每個主題的頂部詞語
        for topic, words in results['topic_words'].items():
            print(f"{topic}: {', '.join(words[:10])}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()