import numpy as np
import pandas as pd
import os
import logging
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import json
import re

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
    
    def run_lda(self, metadata_path, text_column='tokens_str', n_topics=10, callback=None):
        """
        執行LDA主題建模來識別面向
        
        Args:
            metadata_path: BERT元數據文件路徑（CSV）
            text_column: 文本列名
            n_topics: 主題數量
            callback: 進度回調函數
            
        Returns:
            dict: 包含LDA模型和相關結果的字典
        """
        try:
            self.log(f"Starting LDA topic modeling with {n_topics} topics")
            self.log(f"Using metadata from: {metadata_path}")
            
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
                self.log(f"Specified text column '{text_column}' not found, using '{text_column}' instead")
            
            texts = df[text_column].fillna('').tolist()
            self.log(f"Loaded {len(texts)} text entries")
            
            # 創建向量化器
            if callback:
                callback("Creating document-term matrix...", 30)
            
            # 使用TF-IDF進行特徵抽取
            vectorizer = TfidfVectorizer(
                max_df=0.95,  # 忽略在95%以上文檔中出現的詞
                min_df=2,     # 忽略在少於2個文檔中出現的詞
                max_features=5000,  # 最多保留5000個特徵
                stop_words='english'  # 使用英文停用詞
            )
            
            self.log("Vectorizing documents...")
            X = vectorizer.fit_transform(texts)
            
            self.log(f"Created document-term matrix with shape: {X.shape}")
            feature_names = vectorizer.get_feature_names_out()
            
            # 訓練LDA模型
            if callback:
                callback(f"Training LDA model with {n_topics} topics...", 50)
            
            lda_model = LatentDirichletAllocation(
                n_components=n_topics, 
                random_state=42,
                learning_method='online',
                max_iter=10,
                verbose=0
            )
            
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
            for topic_idx, topic in enumerate(topic_word_distributions):
                topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
                topic_top_words[f"Topic_{topic_idx+1}"] = topic_words
                self.log(f"Topic {topic_idx+1}: {', '.join(topic_words[:10])}")
            
            # 獲取文檔-主題分布
            if callback:
                callback("Extracting document-topic distributions...", 80)
            
            doc_topic_distributions = lda_model.transform(X)
            
            # 獲取每個文檔的主要主題
            doc_main_topics = np.argmax(doc_topic_distributions, axis=1)
            df['main_topic'] = [f"Topic_{i+1}" for i in doc_main_topics]
            
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
            
            # 生成可視化
            if callback:
                callback("Generating visualizations...", 95)
            
            # 1. 主題詞雲可視化
            self._plot_top_words(lda_model, feature_names, n_top_words, base_name_without_ext)
            
            # 2. 文檔-主題分布可視化
            self._plot_doc_topics(doc_topic_distributions, n_topics, base_name_without_ext)
            
            if callback:
                callback("LDA topic modeling complete", 100)
            
            # 返回結果
            return {
                'lda_model_path': lda_model_path,
                'vectorizer_path': vectorizer_path,
                'topics_path': topics_path,
                'topic_metadata_path': topic_metadata_path,
                'n_topics': n_topics,
                'topic_words': topic_top_words
            }
            
        except Exception as e:
            self.log(f"Error in LDA topic modeling: {str(e)}", level=logging.ERROR)
            self.log(traceback.format_exc(), level=logging.ERROR)
            if callback:
                callback(f"Error: {str(e)}", -1)
            raise
    
    def _plot_top_words(self, lda_model, feature_names, n_top_words, base_name):
        """生成主題-詞語分布可視化"""
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
            ax.set_title(f'Topic {topic_idx+1}')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_yticklabels(top_features, fontdict={'fontsize': 8})
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle('Top words for each topic', fontsize=16)
        
        # 保存圖形
        vis_path = os.path.join(self.vis_dir, f"{base_name}_topic_words.png")
        plt.savefig(vis_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_doc_topics(self, doc_topic_dist, n_topics, base_name):
        """生成文檔-主題分布可視化"""
        # 計算每個主題的文檔數量
        topic_counts = np.zeros(n_topics)
        doc_main_topics = np.argmax(doc_topic_dist, axis=1)
        
        for topic_idx in range(n_topics):
            topic_counts[topic_idx] = np.sum(doc_main_topics == topic_idx)
        
        # 繪製主題分布柱狀圖
        plt.figure(figsize=(10, 6))
        x = np.arange(n_topics)
        plt.bar(x, topic_counts)
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Documents')
        plt.title('Document Distribution Across Topics')
        plt.xticks(x, [f'Topic {i+1}' for i in range(n_topics)])
        plt.xticks(rotation=45)
        
        # 保存圖形
        vis_path = os.path.join(self.vis_dir, f"{base_name}_doc_topics.png")
        plt.savefig(vis_path, dpi=200, bbox_inches='tight')
        plt.close()

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
            
            # 創建向量化器
            vectorizer = TfidfVectorizer(
                max_df=0.95, min_df=2, max_features=5000, stop_words='english'
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
            plt.close()
            
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