#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分類方法模組 - 支援多種文本分類和主題建模方法
包含：傳統情感分析、LDA、BERTopic、NMF、聚類分析
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod
import os
import json
from datetime import datetime

# 傳統機器學習分類器
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB

# 主題建模和聚類
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 可選的進階庫
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

try:
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class BaseClassificationMethod(ABC):
    """分類方法基類"""
    
    def __init__(self, output_dir: Optional[str] = None, progress_callback=None):
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.results = {}
        
    @abstractmethod
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
            texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """訓練/擬合模型"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray, texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """預測"""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """返回方法名稱"""
        pass
    
    def _notify_progress(self, phase: str, current: int = None, total: int = None, status: str = None):
        """通知進度更新"""
        if self.progress_callback:
            if phase == 'phase':
                self.progress_callback('phase', {
                    'phase_name': status or phase,
                    'current_phase': current or 1,
                    'total_phases': total or 3
                })
            elif phase == 'progress' and current is not None and total is not None:
                self.progress_callback('progress', (current, total))
            elif phase == 'status':
                self.progress_callback('status', status)
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """保存結果到文件"""
        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"結果已保存到：{filepath}")

class SentimentClassificationMethod(BaseClassificationMethod):
    """傳統情感分類方法（原有系統）"""
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化分類模型"""
        model_map = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm_linear': SVC(kernel='linear', probability=True, random_state=42),
            'naive_bayes': GaussianNB()
        }
        
        if XGBOOST_AVAILABLE and self.model_type == 'xgboost':
            model_map['xgboost'] = xgb.XGBClassifier(random_state=42)
        
        if self.model_type not in model_map:
            available_models = ', '.join(model_map.keys())
            raise ValueError(f"不支援的模型類型：{self.model_type}。可用模型：{available_models}")
        
        self.model = model_map[self.model_type]
        logger.info(f"已初始化{self.model_type}模型")
    
    def get_method_name(self) -> str:
        return f"情感分析 ({self.model_type})"
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
            texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """訓練情感分類模型"""
        if labels is None:
            raise ValueError("情感分類需要提供標籤")
        
        self._notify_progress('phase', 1, 3, f'{self.get_method_name()} - 資料預處理')
        
        # 標準化特徵
        features_scaled = self.scaler.fit_transform(features)
        
        # 編碼標籤
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # 分割訓練和測試集
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
        )
        
        self._notify_progress('phase', 2, 3, f'{self.get_method_name()} - 模型訓練')
        
        # 訓練模型
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        self._notify_progress('phase', 3, 3, f'{self.get_method_name()} - 評估')
        
        # 預測
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        test_pred_proba = self.model.predict_proba(X_test)
        
        # 計算指標
        results = {
            'method': self.get_method_name(),
            'model_type': self.model_type,
            'train_accuracy': accuracy_score(y_train, train_pred),
            'test_accuracy': accuracy_score(y_test, test_pred),
            'classification_report': classification_report(y_test, test_pred, 
                                                         target_names=self.label_encoder.classes_, 
                                                         output_dict=True),
            'feature_dim': features.shape[1],
            'n_samples': features.shape[0],
            'n_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 計算F1分數
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_pred, average='weighted')
        results['precision'] = precision
        results['recall'] = recall
        results['f1_score'] = f1
        
        self.results = results
        self.save_results(results, f"sentiment_classification_{self.model_type}_results.json")
        
        logger.info(f"{self.get_method_name()}訓練完成 - 準確率: {results['test_accuracy']:.4f}")
        return results
    
    def predict(self, features: np.ndarray, texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """預測情感"""
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用fit方法")
        
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # 解碼標籤
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        results = {
            'method': self.get_method_name(),
            'predictions': predicted_labels.tolist(),
            'probabilities': probabilities.tolist(),
            'prediction_summary': {
                label: int(np.sum(predicted_labels == label)) 
                for label in self.label_encoder.classes_
            }
        }
        
        return results

class LDATopicMethod(BaseClassificationMethod):
    """LDA主題建模方法"""
    
    def __init__(self, n_topics: int = 5, use_gensim: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.n_topics = n_topics
        self.use_gensim = use_gensim and GENSIM_AVAILABLE
        self.model = None
        self.vectorizer = None
        self.dictionary = None
        self.corpus = None
        self.is_fitted = False
        
        if self.use_gensim and not GENSIM_AVAILABLE:
            logger.warning("Gensim未安裝，將使用sklearn的LDA")
            self.use_gensim = False
    
    def get_method_name(self) -> str:
        engine = "Gensim" if self.use_gensim else "Sklearn"
        return f"LDA主題建模 ({engine}, {self.n_topics}主題)"
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
            texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """訓練LDA模型"""
        if texts is None:
            raise ValueError("LDA需要原始文本進行訓練")
        
        self._notify_progress('phase', 1, 4, f'{self.get_method_name()} - 文本預處理')
        
        if self.use_gensim:
            # 使用Gensim LDA
            results = self._fit_gensim_lda(texts)
        else:
            # 使用Sklearn LDA
            results = self._fit_sklearn_lda(texts)
        
        self.is_fitted = True
        self.results = results
        self.save_results(results, f"lda_topic_modeling_{self.n_topics}topics_results.json")
        
        logger.info(f"{self.get_method_name()}訓練完成")
        return results
    
    def _fit_gensim_lda(self, texts: pd.Series) -> Dict[str, Any]:
        """使用Gensim的LDA"""
        from gensim.utils import simple_preprocess
        from gensim.parsing.preprocessing import STOPWORDS
        
        # 預處理文本
        processed_texts = []
        for text in texts:
            tokens = simple_preprocess(str(text), deacc=True)
            tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 3]
            processed_texts.append(tokens)
        
        # 創建字典和語料庫
        self.dictionary = corpora.Dictionary(processed_texts)
        self.dictionary.filter_extremes(no_below=2, no_above=0.8)
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        self._notify_progress('phase', 2, 4, f'{self.get_method_name()} - 模型訓練')
        
        # 訓練LDA模型
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        self._notify_progress('phase', 3, 4, f'{self.get_method_name()} - 提取主題')
        
        # 提取主題信息
        topics = []
        for topic_id in range(self.n_topics):
            topic_words = self.model.show_topic(topic_id, topn=10)
            topics.append({
                'topic_id': topic_id,
                'words': [{'word': word, 'probability': prob} for word, prob in topic_words]
            })
        
        # 計算主題一致性
        from gensim.models import CoherenceModel
        coherence_model = CoherenceModel(model=self.model, texts=processed_texts, 
                                       dictionary=self.dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        
        results = {
            'method': self.get_method_name(),
            'n_topics': self.n_topics,
            'topics': topics,
            'coherence_score': coherence_score,
            'vocabulary_size': len(self.dictionary),
            'corpus_size': len(self.corpus),
            'engine': 'gensim',
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _fit_sklearn_lda(self, texts: pd.Series) -> Dict[str, Any]:
        """使用Sklearn的LDA"""
        # 創建詞頻向量器
        self.vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        # 向量化文本
        text_vectors = self.vectorizer.fit_transform(texts.astype(str))
        
        self._notify_progress('phase', 2, 4, f'{self.get_method_name()} - 模型訓練')
        
        # 訓練LDA模型
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            learning_method='batch',
            max_iter=25
        )
        
        topic_distributions = self.model.fit_transform(text_vectors)
        
        self._notify_progress('phase', 3, 4, f'{self.get_method_name()} - 提取主題')
        
        # 提取主題詞
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [
                {'word': feature_names[i], 'probability': topic[i]}
                for i in top_words_idx
            ]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words
            })
        
        results = {
            'method': self.get_method_name(),
            'n_topics': self.n_topics,
            'topics': topics,
            'perplexity': self.model.perplexity(text_vectors),
            'log_likelihood': self.model.score(text_vectors),
            'vocabulary_size': len(feature_names),
            'corpus_size': len(texts),
            'engine': 'sklearn',
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def predict(self, features: np.ndarray, texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """預測文檔的主題分佈"""
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用fit方法")
        
        if texts is None:
            raise ValueError("LDA預測需要原始文本")
        
        if self.use_gensim:
            return self._predict_gensim(texts)
        else:
            return self._predict_sklearn(texts)
    
    def _predict_gensim(self, texts: pd.Series) -> Dict[str, Any]:
        """使用Gensim模型預測"""
        from gensim.utils import simple_preprocess
        from gensim.parsing.preprocessing import STOPWORDS
        
        predictions = []
        topic_distributions = []
        
        for text in texts:
            tokens = simple_preprocess(str(text), deacc=True)
            tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 3]
            bow = self.dictionary.doc2bow(tokens)
            
            doc_topics = self.model.get_document_topics(bow)
            
            # 找到主要主題
            if doc_topics:
                main_topic = max(doc_topics, key=lambda x: x[1])
                predictions.append(main_topic[0])
                
                # 構建完整的主題分佈向量
                topic_dist = np.zeros(self.n_topics)
                for topic_id, prob in doc_topics:
                    topic_dist[topic_id] = prob
                topic_distributions.append(topic_dist.tolist())
            else:
                predictions.append(-1)  # 無法分類
                topic_distributions.append([0] * self.n_topics)
        
        results = {
            'method': self.get_method_name(),
            'predictions': predictions,
            'topic_distributions': topic_distributions,
            'prediction_summary': {
                f'Topic_{i}': int(np.sum(np.array(predictions) == i))
                for i in range(self.n_topics)
            }
        }
        
        return results
    
    def _predict_sklearn(self, texts: pd.Series) -> Dict[str, Any]:
        """使用Sklearn模型預測"""
        text_vectors = self.vectorizer.transform(texts.astype(str))
        topic_distributions = self.model.transform(text_vectors)
        
        # 找到每個文檔的主要主題
        predictions = np.argmax(topic_distributions, axis=1)
        
        results = {
            'method': self.get_method_name(),
            'predictions': predictions.tolist(),
            'topic_distributions': topic_distributions.tolist(),
            'prediction_summary': {
                f'Topic_{i}': int(np.sum(predictions == i))
                for i in range(self.n_topics)
            }
        }
        
        return results

class BERTopicMethod(BaseClassificationMethod):
    """BERTopic主題建模方法"""
    
    def __init__(self, n_topics: int = 5, **kwargs):
        super().__init__(**kwargs)
        if not BERTOPIC_AVAILABLE:
            raise ImportError("BERTopic未安裝，請執行：pip install bertopic")
        
        self.n_topics = n_topics
        self.model = None
        self.is_fitted = False
        
    def get_method_name(self) -> str:
        return f"BERTopic主題建模 ({self.n_topics}主題)"
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
            texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """訓練BERTopic模型"""
        if texts is None:
            raise ValueError("BERTopic需要原始文本進行訓練")
        
        self._notify_progress('phase', 1, 3, f'{self.get_method_name()} - 初始化模型')
        
        # 初始化BERTopic模型（使用預訓練的嵌入）
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom')
        
        self.model = BERTopic(
            embedding_model=None,  # 我們將使用傳入的features
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=CountVectorizer(stop_words="english"),
            representation_model=None,
            nr_topics=self.n_topics
        )
        
        self._notify_progress('phase', 2, 3, f'{self.get_method_name()} - 模型訓練')
        
        # 訓練模型
        topics, probabilities = self.model.fit_transform(texts.tolist(), embeddings=features)
        
        self._notify_progress('phase', 3, 3, f'{self.get_method_name()} - 結果整理')
        
        # 整理結果
        topic_info = self.model.get_topic_info()
        
        # 提取主題詞
        topics_detail = []
        for topic_id in topic_info['Topic'].unique():
            if topic_id != -1:  # 排除噪音主題
                topic_words = self.model.get_topic(topic_id)
                topics_detail.append({
                    'topic_id': topic_id,
                    'words': [{'word': word, 'probability': prob} for word, prob in topic_words]
                })
        
        results = {
            'method': self.get_method_name(),
            'n_topics': len(topics_detail),
            'topics': topics_detail,
            'topic_info': topic_info.to_dict(),
            'corpus_size': len(texts),
            'outlier_count': int(np.sum(np.array(topics) == -1)),
            'timestamp': datetime.now().isoformat()
        }
        
        self.is_fitted = True
        self.results = results
        self.save_results(results, f"bertopic_{self.n_topics}topics_results.json")
        
        logger.info(f"{self.get_method_name()}訓練完成")
        return results
    
    def predict(self, features: np.ndarray, texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """預測文檔主題"""
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用fit方法")
        
        if texts is None:
            raise ValueError("BERTopic預測需要原始文本")
        
        topics, probabilities = self.model.transform(texts.tolist(), embeddings=features)
        
        results = {
            'method': self.get_method_name(),
            'predictions': topics,
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'prediction_summary': {
                f'Topic_{topic_id}': int(np.sum(np.array(topics) == topic_id))
                for topic_id in set(topics) if topic_id != -1
            }
        }
        
        return results

class NMFTopicMethod(BaseClassificationMethod):
    """非負矩陣分解（NMF）主題建模方法"""
    
    def __init__(self, n_topics: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.n_topics = n_topics
        self.model = None
        self.vectorizer = None
        self.is_fitted = False
    
    def get_method_name(self) -> str:
        return f"NMF主題建模 ({self.n_topics}主題)"
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
            texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """訓練NMF模型"""
        if texts is None:
            raise ValueError("NMF需要原始文本進行訓練")
        
        self._notify_progress('phase', 1, 3, f'{self.get_method_name()} - 文本向量化')
        
        # 創建TF-IDF向量器
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b',
            max_df=0.8,
            min_df=2
        )
        
        # 向量化文本
        text_vectors = self.vectorizer.fit_transform(texts.astype(str))
        
        self._notify_progress('phase', 2, 3, f'{self.get_method_name()} - 模型訓練')
        
        # 訓練NMF模型
        self.model = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=200,
            alpha=0.1,
            l1_ratio=0.5
        )
        
        topic_distributions = self.model.fit_transform(text_vectors)
        
        self._notify_progress('phase', 3, 3, f'{self.get_method_name()} - 提取主題')
        
        # 提取主題詞
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [
                {'word': feature_names[i], 'probability': topic[i]}
                for i in top_words_idx
            ]
            topics.append({
                'topic_id': topic_idx,
                'words': top_words
            })
        
        # 計算重構誤差
        reconstruction_error = self.model.reconstruction_err_
        
        results = {
            'method': self.get_method_name(),
            'n_topics': self.n_topics,
            'topics': topics,
            'reconstruction_error': reconstruction_error,
            'vocabulary_size': len(feature_names),
            'corpus_size': len(texts),
            'timestamp': datetime.now().isoformat()
        }
        
        self.is_fitted = True
        self.results = results
        self.save_results(results, f"nmf_topic_{self.n_topics}topics_results.json")
        
        logger.info(f"{self.get_method_name()}訓練完成")
        return results
    
    def predict(self, features: np.ndarray, texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """預測文檔的主題分佈"""
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用fit方法")
        
        if texts is None:
            raise ValueError("NMF預測需要原始文本")
        
        text_vectors = self.vectorizer.transform(texts.astype(str))
        topic_distributions = self.model.transform(text_vectors)
        
        # 找到每個文檔的主要主題
        predictions = np.argmax(topic_distributions, axis=1)
        
        results = {
            'method': self.get_method_name(),
            'predictions': predictions.tolist(),
            'topic_distributions': topic_distributions.tolist(),
            'prediction_summary': {
                f'Topic_{i}': int(np.sum(predictions == i))
                for i in range(self.n_topics)
            }
        }
        
        return results

class ClusteringMethod(BaseClassificationMethod):
    """聚類分析方法"""
    
    def __init__(self, method: str = 'kmeans', n_clusters: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.clustering_method = method
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # 初始化聚類模型
        self._init_model()
    
    def _init_model(self):
        """初始化聚類模型"""
        model_map = {
            'kmeans': KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10),
            'agglomerative': AgglomerativeClustering(n_clusters=self.n_clusters),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        
        if self.clustering_method not in model_map:
            available_methods = ', '.join(model_map.keys())
            raise ValueError(f"不支援的聚類方法：{self.clustering_method}。可用方法：{available_methods}")
        
        self.model = model_map[self.clustering_method]
    
    def get_method_name(self) -> str:
        return f"聚類分析 ({self.clustering_method.upper()}, {self.n_clusters}群)"
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None, 
            texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """執行聚類分析"""
        self._notify_progress('phase', 1, 4, f'{self.get_method_name()} - 資料預處理')
        
        # 標準化特徵
        features_scaled = self.scaler.fit_transform(features)
        
        self._notify_progress('phase', 2, 4, f'{self.get_method_name()} - 聚類分析')
        
        # 執行聚類
        cluster_labels = self.model.fit_predict(features_scaled)
        
        self._notify_progress('phase', 3, 4, f'{self.get_method_name()} - 降維視覺化')
        
        # 降維用於視覺化
        if features.shape[1] > 50:
            # 先用PCA降到50維
            pca = PCA(n_components=50)
            features_pca = pca.fit_transform(features_scaled)
            # 再用t-SNE降到2維
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features_pca)
        else:
            # 直接用t-SNE降到2維
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)//4))
            features_2d = tsne.fit_transform(features_scaled)
        
        self._notify_progress('phase', 4, 4, f'{self.get_method_name()} - 結果分析')
        
        # 分析聚類結果
        unique_labels = np.unique(cluster_labels)
        cluster_info = {}
        
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_size = np.sum(mask)
            cluster_info[f'Cluster_{label}'] = {
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(cluster_labels) * 100),
                'center': np.mean(features_2d[mask], axis=0).tolist() if cluster_size > 0 else [0, 0]
            }
        
        # 計算輪廓係數（如果不是DBSCAN且有足夠的群）
        silhouette_score = None
        if self.clustering_method != 'dbscan' and len(unique_labels) > 1:
            from sklearn.metrics import silhouette_score as calc_silhouette
            silhouette_score = calc_silhouette(features_scaled, cluster_labels)
        
        results = {
            'method': self.get_method_name(),
            'clustering_method': self.clustering_method,
            'n_clusters': len(unique_labels),
            'cluster_labels': cluster_labels.tolist(),
            'cluster_info': cluster_info,
            'features_2d': features_2d.tolist(),
            'silhouette_score': silhouette_score,
            'corpus_size': len(features),
            'feature_dim': features.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        
        self.is_fitted = True
        self.results = results
        self.save_results(results, f"clustering_{self.clustering_method}_{self.n_clusters}clusters_results.json")
        
        logger.info(f"{self.get_method_name()}完成")
        return results
    
    def predict(self, features: np.ndarray, texts: Optional[pd.Series] = None) -> Dict[str, Any]:
        """預測新樣本的聚類標籤"""
        if not self.is_fitted:
            raise ValueError("模型尚未訓練，請先調用fit方法")
        
        features_scaled = self.scaler.transform(features)
        
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(features_scaled)
        else:
            # 對於沒有predict方法的聚類器，使用距離最近的聚類中心
            from sklearn.metrics.pairwise import euclidean_distances
            centers = self.model.cluster_centers_ if hasattr(self.model, 'cluster_centers_') else None
            if centers is not None:
                distances = euclidean_distances(features_scaled, centers)
                predictions = np.argmin(distances, axis=1)
            else:
                raise ValueError(f"{self.clustering_method}不支援新樣本預測")
        
        results = {
            'method': self.get_method_name(),
            'predictions': predictions.tolist(),
            'prediction_summary': {
                f'Cluster_{i}': int(np.sum(predictions == i))
                for i in np.unique(predictions)
            }
        }
        
        return results

class ClassificationMethodFactory:
    """分類方法工廠類"""
    
    @staticmethod
    def create_method(method_type: str, **kwargs) -> BaseClassificationMethod:
        """
        創建指定類型的分類方法
        
        Args:
            method_type: 方法類型 ('sentiment', 'lda', 'bertopic', 'nmf', 'clustering')
            **kwargs: 其他參數
            
        Returns:
            BaseClassificationMethod: 相應的分類方法實例
        """
        method_map = {
            'sentiment': SentimentClassificationMethod,
            'lda': LDATopicMethod,
            'bertopic': BERTopicMethod,
            'nmf': NMFTopicMethod,
            'clustering': ClusteringMethod
        }
        
        if method_type.lower() not in method_map:
            available_types = ', '.join(method_map.keys())
            raise ValueError(f"不支援的分類方法：{method_type}。可用方法：{available_types}")
        
        method_class = method_map[method_type.lower()]
        return method_class(**kwargs)
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """獲取可用的分類方法列表"""
        available = ['sentiment', 'lda', 'nmf', 'clustering']
        
        if BERTOPIC_AVAILABLE:
            available.append('bertopic')
        
        return available
    
    @staticmethod
    def get_method_info() -> Dict[str, Dict]:
        """獲取分類方法的詳細信息"""
        info = {
            'sentiment': {
                'name': '情感分析',
                'description': '傳統的有監督情感分類',
                'type': '有監督學習',
                'advantages': '準確率高、可解釋性強',
                'requirements': 'sklearn, xgboost (可選)',
                'needs_labels': True
            },
            'lda': {
                'name': 'LDA主題建模',
                'description': '潛在狄利克雷分配主題建模',
                'type': '無監督學習',
                'advantages': '主題可解釋性強、適合文檔集合分析',
                'requirements': 'sklearn, gensim (可選)',
                'needs_labels': False
            },
            'bertopic': {
                'name': 'BERTopic',
                'description': '基於BERT的主題建模',
                'type': '無監督學習',
                'advantages': '利用預訓練模型、主題品質高',
                'requirements': 'bertopic, umap-learn, hdbscan',
                'needs_labels': False
            },
            'nmf': {
                'name': 'NMF主題建模',
                'description': '非負矩陣分解主題建模',
                'type': '無監督學習',
                'advantages': '計算效率高、主題分離度好',
                'requirements': 'sklearn',
                'needs_labels': False
            },
            'clustering': {
                'name': '聚類分析',
                'description': '基於特徵向量的聚類分析',
                'type': '無監督學習',
                'advantages': '發現數據結構、適合探索性分析',
                'requirements': 'sklearn',
                'needs_labels': False
            }
        }
        
        # 檢查可用性
        info['sentiment']['available'] = True
        info['lda']['available'] = True
        info['nmf']['available'] = True
        info['clustering']['available'] = True
        
        if BERTOPIC_AVAILABLE:
            info['bertopic']['available'] = True
        else:
            info['bertopic']['available'] = False
            info['bertopic']['note'] = '需要安裝bertopic, umap-learn, hdbscan'
        
        return info