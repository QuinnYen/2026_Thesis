"""
面向分類器工廠模組
提供各種面向分類方法的統一接口和管理
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple, Optional, Any
import logging
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from .base_interfaces import BaseAspectClassifier
from .attention_mechanism import AttentionMechanism

logger = logging.getLogger(__name__)


class DefaultAspectClassifier(BaseAspectClassifier):
    """預設面向分類器 - 基於現有的注意力機制系統"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        self.attention_types = self.config.get('attention_types', ['similarity', 'keyword', 'self', 'combined'])
        self.topic_keywords = self.config.get('topic_keywords', None)
        self.attention_mechanism = AttentionMechanism(self.config)
        self.aspect_vectors = None
        self.aspect_names = []
        
        if self.progress_callback:
            self.progress_callback('status', '預設注意力機制分類器已初始化')
    
    def fit_transform(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """使用注意力機制進行面向分類"""
        results = {}
        
        # 載入主題關鍵詞
        if self.topic_keywords is None:
            topic_labels_path = os.path.join(os.path.dirname(__file__), '..', 'utils', 'topic_labels.json')
            if os.path.exists(topic_labels_path):
                with open(topic_labels_path, 'r', encoding='utf-8') as f:
                    topic_data = json.load(f)
                    self.topic_keywords = topic_data.get('topic_keywords', {})
        
        # 使用組合注意力機制作為主要方法
        attention_result, attention_info = self.attention_mechanism.compute_attention(
            embeddings, metadata, 
            attention_type='combined',
            topic_keywords=self.topic_keywords
        )
        
        self.aspect_vectors = attention_result
        self.aspect_names = list(self.topic_keywords.keys()) if self.topic_keywords else [f"Aspect_{i}" for i in range(attention_result.shape[0])]
        
        results = {
            'attention_result': attention_result,
            'attention_info': attention_info,
            'aspect_names': self.aspect_names,
            'method': 'attention_mechanism'
        }
        
        if self.progress_callback:
            self.progress_callback('status', f'預設分類器完成，發現 {len(self.aspect_names)} 個面向')
        
        return attention_result, results
    
    def get_aspect_names(self) -> List[str]:
        return self.aspect_names


class LDAAspectClassifier(BaseAspectClassifier):
    """LDA面向分類器"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        self.n_topics = self.config.get('n_topics', 10)
        self.max_features = self.config.get('max_features', 1000)
        self.random_state = self.config.get('random_state', 42)
        
        self.vectorizer = CountVectorizer(max_features=self.max_features, stop_words='english')
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=20
        )
        self.aspect_names = []
        
        if self.progress_callback:
            self.progress_callback('status', f'LDA分類器已初始化 (主題數: {self.n_topics})')
    
    def fit_transform(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """使用LDA進行主題建模"""
        
        # 確保有文本數據
        if 'text' not in metadata.columns:
            raise ValueError("LDA需要原始文本數據，metadata中必須包含'text'欄位")
        
        texts = metadata['text'].fillna('').astype(str).tolist()
        
        if self.progress_callback:
            self.progress_callback('status', 'LDA正在處理文本...')
        
        # 文本向量化
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        if self.progress_callback:
            self.progress_callback('status', 'LDA正在訓練主題模型...')
        
        # 訓練LDA模型
        self.lda_model.fit(doc_term_matrix)
        
        # 獲取文檔-主題分佈
        doc_topic_dist = self.lda_model.transform(doc_term_matrix)
        
        # 獲取主題-詞語分佈
        topic_word_dist = self.lda_model.components_
        
        # 為每個主題生成名稱
        feature_names = self.vectorizer.get_feature_names_out()
        self.aspect_names = []
        
        for topic_idx, topic in enumerate(topic_word_dist):
            top_words = [feature_names[i] for i in topic.argsort()[::-1][:5]]
            topic_name = f"Topic_{topic_idx}_{'+'.join(top_words[:3])}"
            self.aspect_names.append(topic_name)
        
        # 計算面向向量（使用主題的詞語分佈）
        aspect_vectors = topic_word_dist
        
        results = {
            'doc_topic_dist': doc_topic_dist,
            'topic_word_dist': topic_word_dist,
            'feature_names': feature_names,
            'aspect_names': self.aspect_names,
            'lda_model': self.lda_model,
            'vectorizer': self.vectorizer,
            'method': 'lda'
        }
        
        if self.progress_callback:
            self.progress_callback('status', f'LDA完成，發現 {self.n_topics} 個主題')
        
        return aspect_vectors, results
    
    def get_aspect_names(self) -> List[str]:
        return self.aspect_names


class BERTopicAspectClassifier(BaseAspectClassifier):
    """BERTopic面向分類器"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        self.min_topic_size = self.config.get('min_topic_size', 10)
        self.n_topics = self.config.get('n_topics', 'auto')
        self.topic_model = None
        self.fallback_mode = False
        
        try:
            # 嘗試匯入所有必需的套件
            from bertopic import BERTopic
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
            
            # 配置UMAP和HDBSCAN
            umap_model = UMAP(n_neighbors=15, n_components=5, random_state=42)
            hdbscan_model = HDBSCAN(min_cluster_size=self.min_topic_size, prediction_data=True)
            vectorizer_model = CountVectorizer(stop_words="english")
            
            self.topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model
            )
            
            if self.progress_callback:
                self.progress_callback('status', 'BERTopic分類器已初始化')
                
        except ImportError as e:
            logger.warning(f"BERTopic依賴項缺失，將使用降級方案: {e}")
            self.fallback_mode = True
            
            # 檢查哪些套件缺失
            missing_packages = []
            try:
                import bertopic
            except ImportError:
                missing_packages.append("bertopic")
            try:
                import umap
            except ImportError:
                missing_packages.append("umap-learn")
            try:
                import hdbscan
            except ImportError:
                missing_packages.append("hdbscan")
            
            if self.progress_callback:
                missing_str = ", ".join(missing_packages)
                self.progress_callback('warning', f'BERTopic功能不可用，缺失套件: {missing_str}')
                self.progress_callback('status', '將使用基於NMF的降級主題建模')
            
            logger.info(f"BERTopic降級到NMF，缺失套件: {missing_packages}")
        
        self.aspect_names = []
    
    def fit_transform(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """使用BERTopic進行主題建模，如果不可用則降級到NMF"""
        
        # 確保有文本數據
        if 'text' not in metadata.columns:
            raise ValueError("主題建模需要原始文本數據，metadata中必須包含'text'欄位")
        
        texts = metadata['text'].fillna('').astype(str).tolist()
        
        if self.fallback_mode:
            # 降級到NMF方案
            return self._fallback_nmf_fit_transform(embeddings, metadata, texts)
        
        if self.progress_callback:
            self.progress_callback('status', 'BERTopic正在訓練主題模型...')
        
        try:
            # 使用預計算的嵌入向量進行主題建模
            topics, probs = self.topic_model.fit_transform(texts, embeddings)
            
            # 獲取主題信息
            topic_info = self.topic_model.get_topic_info()
            topic_keywords = {}
            
            for topic_id in topic_info['Topic']:
                if topic_id != -1:  # 排除噪音主題
                    keywords = [word for word, _ in self.topic_model.get_topic(topic_id)]
                    topic_name = f"Topic_{topic_id}_{'+'.join(keywords[:3])}"
                    self.aspect_names.append(topic_name)
                    topic_keywords[topic_name] = keywords
            
            # 計算主題向量（使用主題的關鍵詞表示）
            aspect_vectors = []
            for topic_id in sorted(set(topics)):
                if topic_id != -1:  # 排除噪音主題
                    # 獲取屬於該主題的文檔索引
                    topic_docs = [i for i, t in enumerate(topics) if t == topic_id]
                    if topic_docs:
                        # 計算該主題的平均嵌入向量
                        topic_vector = embeddings[topic_docs].mean(axis=0)
                        aspect_vectors.append(topic_vector)
            
            aspect_vectors = np.array(aspect_vectors)
            
            results = {
                'topics': topics,
                'topic_probs': probs,
                'topic_info': topic_info,
                'topic_keywords': topic_keywords,
                'aspect_names': self.aspect_names,
                'topic_model': self.topic_model,
                'method': 'bertopic'
            }
            
            if self.progress_callback:
                self.progress_callback('status', f'BERTopic完成，發現 {len(self.aspect_names)} 個主題')
            
            return aspect_vectors, results
            
        except Exception as e:
            logger.error(f"BERTopic執行時錯誤，降級到NMF: {e}")
            if self.progress_callback:
                self.progress_callback('warning', f'BERTopic執行失敗，降級到NMF: {str(e)}')
            
            return self._fallback_nmf_fit_transform(embeddings, metadata, texts)
            
    def _fallback_nmf_fit_transform(self, embeddings: np.ndarray, metadata: pd.DataFrame, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        """NMF降級方案"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF
        
        if self.progress_callback:
            self.progress_callback('status', '使用NMF降級方案進行主題建模...')
        
        # 使用TF-IDF向量化
        n_topics = int(self.n_topics) if isinstance(self.n_topics, (int, str)) and str(self.n_topics).isdigit() else 10
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # 執行NMF分解
        nmf_model = NMF(n_components=n_topics, random_state=42, max_iter=200)
        W = nmf_model.fit_transform(doc_term_matrix)  # 文檔-主題矩陣
        H = nmf_model.components_  # 主題-詞語矩陣
        
        # 為每個主題生成名稱
        feature_names = vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(H):
            top_words = [feature_names[i] for i in topic.argsort()[::-1][:5]]
            topic_name = f"Topic_{topic_idx}_{'+'.join(top_words[:3])}"
            self.aspect_names.append(topic_name)
        
        # 計算主題向量（使用主題的詞語分佈）
        aspect_vectors = H
        
        # 生成主題分配（每個文檔的主要主題）
        topics = W.argmax(axis=1)
        topic_probs = W / W.sum(axis=1, keepdims=True)
        
        results = {
            'topics': topics,
            'topic_probs': topic_probs,
            'doc_topic_matrix': W,
            'topic_word_matrix': H,
            'feature_names': feature_names,
            'aspect_names': self.aspect_names,
            'nmf_model': nmf_model,
            'vectorizer': vectorizer,
            'method': 'nmf_fallback'
        }
        
        if self.progress_callback:
            self.progress_callback('status', f'NMF降級方案完成，發現 {n_topics} 個主題')
        
        return aspect_vectors, results
    
    def get_aspect_names(self) -> List[str]:
        return self.aspect_names


class NMFAspectClassifier(BaseAspectClassifier):
    """NMF面向分類器"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        self.n_topics = self.config.get('n_topics', 10)
        self.max_features = self.config.get('max_features', 1000)
        self.random_state = self.config.get('random_state', 42)
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=200
        )
        self.aspect_names = []
        
        if self.progress_callback:
            self.progress_callback('status', f'NMF分類器已初始化 (主題數: {self.n_topics})')
    
    def fit_transform(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """使用NMF進行非負矩陣分解"""
        
        # 確保有文本數據
        if 'text' not in metadata.columns:
            raise ValueError("NMF需要原始文本數據，metadata中必須包含'text'欄位")
        
        texts = metadata['text'].fillna('').astype(str).tolist()
        
        if self.progress_callback:
            self.progress_callback('status', 'NMF正在處理文本...')
        
        # 文本向量化
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        if self.progress_callback:
            self.progress_callback('status', 'NMF正在執行矩陣分解...')
        
        # 執行NMF分解
        W = self.nmf_model.fit_transform(doc_term_matrix)  # 文檔-主題矩陣
        H = self.nmf_model.components_  # 主題-詞語矩陣
        
        # 為每個主題生成名稱
        feature_names = self.vectorizer.get_feature_names_out()
        self.aspect_names = []
        
        for topic_idx, topic in enumerate(H):
            top_words = [feature_names[i] for i in topic.argsort()[::-1][:5]]
            topic_name = f"Topic_{topic_idx}_{'+'.join(top_words[:3])}"
            self.aspect_names.append(topic_name)
        
        # 計算面向向量（使用主題的詞語分佈）
        aspect_vectors = H
        
        results = {
            'doc_topic_matrix': W,
            'topic_word_matrix': H,
            'feature_names': feature_names,
            'aspect_names': self.aspect_names,
            'nmf_model': self.nmf_model,
            'vectorizer': self.vectorizer,
            'method': 'nmf'
        }
        
        if self.progress_callback:
            self.progress_callback('status', f'NMF完成，發現 {self.n_topics} 個主題')
        
        return aspect_vectors, results
    
    def get_aspect_names(self) -> List[str]:
        return self.aspect_names


class AspectFactory:
    """面向分類器工廠類"""
    
    _classifiers = {
        'default': DefaultAspectClassifier,
        'lda': LDAAspectClassifier,
        'bertopic': BERTopicAspectClassifier,
        'nmf': NMFAspectClassifier
    }
    
    @classmethod
    def create_classifier(cls, classifier_type: str, config: Optional[Dict] = None,
                         progress_callback=None) -> BaseAspectClassifier:
        """
        創建指定類型的面向分類器
        
        Args:
            classifier_type: 分類器類型 ('default', 'lda', 'bertopic', 'nmf')
            config: 配置參數
            progress_callback: 進度回調函數
            
        Returns:
            BaseAspectClassifier: 分類器實例
        """
        classifier_type = classifier_type.lower()
        if classifier_type not in cls._classifiers:
            raise ValueError(f"不支援的面向分類器類型: {classifier_type}")
        
        classifier_class = cls._classifiers[classifier_type]
        return classifier_class(config, progress_callback)
    
    @classmethod
    def get_available_classifiers(cls) -> List[str]:
        """獲取可用的分類器列表"""
        return list(cls._classifiers.keys())
    
    @classmethod
    def get_classifier_info(cls, classifier_type: str) -> Dict[str, Any]:
        """獲取分類器信息"""
        classifier_descriptions = {
            'default': {
                'name': '預設注意力機制',
                'description': '基於多種注意力機制的面向分類方法',
                'advantages': ['準確率高', '可解釋性強', '支援多種注意力機制'],
                'disadvantages': ['需要預定義關鍵詞', '計算複雜度較高'],
                'suitable_for': ['有明確面向定義的任務', '需要高準確率的場景']
            },
            'lda': {
                'name': 'LDA主題建模',
                'description': 'Latent Dirichlet Allocation 潛在狄利克雷分配',
                'advantages': ['無監督學習', '可解釋性好', '計算效率高'],
                'disadvantages': ['需要預設主題數', '假設詞袋模型'],
                'suitable_for': ['探索性分析', '文檔主題發現', '中等規模數據集']
            },
            'bertopic': {
                'name': 'BERTopic',
                'description': '基於BERT和聚類的主題建模方法',
                'advantages': ['自動發現主題數', '語義理解強', '效果優秀'],
                'disadvantages': ['計算資源需求高', '需要額外依賴'],
                'suitable_for': ['大規模數據', '需要高質量主題的場景']
            },
            'nmf': {
                'name': 'NMF非負矩陣分解',
                'description': 'Non-negative Matrix Factorization',
                'advantages': ['計算效率高', '結果稀疏', '易於解釋'],
                'disadvantages': ['需要預設主題數', '對初始化敏感'],
                'suitable_for': ['計算資源有限', '需要稀疏表示的場景']
            }
        }
        
        return classifier_descriptions.get(classifier_type.lower(), {})