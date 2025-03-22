import re
import nltk
import spacy
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import defaultdict, Counter

# 確保必要的NLTK資源已下載
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')

# 嘗試加載spaCy模型
try:
    nlp = spacy.load("zh_core_web_sm")  # 中文模型
except:
    try:
        nlp = spacy.load("en_core_web_sm")  # 英文模型
    except:
        print("未安裝spaCy模型，正在使用替代方法...")
        nlp = None

class AspectExtractor:
    """評論面相切割工具類"""
    
    def __init__(self, domain='general'):
        """
        初始化面相切割器
        
        Parameters:
        -----------
        domain : str
            數據領域，可選值為 'restaurant', 'electronics', 'movies', 'general'
        """
        self.domain = domain
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # 載入領域特定的面相詞典
        self.aspect_dict = self._load_aspect_dictionary(domain)
        
        # 正則表達式模式：用於基於規則的面相抽取
        self.patterns = {
            'noun_phrase': r'(?:JJ\s)?(?:NN[PS]?\s)+',  # 形容詞+名詞或純名詞短語
            'aspect_opinion': r'(\w+)/JJ\s(\w+)/NN[PS]?'  # 形容詞+名詞組合
        }
    
    def _load_aspect_dictionary(self, domain):
        """載入預先定義的領域特定面相詞典"""
        
        # 通用面相詞典（跨領域）
        general_aspects = {
            'quality': ['quality', 'condition', 'standard'],
            'price': ['price', 'cost', 'value', 'money', 'worth', 'expensive', 'cheap'],
            'overall': ['overall', 'general', 'generally', 'mostly', 'recommend']
        }
        
        # 餐廳領域面相詞典
        restaurant_aspects = {
            'food': ['food', 'dish', 'meal', 'appetizer', 'dessert', 'taste', 'flavor', 'delicious'],
            'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'manager'],
            'ambience': ['ambience', 'atmosphere', 'decor', 'environment', 'noise', 'music', 'seating'],
            'location': ['location', 'place', 'neighborhood', 'parking']
        }
        
        # 電子產品領域面相詞典
        electronics_aspects = {
            'performance': ['performance', 'speed', 'fast', 'slow', 'powerful', 'capability'],
            'design': ['design', 'look', 'appearance', 'style', 'color', 'size', 'weight'],
            'battery': ['battery', 'charge', 'power', 'life', 'lasting', 'duration'],
            'features': ['feature', 'function', 'capability', 'option'],
            'usability': ['use', 'usability', 'user-friendly', 'interface', 'menu', 'navigation']
        }
        
        # 電影領域面相詞典
        movies_aspects = {
            'plot': ['plot', 'story', 'storyline', 'screenplay', 'narrative', 'script'],
            'acting': ['acting', 'actor', 'actress', 'performance', 'character', 'role', 'cast'],
            'visuals': ['visual', 'effect', 'graphics', 'scene', 'cinematography', 'animation'],
            'sound': ['sound', 'music', 'soundtrack', 'score', 'song'],
            'direction': ['direction', 'director', 'editing', 'pacing', 'screenplay']
        }
        
        # 根據指定領域返回對應詞典
        domain_dict = general_aspects.copy()  # 先包含通用面相
        
        if domain == 'restaurant':
            domain_dict.update(restaurant_aspects)
        elif domain == 'electronics':
            domain_dict.update(electronics_aspects)
        elif domain == 'movies':
            domain_dict.update(movies_aspects)
        # 在general情況下只使用通用面相
        
        return domain_dict
    
    def rule_based_extraction(self, text):
        """使用基於規則的方法提取面相"""
        aspects = []
        
        # 使用NLTK進行詞性標註
        tokens = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(tokens)
        tagged_text = ' '.join([f"{word}/{tag}" for word, tag in tagged])
        
        # 找出所有名詞
        nouns = [word for word, tag in tagged if tag.startswith('NN')]
        
        # 找出通過模式匹配的名詞短語
        noun_phrases = re.findall(self.patterns['noun_phrase'], tagged_text)
        for phrase in noun_phrases:
            words = phrase.split()
            for word in words:
                if '/' in word:
                    word = word.split('/')[0]
                    if not self._is_stopword(word) and len(word) > 2:
                        aspects.append(word)
        
        # 添加所有單獨的名詞（可能是面相）
        for noun in nouns:
            if not self._is_stopword(noun) and len(noun) > 2:
                aspects.append(noun)
        
        # 檢查這些詞是否在面相詞典中
        matched_aspects = defaultdict(list)
        for aspect in aspects:
            aspect = self.lemmatizer.lemmatize(aspect)
            for category, words in self.aspect_dict.items():
                if aspect in words:
                    matched_aspects[category].append(aspect)
        
        return matched_aspects, aspects
    
    def dictionary_based_extraction(self, text):
        """使用詞典匹配方法提取面相"""
        matched_aspects = defaultdict(list)
        text_lower = text.lower()
        
        # 分詞並去除停用詞
        words = nltk.word_tokenize(text_lower)
        filtered_words = [word for word in words if not self._is_stopword(word)]
        
        # 詞形還原
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in filtered_words]
        
        # 檢查詞典匹配
        for word in lemmatized_words:
            for category, aspect_words in self.aspect_dict.items():
                if word in aspect_words:
                    matched_aspects[category].append(word)
        
        return matched_aspects
    
    def topic_modeling_extraction(self, texts, n_topics=5, max_features=1000, method='lda'):
        """
        使用主題模型提取文本集合中的潛在面相
        
        Parameters:
        -----------
        texts : list
            評論文本列表
        n_topics : int
            要提取的主題數量（對應於面相數量）
        max_features : int
            詞彙表大小
        method : str
            使用的主題模型方法，'lda' 或 'nmf'
        
        Returns:
        --------
        topics : list of list
            每個主題的關鍵詞列表
        """
        # 使用TF-IDF向量化文本
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        tfidf = vectorizer.fit_transform(texts)
        
        # 選擇主題模型
        if method == 'lda':
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        else:  # 'nmf'
            model = NMF(n_components=n_topics, random_state=42)
        
        model.fit(tfidf)
        
        # 提取每個主題的關鍵詞
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(model.components_):
            # 獲取前10個關鍵詞
            top_features_idx = topic.argsort()[:-11:-1]
            top_features = [feature_names[i] for i in top_features_idx]
            topics.append(top_features)
        
        return topics
    
    def spacy_based_extraction(self, text):
        """使用spaCy進行面相提取（依賴句法分析）"""
        if nlp is None:
            return defaultdict(list)
        
        aspects = defaultdict(list)
        doc = nlp(text)
        
        # 提取名詞短語作為可能的面相
        for chunk in doc.noun_chunks:
            # 僅考慮較長的名詞短語（可能更有意義）
            if len(chunk.text.split()) > 1:
                # 檢查是否匹配任何已知面相類別
                for category, words in self.aspect_dict.items():
                    for word in chunk.text.lower().split():
                        if word in words and not self._is_stopword(word):
                            aspects[category].append(chunk.text)
        
        return aspects
    
    def _is_stopword(self, word):
        """檢查詞是否為停用詞"""
        return word in self.stop_words or len(word) <= 2
    
    def extract_aspects(self, text, methods=None):
        """
        使用多種方法提取面相並返回合併結果
        
        Parameters:
        -----------
        text : str
            評論文本
        methods : list
            要使用的方法列表，可以是 ['rule', 'dictionary', 'spacy'] 的任意組合
            如果為None，則使用所有可用方法
        
        Returns:
        --------
        aspects : dict
            面相類別到對應面相詞的映射
        """
        if methods is None:
            methods = ['rule', 'dictionary']
            if nlp is not None:
                methods.append('spacy')
        
        all_aspects = defaultdict(list)
        
        for method in methods:
            if method == 'rule':
                categorized_aspects, raw_aspects = self.rule_based_extraction(text)
                for category, aspects in categorized_aspects.items():
                    all_aspects[category].extend(aspects)
            
            elif method == 'dictionary':
                categorized_aspects = self.dictionary_based_extraction(text)
                for category, aspects in categorized_aspects.items():
                    all_aspects[category].extend(aspects)
            
            elif method == 'spacy' and nlp is not None:
                categorized_aspects = self.spacy_based_extraction(text)
                for category, aspects in categorized_aspects.items():
                    all_aspects[category].extend(aspects)
        
        # 移除重複項
        for category in all_aspects:
            all_aspects[category] = list(set(all_aspects[category]))
        
        return dict(all_aspects)
    
    def batch_extract_aspects(self, texts, methods=None):
        """
        批量處理多個評論文本
        
        Parameters:
        -----------
        texts : list
            評論文本列表
        methods : list
            要使用的方法列表
        
        Returns:
        --------
        all_results : list of dict
            每個評論的面相提取結果
        topic_model_results : list of list
            主題模型提取的潛在面相主題
        """
        # 單獨處理每個文本
        individual_results = [self.extract_aspects(text, methods) for text in texts]
        
        # 如果文本數量足夠，使用主題模型找出潛在面相
        topic_model_results = None
        if len(texts) >= 20:  # 至少需要一定數量的文本才能進行主題建模
            topic_model_results = self.topic_modeling_extraction(texts)
        
        return individual_results, topic_model_results
    
    def analyze_aspects_distribution(self, texts, labels=None):
        """
        分析面相在評論集中的分布情況
        
        Parameters:
        -----------
        texts : list
            評論文本列表
        labels : list
            評論對應的情感標籤（正面/負面）
        
        Returns:
        --------
        aspect_stats : dict
            各面相類別的統計信息
        """
        aspect_counts = defaultdict(int)
        aspect_sentiment = defaultdict(list)
        
        # 處理每個評論並統計面相出現頻率
        for i, text in enumerate(texts):
            aspects = self.extract_aspects(text)
            
            for category in aspects:
                # 計算出現次數
                aspect_counts[category] += 1
                
                # 如果有提供情感標籤，則同時記錄面相對應的情感
                if labels is not None and i < len(labels):
                    sentiment = labels[i]
                    # 將情感轉換為數值（假設正面=1，負面=-1，中性=0）
                    if isinstance(sentiment, str):
                        if sentiment.lower() in ['positive', '正面']:
                            sentiment_value = 1
                        elif sentiment.lower() in ['negative', '負面']:
                            sentiment_value = -1
                        else:
                            sentiment_value = 0
                    else:  # 假設是數字評分
                        sentiment_value = 1 if sentiment > 3 else (-1 if sentiment < 3 else 0)
                    
                    aspect_sentiment[category].append(sentiment_value)
        
        # 計算面相統計信息
        aspect_stats = {}
        for category in aspect_counts:
            stats = {
                'count': aspect_counts[category],
                'percentage': aspect_counts[category] / len(texts) * 100
            }
            
            if labels is not None and category in aspect_sentiment:
                sentiments = aspect_sentiment[category]
                if sentiments:
                    stats['avg_sentiment'] = sum(sentiments) / len(sentiments)
                    stats['positive_count'] = sum(1 for s in sentiments if s > 0)
                    stats['negative_count'] = sum(1 for s in sentiments if s < 0)
                    stats['neutral_count'] = sum(1 for s in sentiments if s == 0)
            
            aspect_stats[category] = stats
        
        return aspect_stats


def segment_reviews_by_aspects(df, domain='general', text_column='review', 
                              label_column=None, methods=None):
    """
    將DataFrame中的評論按面相切割，並添加面相相關列
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含評論的DataFrame
    domain : str
        評論領域（'restaurant', 'electronics', 'movies', 'general'）
    text_column : str
        評論文本列的名稱
    label_column : str
        評論情感標籤列的名稱（如果有）
    methods : list
        面相提取使用的方法
    
    Returns:
    --------
    segmented_df : pandas.DataFrame
        包含面相切割結果的DataFrame
    """
    # 初始化面相提取器
    extractor = AspectExtractor(domain=domain)
    
    # 獲取評論文本
    reviews = df[text_column].astype(str).tolist()
    
    # 獲取情感標籤（如果有）
    labels = None
    if label_column is not None and label_column in df.columns:
        labels = df[label_column].tolist()
    
    # 提取面相
    results = []
    for i, review in enumerate(reviews):
        # 獲取當前評論的面相
        aspects = extractor.extract_aspects(review, methods)
        
        # 創建面相列表（用於添加到DataFrame）
        aspect_list = []
        aspect_categories = []
        
        for category, terms in aspects.items():
            if terms:  # 確保有面相詞
                aspect_categories.append(category)
                aspect_list.extend(terms)
        
        # 建立結果字典
        result = {
            'review_id': i,
            'original_review': review,
            'aspects_found': len(aspect_list) > 0,
            'aspect_terms': '|'.join(aspect_list) if aspect_list else '',
            'aspect_categories': '|'.join(aspect_categories) if aspect_categories else '',
            'aspects_count': len(aspect_list)
        }
        
        # 如果有情感標籤，加入結果
        if labels is not None and i < len(labels):
            result['sentiment'] = labels[i]
        
        results.append(result)
    
    # 創建結果DataFrame
    segmented_df = pd.DataFrame(results)
    
    # 進行面相分布分析
    aspect_stats = extractor.analyze_aspects_distribution(reviews, labels)
    
    # 如果文本數量足夠，進行主題建模
    topic_results = None
    if len(reviews) >= 20:
        individual_results, topic_results = extractor.batch_extract_aspects(reviews, methods)
    
    return segmented_df, aspect_stats, topic_results


def create_aspect_specific_dataset(df, segmented_df, aspect_categories=None):
    """
    根據指定的面相類別創建面相特定的數據集
    
    Parameters:
    -----------
    df : pandas.DataFrame
        原始數據集
    segmented_df : pandas.DataFrame
        面相切割後的數據集
    aspect_categories : list
        要包含的面相類別列表
    
    Returns:
    --------
    aspect_df : pandas.DataFrame
        面相特定數據集
    """
    if aspect_categories is None:
        # 默認使用所有出現的面相類別
        all_categories = []
        for cats in segmented_df['aspect_categories']:
            if cats:
                all_categories.extend(cats.split('|'))
        aspect_categories = list(set(all_categories))
    
    # 創建面相特定的資料集
    aspect_specific_rows = []
    
    for i, row in segmented_df.iterrows():
        if not row['aspects_found'] or not row['aspect_categories']:
            continue
        
        # 獲取當前評論的面相類別
        current_categories = row['aspect_categories'].split('|')
        
        # 檢查是否包含指定的面相類別
        for category in aspect_categories:
            if category in current_categories:
                # 創建一個面相特定的行
                aspect_row = {
                    'review_id': row['review_id'],
                    'original_review': row['original_review'],
                    'aspect_category': category,
                    'aspect_terms': '|'.join([term for term in row['aspect_terms'].split('|') 
                                            if any(term in extractor.aspect_dict.get(cat, []) 
                                                  for cat in current_categories)])
                }
                
                # 如果有情感標籤，加入結果
                if 'sentiment' in row:
                    aspect_row['sentiment'] = row['sentiment']
                
                aspect_specific_rows.append(aspect_row)
    
    # 創建結果DataFrame
    aspect_df = pd.DataFrame(aspect_specific_rows)
    
    return aspect_df


if __name__ == "__main__":
    # 測試代碼
    # 初始化面相提取器
    extractor = AspectExtractor(domain='restaurant')
    
    # 測試文本
    test_text = "The food was delicious but the service was terrible. The ambiance was nice though."
    
    # 提取面相
    aspects = extractor.extract_aspects(test_text)
    print("提取的面相:", aspects)
    
    # 測試多個評論
    test_texts = [
        "The pizza was amazing and the staff was very friendly.",
        "The battery life on this phone is terrible, but the camera quality is excellent.",
        "The acting was superb, but the plot was confusing and too complex."
    ]
    
    # 批量提取面相
    individual_results, topic_results = extractor.batch_extract_aspects(test_texts)
    print("\n批量處理結果:")
    for i, result in enumerate(individual_results):
        print(f"評論 {i+1}: {result}")
    
    if topic_results:
        print("\n主題模型結果:")
        for i, topic in enumerate(topic_results):
            print(f"主題 {i+1}: {', '.join(topic)}")