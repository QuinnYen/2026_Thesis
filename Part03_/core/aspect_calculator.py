"""
面向計算模組
此模組負責將文本特徵和主題分布結合起來計算面向向量
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import pickle
import logging
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import defaultdict

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 從utils模組導入工具類
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.config_manager import ConfigManager
from Part03_.core.bert_embedder import BertEmbedder
from Part03_.core.topic_extractor import TopicExtractor
from Part03_.core.attention_mechanism import AttentionProcessor

class AspectCalculator:
    """
    面向向量計算類
    計算評論中的面向特徵向量以用於多面向情感分析
    """
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化面向計算器
        
        Args:
            config: 配置管理器，如果沒有提供則創建新的
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 設置輸出目錄
        self.output_dir = self.config.get('data_settings.output_directory', './Part03_/results/')
        self.aspects_dir = os.path.join(self.output_dir, '04_aspect_vectors')
        os.makedirs(self.aspects_dir, exist_ok=True)
        
        # 初始化日誌
        log_dir = self.config.get('logging.log_dir', './Part03_/logs/')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'aspect_calculator.log')
        
        self.logger = logging.getLogger('aspect_calculator')
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
        
        # 初始化其他組件
        self.bert_embedder = None
        self.topic_extractor = None
        self.attention_processor = None
        self.aspects = {}
        self.aspect_vectors = {}
        self.aspect_labels = {}
        # 儲存面向組合的結果
        self.aspect_combinations = {}
        
    def initialize_components(self, bert_model_name: str = 'bert-base-chinese',
                           use_cuda: bool = True) -> None:
        """
        初始化BERT嵌入生成器和主題提取器
        
        Args:
            bert_model_name: BERT模型名稱
            use_cuda: 是否使用CUDA加速
        """
        # 初始化BERT嵌入生成器
        self.bert_embedder = BertEmbedder(
            model_name=bert_model_name,
            use_cuda=use_cuda,
            config=self.config
        )
        
        # 初始化主題提取器
        self.topic_extractor = TopicExtractor(config=self.config)
        
        # 初始化注意力處理器
        self.attention_processor = AttentionProcessor(config=self.config)
        
        self.logger.info(f"已初始化組件，使用BERT模型: {bert_model_name}")
    
    def calculate_aspect_vectors(self, df: pd.DataFrame, 
                               text_column: str = 'processed_text',
                               id_column: str = 'id', 
                               bert_embeddings: Optional[Dict[str, np.ndarray]] = None,
                               lda_results: Optional[Dict[str, Any]] = None,
                               num_keywords_per_aspect: int = 5,
                               console_output: bool = True) -> Dict[str, Any]:
        """
        計算面向向量
        
        Args:
            df: 輸入數據框
            text_column: 文本列名
            id_column: ID列名
            bert_embeddings: 預先計算的BERT嵌入 {id: vector}
            lda_results: 預先計算的LDA結果
            num_keywords_per_aspect: 每個面向的關鍵詞數量
            console_output: 是否顯示處理進度
            
        Returns:
            面向向量計算結果
        """
        # 設置控制台輸出
        if (console_output):
            log_file, status_file = ConsoleOutputManager.open_console("面向向量計算")
            logger = ConsoleOutputManager.setup_console_logger("aspect_vectors", log_file)
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
            
            # 確保ID列存在
            if id_column not in df.columns:
                error_msg = f"數據集中不存在列 '{id_column}'"
                logger.error(error_msg)
                if console_output:
                    ConsoleOutputManager.mark_process_complete(status_file)
                raise ValueError(error_msg)
            
            # 1. 獲取BERT嵌入向量
            if bert_embeddings is None:
                logger.info("計算BERT嵌入向量...")
                if self.bert_embedder is None:
                    self.initialize_components()
                
                bert_embeddings = self.bert_embedder.process_dataset(
                    df, text_column=text_column, id_column=id_column,
                    console_output=False
                )
            else:
                logger.info("使用預先計算的BERT嵌入向量")
            
            # 檢查每個ID是否都有對應的嵌入向量
            missing_ids = [id_val for id_val in df[id_column].tolist() if id_val not in bert_embeddings]
            if missing_ids:
                logger.warning(f"有 {len(missing_ids)} 個ID沒有對應的BERT嵌入向量")
            
            # 2. 獲取LDA主題模型結果
            if lda_results is None:
                logger.info("計算LDA主題模型...")
                if self.topic_extractor is None:
                    self.topic_extractor = TopicExtractor(config=self.config)
                
                lda_results = self.topic_extractor.extract_topics(
                    df, text_column=text_column, console_output=False
                )
            else:
                logger.info("使用預先計算的LDA主題模型")
            
            # 3. 提取主題詞作為面向詞
            topics = lda_results['topics']
            aspects = {}
            
            for topic in topics:
                topic_id = topic['id']
                # 獲取前N個關鍵詞作為面向詞
                aspect_terms = [word for word, _ in topic['words'][:num_keywords_per_aspect]]
                aspects[topic_id] = aspect_terms
            
            # 記錄面向詞
            self.aspects = aspects
            
            # 建立面向標籤 - 使用前兩個關鍵詞
            aspect_labels = {}
            for topic_id, aspect_terms in aspects.items():
                if len(aspect_terms) >= 2:
                    label = f"{aspect_terms[0]}+{aspect_terms[1]}"
                elif len(aspect_terms) == 1:
                    label = aspect_terms[0]
                else:
                    label = f"面向{topic_id+1}"
                aspect_labels[topic_id] = label
            
            self.aspect_labels = aspect_labels
            
            # 4. 計算每個面向的代表向量
            logger.info("計算面向向量...")
            aspect_vectors = {}
            
            # 獲取文檔-主題分布
            doc_topic_matrix = lda_results['doc_topics']
            
            # 遍歷每個面向
            for topic_id, aspect_terms in aspects.items():
                logger.info(f"計算面向 {topic_id+1} ({aspect_labels[topic_id]}) 的向量表示...")
                
                # 獲取該主題的權重向量
                topic_weights = doc_topic_matrix[:, topic_id]
                
                # 找出權重高於閾值的文檔
                threshold = np.percentile(topic_weights, 80)  # 取前20%
                high_weight_indices = np.where(topic_weights >= threshold)[0]
                
                # 如果沒有高權重文檔，則使用全部文檔
                if len(high_weight_indices) == 0:
                    high_weight_indices = np.arange(len(topic_weights))
                
                # 獲取高權重文檔ID
                high_weight_ids = df.iloc[high_weight_indices][id_column].tolist()
                
                # 從BERT嵌入中提取這些文檔的向量
                high_weight_vectors = np.vstack([bert_embeddings[id_val] for id_val in high_weight_ids 
                                               if id_val in bert_embeddings])
                
                # 計算面向向量 - 使用加權平均
                weights = topic_weights[high_weight_indices]
                weights = weights / np.sum(weights)  # 歸一化
                
                # 計算加權平均
                aspect_vector = np.average(high_weight_vectors, axis=0, weights=weights)
                
                # 歸一化向量
                aspect_vector = aspect_vector / np.linalg.norm(aspect_vector)
                
                aspect_vectors[topic_id] = {
                    'vector': aspect_vector,
                    'keywords': aspect_terms,
                    'label': aspect_labels[topic_id]
                }
            
            self.aspect_vectors = aspect_vectors
            
            # 5. 計算文本-面向關聯度矩陣
            logger.info("計算文本與面向的關聯度...")
            text_aspect_matrix = np.zeros((len(df), len(aspects)))
            
            # 獲取所有文本的ID
            all_ids = df[id_column].tolist()
            
            # 建立ID到索引的映射
            id_to_index = {id_val: idx for idx, id_val in enumerate(all_ids)}
            
            # 遍歷每個面向
            for topic_id, aspect_info in aspect_vectors.items():
                aspect_vector = aspect_info['vector']
                
                # 計算每個文本與該面向的餘弦相似度
                for id_val, bert_vector in bert_embeddings.items():
                    if id_val in id_to_index:  # 確保ID在數據集中
                        idx = id_to_index[id_val]
                        # 計算餘弦相似度
                        similarity = cosine_similarity(
                            bert_vector.reshape(1, -1),
                            aspect_vector.reshape(1, -1)
                        )[0, 0]
                        
                        # 儲存到矩陣
                        text_aspect_matrix[idx, topic_id] = similarity
            
            # 歸一化每行（每個文本的面向分布總和為1）
            row_sums = text_aspect_matrix.sum(axis=1, keepdims=True)
            # 避免除零錯誤
            row_sums[row_sums == 0] = 1.0
            text_aspect_matrix = text_aspect_matrix / row_sums
            
            # 6. 創建結果字典
            result = {
                'aspect_vectors': aspect_vectors,
                'aspects': aspects,
                'aspect_labels': aspect_labels,
                'text_aspect_matrix': text_aspect_matrix,
                'id_to_index': id_to_index
            }
            
            logger.info("面向向量計算完成")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return result
            
        except Exception as e:
            logger.error(f"面向向量計算過程中發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def calculate_aspect_combination(self, aspect_ids: List[int], 
                                   method: str = 'attention_weighted') -> Dict[str, Any]:
        """
        計算多個面向的組合向量表示
        使用注意力機制來權衡不同面向的重要性
        
        Args:
            aspect_ids: 要組合的面向ID列表
            method: 組合方法，可選值:
                   'simple_average' - 簡單平均
                   'attention_weighted' - 注意力加權
                   'multi_head_attention' - 多頭注意力
                   
        Returns:
            組合面向的信息，包括向量表示和標籤
        """
        if not self.aspect_vectors:
            raise RuntimeError("尚未計算面向向量")
        
        if self.attention_processor is None:
            if self.config.get('processing.use_cuda', True):
                use_cuda = True
            else:
                use_cuda = False
            self.attention_processor = AttentionProcessor(config=self.config)
        
        # 確保所有面向ID都存在
        for aspect_id in aspect_ids:
            if aspect_id not in self.aspect_vectors:
                raise ValueError(f"面向ID {aspect_id} 不存在")
        
        # 如果只有一個面向，直接返回
        if len(aspect_ids) == 1:
            return self.aspect_vectors[aspect_ids[0]]
        
        # 生成組合的唯一標識符
        combination_key = '+'.join([str(aid) for aid in sorted(aspect_ids)])
        
        # 如果已經計算過該組合，直接返回
        if combination_key in self.aspect_combinations:
            return self.aspect_combinations[combination_key]
        
        # 提取面向向量和關鍵詞
        vectors = [self.aspect_vectors[a_id]['vector'] for a_id in aspect_ids]
        keywords_lists = [self.aspect_vectors[a_id]['keywords'] for a_id in aspect_ids]
        labels = [self.aspect_vectors[a_id]['label'] for a_id in aspect_ids]
        
        # 根據不同方法計算組合向量
        if method == 'simple_average':
            # 簡單平均
            combined_vector = np.mean(vectors, axis=0)
        
        elif method == 'attention_weighted':
            # 使用自注意力機制進行加權
            # 將向量組織成適合注意力機制的形狀
            stacked_vectors = np.vstack(vectors).reshape(1, len(vectors), -1)  # [1, n_aspects, hidden_dim]
            
            # 應用自注意力機制
            weighted_output, attn_weights = self.attention_processor.apply_self_attention(stacked_vectors)
            combined_vector = weighted_output[0]  # 去除batch維度
            
            # 記錄面向權重
            aspect_weights = {}
            for i, aspect_id in enumerate(aspect_ids):
                aspect_weights[aspect_id] = float(attn_weights[0, i])
            
            self.logger.info(f"面向組合 {combination_key} 的注意力權重: {aspect_weights}")
        
        elif method == 'multi_head_attention':
            # 使用多頭注意力機制
            stacked_vectors = np.vstack(vectors).reshape(1, len(vectors), -1)  # [1, n_aspects, hidden_dim]
            
            # 應用多頭注意力機制
            output, _ = self.attention_processor.apply_multi_head_attention(stacked_vectors)
            
            # 通過平均或最大池化獲取上下文向量
            combined_vector = np.mean(output[0], axis=0)
        
        else:
            raise ValueError(f"不支持的組合方法: {method}")
        
        # 歸一化向量
        combined_vector = combined_vector / np.linalg.norm(combined_vector)
        
        # 合併關鍵詞 (取前3個關鍵詞)
        combined_keywords = []
        for kw_list in keywords_lists:
            combined_keywords.extend(kw_list[:3])
        combined_keywords = list(dict.fromkeys(combined_keywords))[:5]  # 去重並限制數量
        
        # 創建組合標籤
        combined_label = '+'.join(labels)
        
        # 創建組合面向的信息
        combination_info = {
            'vector': combined_vector,
            'keywords': combined_keywords,
            'label': combined_label,
            'component_aspects': aspect_ids,
            'method': method
        }
        
        # 保存到組合緩存
        self.aspect_combinations[combination_key] = combination_info
        
        return combination_info
    
    def compare_aspect_combinations(self, combinations: List[List[int]], 
                                  method: str = 'attention_weighted') -> Dict[str, Any]:
        """
        比較多個面向組合的相似度和差異
        
        Args:
            combinations: 要比較的面向組合列表，每個元素是一個面向ID列表
            method: 組合方法，同calculate_aspect_combination
            
        Returns:
            比較結果，包括相似度矩陣和分析
        """
        # 計算每個組合的向量表示
        combination_vectors = []
        combination_labels = []
        
        for aspect_ids in combinations:
            combo_info = self.calculate_aspect_combination(aspect_ids, method=method)
            combination_vectors.append(combo_info['vector'])
            combination_labels.append(combo_info['label'])
        
        # 計算相似度矩陣
        similarity_matrix = np.zeros((len(combinations), len(combinations)))
        for i in range(len(combinations)):
            for j in range(len(combinations)):
                similarity_matrix[i, j] = cosine_similarity(
                    combination_vectors[i].reshape(1, -1),
                    combination_vectors[j].reshape(1, -1)
                )[0, 0]
        
        # 分析結果
        most_similar_pair = None
        most_similar_score = -1
        least_similar_pair = None
        least_similar_score = 2  # 餘弦相似度範圍為[-1,1]
        
        for i in range(len(combinations)):
            for j in range(i+1, len(combinations)):
                sim_score = similarity_matrix[i, j]
                if sim_score > most_similar_score:
                    most_similar_score = sim_score
                    most_similar_pair = (i, j)
                if sim_score < least_similar_score:
                    least_similar_score = sim_score
                    least_similar_pair = (i, j)
        
        result = {
            'similarity_matrix': similarity_matrix,
            'combinations': combinations,
            'combination_labels': combination_labels,
            'most_similar': {
                'pair': most_similar_pair,
                'score': most_similar_score,
                'labels': [combination_labels[most_similar_pair[0]], combination_labels[most_similar_pair[1]]] if most_similar_pair else None
            },
            'least_similar': {
                'pair': least_similar_pair,
                'score': least_similar_score,
                'labels': [combination_labels[least_similar_pair[0]], combination_labels[least_similar_pair[1]]] if least_similar_pair else None
            }
        }
        
        return result
    
    def visualize_combinations_comparison(self, comparison_result: Dict[str, Any]) -> plt.Figure:
        """
        可視化面向組合的比較結果
        
        Args:
            comparison_result: 來自compare_aspect_combinations的結果
            
        Returns:
            matplotlib圖形對象
        """
        similarity_matrix = comparison_result['similarity_matrix']
        labels = comparison_result['combination_labels']
        
        # 創建熱力圖
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(similarity_matrix, cmap='viridis')
        
        # 添加標簽
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # 旋轉X軸標籤以避免重疊
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加顏色條
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("餘弦相似度", rotation=-90, va="bottom")
        
        # 在每個格子添加數值
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f"{similarity_matrix[i, j]:.2f}",
                               ha="center", va="center", color="w" if similarity_matrix[i, j] < 0.7 else "black")
        
        ax.set_title("面向組合相似度矩陣")
        fig.tight_layout()
        
        return fig
    
    def analyze_text_with_aspect_combinations(self, text: str, combinations: List[List[int]], 
                                           method: str = 'attention_weighted') -> Dict[str, Any]:
        """
        使用不同的面向組合分析文本
        
        Args:
            text: 輸入文本
            combinations: 要使用的面向組合列表，每個元素是一個面向ID列表
            method: 組合方法
            
        Returns:
            分析結果
        """
        if self.bert_embedder is None:
            self.initialize_components()
        
        # 生成文本嵌入
        embeddings = self.bert_embedder.generate_embeddings([text], show_progress=False)
        text_embedding = embeddings[0]
        
        # 計算每個面向組合的向量表示
        combination_vectors = []
        combination_labels = []
        
        for aspect_ids in combinations:
            combo_info = self.calculate_aspect_combination(aspect_ids, method=method)
            combination_vectors.append(combo_info['vector'])
            combination_labels.append(combo_info['label'])
        
        # 計算文本與每個組合的相似度
        similarities = {}
        for i, combo_label in enumerate(combination_labels):
            combo_vector = combination_vectors[i]
            similarity = cosine_similarity(
                text_embedding.reshape(1, -1),
                combo_vector.reshape(1, -1)
            )[0, 0]
            similarities[combo_label] = similarity
        
        # 找出最相關的面向組合
        most_relevant_label = max(similarities, key=similarities.get)
        most_relevant_score = similarities[most_relevant_label]
        most_relevant_index = combination_labels.index(most_relevant_label)
        most_relevant_aspects = combinations[most_relevant_index]
        
        result = {
            'text': text,
            'combinations': {label: {'similarity': similarities[label]} for label in combination_labels},
            'most_relevant': {
                'label': most_relevant_label,
                'aspects': most_relevant_aspects,
                'similarity': most_relevant_score
            }
        }
        
        return result

# 使用示例
if __name__ == "__main__":
    # 初始化面向計算器
    aspect_calc = AspectCalculator()
    aspect_calc.initialize_components()
    
    # 模擬數據
    texts = [
        "這家餐廳的食物非常美味，服務也很好",
        "菜品價格合理，但服務態度需要改進",
        "環境很舒適，但食物味道一般",
        "這家店的服務員態度很好，食物也不錯",
        "價格有點貴，但是食物品質很高",
        "餐廳環境優美，適合約會",
        "服務速度慢，但是食物很美味"
    ]
    
    ids = [f"text_{i}" for i in range(len(texts))]
    df = pd.DataFrame({
        'id': ids,
        'text': texts,
        'processed_text': texts
    })
    
    # 計算面向向量
    result = aspect_calc.calculate_aspect_vectors(df)
    
    # 顯示結果
    print("\n面向信息:")
    for aspect_id, info in aspect_calc.aspect_vectors.items():
        print(f"面向 {aspect_id+1} ({info['label']}): {', '.join(info['keywords'])}")
    
    # 測試單條文本分析
    test_text = "這家餐廳的菜品很美味，雖然價格有點高，但環境不錯"
    analysis = aspect_calc.analyze_text(test_text)
    
    print(f"\n文本: {test_text}")
    print(f"主要面向: {analysis['dominant_aspect']['label']} (相似度: {analysis['dominant_aspect']['similarity']:.2f})")
    print("面向分布:")
    for aspect_id, score in sorted(analysis['aspect_distribution'].items(), key=lambda x: x[1], reverse=True):
        label = aspect_calc.aspect_labels.get(aspect_id, f"面向{aspect_id+1}")
        print(f"  {label}: {score:.2f}")