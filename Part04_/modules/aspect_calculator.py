"""
面向向量計算模組 - 負責計算面向相關句子的平均向量
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import json
import pickle
import logging
import time

# 導入系統模組
from utils.logger import get_logger

# 導入系統自定義模組
from modules.attention_mechanism import create_attention_mechanism, apply_attention_mechanism

# 獲取logger
logger = get_logger("aspect_calculator")

class AspectCalculator:
    """面向向量計算器 - 使用多種注意力機制計算面向向量"""
    
    def __init__(self, config=None):
        """初始化面向向量計算器
        
        Args:
            config: 配置參數字典，可包含以下鍵:
                - output_dir: 輸出目錄
                - attention_types: 要使用的注意力機制類型列表
                - attention_weights: 組合注意力的權重
                - evaluation_metrics: 需計算的評估指標
        """
        self.config = config or {}
        self.logger = logger
        
        # 設置默認配置
        self.output_dir = self.config.get('output_dir', './output/vectors')
        self.attention_types = self.config.get('attention_types', ['no', 'similarity', 'keyword', 'self', 'combined'])
        self.attention_weights = self.config.get('attention_weights', {
            'similarity': 0.33,
            'keyword': 0.33,
            'self': 0.34
        })
        
        # 確保輸出目錄存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 創建可視化目錄
        self.vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def calculate_aspect_vectors(self, embeddings_path, metadata_path, topics_path=None, progress_callback=None):
        """計算面向向量
        
        Args:
            embeddings_path: BERT嵌入向量文件路徑
            metadata_path: 包含主題標籤的元數據文件路徑
            topics_path: 主題詞文件路徑，用於關鍵詞注意力
            progress_callback: 進度回調函數
            
        Returns:
            dict: 包含結果的字典
        """
        try:
            start_time = time.time()
            self.logger.info(f"開始計算面向向量")
            self.logger.info(f"使用嵌入向量: {embeddings_path}")
            self.logger.info(f"使用元數據: {metadata_path}")
            if topics_path:
                self.logger.info(f"使用主題詞: {topics_path}")
            
            # 加載嵌入向量
            self.logger.info(f"正在加載嵌入向量...")
            if progress_callback:
                progress_callback("加載嵌入向量...", 10)
                
            try:
                embeddings_data = np.load(embeddings_path)
                embeddings = embeddings_data['embeddings']
                self.logger.info(f"加載了 {embeddings.shape[0]} 個嵌入向量，維度為 {embeddings.shape[1]}")
            except Exception as e:
                self.logger.error(f"加載嵌入向量時出錯: {str(e)}")
                return None
            
            # 加載元數據
            self.logger.info(f"正在加載元數據...")
            if progress_callback:
                progress_callback("加載元數據...", 20)
                
            try:
                metadata = pd.read_csv(metadata_path)
                self.logger.info(f"加載了 {len(metadata)} 條元數據記錄")
                
                # 檢查主題列
                if 'main_topic' not in metadata.columns:
                    self.logger.error(f"元數據中找不到 'main_topic' 列")
                    return None
                    
                # 檢查主題數
                topics = metadata['main_topic'].unique()
                self.logger.info(f"識別出 {len(topics)} 個主題")
            except Exception as e:
                self.logger.error(f"加載元數據時出錯: {str(e)}")
                return None
            
            # 獲取基礎文件名
            base_name = os.path.basename(metadata_path)
            base_name_without_ext = os.path.splitext(base_name)[0].replace('_with_topics', '')
            
            # 為每種注意力機制計算面向向量
            all_results = {}
            attention_results = {}
            
            # 計算總進度
            total_attention_types = len(self.attention_types)
            
            for i, attention_type in enumerate(self.attention_types):
                # 更新進度
                progress_start = 20 + i * (60 / total_attention_types)
                progress_end = 20 + (i + 1) * (60 / total_attention_types)
                
                if progress_callback:
                    progress_callback(f"計算 {attention_type} 注意力機制的面向向量...", progress_start)
                
                self.logger.info(f"正在計算 {attention_type} 注意力機制的面向向量...")
                
                # 設置權重（僅對組合注意力有效）
                weights = self.attention_weights if attention_type == 'combined' else None
                
                # 應用注意力機制
                result = apply_attention_mechanism(
                    attention_type, embeddings, metadata, 
                    topics_path=topics_path, weights=weights
                )
                
                # 保存結果
                aspect_vectors = result['aspect_vectors']
                metrics = result['metrics']
                
                self.logger.info(f"{attention_type} 注意力機制評估指標:")
                self.logger.info(f"  內聚度: {metrics['coherence']:.4f}")
                self.logger.info(f"  分離度: {metrics['separation']:.4f}")
                self.logger.info(f"  綜合得分: {metrics['combined_score']:.4f}")
                
                # 保存到結果集
                attention_results[attention_type] = {
                    'aspect_vectors': aspect_vectors,
                    'metrics': metrics
                }
                
                # 保存面向向量
                output_path = os.path.join(
                    self.output_dir, 
                    f"{base_name_without_ext}_{attention_type}_aspect_vectors.npz"
                )
                
                # 轉換為NumPy數組以便保存
                aspect_vectors_array = np.zeros((len(aspect_vectors), embeddings.shape[1]))
                aspect_topics = list(aspect_vectors.keys())
                
                for j, topic in enumerate(aspect_topics):
                    aspect_vectors_array[j] = aspect_vectors[topic]
                
                np.savez_compressed(
                    output_path, 
                    aspect_vectors=aspect_vectors_array, 
                    topics=aspect_topics
                )
                
                # 保存元數據
                metadata_output = {
                    'attention_type': attention_type,
                    'metrics': metrics,
                    'topics': aspect_topics
                }
                
                metadata_path = os.path.join(
                    self.output_dir, 
                    f"{base_name_without_ext}_{attention_type}_aspect_metadata.json"
                )
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata_output, f, ensure_ascii=False, indent=2)
                
                all_results[attention_type] = {
                    'vectors_path': output_path,
                    'metadata_path': metadata_path,
                    'metrics': metrics
                }
                
                # 創建可視化
                if progress_callback:
                    progress_callback(f"為 {attention_type} 注意力機制創建可視化...", progress_end - 5)
                
                # 可視化: t-SNE降維
                vis_path = self._visualize_tsne(
                    embeddings, metadata, aspect_vectors,
                    f"{base_name_without_ext}_{attention_type}"
                )
                
                if vis_path:
                    all_results[attention_type]['visualization'] = vis_path
            
            # 創建比較可視化
            if progress_callback:
                progress_callback("創建注意力機制比較可視化...", 80)
                
            # 比較不同注意力機制的評估指標
            comparison_path = self._create_comparison_visualization(
                attention_results, f"{base_name_without_ext}_comparison"
            )
            
            if comparison_path:
                all_results['comparison'] = {
                    'visualization': comparison_path
                }
            
            # 找到最佳注意力機制
            best_attention = self._find_best_attention(attention_results)
            
            # 計算完成時間
            elapsed_time = time.time() - start_time
            self.logger.info(f"面向向量計算完成，共 {len(self.attention_types)} 種注意力機制")
            self.logger.info(f"最佳注意力機制: {best_attention['type']}，綜合得分: {best_attention['score']:.4f}")
            self.logger.info(f"處理耗時: {elapsed_time:.2f} 秒")
            
            # 完成並返回結果
            if progress_callback:
                progress_callback("面向向量計算完成", 100)
                
            return {
                'all_results': all_results,
                'best_attention': best_attention,
                'comparison_visualization': comparison_path
            }
            
        except Exception as e:
            self.logger.error(f"計算面向向量時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _find_best_attention(self, attention_results):
        """找到最佳注意力機制
        
        Args:
            attention_results: 注意力機制結果字典
            
        Returns:
            dict: 最佳注意力機制信息
        """
        best_score = -float('inf')
        best_type = None
        
        for attention_type, result in attention_results.items():
            metrics = result['metrics']
            score = metrics['combined_score']
            
            if score > best_score:
                best_score = score
                best_type = attention_type
        
        # 返回最佳機制信息
        return {
            'type': best_type,
            'score': best_score,
            'metrics': attention_results[best_type]['metrics']
        }
    
    def _visualize_tsne(self, embeddings, metadata, aspect_vectors, output_prefix):
        """使用t-SNE可視化文檔嵌入和面向向量
        
        Args:
            embeddings: 嵌入向量
            metadata: 元數據
            aspect_vectors: 面向向量字典
            output_prefix: 輸出文件前綴
            
        Returns:
            str: 可視化圖片路徑
        """
        try:
            # 抽樣文檔（如果太多）
            max_docs = 5000
            if len(embeddings) > max_docs:
                sample_indices = np.random.choice(len(embeddings), max_docs, replace=False)
                sampled_embeddings = embeddings[sample_indices]
                sampled_topics = metadata['main_topic'].iloc[sample_indices].values
            else:
                sampled_embeddings = embeddings
                sampled_topics = metadata['main_topic'].values
            
            # 獲取面向向量列表
            aspect_topics = list(aspect_vectors.keys())
            aspect_vecs = np.array([aspect_vectors[topic] for topic in aspect_topics])
            
            # 合併文檔嵌入和面向向量
            combined_embeddings = np.vstack([sampled_embeddings, aspect_vecs])
            
            # 創建標籤
            doc_labels = [f'Doc: {topic}' for topic in sampled_topics]
            aspect_labels = [f'Aspect: {topic}' for topic in aspect_topics]
            combined_labels = doc_labels + aspect_labels
            
            # 創建類型標記
            doc_types = ['document'] * len(sampled_embeddings)
            aspect_types = ['aspect'] * len(aspect_vecs)
            combined_types = doc_types + aspect_types
            
            # 使用t-SNE降維到2D
            perplexity = min(30, len(combined_embeddings) - 1)
            tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=perplexity,
                n_iter=1000
            )
            tsne_results = tsne.fit_transform(combined_embeddings)
            
            # 分離文檔和面向向量的結果
            doc_tsne = tsne_results[:len(sampled_embeddings)]
            aspect_tsne = tsne_results[len(sampled_embeddings):]
            
            # 為每個主題分配顏色
            unique_topics = sorted(list(set(sampled_topics)))
            topic_to_color = {topic: i for i, topic in enumerate(unique_topics)}
            
            # 創建顏色映射
            doc_colors = [topic_to_color[topic] for topic in sampled_topics]
            
            # 繪製t-SNE圖
            plt.figure(figsize=(12, 10))
            
            # 繪製文檔點
            scatter = plt.scatter(
                doc_tsne[:, 0], doc_tsne[:, 1],
                c=doc_colors,
                cmap='tab20',
                alpha=0.6,
                s=10
            )
            
            # 為面向向量添加大的標記
            for i, topic in enumerate(aspect_topics):
                color_idx = topic_to_color.get(topic, 0)
                color = plt.cm.tab20(color_idx)
                plt.scatter(
                    aspect_tsne[i, 0], aspect_tsne[i, 1],
                    c=[color],
                    s=200,
                    marker='*',
                    edgecolors='black',
                    label=f'Aspect: {topic}'
                )
                
                # 添加文本標籤
                plt.text(
                    aspect_tsne[i, 0], aspect_tsne[i, 1],
                    topic.split('_')[-1] if '_' in topic else topic,
                    fontsize=12,
                    ha='center',
                    va='bottom'
                )
            
            # 添加圖例
            legend1 = plt.legend(*scatter.legend_elements(),
                               loc="upper right", title="Topics")
            plt.gca().add_artist(legend1)
            
            # 添加標題和軸標籤
            plt.title('文檔嵌入和面向向量的t-SNE視覺化')
            plt.xlabel('t-SNE維度 1')
            plt.ylabel('t-SNE維度 2')
            
            plt.tight_layout()
            
            # 保存圖片
            output_path = os.path.join(self.vis_dir, f"{output_prefix}_tsne.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"創建t-SNE可視化時出錯: {str(e)}")
            return None
    
    def _create_comparison_visualization(self, attention_results, output_prefix):
        """創建注意力機制比較可視化
        
        Args:
            attention_results: 注意力機制結果字典
            output_prefix: 輸出文件前綴
            
        Returns:
            str: 可視化圖片路徑
        """
        try:
            # 提取評估指標
            attention_types = list(attention_results.keys())
            coherence_scores = [attention_results[t]['metrics']['coherence'] for t in attention_types]
            separation_scores = [attention_results[t]['metrics']['separation'] for t in attention_types]
            combined_scores = [attention_results[t]['metrics']['combined_score'] for t in attention_types]
            
            # 創建圖表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 繪製內聚度比較
            axes[0, 0].bar(attention_types, coherence_scores)
            axes[0, 0].set_title('內聚度比較 (越高越好)')
            axes[0, 0].set_xlabel('注意力機制')
            axes[0, 0].set_ylabel('內聚度得分')
            axes[0, 0].set_ylim(0, max(coherence_scores) * 1.2)
            
            # 添加數值標籤
            for i, score in enumerate(coherence_scores):
                axes[0, 0].text(i, score, f'{score:.4f}', ha='center', va='bottom')
            
            # 繪製分離度比較
            axes[0, 1].bar(attention_types, separation_scores)
            axes[0, 1].set_title('分離度比較 (越高越好)')
            axes[0, 1].set_xlabel('注意力機制')
            axes[0, 1].set_ylabel('分離度得分')
            axes[0, 1].set_ylim(0, max(separation_scores) * 1.2)
            
            # 添加數值標籤
            for i, score in enumerate(separation_scores):
                axes[0, 1].text(i, score, f'{score:.4f}', ha='center', va='bottom')
            
            # 繪製綜合得分比較
            axes[1, 0].bar(attention_types, combined_scores)
            axes[1, 0].set_title('綜合得分比較 (越高越好)')
            axes[1, 0].set_xlabel('注意力機制')
            axes[1, 0].set_ylabel('綜合得分')
            axes[1, 0].set_ylim(0, max(combined_scores) * 1.2)
            
            # 添加數值標籤
            for i, score in enumerate(combined_scores):
                axes[1, 0].text(i, score, f'{score:.4f}', ha='center', va='bottom')
            
            # 繪製雷達圖
            ax = axes[1, 1]
            
            # 準備雷達圖數據
            metrics = ['內聚度', '分離度', '綜合得分']
            num_metrics = len(metrics)
            
            # 計算角度
            angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
            angles += angles[:1]  # 閉合雷達圖
            
            # 初始化雷達圖
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            
            # 繪製每種注意力機制的雷達圖
            for i, attention_type in enumerate(attention_types):
                values = [
                    attention_results[attention_type]['metrics']['coherence'],
                    attention_results[attention_type]['metrics']['separation'],
                    attention_results[attention_type]['metrics']['combined_score']
                ]
                values += values[:1]  # 閉合雷達圖
                
                ax.plot(angles, values, linewidth=2, label=attention_type)
                ax.fill(angles, values, alpha=0.1)
            
            ax.set_title('注意力機制評估指標雷達圖')
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            
            # 保存圖片
            output_path = os.path.join(self.vis_dir, f"{output_prefix}_comparison.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            return output_path
        except Exception as e:
            self.logger.error(f"創建比較可視化時出錯: {str(e)}")
            return None
    
    def export_results(self, vectors_path, output_format='csv'):
        """導出面向向量為不同格式
        
        Args:
            vectors_path: 面向向量文件路徑
            output_format: 輸出格式，'csv', 'json'或'pickle'
            
        Returns:
            str: 導出文件路徑
        """
        try:
            self.logger.info(f"導出面向向量: {vectors_path}")
            
            # 載入面向向量
            vectors_data = np.load(vectors_path, allow_pickle=True)
            vectors = vectors_data['aspect_vectors']
            topics = list(vectors_data['topics'])
            
            # 生成輸出文件名
            base_name = os.path.basename(vectors_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            
            if output_format == 'csv':
                # 轉換為DataFrame並導出為CSV
                df = pd.DataFrame(vectors)
                df['topic'] = topics
                output_path = os.path.join(self.output_dir, f"{base_name_without_ext}.csv")
                df.to_csv(output_path, index=False)
                
            elif output_format == 'json':
                # 構建JSON對象並導出
                json_data = {}
                for i, topic in enumerate(topics):
                    json_data[topic] = vectors[i].tolist()
                
                output_path = os.path.join(self.output_dir, f"{base_name_without_ext}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
            elif output_format == 'pickle':
                # 構建字典並導出為pickle
                pickle_data = {
                    'aspect_vectors': vectors,
                    'topics': topics
                }
                output_path = os.path.join(self.output_dir, f"{base_name_without_ext}.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(pickle_data, f)
                
            else:
                self.logger.error(f"不支持的輸出格式: {output_format}")
                return None
                
            self.logger.info(f"成功導出到: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"導出面向向量時出錯: {str(e)}")
            return None


def calculate_aspect_vectors(embeddings_path, metadata_path, topics_path=None, 
                          attention_types=None, output_dir='./output/vectors', progress_callback=None):
    """計算面向向量的便捷函數
    
    Args:
        embeddings_path: BERT嵌入向量文件路徑
        metadata_path: 包含主題標籤的元數據文件路徑
        topics_path: 主題詞文件路徑，用於關鍵詞注意力
        attention_types: 要使用的注意力機制列表，如果為None則使用所有
        output_dir: 輸出目錄
        progress_callback: 進度回調函數
        
    Returns:
        dict: 包含結果的字典
    """
    # 使用默認注意力機制列表
    if attention_types is None:
        attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
    
    # 創建配置
    config = {
        'output_dir': output_dir,
        'attention_types': attention_types
    }
    
    # 創建計算器
    calculator = AspectCalculator(config)
    
    # 計算面向向量
    return calculator.calculate_aspect_vectors(
        embeddings_path, metadata_path, topics_path, progress_callback
    )


# 測試代碼
if __name__ == "__main__":
    # 配置日誌級別
    logger.setLevel(logging.INFO)
    
    # 測試面向向量計算
    embeddings_path = "path/to/your/bert_embeddings.npz"
    metadata_path = "path/to/your/with_topics.csv"
    topics_path = "path/to/your/topics.json"
    
    if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
        # 定義進度回調函數
        def progress_callback(message, percentage):
            print(f"{message} - {percentage}%")
        
        # 計算面向向量
        result = calculate_aspect_vectors(
            embeddings_path, metadata_path, topics_path,
            progress_callback=progress_callback
        )
        
        if result:
            print(f"最佳注意力機制: {result['best_attention']['type']}")
            print(f"綜合得分: {result['best_attention']['score']:.4f}")
        else:
            print("面向向量計算失敗")
    else:
        logger.warning(f"測試文件不存在")