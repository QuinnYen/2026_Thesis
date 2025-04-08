"""
面向向量計算器
"""

import numpy as np
import pandas as pd
import os
import logging
import traceback
import json
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from src.settings.visualization_config import apply_chinese_to_plot, check_chinese_display
import seaborn as sns

import matplotlib.pyplot as plt
plt.ioff()

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('aspect_vector_calculator')

class AspectVectorCalculator:
    """計算面向相關句子的平均向量"""
    
    def __init__(self, output_dir='./Part02_/results', logger=None):
        """
        初始化面向向量計算器
        
        Args:
            output_dir: 輸出目錄
            logger: 日誌器
        """
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 創建可視化子目錄
        self.vis_dir = os.path.join(output_dir, 'visualizations')
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
    
    def log(self, message, level=logging.INFO):
        """統一的日誌處理方法"""
        if self.logger:
            self.logger.log(level, message)
    
    def calculate_aspect_vectors(self, embeddings_path, topic_metadata_path, topic_keywords_path=None, attention_params=None, callback=None):
        """
        計算面向向量 - 支援多種注意力機制
        
        Args:
            embeddings_path: BERT嵌入向量路徑
            topic_metadata_path: 主題元數據路徑
            topic_keywords_path: 主題關鍵詞路徑（可選）
            attention_params: 注意力機制參數
            callback: 進度回調函數
        """
        try:
            # 加載BERT嵌入向量
            embeddings_data = np.load(embeddings_path)
            embeddings = embeddings_data['embeddings']
            
            # 加載主題元數據
            df = pd.read_csv(topic_metadata_path)
            
            # 加載主題關鍵詞（如果提供）
            topic_keywords = None
            if topic_keywords_path and os.path.exists(topic_keywords_path):
                with open(topic_keywords_path, 'r', encoding='utf-8') as f:
                    topic_keywords = json.load(f)
                    
            # 決定使用哪種注意力機制
            attention_type = "none"
            attention_weights = {}
            
            if attention_params:
                attention_type = attention_params.get("type", "無")
                attention_weights = attention_params.get("weights", {})
            
            # 初始化相應的注意力機制
            from src.attention_mechanism import (
                NoAttention, SimilarityAttention, 
                KeywordGuidedAttention, SelfAttention, CombinedAttention
            )
            
            if attention_type == "相似度注意力":
                attention_mech = SimilarityAttention(self.logger)
            elif attention_type == "關鍵詞注意力":
                attention_mech = KeywordGuidedAttention(self.logger)
            elif attention_type == "自注意力":
                attention_mech = SelfAttention(self.logger)
            elif attention_type == "組合注意力":
                attention_mech = CombinedAttention(self.logger)
            else:
                attention_mech = NoAttention(self.logger)
            
            # 計算面向向量
            if callback:
                callback("計算面向向量...", 50)
            
            aspect_vectors, attention_data = attention_mech.compute_aspect_vectors(
                embeddings, df, 
                topic_keywords=topic_keywords,
                weights=attention_weights
            )
            
            # 評估注意力機制效能
            if callback:
                callback("評估注意力機制效能...", 70)
            
            metrics = attention_mech.evaluate(aspect_vectors, embeddings, df)
            
            # 保存面向向量和元數據
            if callback:
                callback("保存結果...", 80)
            
            # 生成輸出文件名
            base_name = os.path.basename(topic_metadata_path)
            base_name_without_ext = os.path.splitext(base_name)[0].replace('_with_topics', '')
            
            # 添加注意力類型到文件名
            attention_suffix = "_" + attention_type.replace(" ", "_")
            output_base_name = base_name_without_ext + attention_suffix
            
            # 保存面向向量
            aspect_vectors_path = os.path.join(self.output_dir, f"{output_base_name}_aspect_vectors.npz")
            
            # 轉換為NumPy數組
            aspect_vectors_array = np.zeros((len(aspect_vectors), embeddings.shape[1]))
            topics = list(aspect_vectors.keys())
            
            for i, topic in enumerate(topics):
                aspect_vectors_array[i] = aspect_vectors[topic]
            
            np.savez_compressed(
                aspect_vectors_path, 
                aspect_vectors=aspect_vectors_array, 
                topics=topics,
                attention_weights=attention_data["weights"]
            )
            
            # 保存元數據
            aspect_metadata = {
                'topics': topics,
                'attention_type': attention_type,
                'attention_params': attention_weights,
                'metrics': metrics,
                'embedding_dim': embeddings.shape[1]
            }
            
            aspect_metadata_path = os.path.join(self.output_dir, f"{output_base_name}_aspect_metadata.json")
            with open(aspect_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(aspect_metadata, f, ensure_ascii=False, indent=2)
            
            # 返回結果
            result = {
                'aspect_vectors_path': aspect_vectors_path,
                'aspect_metadata_path': aspect_metadata_path,
                'attention_type': attention_type,
                'metrics': metrics,
                'topics': topics
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"錯誤: {str(e)}")
            raise
    
    def _visualize_aspect_vectors(self, embeddings, df, metadata_path):
        """
        使用t-SNE降維並可視化面向向量 - 修復版
        """
        self.log("Visualizing aspect vectors using t-SNE")

        # 如果數據太大，隨機選擇一部分進行可視化
        sample_size = min(5000, len(embeddings))
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
            sample_topics = df['main_topic'].iloc[indices].values
        else:
            sample_embeddings = embeddings
            sample_topics = df['main_topic'].values
        
        # 檢查樣本數量，如果太少則跳過t-SNE
        if len(sample_embeddings) < 5:
            self.log(f"Too few samples ({len(sample_embeddings)}), skipping t-SNE visualization", level=logging.WARNING)
            
            # 生成一個簡單的條形圖替代
            base_name = os.path.basename(metadata_path)
            base_name_without_ext = os.path.splitext(base_name)[0].replace('_with_topics', '')
            tsne_plot_path = os.path.join(self.vis_dir, f"{base_name_without_ext}_topic_counts.png")
            
            # 統計每個主題的文檔數量
            topic_counts = df['main_topic'].value_counts()
            plt.figure(figsize=(10, 6))
            topic_counts.plot(kind='bar')
            plt.xlabel('主題')
            plt.ylabel('文檔數量')
            plt.title('各主題文檔數量統計')
            plt.tight_layout()
            plt.savefig(tsne_plot_path, dpi=300)
            plt.close('all')  # 關閉所有圖表
            
            return {
                'tsne_plot_path': tsne_plot_path
            }
        
        # 動態調整perplexity，確保它小於樣本數量
        perplexity = min(30, len(sample_embeddings) - 1)
        self.log(f"Running t-SNE on {len(sample_embeddings)} embeddings with perplexity={perplexity}")
        
        # 使用t-SNE降維到2D
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=perplexity, 
            max_iter=1000
        )
        tsne_results = tsne.fit_transform(sample_embeddings)
        
        # 繪製t-SNE結果
        plt.figure(figsize=(12, 10))
        
        # 將主題標籤轉換為數字
        topic_labels = np.unique(sample_topics)
        topic_to_id = {topic: i for i, topic in enumerate(topic_labels)}
        numeric_topics = np.array([topic_to_id[topic] for topic in sample_topics])
        
        # 使用顏色區分不同主題
        scatter = plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1], 
            c=numeric_topics, 
            cmap='tab20', 
            alpha=0.6, 
            s=10
        )
        
        # 添加標題和標籤
        plt.title('文檔嵌入向量的t-SNE視覺化（按主題著色）')
        plt.xlabel('t-SNE維度 1')
        plt.ylabel('t-SNE維度 2')
        
        # 修復圖例 - 將topic_labels轉換為列表以避免NumPy數組的布爾操作問題
        legend_elements = scatter.legend_elements()[0]
        topic_labels_list = list(topic_labels)  # 轉換為Python列表
        
        try:
            # 修復版本的圖例創建
            if len(legend_elements) > 0 and len(topic_labels_list) > 0:
                legend1 = plt.legend(
                    handles=legend_elements,
                    labels=topic_labels_list,
                    title="主題",
                    loc="upper right"
                )
                plt.gca().add_artist(legend1)
        except Exception as e:
            self.log(f"Warning: Failed to create legend: {str(e)}", level=logging.WARNING)
        
        plt.tight_layout()
        
        # 保存圖形
        base_name = os.path.basename(metadata_path)
        base_name_without_ext = os.path.splitext(base_name)[0].replace('_with_topics', '')
        tsne_plot_path = os.path.join(self.vis_dir, f"{base_name_without_ext}_tsne.png")
        plt.savefig(tsne_plot_path, dpi=300, bbox_inches='tight')
        plt.close('all')  # 關閉所有圖表
        
        self.log(f"t-SNE visualization saved to: {tsne_plot_path}")
        
        return {
            'tsne_plot_path': tsne_plot_path
        }
    
    def export_aspect_vectors(self, aspect_vectors_path, output_format='csv', callback=None):
        """
        導出面向向量為不同格式
        
        Args:
            aspect_vectors_path: 面向向量文件路徑
            output_format: 輸出格式，可以是'csv', 'json', 或 'pickle'
            callback: 進度回調函數
            
        Returns:
            str: 導出文件的路徑
        """
        try:
            self.log(f"Exporting aspect vectors from: {aspect_vectors_path}")
            self.log(f"Output format: {output_format}")
            
            # 讀取面向向量
            if callback:
                callback("Loading aspect vectors...", 20)
            
            # 使用 allow_pickle=True 參數讀取包含 Python 對象的 NumPy 檔案
            aspect_data = np.load(aspect_vectors_path, allow_pickle=True)
            aspect_vectors = aspect_data['aspect_vectors']
            topics = aspect_data['topics']
            
            # 準備輸出文件名
            base_name = os.path.basename(aspect_vectors_path)
            base_name_without_ext = os.path.splitext(base_name)[0].replace('_aspect_vectors', '')
            
            if callback:
                callback("Preparing export data...", 50)
            
            if output_format == 'csv':
                # 轉換為DataFrame並導出為CSV
                df = pd.DataFrame(aspect_vectors)
                df['topic'] = topics
                output_path = os.path.join(self.output_dir, f"{base_name_without_ext}_aspect_vectors.csv")
                df.to_csv(output_path, index=False)
            
            elif output_format == 'json':
                # 構建JSON對象並導出
                json_data = {}
                for i, topic in enumerate(topics):
                    json_data[str(topic)] = aspect_vectors[i].tolist()
                
                output_path = os.path.join(self.output_dir, f"{base_name_without_ext}_aspect_vectors.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            elif output_format == 'pickle':
                # 構建字典並導出為pickle
                pickle_data = {
                    'aspect_vectors': aspect_vectors,
                    'topics': topics
                }
                output_path = os.path.join(self.output_dir, f"{base_name_without_ext}_aspect_vectors.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(pickle_data, f)
            
            else:
                raise ValueError(f"不支持的輸出格式: {output_format}")
            
            if callback:
                callback(f"Export complete: {output_path}", 100)
            
            self.log(f"Aspect vectors exported to: {output_path}")
            return output_path
            
        except Exception as e:
            self.log(f"Error exporting aspect vectors: {str(e)}", level=logging.ERROR)
            self.log(traceback.format_exc(), level=logging.ERROR)
            if callback:
                callback(f"Error: {str(e)}", -1)
            raise

    def __del__(self):
        """確保在對象被銷毀時清理所有圖表資源"""
        plt.close('all')

# 使用示例
if __name__ == "__main__":
    calculator = AspectVectorCalculator()
    
    def print_progress(message, percentage):
        print(f"{message} ({percentage}%)")
    
    try:
        # 假設我們已經有了BERT嵌入和主題元數據
        embeddings_path = "./Part02_/results/processed_reviews_bert_embeddings.npz"
        topic_metadata_path = "./Part02_/results/processed_reviews_with_topics.csv"
        
        # 計算面向向量
        results = calculator.calculate_aspect_vectors(
            embeddings_path,
            topic_metadata_path,
            callback=print_progress
        )
        
        print(f"Aspect vectors saved to: {results['aspect_vectors_path']}")
        print(f"Aspect metadata saved to: {results['aspect_metadata_path']}")
        print(f"t-SNE visualization saved to: {results['tsne_plot_path']}")
        
        # 導出為不同格式
        csv_path = calculator.export_aspect_vectors(
            results['aspect_vectors_path'],
            output_format='csv',
            callback=print_progress
        )
        print(f"Exported to CSV: {csv_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()