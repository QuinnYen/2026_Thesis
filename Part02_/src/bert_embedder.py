"""
BERT嵌入提取器
這個腳本用於從文本中提取BERT嵌入。
"""

import torch
import numpy as np
import pandas as pd
import os
import logging
import traceback
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('bert_embedder')

class TextDataset(Dataset):
    """BERT處理用的文本數據集"""
    
    def __init__(self, texts, tokenizer, max_length=512):
        """
        初始化文本數據集
        
        Args:
            texts: 文本列表
            tokenizer: BERT分詞器
            max_length: 最大序列長度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 使用BERT分詞器處理文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 壓縮批次維度
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'text': text
        }

class BertEmbedder:
    """使用BERT模型提取文本的語義表示"""
    
    def __init__(self, model_name='bert-base-uncased', output_dir='./Part02_/results', device=None, logger=None, force_cpu=False):
        """
        初始化BERT編碼器
        
        Args:
            model_name: 使用的BERT模型名稱
            output_dir: 輸出目錄
            device: 計算設備，如果為None，則自動選擇
            logger: 日誌器
            force_cpu: 是否強制使用CPU即使有GPU可用
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        
        # 確保輸出目錄存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 檢查CUDA可用性
        self.cuda_available = torch.cuda.is_available()
        
        # 設置設備 - 自動檢測並選擇
        if device is None:
            if self.cuda_available and not force_cpu:
                self.device = torch.device('cuda')
                self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
                try:
                    mem_info = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                    self.logger.info(f"GPU記憶體: {mem_info:.2f} GB")
                except Exception as e:
                    self.logger.warning(f"無法獲取GPU記憶體資訊: {str(e)}")
            else:
                self.device = torch.device('cpu')
                if force_cpu:
                    self.logger.info("根據設定強制使用CPU進行處理")
                elif not self.cuda_available:
                    self.logger.info("找不到可用的GPU，將使用CPU進行處理")
                    
                    # 提供診斷信息
                    self.logger.info("CUDA診斷信息:")
                    self.logger.info(f"- PyTorch版本: {torch.__version__}")
                    self.logger.info(f"- CUDA版本 (PyTorch): {torch.version.cuda}")
                    
                    # 嘗試獲取系統CUDA信息
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                        if result.returncode == 0:
                            self.logger.info("系統已安裝NVIDIA驅動程序，但PyTorch無法使用CUDA")
                            self.logger.info("可能需要重新安裝支援CUDA的PyTorch版本")
                        else:
                            self.logger.info("系統未檢測到NVIDIA驅動程序")
                    except:
                        self.logger.info("無法執行nvidia-smi檢查")
        else:
            self.device = device
        
        self.logger.info(f"Using device: {self.device}")
        
        # 加載分詞器和模型
        try:
            self.logger.info(f"Loading BERT model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # 設置為評估模式
            self.logger.info(f"BERT model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading BERT model: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def log(self, message, level=logging.INFO):
        """統一的日誌處理方法"""
        if self.logger:
            self.logger.log(level, message)
    
    def extract_embeddings(self, data_path, text_column='clean_text', batch_size=16, callback=None):
        """
        從文本中提取BERT嵌入
        
        Args:
            data_path: 數據文件路徑（CSV）
            text_column: 文本列名
            batch_size: 批處理大小
            callback: 進度回調函數
            
        Returns:
            embeddings_path: 嵌入向量保存路徑
        """
        try:
            self.log(f"Starting to process file: {data_path}")
            self.log(f"Using model: {self.model_name}")
            
            # 讀取數據
            if callback:
                callback("Loading data...", 10)
            
            self.logger.info(f"Reading data from: {data_path}")
            df = pd.read_csv(data_path)
            
            if text_column not in df.columns:
                # 嘗試找到文本列
                if 'clean_text' in df.columns:
                    text_column = 'clean_text'
                elif 'text' in df.columns:
                    text_column = 'text'
                else:
                    text_column = df.columns[0]
                self.logger.warning(f"Specified text column '{text_column}' not found, using '{text_column}' instead")
            
            texts = df[text_column].tolist()
            self.logger.info(f"Loaded {len(texts)} text entries")
            
            # 創建數據集和數據加載器
            if callback:
                callback("Preparing BERT processing...", 20)
            
            dataset = TextDataset(
                texts=texts,
                tokenizer=self.tokenizer,
                max_length=512
            )
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            # 提取BERT嵌入
            if callback:
                callback("Extracting BERT embeddings...", 30)
            
            embeddings = []
            original_texts = []
            
            # 使用tqdm顯示進度
            progress_bar = tqdm(dataloader, desc="Extracting BERT embeddings")
            for i, batch in enumerate(progress_bar):
                # 更新進度回調
                if callback and i % 10 == 0:  # 每10批更新一次UI
                    progress = 30 + (i / len(dataloader) * 60)
                    callback(f"Processing batch {i+1}/{len(dataloader)}...", progress)
                
                # 將張量移到設備上
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 不計算梯度
                with torch.no_grad():
                    # 獲取BERT輸出
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    # 獲取[CLS]標記的表示（最後一層）
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    # 將嵌入添加到列表中
                    embeddings.append(cls_embeddings)
                    original_texts.extend(batch['text'])
            
            # 將嵌入連接成一個數組
            embeddings = np.vstack(embeddings)
            self.logger.info(f"Generated {embeddings.shape[0]} embedding vectors with dimension {embeddings.shape[1]}")
            
            # 將嵌入與原始文本和ID合併
            if callback:
                callback("Preparing to save results...", 90)
            
            # 創建一個新的DataFrame來保存嵌入和文本
            embeddings_df = pd.DataFrame({
                'text': original_texts
            })
            
            # 添加所有原始列（除了嵌入）
            for col in df.columns:
                if col != text_column:
                    embeddings_df[col] = df[col].values
            
            # 生成輸出文件路徑
            base_name = os.path.basename(data_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            embeddings_path = os.path.join(self.output_dir, f"{base_name_without_ext}_bert_embeddings.npz")
            metadata_path = os.path.join(self.output_dir, f"{base_name_without_ext}_bert_metadata.csv")
            
            # 保存嵌入向量（使用NumPy的壓縮格式）
            np.savez_compressed(embeddings_path, embeddings=embeddings)
            
            # 保存元數據（包含原始文本和其他列）
            embeddings_df.to_csv(metadata_path, index=False)
            
            self.logger.info(f"Embeddings saved to: {embeddings_path}")
            self.logger.info(f"Metadata saved to: {metadata_path}")
            
            if callback:
                callback("BERT embedding extraction complete", 100)
            
            return {
                'embeddings_path': embeddings_path,
                'metadata_path': metadata_path,
                'embedding_dim': embeddings.shape[1]
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting BERT embeddings: {str(e)}")
            self.logger.error(traceback.format_exc())
            if callback:
                callback(f"Error: {str(e)}", -1)
            raise

    def mean_pooling(self, model_output, attention_mask):
        """
        對模型輸出進行平均池化
        
        Args:
            model_output: 模型輸出
            attention_mask: 注意力遮罩
            
        Returns:
            sentence_embeddings: 句子嵌入
        """
        # 首先抽取最後一個隱藏狀態
        token_embeddings = model_output.last_hidden_state
        
        # 擴展注意力遮罩
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 對所有標記進行平均池化
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

# 使用示例
if __name__ == "__main__":
    # 初始化BERT編碼器
    embedder = BertEmbedder()
    
    # 從CSV文件中提取BERT嵌入
    result = embedder.extract_embeddings(
        "data_processed/processed_reviews.csv",
        text_column="clean_text"
    )
    
    print(f"Embeddings saved to: {result['embeddings_path']}")
    print(f"Embedding dimension: {result['embedding_dim']}")