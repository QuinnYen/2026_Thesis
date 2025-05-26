"""
BERT 嵌入模組 - 負責使用BERT模型生成文本嵌入
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import logging
import time
import random
from tqdm import tqdm  # 添加缺少的 tqdm 導入

# 固定所有隨機種子，確保結果可重現
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 導入系統日誌模組
from utils.logger import get_logger

# 獲取logger
logger = get_logger("bert_embedder")

class TextDataset(Dataset):
    """用於BERT處理的文本數據集類"""
    
    def __init__(self, texts, tokenizer, max_length=128):
        """初始化文本數據集
        
        Args:
            texts: 文本列表
            tokenizer: BERT標記器
            max_length: 最大序列長度
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 使用BERT標記器處理文本
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
    
    def __init__(self, config=None):
        """初始化BERT嵌入器
        
        Args:
            config: 配置字典，可包含以下鍵:
                - model_name: BERT模型名稱
                - batch_size: 批次大小
                - max_length: 最大序列長度
                - use_gpu: 是否使用GPU
                - output_dir: 輸出目錄
                - random_seed: 隨機種子
        """
        self.config = config or {}
        self.logger = logger
        
        # 設置默認配置
        self.model_name = self.config.get('model_name', 'bert-base-uncased')
        self.batch_size = self.config.get('batch_size', 32)
        self.max_length = self.config.get('max_length', 128)
        self.use_gpu = self.config.get('use_gpu', torch.cuda.is_available())
        self.output_dir = self.config.get('output_dir', os.path.join('Part04_', '1_output', 'embeddings'))
        self.random_seed = self.config.get('random_seed', 42)
        
        # 固定隨機種子，確保結果可重現
        import random
        import numpy as np
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 檢查CUDA可用性
        self.cuda_available = torch.cuda.is_available()
        
        # 設置設備
        if self.cuda_available and self.use_gpu:
            self.device = torch.device('cuda')
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
            try:
                mem_info = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                self.logger.info(f"GPU記憶體: {mem_info:.2f} GB")
            except Exception as e:
                self.logger.warning(f"無法獲取GPU記憶體資訊: {str(e)}")
        else:
            self.device = torch.device('cpu')
            if self.use_gpu and not self.cuda_available:
                self.logger.warning("找不到可用的GPU，將使用CPU進行處理")
            else:
                self.logger.info("使用CPU進行處理")
        
        # 延遲創建輸出目錄，直到實際需要時才創建
        # os.makedirs(self.output_dir, exist_ok=True)
    
    def update_config(self, new_config):
        """更新BERT編碼器配置
        
        Args:
            new_config: 新的配置參數字典
        """
        if not new_config:
            return
            
        self.logger.info("更新BERT編碼器配置")
        
        # 更新配置字典
        if isinstance(new_config, dict):
            self.config.update(new_config)
            
            # 更新類屬性
            self.model_name = self.config.get('model_name', self.model_name)
            self.max_length = self.config.get('max_length', self.max_length)
            self.batch_size = self.config.get('batch_size', self.batch_size)
            self.output_dir = self.config.get('output_dir', self.output_dir)
            
            # 延遲創建輸出目錄，直到實際需要時才創建
            # os.makedirs(self.output_dir, exist_ok=True)
            
            # 如果模型名稱變更，則需要重新加載模型
            if hasattr(self, 'model') and self.model_name != new_config.get('model_name'):
                self.logger.info(f"模型名稱已變更，需要重新加載模型: {new_config.get('model_name')}")
                delattr(self, 'model')
                delattr(self, 'tokenizer')
            
            # 重新設置設備（如果use_gpu設置變更）
            use_gpu = self.config.get('use_gpu', True)
            if self.cuda_available and use_gpu:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                
            self.logger.debug(f"BERT編碼器配置更新完成")
        else:
            self.logger.warning(f"無效的配置格式: {type(new_config)}")
    
    def load_model(self):
        """加載BERT模型和標記器"""
        try:
            self.logger.info(f"正在加載BERT模型: {self.model_name}")
            
            # 加載標記器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 加載模型
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # 設置為評估模式
            
            self.logger.info("BERT模型加載成功")
            return True
        except Exception as e:
            self.logger.error(f"加載BERT模型失敗: {str(e)}")
            return False
    
    def get_embeddings(self, text):
        """為單個文本生成BERT嵌入向量
        
        Args:
            text: 要進行嵌入的文本字符串
            
        Returns:
            numpy.ndarray: 文本的嵌入向量
        """
        try:
            # 加載模型（如果尚未加載）
            if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
                if not self.load_model():
                    return None
                    
            # 轉換為列表形式以便批處理
            text = str(text)
            
            # 使用BERT標記器處理文本
            encoded_input = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            # 將輸入移到設備上
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
            
            # 不計算梯度
            with torch.no_grad():
                # 獲取BERT輸出
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 獲取[CLS]標記的表示（最後一層）作為整個句子的表示
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                
            return cls_embedding
            
        except Exception as e:
            self.logger.error(f"生成BERT嵌入向量時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def extract_embeddings(self, data_path, text_column='clean_text', progress_callback=None):
        """從處理後的數據中提取BERT嵌入向量
        
        Args:
            data_path: 處理後的數據文件路徑
            text_column: 文本列名
            progress_callback: 進度回調函數
            
        Returns:
            dict: 包含嵌入向量文件路徑和元數據文件路徑的字典
        """
        try:
            start_time = time.time()
            self.logger.info(f"開始從 {data_path} 提取BERT嵌入向量")
            
            # 加載模型（如果尚未加載）
            if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
                if not self.load_model():
                    return None
            
            # 加載數據
            self.logger.info(f"正在加載數據...")
            if progress_callback:
                progress_callback("加載數據文件...", 10)
                
            df = pd.read_csv(data_path)
            
            # 檢查文本列是否存在
            if text_column not in df.columns:
                # 嘗試找到文本列
                if 'clean_text' in df.columns:
                    text_column = 'clean_text'
                elif 'tokens_str' in df.columns:
                    text_column = 'tokens_str'
                elif 'text' in df.columns:
                    text_column = 'text'
                else:
                    self.logger.error(f"找不到文本列 '{text_column}'，可用列: {', '.join(df.columns)}")
                    return None
                    
                self.logger.warning(f"指定的文本列 '{text_column}' 不存在，使用 '{text_column}' 代替")
            
            # 獲取文本列表
            texts = df[text_column].fillna('').tolist()
            self.logger.info(f"共 {len(texts)} 條文本")
            
            # 創建數據集和數據加載器
            self.logger.info(f"正在準備數據...")
            if progress_callback:
                progress_callback("準備BERT處理...", 20)
                
            dataset = TextDataset(
                texts=texts,
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
            
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False
            )
            
            # 提取BERT嵌入
            self.logger.info(f"正在提取BERT嵌入向量...")
            if progress_callback:
                progress_callback("提取BERT嵌入向量...", 30)
                
            embeddings = []
            texts_processed = []
            
            # 使用tqdm顯示進度條
            progress_bar = tqdm(dataloader, desc="提取BERT嵌入向量", disable=None)
            for i, batch in enumerate(progress_bar):
                # 更新進度
                if progress_callback and i % 5 == 0:  # 每5批更新一次UI
                    progress = 30 + int((i / len(dataloader)) * 60)
                    progress_callback(f"處理第 {i+1}/{len(dataloader)} 批...", progress)
                
                # 將張量移動到設備上
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
                    
                    # 將嵌入和文本添加到列表中
                    embeddings.append(cls_embeddings)
                    texts_processed.extend(batch['text'])
            
            # 將嵌入連接成一個數組
            embeddings = np.vstack(embeddings)
            self.logger.info(f"生成了 {embeddings.shape[0]} 個維度為 {embeddings.shape[1]} 的嵌入向量")
            
            # 準備保存結果
            if progress_callback:
                progress_callback("準備保存結果...", 90)
                
            # 確保輸出目錄存在
            os.makedirs(self.output_dir, exist_ok=True)
                
            # 生成輸出文件路徑
            base_name = os.path.basename(data_path)
            base_name_without_ext = os.path.splitext(base_name)[0]
            
            # 創建一個新的DataFrame來保存嵌入相關的元數據
            metadata_df = pd.DataFrame({'text': texts_processed})
            
            # 添加原始DataFrame的所有列（除了文本列）
            for col in df.columns:
                if col != text_column:
                    metadata_df[col] = df[col].values
            
            # 生成保存路徑
            embeddings_path = os.path.join(self.output_dir, f"{base_name_without_ext}_bert_embeddings.npz")
            metadata_path = os.path.join(self.output_dir, f"{base_name_without_ext}_bert_metadata.csv")
            
            # 保存嵌入向量（使用NumPy的壓縮格式）
            np.savez_compressed(embeddings_path, embeddings=embeddings)
            
            # 保存元數據（包含原始文本和其他列）
            metadata_df.to_csv(metadata_path, index=False)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"BERT嵌入向量提取完成，耗時: {elapsed_time:.2f} 秒")
            self.logger.info(f"嵌入向量已保存至: {embeddings_path}")
            self.logger.info(f"元數據已保存至: {metadata_path}")
            
            if progress_callback:
                progress_callback("BERT嵌入向量提取完成", 100)
            
            # 返回結果
            return {
                'embeddings_path': embeddings_path,
                'metadata_path': metadata_path,
                'embedding_dim': embeddings.shape[1],
                'num_embeddings': embeddings.shape[0]
            }
            
        except Exception as e:
            self.logger.error(f"提取BERT嵌入向量時出錯: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def mean_pooling(self, model_output, attention_mask):
        """對模型輸出進行平均池化
        
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


def extract_embeddings_from_file(file_path, output_dir='./output/embeddings', model_name='bert-base-uncased', 
                                 batch_size=16, max_length=128, use_gpu=True, progress_callback=None):
    """從處理後的文件中提取BERT嵌入向量的方便函數
    
    Args:
        file_path: 處理後的數據文件路徑
        output_dir: 輸出目錄
        model_name: BERT模型名稱
        batch_size: 批次大小
        max_length: 最大序列長度
        use_gpu: 是否使用GPU加速
        progress_callback: 進度回調函數
        
    Returns:
        dict: 包含嵌入向量文件路徑和元數據文件路徑的字典
    """
    # 創建配置
    config = {
        'model_name': model_name,
        'batch_size': batch_size,
        'max_length': max_length,
        'use_gpu': use_gpu,
        'output_dir': output_dir
    }
    
    # 創建BERT嵌入器
    embedder = BertEmbedder(config)
    
    # 提取嵌入向量
    return embedder.extract_embeddings(file_path, progress_callback=progress_callback)


# 測試代碼
if __name__ == "__main__":
    # 配置日誌級別
    logger.setLevel(logging.INFO)
    
    # 測試BERT嵌入提取
    test_file = "path/to/your/processed_data.csv"
    
    if os.path.exists(test_file):
        # 定義進度回調函數
        def progress_callback(message, percentage):
            print(f"{message} - {percentage}%")
        
        # 提取嵌入向量
        result = extract_embeddings_from_file(
            test_file,
            progress_callback=progress_callback
        )
        
        if result:
            print(f"嵌入向量已保存至: {result['embeddings_path']}")
            print(f"元數據已保存至: {result['metadata_path']}")
            print(f"嵌入維度: {result['embedding_dim']}")
        else:
            print("提取嵌入向量失敗")
    else:
        logger.warning(f"測試文件不存在: {test_file}")