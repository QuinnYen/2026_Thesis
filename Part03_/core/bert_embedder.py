"""
BERT嵌入模組
此模組負責使用BERT模型生成文本嵌入向量
"""

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, logging
import pickle
import time
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Tuple, Any
import logging as py_logging
import sys
from pathlib import Path

# 添加專案根目錄到模組搜尋路徑
sys.path.append(os.path.abspath('.'))

# 從utils模組導入工具類
from Part03_.utils.console_output import ConsoleOutputManager
from Part03_.utils.config_manager import ConfigManager

# 設置transformers日誌級別
logging.set_verbosity_error()

class BertEmbedder:
    """
    BERT嵌入向量生成類
    使用預訓練的BERT模型將文本轉換為語義向量
    """
    
    def __init__(self, model_name: str = 'bert-base-chinese', 
                use_cuda: bool = True, 
                cache_dir: Optional[str] = None,
                config: Optional[ConfigManager] = None):
        """
        初始化BERT嵌入生成器
        
        Args:
            model_name: BERT模型名稱或路徑
            use_cuda: 是否使用CUDA加速
            cache_dir: 模型快取目錄
            config: 配置管理器，如果沒有提供則創建新的
        """
        # 初始化配置
        self.config = config if config else ConfigManager()
        
        # 從配置中獲取模型參數
        self.model_name = model_name or self.config.get('model_settings.bert.model_name', 'bert-base-chinese')
        self.use_cuda = use_cuda if use_cuda is not None else self.config.get('processing.use_cuda', True)
        self.cache_dir = cache_dir or self.config.get('model_settings.bert.cache_dir', './Part03_/models/bert')
        
        # 確保快取目錄存在
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # 檢查CUDA可用性
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')
        
        # 初始化日誌
        self.logger = py_logging.getLogger('bert_embedder')
        self.logger.setLevel(py_logging.INFO)
        
        # 移除所有處理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 添加控制台處理器
        handler = py_logging.StreamHandler()
        handler.setFormatter(py_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        
        self.logger.info(f"使用設備: {self.device}")
        
        # 初始化模型和分詞器
        self.tokenizer = None
        self.model = None
        
        # 其他參數
        self.max_length = self.config.get('model_settings.bert.max_length', 128)
        self.batch_size = self.config.get('model_settings.bert.batch_size', 16)
        self.output_dir = self.config.get('data_settings.output_directory', './Part03_/results/')
        
        # 創建嵌入輸出目錄
        self.embeddings_dir = os.path.join(self.output_dir, '02_bert_embeddings')
        os.makedirs(self.embeddings_dir, exist_ok=True)
    
    def load_model(self) -> Tuple[bool, str]:
        """
        載入BERT模型和分詞器
        
        Returns:
            成功標誌, 錯誤信息(如果有)
        """
        try:
            self.logger.info(f"正在載入模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # 將模型移動到設備
            self.model = self.model.to(self.device)
            
            # 設置為評估模式
            self.model.eval()
            
            self.logger.info("模型載入成功")
            return True, ""
            
        except Exception as e:
            error_msg = f"載入模型時發生錯誤: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        對token嵌入進行平均池化
        
        Args:
            token_embeddings: token嵌入張量 [batch_size, seq_length, hidden_size]
            attention_mask: 注意力掩碼 [batch_size, seq_length]
            
        Returns:
            句子嵌入 [batch_size, hidden_size]
        """
        # 擴展注意力掩碼以適應嵌入維度
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 對掩碼中的token嵌入求和並除以掩碼總和
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        為一組文本生成嵌入向量
        
        Args:
            texts: 文本列表
            show_progress: 是否顯示進度條
            
        Returns:
            嵌入向量數組 [num_texts, hidden_size]
        """
        # 確保模型已載入
        if self.model is None or self.tokenizer is None:
            success, error = self.load_model()
            if not success:
                raise RuntimeError(f"無法生成嵌入: {error}")
        
        all_embeddings = []
        
        # 分批處理文本
        batch_iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, desc="生成嵌入向量")
        
        try:
            with torch.no_grad():
                for i in batch_iterator:
                    # 顯式進行垃圾回收以釋放CUDA記憶體
                    if i > 0 and i % 10 == 0:  
                        torch.cuda.empty_cache()
                    
                    batch_texts = texts[i:i + self.batch_size]
                    
                    # 編碼文本
                    encoded_input = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    )
                    
                    # 移動到相應設備
                    encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
                    
                    # 計算token嵌入
                    model_output = self.model(**encoded_input)
                    
                    # 使用平均池化獲取句子嵌入
                    sentence_embeddings = self.mean_pooling(
                        model_output.last_hidden_state,
                        encoded_input['attention_mask']
                    )
                    
                    # 轉換為numpy並添加到結果列表
                    batch_embeddings = sentence_embeddings.cpu().numpy()
                    all_embeddings.append(batch_embeddings)
                    
                    # 明確釋放不再需要的PyTorch張量
                    del encoded_input, model_output, sentence_embeddings
                    
            # 合併所有批次的嵌入
            return np.vstack(all_embeddings)
        except Exception as e:
            self.logger.error(f"生成嵌入時出錯: {str(e)}")
            # 確保在發生異常時釋放CUDA記憶體
            torch.cuda.empty_cache()
            raise
    
    def process_dataset(self, df: pd.DataFrame, text_column: str = 'processed_text',
                      id_column: Optional[str] = None, output_file: Optional[str] = None,
                      console_output: bool = True) -> Dict[str, np.ndarray]:
        """
        處理數據集並生成嵌入向量
        
        Args:
            df: 包含文本數據的DataFrame
            text_column: 文本列名
            id_column: 文本ID列名，如果提供則作為索引
            output_file: 輸出文件路徑
            console_output: 是否在控制台顯示處理進度
            
        Returns:
            文本ID和嵌入向量的字典
        """
        # 設置控制台輸出
        if console_output:
            log_file, status_file = ConsoleOutputManager.open_console("BERT語義提取")
            logger = ConsoleOutputManager.setup_console_logger("bert_embedding", log_file)
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
            
            # 獲取文本列表
            texts = df[text_column].tolist()
            logger.info(f"開始為 {len(texts)} 條文本生成BERT嵌入向量")
            
            # 生成嵌入向量
            embeddings = self.generate_embeddings(texts, show_progress=False)
            
            logger.info(f"嵌入向量生成完成，形狀: {embeddings.shape}")
            
            # 創建結果字典
            if id_column and id_column in df.columns:
                ids = df[id_column].tolist()
                result = {id_val: emb for id_val, emb in zip(ids, embeddings)}
            else:
                result = {i: emb for i, emb in enumerate(embeddings)}
            
            # 如果提供了輸出文件路徑，保存結果
            if output_file:
                # 確保輸出目錄存在
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                with open(output_file, 'wb') as f:
                    pickle.dump(result, f)
                logger.info(f"嵌入向量已保存到: {output_file}")
            
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            
            return result
            
        except Exception as e:
            logger.error(f"生成嵌入向量時發生錯誤: {str(e)}")
            if console_output:
                ConsoleOutputManager.mark_process_complete(status_file)
            raise
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], dataset_name: str) -> str:
        """
        保存嵌入向量到文件
        
        Args:
            embeddings: 嵌入向量字典 {id: vector}
            dataset_name: 數據集名稱
            
        Returns:
            保存的文件路徑
        """
        # 生成文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{dataset_name}_embeddings_{timestamp}.pkl"
        file_path = os.path.join(self.embeddings_dir, filename)
        
        # 保存到文件
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        
        self.logger.info(f"嵌入向量已保存到: {file_path}")
        return file_path
    
    def load_embeddings(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        從文件載入嵌入向量
        
        Args:
            file_path: 嵌入向量文件路徑
            
        Returns:
            嵌入向量字典 {id: vector}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"嵌入向量文件不存在: {file_path}")
        
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        self.logger.info(f"已從 {file_path} 載入嵌入向量")
        return embeddings
    
    def generate_contextual_embeddings(self, text: str, aspect_terms: List[str]) -> Dict[str, np.ndarray]:
        """
        為給定文本中的面向詞生成上下文感知的嵌入向量
        
        Args:
            text: 輸入文本
            aspect_terms: 面向詞列表
            
        Returns:
            面向詞和其上下文嵌入向量的字典
        """
        # 確保模型已載入
        if self.model is None or self.tokenizer is None:
            success, error = self.load_model()
            if not success:
                raise RuntimeError(f"無法生成嵌入: {error}")
        
        # 標記文本中的面向詞位置
        term_positions = {}
        for term in aspect_terms:
            lower_text = text.lower()
            lower_term = term.lower()
            start_idx = lower_text.find(lower_term)
            
            if start_idx != -1:
                term_positions[term] = (start_idx, start_idx + len(term))
        
        # 如果沒有找到面向詞
        if not term_positions:
            # 生成整個文本的嵌入
            embeddings = self.generate_embeddings([text], show_progress=False)[0]
            return {term: embeddings for term in aspect_terms}
        
        # 為每個面向詞生成上下文嵌入
        result = {}
        
        with torch.no_grad():
            # 編碼文本
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 移動到相應設備
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # 計算token嵌入
            outputs = self.model(**encoded_input)
            
            # 獲取token嵌入
            token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
            
            # 獲取token與原始文本位置的映射
            tokens = self.tokenizer.tokenize(text)
            word_ids = self.tokenizer(text, add_special_tokens=True).word_ids()
            
            # 為每個面向詞生成上下文嵌入
            for term, (start_pos, end_pos) in term_positions.items():
                # 找到對應的token索引
                token_indices = []
                for i, word_id in enumerate(word_ids):
                    if word_id is not None and start_pos <= self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids([tokens[word_id-1]])).find(text[word_id:word_id+1]) <= end_pos:
                        token_indices.append(i)
                
                if token_indices:
                    # 使用相關token的嵌入向量的平均值
                    aspect_embedding = np.mean(token_embeddings[token_indices], axis=0)
                    result[term] = aspect_embedding
                else:
                    # 如果找不到對應的token，使用整個文本的嵌入
                    result[term] = self.mean_pooling(
                        outputs.last_hidden_state,
                        encoded_input['attention_mask']
                    ).cpu().numpy()[0]
        
        # 檢查是否有未找到的面向詞
        missing_aspects = [term for term in aspect_terms if term not in result]
        if missing_aspects:
            # 生成整個文本的嵌入
            full_embedding = self.mean_pooling(
                outputs.last_hidden_state,
                encoded_input['attention_mask']
            ).cpu().numpy()[0]
            
            # 對於未找到的面向詞，使用整個文本的嵌入
            for term in missing_aspects:
                result[term] = full_embedding
        
        return result

# 使用示例
if __name__ == "__main__":
    # 初始化嵌入器
    embedder = BertEmbedder()
    
    # 測試簡單的文本嵌入
    texts = [
        "這家餐廳的食物非常美味",
        "服務有點差，但價格便宜",
        "環境很好，服務員態度友善"
    ]
    
    # 生成嵌入
    embeddings = embedder.generate_embeddings(texts)
    
    print(f"嵌入向量形狀: {embeddings.shape}")
    
    # 測試上下文嵌入
    text = "這家餐廳的食物很好吃，但服務態度不太友善，價格還算合理。"
    aspects = ["食物", "服務", "價格"]
    
    aspect_embeddings = embedder.generate_contextual_embeddings(text, aspects)
    
    for aspect, emb in aspect_embeddings.items():
        print(f"{aspect} 嵌入向量形狀: {emb.shape}")
