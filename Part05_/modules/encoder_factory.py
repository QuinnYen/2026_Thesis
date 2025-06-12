"""
文本編碼器工廠模組
提供各種文本編碼器的統一接口和管理
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Any
from transformers import (
    BertTokenizer, BertModel,
    GPT2Tokenizer, GPT2Model, 
    T5Tokenizer, T5EncoderModel,
    AutoTokenizer, AutoModel
)
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from .base_interfaces import BaseTextEncoder

logger = logging.getLogger(__name__)


class BertEncoder(BaseTextEncoder):
    """BERT編碼器實現"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        self.model_name = self.config.get('model_name', 'bert-base-uncased')
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 32)
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            if self.progress_callback:
                self.progress_callback('status', f'BERT編碼器已載入 ({self.device})')
                
        except Exception as e:
            logger.error(f"BERT編碼器初始化失敗: {e}")
            raise
    
    def encode(self, texts: Union[pd.Series, List[str]]) -> np.ndarray:
        """編碼文本為BERT向量"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        embeddings = []
        total_batches = len(texts) // self.batch_size + (1 if len(texts) % self.batch_size > 0 else 0)
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # 編碼文本
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 移到設備
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 獲取嵌入向量
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
                
                # 更新進度
                if self.progress_callback:
                    progress = (i // self.batch_size + 1) / total_batches * 100
                    self.progress_callback('progress', f'BERT編碼進度: {progress:.1f}%')
        
        return np.vstack(embeddings)
    
    def get_embedding_dim(self) -> int:
        return 768  # BERT-base的維度


class GPTEncoder(BaseTextEncoder):
    """GPT編碼器實現"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        self.model_name = self.config.get('model_name', 'gpt2')
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 32)
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2Model.from_pretrained(self.model_name)
            
            # GPT2沒有pad token，需要設置
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            if self.progress_callback:
                self.progress_callback('status', f'GPT編碼器已載入 ({self.device})')
                
        except Exception as e:
            logger.error(f"GPT編碼器初始化失敗: {e}")
            raise
    
    def encode(self, texts: Union[pd.Series, List[str]]) -> np.ndarray:
        """編碼文本為GPT向量"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        embeddings = []
        total_batches = len(texts) // self.batch_size + (1 if len(texts) % self.batch_size > 0 else 0)
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # 編碼文本
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 移到設備
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 獲取最後一層隱藏狀態，取最後一個token的向量
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # 使用attention mask來獲取實際的最後一個token
                batch_embeddings = []
                for j, mask in enumerate(attention_mask):
                    actual_length = mask.sum().item()
                    last_hidden = outputs.last_hidden_state[j, actual_length-1, :]
                    batch_embeddings.append(last_hidden.cpu().numpy())
                
                embeddings.append(np.array(batch_embeddings))
                
                # 更新進度
                if self.progress_callback:
                    progress = (i // self.batch_size + 1) / total_batches * 100
                    self.progress_callback('progress', f'GPT編碼進度: {progress:.1f}%')
        
        return np.vstack(embeddings)
    
    def get_embedding_dim(self) -> int:
        return 768  # GPT2的維度


class T5Encoder(BaseTextEncoder):
    """T5編碼器實現"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        self.model_name = self.config.get('model_name', 't5-base')
        self.max_length = self.config.get('max_length', 512)
        self.batch_size = self.config.get('batch_size', 32)
        
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5EncoderModel.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            if self.progress_callback:
                self.progress_callback('status', f'T5編碼器已載入 ({self.device})')
                
        except Exception as e:
            logger.error(f"T5編碼器初始化失敗: {e}")
            raise
    
    def encode(self, texts: Union[pd.Series, List[str]]) -> np.ndarray:
        """編碼文本為T5向量"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        embeddings = []
        total_batches = len(texts) // self.batch_size + (1 if len(texts) % self.batch_size > 0 else 0)
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # 編碼文本
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # 移到設備
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                
                # 獲取編碼器輸出
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)
                
                # 更新進度
                if self.progress_callback:
                    progress = (i // self.batch_size + 1) / total_batches * 100
                    self.progress_callback('progress', f'T5編碼進度: {progress:.1f}%')
        
        return np.vstack(embeddings)
    
    def get_embedding_dim(self) -> int:
        return 768  # T5-base的維度


class CNNEncoder(BaseTextEncoder):
    """CNN文本編碼器實現"""
    
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, filter_sizes, num_filters, output_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.convs = nn.ModuleList([
                nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
            ])
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
        def forward(self, x):
            x = self.embedding(x)
            x = x.unsqueeze(1)
            x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
            x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
            x = torch.cat(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        self.embed_dim = self.config.get('embed_dim', 100)
        self.filter_sizes = self.config.get('filter_sizes', [3, 4, 5])
        self.num_filters = self.config.get('num_filters', 100)
        self.output_dim = self.config.get('output_dim', 256)
        self.max_length = self.config.get('max_length', 200)
        self.vocab_size = self.config.get('vocab_size', 10000)
        
        # 初始化TF-IDF向量化器用於詞彙映射
        self.vectorizer = TfidfVectorizer(max_features=self.vocab_size)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.progress_callback:
            self.progress_callback('status', f'CNN編碼器已初始化 ({self.device})')
    
    def _build_vocab(self, texts: List[str]):
        """構建詞彙表"""
        self.vectorizer.fit(texts)
        self.vocab_size = len(self.vectorizer.vocabulary_)
        
        # 初始化CNN模型
        self.model = self.TextCNN(
            self.vocab_size, 
            self.embed_dim, 
            self.filter_sizes, 
            self.num_filters, 
            self.output_dim
        )
        self.model.to(self.device)
        self.model.eval()
    
    def _texts_to_sequences(self, texts: List[str]) -> torch.Tensor:
        """將文本轉換為序列"""
        sequences = []
        for text in texts:
            words = text.lower().split()[:self.max_length]
            sequence = []
            for word in words:
                if word in self.vectorizer.vocabulary_:
                    sequence.append(self.vectorizer.vocabulary_[word])
                else:
                    sequence.append(0)  # UNK token
            
            # 填充到固定長度
            if len(sequence) < self.max_length:
                sequence.extend([0] * (self.max_length - len(sequence)))
            
            sequences.append(sequence)
        
        return torch.tensor(sequences, dtype=torch.long)
    
    def encode(self, texts: Union[pd.Series, List[str]]) -> np.ndarray:
        """編碼文本為CNN向量"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # 如果模型未構建，先構建詞彙表
        if self.model is None:
            if self.progress_callback:
                self.progress_callback('status', 'CNN正在構建詞彙表...')
            self._build_vocab(texts)
        
        # 轉換為序列
        sequences = self._texts_to_sequences(texts)
        sequences = sequences.to(self.device)
        
        embeddings = []
        batch_size = 32
        total_batches = len(sequences) // batch_size + (1 if len(sequences) % batch_size > 0 else 0)
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_sequences = sequences[i:i + batch_size]
                batch_embeddings = self.model(batch_sequences)
                embeddings.append(batch_embeddings.cpu().numpy())
                
                # 更新進度
                if self.progress_callback:
                    progress = (i // batch_size + 1) / total_batches * 100
                    self.progress_callback('progress', f'CNN編碼進度: {progress:.1f}%')
        
        return np.vstack(embeddings)
    
    def get_embedding_dim(self) -> int:
        return self.output_dim


class ELMoEncoder(BaseTextEncoder):
    """ELMo編碼器實現（簡化版，使用預訓練模型）"""
    
    def __init__(self, config: Optional[Dict] = None, progress_callback=None):
        super().__init__(config, progress_callback)
        
        try:
            # 使用allennlp的ELMo實現
            from allennlp.modules.elmo import Elmo, batch_to_ids
            
            self.options_file = self.config.get('options_file', 
                "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json")
            self.weight_file = self.config.get('weight_file',
                "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5")
            
            self.elmo = Elmo(self.options_file, self.weight_file, 2, dropout=0)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.elmo.to(self.device)
            
            if self.progress_callback:
                self.progress_callback('status', f'ELMo編碼器已載入 ({self.device})')
                
        except ImportError:
            # 如果allennlp不可用，使用簡化的實現
            if self.progress_callback:
                self.progress_callback('status', 'ELMo需要allennlp庫，使用TF-IDF替代')
            self.vectorizer = TfidfVectorizer(max_features=1024)
            self.use_tfidf = True
        except Exception as e:
            logger.error(f"ELMo編碼器初始化失敗: {e}")
            raise
    
    def encode(self, texts: Union[pd.Series, List[str]]) -> np.ndarray:
        """編碼文本為ELMo向量"""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        if hasattr(self, 'use_tfidf') and self.use_tfidf:
            # 使用TF-IDF作為替代
            if not hasattr(self.vectorizer, 'vocabulary_'):
                embeddings = self.vectorizer.fit_transform(texts)
            else:
                embeddings = self.vectorizer.transform(texts)
            return embeddings.toarray()
        
        # 使用真正的ELMo
        from allennlp.modules.elmo import batch_to_ids
        
        embeddings = []
        batch_size = 32
        total_batches = len(texts) // batch_size + (1 if len(texts) % batch_size > 0 else 0)
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 分詞
            tokenized_texts = [text.split() for text in batch_texts]
            character_ids = batch_to_ids(tokenized_texts).to(self.device)
            
            with torch.no_grad():
                embeddings_dict = self.elmo(character_ids)
                batch_embeddings = embeddings_dict['elmo_representations'][0].mean(dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())
            
            # 更新進度
            if self.progress_callback:
                progress = (i // batch_size + 1) / total_batches * 100
                self.progress_callback('progress', f'ELMo編碼進度: {progress:.1f}%')
        
        return np.vstack(embeddings)
    
    def get_embedding_dim(self) -> int:
        if hasattr(self, 'use_tfidf') and self.use_tfidf:
            return 1024
        return 1024  # ELMo的標準維度


class EncoderFactory:
    """編碼器工廠類"""
    
    _encoders = {
        'bert': BertEncoder,
        'gpt': GPTEncoder,
        't5': T5Encoder,
        'cnn': CNNEncoder,
        'elmo': ELMoEncoder
    }
    
    @classmethod
    def create_encoder(cls, encoder_type: str, config: Optional[Dict] = None, 
                      progress_callback=None) -> BaseTextEncoder:
        """
        創建指定類型的編碼器
        
        Args:
            encoder_type: 編碼器類型 ('bert', 'gpt', 't5', 'cnn', 'elmo')
            config: 配置參數
            progress_callback: 進度回調函數
            
        Returns:
            BaseTextEncoder: 編碼器實例
        """
        encoder_type = encoder_type.lower()
        if encoder_type not in cls._encoders:
            raise ValueError(f"不支援的編碼器類型: {encoder_type}")
        
        encoder_class = cls._encoders[encoder_type]
        return encoder_class(config, progress_callback)
    
    @classmethod
    def get_available_encoders(cls) -> List[str]:
        """獲取可用的編碼器列表"""
        return list(cls._encoders.keys())
    
    @classmethod
    def get_encoder_info(cls, encoder_type: str) -> Dict[str, Any]:
        """獲取編碼器信息"""
        encoder_descriptions = {
            'bert': {
                'name': 'BERT',
                'description': 'Bidirectional Encoder Representations from Transformers',
                'embedding_dim': 768,
                'advantages': ['語義理解強', '預訓練效果好', '支援多種下游任務'],
                'disadvantages': ['計算資源需求高', '推理速度較慢']
            },
            'gpt': {
                'name': 'GPT',
                'description': 'Generative Pre-trained Transformer',
                'embedding_dim': 768,
                'advantages': ['生成能力強', '語言模型效果好'],
                'disadvantages': ['單向注意力', '在某些NLU任務上不如BERT']
            },
            't5': {
                'name': 'T5',
                'description': 'Text-to-Text Transfer Transformer',
                'embedding_dim': 768,
                'advantages': ['統一的text-to-text框架', '多任務學習效果好'],
                'disadvantages': ['模型較大', '計算開銷高']
            },
            'cnn': {
                'name': 'CNN',
                'description': 'Convolutional Neural Network for Text',
                'embedding_dim': 256,
                'advantages': ['計算效率高', '並行處理能力強', '局部特徵提取好'],
                'disadvantages': ['無法捕捉長距離依賴', '語義理解相對較弱']
            },
            'elmo': {
                'name': 'ELMo',
                'description': 'Embeddings from Language Models',
                'embedding_dim': 1024,
                'advantages': ['上下文相關嵌入', '字符級建模'],
                'disadvantages': ['計算較慢', '相比Transformer效果稍差']
            }
        }
        
        return encoder_descriptions.get(encoder_type.lower(), {})