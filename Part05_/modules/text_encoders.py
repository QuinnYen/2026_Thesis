#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本編碼器模組 - 支援多種文本向量化方法
包含：BERT、GPT、T5、CNN、ELMo
"""

import torch
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Tuple
import logging
import os
from abc import ABC, abstractmethod
from tqdm import tqdm

# Transformers相關導入
try:
    from transformers import (
        BertTokenizer, BertModel,
        GPT2Tokenizer, GPT2Model,
        T5Tokenizer, T5EncoderModel,
        AutoTokenizer, AutoModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ELMo相關導入
try:
    from allennlp.modules.elmo import Elmo, batch_to_ids
    ELMO_AVAILABLE = True
except ImportError:
    ELMO_AVAILABLE = False

# CNN相關導入
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class BaseTextEncoder(ABC):
    """文本編碼器基類"""
    
    def __init__(self, output_dir: Optional[str] = None, progress_callback=None):
        self.output_dir = output_dir
        self.progress_callback = progress_callback
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用設備: {self.device}")
        
    @abstractmethod
    def encode(self, texts: pd.Series) -> np.ndarray:
        """編碼文本為向量"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """返回嵌入向量維度"""
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

class BertEncoder(BaseTextEncoder):
    """BERT編碼器"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', **kwargs):
        super().__init__(**kwargs)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers未安裝，請執行：pip install transformers")
        
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def get_embedding_dim(self) -> int:
        return 768 if 'base' in self.model_name else 1024
    
    def encode(self, texts: pd.Series) -> np.ndarray:
        embeddings = []
        batch_size = 32
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        self._notify_progress('phase', 1, 3, f'BERT編碼 ({self.model_name})')
        self._notify_progress('status', status=f"開始BERT編碼：{len(texts)} 條文本")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT編碼"):
            batch_texts = texts.iloc[i:i+batch_size].tolist()
            encoded = self.tokenizer(batch_texts, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=512, 
                                   return_tensors='pt')
            
            with torch.no_grad():
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                # 使用[CLS]標記的輸出作為文本表示
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
            
            if self.progress_callback:
                batch_num = i // batch_size + 1
                self.progress_callback('progress', (batch_num, total_batches))
        
        embeddings = np.vstack(embeddings)
        self._save_embeddings(embeddings, "02_bert_embeddings.npy")
        return embeddings
    
    def _save_embeddings(self, embeddings: np.ndarray, filename: str):
        if self.output_dir:
            encoder_type = self.__class__.__name__.lower().replace('encoder', '')
            try:
                from ..utils.storage_manager import StorageManager
                storage_manager = StorageManager(self.output_dir)
                output_file = storage_manager.save_embeddings(embeddings, encoder_type)
                logger.info(f"已保存{encoder_type.upper()}特徵向量到：{output_file}")
            except Exception as e:
                output_file = os.path.join(self.output_dir, filename)
                np.save(output_file, embeddings)
                logger.info(f"已保存{encoder_type.upper()}特徵向量到：{output_file}")

class GPTEncoder(BaseTextEncoder):
    """GPT編碼器"""
    
    def __init__(self, model_name: str = 'gpt2', **kwargs):
        super().__init__(**kwargs)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers未安裝，請執行：pip install transformers")
        
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        
        # GPT2沒有pad_token，使用eos_token作為pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
    def get_embedding_dim(self) -> int:
        return self.model.config.n_embd  # GPT-2: 768, GPT-2-medium: 1024, etc.
    
    def encode(self, texts: pd.Series) -> np.ndarray:
        embeddings = []
        batch_size = 16  # GPT模型較大，使用較小的batch size
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        self._notify_progress('phase', 1, 3, f'GPT編碼 ({self.model_name})')
        self._notify_progress('status', status=f"開始GPT編碼：{len(texts)} 條文本")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="GPT編碼"):
            batch_texts = texts.iloc[i:i+batch_size].tolist()
            encoded = self.tokenizer(batch_texts, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=512, 
                                   return_tensors='pt')
            
            with torch.no_grad():
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                # 使用最後一個token的hidden state作為文本表示
                last_hidden_states = outputs.last_hidden_state
                # 找到每個序列的最後一個非pad token的位置
                attention_mask = encoded['attention_mask']
                sequence_lengths = attention_mask.sum(dim=1) - 1  # -1因為索引從0開始
                
                batch_embeddings = []
                for j, seq_len in enumerate(sequence_lengths):
                    batch_embeddings.append(last_hidden_states[j, seq_len, :].cpu().numpy())
                
                embeddings.append(np.stack(batch_embeddings))
            
            if self.progress_callback:
                batch_num = i // batch_size + 1
                self.progress_callback('progress', (batch_num, total_batches))
        
        embeddings = np.vstack(embeddings)
        self._save_embeddings(embeddings, "02_gpt_embeddings.npy")
        return embeddings
    
    def _save_embeddings(self, embeddings: np.ndarray, filename: str):
        if self.output_dir:
            encoder_type = self.__class__.__name__.lower().replace('encoder', '')
            try:
                from ..utils.storage_manager import StorageManager
                storage_manager = StorageManager(self.output_dir)
                output_file = storage_manager.save_embeddings(embeddings, encoder_type)
                logger.info(f"已保存{encoder_type.upper()}特徵向量到：{output_file}")
            except Exception as e:
                output_file = os.path.join(self.output_dir, filename)
                np.save(output_file, embeddings)
                logger.info(f"已保存{encoder_type.upper()}特徵向量到：{output_file}")

class T5Encoder(BaseTextEncoder):
    """T5編碼器"""
    
    def __init__(self, model_name: str = 't5-base', **kwargs):
        super().__init__(**kwargs)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers未安裝，請執行：pip install transformers")
        
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        # 使用T5EncoderModel只取encoder部分
        self.model = T5EncoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def get_embedding_dim(self) -> int:
        return self.model.config.d_model  # T5-base: 768, T5-large: 1024
    
    def encode(self, texts: pd.Series) -> np.ndarray:
        embeddings = []
        batch_size = 16
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        self._notify_progress('phase', 1, 3, f'T5編碼 ({self.model_name})')
        self._notify_progress('status', status=f"開始T5編碼：{len(texts)} 條文本")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="T5編碼"):
            batch_texts = texts.iloc[i:i+batch_size].tolist()
            # T5需要添加前綴，用於文本分類任務
            batch_texts = [f"classify: {text}" for text in batch_texts]
            
            encoded = self.tokenizer(batch_texts, 
                                   padding=True, 
                                   truncation=True, 
                                   max_length=512, 
                                   return_tensors='pt')
            
            with torch.no_grad():
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                # 使用平均池化作為文本表示
                last_hidden_states = outputs.last_hidden_state
                attention_mask = encoded['attention_mask'].unsqueeze(-1)
                
                # 計算加權平均（排除padding tokens）
                masked_embeddings = last_hidden_states * attention_mask
                sum_embeddings = masked_embeddings.sum(dim=1)
                seq_lengths = attention_mask.sum(dim=1)
                batch_embeddings = sum_embeddings / seq_lengths
                
                embeddings.append(batch_embeddings.cpu().numpy())
            
            if self.progress_callback:
                batch_num = i // batch_size + 1
                self.progress_callback('progress', (batch_num, total_batches))
        
        embeddings = np.vstack(embeddings)
        self._save_embeddings(embeddings, "02_t5_embeddings.npy")
        return embeddings
    
    def _save_embeddings(self, embeddings: np.ndarray, filename: str):
        if self.output_dir:
            encoder_type = self.__class__.__name__.lower().replace('encoder', '')
            try:
                from ..utils.storage_manager import StorageManager
                storage_manager = StorageManager(self.output_dir)
                output_file = storage_manager.save_embeddings(embeddings, encoder_type)
                logger.info(f"已保存{encoder_type.upper()}特徵向量到：{output_file}")
            except Exception as e:
                output_file = os.path.join(self.output_dir, filename)
                np.save(output_file, embeddings)
                logger.info(f"已保存{encoder_type.upper()}特徵向量到：{output_file}")

class CNNEncoder(BaseTextEncoder):
    """CNN文本編碼器"""
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 300, 
                 num_filters: int = 100, filter_sizes: List[int] = None, **kwargs):
        super().__init__(**kwargs)
        
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.output_dim = num_filters * len(filter_sizes)
        
        # 構建CNN模型
        self.model = self._build_cnn_model()
        self.model.to(self.device)
        
        # 詞彙表和分詞器
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def _build_cnn_model(self):
        """構建CNN模型"""
        class TextCNN(nn.Module):
            def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.convs = nn.ModuleList([
                    nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
                    for fs in filter_sizes
                ])
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                # x shape: (batch_size, seq_len)
                embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
                embedded = embedded.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
                
                conv_outputs = []
                for conv in self.convs:
                    conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, new_seq_len)
                    pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
                    conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
                
                output = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
                output = self.dropout(output)
                return output
        
        return TextCNN(self.vocab_size, self.embedding_dim, self.num_filters, self.filter_sizes)
    
    def _build_vocabulary(self, texts: pd.Series):
        """構建詞彙表"""
        from collections import Counter
        
        # 簡單的分詞（可以根據需要改進）
        all_words = []
        for text in texts:
            words = str(text).lower().split()
            all_words.extend(words)
        
        # 統計詞頻並選擇top vocab_size-2 個詞（留出UNK和PAD）
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        # 構建詞彙表
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, _) in enumerate(most_common, 2):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        logger.info(f"構建詞彙表完成，詞彙量：{len(self.word_to_idx)}")
    
    def _text_to_indices(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """將文本轉換為索引序列"""
        indices_list = []
        
        for text in texts:
            words = str(text).lower().split()
            indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 是 <UNK>
            
            # 截斷或填充
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices.extend([0] * (max_length - len(indices)))  # 0 是 <PAD>
            
            indices_list.append(indices)
        
        return torch.tensor(indices_list, dtype=torch.long)
    
    def get_embedding_dim(self) -> int:
        return self.output_dim
    
    def encode(self, texts: pd.Series) -> np.ndarray:
        # 構建詞彙表
        self._notify_progress('phase', 1, 4, 'CNN編碼 - 構建詞彙表')
        self._build_vocabulary(texts)
        
        embeddings = []
        batch_size = 64
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        self._notify_progress('phase', 2, 4, 'CNN編碼 - 特徵提取')
        self._notify_progress('status', status=f"開始CNN編碼：{len(texts)} 條文本")
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="CNN編碼"):
                batch_texts = texts.iloc[i:i+batch_size].tolist()
                batch_indices = self._text_to_indices(batch_texts)
                batch_indices = batch_indices.to(self.device)
                
                batch_embeddings = self.model(batch_indices)
                embeddings.append(batch_embeddings.cpu().numpy())
                
                if self.progress_callback:
                    batch_num = i // batch_size + 1
                    self.progress_callback('progress', (batch_num, total_batches))
        
        embeddings = np.vstack(embeddings)
        self._save_embeddings(embeddings, "02_cnn_embeddings.npy")
        return embeddings
    
    def _save_embeddings(self, embeddings: np.ndarray, filename: str):
        if self.output_dir:
            encoder_type = 'cnn'
            try:
                from ..utils.storage_manager import StorageManager
                storage_manager = StorageManager(self.output_dir)
                # 保存特徵向量
                output_file = storage_manager.save_embeddings(embeddings, encoder_type)
                # 保存詞彙表到同一目錄
                encoding_dir = storage_manager.get_encoding_dir(encoder_type)
                vocab_file = encoding_dir / "cnn_vocabulary.json"
                import json
                with open(vocab_file, 'w', encoding='utf-8') as f:
                    json.dump(self.word_to_idx, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存CNN特徵向量到：{output_file}")
                logger.info(f"已保存CNN詞彙表到：{vocab_file}")
            except Exception as e:
                # 回退到直接保存方式
                output_file = os.path.join(self.output_dir, filename)
                np.save(output_file, embeddings)
                vocab_file = os.path.join(self.output_dir, "cnn_vocabulary.json")
                import json
                with open(vocab_file, 'w', encoding='utf-8') as f:
                    json.dump(self.word_to_idx, f, ensure_ascii=False, indent=2)
                logger.info(f"已保存CNN特徵向量到：{output_file}")
                logger.info(f"已保存CNN詞彙表到：{vocab_file}")

class ELMoEncoder(BaseTextEncoder):
    """ELMo編碼器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not ELMO_AVAILABLE:
            raise ImportError("AllenNLP未安裝，請執行：pip install allennlp")
        
        # ELMo模型參數
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        
        # 初始化ELMo模型
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        self.elmo.to(self.device)
        
    def get_embedding_dim(self) -> int:
        return 1024  # ELMo輸出維度
    
    def encode(self, texts: pd.Series) -> np.ndarray:
        embeddings = []
        batch_size = 8  # ELMo模型較大，使用小批量
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        self._notify_progress('phase', 1, 3, 'ELMo編碼')
        self._notify_progress('status', status=f"開始ELMo編碼：{len(texts)} 條文本")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="ELMo編碼"):
            batch_texts = texts.iloc[i:i+batch_size].tolist()
            # 將文本轉換為字符ID
            character_ids = batch_to_ids([text.split() for text in batch_texts])
            character_ids = character_ids.to(self.device)
            
            with torch.no_grad():
                embeddings_dict = self.elmo(character_ids)
                # 使用ELMo的輸出表示（通常是所有層的加權平均）
                elmo_embeddings = embeddings_dict['elmo_representations'][0]
                
                # 計算每個句子的平均嵌入
                mask = embeddings_dict['mask']
                masked_embeddings = elmo_embeddings * mask.unsqueeze(-1)
                sentence_lengths = mask.sum(dim=1).float()
                
                batch_embeddings = masked_embeddings.sum(dim=1) / sentence_lengths.unsqueeze(-1)
                embeddings.append(batch_embeddings.cpu().numpy())
            
            if self.progress_callback:
                batch_num = i // batch_size + 1
                self.progress_callback('progress', (batch_num, total_batches))
        
        embeddings = np.vstack(embeddings)
        self._save_embeddings(embeddings, "02_elmo_embeddings.npy")
        return embeddings
    
    def _save_embeddings(self, embeddings: np.ndarray, filename: str):
        if self.output_dir:
            encoder_type = 'elmo'
            try:
                from ..utils.storage_manager import StorageManager
                storage_manager = StorageManager(self.output_dir)
                output_file = storage_manager.save_embeddings(embeddings, encoder_type)
                logger.info(f"已保存{encoder_type.upper()}特徵向量到：{output_file}")
            except Exception as e:
                output_file = os.path.join(self.output_dir, filename)
                np.save(output_file, embeddings)
                logger.info(f"已保存{encoder_type.upper()}特徵向量到：{output_file}")

class TextEncoderFactory:
    """文本編碼器工廠類"""
    
    @staticmethod
    def create_encoder(encoder_type: str, **kwargs) -> BaseTextEncoder:
        """
        創建指定類型的文本編碼器
        
        Args:
            encoder_type: 編碼器類型 ('bert', 'gpt', 't5', 'cnn', 'elmo')
            **kwargs: 其他參數
            
        Returns:
            BaseTextEncoder: 相應的編碼器實例
        """
        encoder_map = {
            'bert': BertEncoder,
            'gpt': GPTEncoder,
            't5': T5Encoder,
            'cnn': CNNEncoder,
            'elmo': ELMoEncoder
        }
        
        if encoder_type.lower() not in encoder_map:
            available_types = ', '.join(encoder_map.keys())
            raise ValueError(f"不支援的編碼器類型：{encoder_type}。可用類型：{available_types}")
        
        encoder_class = encoder_map[encoder_type.lower()]
        return encoder_class(**kwargs)
    
    @staticmethod
    def get_available_encoders() -> List[str]:
        """獲取可用的編碼器類型列表"""
        available = ['bert', 'gpt', 't5', 'cnn']
        
        if ELMO_AVAILABLE:
            available.append('elmo')
        
        return available
    
    @staticmethod
    def get_encoder_info() -> Dict[str, Dict]:
        """獲取編碼器的詳細信息"""
        info = {
            'bert': {
                'name': 'BERT',
                'description': 'Bidirectional Encoder Representations from Transformers',
                'embedding_dim': '768 (base) / 1024 (large)',
                'advantages': '雙向理解、預訓練效果好',
                'requirements': 'transformers'
            },
            'gpt': {
                'name': 'GPT',
                'description': 'Generative Pre-trained Transformer',
                'embedding_dim': '768 (GPT-2) / 1024+ (larger models)',
                'advantages': '生成能力強、上下文理解好',
                'requirements': 'transformers'
            },
            't5': {
                'name': 'T5',
                'description': 'Text-to-Text Transfer Transformer',
                'embedding_dim': '768 (base) / 1024 (large)',
                'advantages': '統一的text-to-text框架',
                'requirements': 'transformers'
            },
            'cnn': {
                'name': 'CNN',
                'description': 'Convolutional Neural Network for text',
                'embedding_dim': '可自定義 (預設: 300)',
                'advantages': '速度快、對局部特徵敏感',
                'requirements': 'pytorch (內建)'
            },
            'elmo': {
                'name': 'ELMo',
                'description': 'Embeddings from Language Models',
                'embedding_dim': '1024',
                'advantages': '上下文相關嵌入、字符級別',
                'requirements': 'allennlp'
            }
        }
        
        # 檢查可用性
        if not TRANSFORMERS_AVAILABLE:
            for encoder in ['bert', 'gpt', 't5']:
                info[encoder]['available'] = False
                info[encoder]['note'] = '需要安裝transformers'
        else:
            for encoder in ['bert', 'gpt', 't5']:
                info[encoder]['available'] = True
        
        info['cnn']['available'] = True  # PyTorch通常是可用的
        
        if ELMO_AVAILABLE:
            info['elmo']['available'] = True
        else:
            info['elmo']['available'] = False
            info['elmo']['note'] = '需要安裝allennlp'
        
        return info