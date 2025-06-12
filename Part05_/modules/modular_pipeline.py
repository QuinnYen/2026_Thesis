"""
模組化流水線處理器
整合不同的文本編碼器和面向分類器，提供靈活的組合選擇
"""

import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime
from .base_interfaces import BasePipeline
from .encoder_factory import EncoderFactory
from .aspect_factory import AspectFactory
from .text_preprocessor import TextPreprocessor
from .sentiment_classifier import SentimentClassifier
from .run_manager import RunManager

logger = logging.getLogger(__name__)


class ModularPipeline(BasePipeline):
    """模組化流水線處理器"""
    
    def __init__(self, 
                 encoder_type: str = 'bert',
                 aspect_type: str = 'default',
                 encoder_config: Optional[Dict] = None,
                 aspect_config: Optional[Dict] = None,
                 output_dir: Optional[str] = None,
                 progress_callback=None):
        """
        初始化模組化流水線
        
        Args:
            encoder_type: 編碼器類型 ('bert', 'gpt', 't5', 'cnn', 'elmo')
            aspect_type: 面向分類器類型 ('default', 'lda', 'bertopic', 'nmf')
            encoder_config: 編碼器配置
            aspect_config: 面向分類器配置
            output_dir: 輸出目錄
            progress_callback: 進度回調函數
        """
        # 創建編碼器和分類器
        text_encoder = EncoderFactory.create_encoder(encoder_type, encoder_config, progress_callback)
        aspect_classifier = AspectFactory.create_classifier(aspect_type, aspect_config, progress_callback)
        
        # 調用父類初始化
        config = {
            'encoder_type': encoder_type,
            'aspect_type': aspect_type,
            'encoder_config': encoder_config or {},
            'aspect_config': aspect_config or {}
        }
        super().__init__(text_encoder, aspect_classifier, config)
        
        self.output_dir = output_dir or self._create_output_dir()
        self.progress_callback = progress_callback
        self.run_manager = RunManager(self.output_dir)
        
        # 初始化其他組件
        self.preprocessor = TextPreprocessor(self.output_dir)
        self.sentiment_classifier = None
        
        # 記錄配置信息
        self.pipeline_config = {
            'encoder_type': encoder_type,
            'aspect_type': aspect_type,
            'encoder_config': encoder_config or {},
            'aspect_config': aspect_config or {},
            'timestamp': datetime.now().isoformat(),
            'output_dir': self.output_dir
        }
        
        logger.info(f"模組化流水線已初始化: {encoder_type} + {aspect_type}")
        
        if self.progress_callback:
            self.progress_callback('status', f'流水線已初始化: {encoder_type.upper()} + {aspect_type.upper()}')
    
    def _create_output_dir(self) -> str:
        """創建輸出目錄"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', f'run_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """執行完整的處理流水線"""
        results = {}
        start_time = time.time()
        
        try:
            # 1. 數據預處理
            if self.progress_callback:
                self.progress_callback('status', '開始文本預處理...')
            
            preprocessed_data = self._preprocess_data(data)
            results['preprocessed_data'] = preprocessed_data
            
            # 2. 文本編碼
            if self.progress_callback:
                self.progress_callback('status', f'開始{self.config["encoder_type"].upper()}編碼...')
            
            embeddings = self._encode_texts(preprocessed_data)
            results['embeddings'] = embeddings
            
            # 3. 面向分類
            if self.progress_callback:
                self.progress_callback('status', f'開始{self.config["aspect_type"].upper()}面向分類...')
            
            aspect_vectors, aspect_info = self._classify_aspects(embeddings, preprocessed_data)
            results['aspect_vectors'] = aspect_vectors
            results['aspect_info'] = aspect_info
            
            # 4. 情感分析（可選）
            if 'sentiment' in preprocessed_data.columns:
                if self.progress_callback:
                    self.progress_callback('status', '開始情感分析...')
                
                sentiment_results = self._analyze_sentiment(embeddings, preprocessed_data)
                results['sentiment_results'] = sentiment_results
            
            # 5. 保存結果
            if self.progress_callback:
                self.progress_callback('status', '保存結果...')
            
            self._save_results(results)
            
            # 計算總耗時
            total_time = time.time() - start_time
            results['processing_time'] = total_time
            results['pipeline_config'] = self.pipeline_config
            
            if self.progress_callback:
                self.progress_callback('status', f'流水線處理完成 (耗時: {total_time:.2f}秒)')
            
            logger.info(f"流水線處理完成，耗時: {total_time:.2f}秒")
            
        except Exception as e:
            logger.error(f"流水線處理失敗: {e}")
            if self.progress_callback:
                self.progress_callback('error', f'處理失敗: {str(e)}')
            raise
        
        return results
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """數據預處理"""
        # 確保有text欄位
        if 'text' not in data.columns:
            if 'review' in data.columns:
                data = data.rename(columns={'review': 'text'})
            elif 'comment' in data.columns:
                data = data.rename(columns={'comment': 'text'})
            else:
                raise ValueError("數據中必須包含 'text'、'review' 或 'comment' 欄位")
        
        # 使用文本預處理器
        preprocessed_data = self.preprocessor.preprocess_dataframe(data)
        
        # 保存預處理結果
        preprocessing_dir = os.path.join(self.output_dir, '01_preprocessing')
        os.makedirs(preprocessing_dir, exist_ok=True)
        
        output_path = os.path.join(preprocessing_dir, '01_preprocessed_data.csv')
        preprocessed_data.to_csv(output_path, index=False, encoding='utf-8')
        
        return preprocessed_data
    
    def _encode_texts(self, data: pd.DataFrame) -> np.ndarray:
        """文本編碼"""
        texts = data['text'].fillna('').astype(str)
        embeddings = self.text_encoder.encode(texts)
        
        # 保存編碼結果
        encoding_dir = os.path.join(self.output_dir, '02_encoding')
        os.makedirs(encoding_dir, exist_ok=True)
        
        embeddings_path = os.path.join(encoding_dir, f'02_{self.config["encoder_type"]}_embeddings.npy')
        np.save(embeddings_path, embeddings)
        
        # 保存編碼器信息
        encoder_info = self.text_encoder.get_encoder_info()
        info_path = os.path.join(encoding_dir, f'encoder_info_{self.config["encoder_type"]}.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(encoder_info, f, ensure_ascii=False, indent=2)
        
        return embeddings
    
    def _classify_aspects(self, embeddings: np.ndarray, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """面向分類"""
        aspect_vectors, aspect_info = self.aspect_classifier.fit_transform(embeddings, data)
        
        # 保存面向分類結果
        aspect_dir = os.path.join(self.output_dir, '03_aspect_classification')
        os.makedirs(aspect_dir, exist_ok=True)
        
        # 保存面向向量
        vectors_path = os.path.join(aspect_dir, f'03_{self.config["aspect_type"]}_aspect_vectors.npy')
        np.save(vectors_path, aspect_vectors)
        
        # 保存面向信息
        info_path = os.path.join(aspect_dir, f'aspect_info_{self.config["aspect_type"]}.json')
        
        # 轉換numpy數組為列表以便JSON序列化
        serializable_info = {}
        for key, value in aspect_info.items():
            if isinstance(value, np.ndarray):
                serializable_info[key] = value.tolist()
            elif hasattr(value, 'tolist'):
                serializable_info[key] = value.tolist()
            else:
                serializable_info[key] = value
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_info, f, ensure_ascii=False, indent=2)
        
        return aspect_vectors, aspect_info
    
    def _analyze_sentiment(self, embeddings: np.ndarray, data: pd.DataFrame) -> Dict:
        """情感分析（可選）"""
        if self.sentiment_classifier is None:
            classifier_config = {
                'classifier_type': 'xgboost',
                'use_gpu': True
            }
            self.sentiment_classifier = SentimentClassifier(
                output_dir=self.output_dir,
                config=classifier_config,
                progress_callback=self.progress_callback
            )
        
        # 訓練情感分類器
        X_train = embeddings
        y_train = data['sentiment'] if 'sentiment' in data.columns else None
        
        if y_train is not None:
            results = self.sentiment_classifier.train_and_evaluate(X_train, y_train)
            
            # 保存情感分析結果
            sentiment_dir = os.path.join(self.output_dir, '04_sentiment_analysis')
            os.makedirs(sentiment_dir, exist_ok=True)
            
            results_path = os.path.join(sentiment_dir, 'sentiment_results.json')
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            return results
        else:
            logger.warning("沒有情感標籤，跳過情感分析")
            return {}
    
    def _save_results(self, results: Dict):
        """保存完整結果"""
        # 創建完整結果文件
        complete_results = {
            'pipeline_info': self.get_pipeline_info(),
            'processing_summary': {
                'encoder_type': self.config['encoder_type'],
                'aspect_type': self.config['aspect_type'],
                'processing_time': results.get('processing_time', 0),
                'data_size': len(results.get('preprocessed_data', [])),
                'embedding_dim': self.text_encoder.get_embedding_dim(),
                'aspect_count': len(self.aspect_classifier.get_aspect_names()),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # 添加面向名稱
        if 'aspect_info' in results:
            complete_results['aspect_names'] = self.aspect_classifier.get_aspect_names()
        
        # 保存到根目錄
        results_path = os.path.join(self.output_dir, 'complete_pipeline_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(complete_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"完整結果已保存至: {results_path}")
    
    def get_available_combinations(self) -> Dict[str, Any]:
        """獲取可用的編碼器和分類器組合"""
        return {
            'encoders': {
                'available': EncoderFactory.get_available_encoders(),
                'info': {enc: EncoderFactory.get_encoder_info(enc) 
                        for enc in EncoderFactory.get_available_encoders()}
            },
            'aspect_classifiers': {
                'available': AspectFactory.get_available_classifiers(),
                'info': {cls: AspectFactory.get_classifier_info(cls) 
                        for cls in AspectFactory.get_available_classifiers()}
            },
            'recommended_combinations': [
                {'encoder': 'bert', 'aspect': 'default', 'scenario': '高準確率需求'},
                {'encoder': 'bert', 'aspect': 'bertopic', 'scenario': '自動主題發現'},
                {'encoder': 't5', 'aspect': 'lda', 'scenario': '平衡效果與效率'},
                {'encoder': 'cnn', 'aspect': 'nmf', 'scenario': '快速處理'},
                {'encoder': 'gpt', 'aspect': 'default', 'scenario': '生成式理解'}
            ]
        }
    
    def compare_methods(self, data: pd.DataFrame, 
                       encoder_types: List[str] = None,
                       aspect_types: List[str] = None) -> Dict[str, Any]:
        """比較不同方法的效果"""
        encoder_types = encoder_types or ['bert', 'gpt', 't5']
        aspect_types = aspect_types or ['default', 'lda', 'bertopic']
        
        comparison_results = {}
        
        for encoder_type in encoder_types:
            for aspect_type in aspect_types:
                try:
                    if self.progress_callback:
                        self.progress_callback('status', f'測試組合: {encoder_type} + {aspect_type}')
                    
                    # 創建臨時流水線
                    temp_pipeline = ModularPipeline(
                        encoder_type=encoder_type,
                        aspect_type=aspect_type,
                        progress_callback=self.progress_callback
                    )
                    
                    # 處理數據
                    start_time = time.time()
                    results = temp_pipeline.process(data.sample(min(1000, len(data))))
                    processing_time = time.time() - start_time
                    
                    # 記錄結果
                    combination_key = f"{encoder_type}_{aspect_type}"
                    comparison_results[combination_key] = {
                        'encoder_type': encoder_type,
                        'aspect_type': aspect_type,
                        'processing_time': processing_time,
                        'aspect_count': len(results.get('aspect_info', {}).get('aspect_names', [])),
                        'embedding_dim': temp_pipeline.text_encoder.get_embedding_dim(),
                        'success': True
                    }
                    
                except Exception as e:
                    logger.error(f"組合 {encoder_type}+{aspect_type} 失敗: {e}")
                    combination_key = f"{encoder_type}_{aspect_type}"
                    comparison_results[combination_key] = {
                        'encoder_type': encoder_type,
                        'aspect_type': aspect_type,
                        'error': str(e),
                        'success': False
                    }
        
        return comparison_results


def create_pipeline(encoder_type: str = 'bert',
                   aspect_type: str = 'default',
                   **kwargs) -> ModularPipeline:
    """
    便捷函數：創建模組化流水線
    
    Args:
        encoder_type: 編碼器類型
        aspect_type: 面向分類器類型
        **kwargs: 其他參數
        
    Returns:
        ModularPipeline: 流水線實例
    """
    return ModularPipeline(
        encoder_type=encoder_type,
        aspect_type=aspect_type,
        **kwargs
    )