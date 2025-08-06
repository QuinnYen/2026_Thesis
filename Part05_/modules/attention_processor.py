"""
注意力處理器 - 負責執行注意力機制分析和比較
"""

import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from tqdm import tqdm

from .attention_analyzer import AttentionAnalyzer
from .bert_encoder import BertEncoder
from .run_manager import RunManager

# 匯入錯誤處理工具
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.error_handler import handle_error, handle_warning, handle_info
from utils.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class AttentionProcessor:
    """注意力機制處理器，用於執行完整的注意力分析流程"""
    
    def __init__(self, output_dir: Optional[str] = None, config: Optional[Dict] = None, progress_callback=None, encoder_type: str = 'bert'):
        """
        初始化注意力處理器
        
        Args:
            output_dir: 輸出目錄
            config: 配置參數
            progress_callback: 進度回調函數
            encoder_type: 編碼器類型 (bert, gpt, t5, cnn, elmo)
        """
        self.output_dir = output_dir
        self.config = config or {}
        self.progress_callback = progress_callback
        self.encoder_type = encoder_type
        self.run_manager = RunManager(output_dir) if output_dir else None
        
        # 初始化儲存管理器
        self.storage_manager = StorageManager(output_dir) if output_dir else None
        
        # 初始化組件
        self.bert_encoder = None
        self.attention_analyzer = None
        
        logger.info(f"注意力處理器已初始化，使用 {encoder_type.upper()} 編碼器")
        if self.storage_manager:
            logger.info("📁 儲存管理器已啟用，將優化檔案儲存")
    
    def process_with_attention(self, 
                             input_file: str,
                             attention_types: List[str] = None,
                             topics_path: Optional[str] = None,
                             attention_weights: Optional[Dict] = None,
                             save_results: bool = True) -> Dict[str, Any]:
        """
        執行完整的注意力機制分析流程
        
        Args:
            input_file: 輸入的預處理數據文件
            attention_types: 要測試的注意力機制類型
            topics_path: 關鍵詞文件路徑
            attention_weights: 組合注意力權重配置
            save_results: 是否保存結果
            
        Returns:
            Dict: 完整的分析結果
        """
        try:
            start_time = datetime.now()
            print("🔄 開始注意力機制分析流程")
            
            # 設定默認的注意力機制類型
            if attention_types is None:
                attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
            
            total_steps = 6  # 總共6個主要步驟
            
            # 1. 讀取預處理數據
            print(f"\n📊 步驟 1/{total_steps}: 讀取數據...")
            if self.progress_callback:
                self.progress_callback('phase', {
                    'phase_name': '讀取預處理數據',
                    'current_phase': 1,
                    'total_phases': total_steps
                })
                self.progress_callback('status', f'讀取數據: {input_file}')
            
            logger.info(f"讀取數據: {input_file}")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"找不到輸入文件: {input_file}")
            
            df = pd.read_csv(input_file)
            print(f"✅ 成功讀取 {len(df)} 條數據")
            logger.info(f"成功讀取 {len(df)} 條數據")
            
            # 創建輸入文件參考記錄（不複製原始數據集）
            if self.storage_manager:
                self.storage_manager.create_input_reference(input_file)
            
            if self.progress_callback:
                self.progress_callback('status', f'✅ 成功讀取 {len(df)} 條數據')
            
            # 2. 檢查必要欄位
            print(f"\n🔍 步驟 2/{total_steps}: 檢查數據欄位...")
            if self.progress_callback:
                self.progress_callback('phase', {
                    'phase_name': '檢查數據欄位',
                    'current_phase': 2,
                    'total_phases': total_steps
                })
            
            text_column = self._find_text_column(df)
            if text_column is None:
                raise ValueError("找不到有效的文本欄位")
            print(f"✅ 使用文本欄位: {text_column}")
            
            if self.progress_callback:
                self.progress_callback('status', f'✅ 使用文本欄位: {text_column}')
            
            # 3. 初始化編碼器和獲取特徵向量
            print(f"\n🤖 步驟 3/{total_steps}: 處理{self.encoder_type.upper()}特徵向量...")
            if self.progress_callback:
                self.progress_callback('phase', {
                    'phase_name': f'處理{self.encoder_type.upper()}特徵向量',
                    'current_phase': 3,
                    'total_phases': total_steps
                })
            
            embeddings = self._get_embeddings(df, text_column, self.encoder_type)
            print(f"✅ 特徵向量準備完成 (形狀: {embeddings.shape})")
            
            if self.progress_callback:
                self.progress_callback('status', f'✅ 特徵向量準備完成 (形狀: {embeddings.shape})')
            
            # 4. 準備元數據
            print(f"\n📋 步驟 4/{total_steps}: 準備元數據...")
            metadata = self._prepare_metadata(df)
            print(f"✅ 元數據準備完成")
            
            # 5. 初始化注意力分析器
            print(f"\n⚙️ 步驟 5/{total_steps}: 初始化注意力分析器...")
            if self.attention_analyzer is None:
                topic_labels_path = self._find_topic_labels_path()
                self.attention_analyzer = AttentionAnalyzer(
                    topic_labels_path=topic_labels_path,
                    config=self.config
                )
            print(f"✅ 注意力分析器已初始化")
            
            # 6. 執行注意力分析
            print(f"\n🔬 步驟 6/{total_steps}: 執行注意力機制分析...")
            print(f"將測試以下注意力機制: {', '.join(attention_types)}")
            
            results = self.attention_analyzer.analyze_with_attention(
                embeddings=embeddings,
                metadata=metadata,
                attention_types=attention_types,
                topics_path=topics_path,
                attention_weights=attention_weights
            )
            
            # 7. 添加處理信息
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            results['processing_info'] = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'processing_time_seconds': processing_time,
                'input_file': input_file,
                'data_samples': len(df),
                'text_column': text_column,
                'embeddings_shape': embeddings.shape,
                'attention_types_tested': attention_types or ['no', 'similarity', 'keyword', 'self', 'combined']
            }
            
            # 8. 保存結果
            if save_results and self.output_dir:
                print(f"\n💾 保存分析結果...")
                if self.storage_manager:
                    # 使用儲存管理器保存結果
                    self.storage_manager.save_analysis_results(
                        results, 
                        "attention_analysis", 
                        "attention_analysis_results.json"
                    )
                    # 生成摘要報告
                    summary = self.storage_manager.generate_summary_report()
                    print(f"📊 已生成儲存摘要報告，共 {summary['total_files']} 個文件")
                else:
                    self._save_analysis_results(results)
                print(f"✅ 結果已保存至: {self.output_dir}")
            
            print(f"\n🎉 注意力機制分析完成！")
            print(f"📊 總耗時: {processing_time:.2f} 秒")
            
            # 顯示結果摘要
            if 'comparison' in results and 'summary' in results['comparison']:
                summary = results['comparison']['summary']
                print(f"🏆 最佳注意力機制: {summary.get('best_mechanism', 'N/A')}")
                print(f"📈 最佳綜合得分: {summary.get('best_score', 0):.4f}")
            
            logger.info(f"注意力機制分析完成，耗時 {processing_time:.2f} 秒")
            return results
            
        except Exception as e:
            handle_error(e, "注意力機制分析", show_traceback=True)
            raise
    
    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """尋找有效的文本欄位"""
        text_columns = ['processed_text', 'clean_text', 'text', 'review', 'content']
        
        for col in text_columns:
            if col in df.columns and not df[col].isna().all():
                logger.info(f"使用文本欄位: {col}")
                return col
        
        return None
    
    def _get_embeddings(self, df: pd.DataFrame, text_column: str, encoder_type: str = 'bert') -> np.ndarray:
        """獲取或生成文本特徵向量，支援多種編碼器"""
        # 使用儲存管理器檢查已存在的特徵向量
        embeddings_file = None
        if self.storage_manager:
            existing_path = self.storage_manager.check_existing_embeddings(encoder_type)
            if existing_path:
                embeddings_file = existing_path
        elif self.output_dir:
            # 舊版本兼容：手動檢查編碼目錄
            if any(subdir in self.output_dir for subdir in ["01_preprocessing", "02_bert_encoding", "02_encoding", "03_attention_testing", "04_analysis"]):
                run_dir = os.path.dirname(self.output_dir)
            else:
                run_dir = self.output_dir
            
            # 使用儲存管理器獲取編碼器特定目錄
            try:
                from ..config.paths import get_path_config
                path_config = get_path_config()
                encoding_dir_name = path_config.get_subdirectory_name("encoding", encoder_type)
                encoding_dir = os.path.join(run_dir, encoding_dir_name)
                
                if os.path.exists(encoding_dir):
                    # 使用配置獲取正確的檔案模式
                    embeddings_pattern = path_config.get_file_pattern("embeddings", encoder_type)
                    candidate_file = os.path.join(encoding_dir, embeddings_pattern)
                    if os.path.exists(candidate_file):
                        embeddings_file = candidate_file
                        logger.info(f"找到{encoder_type.upper()}特徵向量檔案: {candidate_file}")
                        
            except Exception as e:
                logger.warning(f"使用配置檢查特徵向量時發生錯誤: {e}")
                # 回退到舊的檢查方式
                encoding_dirs = [
                    os.path.join(run_dir, f"02_{encoder_type}_encoding"),
                    os.path.join(run_dir, "02_bert_encoding")
                ]
                
                for encoding_dir in encoding_dirs:
                    if os.path.exists(encoding_dir):
                        possible_files = [
                            f"02_{encoder_type}_embeddings.npy",
                            "02_bert_embeddings.npy",
                            f"{encoder_type}_embeddings.npy",
                            "embeddings.npy"
                        ]
                        
                        for filename in possible_files:
                            candidate_file = os.path.join(encoding_dir, filename)
                            if os.path.exists(candidate_file):
                                embeddings_file = candidate_file
                                break
                        
                        if embeddings_file:
                            break
            
        # 如果當前目錄沒有，搜索所有run目錄中的特徵向量文件
        existing_embeddings_file = None
        if not (embeddings_file and os.path.exists(embeddings_file)):
            existing_embeddings_file = self._find_existing_embeddings(encoder_type)
            
        if embeddings_file and os.path.exists(embeddings_file):
            print(f"   📂 發現已存在的 {encoder_type.upper()} 特徵向量文件，正在載入...")
            logger.info(f"載入已存在的{encoder_type.upper()}特徵向量: {embeddings_file}")
            embeddings = np.load(embeddings_file)
            print(f"   ✅ {encoder_type.upper()}特徵向量載入完成 (形狀: {embeddings.shape})")
            logger.info(f"{encoder_type.upper()}特徵向量形狀: {embeddings.shape}")
        elif existing_embeddings_file:
            print(f"   📂 發現已存在的 {encoder_type.upper()} 特徵向量文件，正在載入...")
            print(f"   📁 來源: {existing_embeddings_file}")
            logger.info(f"載入已存在的{encoder_type.upper()}特徵向量: {existing_embeddings_file}")
            embeddings = np.load(existing_embeddings_file)
            print(f"   ✅ {encoder_type.upper()}特徵向量載入完成 (形狀: {embeddings.shape})")
            logger.info(f"{encoder_type.upper()}特徵向量形狀: {embeddings.shape}")
            
            # 將找到的特徵向量複製到當前編碼目錄，以便後續使用
            if self.output_dir and embeddings_file:
                try:
                    import shutil
                    os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
                    shutil.copy2(existing_embeddings_file, embeddings_file)
                    print(f"   📋 已複製{encoder_type.upper()}特徵向量到編碼目錄: {embeddings_file}")
                    logger.info(f"已複製{encoder_type.upper()}特徵向量到編碼目錄: {embeddings_file}")
                except Exception as e:
                    logger.warning(f"複製{encoder_type.upper()}特徵向量文件時發生錯誤: {str(e)}")
        else:
            # 生成新的特徵向量（支援多種編碼器）
            print(f"   🔄 未發現已存在的特徵向量，開始生成新的 {encoder_type.upper()} 特徵向量...")
            logger.info(f"生成 {encoder_type.upper()} 特徵向量...")
            
            # 根據編碼器類型選擇合適的編碼器
            if encoder_type == 'bert':
                if self.bert_encoder is None:
                    self.bert_encoder = BertEncoder(output_dir=self.output_dir)
                encoder = self.bert_encoder
            else:
                # 對於其他編碼器類型，使用模組化架構
                try:
                    from .text_encoders import TextEncoderFactory
                    encoder = TextEncoderFactory.create_encoder(encoder_type, output_dir=self.output_dir, progress_callback=self.progress_callback)
                except Exception as e:
                    # 如果模組化架構不可用，回退到BERT
                    print(f"   ⚠️ {encoder_type.upper()} 編碼器不可用 ({e})，回退使用BERT")
                    logger.warning(f"{encoder_type.upper()} 編碼器不可用，回退使用BERT: {e}")
                    if self.bert_encoder is None:
                        self.bert_encoder = BertEncoder(output_dir=self.output_dir)
                    encoder = self.bert_encoder
                    # 更新編碼器類型以保持一致性
                    self.encoder_type = 'bert'
                    encoder_type = 'bert'  # 更新區域變數
            
            embeddings = encoder.encode(df[text_column])
            print(f"   ✅ {encoder_type.upper()} 特徵向量生成完成 (形狀: {embeddings.shape})")
            logger.info(f"生成的特徵向量形狀: {embeddings.shape}")
        
        return embeddings
    
    def _find_existing_embeddings(self, encoder_type: str = 'bert') -> Optional[str]:
        """
        在所有run目錄中搜索已存在的文本特徵向量文件，支援多種編碼器
        
        Args:
            encoder_type: 編碼器類型 (bert, gpt, t5, cnn, elmo)
        
        Returns:
            Optional[str]: 找到的最新特徵向量文件路徑，如果沒找到則返回None
        """
        try:
            if not self.output_dir:
                return None
                
            # 獲取基礎輸出目錄（通常是output或包含output的目錄）
            base_dir = self.output_dir
            while base_dir and not base_dir.endswith('output'):
                parent_dir = os.path.dirname(base_dir)
                if parent_dir == base_dir:  # 已經到達根目錄
                    break
                base_dir = parent_dir
                
            # 如果沒找到output目錄，嘗試查找兄弟目錄中的output
            if not base_dir.endswith('output'):
                current_parent = os.path.dirname(self.output_dir)
                potential_output = os.path.join(current_parent, 'output')
                if os.path.exists(potential_output):
                    base_dir = potential_output
                else:
                    # 最後嘗試在Part05_目錄下查找output
                    part05_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    potential_output = os.path.join(part05_dir, 'output')
                    if os.path.exists(potential_output):
                        base_dir = potential_output
                    else:
                        logger.warning(f"無法找到output目錄，搜索路徑: {base_dir}")
                        return None
            
            logger.info(f"在目錄中搜索 {encoder_type.upper()} 特徵向量: {base_dir}")
            
            # 搜索所有run_*目錄
            embeddings_files = []
            if os.path.exists(base_dir):
                run_dirs = [item for item in os.listdir(base_dir) if item.startswith('run_')]
                logger.info(f"找到 {len(run_dirs)} 個run目錄: {run_dirs}")
                
                # 使用路徑配置系統
                try:
                    from ..config.paths import get_path_config
                    path_config = get_path_config()
                    
                    for item in run_dirs:
                        run_dir = os.path.join(base_dir, item)
                        if os.path.isdir(run_dir):
                            logger.info(f"檢查run目錄: {run_dir}")
                            
                            # 使用統一的編碼目錄結構
                            try:
                                encoding_dir_name = path_config.get_subdirectory_name("encoding")
                                encoding_dir = os.path.join(run_dir, encoding_dir_name)
                                logger.info(f"檢查編碼目錄: {encoding_dir} (存在: {os.path.exists(encoding_dir)})")
                                
                                if os.path.exists(encoding_dir):
                                    embeddings_pattern = path_config.get_file_pattern("embeddings", encoder_type)
                                    embeddings_file = os.path.join(encoding_dir, embeddings_pattern)
                                    logger.info(f"檢查文件: {embeddings_file} (存在: {os.path.exists(embeddings_file)})")
                                    
                                    if os.path.exists(embeddings_file):
                                        logger.info(f"驗證文件: {embeddings_file}, 編碼器類型: {encoder_type}")
                                        if self._validate_embeddings_file(embeddings_file, encoder_type):
                                            mtime = os.path.getmtime(embeddings_file)
                                            embeddings_files.append((embeddings_file, mtime))
                                            logger.info(f"✅ 找到並驗證 {encoder_type.upper()} 特徵向量: {embeddings_file}")
                                        else:
                                            logger.info(f"❌ 文件驗證失敗: {embeddings_file}")
                                
                            except Exception as e:
                                logger.warning(f"使用配置檢查時發生錯誤: {e}，使用舊方法")
                                # 回退到舊的檢查方式
                                encoding_dirs = [
                                    ('02_encoding', [
                                        f'02_{encoder_type}_embeddings.npy',
                                        f'{encoder_type}_embeddings.npy',
                                        'embeddings.npy'
                                    ]),
                                    ('02_bert_encoding', [
                                        f'02_{encoder_type}_embeddings.npy',
                                        f'{encoder_type}_embeddings.npy',
                                        '02_bert_embeddings.npy',
                                        'bert_embeddings.npy'
                                    ])
                                ]
                                
                                for dir_name, file_patterns in encoding_dirs:
                                    encoding_dir = os.path.join(run_dir, dir_name)
                                    logger.info(f"檢查編碼目錄: {encoding_dir} (存在: {os.path.exists(encoding_dir)})")
                                    if os.path.exists(encoding_dir):
                                        for pattern in file_patterns:
                                            embeddings_file = os.path.join(encoding_dir, pattern)
                                            logger.info(f"檢查文件: {embeddings_file} (存在: {os.path.exists(embeddings_file)})")
                                            if os.path.exists(embeddings_file):
                                                logger.info(f"驗證文件: {embeddings_file}")
                                                if self._validate_embeddings_file(embeddings_file, encoder_type):
                                                    mtime = os.path.getmtime(embeddings_file)
                                                    embeddings_files.append((embeddings_file, mtime))
                                                    logger.info(f"✅ 找到並驗證 {encoder_type.upper()} 特徵向量: {embeddings_file}")
                                                else:
                                                    logger.info(f"❌ 文件驗證失敗: {embeddings_file}")
                                                break
                    
                except ImportError as e:
                    logger.warning(f"無法導入路徑配置模組: {e}，使用舊方法")
                    # 完全回退到舊的檢查方式
                    for item in run_dirs:
                        run_dir = os.path.join(base_dir, item)
                        if os.path.isdir(run_dir):
                            logger.info(f"檢查run目錄: {run_dir}")
                            encoding_dirs = [
                                ('02_encoding', [
                                    f'02_{encoder_type}_embeddings.npy',
                                    f'{encoder_type}_embeddings.npy',
                                    'embeddings.npy'
                                ]),
                                ('02_bert_encoding', [
                                    f'02_{encoder_type}_embeddings.npy',
                                    f'{encoder_type}_embeddings.npy',
                                    '02_bert_embeddings.npy',
                                    'bert_embeddings.npy'
                                ])
                            ]
                            
                            for dir_name, file_patterns in encoding_dirs:
                                encoding_dir = os.path.join(run_dir, dir_name)
                                logger.info(f"檢查編碼目錄: {encoding_dir} (存在: {os.path.exists(encoding_dir)})")
                                if os.path.exists(encoding_dir):
                                    for pattern in file_patterns:
                                        embeddings_file = os.path.join(encoding_dir, pattern)
                                        logger.info(f"檢查文件: {embeddings_file} (存在: {os.path.exists(embeddings_file)})")
                                        if os.path.exists(embeddings_file):
                                            logger.info(f"驗證文件: {embeddings_file}")
                                            if self._validate_embeddings_file(embeddings_file, encoder_type):
                                                mtime = os.path.getmtime(embeddings_file)
                                                embeddings_files.append((embeddings_file, mtime))
                                                logger.info(f"✅ 找到並驗證 {encoder_type.upper()} 特徵向量: {embeddings_file}")
                                            else:
                                                logger.info(f"❌ 文件驗證失敗: {embeddings_file}")
                                            break
            
            # 如果找到文件，返回最新的
            if embeddings_files:
                # 按修改時間排序，返回最新的
                embeddings_files.sort(key=lambda x: x[1], reverse=True)
                latest_file = embeddings_files[0][0]
                logger.info(f"選擇最新的 {encoder_type.upper()} 特徵向量文件: {latest_file}")
                return latest_file
            else:
                logger.info(f"未找到任何已存在的 {encoder_type.upper()} 特徵向量文件")
                return None
                
        except Exception as e:
            logger.error(f"搜索已存在 {encoder_type.upper()} 特徵向量時發生錯誤: {str(e)}")
            return None
    
    def _validate_embeddings_file(self, file_path: str, encoder_type: str) -> bool:
        """驗證特徵向量檔案是否對應正確的編碼器類型"""
        try:
            # 基本檔案名檢查
            filename = os.path.basename(file_path)
            logger.info(f"驗證文件: {filename}, 編碼器類型: {encoder_type}")
            
            # 如果檔案名包含編碼器類型，直接檢查
            if encoder_type in filename.lower():
                logger.info(f"✅ 驗證通過: 檔案名包含編碼器類型 '{encoder_type}'")
                return True
            
            # 如果是舊的BERT檔案格式，只接受BERT編碼器
            if 'bert' in filename.lower() and encoder_type == 'bert':
                return True
            
            # 如果是通用檔案名，嘗試通過目錄結構或檔案內容推斷
            if filename in ['embeddings.npy', '02_embeddings.npy']:
                # 檢查同目錄下是否有編碼器信息檔案
                dir_path = os.path.dirname(file_path)
                info_files = [
                    f'encoder_info_{encoder_type}.json',
                    f'{encoder_type}_info.json',
                    'encoder_info.json'
                ]
                
                for info_file in info_files:
                    info_path = os.path.join(dir_path, info_file)
                    if os.path.exists(info_path):
                        try:
                            import json
                            with open(info_path, 'r', encoding='utf-8') as f:
                                info = json.load(f)
                                # 檢查encoder_type或從檔名推斷
                                if (info.get('encoder_type') == encoder_type or 
                                    encoder_type in info_file.lower()):
                                    return True
                        except:
                            pass
                
                # 如果沒有信息檔案，但是特定的編碼器目錄，也接受
                if '02_bert_encoding' in file_path and encoder_type == 'bert':
                    return True
                if '02_encoding' in file_path:
                    return True  # 新的模組化架構，通常是對應的編碼器
            
            return False
            
        except Exception as e:
            logger.warning(f"驗證特徵向量檔案時發生錯誤: {str(e)}")
            return True  # 如果驗證失敗，就接受檔案
    
    def _prepare_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """準備用於注意力分析的元數據"""
        metadata = df.copy()
        
        # 檢查是否有情感標籤
        if 'sentiment' not in metadata.columns:
            # 如果沒有情感標籤，可以嘗試從其他欄位推斷或創建假設標籤
            if 'label' in metadata.columns:
                metadata['sentiment'] = metadata['label']
            elif 'rating' in metadata.columns:
                # 基於評分創建情感標籤
                metadata['sentiment'] = metadata['rating'].apply(self._rating_to_sentiment)
            else:
                # 創建隨機情感標籤用於測試
                sentiments = ['positive', 'negative', 'neutral']
                metadata['sentiment'] = np.random.choice(sentiments, size=len(metadata))
                logger.warning("未找到情感標籤，已創建隨機標籤用於測試")
        
        logger.info(f"情感標籤分布: {metadata['sentiment'].value_counts().to_dict()}")
        return metadata
    
    def _rating_to_sentiment(self, rating):
        """將評分轉換為情感標籤"""
        if pd.isna(rating):
            return 'neutral'
        elif rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'negative'
        else:
            return 'neutral'
    
    def _find_topic_labels_path(self) -> Optional[str]:
        """尋找主題標籤文件"""
        possible_paths = [
            os.path.join(self.output_dir, "topic_labels.json") if self.output_dir else None,
            "utils/topic_labels.json",
            "Part05_/utils/topic_labels.json"
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                logger.info(f"找到主題標籤文件: {path}")
                return path
        
        logger.warning("未找到主題標籤文件")
        return None
    
    def _save_analysis_results(self, results: Dict[str, Any]):
        """保存分析結果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存完整結果
            results_file = os.path.join(self.output_dir, f"03_attention_analysis_{timestamp}.json")
            self.attention_analyzer.save_results(results, results_file)
            
            # 保存簡化的比較報告
            if 'comparison' in results:
                report_file = os.path.join(self.output_dir, f"03_attention_comparison_{timestamp}.json")
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump(results['comparison'], f, ensure_ascii=False, indent=2)
                
                logger.info(f"比較報告已保存: {report_file}")
            
            # 保存面向向量（如果存在）
            for attention_type, result in results.items():
                if attention_type in ['processing_info', 'comparison']:
                    continue
                    
                if 'aspect_vectors' in result:
                    vectors_file = os.path.join(self.output_dir, f"03_aspect_vectors_{attention_type}_{timestamp}.npy")
                    
                    # 轉換為數組格式保存
                    aspect_vectors = result['aspect_vectors']
                    if aspect_vectors:
                        vector_array = np.array(list(aspect_vectors.values()))
                        np.save(vectors_file, vector_array)
                        logger.info(f"面向向量已保存: {vectors_file}")
            
        except Exception as e:
            logger.error(f"保存結果時發生錯誤: {str(e)}")
    
    def compare_attention_mechanisms(self, 
                                   attention_types: List[str] = None,
                                   input_file: str = None,
                                   topics_path: Optional[str] = None) -> Dict[str, Any]:
        """
        專門用於比較不同注意力機制效果的方法
        
        Args:
            attention_types: 要比較的注意力機制類型
            input_file: 輸入數據文件
            topics_path: 關鍵詞文件路徑
            
        Returns:
            Dict: 比較結果
        """
        if attention_types is None:
            attention_types = ['no', 'similarity', 'keyword', 'self', 'combined']
        
        logger.info(f"開始比較 {len(attention_types)} 種注意力機制")
        
        # 執行分析
        results = self.process_with_attention(
            input_file=input_file,
            attention_types=attention_types,
            topics_path=topics_path,
            save_results=True
        )
        
        # 提取比較結果
        comparison = results.get('comparison', {})
        
        # 生成詳細比較報告
        report = self._generate_comparison_report(results, attention_types)
        
        return {
            'comparison': comparison,
            'detailed_report': report,
            'processing_info': results.get('processing_info', {})
        }
    
    def _generate_comparison_report(self, results: Dict[str, Any], attention_types: List[str]) -> Dict[str, Any]:
        """生成詳細的比較報告"""
        report = {
            'summary': {},
            'detailed_metrics': {},
            'recommendations': []
        }
        
        # 收集指標
        metrics_data = []
        for attention_type in attention_types:
            if attention_type in results:
                metrics = results[attention_type].get('metrics', {})
                metrics_data.append({
                    'type': attention_type,
                    'coherence': metrics.get('coherence', 0),
                    'separation': metrics.get('separation', 0),
                    'combined_score': metrics.get('combined_score', 0)
                })
        
        if metrics_data:
            # 找出最佳機制
            best_combined = max(metrics_data, key=lambda x: x['combined_score'])
            best_coherence = max(metrics_data, key=lambda x: x['coherence'])
            best_separation = max(metrics_data, key=lambda x: x['separation'])
            
            report['summary'] = {
                'best_overall': best_combined['type'],
                'best_coherence': best_coherence['type'],
                'best_separation': best_separation['type'],
                'total_mechanisms_tested': len(metrics_data)
            }
            
            report['detailed_metrics'] = metrics_data
            
            # 生成建議
            if best_combined['type'] == 'combined':
                report['recommendations'].append("組合注意力機制表現最佳，建議在生產環境中使用")
            elif best_combined['type'] == 'similarity':
                report['recommendations'].append("相似度注意力機制表現最佳，適合語義相似性重要的任務")
            elif best_combined['type'] == 'keyword':
                report['recommendations'].append("關鍵詞注意力機制表現最佳，適合特定術語重要的任務")
            elif best_combined['type'] == 'self':
                report['recommendations'].append("自注意力機制表現最佳，適合文檔間關係複雜的任務")
            
            # 性能差異分析
            scores = [item['combined_score'] for item in metrics_data]
            score_std = np.std(scores)
            if score_std < 0.05:
                report['recommendations'].append("各機制性能差異較小，可考慮計算成本選擇簡單機制")
            else:
                report['recommendations'].append("各機制性能差異明顯，建議選擇最佳性能的機制")
        
        return report
    
    def load_previous_results(self, results_file: str) -> Dict[str, Any]:
        """載入之前的分析結果"""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logger.info(f"成功載入結果: {results_file}")
            return results
            
        except Exception as e:
            logger.error(f"載入結果失敗: {str(e)}")
            return {}
    
    def export_comparison_report(self, results: Dict[str, Any], output_file: str):
        """匯出比較報告為可讀格式"""
        try:
            comparison = results.get('comparison', {})
            
            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("注意力機制比較報告")
            report_lines.append("=" * 60)
            report_lines.append("")
            
            # 總體摘要
            if 'summary' in comparison:
                summary = comparison['summary']
                report_lines.append("總體摘要:")
                report_lines.append(f"  最佳機制: {summary.get('best_mechanism', 'N/A')}")
                report_lines.append(f"  最佳得分: {summary.get('best_score', 0):.4f}")
                report_lines.append(f"  測試機制數: {summary.get('total_mechanisms', 0)}")
                report_lines.append("")
            
            # 詳細排名
            rankings = ['coherence_ranking', 'separation_ranking', 'combined_ranking']
            ranking_names = ['內聚度排名', '分離度排名', '綜合得分排名']
            
            for ranking, name in zip(rankings, ranking_names):
                if ranking in comparison:
                    report_lines.append(f"{name}:")
                    for i, (mechanism, score) in enumerate(comparison[ranking], 1):
                        report_lines.append(f"  {i}. {mechanism}: {score:.4f}")
                    report_lines.append("")
            
            # 寫入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"比較報告已匯出: {output_file}")
            
        except Exception as e:
            logger.error(f"匯出報告失敗: {str(e)}")


# 測試代碼
if __name__ == "__main__":
    # 配置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 測試注意力處理器
    processor = AttentionProcessor()
    
    # 創建測試數據
    test_data = pd.DataFrame({
        'text': [
            'This is a great product!',
            'I love this item very much.',
            'Not bad, could be better.',
            'Terrible quality, very disappointed.',
            'Worst purchase ever made.'
        ],
        'sentiment': ['positive', 'positive', 'neutral', 'negative', 'negative']
    })
    
    test_file = 'test_data.csv'
    test_data.to_csv(test_file, index=False)
    
    try:
        # 執行比較
        results = processor.compare_attention_mechanisms(
            input_file=test_file,
            attention_types=['no', 'similarity', 'combined']
        )
        
        print("比較完成！")
        print(f"最佳機制: {results['comparison']['summary']['best_mechanism']}")
        
        # 清理測試文件
        os.remove(test_file)
        
    except Exception as e:
        print(f"測試失敗: {str(e)}")
        if os.path.exists(test_file):
            os.remove(test_file) 