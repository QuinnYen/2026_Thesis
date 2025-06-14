#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
儲存管理工具
統一管理run資料夾內的檔案儲存，避免重複儲存原始數據集
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import shutil
import logging

logger = logging.getLogger(__name__)

class StorageManager:
    """
    儲存管理器
    確保所有操作結果都正確儲存到run資料夾內，不重複儲存原始數據集
    """
    
    def __init__(self, run_dir: str):
        """
        初始化儲存管理器
        
        Args:
            run_dir: run資料夾路徑
        """
        self.run_dir = Path(run_dir)
        self.subdirs = {
            'preprocessing': self.run_dir / '01_preprocessing',
            'encoding': self.run_dir / '02_bert_encoding',
            'attention': self.run_dir / '03_attention_testing',
            'analysis': self.run_dir / '04_analysis'
        }
        
        # 創建所有子目錄
        self._ensure_directories()
        
        # 初始化文件記錄
        self.file_registry = {}
        self._load_file_registry()
    
    def _ensure_directories(self):
        """確保所有必要的目錄存在"""
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 已確保run目錄結構存在: {self.run_dir}")
    
    def _load_file_registry(self):
        """載入文件記錄"""
        registry_file = self.run_dir / 'file_registry.json'
        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    self.file_registry = json.load(f)
            except Exception as e:
                logger.warning(f"載入文件記錄失敗: {e}")
                self.file_registry = {}
        else:
            self.file_registry = {}
    
    def _save_file_registry(self):
        """儲存文件記錄"""
        registry_file = self.run_dir / 'file_registry.json'
        try:
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.file_registry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"儲存文件記錄失敗: {e}")
    
    def save_processed_data(self, data: pd.DataFrame, stage: str, 
                          filename: str, metadata: Optional[Dict] = None) -> str:
        """
        儲存處理過的數據（而非原始數據集）
        
        Args:
            data: 處理過的數據
            stage: 處理階段 ('preprocessing', 'encoding', 'attention', 'analysis')
            filename: 檔案名稱
            metadata: 額外的元數據信息
            
        Returns:
            str: 儲存的檔案路徑
        """
        if stage not in self.subdirs:
            raise ValueError(f"無效的處理階段: {stage}")
        
        # 只儲存處理後的結果，不儲存原始數據
        output_path = self.subdirs[stage] / filename
        
        # 儲存數據
        data.to_csv(output_path, index=False, encoding='utf-8')
        
        # 記錄文件信息
        file_info = {
            'path': str(output_path),
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'rows': len(data),
            'columns': list(data.columns),
            'file_size': output_path.stat().st_size,
            'metadata': metadata or {}
        }
        
        self.file_registry[filename] = file_info
        self._save_file_registry()
        
        logger.info(f"✅ 已儲存處理數據到: {output_path} ({len(data)} 行)")
        return str(output_path)
    
    def save_embeddings(self, embeddings: np.ndarray, encoder_type: str, 
                       metadata: Optional[Dict] = None) -> str:
        """
        儲存特徵向量
        
        Args:
            embeddings: 特徵向量矩陣
            encoder_type: 編碼器類型
            metadata: 元數據
            
        Returns:
            str: 儲存的檔案路徑
        """
        filename = f"02_{encoder_type}_embeddings.npy"
        output_path = self.subdirs['encoding'] / filename
        
        # 儲存特徵向量
        np.save(output_path, embeddings)
        
        # 記錄文件信息
        file_info = {
            'path': str(output_path),
            'stage': 'encoding',
            'encoder_type': encoder_type,
            'timestamp': datetime.now().isoformat(),
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'file_size': output_path.stat().st_size,
            'metadata': metadata or {}
        }
        
        self.file_registry[filename] = file_info
        self._save_file_registry()
        
        logger.info(f"✅ 已儲存{encoder_type.upper()}特徵向量: {output_path} {embeddings.shape}")
        return str(output_path)
    
    def save_analysis_results(self, results: Dict[str, Any], 
                            analysis_type: str, filename: str) -> str:
        """
        儲存分析結果
        
        Args:
            results: 分析結果
            analysis_type: 分析類型
            filename: 檔案名稱
            
        Returns:
            str: 儲存的檔案路徑
        """
        # 根據分析類型選擇目錄
        if 'attention' in analysis_type.lower():
            output_path = self.subdirs['attention'] / filename
            stage = 'attention'
        else:
            output_path = self.subdirs['analysis'] / filename
            stage = 'analysis'
        
        # 儲存JSON結果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # 記錄文件信息
        file_info = {
            'path': str(output_path),
            'stage': stage,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'file_size': output_path.stat().st_size,
            'keys': list(results.keys()) if isinstance(results, dict) else None
        }
        
        self.file_registry[filename] = file_info
        self._save_file_registry()
        
        logger.info(f"✅ 已儲存{analysis_type}分析結果: {output_path}")
        return str(output_path)
    
    def create_input_reference(self, original_file_path: str) -> str:
        """
        創建原始輸入文件的參考記錄（不複製文件）
        
        Args:
            original_file_path: 原始文件路徑
            
        Returns:
            str: 參考記錄路徑
        """
        original_path = Path(original_file_path)
        
        # 創建參考文件而不是複製原始文件
        reference_info = {
            'original_path': str(original_path.absolute()),
            'filename': original_path.name,
            'file_size': original_path.stat().st_size if original_path.exists() else 0,
            'timestamp': datetime.now().isoformat(),
            'note': '此為原始數據集的參考記錄，未複製實際文件以節省空間'
        }
        
        reference_file = self.run_dir / 'input_reference.json'
        with open(reference_file, 'w', encoding='utf-8') as f:
            json.dump(reference_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 已創建輸入文件參考: {original_file_path}")
        return str(reference_file)
    
    def check_existing_embeddings(self, encoder_type: str) -> Optional[str]:
        """
        檢查是否已存在相同編碼器的特徵向量
        
        Args:
            encoder_type: 編碼器類型
            
        Returns:
            Optional[str]: 已存在的特徵向量文件路徑，如果不存在則返回None
        """
        filename = f"02_{encoder_type}_embeddings.npy"
        
        # 先檢查當前run目錄
        current_path = self.subdirs['encoding'] / filename
        if current_path.exists():
            logger.info(f"🔍 發現當前run中的{encoder_type.upper()}特徵向量: {current_path}")
            return str(current_path)
        
        # 檢查其他run目錄（如果需要的話）
        parent_dir = self.run_dir.parent
        if parent_dir.exists():
            for run_folder in parent_dir.glob('run_*'):
                if run_folder.is_dir() and run_folder != self.run_dir:
                    other_embedding = run_folder / '02_bert_encoding' / filename
                    if other_embedding.exists():
                        logger.info(f"🔍 在其他run中發現{encoder_type.upper()}特徵向量: {other_embedding}")
                        # 可以選擇複製到當前run或直接使用
                        return str(other_embedding)
        
        return None
    
    def get_file_info(self, filename: str) -> Optional[Dict]:
        """
        獲取文件信息
        
        Args:
            filename: 檔案名稱
            
        Returns:
            Optional[Dict]: 文件信息，如果不存在則返回None
        """
        return self.file_registry.get(filename)
    
    def list_files_by_stage(self, stage: str) -> List[Dict]:
        """
        按階段列出文件
        
        Args:
            stage: 處理階段
            
        Returns:
            List[Dict]: 該階段的文件列表
        """
        return [info for info in self.file_registry.values() 
                if info.get('stage') == stage]
    
    def cleanup_temp_files(self):
        """清理臨時文件"""
        temp_patterns = ['temp_*', '*.tmp', '*_temp.*']
        cleaned_files = []
        
        for pattern in temp_patterns:
            for temp_file in self.run_dir.rglob(pattern):
                if temp_file.is_file():
                    try:
                        temp_file.unlink()
                        cleaned_files.append(str(temp_file))
                    except Exception as e:
                        logger.warning(f"清理臨時文件失敗 {temp_file}: {e}")
        
        if cleaned_files:
            logger.info(f"🧹 已清理 {len(cleaned_files)} 個臨時文件")
        
        return cleaned_files
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        生成run資料夾的摘要報告
        
        Returns:
            Dict[str, Any]: 摘要報告
        """
        summary = {
            'run_directory': str(self.run_dir),
            'created_time': datetime.now().isoformat(),
            'total_files': len(self.file_registry),
            'stages': {},
            'total_size_mb': 0,
            'file_details': []
        }
        
        # 計算各階段的統計信息
        for stage in self.subdirs.keys():
            stage_files = self.list_files_by_stage(stage)
            summary['stages'][stage] = {
                'file_count': len(stage_files),
                'files': [f['path'] for f in stage_files]
            }
        
        # 計算總大小
        total_size = 0
        for file_info in self.file_registry.values():
            file_size = file_info.get('file_size', 0)
            total_size += file_size
            summary['file_details'].append({
                'filename': Path(file_info['path']).name,
                'stage': file_info.get('stage'),
                'size_mb': round(file_size / (1024 * 1024), 2),
                'timestamp': file_info.get('timestamp')
            })
        
        summary['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        
        # 儲存摘要報告
        summary_file = self.run_dir / 'storage_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 已生成儲存摘要報告: {summary_file}")
        return summary
    
    def verify_storage_integrity(self) -> Dict[str, Any]:
        """
        驗證儲存完整性
        
        Returns:
            Dict[str, Any]: 驗證結果
        """
        verification = {
            'verified_files': [],
            'missing_files': [],
            'corrupted_files': [],
            'total_verified': 0,
            'verification_time': datetime.now().isoformat()
        }
        
        for filename, file_info in self.file_registry.items():
            file_path = Path(file_info['path'])
            
            if not file_path.exists():
                verification['missing_files'].append({
                    'filename': filename,
                    'expected_path': str(file_path),
                    'stage': file_info.get('stage')
                })
                continue
            
            # 檢查文件大小
            try:
                current_size = file_path.stat().st_size
                expected_size = file_info.get('file_size', 0)
                
                if current_size != expected_size:
                    verification['corrupted_files'].append({
                        'filename': filename,
                        'path': str(file_path),
                        'expected_size': expected_size,
                        'current_size': current_size
                    })
                else:
                    verification['verified_files'].append({
                        'filename': filename,
                        'path': str(file_path),
                        'stage': file_info.get('stage')
                    })
                    verification['total_verified'] += 1
                    
            except Exception as e:
                verification['corrupted_files'].append({
                    'filename': filename,
                    'path': str(file_path),
                    'error': str(e)
                })
        
        # 儲存驗證結果
        verification_file = self.run_dir / 'storage_verification.json'
        with open(verification_file, 'w', encoding='utf-8') as f:
            json.dump(verification, f, ensure_ascii=False, indent=2)
        
        logger.info(f"🔍 儲存驗證完成: {verification['total_verified']} 個文件正常")
        if verification['missing_files']:
            logger.warning(f"⚠️ 發現 {len(verification['missing_files'])} 個遺失文件")
        if verification['corrupted_files']:
            logger.warning(f"⚠️ 發現 {len(verification['corrupted_files'])} 個損壞文件")
        
        return verification 