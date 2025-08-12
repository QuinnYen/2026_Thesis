"""
分析結果報告生成器 - 將所有步驟的分析結果輸出為詳細的txt報告
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

class AnalysisReportGenerator:
    """分析結果報告生成器"""
    
    def __init__(self, output_dir: str = None):
        """初始化報告生成器"""
        self.output_dir = output_dir or "."
        self.report_filename = None
        
    def generate_comprehensive_report(self, 
                                   pipeline_results: Dict[str, Any],
                                   output_filename: str = None) -> str:
        """
        生成完整的分析結果報告
        
        Args:
            pipeline_results: 融合管線的完整結果
            output_filename: 輸出檔案名稱
            
        Returns:
            生成的報告檔案路徑
        """
        # 生成檔案名稱
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"情感分析_方案A_完整報告_{timestamp}.txt"
        
        self.report_filename = os.path.join(self.output_dir, output_filename)
        
        # 生成報告內容
        report_content = self._build_report_content(pipeline_results)
        
        # 寫入檔案
        with open(self.report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📝 完整分析報告已生成: {self.report_filename}")
        return self.report_filename
    
    def _build_report_content(self, results: Dict[str, Any]) -> str:
        """構建報告內容"""
        lines = []
        
        # 報告標題
        lines.append("=" * 100)
        lines.append("情感分析系統 - 方案A完整分析報告")
        lines.append("=" * 100)
        lines.append(f"報告生成時間: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        lines.append("")
        
        # 系統資訊
        lines.extend(self._generate_system_info(results))
        
        # 數據預處理結果
        lines.extend(self._generate_preprocessing_section(results))
        
        # 注意力機制分析
        lines.extend(self._generate_attention_analysis(results))
        
        # GFN門控融合分析
        lines.extend(self._generate_gfn_analysis(results))
        
        # 分類結果分析
        lines.extend(self._generate_classification_analysis(results))
        
        # 性能比較分析
        lines.extend(self._generate_performance_analysis(results))
        
        # 階段執行時間分析
        lines.extend(self._generate_timing_analysis(results))
        
        # 結論和建議
        lines.extend(self._generate_conclusions(results))
        
        return "\n".join(lines)
    
    def _generate_system_info(self, results: Dict[str, Any]) -> List[str]:
        """生成系統資訊部分"""
        lines = []
        lines.append("📋 系統配置資訊")
        lines.append("-" * 80)
        
        pipeline_info = results.get('pipeline_info', {})
        lines.append(f"架構類型: {pipeline_info.get('architecture', 'N/A')}")
        lines.append(f"編碼器: {pipeline_info.get('encoder_type', 'N/A').upper()}")
        lines.append(f"開始時間: {pipeline_info.get('start_time', 'N/A')}")
        
        # 數據基本資訊
        preprocessing = results.get('preprocessing_results', {})
        lines.append(f"原始數據規模: {preprocessing.get('original_shape', 'N/A')}")
        lines.append(f"處理後數據規模: {preprocessing.get('processed_shape', 'N/A')}")
        
        lines.append("")
        return lines
    
    def _generate_preprocessing_section(self, results: Dict[str, Any]) -> List[str]:
        """生成數據預處理結果部分"""
        lines = []
        lines.append("📊 數據預處理結果")
        lines.append("-" * 80)
        
        preprocessing = results.get('preprocessing_results', {})
        
        # 情感標籤分佈
        sentiment_encoding = preprocessing.get('sentiment_encoding', {})
        if sentiment_encoding:
            lines.append("情感標籤編碼分佈:")
            for label, count in sentiment_encoding.items():
                lines.append(f"  {label}: {count} 條記錄")
        
        # 文本預處理統計
        text_stats = preprocessing.get('text_statistics', {})
        if text_stats:
            lines.append("\n文本預處理統計:")
            lines.append(f"  平均文本長度: {text_stats.get('avg_length', 'N/A')}")
            lines.append(f"  最長文本: {text_stats.get('max_length', 'N/A')} 詞")
            lines.append(f"  最短文本: {text_stats.get('min_length', 'N/A')} 詞")
        
        lines.append("")
        return lines
    
    def _generate_attention_analysis(self, results: Dict[str, Any]) -> List[str]:
        """生成注意力機制分析部分"""
        lines = []
        lines.append("🧠 注意力機制分析")
        lines.append("-" * 80)
        
        # 查找注意力階段
        stages = results.get('pipeline_info', {}).get('stages', [])
        attention_stage = next((s for s in stages if s.get('stage') == 3), None)
        
        if attention_stage:
            lines.append("三種注意力機制並行計算結果:")
            
            # 特徵形狀
            feature_shapes = attention_stage.get('feature_shapes', {})
            for mechanism, shape in feature_shapes.items():
                lines.append(f"  {mechanism.capitalize()} Attention: {shape}")
            
            # 注意力機制詳細資訊
            attention_info = attention_stage.get('attention_info', {})
            if attention_info:
                lines.append(f"\n計算的注意力機制數量: {attention_info.get('mechanisms_computed', 0)}")
                
            lines.append(f"\n執行時間: {attention_stage.get('duration_seconds', 0):.2f} 秒")
        
        lines.append("")
        return lines
    
    def _generate_gfn_analysis(self, results: Dict[str, Any]) -> List[str]:
        """生成GFN門控融合分析部分"""
        lines = []
        lines.append("⚡ GFN門控融合網路分析")
        lines.append("-" * 80)
        
        # 查找GFN融合階段
        stages = results.get('pipeline_info', {}).get('stages', [])
        fusion_stage = next((s for s in stages if s.get('stage') == 4), None)
        
        if fusion_stage:
            fusion_info = fusion_stage.get('fusion_info', {})
            
            # 權重分配分析
            avg_weights = fusion_info.get('average_weights', {})
            if avg_weights:
                lines.append("智能權重學習結果:")
                total_weight = sum(avg_weights.values())
                
                for mechanism, weight in avg_weights.items():
                    percentage = (weight / total_weight * 100) if total_weight > 0 else 0
                    lines.append(f"  {mechanism.capitalize()}: {weight:.4f} ({percentage:.1f}%)")
                
                lines.append(f"  權重總和: {total_weight:.4f}")
                
                # 權重分析
                lines.append("\n權重分析:")
                max_mechanism = max(avg_weights, key=avg_weights.get)
                min_mechanism = min(avg_weights, key=avg_weights.get)
                lines.append(f"  主導機制: {max_mechanism.capitalize()} ({avg_weights[max_mechanism]:.4f})")
                lines.append(f"  次要機制: {min_mechanism.capitalize()} ({avg_weights[min_mechanism]:.4f})")
                
                # 權重均衡性分析
                weight_values = list(avg_weights.values())
                mean_weight = sum(weight_values) / len(weight_values)
                weight_variance = sum((w - mean_weight) ** 2 for w in weight_values) / len(weight_values)
                if weight_variance < 0.01:
                    balance_status = "高度均衡"
                elif weight_variance < 0.05:
                    balance_status = "適度均衡"
                else:
                    balance_status = "偏向特定機制"
                lines.append(f"  權重分佈: {balance_status} (方差: {weight_variance:.6f})")
            
            # 融合特徵資訊
            fused_shape = fusion_stage.get('fused_features_shape')
            if fused_shape:
                lines.append(f"\n融合特徵維度: {fused_shape}")
            
            lines.append(f"\n執行時間: {fusion_stage.get('duration_seconds', 0):.2f} 秒")
        
        lines.append("")
        return lines
    
    def _generate_classification_analysis(self, results: Dict[str, Any]) -> List[str]:
        """生成分類結果分析部分"""
        lines = []
        lines.append("🎯 分類結果分析")
        lines.append("-" * 80)
        
        # 查找分類階段
        stages = results.get('pipeline_info', {}).get('stages', [])
        classification_stage = next((s for s in stages if s.get('stage') == 5), None)
        
        if classification_stage:
            classification_results = classification_stage.get('classification_results', {})
            
            lines.append("模型性能指標:")
            metrics = [
                ('準確率', 'test_accuracy'),
                ('F1分數', 'test_f1'),
                ('精確度', 'test_precision'),
                ('召回率', 'test_recall')
            ]
            
            for metric_name, metric_key in metrics:
                value = classification_results.get(metric_key, 0)
                lines.append(f"  {metric_name}: {value:.4f} ({value*100:.2f}%)")
            
            lines.append(f"\n執行時間: {classification_stage.get('duration_seconds', 0):.2f} 秒")
        
        # 分類報告詳細資訊
        classification_full = results.get('classification_results', {})
        if classification_full:
            # 混淆矩陣
            confusion_matrix = classification_full.get('confusion_matrix')
            if confusion_matrix is not None:
                lines.append("\n混淆矩陣:")
                if isinstance(confusion_matrix, (list, tuple)):
                    for i, row in enumerate(confusion_matrix):
                        row_str = "  " + " ".join([f"{val:4d}" for val in row])
                        lines.append(row_str)
            
            # 分類報告
            classification_report = classification_full.get('classification_report_dict')
            if classification_report:
                lines.append("\n各類別詳細性能:")
                class_names = {0: '負面', 1: '中性', 2: '正面'}
                
                for class_idx in [0, 1, 2]:
                    if str(class_idx) in classification_report:
                        class_info = classification_report[str(class_idx)]
                        class_name = class_names.get(class_idx, f"類別{class_idx}")
                        lines.append(f"  {class_name}:")
                        lines.append(f"    精確度: {class_info.get('precision', 0):.4f}")
                        lines.append(f"    召回率: {class_info.get('recall', 0):.4f}")
                        lines.append(f"    F1分數: {class_info.get('f1-score', 0):.4f}")
                        lines.append(f"    支援樣本: {class_info.get('support', 0)}")
        
        lines.append("")
        return lines
    
    def _generate_performance_analysis(self, results: Dict[str, Any]) -> List[str]:
        """生成性能比較分析部分"""
        lines = []
        lines.append("📈 性能分析與比較")
        lines.append("-" * 80)
        
        # 查找準確率分析階段
        stages = results.get('pipeline_info', {}).get('stages', [])
        analysis_stage = next((s for s in stages if s.get('stage') == 6), None)
        
        if analysis_stage:
            accuracy_analysis = analysis_stage.get('accuracy_analysis', {})
            
            # 原始標籤vs預測標籤比較
            if accuracy_analysis:
                lines.append("原始標籤與預測結果比較:")
                lines.append(f"  整體一致性: 已完成分析")
                
            lines.append(f"\n執行時間: {analysis_stage.get('duration_seconds', 0):.2f} 秒")
        
        # 融合vs傳統方法比較
        fusion_results = results.get('fusion_comparison', {})
        if fusion_results:
            lines.append("\n方案A融合架構 vs 傳統方法:")
            lines.append(f"  GFN融合準確率: {fusion_results.get('fusion_accuracy', 0):.4f}")
            lines.append(f"  傳統方法準確率: {fusion_results.get('baseline_accuracy', 0):.4f}")
            improvement = fusion_results.get('improvement', 0)
            lines.append(f"  性能提升: +{improvement:.4f} ({improvement*100:.2f}%)")
        
        lines.append("")
        return lines
    
    def _generate_timing_analysis(self, results: Dict[str, Any]) -> List[str]:
        """生成時間分析部分"""
        lines = []
        lines.append("⏱️ 執行時間分析")
        lines.append("-" * 80)
        
        stages = results.get('pipeline_info', {}).get('stages', [])
        total_time = sum(stage.get('duration_seconds', 0) for stage in stages)
        
        lines.append("各階段執行時間:")
        for stage in stages:
            stage_num = stage.get('stage', 0)
            stage_name = stage.get('name', '未知階段')
            duration = stage.get('duration_seconds', 0)
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            
            lines.append(f"  階段{stage_num} - {stage_name}:")
            lines.append(f"    執行時間: {duration:.2f} 秒 ({percentage:.1f}%)")
        
        lines.append(f"\n總執行時間: {total_time:.2f} 秒")
        
        # 性能瓶頸分析
        if stages:
            slowest_stage = max(stages, key=lambda x: x.get('duration_seconds', 0))
            lines.append(f"最耗時階段: 階段{slowest_stage.get('stage')} - {slowest_stage.get('name')}")
            lines.append(f"耗時: {slowest_stage.get('duration_seconds', 0):.2f} 秒")
        
        lines.append("")
        return lines
    
    def _generate_conclusions(self, results: Dict[str, Any]) -> List[str]:
        """生成結論和建議部分"""
        lines = []
        lines.append("💡 結論與建議")
        lines.append("-" * 80)
        
        # 方案A架構優勢分析
        lines.append("方案A架構特點:")
        lines.append("  ✅ 三種注意力機制並行計算，充分捕捉文本特徵")
        lines.append("  ✅ GFN門控融合網路智能學習最優權重分配")
        lines.append("  ✅ 特徵對齊確保維度一致性，避免計算錯誤")
        lines.append("  ✅ End-to-End訓練，自動優化整體性能")
        
        # 性能評估
        stages = results.get('pipeline_info', {}).get('stages', [])
        classification_stage = next((s for s in stages if s.get('stage') == 5), None)
        
        if classification_stage:
            accuracy = classification_stage.get('classification_results', {}).get('test_accuracy', 0)
            if accuracy >= 0.9:
                performance_level = "優秀"
            elif accuracy >= 0.8:
                performance_level = "良好"
            elif accuracy >= 0.7:
                performance_level = "中等"
            else:
                performance_level = "待改進"
            
            lines.append(f"\n模型性能評估: {performance_level} (準確率: {accuracy:.4f})")
        
        # GFN權重分析建議
        fusion_stage = next((s for s in stages if s.get('stage') == 4), None)
        if fusion_stage:
            avg_weights = fusion_stage.get('fusion_info', {}).get('average_weights', {})
            if avg_weights:
                max_mechanism = max(avg_weights, key=avg_weights.get)
                lines.append(f"\nGFN學習結果: {max_mechanism.capitalize()} 注意力機制貢獻最大")
                
                weight_values = list(avg_weights.values())
                mean_weight = sum(weight_values) / len(weight_values) 
                weight_variance = sum((w - mean_weight) ** 2 for w in weight_values) / len(weight_values)
                if weight_variance > 0.05:
                    lines.append("建議: 權重分佈不均勻，可考慮調整注意力機制參數")
                else:
                    lines.append("權重分配合理，各注意力機制協調配合")
        
        # 改進建議
        lines.append("\n改進建議:")
        lines.append("  🔧 可嘗試調整GFN隱藏層維度以優化融合效果")
        lines.append("  🔧 考慮增加數據預處理步驟以提高文本質量")
        lines.append("  🔧 可實驗不同的注意力機制組合")
        lines.append("  🔧 適當調整訓練參數以避免過擬合")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("報告結束")
        lines.append("=" * 100)
        
        return lines