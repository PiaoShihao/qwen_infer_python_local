#!/usr/bin/env python3
"""
图像美学分析器
专门用于处理图像美学评价任务的辅助模块
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class AestheticScore:
    """美学评分数据类"""
    composition: float  # 构图
    focal_length: float  # 焦段
    contrast_exposure_brightness: float  # 对比度&曝光度&亮度
    overall: float  # 综合评分
    
    def to_dict(self) -> Dict:
        return {
            "composition": self.composition,
            "focal_length": self.focal_length, 
            "contrast_exposure_brightness": self.contrast_exposure_brightness,
            "overall": self.overall
        }


@dataclass 
class AestheticAnalysis:
    """美学分析结果数据类"""
    composition_analysis: str
    focal_length_analysis: str
    contrast_exposure_brightness_analysis: str
    overall_evaluation: str
    suggestions: str
    scores: AestheticScore
    
    def to_dict(self) -> Dict:
        return {
            "composition_analysis": self.composition_analysis,
            "focal_length_analysis": self.focal_length_analysis,
            "contrast_exposure_brightness_analysis": self.contrast_exposure_brightness_analysis,
            "overall_evaluation": self.overall_evaluation,
            "suggestions": self.suggestions,
            "scores": self.scores.to_dict()
        }


class AestheticAnalyzer:
    """图像美学分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def parse_response(self, response: str) -> AestheticAnalysis:
        """
        解析模型响应，提取结构化的美学分析结果
        
        Args:
            response: 模型生成的响应文本
            
        Returns:
            结构化的美学分析结果
        """
        try:
            # 初始化默认值
            composition_analysis = ""
            focal_length_analysis = ""
            contrast_exposure_brightness_analysis = ""
            overall_evaluation = ""
            suggestions = ""
            
            # 默认评分
            composition_score = 0.0
            focal_length_score = 0.0
            contrast_exposure_brightness_score = 0.0
            overall_score = 0.0
            
            # 分析维度部分
            dimension_pattern = r"维度分析与评分：(.*?)综合评分："
            dimension_match = re.search(dimension_pattern, response, re.DOTALL)
            
            if dimension_match:
                dimension_content = dimension_match.group(1)
                
                # 提取构图分析
                composition_pattern = r"构图：(.*?)(?=焦段：|对比度|$)"
                composition_match = re.search(composition_pattern, dimension_content, re.DOTALL)
                if composition_match:
                    composition_text = composition_match.group(1).strip()
                    composition_analysis = self._clean_text(composition_text)
                    composition_score = self._extract_score(composition_text)
                
                # 提取焦段分析
                focal_pattern = r"焦段：(.*?)(?=对比度|$)"
                focal_match = re.search(focal_pattern, dimension_content, re.DOTALL)
                if focal_match:
                    focal_text = focal_match.group(1).strip()
                    focal_length_analysis = self._clean_text(focal_text)
                    focal_length_score = self._extract_score(focal_text)
                
                # 提取对比度&曝光度&亮度分析
                contrast_pattern = r"对比度&曝光度&亮度：(.*?)(?=$|\n\n)"
                contrast_match = re.search(contrast_pattern, dimension_content, re.DOTALL)
                if contrast_match:
                    contrast_text = contrast_match.group(1).strip()
                    contrast_exposure_brightness_analysis = self._clean_text(contrast_text)
                    contrast_exposure_brightness_score = self._extract_score(contrast_text)
            
            # 提取综合评分
            overall_score_pattern = r"综合评分：.*?(\d+\.?\d*)"
            overall_score_match = re.search(overall_score_pattern, response)
            if overall_score_match:
                overall_score = float(overall_score_match.group(1))
            
            # 提取综合评价与建议
            evaluation_pattern = r"综合评价与建议：(.*?)(?=$|\n\n\n)"
            evaluation_match = re.search(evaluation_pattern, response, re.DOTALL)
            if evaluation_match:
                evaluation_text = evaluation_match.group(1).strip()
                
                # 分离评价和建议
                if "建议" in evaluation_text:
                    parts = evaluation_text.split("建议", 1)
                    overall_evaluation = parts[0].strip()
                    suggestions = "建议" + parts[1].strip()
                else:
                    overall_evaluation = evaluation_text
            
            # 创建评分对象
            scores = AestheticScore(
                composition=composition_score,
                focal_length=focal_length_score,
                contrast_exposure_brightness=contrast_exposure_brightness_score,
                overall=overall_score
            )
            
            # 创建分析结果对象
            analysis = AestheticAnalysis(
                composition_analysis=composition_analysis,
                focal_length_analysis=focal_length_analysis,
                contrast_exposure_brightness_analysis=contrast_exposure_brightness_analysis,
                overall_evaluation=overall_evaluation,
                suggestions=suggestions,
                scores=scores
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"解析响应失败: {e}")
            # 返回默认分析结果
            return self._get_default_analysis()
    
    def _extract_score(self, text: str) -> float:
        """从文本中提取评分"""
        # 查找评分模式：X.X分 或 X分
        score_pattern = r"(\d+\.?\d*)分"
        matches = re.findall(score_pattern, text)
        
        if matches:
            try:
                return float(matches[-1])  # 取最后一个匹配的分数
            except ValueError:
                return 0.0
        
        return 0.0
    
    def _clean_text(self, text: str) -> str:
        """清理文本，移除多余的空白字符"""
        # 移除多余的空白字符和换行符
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return cleaned
    
    def _get_default_analysis(self) -> AestheticAnalysis:
        """获取默认的分析结果"""
        default_scores = AestheticScore(
            composition=5.0,
            focal_length=5.0,
            contrast_exposure_brightness=5.0,
            overall=5.0
        )
        
        return AestheticAnalysis(
            composition_analysis="构图分析暂不可用",
            focal_length_analysis="焦段分析暂不可用", 
            contrast_exposure_brightness_analysis="对比度&曝光度&亮度分析暂不可用",
            overall_evaluation="整体评价暂不可用",
            suggestions="建议暂不可用",
            scores=default_scores
        )
    
    def format_analysis_report(self, analysis: AestheticAnalysis, image_path: str) -> str:
        """
        格式化美学分析报告
        
        Args:
            analysis: 美学分析结果
            image_path: 图像路径
            
        Returns:
            格式化的报告文本
        """
        report = f"""
{'='*60}
图像美学分析报告
{'='*60}

分析图像: {image_path}

【维度分析与评分】

🎨 构图分析 (评分: {analysis.scores.composition}/10)
{analysis.composition_analysis}

📷 焦段分析 (评分: {analysis.scores.focal_length}/10) 
{analysis.focal_length_analysis}

💡 对比度&曝光度&亮度分析 (评分: {analysis.scores.contrast_exposure_brightness}/10)
{analysis.contrast_exposure_brightness_analysis}

【综合评价】
综合评分: {analysis.scores.overall}/10

{analysis.overall_evaluation}

【改进建议】
{analysis.suggestions}

{'='*60}
分析完成
{'='*60}
"""
        return report
    
    def save_analysis_json(self, analysis: AestheticAnalysis, output_path: str, image_path: str):
        """
        保存分析结果为 JSON 格式
        
        Args:
            analysis: 美学分析结果
            output_path: 输出文件路径
            image_path: 图像路径
        """
        try:
            result = {
                "image_path": image_path,
                "timestamp": "2024-01-01T00:00:00Z",  # 占位符时间戳
                "analysis": analysis.to_dict()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"分析结果已保存为 JSON: {output_path}")
            
        except Exception as e:
            self.logger.error(f"保存 JSON 失败: {e}")
    
    def generate_summary_report(self, analyses: List[Tuple[str, AestheticAnalysis]]) -> str:
        """
        生成多张图像的汇总报告
        
        Args:
            analyses: (图像路径, 分析结果) 的列表
            
        Returns:
            汇总报告文本
        """
        if not analyses:
            return "没有分析结果可供汇总"
        
        total_images = len(analyses)
        avg_overall = sum(analysis.scores.overall for _, analysis in analyses) / total_images
        avg_composition = sum(analysis.scores.composition for _, analysis in analyses) / total_images
        avg_focal = sum(analysis.scores.focal_length for _, analysis in analyses) / total_images
        avg_contrast = sum(analysis.scores.contrast_exposure_brightness for _, analysis in analyses) / total_images
        
        # 找出评分最高和最低的图像
        best_image = max(analyses, key=lambda x: x[1].scores.overall)
        worst_image = min(analyses, key=lambda x: x[1].scores.overall)
        
        report = f"""
{'='*60}
图像美学分析汇总报告
{'='*60}

分析图像总数: {total_images}

【平均评分】
- 综合评分: {avg_overall:.2f}/10
- 构图评分: {avg_composition:.2f}/10  
- 焦段评分: {avg_focal:.2f}/10
- 对比度&曝光度&亮度评分: {avg_contrast:.2f}/10

【最佳表现】
图像: {Path(best_image[0]).name}
综合评分: {best_image[1].scores.overall}/10

【需要改进】
图像: {Path(worst_image[0]).name}
综合评分: {worst_image[1].scores.overall}/10

【详细结果】
"""
        
        for i, (image_path, analysis) in enumerate(analyses, 1):
            report += f"""
{i}. {Path(image_path).name}
   综合评分: {analysis.scores.overall}/10
   构图: {analysis.scores.composition}/10 | 焦段: {analysis.scores.focal_length}/10 | 对比度&曝光度&亮度: {analysis.scores.contrast_exposure_brightness}/10
"""
        
        report += f"\n{'='*60}\n汇总完成\n{'='*60}\n"
        
        return report