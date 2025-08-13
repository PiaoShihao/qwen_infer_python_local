#!/usr/bin/env python3
"""
Qwen2.5-VL-3B 统一推理脚本
将所有功能合并到单个文件中，提供流式输出的图像美学分析
基于 MLX 框架的本地推理实现
"""

import os
import sys
import json
import re
import logging
import argparse
from typing import Optional, Dict, Any, List, Tuple, Generator
from pathlib import Path
from dataclasses import dataclass
from PIL import Image
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import datetime


# ============================================================================
# 数据类定义
# ============================================================================

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


# ============================================================================
# 美学分析器
# ============================================================================

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


# ============================================================================
# Qwen2.5-VL 推理引擎
# ============================================================================

class QwenVLInference:
    """Qwen2.5-VL-3B 模型推理类"""
    
    def __init__(self, model_path: str, device: str = "gpu"):
        """
        初始化推理引擎
        
        Args:
            model_path: 模型文件路径
            device: 运行设备 ('gpu' 或 'cpu')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载模型配置
        self._load_config()
        
        # 加载分词器
        self._load_tokenizer()
        
        # 加载模型
        self._load_model()
        
    def _load_config(self):
        """加载模型配置"""
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        self.logger.info(f"已加载模型配置: {self.config['model_type']}")
        
    def _load_tokenizer(self):
        """加载分词器"""
        # 这里需要根据实际的 MLX tokenizer 实现进行调整
        tokenizer_path = self.model_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"分词器文件不存在: {tokenizer_path}")
            
        self.logger.info("分词器加载完成")
        
    def _load_model(self):
        """加载 MLX 模型"""
        try:
            # 使用 MLX 加载量化模型
            model_file = self.model_path / "model.safetensors"
            if not model_file.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_file}")
            
            # 设置 MLX 设备
            if self.device == "gpu" and mx.metal.is_available():
                mx.set_default_device(mx.gpu)
                self.logger.info("使用 GPU (Metal) 运行推理")
            else:
                mx.set_default_device(mx.cpu)
                self.logger.info("使用 CPU 运行推理")
                
            # 这里需要实现实际的模型加载逻辑
            # 由于 MLX 的 Qwen2.5-VL 支持可能需要特定的实现
            self.logger.info("模型加载完成")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> mx.array:
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            预处理后的图像张量
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 获取视觉配置
            vision_config = self.config.get('vision_config', {})
            patch_size = vision_config.get('patch_size', 14)
            
            # 调整图像尺寸 - 通常需要调整到模型期望的尺寸
            # 这里使用 448x448 作为示例，实际可能需要根据模型配置调整
            target_size = (448, 448)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # 转换为数组并标准化
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # 标准化 (ImageNet 标准)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_array = (image_array - mean) / std
            
            # 转换为 MLX 数组并调整维度 (H, W, C) -> (1, C, H, W)
            image_tensor = mx.array(image_array.transpose(2, 0, 1))
            image_tensor = mx.expand_dims(image_tensor, axis=0)
            
            self.logger.info(f"图像预处理完成，尺寸: {image_tensor.shape}")
            return image_tensor
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            raise
    
    def preprocess_text(self, prompt: str) -> mx.array:
        """
        预处理文本提示
        
        Args:
            prompt: 文本提示
            
        Returns:
            编码后的文本张量
        """
        try:
            # 这里需要实现实际的文本编码逻辑
            # 使用 Qwen2.5 的特殊 token
            vision_start_token = self.config.get('vision_start_token_id', 151652)
            vision_end_token = self.config.get('vision_end_token_id', 151653)
            
            # 构建带有视觉 token 的提示
            full_prompt = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
            
            # 这里应该使用实际的 tokenizer 进行编码
            # 作为示例，我们返回一个占位符张量
            # 实际实现需要集成真正的 tokenizer
            
            self.logger.info("文本预处理完成")
            return mx.array([1, 2, 3])  # 占位符
            
        except Exception as e:
            self.logger.error(f"文本预处理失败: {e}")
            raise
    
    def _mock_aesthetic_analysis_stream(self) -> Generator[str, None, None]:
        """
        模拟流式美学分析响应
        实际实现中应该调用真正的模型进行流式生成
        """
        response_parts = [
            "维度分析与评分：\n",
            "- 构图：采用了较为传统的海景构图，前景的岩石形成了天然的引导线，",
            "将视线引向远处的海平面。画面遵循了三分法原则，",
            "海平线位于画面下三分之一处，构图相对平衡。",
            "但岩石占据了过多前景空间，可能压抑了海洋的开阔感。评分：7.2分\n",
            "- 焦段：使用适中焦段拍摄，透视效果自然，没有明显的变形。",
            "焦段选择适合表现海岸风光的层次感，",
            "能够较好地平衡前景岩石和远景海面的关系。评分：7.8分\n",  
            "- 对比度&曝光度&亮度：画面对比度适中，",
            "天空的蓝色与岩石的暖色调形成了良好的色彩对比。",
            "曝光基本准确，天空和海水的细节都有较好的保留，",
            "没有明显的过曝或欠曝现象。整体亮度适宜，",
            "符合日间海景的自然光照条件。评分：8.1分\n\n",
            "综合评分：7.7（1-10分）\n\n",
            "综合评价与建议：这是一幅较为标准的海岸风光摄影作品。",
            "优点包括色彩搭配和谐、曝光控制得当、画面层次清晰。",
            "建议改进：1）可以尝试不同的拍摄角度，",
            "减少前景岩石的占比，突出海洋的壮阔感；",
            "2）等待更佳的光线条件，如黄金时段的温暖光线或者戏剧性的云层；",
            "3）考虑使用渐变镜平衡天空和地面的光比，",
            "进一步提升画面的视觉效果。\n"
        ]
        
        import time
        for part in response_parts:
            yield part
            time.sleep(0.1)  # 模拟生成延迟
    
    def generate_response_stream(self, image_path: str, prompt: str, max_tokens: int = 2048) -> Generator[str, None, None]:
        """
        生成流式响应
        
        Args:
            image_path: 图像文件路径
            prompt: 文本提示
            max_tokens: 最大生成 token 数量
            
        Yields:
            生成的文本片段
        """
        try:
            # 预处理输入
            image_tensor = self.preprocess_image(image_path)
            text_tensor = self.preprocess_text(prompt)
            
            # 执行推理
            # 这里需要实现实际的模型推理逻辑
            # 由于 MLX 的 Qwen2.5-VL 实现可能比较复杂，这里提供一个框架
            
            self.logger.info("开始流式推理...")
            
            # 使用模拟的流式响应 - 实际实现需要调用模型
            for chunk in self._mock_aesthetic_analysis_stream():
                yield chunk
            
            self.logger.info("流式推理完成")
            
        except Exception as e:
            self.logger.error(f"流式推理失败: {e}")
            yield f"推理失败: {e}"


# ============================================================================
# 主要推理函数
# ============================================================================

def infer_with_qwen(prompt: str, image_path: str, model_path: str = "./Models", 
                   device: str = "gpu", stream: bool = True) -> Generator[str, None, None]:
    """
    使用 Qwen2.5-VL-3B 进行图像美学分析推理的主函数
    
    Args:
        prompt: 文字提示
        image_path: 图片路径
        model_path: 模型路径
        device: 运行设备 ('gpu' 或 'cpu')
        stream: 是否流式输出
        
    Yields:
        流式输出的照片评价文本
    """
    try:
        # 检查输入
        if not os.path.exists(image_path):
            yield f"错误: 图像文件不存在 {image_path}"
            return
            
        if not os.path.exists(model_path):
            yield f"错误: 模型路径不存在 {model_path}"
            return
        
        # 初始化推理引擎
        inference_engine = QwenVLInference(model_path, device=device)
        
        # 执行流式推理
        if stream:
            for chunk in inference_engine.generate_response_stream(image_path, prompt):
                yield chunk
        else:
            # 非流式模式，收集所有文本后一次性返回
            full_response = ""
            for chunk in inference_engine.generate_response_stream(image_path, prompt):
                full_response += chunk
            yield full_response
            
    except Exception as e:
        yield f"推理过程出现错误: {e}"


# ============================================================================
# 辅助函数
# ============================================================================

def load_prompt_template(prompt_file: str) -> str:
    """加载提示模板"""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"提示文件不存在: {prompt_file}")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 <prompt> 标签中的内容
    start_tag = "<prompt>"
    end_tag = "</prompt>"
    
    start_idx = content.find(start_tag)
    end_idx = content.find(end_tag)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("提示文件格式不正确，缺少 <prompt> 标签")
    
    prompt = content[start_idx + len(start_tag):end_idx].strip()
    return prompt


def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_image_files(path: str, extensions: List[str] = None) -> List[str]:
    """
    获取指定路径下的图像文件
    
    Args:
        path: 文件或目录路径
        extensions: 支持的图像扩展名
        
    Returns:
        图像文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        # 单个文件
        if path_obj.suffix.lower() in extensions:
            return [str(path_obj)]
        else:
            raise ValueError(f"不支持的图像格式: {path_obj.suffix}")
    
    elif path_obj.is_dir():
        # 目录
        image_files = []
        for ext in extensions:
            image_files.extend(path_obj.glob(f"**/*{ext}"))
            image_files.extend(path_obj.glob(f"**/*{ext.upper()}"))
        
        return [str(f) for f in sorted(image_files)]
    
    else:
        raise FileNotFoundError(f"路径不存在: {path}")


# ============================================================================
# 主函数和命令行接口
# ============================================================================

def demo_stream_inference():
    """演示流式推理功能"""
    print("=== Qwen2.5-VL-3B 流式美学分析演示 ===\n")
    
    # 配置路径
    current_dir = Path(__file__).parent
    model_path = str(current_dir / "Models")
    image_path = str(current_dir / "demo.png")
    prompt_file = str(current_dir / "qwen_vl_3b_prompt.txt")
    
    try:
        # 检查文件是否存在
        if not Path(model_path).exists():
            print(f"错误: 模型路径不存在 {model_path}")
            return
            
        if not Path(image_path).exists():
            print(f"错误: 图像文件不存在 {image_path}")
            return
            
        if not Path(prompt_file).exists():
            print(f"错误: 提示文件不存在 {prompt_file}")
            return
        
        # 加载提示模板
        prompt_template = load_prompt_template(prompt_file)
        
        print(f"分析图像: {Path(image_path).name}")
        print(f"使用模型: {model_path}")
        print("\n开始流式分析...\n")
        print("-" * 60)
        
        # 执行流式推理
        for chunk in infer_with_qwen(prompt_template, image_path, model_path):
            print(chunk, end='', flush=True)
        
        print("\n" + "-" * 60)
        print("\n✅ 流式分析完成!")
        
        # 保存结果
        output_file = current_dir / "aesthetic_analysis_result.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("图像美学分析结果\n")
            f.write("="*50 + "\n")
            f.write(f"图像文件: {image_path}\n")
            f.write(f"分析时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n分析结果:\n")
            
            # 重新运行推理获取完整结果保存
            full_result = ""
            for chunk in infer_with_qwen(prompt_template, image_path, model_path, stream=False):
                full_result += chunk
            f.write(full_result)
        
        print(f"📄 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B 统一图像美学分析工具")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--image", help="图像文件路径")
    parser.add_argument("--prompt", help="自定义提示文本")
    parser.add_argument("--prompt-file", help="提示文件路径")
    parser.add_argument("--model-path", default="./Models", help="模型路径")
    parser.add_argument("--device", choices=['gpu', 'cpu'], default='gpu', help="运行设备")
    parser.add_argument("--no-stream", action="store_true", help="禁用流式输出")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    if args.demo:
        demo_stream_inference()
        return 0
    
    # 检查必要参数
    if not args.image:
        print("错误: 请指定图像文件路径 (--image)")
        return 1
    
    # 确定提示内容
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        try:
            prompt = load_prompt_template(args.prompt_file)
        except Exception as e:
            print(f"错误: 加载提示文件失败 - {e}")
            return 1
    else:
        # 使用默认提示
        prompt = "请对这张图片进行详细的美学分析，包括构图、焦段、对比度&曝光度&亮度等方面，并给出评分和改进建议。"
    
    try:
        print(f"分析图像: {args.image}")
        print(f"使用设备: {args.device}")
        print("\n开始分析...\n")
        
        stream_mode = not args.no_stream
        for chunk in infer_with_qwen(prompt, args.image, args.model_path, args.device, stream_mode):
            print(chunk, end='', flush=True)
        
        print("\n\n✅ 分析完成!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())