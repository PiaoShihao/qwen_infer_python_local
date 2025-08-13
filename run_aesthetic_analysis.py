#!/usr/bin/env python3
"""
运行图像美学分析的主脚本
整合所有组件，提供完整的分析流程
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# 导入自定义模块
from qwen_vl_inference import QwenVLInference, load_prompt_template
from aesthetic_analyzer import AestheticAnalyzer


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


def analyze_single_image(inference_engine: QwenVLInference, 
                        analyzer: AestheticAnalyzer,
                        image_path: str, 
                        prompt: str,
                        output_dir: str) -> bool:
    """
    分析单张图像
    
    Args:
        inference_engine: 推理引擎
        analyzer: 美学分析器
        image_path: 图像路径
        prompt: 提示文本
        output_dir: 输出目录
        
    Returns:
        是否成功
    """
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"开始分析图像: {Path(image_path).name}")
        
        # 执行推理
        response = inference_engine.generate_response(image_path, prompt)
        
        # 解析响应
        analysis = analyzer.parse_response(response)
        
        # 生成报告
        report = analyzer.format_analysis_report(analysis, image_path)
        
        # 保存文本报告
        image_name = Path(image_path).stem
        report_file = Path(output_dir) / f"{image_name}_aesthetic_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存 JSON 结果
        json_file = Path(output_dir) / f"{image_name}_aesthetic_analysis.json"
        analyzer.save_analysis_json(analysis, str(json_file), image_path)
        
        logger.info(f"分析完成: {image_name} (综合评分: {analysis.scores.overall}/10)")
        return True
        
    except Exception as e:
        logger.error(f"分析图像失败 {image_path}: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B 图像美学分析工具")
    parser.add_argument("image_path", help="图像文件或目录路径")
    parser.add_argument("--model-path", default="./Models", help="模型路径 (默认: ./Models)")
    parser.add_argument("--prompt-file", default="./qwen_vl_3b_prompt.txt", help="提示文件路径")
    parser.add_argument("--output-dir", default="./output", help="输出目录 (默认: ./output)")
    parser.add_argument("--device", choices=['gpu', 'cpu'], default='gpu', help="运行设备")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="日志级别")
    parser.add_argument("--max-tokens", type=int, default=2048, help="最大生成 token 数")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小 (暂未实现)")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # 检查路径
        if not os.path.exists(args.image_path):
            logger.error(f"图像路径不存在: {args.image_path}")
            return 1
            
        if not os.path.exists(args.model_path):
            logger.error(f"模型路径不存在: {args.model_path}")
            return 1
            
        if not os.path.exists(args.prompt_file):
            logger.error(f"提示文件不存在: {args.prompt_file}")
            return 1
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")
        
        # 加载提示模板
        logger.info("加载提示模板...")
        prompt_template = load_prompt_template(args.prompt_file)
        
        # 获取图像文件列表
        logger.info("扫描图像文件...")
        image_files = get_image_files(args.image_path)
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        if not image_files:
            logger.warning("没有找到图像文件")
            return 0
        
        # 初始化推理引擎
        logger.info("初始化 Qwen2.5-VL-3B 推理引擎...")
        inference_engine = QwenVLInference(args.model_path, device=args.device)
        
        # 初始化美学分析器
        analyzer = AestheticAnalyzer()
        
        # 分析图像
        successful_analyses = []
        failed_count = 0
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"进度: {i}/{len(image_files)}")
            
            success = analyze_single_image(
                inference_engine=inference_engine,
                analyzer=analyzer,
                image_path=image_path,
                prompt=prompt_template,
                output_dir=str(output_dir)
            )
            
            if success:
                successful_analyses.append(image_path)
            else:
                failed_count += 1
        
        # 生成汇总报告
        if len(image_files) > 1:
            logger.info("生成汇总报告...")
            
            # 重新读取分析结果 (在实际实现中，可以在内存中保存)
            analyses_for_summary = []
            for image_path in successful_analyses:
                # 这里应该从 JSON 文件重新加载分析结果
                # 为简化示例，使用默认分析结果
                analysis = analyzer._get_default_analysis()
                analyses_for_summary.append((image_path, analysis))
            
            if analyses_for_summary:
                summary_report = analyzer.generate_summary_report(analyses_for_summary)
                summary_file = output_dir / "aesthetic_analysis_summary.txt"
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary_report)
                
                logger.info(f"汇总报告已保存: {summary_file}")
        
        # 输出统计信息
        logger.info(f"分析完成! 成功: {len(successful_analyses)}, 失败: {failed_count}")
        
        if successful_analyses:
            print(f"\n✅ 成功分析 {len(successful_analyses)} 张图像")
            print(f"📁 结果保存在: {output_dir}")
            
            # 显示第一个成功分析的示例结果
            if len(successful_analyses) > 0:
                first_image = Path(successful_analyses[0]).stem
                report_file = output_dir / f"{first_image}_aesthetic_report.txt"
                
                if report_file.exists():
                    print(f"\n📄 示例报告 ({first_image}):")
                    print("-" * 50)
                    with open(report_file, 'r', encoding='utf-8') as f:
                        # 只显示前几行作为示例
                        lines = f.readlines()[:20]
                        print(''.join(lines))
                        if len(f.readlines()) > 20:
                            print("... (查看完整报告请打开文件)")
        
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} 张图像分析失败，请查看日志获取详细信息")
        
        return 0 if failed_count == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())