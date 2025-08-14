#!/usr/bin/env python3
"""
多模态推理模块 - 基于 MLX 和 Qwen2.5-VL 模型实现图片美学分析
"""

from typing import Generator, Optional, Union
from pathlib import Path

from mlx_vlm.utils import load
from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template


class MultimodalInference:
    """多模态推理类，封装了模型加载和推理功能"""
    
    def __init__(self, model_path: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"):
        """
        初始化多模态推理类
        
        Args:
            model_path: 模型路径或 Hugging Face 仓库名称
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """加载模型和处理器"""
        try:
            print(f"正在加载模型: {self.model_path}")
            self.model, self.processor = load(
                self.model_path, 
                trust_remote_code=True
            )
            print("模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def generate_stream(
        self,
        image_path: Union[str, Path],
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop_on_repeat: bool = True,
        repeat_threshold: int = 3,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        流式生成文本响应
        
        Args:
            image_path: 图片路径
            prompt: 文本提示
            max_tokens: 最大生成token数
            temperature: 温度参数，控制生成的随机性
            stop_on_repeat: 是否在检测到重复文本时停止
            repeat_threshold: 重复多少次后停止
            **kwargs: 其他生成参数
            
        Yields:
            str: 流式生成的文本片段
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("模型未正确加载")
        
        # 确保图片路径是字符串格式
        if isinstance(image_path, Path):
            image_path = str(image_path)
        
        try:
            generated_text = ""
            recent_segments = []
            
            # 应用聊天模板格式化提示词
            formatted_prompt = apply_chat_template(
                processor=self.processor,
                config=self.model.config,
                prompt=prompt,
                num_images=1,
                num_audios=0
            )
            
            # 添加重复抑制参数
            kwargs_with_repetition_penalty = {
                'repetition_penalty': 1.1,  # 轻微的重复抑制
                'repetition_context_size': 50,  # 考虑前50个token
                **kwargs
            }
            
            # 使用 MLX VLM 的流式生成功能
            for response in stream_generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted_prompt,
                image=[image_path],
                max_tokens=max_tokens,
                temperature=max(temperature, 0.1),  # 确保最小温度为0.1避免完全确定性
                **kwargs_with_repetition_penalty
            ):
                if response.text:
                    # 检测重复文本和无意义循环
                    if stop_on_repeat:
                        # 检测短片段重复
                        recent_segments.append(response.text)
                        if len(recent_segments) > repeat_threshold:
                            recent_segments = recent_segments[-repeat_threshold:]
                        
                        # 检测长片段重复
                        if len(generated_text) > 50:
                            # 检查是否有长片段在重复
                            last_50_chars = generated_text[-50:]
                            if (last_50_chars in generated_text[:-50] and 
                                len(response.text.strip()) > 0):
                                break
                        
                        # 如果最近的片段都相同，则停止生成
                        if (len(recent_segments) >= repeat_threshold and 
                            len(set(recent_segments)) == 1 and
                            len(response.text.strip()) > 3):
                            break
                        
                        # 检测"请上传"等无效输出
                        if ("请上传" in generated_text or 
                            "上传照片" in generated_text or
                            "期待您的" in generated_text) and len(generated_text) > 20:
                            break
                    
                    generated_text += response.text
                    yield response.text
                    
        except Exception as e:
            yield f"推理过程中出现错误: {e}"
    
    def generate(
        self,
        image_path: Union[str, Path],
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        生成完整的文本响应（非流式）
        
        Args:
            image_path: 图片路径
            prompt: 文本提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            **kwargs: 其他生成参数
            
        Returns:
            str: 完整的生成文本
        """
        full_text = ""
        for text_chunk in self.generate_stream(
            image_path, prompt, max_tokens, temperature, **kwargs
        ):
            full_text += text_chunk
        
        return full_text


def aesthetic_analysis(
    image_path: Union[str, Path], 
    prompt_file: Optional[Union[str, Path]] = None,
    model_path: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    max_tokens: int = 512,
    temperature: float = 0.0,
    stream: bool = True
) -> Union[str, Generator[str, None, None]]:
    """
    对图片进行美学分析的便捷函数
    
    Args:
        image_path: 图片路径
        prompt_file: 提示词文件路径，如果为None则使用默认提示词
        model_path: 模型路径
        max_tokens: 最大生成token数
        temperature: 温度参数
        stream: 是否使用流式输出
        
    Returns:
        如果stream=True，返回Generator[str, None, None]
        如果stream=False，返回完整文本str
    """
    # 直接的图片分析提示词（简洁有效）
    prompt = "Analyze this image in detail, focusing on composition, colors, lighting, and overall aesthetic quality."
    
    # 如果指定了提示词文件，尝试读取并转换为直接分析格式
    if prompt_file is not None:
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                # 移除 <prompt> 标签
                file_content = file_content.replace('<prompt>', '').replace('</prompt>', '').strip()
                
                # 转换为直接分析格式，避免对话式提示
                if len(file_content) > 50:
                    # 提取核心分析要求，重新组织为直接分析格式
                    prompt = f"""请直接分析这张图片的以下方面：

1. 构图：分析构图方式和元素布局的合理性
2. 焦段：评价焦段选择和透视效果
3. 对比度：判断明暗对比和色彩对比效果
4. 曝光度：分析曝光准确性和细节呈现
5. 亮度：评价整体亮度和分布情况

开始分析："""
        except FileNotFoundError:
            print(f"提示词文件 {prompt_file} 未找到，使用默认提示词")
    
    # 创建推理实例
    inference = MultimodalInference(model_path)
    
    if stream:
        return inference.generate_stream(
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
    else:
        return inference.generate(
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )


def main():
    """主函数，用于测试和演示"""
    import argparse
    
    parser = argparse.ArgumentParser(description="多模态图片美学分析工具")
    parser.add_argument(
        "--image", 
        type=str, 
        default="demo.png",
        help="图片路径"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="qwen_vl_3b_prompt.txt",
        help="提示词文件路径"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        help="模型路径"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="温度参数"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="禁用流式输出"
    )
    
    args = parser.parse_args()
    
    print(f"正在分析图片: {args.image}")
    print(f"使用模型: {args.model}")
    print("=" * 50)
    
    try:
        if args.no_stream:
            # 非流式输出
            result = aesthetic_analysis(
                image_path=args.image,
                prompt_file=args.prompt_file,
                model_path=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=False
            )
            print(result)
        else:
            # 流式输出
            for text_chunk in aesthetic_analysis(
                image_path=args.image,
                prompt_file=args.prompt_file,
                model_path=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True
            ):
                print(text_chunk, end="", flush=True)
            print()  # 最后换行
            
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "=" * 50)
    print("分析完成")


if __name__ == "__main__":
    main()