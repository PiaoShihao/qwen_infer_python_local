#!/usr/bin/env python3
"""
多模态推理模块 - 基于 MLX 和 Qwen2.5-VL 模型实现图片美学分析
"""

from typing import Generator, Optional, Union
from pathlib import Path
import os

from mlx_vlm.utils import load
from mlx_vlm.generate import stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("警告: PIL/Pillow 未安装，只支持 PNG 格式")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    HEIF_SUPPORT = False


# 支持的图片格式
SUPPORTED_FORMATS = {
    '.png': 'PNG',
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.webp': 'WEBP',
    '.bmp': 'BMP',
    '.tiff': 'TIFF',
    '.tif': 'TIFF',
    '.gif': 'GIF',
    '.heic': 'HEIF',
    '.heif': 'HEIF'
}


def get_image_format(image_path: Union[str, Path]) -> Optional[str]:
    """
    获取图片格式
    
    Args:
        image_path: 图片路径
        
    Returns:
        图片格式字符串，如果不支持则返回 None
    """
    path = Path(image_path)
    extension = path.suffix.lower()
    
    if extension in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[extension]
    return None


def validate_image_file(image_path: Union[str, Path]) -> tuple:
    """
    验证图片文件是否存在且格式支持
    
    Args:
        image_path: 图片路径
        
    Returns:
        tuple[bool, str]: (是否有效, 错误信息或格式信息)
    """
    path = Path(image_path)
    
    # 检查文件是否存在
    if not path.exists():
        return False, f"文件不存在: {path}"
    
    if not path.is_file():
        return False, f"路径不是文件: {path}"
    
    # 检查文件扩展名
    image_format = get_image_format(path)
    if image_format is None:
        supported_exts = ', '.join(SUPPORTED_FORMATS.keys())
        return False, f"不支持的图片格式，支持的格式: {supported_exts}"
    
    # 对于 HEIC/HEIF 格式，检查是否有相应支持
    if image_format == 'HEIF' and not HEIF_SUPPORT:
        return False, "HEIC/HEIF 格式需要安装 pillow-heif 库: pip install pillow-heif"
    
    # 检查 PIL 是否可用
    if not PIL_AVAILABLE and image_format != 'PNG':
        return False, f"需要安装 Pillow 库来支持 {image_format} 格式: pip install Pillow"
    
    return True, f"有效的 {image_format} 格式图片"


def resize_image_if_needed(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    如果图片太大，调整图片尺寸以减少内存使用
    
    Args:
        image: PIL Image 对象
        max_size: 最大尺寸（宽度或高度）
        
    Returns:
        调整后的 PIL Image 对象
    """
    width, height = image.size
    
    # 如果图片已经足够小，直接返回
    if max(width, height) <= max_size:
        return image
    
    # 计算缩放比例
    scale = max_size / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    print(f"图片尺寸过大 ({width}x{height})，调整为 ({new_width}x{new_height}) 以减少内存使用")
    
    # 使用高质量的重采样方法
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def convert_image_if_needed(image_path: Union[str, Path], target_format: str = 'PNG', max_size: int = 1024) -> str:
    """
    如果需要，将图片转换为目标格式并调整尺寸
    
    Args:
        image_path: 输入图片路径
        target_format: 目标格式 (默认 PNG)
        max_size: 最大尺寸限制
        
    Returns:
        处理后的图片路径（可能是原路径或临时文件路径）
    """
    if not PIL_AVAILABLE:
        return str(image_path)
    
    path = Path(image_path)
    image_format = get_image_format(path)
    
    try:
        # 打开图片
        with Image.open(path) as img:
            # 调整图片尺寸
            img = resize_image_if_needed(img, max_size)
            
            # 检查是否需要转换格式或调整尺寸
            original_size = Image.open(path).size
            needs_conversion = image_format != target_format
            needs_resize = max(original_size) > max_size
            
            # 如果不需要任何处理，直接返回原路径
            if not needs_conversion and not needs_resize:
                return str(path)
            
            # 转换为适当的颜色模式
            if img.mode in ('RGBA', 'LA', 'P'):
                if target_format == 'JPEG':
                    # JPEG 不支持透明度，转换为白色背景
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if len(img.split()) == 4 else None)
                    img = rgb_img
                elif target_format == 'PNG':
                    img = img.convert('RGBA')
            elif img.mode != 'RGB' and target_format in ('JPEG', 'BMP'):
                img = img.convert('RGB')
            
            # 生成临时文件名
            temp_ext = '.png' if target_format == 'PNG' else '.jpg'
            temp_path = path.parent / f"{path.stem}_converted{temp_ext}"
            
            # 保存处理后的图片
            img.save(temp_path, format=target_format, optimize=True)
            return str(temp_path)
            
    except Exception as e:
        print(f"图片处理失败: {e}")
        return str(image_path)


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
        max_size: int = 1024,
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
            max_size: 图片最大尺寸限制（减少内存使用）
            **kwargs: 其他生成参数
            
        Yields:
            str: 流式生成的文本片段
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("模型未正确加载")
        
        # 验证图片文件
        is_valid, message = validate_image_file(image_path)
        if not is_valid:
            yield f"图片文件错误: {message}"
            return
        
        print(f"图片验证成功: {message}")
        
        # 如果需要，转换图片格式和调整尺寸
        processed_image_path = convert_image_if_needed(image_path, target_format='PNG', max_size=max_size)
        if processed_image_path != str(image_path):
            print(f"图片已处理: {image_path} -> {processed_image_path}")
        
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
            
            # 使用 MLX VLM 的流式生成功能，带内存错误处理
            try:
                for response in stream_generate(
                    model=self.model,
                    processor=self.processor,
                    prompt=formatted_prompt,
                    image=[processed_image_path],
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
                error_msg = str(e)
                if ('memory' in error_msg.lower() or 'insufficient' in error_msg.lower() or 
                    'kIOGPUCommandBufferCallbackErrorOutOfMemory' in error_msg):
                    yield f"GPU 内存不足错误。尝试使用更小的图片或减少 max_tokens 参数。"
                    clear_mlx_memory()
                    
                    # 如果图片尺寸还能进一步减小，尝试重新处理
                    if max_size > 256:
                        yield f"尝试使用更小的图片尺寸重新处理..."
                        smaller_processed_path = convert_image_if_needed(image_path, target_format='PNG', max_size=max_size//2)
                        try:
                            for response in stream_generate(
                                model=self.model,
                                processor=self.processor,
                                prompt=formatted_prompt,
                                image=[smaller_processed_path],
                                max_tokens=min(max_tokens, 256),  # 减少 token 数
                                temperature=max(temperature, 0.1),
                                **kwargs_with_repetition_penalty
                            ):
                                if response.text:
                                    yield response.text
                        except Exception as retry_e:
                            yield f"重试失败: {retry_e}"
                        finally:
                            # 清理小尺寸临时文件
                            if smaller_processed_path != processed_image_path and os.path.exists(smaller_processed_path):
                                try:
                                    os.remove(smaller_processed_path)
                                except:
                                    pass
                    else:
                        yield f"无法进一步减小图片尺寸，请尝试使用更小的图片文件。"
                else:
                    yield f"推理过程中出现错误: {e}"
                    
        except Exception as e:
            yield f"推理过程中出现错误: {e}"
        finally:
            # 清理临时文件
            if processed_image_path != str(image_path) and os.path.exists(processed_image_path):
                try:
                    os.remove(processed_image_path)
                    print(f"已删除临时文件: {processed_image_path}")
                except Exception as e:
                    print(f"删除临时文件失败: {e}")
    
    def generate(
        self,
        image_path: Union[str, Path],
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_size: int = 1024,
        **kwargs
    ) -> str:
        """
        生成完整的文本响应（非流式）
        
        Args:
            image_path: 图片路径
            prompt: 文本提示
            max_tokens: 最大生成token数
            temperature: 温度参数
            max_size: 图片最大尺寸限制
            **kwargs: 其他生成参数
            
        Returns:
            str: 完整的生成文本
        """
        full_text = ""
        for text_chunk in self.generate_stream(
            image_path, prompt, max_tokens, temperature, max_size=max_size, **kwargs
        ):
            full_text += text_chunk
        
        return full_text


def get_supported_formats() -> dict:
    """
    获取当前支持的图片格式信息
    
    Returns:
        dict: 包含支持的格式和状态信息
    """
    formats_info = {
        'supported_extensions': list(SUPPORTED_FORMATS.keys()),
        'pil_available': PIL_AVAILABLE,
        'heif_support': HEIF_SUPPORT,
        'status': {}
    }
    
    for ext, fmt in SUPPORTED_FORMATS.items():
        if fmt == 'HEIF':
            formats_info['status'][ext] = "支持" if HEIF_SUPPORT else "需要 pillow-heif"
        else:
            formats_info['status'][ext] = "支持" if PIL_AVAILABLE else "需要 Pillow"
    
    return formats_info


def print_supported_formats():
    """打印支持的图片格式信息"""
    info = get_supported_formats()
    print("=" * 50)
    print("支持的图片格式:")
    print("=" * 50)
    
    for ext, status in info['status'].items():
        fmt = SUPPORTED_FORMATS[ext]
        print(f"{ext:<8} ({fmt:<8}) - {status}")
    
    print("\n库支持状态:")
    print(f"PIL/Pillow: {'✓' if info['pil_available'] else '✗'}")
    print(f"HEIF 支持:  {'✓' if info['heif_support'] else '✗'}")
    
    if not info['heif_support']:
        print("\n要支持 HEIC/HEIF 格式，请安装: pip install pillow-heif")
    
    print("=" * 50)


def clear_mlx_memory():
    """清理 MLX 内存"""
    if MLX_AVAILABLE:
        try:
            mx.metal.clear_cache()
            print("已清理 MLX 内存缓存")
        except Exception as e:
            print(f"清理 MLX 内存失败: {e}")


def handle_memory_error(func):
    """装饰器：处理内存不足错误"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if ('memory' in error_msg or 'insufficient' in error_msg or 
                'kIOGPUCommandBufferCallbackErrorOutOfMemory' in str(e)):
                print("检测到 GPU 内存不足错误，尝试清理内存后重试...")
                clear_mlx_memory()
                
                # 尝试使用更小的图片尺寸重试
                if 'image_path' in kwargs:
                    print("尝试使用更小的图片尺寸重试...")
                    kwargs['max_size'] = kwargs.get('max_size', 1024) // 2
                    
                try:
                    return func(*args, **kwargs)
                except Exception as retry_e:
                    raise RuntimeError(f"GPU 内存不足，即使降低图片尺寸后仍然失败: {retry_e}")
            else:
                raise e
    return wrapper


def aesthetic_analysis(
    image_path: Union[str, Path], 
    prompt_file: Optional[Union[str, Path]] = None,
    model_path: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    max_tokens: int = 512,
    temperature: float = 0.0,
    stream: bool = True,
    max_size: int = 1024
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
        max_size: 图片最大尺寸限制
        
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
            temperature=temperature,
            max_size=max_size
        )
    else:
        return inference.generate(
            image_path=image_path,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            max_size=max_size
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
    parser.add_argument(
        "--show-formats",
        action="store_true",
        help="显示支持的图片格式"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="图片最大尺寸限制（减少内存使用）"
    )
    
    args = parser.parse_args()
    
    # 如果用户要求显示格式信息
    if args.show_formats:
        print_supported_formats()
        return
    
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
                stream=False,
                max_size=args.max_size
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
                stream=True,
                max_size=args.max_size
            ):
                print(text_chunk, end="", flush=True)
            print()  # 最后换行
            
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n" + "=" * 50)
    print("分析完成")


if __name__ == "__main__":
    main()