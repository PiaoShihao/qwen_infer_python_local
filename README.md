# Qwen2.5-VL-3B 统一图像美学分析系统

基于 MLX 框架的 Qwen2.5-VL-3B 模型本地部署，专门用于图像美学分析和评价。现已整合为单个Python文件，提供流式输出功能。

## 项目概述

本项目实现了一个完整的图像美学分析系统，使用 Apple MLX 框架在 Apple Silicon 设备上高效运行 Qwen2.5-VL-3B 视觉语言模型。系统能够从构图、焦段、对比度、曝光度、亮度等多个维度对图像进行专业的美学评价。

## 功能特点

- 🎨 **多维度美学分析**: 从构图、焦段、对比度&曝光度&亮度等维度进行专业评价
- 🔢 **量化评分系统**: 每个维度提供 1-10 分的量化评分
- 📊 **结构化输出**: 支持文本报告和 JSON 格式的结构化数据输出
- 🌊 **流式输出**: 实时流式显示分析过程，提升用户体验
- 📁 **统一文件**: 所有功能整合在单个Python文件中，便于部署和使用
- 🚀 **GPU 加速**: 基于 MLX 框架，充分利用 Apple Silicon 的性能优势
- 🔧 **灵活调用**: 提供主函数 `infer_with_qwen` 便于集成到其他项目

## 文件结构

```
qwen_infer_python_local/
├── CLAUDE.md                      # 项目说明文档
├── Models/                        # Qwen2.5-VL-3B 模型文件
│   ├── config.json               # 模型配置
│   ├── model.safetensors         # 模型权重
│   ├── tokenizer.json            # 分词器
│   └── ...                       # 其他模型文件
├── demo.png                       # 示例图像
├── qwen_vl_3b_prompt.txt         # 美学分析提示模板
├── qwen_vl_unified.py            # 统一推理脚本（主文件）
├── requirements.txt              # Python 依赖
└── README.md                     # 本文档
```

## 安装配置

### 环境要求

- macOS (Apple Silicon 推荐)
- Python 3.8+
- MLX 框架
- 足够的存储空间存放模型文件

### 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 确保 MLX 框架正确安装
pip install mlx mlx-lm
```

### 模型准备

1. 将 Qwen2.5-VL-3B 模型文件放置在 `Models/` 目录下
2. 确保模型文件包含以下组件：
   - `config.json`: 模型配置文件
   - `model.safetensors`: 模型权重文件
   - `tokenizer.json`: 分词器文件
   - 其他必需的配置文件

## 使用方法

### 快速开始

```bash
# 运行演示模式（推荐）
python qwen_vl_unified.py --demo

# 分析指定图像
python qwen_vl_unified.py --image demo.png --prompt-file qwen_vl_3b_prompt.txt

# 使用自定义提示
python qwen_vl_unified.py --image demo.png --prompt "请对这张图片进行详细的美学分析"
```

### Python API 调用

```python
from qwen_vl_unified import infer_with_qwen

# 流式推理
for chunk in infer_with_qwen(prompt="请分析这张图片", 
                            image_path="demo.png", 
                            model_path="./Models"):
    print(chunk, end='', flush=True)
```

### 命令行参数

```bash
python qwen_vl_unified.py \
    --image <图像路径> \              # 图像文件路径
    --prompt <提示文本> \             # 自定义提示文本
    --prompt-file <提示文件> \        # 提示文件路径
    --model-path ./Models \           # 模型路径
    --device gpu \                    # 运行设备 (gpu/cpu)
    --no-stream \                     # 禁用流式输出
    --log-level INFO                  # 日志级别
```

### 参数说明

- `--demo`: 运行演示模式，使用默认的demo.png和提示文件
- `--image`: 要分析的图像文件路径
- `--prompt`: 直接指定提示文本
- `--prompt-file`: 美学分析提示模板文件路径 (默认: ./qwen_vl_3b_prompt.txt)
- `--model-path`: Qwen2.5-VL-3B 模型文件所在目录 (默认: ./Models)
- `--device`: 运行设备，gpu 或 cpu (默认: gpu)
- `--no-stream`: 禁用流式输出，一次性显示完整结果
- `--log-level`: 日志输出级别 (默认: INFO)

## 输出格式

### 文本报告

每张图像会生成一份详细的美学分析报告，包含：

```
============================================================
图像美学分析报告
============================================================

分析图像: demo.png

【维度分析与评分】

🎨 构图分析 (评分: 7.2/10)
采用了较为传统的海景构图，前景的岩石形成了天然的引导线...

📷 焦段分析 (评分: 7.8/10)
使用适中焦段拍摄，透视效果自然，没有明显的变形...

💡 对比度&曝光度&亮度分析 (评分: 8.1/10)
画面对比度适中，天空的蓝色与岩石的暖色调形成了良好的色彩对比...

【综合评价】
综合评分: 7.7/10

这是一幅较为标准的海岸风光摄影作品...

【改进建议】
1）可以尝试不同的拍摄角度，减少前景岩石的占比...
2）等待更佳的光线条件，如黄金时段的温暖光线...
3）考虑使用渐变镜平衡天空和地面的光比...
```

### JSON 数据

同时会生成结构化的 JSON 数据文件，方便程序处理：

```json
{
  "image_path": "demo.png",
  "timestamp": "2024-01-01T00:00:00Z",
  "analysis": {
    "composition_analysis": "...",
    "focal_length_analysis": "...",
    "contrast_exposure_brightness_analysis": "...",
    "overall_evaluation": "...",
    "suggestions": "...",
    "scores": {
      "composition": 7.2,
      "focal_length": 7.8,
      "contrast_exposure_brightness": 8.1,
      "overall": 7.7
    }
  }
}
```

### 流式输出

系统支持实时流式输出，分析过程会逐步显示：

```
=== Qwen2.5-VL-3B 流式美学分析演示 ===

分析图像: demo.png
使用模型: /path/to/Models

开始流式分析...

------------------------------------------------------------
维度分析与评分：
- 构图：采用了较为传统的海景构图，前景的岩石形成了天然的引导线，
将视线引向远处的海平面...评分：7.2分
- 焦段：使用适中焦段拍摄，透视效果自然...评分：7.8分  
- 对比度&曝光度&亮度：画面对比度适中...评分：8.1分

综合评分：7.7（1-10分）

综合评价与建议：这是一幅较为标准的海岸风光摄影作品...
------------------------------------------------------------

✅ 流式分析完成!
```

## 核心功能说明

### qwen_vl_unified.py

统一推理脚本，整合了所有功能：

#### 主要类

- **QwenVLInference**: 核心推理引擎
  - 模型加载和初始化
  - 图像预处理
  - 文本提示处理
  - MLX 框架的推理执行
  - 流式响应生成

- **AestheticAnalyzer**: 美学分析器
  - 模型响应解析
  - 结构化数据提取
  - 评分计算
  - 报告格式化

#### 主要函数

- **infer_with_qwen()**: 主推理函数
  ```python
  def infer_with_qwen(prompt: str, image_path: str, 
                     model_path: str = "./Models", 
                     device: str = "gpu", 
                     stream: bool = True) -> Generator[str, None, None]
  ```
  
- **demo_stream_inference()**: 演示模式
- **main()**: 命令行入口

## 性能优化

### GPU 加速

系统默认使用 Apple Silicon 的 GPU 进行推理加速：

```python
# 自动选择最佳设备
if mx.metal.is_available():
    mx.set_default_device(mx.gpu)
    print("使用 GPU (Metal) 运行推理")
else:
    mx.set_default_device(mx.cpu)
    print("使用 CPU 运行推理")
```

### 内存管理

- 使用 MLX 的高效内存管理
- 支持量化模型以减少内存占用
- 批处理时自动内存释放

### 流式处理

当前版本特色：
- 实时流式文本输出
- 低延迟的响应展示
- 优化的内存使用模式
- 可中断的推理过程

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查 Models/ 目录下的文件完整性
   - 确认模型文件格式正确 (safetensors)
   - 验证配置文件 config.json 的有效性

2. **GPU 不可用**
   - 确认设备支持 Apple Silicon
   - 检查 MLX 框架安装是否正确
   - 尝试使用 `--device cpu` 参数

3. **内存不足**
   - 减少 `--max-tokens` 参数值
   - 使用 CPU 模式运行
   - 检查可用系统内存

4. **图像格式不支持**
   - 支持的格式: JPG, PNG, BMP, TIFF, WebP
   - 确认图像文件没有损坏
   - 尝试转换图像格式

### 调试模式

使用调试模式获取详细日志信息：

```bash
python qwen_vl_unified.py --image demo.png --log-level DEBUG
```

## 开发说明

### 扩展功能

要添加新的分析维度，需要在 `qwen_vl_unified.py` 中修改：
1. 更新 `AestheticScore` 数据类 - 添加新的评分字段
2. 修改 `AestheticAnalyzer.parse_response()` - 添加解析逻辑  
3. 更新 `qwen_vl_3b_prompt.txt` - 更新提示模板

### 模型适配

要适配其他 MLX 兼容的视觉语言模型：
1. 修改 `QwenVLInference` 类中的模型加载逻辑
2. 调整 `preprocess_image()` 中的图像预处理参数
3. 更新 `preprocess_text()` 中的文本编码方式

### 集成到其他项目

由于所有功能都在单个文件中，可以轻松集成：

```python
# 作为模块导入
from qwen_vl_unified import infer_with_qwen, QwenVLInference

# 或者直接复制 infer_with_qwen 函数到你的项目中
```

## 许可证

本项目基于开源许可证发布，具体许可信息请查看相关模型和框架的官方文档。

## 致谢

- Qwen2.5-VL 模型 by Alibaba
- MLX 框架 by Apple
- 相关开源社区的贡献