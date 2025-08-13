# Qwen2.5-VL-3B 图像美学分析系统

基于 MLX 框架的 Qwen2.5-VL-3B 模型本地部署，专门用于图像美学分析和评价。

## 项目概述

本项目实现了一个完整的图像美学分析系统，使用 Apple MLX 框架在 Apple Silicon 设备上高效运行 Qwen2.5-VL-3B 视觉语言模型。系统能够从构图、焦段、对比度、曝光度、亮度等多个维度对图像进行专业的美学评价。

## 功能特点

- 🎨 **多维度美学分析**: 从构图、焦段、对比度&曝光度&亮度等维度进行专业评价
- 🔢 **量化评分系统**: 每个维度提供 1-10 分的量化评分
- 📊 **结构化输出**: 支持文本报告和 JSON 格式的结构化数据输出
- 🖼️ **批量处理**: 支持单张图像或批量处理多张图像
- 📈 **汇总报告**: 多张图像分析后自动生成汇总统计报告
- 🚀 **GPU 加速**: 基于 MLX 框架，充分利用 Apple Silicon 的性能优势

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
├── qwen_vl_inference.py          # 核心推理引擎
├── aesthetic_analyzer.py         # 美学分析器
├── run_aesthetic_analysis.py     # 主运行脚本
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

### 基本使用

```bash
# 分析单张图像
python run_aesthetic_analysis.py demo.png

# 分析指定目录下的所有图像
python run_aesthetic_analysis.py /path/to/images/

# 指定输出目录
python run_aesthetic_analysis.py demo.png --output-dir ./results
```

### 高级参数

```bash
python run_aesthetic_analysis.py <图像路径> \
    --model-path ./Models \           # 模型路径
    --prompt-file ./qwen_vl_3b_prompt.txt \  # 提示文件
    --output-dir ./output \           # 输出目录
    --device gpu \                    # 运行设备 (gpu/cpu)
    --log-level INFO \                # 日志级别
    --max-tokens 2048                 # 最大生成 token 数
```

### 参数说明

- `image_path`: 要分析的图像文件或包含图像的目录路径
- `--model-path`: Qwen2.5-VL-3B 模型文件所在目录 (默认: ./Models)
- `--prompt-file`: 美学分析提示模板文件路径 (默认: ./qwen_vl_3b_prompt.txt)
- `--output-dir`: 分析结果输出目录 (默认: ./output)
- `--device`: 运行设备，gpu 或 cpu (默认: gpu)
- `--log-level`: 日志输出级别 (默认: INFO)
- `--max-tokens`: 模型生成的最大 token 数量 (默认: 2048)

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

### 汇总报告

批量分析多张图像时，会自动生成汇总统计报告：

```
============================================================
图像美学分析汇总报告
============================================================

分析图像总数: 5

【平均评分】
- 综合评分: 7.45/10
- 构图评分: 7.20/10
- 焦段评分: 7.60/10
- 对比度&曝光度&亮度评分: 7.55/10

【最佳表现】
图像: best_photo.jpg
综合评分: 8.9/10

【需要改进】
图像: needs_improvement.jpg
综合评分: 6.1/10
```

## 核心模块说明

### qwen_vl_inference.py

核心推理引擎，负责：
- 模型加载和初始化
- 图像预处理
- 文本提示处理
- MLX 框架的推理执行
- 响应生成

### aesthetic_analyzer.py

美学分析器，负责：
- 模型响应解析
- 结构化数据提取
- 评分计算
- 报告格式化
- JSON 数据保存

### run_aesthetic_analysis.py

主运行脚本，提供：
- 命令行接口
- 批量处理功能
- 进度监控
- 错误处理
- 汇总报告生成

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

### 并行处理

当前版本支持单张图像处理，未来版本计划支持：
- 多图像并行推理
- 异步 I/O 操作
- 流式处理大批量图像

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
python run_aesthetic_analysis.py demo.png --log-level DEBUG
```

## 开发说明

### 扩展功能

要添加新的分析维度，需要修改：
1. `qwen_vl_3b_prompt.txt` - 更新提示模板
2. `aesthetic_analyzer.py` - 添加解析逻辑
3. `AestheticScore` 数据类 - 添加新的评分字段

### 模型适配

要适配其他 MLX 兼容的视觉语言模型：
1. 修改 `qwen_vl_inference.py` 中的模型加载逻辑
2. 调整图像预处理参数
3. 更新文本编码方式

### Swift 迁移准备

当前 Python 实现为后续 Swift 移植做了以下准备：
- 清晰的模块化设计
- 标准化的数据结构
- MLX 框架的统一使用
- 详细的接口文档

## 许可证

本项目基于开源许可证发布，具体许可信息请查看相关模型和框架的官方文档。

## 致谢

- Qwen2.5-VL 模型 by Alibaba
- MLX 框架 by Apple
- 相关开源社区的贡献