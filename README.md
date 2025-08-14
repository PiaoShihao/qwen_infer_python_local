# Qwen2.5-VL 多模态推理工具

基于 MLX 和 Qwen2.5-VL-3B-Instruct-4bit 模型的本地多模态推理工具，用于图片美学分析。

## 环境要求

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- 16GB+ 内存推荐

## 安装

```bash
# 创建环境
conda create -n qwenvl python=3.10
conda activate qwenvl

# 安装依赖
pip install -r requirements.txt
```

## 使用示例

### multimodal_inference.py 调用示例

```python
from multimodal_inference import MultimodalInference, aesthetic_analysis

# 方法一：使用 MultimodalInference 类
inference = MultimodalInference()

# 流式生成
print("=== 流式输出 ===")
for text_chunk in inference.generate_stream(
    image_path="demo.png",
    prompt="请详细分析这张图片的美学特点，从构图、色彩、光影等方面评价",
    max_tokens=300,
    temperature=0.3
):
    print(text_chunk, end="", flush=True)
print("\n")

# 非流式生成
print("=== 完整输出 ===")
result = inference.generate(
    image_path="demo.png",
    prompt="请用专业摄影角度分析这张图片",
    max_tokens=200
)
print(result)

# 方法二：使用便捷函数（推荐用于美学分析）
print("\n=== 美学分析 ===")
for text_chunk in aesthetic_analysis(
    image_path="demo.png",
    max_tokens=400,
    stream=True
):
    print(text_chunk, end="", flush=True)
```

### 命令行使用

```bash
# 激活环境
conda activate qwenvl

# 基本用法
python multimodal_inference.py --image demo.png

# 推荐参数（获得更好的分析效果）
python multimodal_inference.py --image demo.png --max-tokens 300 --temperature 0.3
```

## 项目文件

```
├── multimodal_inference.py     # 主程序文件
├── requirements.txt            # 依赖包列表
├── demo.png / demo2.png        # 示例图片
└── README.md                   # 使用说明
```

## 说明

- 首次运行会自动下载模型（约2-3GB）
- 支持流式和非流式输出
- 内置重复文本检测机制
- 专为图片美学分析优化