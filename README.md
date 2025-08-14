# Qwen2.5-VL 多模态推理工具

基于 MLX 和 Qwen2.5-VL-3B-Instruct-4bit 模型的本地多模态推理工具，用于图片美学分析。

## 环境要求

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10+
- 16GB+ 内存推荐

## 安装

```bash
conda activate qwenvl

# 安装依赖
pip install -r requirements.txt
```

### 命令行使用

```bash
# 激活环境
conda activate qwenvl

# 基本用法
python multimodal_inference.py --image demo.png

# 自定义参数
python multimodal_inference.py --image demo.png --max-tokens 300 --temperature 0.1
python multimodal_inference.py --image demo.HEIC --max-tokens 300
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