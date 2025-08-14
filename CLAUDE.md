# CLAUDE.md

本文档为 Claude Code（claude.ai/code）在处理本仓库代码时提供指导。

## 项目概述
这是一个基于本地的位于 Models 文件夹下的Qwen2.5-VL-3B 模型，使用 mlx 完成部署，可以基于 demo.png和qwen_vl_3b_prompt.txt完成多模态的推理，来提供对于相片的美学分析。

正常的 Python 上可以使用 
python -m mlx_vlm.generate --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit --max-tokens 100 --temp 0.0 --prompt "Describe this image." --image ./demo.png
来执行推理，请参考如上命令实现一个单独的 Python 文件来完成多模态的推理，定义一个单独的函数，输入是图片和文字 prompt，输出是流式的文字评价。

执行 git commit 时，commit 信息优先使用中文。

使用 conda 环境是 qwenvl
