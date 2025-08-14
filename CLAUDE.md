# CLAUDE.md

本文档为 Claude Code（claude.ai/code）在处理本仓库代码时提供指导。

## 项目概述
这是一个基于本地的位于 Models 文件夹下的Qwen2.5-VL-3B 模型，使用 mlx 完成部署，可以基于 demo.png和qwen_vl_3b_prompt.txt完成多模态的推理，来提供对于相片的美学分析。
目前在 multimodel_inference.py 已实现基于 Python的推理功能，我现在希望实现在 iOS 应用上可以运行的 swift 语言版本，请基于./mlx-swift这个包来实现同样的功能的 swift 语言版本，我期望实现一个函数，输入是文字 prompt 和图片，输出是经过本地部署的 Qwen 模型输出的美学分析与评价。

执行 git commit 时，commit 信息优先使用中文。

使用 conda 环境是 qwenvl
