# CLAUDE.md

本文档为 Claude Code（claude.ai/code）在处理本仓库代码时提供指导。

## 项目概述
这是一个基于本地的位于 Models 文件夹下的Qwen2.5-VL-3B 模型，使用 mlx 完成部署，可以基于 demo.png和qwen_vl_3b_prompt.txt完成多模态的推理，来提供对于相片的美学分析。

执行 git commit 时，commit 信息优先使用中文。

你的projection显示会超出token限制。要多用实际token，少用projection，Claude Code减少projection使用的策略是：
  1. 减少文件读取量 - 使用head_limit参数或指定行范围，避免读取整个大文件
  2. 精准搜索 - 用Grep和Glob工具定位具体内容，而不是广泛浏览
  3. 分批处理 - 将大任务分解成小步骤，每次只处理必要的文件
  4. 使用Task代理 - 让专门的代理处理复杂搜索，减少主会话的token消耗
  5. 避免重复读取 - 缓存已读文件信息，不重复加载相同内容
