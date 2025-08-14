# Qwen2.5-VL Swift 完整实现

基于 mlx-swift 的 Qwen2.5-VL-3B 多模态推理 Swift 完整实现，严格按照模型架构实现，包括完整的 tokenization、image encoding、cross-modal attention 等组件，支持本地图片美学分析。

## 功能特性

- 🏗️ **完整模型架构**: 严格按照 Qwen2.5-VL 架构实现，包括：
  - Qwen2.5 语言模型（36层 Transformer）
  - Vision Transformer（32层，带全注意力块）
  - Multi-Head Grouped Query Attention
  - RoPE 位置编码和 RMSNorm
  - SwiGLU 激活函数
- 🔤 **完整 Tokenizer**: 
  - BPE 编码/解码
  - 特殊 tokens 处理
  - 聊天模板格式化
- 🖼️ **多格式图片支持**: PNG, JPEG, HEIC, WebP, BMP, TIFF, GIF
- 🧠 **跨模态融合**: 真正的 cross-modal attention 机制
- 🎯 **美学分析**: 专门针对图片美学质量的分析功能
- 📱 **iOS/macOS 兼容**: 支持在 iOS 和 macOS 平台运行
- 🔄 **流式输出**: 支持实时流式文本生成
- 💾 **内存优化**: 自动调整图片尺寸以减少内存使用
- ⚡ **本地推理**: 基于本地部署的 Qwen 模型，无需网络连接
- 🔧 **量化支持**: 支持 4-bit 量化模型推理

## 项目结构

```
QwenVLSwift/
├── Package.swift                             # Swift 包配置文件
├── Sources/
│   ├── QwenVLSwift/
│   │   ├── Qwen25VLModel.swift              # 完整模型架构实现
│   │   ├── Qwen25VLInferencePipeline.swift  # 推理管道
│   │   └── QwenVLInference.swift            # 高级 API 接口
│   └── QwenVLExample/
│       └── QwenVLExample.swift              # 完整示例程序
├── Tests/
│   └── QwenVLSwiftTests/
│       └── QwenVLSwiftTests.swift           # 单元测试
└── Models/                                  # 模型文件目录
    └── models--mlx-community--Qwen2.5-VL-3B-Instruct-4bit/
        └── snapshots/46d4cf06a06ffc1a766c214174f9cbed2f45bcab/
            ├── config.json                  # 模型配置
            ├── model.safetensors           # 模型权重
            ├── vocab.json                  # 词汇表
            ├── merges.txt                 # BPE merges
            └── tokenizer_config.json      # Tokenizer 配置
```

## 架构说明

### 核心组件

1. **Qwen25VLForConditionalGeneration**: 主模型类，包含：
   - `Qwen25LanguageModel`: 语言模型部分
   - `VisionTransformer`: 视觉编码器
   - `visualProjector`: 视觉特征投影层

2. **Qwen25Tokenizer**: 完整的 BPE tokenizer 实现
   - 支持特殊 tokens (vision_start, vision_end, image_pad 等)
   - 聊天模板格式化
   - 编码/解码功能

3. **VisionTransformer**: 视觉编码器
   - Patch embedding
   - 32层 transformer blocks
   - 空间注意力和全注意力块
   - 输出投影到语言模型维度

4. **Qwen25VLInferencePipeline**: 推理管道
   - 模型加载和权重映射
   - 图片预处理
   - 多模态输入构建
   - 生成和采样逻辑
```

## 安装依赖

### 1. mlx-swift 包

在 Xcode 中添加包依赖：
```
https://github.com/ml-explore/mlx-swift.git
```

或在 `Package.swift` 中添加：
```swift
dependencies: [
    .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.10.0")
]
```

### 2. 模型文件

确保本地 `Models` 目录下包含 Qwen2.5-VL-3B 模型文件。

## 使用方法

### 基本用法

```swift
import QwenVLSwift

// 简单的美学分析
do {
    let result = try aestheticAnalysis(imagePath: "demo.png")
    print(result)
} catch {
    print("分析失败: \(error)")
}
```

### 高级用法

```swift
// 创建推理实例
let inference = QwenVLInference(
    modelPath: "./Models/models--mlx-community--Qwen2.5-VL-3B-Instruct-4bit/snapshots/46d4cf06a06ffc1a766c214174f9cbed2f45bcab",
    maxTokens: 512,
    temperature: 0.0
)

// 同步分析
do {
    let result = try inference.generateAestheticAnalysis(
        imagePath: "demo.png",
        prompt: "请分析这张图片的美学特点",
        maxSize: 1024
    )
    print(result)
} catch {
    print("分析失败: \(error)")
}
```

### 流式输出

```swift
// 流式分析
inference.generateAestheticAnalysisStream(
    imagePath: "demo.png",
    prompt: "请分析这张图片的美学特点"
) { result in
    switch result {
    case .success(let chunk):
        print(chunk, terminator: "")
    case .failure(let error):
        print("错误: \(error)")
    }
}
```

## iOS 集成示例

```swift
import UIKit
import QwenVLSwift

class ViewController: UIViewController {
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var analysisTextView: UITextView!
    
    let inference = QwenVLInference()
    
    @IBAction func analyzeImage(_ sender: UIButton) {
        guard let image = imageView.image else { return }
        
        // 保存图片到临时文件
        let tempURL = saveImageToTemp(image)
        
        // 开始分析
        analysisTextView.text = "正在分析..."
        
        inference.generateAestheticAnalysisStream(
            imagePath: tempURL.path,
            prompt: "请分析这张照片的美学特点",
            maxSize: 512  // iOS 上使用较小尺寸以节省内存
        ) { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let chunk):
                    self?.analysisTextView.text += chunk
                case .failure(let error):
                    self?.analysisTextView.text = "分析失败: \(error.localizedDescription)"
                }
            }
        }
    }
    
    private func saveImageToTemp(_ image: UIImage) -> URL {
        let tempDir = FileManager.default.temporaryDirectory
        let tempURL = tempDir.appendingPathComponent(UUID().uuidString + ".png")
        
        if let data = image.pngData() {
            try? data.write(to: tempURL)
        }
        
        return tempURL
    }
}
```

## 编译和运行

### 命令行编译

```bash
# 编译
xcodebuild build -scheme QwenVLSwift-Package -destination 'platform=OS X'

# 运行示例
xcodebuild build -scheme QwenVLExample -destination 'platform=OS X'
./QwenVLExample
```

### Xcode 编译

1. 打开 Xcode
2. 选择 "File" → "Open" → 选择项目目录
3. 选择目标平台（iOS/macOS）
4. 点击 "Build" 按钮

## 性能优化建议

### 内存使用

- 在 iOS 设备上，建议设置 `maxSize` 参数为 512 或更小
- 及时调用 `clearMemory()` 方法清理内存
- 处理大量图片时，考虑批量处理以避免内存峰值

### 推理速度

- 使用较小的 `maxTokens` 值可以减少推理时间
- 设置合适的 `temperature` 值平衡生成质量和速度
- 预先加载模型以避免重复加载开销

## 错误处理

```swift
do {
    let result = try aestheticAnalysis(imagePath: "demo.png")
    print(result)
} catch QwenVLError.modelLoadFailed(let message) {
    print("模型加载失败: \(message)")
} catch QwenVLError.imageProcessingFailed(let message) {
    print("图片处理失败: \(message)")
} catch QwenVLError.inferenceError(let message) {
    print("推理错误: \(message)")
} catch {
    print("未知错误: \(error)")
}
```

## 支持的图片格式

- PNG (.png)
- JPEG (.jpg, .jpeg)
- HEIC (.heic, .heif)
- WebP (.webp)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- GIF (.gif)

## 注意事项

1. **模型依赖**: 需要本地部署 Qwen2.5-VL-3B 模型
2. **内存需求**: 推理过程需要足够的 GPU 内存
3. **平台限制**: 目前仅支持 Apple Silicon 设备
4. **网络**: 完全离线运行，无需网络连接

## 故障排除

### 模型加载失败
- 检查模型文件路径是否正确
- 确认模型文件完整性
- 验证 safetensors 文件是否存在

### 内存不足
- 减小 `maxSize` 参数
- 降低 `maxTokens` 值
- 调用 `clearMemory()` 清理缓存

### 图片处理失败
- 检查图片文件是否存在
- 确认图片格式是否支持
- 验证图片文件是否损坏

## 许可证

本项目遵循与原始 mlx-swift 相同的许可证条款。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个实现。