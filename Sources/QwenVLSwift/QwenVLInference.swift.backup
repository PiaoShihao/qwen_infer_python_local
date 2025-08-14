// Copyright © 2024 Apple Inc.
// Qwen2.5-VL 高级推理 API

import Foundation
import MLX
import MLXNN
import CoreImage
import Vision
import UniformTypeIdentifiers

/// 错误类型定义
public enum QwenVLError: Error, LocalizedError {
    case modelLoadFailed(String)
    case imageProcessingFailed(String)
    case inferenceError(String)
    case fileNotFound(String)
    case unsupportedImageFormat(String)
    case memoryError(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let message):
            return "模型加载失败: \(message)"
        case .imageProcessingFailed(let message):
            return "图片处理失败: \(message)"
        case .inferenceError(let message):
            return "推理错误: \(message)"
        case .fileNotFound(let message):
            return "文件未找到: \(message)"
        case .unsupportedImageFormat(let message):
            return "不支持的图片格式: \(message)"
        case .memoryError(let message):
            return "内存错误: \(message)"
        }
    }
}

/// 支持的图片格式
public struct SupportedImageFormats {
    public static let supportedExtensions: Set<String> = ["png", "jpg", "jpeg", "heic", "heif", "webp", "bmp", "tiff", "tif", "gif"]
    
    public static func isSupported(_ extension: String) -> Bool {
        return supportedExtensions.contains(extension.lowercased())
    }
}

/// 图片处理工具类
public class ImageProcessor {
    
    /// 验证图片文件是否存在且格式支持
    /// - Parameter imagePath: 图片文件路径
    /// - Returns: 验证结果和消息
    public static func validateImageFile(at imagePath: String) -> (isValid: Bool, message: String) {
        let url = URL(fileURLWithPath: imagePath)
        
        // 检查文件是否存在
        guard FileManager.default.fileExists(atPath: imagePath) else {
            return (false, "文件不存在: \(imagePath)")
        }
        
        // 检查是否为文件而非目录
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: imagePath, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            return (false, "路径不是文件: \(imagePath)")
        }
        
        // 检查文件扩展名
        let pathExtension = url.pathExtension.lowercased()
        guard SupportedImageFormats.isSupported(pathExtension) else {
            let supportedFormats = SupportedImageFormats.supportedExtensions.joined(separator: ", ")
            return (false, "不支持的图片格式，支持的格式: \(supportedFormats)")
        }
        
        return (true, "有效的 \(pathExtension.uppercased()) 格式图片")
    }
    
    /// 加载并处理图片
    /// - Parameters:
    ///   - imagePath: 图片文件路径
    ///   - maxSize: 最大尺寸限制
    /// - Returns: 处理后的 CIImage
    public static func loadAndProcessImage(from imagePath: String, maxSize: Int = 1024) throws -> CIImage {
        let url = URL(fileURLWithPath: imagePath)
        
        // 验证图片文件
        let validation = validateImageFile(at: imagePath)
        guard validation.isValid else {
            throw QwenVLError.imageProcessingFailed(validation.message)
        }
        
        // 加载图片
        guard let ciImage = CIImage(contentsOf: url) else {
            throw QwenVLError.imageProcessingFailed("无法加载图片: \(imagePath)")
        }
        
        // 检查是否需要调整尺寸
        let imageSize = ciImage.extent
        let currentMaxSize = max(imageSize.width, imageSize.height)
        
        if currentMaxSize <= CGFloat(maxSize) {
            return ciImage
        }
        
        // 计算缩放比例
        let scale = CGFloat(maxSize) / currentMaxSize
        let transform = CGAffineTransform(scaleX: scale, y: scale)
        let resizedImage = ciImage.transformed(by: transform)
        
        print("图片尺寸过大 (\(Int(imageSize.width))x\(Int(imageSize.height)))，调整为 (\(Int(resizedImage.extent.width))x\(Int(resizedImage.extent.height))) 以减少内存使用")
        
        return resizedImage
    }
    
    /// 将 CIImage 转换为 MLXArray
    /// - Parameter ciImage: 输入的 CIImage
    /// - Returns: 转换后的 MLXArray，形状为 [height, width, channels]
    public static func ciImageToMLXArray(_ ciImage: CIImage) throws -> MLXArray {
        let context = CIContext()
        
        // 确保图片在正确的颜色空间
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        let format = CIFormat.RGBA8
        
        let width = Int(ciImage.extent.width)
        let height = Int(ciImage.extent.height)
        
        // 创建 bitmap 缓冲区
        let bytesPerPixel = 4 // RGBA
        let bytesPerRow = width * bytesPerPixel
        let bufferSize = height * bytesPerRow
        
        var pixelBuffer = [UInt8](repeating: 0, count: bufferSize)
        
        // 渲染图片到缓冲区
        context.render(ciImage, toBitmap: &pixelBuffer, rowBytes: bytesPerRow, bounds: ciImage.extent, format: format, colorSpace: colorSpace)
        
        // 转换为 Float32 数组并标准化到 [0, 1]
        let floatPixels = pixelBuffer.map { Float($0) / 255.0 }
        
        // 重新组织数据为 [height, width, channels] 格式，只取 RGB 通道
        var rgbData = [Float](repeating: 0, count: height * width * 3)
        for y in 0..<height {
            for x in 0..<width {
                let srcIndex = (y * width + x) * 4 // RGBA
                let dstIndex = (y * width + x) * 3 // RGB
                
                rgbData[dstIndex] = floatPixels[srcIndex]     // R
                rgbData[dstIndex + 1] = floatPixels[srcIndex + 1] // G
                rgbData[dstIndex + 2] = floatPixels[srcIndex + 2] // B
            }
        }
        
        // 创建 MLXArray
        return MLXArray(rgbData, [height, width, 3])
    }
}

/// Qwen2.5-VL 高级推理接口
public class QwenVLInference {
    private let pipeline: Qwen25VLInferencePipeline
    private let modelPath: String
    private let maxTokens: Int
    private let temperature: Float
    
    /// 初始化推理类
    /// - Parameters:
    ///   - modelPath: 模型路径，默认使用本地 Models 目录下的模型
    ///   - maxTokens: 最大生成 token 数
    ///   - temperature: 温度参数，控制生成的随机性
    public init(modelPath: String = "./Models/models--mlx-community--Qwen2.5-VL-3B-Instruct-4bit/snapshots/46d4cf06a06ffc1a766c214174f9cbed2f45bcab", 
                maxTokens: Int = 512, 
                temperature: Float = 0.0) throws {
        self.modelPath = modelPath
        self.maxTokens = maxTokens
        self.temperature = temperature
        
        // 初始化底层推理管道
        self.pipeline = try Qwen25VLInferencePipeline(modelPath: modelPath)
    }
    
    /// 加载模型 (已在初始化时完成)
    /// - Throws: 模型加载失败时抛出错误
    public func loadModel() throws {
        // 模型已在 init 中加载
        print("模型已就绪")
    }
    
    /// 生成美学分析
    /// - Parameters:
    ///   - imagePath: 图片路径
    ///   - prompt: 文本提示
    ///   - maxSize: 图片最大尺寸限制
    /// - Returns: 美学分析结果
    public func generateAestheticAnalysis(imagePath: String, 
                                        prompt: String, 
                                        maxSize: Int = 1024) throws -> String {
        // 加载并处理图片
        let ciImage = try ImageProcessor.loadAndProcessImage(from: imagePath, maxSize: maxSize)
        
        // 构建用于美学分析的提示词
        let aestheticPrompt = """
        请直接分析这张图片的以下方面：

        1. 构图：分析构图方式和元素布局的合理性
        2. 焦段：评价焦段选择和透视效果
        3. 对比度：判断明暗对比和色彩对比效果
        4. 曝光度：分析曝光准确性和细节呈现
        5. 亮度：评价整体亮度和分布情况

        开始分析：
        """
        
        // 使用真实的推理管道
        return try pipeline.generate(
            prompt: aestheticPrompt,
            image: ciImage,
            maxNewTokens: maxTokens,
            temperature: temperature
        )
    }
    
    /// 流式生成美学分析
    /// - Parameters:
    ///   - imagePath: 图片路径
    ///   - prompt: 文本提示
    ///   - maxSize: 图片最大尺寸限制
    ///   - completion: 流式输出回调
    public func generateAestheticAnalysisStream(imagePath: String,
                                              prompt: String,
                                              maxSize: Int = 1024,
                                              completion: @escaping (Result<String, Error>) -> Void) {
        do {
            // 加载并处理图片
            let ciImage = try ImageProcessor.loadAndProcessImage(from: imagePath, maxSize: maxSize)
            
            // 构建用于美学分析的提示词
            let aestheticPrompt = """
            请直接分析这张图片的以下方面：

            1. 构图：分析构图方式和元素布局的合理性
            2. 焦段：评价焦段选择和透视效果
            3. 对比度：判断明暗对比和色彩对比效果
            4. 曝光度：分析曝光准确性和细节呈现
            5. 亮度：评价整体亮度和分布情况

            开始分析：
            """
            
            // 使用真实的流式推理
            try pipeline.generateStream(
                prompt: aestheticPrompt,
                image: ciImage,
                maxNewTokens: maxTokens,
                temperature: temperature
            ) { chunk in
                completion(.success(chunk))
            }
        } catch {
            completion(.failure(error))
        }
    }
    
    /// 通用生成方法
    /// - Parameters:
    ///   - prompt: 文本提示
    ///   - imagePath: 图片路径（可选）
    ///   - maxSize: 图片最大尺寸限制
    /// - Returns: 生成的文本
    public func generate(prompt: String, imagePath: String? = nil, maxSize: Int = 1024) throws -> String {
        var ciImage: CIImage? = nil
        
        if let imagePath = imagePath {
            ciImage = try ImageProcessor.loadAndProcessImage(from: imagePath, maxSize: maxSize)
        }
        
        return try pipeline.generate(
            prompt: prompt,
            image: ciImage,
            maxNewTokens: maxTokens,
            temperature: temperature
        )
    }
    
    /// 清理内存
    public func clearMemory() {
        pipeline.clearCache()
        print("已清理 MLX 内存缓存")
    }
}

/// 便捷函数：对图片进行美学分析
/// - Parameters:
///   - imagePath: 图片路径
///   - promptFile: 提示词文件路径（可选）
///   - modelPath: 模型路径
///   - maxTokens: 最大生成token数
///   - temperature: 温度参数
///   - maxSize: 图片最大尺寸限制
/// - Returns: 美学分析结果
public func aestheticAnalysis(imagePath: String,
                            promptFile: String? = nil,
                            modelPath: String = "./Models/models--mlx-community--Qwen2.5-VL-3B-Instruct-4bit/snapshots/46d4cf06a06ffc1a766c214174f9cbed2f45bcab",
                            maxTokens: Int = 512,
                            temperature: Float = 0.0,
                            maxSize: Int = 1024) throws -> String {
    
    let inference = try QwenVLInference(modelPath: modelPath, maxTokens: maxTokens, temperature: temperature)
    
    // 读取提示词文件（如果提供）
    var prompt = "Analyze this image in detail, focusing on composition, colors, lighting, and overall aesthetic quality."
    
    if let promptFile = promptFile {
        do {
            let fileContent = try String(contentsOfFile: promptFile, encoding: .utf8)
            // 移除可能的 XML 标签
            let cleanedContent = fileContent
                .replacingOccurrences(of: "<prompt>", with: "")
                .replacingOccurrences(of: "</prompt>", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            
            if cleanedContent.count > 50 {
                prompt = """
                请直接分析这张图片的以下方面：

                1. 构图：分析构图方式和元素布局的合理性
                2. 焦段：评价焦段选择和透视效果
                3. 对比度：判断明暗对比和色彩对比效果
                4. 曝光度：分析曝光准确性和细节呈现
                5. 亮度：评价整体亮度和分布情况

                开始分析：
                """
            }
        } catch {
            print("提示词文件 \(promptFile) 读取失败，使用默认提示词")
        }
    }
    
    return try inference.generateAestheticAnalysis(imagePath: imagePath, prompt: prompt, maxSize: maxSize)
}

// MARK: - 示例用法

/*
 使用示例：
 
 // 基本用法
 do {
     let result = try aestheticAnalysis(imagePath: "demo.png")
     print(result)
 } catch {
     print("分析失败: \(error)")
 }
 
 // 高级用法
 let inference = QwenVLInference()
 
 // 流式输出
 inference.generateAestheticAnalysisStream(imagePath: "demo.png", prompt: "分析这张图片") { result in
     switch result {
     case .success(let chunk):
         print(chunk, terminator: "")
     case .failure(let error):
         print("错误: \(error)")
     }
 }
 */