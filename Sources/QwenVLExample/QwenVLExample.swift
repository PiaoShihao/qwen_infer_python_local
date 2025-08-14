// Copyright © 2024 Apple Inc.

import Foundation
import QwenVLSwift

/// 示例程序：演示 Qwen2.5-VL Swift 完整推理功能
@main
struct QwenVLExample {
    static func main() {
        print("Qwen2.5-VL Swift 完整推理示例")
        print("=" * 50)
        
        // 示例 1: 基本美学分析
        testBasicAestheticAnalysis()
        
        // 示例 2: 流式输出
        testStreamingAnalysis()
        
        // 示例 3: 自定义参数
        testCustomParameters()
        
        // 示例 4: 直接使用推理管道
        testDirectPipelineUsage()
        
        print("\n示例运行完成")
    }
    
    /// 测试基本美学分析功能
    static func testBasicAestheticAnalysis() {
        print("\n测试 1: 基本美学分析")
        print("-" * 30)
        
        let imagePath = "demo.png"
        
        do {
            print("正在初始化 Qwen2.5-VL 模型...")
            let result = try aestheticAnalysis(imagePath: imagePath)
            print("分析结果:")
            print(result)
        } catch {
            print("分析失败: \(error.localizedDescription)")
        }
    }
    
    /// 测试流式输出功能
    static func testStreamingAnalysis() {
        print("\n测试 2: 流式输出")
        print("-" * 30)
        
        let imagePath = "demo2.png"
        
        do {
            print("正在初始化推理系统...")
            let inference = try QwenVLInference()
            
            let group = DispatchGroup()
            group.enter()
            
            print("开始流式分析...")
            inference.generateAestheticAnalysisStream(
                imagePath: imagePath,
                prompt: "请分析这张图片的美学特点"
            ) { result in
                switch result {
                case .success(let chunk):
                    print(chunk, terminator: "")
                    fflush(stdout)
                case .failure(let error):
                    print("\n流式分析错误: \(error.localizedDescription)")
                    group.leave()
                }
            }
            
            // 等待流式输出完成
            DispatchQueue.global().asyncAfter(deadline: .now() + 10) {
                group.leave()
            }
            
            group.wait()
            print("\n流式分析完成")
        } catch {
            print("流式分析初始化失败: \(error.localizedDescription)")
        }
    }
    
    /// 测试自定义参数
    static func testCustomParameters() {
        print("\n测试 3: 自定义参数")
        print("-" * 30)
        
        let imagePath = "demo3.png"
        let promptFile = "qwen_vl_3b_prompt.txt"
        
        do {
            print("使用自定义参数进行分析...")
            let result = try aestheticAnalysis(
                imagePath: imagePath,
                promptFile: promptFile,
                maxTokens: 256,
                temperature: 0.1,
                maxSize: 512
            )
            print("自定义参数分析结果:")
            print(result)
        } catch {
            print("自定义参数分析失败: \(error.localizedDescription)")
        }
    }
    
    /// 测试直接使用推理管道
    static func testDirectPipelineUsage() {
        print("\n测试 4: 直接使用推理管道")
        print("-" * 30)
        
        do {
            print("正在初始化 Qwen2.5-VL 推理管道...")
            let pipeline = try Qwen25VLInferencePipeline(
                modelPath: "./Models/models--mlx-community--Qwen2.5-VL-3B-Instruct-4bit/snapshots/46d4cf06a06ffc1a766c214174f9cbed2f45bcab"
            )
            
            // 测试纯文本生成
            print("\n4.1 纯文本生成测试:")
            let textResult = try pipeline.generate(
                prompt: "请介绍一下人工智能的发展历史",
                maxNewTokens: 100,
                temperature: 0.3
            )
            print("纯文本生成结果: \(textResult)")
            
            // 测试图文混合
            print("\n4.2 图文混合推理测试:")
            if let ciImage = ImageProcessor.loadAndProcessImage(from: "demo.png", maxSize: 512) {
                let multimodalResult = try pipeline.generate(
                    prompt: "请详细描述这张图片的内容和美学特点",
                    image: ciImage,
                    maxNewTokens: 200,
                    temperature: 0.2
                )
                print("多模态分析结果: \(multimodalResult)")
            }
            
            // 清理缓存
            pipeline.clearCache()
            print("测试完成，已清理缓存")
            
        } catch {
            print("直接管道测试失败: \(error.localizedDescription)")
        }
    }
}

/// 图片验证示例
struct ImageValidationExample {
    static func validateImages() {
        print("\n图片验证示例")
        print("-" * 30)
        
        let imagePaths = ["demo.png", "demo2.png", "demo3.png", "demo.HEIC"]
        
        for imagePath in imagePaths {
            let validation = ImageProcessor.validateImageFile(at: imagePath)
            let status = validation.isValid ? "✓" : "✗"
            print("\(status) \(imagePath): \(validation.message)")
        }
    }
}

/// 支持格式信息示例
struct SupportedFormatsExample {
    static func showSupportedFormats() {
        print("\n支持的图片格式")
        print("-" * 30)
        
        print("支持的扩展名:")
        for ext in SupportedImageFormats.supportedExtensions.sorted() {
            print("  .\(ext)")
        }
        
        print("\n格式检查示例:")
        let testExtensions = ["png", "jpg", "heic", "pdf", "txt"]
        for ext in testExtensions {
            let supported = SupportedImageFormats.isSupported(ext)
            let status = supported ? "✓" : "✗"
            print("  \(status) .\(ext)")
        }
    }
}

// 扩展 String 以支持重复操作符
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

/// 使用指南
/*
 编译和运行指南:
 
 1. 确保安装了 mlx-swift 依赖:
    - 在 Xcode 中添加 mlx-swift 包依赖
    - 或在 Package.swift 中添加依赖
 
 2. 编译:
    xcodebuild build -scheme QwenVLExample -destination 'platform=OS X'
 
 3. 运行:
    ./QwenVLExample
 
 4. iOS 集成:
    - 将 QwenVLInference.swift 添加到 iOS 项目
    - 确保在 iOS 设备上有足够的内存
    - 使用较小的 maxSize 参数以减少内存使用
 
 示例代码在 iOS 中的使用:
 
 ```swift
 class ViewController: UIViewController {
     let inference = QwenVLInference()
     
     func analyzeImage(_ image: UIImage) {
         // 保存临时图片文件
         let tempURL = // ... 创建临时文件
         
         inference.generateAestheticAnalysisStream(
             imagePath: tempURL.path,
             prompt: "分析这张照片",
             maxSize: 512  // iOS 上使用较小尺寸
         ) { result in
             DispatchQueue.main.async {
                 switch result {
                 case .success(let chunk):
                     // 更新 UI
                 case .failure(let error):
                     // 处理错误
                 }
             }
         }
     }
 }
 ```
 */