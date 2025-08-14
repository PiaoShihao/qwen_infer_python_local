// Simplified Qwen2.5-VL Example
// Copyright © 2024 Apple Inc.

import Foundation
import QwenVLSwift

@main
struct SimplifiedQwenVLExample {
    static func main() {
        print("Simplified Qwen2.5-VL Swift Example")
        print("=" * 50)
        
        // Test basic model initialization
        testModelInitialization()
        
        // Test aesthetic analysis function
        testAestheticAnalysis()
        
        print("\n示例运行完成")
    }
    
    /// 测试模型初始化
    static func testModelInitialization() {
        print("\n测试 1: 模型初始化")
        print("-" * 30)
        
        do {
            print("正在初始化 Simplified Qwen2.5-VL 模型...")
            let inference = try SimplifiedQwenVLInference()
            print("✅ 模型初始化成功")
            
            // Test text generation without image
            let result = try inference.generate(prompt: "Hello, world!")
            print("文本生成测试结果: \(result)")
            
        } catch {
            print("❌ 模型初始化失败: \(error.localizedDescription)")
        }
    }
    
    /// 测试美学分析功能
    static func testAestheticAnalysis() {
        print("\n测试 2: 美学分析")
        print("-" * 30)
        
        let testImages = ["demo.png", "demo2.png", "demo3.png"]
        
        for imagePath in testImages {
            print("\n正在分析图片: \(imagePath)")
            
            // Check if image exists
            if !FileManager.default.fileExists(atPath: imagePath) {
                print("⚠️  图片文件不存在: \(imagePath)")
                continue
            }
            
            do {
                let result = try aestheticAnalysis(imagePath: imagePath)
                print("✅ 分析结果: \(result)")
            } catch {
                print("❌ 分析失败: \(error.localizedDescription)")
            }
        }
    }
}

// Extension to support string repetition
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}

// MARK: - Additional Test Functions

struct ImageValidationTest {
    static func validateImages() {
        print("\n图片验证测试")
        print("-" * 30)
        
        let imagePaths = ["demo.png", "demo2.png", "demo3.png", "demo.HEIC"]
        
        for imagePath in imagePaths {
            let exists = FileManager.default.fileExists(atPath: imagePath)
            let status = exists ? "✓" : "✗"
            print("\(status) \(imagePath)")
        }
    }
}

struct ModelConfigTest {
    static func testConfiguration() {
        print("\n配置测试")
        print("-" * 30)
        
        let config = QwenVLConfig()
        print("文本模型配置:")
        print("  - 隐藏层大小: \(config.textConfig.hiddenSize)")
        print("  - 层数: \(config.textConfig.numLayers)")
        print("  - 注意力头数: \(config.textConfig.numAttentionHeads)")
        
        print("视觉模型配置:")
        print("  - 隐藏层大小: \(config.visionConfig.hiddenSize)")
        print("  - 图片大小: \(config.visionConfig.imageSize)")
        print("  - 补丁大小: \(config.visionConfig.patchSize)")
        print("  - 层数: \(config.visionConfig.numLayers)")
    }
}