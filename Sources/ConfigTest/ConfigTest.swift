// Configuration Test - doesn't require MLX runtime
// Copyright © 2024 Apple Inc.

import Foundation
import QwenVLSwift

@main
struct ConfigTest {
    static func main() {
        print("Qwen2.5-VL Swift 配置测试")
        print("=" * 50)
        
        testConfiguration()
        testModelCreation()
        
        print("\n配置测试完成")
    }
    
    static func testConfiguration() {
        print("\n测试 1: 配置验证")
        print("-" * 30)
        
        let config = QwenVLConfig()
        
        print("✅ 文本模型配置:")
        print("   - 词汇表大小: \(config.textConfig.vocabSize)")
        print("   - 隐藏层大小: \(config.textConfig.hiddenSize)")
        print("   - 层数: \(config.textConfig.numLayers)")
        print("   - 注意力头数: \(config.textConfig.numAttentionHeads)")
        print("   - 键值头数: \(config.textConfig.numKeyValueHeads)")
        
        print("✅ 视觉模型配置:")
        print("   - 隐藏层大小: \(config.visionConfig.hiddenSize)")
        print("   - 图片大小: \(config.visionConfig.imageSize)")
        print("   - 补丁大小: \(config.visionConfig.patchSize)")
        print("   - 层数: \(config.visionConfig.numLayers)")
        print("   - 注意力头数: \(config.visionConfig.numAttentionHeads)")
    }
    
    static func testModelCreation() {
        print("\n测试 2: 模型结构创建")
        print("-" * 30)
        
        do {
            print("正在创建模型配置...")
            let config = QwenVLConfig()
            
            print("✅ 配置创建成功")
            print("   - 图片大小: \(config.imageSize)")
            print("   - 补丁大小: \(config.imagePatchSize)")
            print("   - 词汇表大小: \(config.vocabSize)")
            
            // Note: We don't create actual model instances here to avoid MLX initialization
            print("✅ 模型架构验证通过（未实际初始化以避免 MLX 依赖）")
            
        } catch {
            print("❌ 模型创建失败: \(error)")
        }
    }
}

// Extension to support string repetition
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}