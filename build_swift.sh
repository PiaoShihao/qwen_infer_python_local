#!/bin/bash

# Qwen VL Swift 构建脚本
# Copyright © 2024 Apple Inc.

set -e

echo "🚀 开始构建 Qwen2.5-VL Swift 完整实现..."

# 检查 Xcode 是否安装
if ! command -v xcodebuild &> /dev/null; then
    echo "❌ 错误: 未找到 xcodebuild。请确保已安装 Xcode。"
    exit 1
fi

# 检查 Swift 版本
SWIFT_VERSION=$(swift --version | head -n1)
echo "📋 使用 Swift 版本: $SWIFT_VERSION"

# 清理之前的构建
echo "🧹 清理之前的构建..."
rm -rf .build

# 创建必要的目录
echo "📁 创建目录结构..."
mkdir -p Models
mkdir -p .build

# 检查模型文件是否存在
MODEL_PATH="./Models/models--mlx-community--Qwen2.5-VL-3B-Instruct-4bit"
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  警告: 模型文件不存在于 $MODEL_PATH"
    echo "   请确保已下载 Qwen2.5-VL-3B 模型文件"
fi

# 检查测试图片是否存在
for img in "demo.png" "demo2.png" "demo3.png"; do
    if [ ! -f "$img" ]; then
        echo "⚠️  警告: 测试图片 $img 不存在"
    fi
done

# 构建选项
BUILD_CONFIG="debug"
DESTINATION="platform=macOS"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --release)
            BUILD_CONFIG="release"
            shift
            ;;
        --ios)
            DESTINATION="platform=iOS Simulator,name=iPhone 15"
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --release    构建 release 版本 (默认: debug)"
            echo "  --ios        为 iOS 模拟器构建 (默认: macOS)"
            echo "  --help       显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

echo "🔧 构建配置: $BUILD_CONFIG"
echo "🎯 目标平台: $DESTINATION"

# 解析依赖
echo "📦 解析 Swift 包依赖..."
swift package resolve

# 构建库
echo "🏗️  构建 QwenVLSwift 库..."
xcodebuild build \
    -scheme QwenVLSwift-Package \
    -destination "$DESTINATION" \
    -configuration $BUILD_CONFIG \
    -quiet

if [ $? -eq 0 ]; then
    echo "✅ QwenVLSwift 库构建成功"
else
    echo "❌ QwenVLSwift 库构建失败"
    exit 1
fi

# 构建示例程序 (仅在 macOS 下)
if [[ "$DESTINATION" == "platform=macOS" ]]; then
    echo "🏗️  构建配置测试程序..."
    swift build --target ConfigTest

    if [ $? -eq 0 ]; then
        echo "✅ 配置测试程序构建成功"
        echo "🚀 运行配置测试程序..."
        swift run ConfigTest
    else
        echo "❌ 配置测试程序构建失败"
        exit 1
    fi
fi

# 运行测试
echo "🧪 运行单元测试..."
xcodebuild test \
    -scheme QwenVLSwift-Package \
    -destination "$DESTINATION" \
    -quiet

if [ $? -eq 0 ]; then
    echo "✅ 所有测试通过"
else
    echo "⚠️  一些测试失败"
fi
