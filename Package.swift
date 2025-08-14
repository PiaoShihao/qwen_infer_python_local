// swift-tools-version: 5.8
// Copyright © 2024 Apple Inc.

import PackageDescription

let package = Package(
    name: "QwenVLSwift",
    platforms: [
        .macOS("13.3"),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "QwenVLSwift",
            targets: ["QwenVLSwift"]
        ),
        .executable(
            name: "QwenVLExample",
            targets: ["QwenVLExample"]
        ),
        .executable(
            name: "ConfigTest", 
            targets: ["ConfigTest"]
        )
    ],
    dependencies: [
        // MLX Swift 依赖
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.10.0")
    ],
    targets: [
        .target(
            name: "QwenVLSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Sources/QwenVLSwift"
        ),
        .executableTarget(
            name: "QwenVLExample",
            dependencies: [
                "QwenVLSwift",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ],
            path: "Sources/QwenVLExample"
        ),
        .executableTarget(
            name: "ConfigTest",
            dependencies: [
                "QwenVLSwift"
            ],
            path: "Sources/ConfigTest"
        ),
        .testTarget(
            name: "QwenVLSwiftTests",
            dependencies: ["QwenVLSwift"],
            path: "Tests/QwenVLSwiftTests"
        )
    ]
)