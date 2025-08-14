// swift-tools-version: 5.8
// Copyright © 2024 Apple Inc.

import PackageDescription

let package = Package(
    name: "QwenVLSwift",
    platforms: [
        .macOS(.v13),
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
        .testTarget(
            name: "QwenVLSwiftTests",
            dependencies: ["QwenVLSwift"],
            path: "Tests/QwenVLSwiftTests"
        )
    ]
)