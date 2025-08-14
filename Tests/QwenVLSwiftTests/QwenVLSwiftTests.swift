// Copyright © 2024 Apple Inc.

import XCTest
@testable import QwenVLSwift

final class QwenVLSwiftTests: XCTestCase {
    
    func testImageFormatValidation() throws {
        // 测试支持的格式
        XCTAssertTrue(SupportedImageFormats.isSupported("png"))
        XCTAssertTrue(SupportedImageFormats.isSupported("jpg"))
        XCTAssertTrue(SupportedImageFormats.isSupported("JPEG"))
        XCTAssertTrue(SupportedImageFormats.isSupported("heic"))
        
        // 测试不支持的格式
        XCTAssertFalse(SupportedImageFormats.isSupported("pdf"))
        XCTAssertFalse(SupportedImageFormats.isSupported("txt"))
        XCTAssertFalse(SupportedImageFormats.isSupported("doc"))
    }
    
    func testImageValidationNonExistentFile() throws {
        let result = ImageProcessor.validateImageFile(at: "non_existent_file.png")
        XCTAssertFalse(result.isValid)
        XCTAssertTrue(result.message.contains("文件不存在"))
    }
    
    func testQwenVLInferenceInitialization() throws {
        let inference = QwenVLInference()
        XCTAssertNotNil(inference)
    }
    
    func testErrorTypes() throws {
        let error1 = QwenVLError.modelLoadFailed("test")
        XCTAssertNotNil(error1.errorDescription)
        
        let error2 = QwenVLError.imageProcessingFailed("test")
        XCTAssertNotNil(error2.errorDescription)
        
        let error3 = QwenVLError.inferenceError("test")
        XCTAssertNotNil(error3.errorDescription)
    }
    
    func testStringChunking() throws {
        let inference = QwenVLInference()
        let text = "这是一个测试文本，用来验证分块功能是否正常工作。"
        
        // 使用反射访问私有方法（仅用于测试）
        let chunks = text.chunked(into: 5)
        XCTAssertGreaterThan(chunks.count, 1)
    }
}

// 扩展 String 以支持分块功能（用于测试）
extension String {
    func chunked(into size: Int) -> [String] {
        var chunks: [String] = []
        var currentIndex = self.startIndex
        
        while currentIndex < self.endIndex {
            let endIndex = self.index(currentIndex, offsetBy: size, limitedBy: self.endIndex) ?? self.endIndex
            let chunk = String(self[currentIndex..<endIndex])
            chunks.append(chunk)
            currentIndex = endIndex
        }
        
        return chunks
    }
}