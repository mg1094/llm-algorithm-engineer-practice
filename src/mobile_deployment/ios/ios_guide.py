"""
iOS移动端部署示例代码和使用指南
"""
# 这是一个文档文件，包含iOS部署的完整指南

IOS_DEPLOYMENT_GUIDE = """
# iOS移动端YOLO部署指南

## 概述

本指南介绍如何在iOS应用中部署YOLOv8模型，实现实时目标检测功能。

## 前置要求

1. macOS系统
2. Xcode 12.0+ （苹果官方的开发工具，用于构建和测试iOS应用）
3. iOS 13.0+
4. Python环境（用于模型转换）

## 步骤1: 模型转换

使用项目中的转换工具将YOLOv8模型转换为CoreML格式：

```python
from src.mobile_deployment.converters.yolo_converter import YOLOv8MobileConverter

converter = YOLOv8MobileConverter(model_size='n')
results = converter.convert_all(
    output_dir="./mobile/models",
    model_name="yolov8n"
)
```

转换后会生成 `yolov8n.mlmodel` 文件。

## 步骤2: 集成到Xcode项目

### 2.1 添加模型文件

1. 在Xcode中创建新项目或打开现有项目
2. 将 `.mlmodel` 文件拖拽到项目导航器中
3. 确保 "Copy items if needed" 和 "Add to targets" 已勾选

### 2.2 配置Info.plist

添加相机权限：

```xml
<key>NSCameraUsageDescription</key>
<string>需要访问相机以进行实时目标检测</string>
```

## 步骤3: Swift代码实现

### 3.1 创建YOLODetector类

```swift
import UIKit
import CoreML
import Vision
import AVFoundation

class YOLODetector {
    private var model: VNCoreMLModel?
    private var requests: [VNRequest] = []
    
    init() {
        setupModel()
    }
    
    private func setupModel() {
        guard let modelURL = Bundle.main.url(forResource: "yolov8n", withExtension: "mlmodel") else {
            fatalError("无法找到模型文件")
        }
        
        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            model = try VNCoreMLModel(for: mlModel)
            
            // 创建检测请求
            let request = VNCoreMLRequest(model: model!) { [weak self] request, error in
                self?.processDetections(request.results)
            }
            request.imageCropAndScaleOption = .scaleFill
            requests = [request]
        } catch {
            fatalError("模型加载失败: \\(error)")
        }
    }
    
    func detect(image: UIImage, completion: @escaping ([Detection]) -> Void) {
        guard let ciImage = CIImage(image: image) else {
            completion([])
            return
        }
        
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        
        do {
            try handler.perform(requests)
            // 检测结果会在processDetections中处理
        } catch {
            print("检测失败: \\(error)")
            completion([])
        }
    }
    
    private func processDetections(_ results: [Any]?) {
        guard let results = results as? [VNRecognizedObjectObservation] else {
            return
        }
        
        var detections: [Detection] = []
        for observation in results {
            let boundingBox = observation.boundingBox
            let confidence = observation.confidence
            
            let detection = Detection(
                boundingBox: boundingBox,
                confidence: Float(confidence),
                label: observation.labels.first?.identifier ?? "unknown"
            )
            detections.append(detection)
        }
        
        // 调用回调函数
        // completion(detections)
    }
}

struct Detection {
    let boundingBox: CGRect
    let confidence: Float
    let label: String
}
```

### 3.2 实时相机检测

```swift
import AVFoundation

class CameraViewController: UIViewController {
    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var detector: YOLODetector!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        detector = YOLODetector()
        setupCamera()
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high
        
        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { return }
        let videoInput: AVCaptureDeviceInput
        
        do {
            videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
        } catch {
            return
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        captureSession.startRunning()
    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, 
                      didOutput sampleBuffer: CMSampleBuffer, 
                      from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)
        
        // 执行检测（异步处理，避免阻塞相机）
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.detector.detect(image: uiImage) { detections in
                DispatchQueue.main.async {
                    // 更新UI显示检测结果
                    self?.updateDetections(detections)
                }
            }
        }
    }
    
    private func updateDetections(_ detections: [Detection]) {
        // 在界面上绘制检测框
        // 实现UI更新逻辑
    }
}
```

## 步骤4: 性能优化

### 4.1 降低输入分辨率

```swift
// 使用较小的输入尺寸可以显著提升速度
let request = VNCoreMLRequest(model: model!)
request.imageCropAndScaleOption = .scaleFill
```

### 4.2 使用Metal加速

CoreML会自动使用Metal进行GPU加速，无需额外配置。

### 4.3 批处理优化

对于图片检测，可以批量处理多张图片：

```swift
let handler = VNImageRequestHandler(ciImages: images, options: [:])
```

## 步骤5: 测试和调试

1. 在真机上测试性能（模拟器性能不准确）
2. 监控内存使用情况
3. 测试不同光照条件下的检测效果
4. 优化NMS阈值和置信度阈值

## 常见问题

### Q: 模型太大怎么办？
A: 使用YOLOv8n（nano）版本，或者进一步量化模型。

### Q: 检测速度慢？
A: 
- 降低输入分辨率
- 使用GPU加速（Metal）
- 减少检测的类别数量

### Q: 内存占用过高？
A:
- 优化图像预处理
- 及时释放不需要的资源
- 使用内存池

## 参考资料

- [CoreML官方文档](https://developer.apple.com/documentation/coreml)
- [Vision框架文档](https://developer.apple.com/documentation/vision)
- [AVFoundation文档](https://developer.apple.com/documentation/avfoundation)
"""

# 保存为Markdown文件
if __name__ == "__main__":
    with open("docs/mobile/ios_deployment_guide.md", "w", encoding="utf-8") as f:
        f.write(IOS_DEPLOYMENT_GUIDE)
    print("iOS部署指南已生成: docs/mobile/ios_deployment_guide.md")

