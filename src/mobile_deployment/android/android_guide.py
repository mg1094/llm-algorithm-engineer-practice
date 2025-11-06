"""
Android移动端部署示例代码和使用指南
"""
# 这是一个文档文件，包含Android部署的完整指南

ANDROID_DEPLOYMENT_GUIDE = """
# Android移动端YOLO部署指南

## 概述

本指南介绍如何在Android应用中部署YOLOv8模型，实现实时目标检测功能。

## 前置要求

1. Android Studio Arctic Fox或更高版本
2. Android SDK API 21+ (Android 5.0+)
3. Python环境（用于模型转换）
4. NDK（可选，用于原生代码）

## 步骤1: 模型转换

使用项目中的转换工具将YOLOv8模型转换为TFLite格式：

```python
from src.mobile_deployment.converters.yolo_converter import YOLOv8MobileConverter

converter = YOLOv8MobileConverter(model_size='n')
results = converter.convert_all(
    output_dir="./mobile/models",
    model_name="yolov8n"
)
```

转换后会生成 `yolov8n.tflite` 文件。

## 步骤2: 集成到Android项目

### 2.1 添加依赖

在 `app/build.gradle` 中添加TensorFlow Lite依赖：

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'  // GPU加速（可选）
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

### 2.2 添加模型文件

1. 在 `app/src/main/assets/` 目录下创建 `models` 文件夹
2. 将 `yolov8n.tflite` 文件复制到该目录

### 2.3 配置权限

在 `AndroidManifest.xml` 中添加相机权限：

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera" android:required="true" />
```

## 步骤3: Kotlin代码实现

### 3.1 创建YOLODetector类

```kotlin
package com.example.yolodetector

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

class YOLODetector(context: Context) {
    private var interpreter: Interpreter? = null
    private val inputImageWidth = 640
    private val inputImageHeight = 640
    private val numClasses = 80  // COCO数据集类别数
    private val numBoxes = 8400  // YOLOv8默认anchor数量
    private val confidenceThreshold = 0.5f
    private val nmsThreshold = 0.45f
    
    init {
        loadModel(context)
    }
    
    private fun loadModel(context: Context) {
        try {
            val model = loadModelFile(context, "yolov8n.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4)  // 使用4个线程
                // 使用GPU加速（可选）
                // setUseGpuDelegateV2(true)
            }
            interpreter = Interpreter(model, options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    private fun loadModelFile(context: Context, modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    fun detect(bitmap: Bitmap): List<Detection> {
        val resizedBitmap = Bitmap.createScaledBitmap(
            bitmap, inputImageWidth, inputImageHeight, true
        )
        
        val inputBuffer = preprocessImage(resizedBitmap)
        val outputBuffer = Array(1) { Array(numBoxes) { FloatArray(4 + numClasses) } }
        
        interpreter?.run(inputBuffer, outputBuffer)
        
        return postprocess(outputBuffer[0], bitmap.width, bitmap.height)
    }
    
    private fun preprocessImage(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val inputBuffer = Array(1) {
            Array(inputImageHeight) {
                Array(inputImageWidth) { FloatArray(3) }
            }
        }
        
        val intValues = IntArray(inputImageWidth * inputImageHeight)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        
        for (y in 0 until inputImageHeight) {
            for (x in 0 until inputImageWidth) {
                val pixel = intValues[y * inputImageWidth + x]
                
                // 归一化到[0, 1]
                inputBuffer[0][y][x][0] = ((pixel shr 16) and 0xFF) / 255.0f
                inputBuffer[0][y][x][1] = ((pixel shr 8) and 0xFF) / 255.0f
                inputBuffer[0][y][x][2] = (pixel and 0xFF) / 255.0f
            }
        }
        
        return inputBuffer
    }
    
    private fun postprocess(
        output: Array<FloatArray>,
        imageWidth: Int,
        imageHeight: Int
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        
        for (i in output.indices) {
            val box = output[i]
            
            // 提取边界框坐标和置信度
            val x = box[0]
            val y = box[1]
            val w = box[2]
            val h = box[3]
            
            // 找到最大置信度的类别
            var maxConf = 0f
            var maxClass = 0
            for (j in 4 until box.size) {
                if (box[j] > maxConf) {
                    maxConf = box[j]
                    maxClass = j - 4
                }
            }
            
            val confidence = maxConf
            if (confidence >= confidenceThreshold) {
                // 转换坐标到原始图像尺寸
                val left = (x - w / 2) * imageWidth / inputImageWidth
                val top = (y - h / 2) * imageHeight / inputImageHeight
                val right = (x + w / 2) * imageWidth / inputImageWidth
                val bottom = (y + h / 2) * imageHeight / inputImageHeight
                
                detections.add(
                    Detection(
                        boundingBox = android.graphics.RectF(
                            left, top, right, bottom
                        ),
                        confidence = confidence,
                        classId = maxClass
                    )
                )
            }
        }
        
        // 非极大值抑制（NMS）
        return nms(detections)
    }
    
    private fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val selected = mutableListOf<Detection>()
        
        for (detection in sorted) {
            var shouldAdd = true
            for (selectedDetection in selected) {
                val iou = calculateIoU(detection.boundingBox, selectedDetection.boundingBox)
                if (iou > nmsThreshold) {
                    shouldAdd = false
                    break
                }
            }
            if (shouldAdd) {
                selected.add(detection)
            }
        }
        
        return selected
    }
    
    private fun calculateIoU(box1: android.graphics.RectF, box2: android.graphics.RectF): Float {
        val intersectionLeft = max(box1.left, box2.left)
        val intersectionTop = max(box1.top, box2.top)
        val intersectionRight = min(box1.right, box2.right)
        val intersectionBottom = min(box1.bottom, box2.bottom)
        
        if (intersectionRight <= intersectionLeft || intersectionBottom <= intersectionTop) {
            return 0f
        }
        
        val intersectionArea = (intersectionRight - intersectionLeft) * 
                              (intersectionBottom - intersectionTop)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = box1Area + box2Area - intersectionArea
        
        return intersectionArea / unionArea
    }
    
    fun close() {
        interpreter?.close()
    }
}

data class Detection(
    val boundingBox: android.graphics.RectF,
    val confidence: Float,
    val classId: Int
)
```

### 3.2 实时相机检测

```kotlin
package com.example.yolodetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.hardware.camera2.*
import android.os.Bundle
import android.view.Surface
import android.view.SurfaceView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

class CameraActivity : AppCompatActivity() {
    private lateinit var cameraManager: CameraManager
    private var cameraDevice: CameraDevice? = null
    private lateinit var detector: YOLODetector
    private var imageReader: ImageReader? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        detector = YOLODetector(this)
        cameraManager = getSystemService(CAMERA_SERVICE) as CameraManager
        
        if (checkCameraPermission()) {
            openCamera()
        } else {
            requestCameraPermission()
        }
    }
    
    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }
    
    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST_CODE
        )
    }
    
    private fun openCamera() {
        try {
            val cameraId = cameraManager.cameraIdList[0]
            val characteristics = cameraManager.getCameraCharacteristics(cameraId)
            val streamConfigurationMap = characteristics.get(
                CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
            )
            
            val imageDimension = streamConfigurationMap?.getOutputSizes(
                ImageReader::class.java
            )?.maxByOrNull { it.width * it.height }
            
            imageReader = ImageReader.newInstance(
                imageDimension?.width ?: 1920,
                imageDimension?.height ?: 1080,
                ImageFormat.YUV_420_888,
                2
            )
            
            imageReader?.setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage()
                processImage(image)
                image?.close()
            }, null)
            
            if (ActivityCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                return
            }
            
            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    createCaptureSession()
                }
                
                override fun onDisconnected(camera: CameraDevice) {
                    camera.close()
                    cameraDevice = null
                }
                
                override fun onError(camera: CameraDevice, error: Int) {
                    camera.close()
                    cameraDevice = null
                    Toast.makeText(this@CameraActivity, "相机错误", Toast.LENGTH_SHORT).show()
                }
            }, null)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    private fun createCaptureSession() {
        val surface = imageReader?.surface
        val captureRequestBuilder = cameraDevice?.createCaptureRequest(
            CameraDevice.TEMPLATE_PREVIEW
        )
        captureRequestBuilder?.addTarget(surface)
        
        cameraDevice?.createCaptureSession(
            listOf(surface!!),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    val captureRequest = captureRequestBuilder?.build()
                    session.setRepeatingRequest(captureRequest!!, null, null)
                }
                
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Toast.makeText(this@CameraActivity, "配置失败", Toast.LENGTH_SHORT).show()
                }
            },
            null
        )
    }
    
    private fun processImage(image: android.media.Image) {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        
        // 在后台线程执行检测
        Thread {
            val detections = detector.detect(bitmap)
            
            // 在主线程更新UI
            runOnUiThread {
                updateDetections(detections)
            }
        }.start()
    }
    
    private fun updateDetections(detections: List<Detection>) {
        // 在界面上绘制检测框
        // 实现UI更新逻辑
    }
    
    override fun onDestroy() {
        super.onDestroy()
        detector.close()
        cameraDevice?.close()
    }
    
    companion object {
        private const val CAMERA_PERMISSION_REQUEST_CODE = 1001
    }
}
```

## 步骤4: 性能优化

### 4.1 使用GPU加速

```kotlin
val options = Interpreter.Options().apply {
    val delegate = GpuDelegate()
    addDelegate(delegate)
}
interpreter = Interpreter(model, options)
```

### 4.2 使用NNAPI（Android 8.1+）

```kotlin
val options = Interpreter.Options().apply {
    setUseNNAPI(true)
}
interpreter = Interpreter(model, options)
```

### 4.3 降低输入分辨率

```kotlin
private val inputImageWidth = 416  // 从640降低到416
private val inputImageHeight = 416
```

## 步骤5: 测试和调试

1. 在真机上测试性能（模拟器性能不准确）
2. 使用Android Profiler监控内存和CPU
3. 测试不同Android版本的兼容性
4. 优化NMS阈值和置信度阈值

## 常见问题

### Q: 模型太大怎么办？
A: 使用量化后的TFLite模型，或者使用YOLOv8n版本。

### Q: 检测速度慢？
A: 
- 使用GPU加速（GpuDelegate）
- 使用NNAPI
- 降低输入分辨率
- 减少检测的类别数量

### Q: 内存占用过高？
A:
- 优化图像预处理
- 及时释放Image对象
- 使用对象池

## 参考资料

- [TensorFlow Lite官方文档](https://www.tensorflow.org/lite)
- [Android相机2 API文档](https://developer.android.com/reference/android/hardware/camera2/package-summary)
- [Kotlin官方文档](https://kotlinlang.org/docs/home.html)
"""

# 保存为Markdown文件
if __name__ == "__main__":
    with open("docs/mobile/android_deployment_guide.md", "w", encoding="utf-8") as f:
        f.write(ANDROID_DEPLOYMENT_GUIDE)
    print("Android部署指南已生成: docs/mobile/android_deployment_guide.md")

