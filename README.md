# 大模型算法工程师实践项目

这是一个综合性的深度学习实践项目，涵盖了**大模型算法工程师**岗位所需的核心技能栈。

## 📋 项目概述

本项目整合了以下核心功能模块：

- ✅ **数据处理** - 使用Pandas和NumPy进行数据采集、清洗、预处理和深度分析
- ✅ **Web爬虫** - Python网页数据采集工具
- ✅ **模型训练与评估** - PyTorch模型训练、评估和可视化
- ✅ **模型优化** - 量化、剪枝、蒸馏等优化技术
- ✅ **多模态识别** - 图像识别、视频识别、声纹识别
- ✅ **模型部署** - Flask、FastAPI、Gradio等多种部署方式
- ✅ **移动端部署** - YOLOv8模型在iOS/Android上的部署（CoreML/TFLite）

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python main.py
```

## 📁 项目结构

```
sheng_cheng/
├── src/                          # 源代码目录
│   ├── data_processing/         # 数据处理模块
│   │   └── processor.py         # 数据处理类
│   ├── scraper/                 # Web爬虫模块
│   │   └── web_scraper.py       # 爬虫类
│   ├── model/                   # 模型训练模块
│   │   └── trainer.py           # 训练器和评估器
│   ├── optimization/            # 模型优化模块
│   │   └── optimizer.py         # 量化、剪枝、蒸馏
│   ├── multimodal/              # 多模态识别模块
│   │   └── recognizer.py        # 图像、视频、声纹识别
│   ├── deployment/              # 模型部署模块
│   │   └── server.py            # Flask/FastAPI/Gradio服务
│   └── mobile_deployment/       # 移动端部署模块
│       ├── converters/          # 模型转换器
│       │   └── yolo_converter.py # YOLOv8转换工具
│       ├── utils/               # 工具函数
│       │   └── performance_evaluator.py # 性能评估
│       ├── ios/                 # iOS部署示例
│       └── android/             # Android部署示例
├── data/                         # 数据目录
│   ├── raw/                     # 原始数据
│   ├── processed/               # 处理后的数据
│   └── cache/                   # 缓存数据
├── models/                       # 模型文件目录
├── mobile/                       # 移动端文件目录
│   ├── models/                  # 移动端模型文件
│   ├── ios/                     # iOS项目文件
│   └── android/                 # Android项目文件
├── docs/                         # 文档目录
│   └── mobile/                  # 移动端部署文档
├── checkpoints/                 # 模型检查点
├── logs/                        # 日志文件
├── config.yaml                  # 配置文件
├── requirements.txt             # 依赖包列表
├── main.py                      # 主程序入口
└── README.md                    # 项目说明文档
```

## 🔧 核心功能模块

### 1. 数据处理模块 (`src/data_processing/`)

使用Pandas和NumPy进行数据采集、清洗、预处理和深度分析。

**主要功能：**
- 数据加载（CSV、Excel、JSON、Parquet）
- 数据清洗（去重、缺失值处理）
- 特征工程（标准化、编码）
- 深度数据分析（统计、相关性分析）
- 训练集/测试集划分

**示例用法：**

```python
from src.data_processing.processor import DataProcessor

processor = DataProcessor()
df = processor.load_data("data.csv")
cleaned_df = processor.data_cleaning(df)
processed_df = processor.feature_engineering(cleaned_df)
analysis = processor.analyze_data(processed_df)
```

### 2. Web爬虫模块 (`src/scraper/`)

Python网页数据采集工具，支持文本、链接、图片提取。

**主要功能：**
- 网页内容获取
- HTML解析
- 文本、链接、图片提取
- 批量爬取
- 数据保存

**示例用法：**

```python
from src.scraper.web_scraper import WebScraper

scraper = WebScraper(delay=1.0)
data = scraper.scrape(
    'https://example.com',
    extract_text=True,
    extract_links=True,
    extract_images=True
)
```

### 3. 模型训练与评估模块 (`src/model/`)

使用PyTorch进行模型训练、评估和可视化。

**主要功能：**
- 数据集类（TextDataset）
- 模型训练器（ModelTrainer）
- 模型评估器（ModelEvaluator）
- 混淆矩阵可视化
- 训练历史记录

**示例用法：**

```python
from src.model.trainer import ModelTrainer, SimpleCNN

model = SimpleCNN(vocab_size=10000, num_classes=2)
trainer = ModelTrainer(model)
history = trainer.train(train_loader, val_loader, num_epochs=10)
```

### 4. 模型优化模块 (`src/optimization/`)

实现量化、剪枝、蒸馏等模型优化技术。

**主要功能：**
- 动态/静态量化（8位、4位）
- 权重剪枝（L1、L2、随机）
- 知识蒸馏
- ONNX模型导出
- 模型大小统计

**示例用法：**

```python
from src.optimization.optimizer import ModelOptimizer

optimizer = ModelOptimizer(model)
optimized_model = optimizer.optimize(
    quantization=True,
    pruning=True,
    pruning_ratio=0.3,
    quantization_bits=8
)
```

### 5. 多模态识别模块 (`src/multimodal/`)

图像识别、视频识别、声纹识别功能。

**主要功能：**
- 图像识别（特征提取、目标检测、分类）
- 视频识别（帧提取、场景检测、分类）
- 声纹识别（MFCC特征、声纹验证、说话人识别）
- 多模态统一接口

**示例用法：**

```python
from src.multimodal.recognizer import MultimodalRecognizer

recognizer = MultimodalRecognizer()
result = recognizer.recognize("image.jpg", media_type="image")
```

### 6. 模型部署模块 (`src/deployment/`)

使用Flask、FastAPI、Gradio部署模型服务。

**主要功能：**
- Flask REST API服务
- FastAPI高性能服务
- Gradio交互界面
- 模型导出（TorchScript、ONNX）

**示例用法：**

```python
from src.deployment.server import ModelDeploymentManager

manager = ModelDeploymentManager(model, "my_model")
manager.deploy_fastapi(host="0.0.0.0", port=8000)
# 或
manager.deploy_gradio(input_type="text", server_port=7860)
```

### 7. 移动端部署模块 (`src/mobile_deployment/`) 🆕

YOLOv8模型在iOS和Android移动端的部署工具和示例代码。

**主要功能：**
- YOLOv8模型转换（PyTorch → ONNX → CoreML/TFLite）
- iOS部署（Swift + CoreML）
- Android部署（Kotlin + TFLite）
- 性能评估工具（推理速度、内存占用）
- 实时目标检测和图片检测支持

**模型转换示例：**

```python
from src.mobile_deployment.converters.yolo_converter import YOLOv8MobileConverter

# 创建转换器（使用nano版本，最适合移动端）
converter = YOLOv8MobileConverter(model_size='n')

# 转换所有格式（ONNX、CoreML、TFLite）
results = converter.convert_all(
    output_dir="./mobile/models",
    model_name="yolov8n_mobile"
)
```

**性能评估示例：**

```python
from src.mobile_deployment.utils.performance_evaluator import MobilePerformanceEvaluator

# 评估ONNX模型性能
evaluator = MobilePerformanceEvaluator(
    model_path="./mobile/models/yolov8n.onnx",
    model_type='onnx'
)
report = evaluator.comprehensive_evaluation()
```

**iOS部署：**
- 详细指南：`docs/mobile/ios_deployment_guide.md`
- 示例代码：`src/mobile_deployment/ios/ios_guide.py`
- 支持实时相机检测和图片检测

**Android部署：**
- 详细指南：`docs/mobile/android_deployment_guide.md`
- 示例代码：`src/mobile_deployment/android/android_guide.py`
- 支持GPU加速（GpuDelegate）和NNAPI加速

**移动端部署特点：**
- ✅ 支持YOLOv8n（nano版本，速度最快）
- ✅ 自动模型优化（量化、简化）
- ✅ 完整的iOS/Android示例代码
- ✅ 性能评估工具
- ✅ 实时检测和图片检测支持

## ⚙️ 配置说明

项目使用 `config.yaml` 进行配置管理，主要配置项包括：

- **数据配置** - 数据目录路径
- **模型配置** - 模型参数、训练参数
- **优化配置** - 量化、剪枝、蒸馏参数
- **部署配置** - 服务端口、worker数量
- **爬虫配置** - 请求延迟、超时时间
- **多模态配置** - 图像、视频、音频参数

## 📊 模型评估指标

项目支持以下评估指标：

- **准确率 (Accuracy)**
- **精确率 (Precision)**
- **召回率 (Recall)**
- **F1分数 (F1-Score)**
- **混淆矩阵 (Confusion Matrix)**

## 🎯 技能覆盖

本项目覆盖了**大模型算法工程师**岗位所需的核心技能：

### 编程能力
- ✅ Python（数据处理、爬虫、模型开发）
- ✅ PyTorch/TensorFlow深度学习框架
- ✅ OpenCV图像处理

### 数据处理
- ✅ Pandas数据分析和处理
- ✅ NumPy数值计算
- ✅ 数据清洗和特征工程

### 模型优化
- ✅ 量化（8位、4位）
- ✅ 剪枝（L1、L2、随机）
- ✅ 知识蒸馏
- ✅ 模型部署优化

### 多模态技术
- ✅ 图像识别
- ✅ 视频识别
- ✅ 声纹识别

### 工程化能力
- ✅ Web爬虫开发
- ✅ 模型部署（Flask、FastAPI、Gradio）
- ✅ 模型导出（ONNX、TorchScript）
- ✅ 移动端部署（iOS CoreML、Android TFLite）
- ✅ YOLO系列模型移动端优化

## 📚 使用示例

### 完整流程示例

```python
# 1. 数据处理
from src.data_processing.processor import DataProcessor
processor = DataProcessor()
df = processor.load_data("data.csv")
cleaned_df = processor.data_cleaning(df)
processed_df = processor.feature_engineering(cleaned_df)

# 2. 模型训练
from src.model.trainer import ModelTrainer, SimpleCNN
model = SimpleCNN(vocab_size=10000, num_classes=2)
trainer = ModelTrainer(model)
history = trainer.train(train_loader, val_loader, num_epochs=10)

# 3. 模型优化
from src.optimization.optimizer import ModelOptimizer
optimizer = ModelOptimizer(model)
optimized_model = optimizer.optimize(quantization=True, pruning=True)

# 4. 模型部署
from src.deployment.server import ModelDeploymentManager
manager = ModelDeploymentManager(optimized_model, "my_model")
manager.deploy_fastapi(host="0.0.0.0", port=8000)

# 5. 移动端部署（YOLOv8）
from src.mobile_deployment.converters.yolo_converter import YOLOv8MobileConverter

# 转换模型为移动端格式
converter = YOLOv8MobileConverter(model_size='n')
results = converter.convert_all(
    output_dir="./mobile/models",
    model_name="yolov8n"
)
# 生成的文件：
# - yolov8n.onnx (ONNX格式)
# - yolov8n.mlmodel (iOS CoreML格式)
# - yolov8n.tflite (Android TFLite格式)
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📝 许可证

MIT License

## 🔗 相关资源

- [PyTorch官方文档](https://pytorch.org/docs/)
- [Pandas官方文档](https://pandas.pydata.org/docs/)
- [FastAPI官方文档](https://fastapi.tiangolo.com/)
- [YOLOv8官方文档](https://docs.ultralytics.com/)
- [CoreML官方文档](https://developer.apple.com/documentation/coreml)
- [TensorFlow Lite文档](https://www.tensorflow.org/lite)
- [iOS部署指南](docs/mobile/ios_deployment_guide.md)
- [Android部署指南](docs/mobile/android_deployment_guide.md)

## 📧 联系方式

如有问题或建议，欢迎提交Issue。

---

**注意：** 本项目为学习和实践用途，部分功能需要根据实际需求进行调整和优化。

