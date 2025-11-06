"""
移动端部署示例脚本
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mobile_deployment.converters.yolo_converter import YOLOv8MobileConverter
from src.mobile_deployment.utils.performance_evaluator import MobilePerformanceEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """移动端部署示例"""
    print("\n" + "="*60)
    print("移动端YOLOv8部署示例")
    print("="*60)
    
    try:
        # 1. 模型转换
        print("\n步骤1: 转换YOLOv8模型为移动端格式...")
        converter = YOLOv8MobileConverter(model_size='n')
        
        # 注意：实际转换需要下载模型，这里仅演示接口
        print("YOLOv8MobileConverter已创建")
        print("使用方法:")
        print("  converter = YOLOv8MobileConverter(model_size='n')")
        print("  results = converter.convert_all(output_dir='./mobile/models')")
        
        # 2. 性能评估示例
        print("\n步骤2: 性能评估工具...")
        print("MobilePerformanceEvaluator可用于评估模型性能")
        print("使用方法:")
        print("  evaluator = MobilePerformanceEvaluator(model_path, model_type='onnx')")
        print("  report = evaluator.comprehensive_evaluation()")
        
        # 3. 部署指南
        print("\n步骤3: 查看部署指南...")
        print("iOS部署指南: docs/mobile/ios_deployment_guide.md")
        print("Android部署指南: docs/mobile/android_deployment_guide.md")
        
        print("\n" + "="*60)
        print("移动端部署模块已就绪！")
        print("="*60)
        print("\n📱 支持的平台:")
        print("  - iOS (CoreML)")
        print("  - Android (TensorFlow Lite)")
        print("\n🎯 支持的场景:")
        print("  - 实时目标检测")
        print("  - 图片目标检测")
        print("\n⚡ 性能优化:")
        print("  - 模型量化")
        print("  - GPU加速")
        print("  - NNAPI加速 (Android)")
        print("="*60)
        
    except ImportError as e:
        logger.warning(f"部分依赖未安装: {e}")
        logger.info("请安装移动端部署依赖:")
        logger.info("  pip install ultralytics coremltools tensorflow")
    except Exception as e:
        logger.error(f"示例运行出错: {e}")


if __name__ == "__main__":
    main()



