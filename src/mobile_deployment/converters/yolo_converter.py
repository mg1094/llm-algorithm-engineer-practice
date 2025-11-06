"""
移动端模型转换器 - 将PyTorch/YOLO模型转换为移动端格式（CoreML/TFLite）
"""
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobileModelConverter:
    """移动端模型转换器基类"""
    
    def __init__(self, model: torch.nn.Module, input_shape: Tuple[int, ...] = (1, 3, 640, 640)):
        """
        初始化转换器
        
        Args:
            model: PyTorch模型
            input_shape: 输入形状 (batch, channels, height, width)
        """
        self.model = model
        self.model.eval()
        self.input_shape = input_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def export_to_onnx(self, output_path: str, opset_version: int = 11, 
                      simplify: bool = True) -> str:
        """
        导出模型为ONNX格式
        
        Args:
            output_path: 输出路径
            opset_version: ONNX opset版本
            simplify: 是否简化模型
            
        Returns:
            ONNX模型路径
        """
        self.model.eval()
        dummy_input = torch.randn(*self.input_shape).to(self.device)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 导出ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )
        
        logger.info(f"ONNX模型已导出: {output_path}")
        
        # 简化模型
        if simplify:
            try:
                from onnxsim import simplify
                onnx_model = onnx.load(str(output_path))
                simplified_model, check = simplify(onnx_model)
                if check:
                    onnx.save(simplified_model, str(output_path))
                    logger.info(f"ONNX模型已简化: {output_path}")
            except Exception as e:
                logger.warning(f"ONNX模型简化失败: {e}")
        
        return str(output_path)
    
    def validate_onnx(self, onnx_path: str, test_input: Optional[torch.Tensor] = None) -> bool:
        """
        验证ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            test_input: 测试输入（可选）
            
        Returns:
            验证是否通过
        """
        try:
            # 检查ONNX模型
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX模型验证通过")
            
            # 如果提供了测试输入，进行推理测试
            if test_input is not None:
                session = ort.InferenceSession(onnx_path)
                input_name = session.get_inputs()[0].name
                output = session.run(None, {input_name: test_input.cpu().numpy()})
                logger.info(f"ONNX推理测试成功，输出形状: {output[0].shape}")
            
            return True
        except Exception as e:
            logger.error(f"ONNX模型验证失败: {e}")
            return False


class YOLOv8MobileConverter(MobileModelConverter):
    """YOLOv8移动端转换器"""
    
    def __init__(self, model_path: Optional[str] = None, model_size: str = 'n'):
        """
        初始化YOLOv8转换器
        
        Args:
            model_path: 模型路径（.pt文件），如果为None则使用预训练模型
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')，n为最小最快
        """
        try:
            from ultralytics import YOLO
            
            if model_path:
                self.yolo_model = YOLO(model_path)
            else:
                # 使用预训练模型，nano版本最适合移动端
                model_name = f'yolov8{model_size}.pt'
                self.yolo_model = YOLO(model_name)
                logger.info(f"加载预训练模型: {model_name}")
            
            # YOLOv8的输入尺寸通常是640x640
            super().__init__(self.yolo_model.model, input_shape=(1, 3, 640, 640))
            
        except ImportError:
            logger.error("ultralytics未安装，请安装: pip install ultralytics")
            raise
    
    def export_to_coreml(self, onnx_path: str, output_path: str, 
                        input_name: str = 'image',
                        output_name: str = 'output',
                        image_scale: float = 1.0 / 255.0,
                        red_bias: float = 0.0,
                        green_bias: float = 0.0,
                        blue_bias: float = 0.0) -> str:
        """
        将ONNX模型转换为CoreML格式（iOS）
        
        Args:
            onnx_path: ONNX模型路径
            output_path: CoreML输出路径
            input_name: 输入名称
            output_name: 输出名称
            image_scale: 图像缩放因子
            red_bias/green_bias/blue_bias: RGB通道偏移
            
        Returns:
            CoreML模型路径
        """
        try:
            import coremltools as ct
            
            # 加载ONNX模型
            mlmodel = ct.convert(
                onnx_path,
                source='onnx',
                inputs=[ct.TensorType(name=input_name, shape=(1, 3, 640, 640))]
            )
            
            # 设置输入规格（图像输入）
            spec = mlmodel.get_spec()
            ct.models.utils.update_image_scale_range(spec, image_scale)
            
            # 保存CoreML模型
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mlmodel.save(str(output_path))
            
            logger.info(f"CoreML模型已导出: {output_path}")
            logger.info("CoreML模型可以直接在iOS项目中使用")
            
            return str(output_path)
            
        except ImportError:
            logger.error("coremltools未安装，请安装: pip install coremltools")
            raise
        except Exception as e:
            logger.error(f"CoreML转换失败: {e}")
            raise
    
    def export_to_tflite(self, onnx_path: str, output_path: str,
                        quantize: bool = True,
                        target_spec: Optional[Dict] = None) -> str:
        """
        将ONNX模型转换为TensorFlow Lite格式（Android）
        
        Args:
            onnx_path: ONNX模型路径
            output_path: TFLite输出路径
            quantize: 是否量化（INT8量化可显著减小模型大小）
            target_spec: 目标设备规格（可选）
            
        Returns:
            TFLite模型路径
        """
        try:
            import tf2onnx
            import tensorflow as tf
            from onnx_tf.backend import prepare
            
            # 方法1: 通过ONNX→TensorFlow→TFLite
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 转换为TensorFlow SavedModel
            tf_rep = prepare(onnx_model)
            temp_tf_dir = Path(output_path).parent / 'temp_tf_model'
            tf_rep.export_graph(str(temp_tf_dir))
            
            # 转换为TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(temp_tf_dir))
            
            # 量化选项
            if quantize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]  # 或 tf.int8
            
            # 转换
            tflite_model = converter.convert()
            
            # 保存TFLite模型
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            # 清理临时文件
            import shutil
            shutil.rmtree(temp_tf_dir, ignore_errors=True)
            
            model_size_mb = len(tflite_model) / (1024 * 1024)
            logger.info(f"TFLite模型已导出: {output_path}")
            logger.info(f"模型大小: {model_size_mb:.2f} MB")
            logger.info("TFLite模型可以直接在Android项目中使用")
            
            return str(output_path)
            
        except ImportError as e:
            logger.error(f"依赖未安装: {e}")
            logger.error("请安装: pip install tensorflow onnx-tf")
            raise
        except Exception as e:
            logger.error(f"TFLite转换失败: {e}")
            raise
    
    def optimize_for_mobile(self, onnx_path: str, 
                           target: str = 'mobile',
                           output_path: Optional[str] = None) -> str:
        """
        为移动端优化ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            target: 目标平台 ('mobile', 'ios', 'android')
            output_path: 输出路径（可选）
            
        Returns:
            优化后的模型路径
        """
        if output_path is None:
            output_path = Path(onnx_path).with_name(
                f"{Path(onnx_path).stem}_optimized_{target}.onnx"
            )
        
        # 加载和简化模型
        onnx_model = onnx.load(onnx_path)
        
        # 应用优化
        # 1. 常量折叠
        # 2. 算子融合
        # 3. 死代码消除
        
        try:
            from onnxsim import simplify
            simplified_model, check = simplify(onnx_model)
            if check:
                onnx.save(simplified_model, output_path)
                logger.info(f"移动端优化模型已保存: {output_path}")
            else:
                logger.warning("模型简化失败，使用原始模型")
                onnx.save(onnx_model, output_path)
        except Exception as e:
            logger.warning(f"优化过程出错: {e}")
            onnx.save(onnx_model, output_path)
        
        return str(output_path)
    
    def convert_all(self, output_dir: str = "./mobile/models",
                   model_name: str = "yolov8n") -> Dict[str, str]:
        """
        一次性转换所有格式
        
        Args:
            output_dir: 输出目录
            model_name: 模型名称
            
        Returns:
            转换后的模型路径字典
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 1. 导出ONNX
        logger.info("步骤1: 导出ONNX模型...")
        onnx_path = self.export_to_onnx(
            output_dir / f"{model_name}.onnx"
        )
        results['onnx'] = onnx_path
        
        # 2. 验证ONNX
        logger.info("步骤2: 验证ONNX模型...")
        self.validate_onnx(onnx_path)
        
        # 3. 优化ONNX
        logger.info("步骤3: 优化ONNX模型...")
        optimized_onnx = self.optimize_for_mobile(onnx_path, 'mobile')
        results['onnx_optimized'] = optimized_onnx
        
        # 4. 转换为CoreML (iOS)
        logger.info("步骤4: 转换为CoreML格式 (iOS)...")
        try:
            coreml_path = self.export_to_coreml(
                optimized_onnx,
                output_dir / f"{model_name}.mlmodel"
            )
            results['coreml'] = coreml_path
        except Exception as e:
            logger.warning(f"CoreML转换失败: {e}")
        
        # 5. 转换为TFLite (Android)
        logger.info("步骤5: 转换为TFLite格式 (Android)...")
        try:
            tflite_path = self.export_to_tflite(
                optimized_onnx,
                output_dir / f"{model_name}.tflite",
                quantize=True
            )
            results['tflite'] = tflite_path
        except Exception as e:
            logger.warning(f"TFLite转换失败: {e}")
        
        logger.info("=" * 60)
        logger.info("模型转换完成！")
        logger.info("=" * 60)
        for format_name, path in results.items():
            logger.info(f"{format_name.upper()}: {path}")
        
        return results


if __name__ == "__main__":
    # 示例用法
    logger.info("YOLOv8移动端转换器示例")
    
    # 创建转换器（使用nano版本，最适合移动端）
    converter = YOLOv8MobileConverter(model_size='n')
    
    # 转换所有格式
    results = converter.convert_all(
        output_dir="./mobile/models",
        model_name="yolov8n_mobile"
    )
    
    print("\n转换结果:", results)

