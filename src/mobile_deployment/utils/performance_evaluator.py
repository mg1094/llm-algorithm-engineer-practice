"""
移动端性能评估工具 - 评估模型在移动端的推理速度、内存占用等
"""
import time
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MobilePerformanceEvaluator:
    """移动端性能评估器"""
    
    def __init__(self, model_path: str, model_type: str = 'onnx'):
        """
        初始化评估器
        
        Args:
            model_path: 模型路径
            model_type: 模型类型 ('onnx', 'coreml', 'tflite')
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        if self.model_type == 'onnx':
            import onnxruntime as ort
            self.model = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']  # 移动端通常使用CPU
            )
        elif self.model_type == 'tflite':
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            interpreter.allocate_tensors()
            self.model = interpreter
        else:
            logger.warning(f"模型类型 {self.model_type} 暂不支持")
    
    def evaluate_inference_speed(self, input_shape: tuple = (1, 3, 640, 640),
                                num_runs: int = 100,
                                warmup_runs: int = 10) -> Dict[str, float]:
        """
        评估推理速度
        
        Args:
            input_shape: 输入形状
            num_runs: 运行次数
            warmup_runs: 预热次数
            
        Returns:
            性能指标字典
        """
        # 准备输入数据
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 预热
        for _ in range(warmup_runs):
            self._inference_once(dummy_input)
        
        # 正式测试
        inference_times = []
        for _ in range(num_runs):
            start_time = time.time()
            self._inference_once(dummy_input)
            inference_times.append(time.time() - start_time)
        
        # 计算统计信息
        inference_times = np.array(inference_times)
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        fps = 1.0 / avg_time
        
        results = {
            'average_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'min_time_ms': min_time * 1000,
            'max_time_ms': max_time * 1000,
            'fps': fps,
            # p50_ms: 50分位（中位数）推理时间（毫秒），即一半的数据低于这个耗时
            'p50_ms': np.percentile(inference_times, 50) * 1000,   
            # p95_ms: 95分位推理时间（毫秒），即95%的推理耗时小于该值，越低说明延时越可控
            'p95_ms': np.percentile(inference_times, 95) * 1000,
            # p99_ms: 99分位推理时间（毫秒），评估极端慢的推理极少发生（抖动情况）
            'p99_ms': np.percentile(inference_times, 99) * 1000
        }
        
        logger.info(f"推理速度评估完成:")
        logger.info(f"  平均时间: {results['average_time_ms']:.2f} ms")
        logger.info(f"  FPS: {results['fps']:.2f}")
        logger.info(f"  P95: {results['p95_ms']:.2f} ms")
        
        return results
    
    def _inference_once(self, input_data: np.ndarray):
        """执行一次推理"""
        if self.model_type == 'onnx':
            input_name = self.model.get_inputs()[0].name
            self.model.run(None, {input_name: input_data})
        elif self.model_type == 'tflite':
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            self.model.set_tensor(input_details[0]['index'], input_data)
            self.model.invoke()
            self.model.get_tensor(output_details[0]['index'])
    
    def evaluate_memory_usage(self, input_shape: tuple = (1, 3, 640, 640)) -> Dict[str, float]:
        """
        评估内存占用
        
        Args:
            input_shape: 输入形状
            
        Returns:
            内存指标字典
        """
        process = psutil.Process(os.getpid())
        
        # 获取模型文件大小
        model_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        
        # 获取内存使用
        memory_before = process.memory_info().rss / (1024 * 1024)
        
        # 执行推理
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        self._inference_once(dummy_input)
        
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_used = memory_after - memory_before
        
        results = {
            'model_size_mb': model_size_mb,
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'memory_used_mb': memory_used,
            'peak_memory_mb': process.memory_info().peak_wss / (1024 * 1024) if hasattr(process.memory_info(), 'peak_wss') else memory_after
        }
        
        logger.info(f"内存占用评估:")
        logger.info(f"  模型大小: {results['model_size_mb']:.2f} MB")
        logger.info(f"  推理内存: {results['memory_used_mb']:.2f} MB")
        
        return results
    
    def evaluate_model_complexity(self) -> Dict[str, int]:
        """
        评估模型复杂度（参数量、计算量等）
        
        Returns:
            复杂度指标字典
        """
        if self.model_type == 'onnx':
            import onnx
            model = onnx.load(str(self.model_path))
            
            # 计算参数量
            total_params = 0
            for initializer in model.graph.initializer:
                param_count = 1
                for dim in initializer.dims:
                    param_count *= dim
                total_params += param_count
            
            # 计算节点数
            node_count = len(model.graph.node)
            
            results = {
                'total_parameters': total_params,
                'total_nodes': node_count,
                'input_shapes': [str(input.shape) for input in model.graph.input],
                'output_shapes': [str(output.shape) for output in model.graph.output]
            }
        else:
            results = {
                'total_parameters': 0,
                'total_nodes': 0,
                'note': '仅支持ONNX模型'
            }
        
        logger.info(f"模型复杂度:")
        logger.info(f"  参数量: {results.get('total_parameters', 0):,}")
        logger.info(f"  节点数: {results.get('total_nodes', 0)}")
        
        return results
    
    def comprehensive_evaluation(self, input_shape: tuple = (1, 3, 640, 640),
                               num_runs: int = 100) -> Dict:
        """
        综合评估
        
        Args:
            input_shape: 输入形状
            num_runs: 推理运行次数
            
        Returns:
            完整的评估报告
        """
        logger.info("=" * 60)
        logger.info("开始综合性能评估...")
        logger.info("=" * 60)
        
        report = {
            'model_path': str(self.model_path),
            'model_type': self.model_type,
            'input_shape': input_shape,
            'inference_speed': self.evaluate_inference_speed(input_shape, num_runs),
            'memory_usage': self.evaluate_memory_usage(input_shape),
            'model_complexity': self.evaluate_model_complexity()
        }
        
        logger.info("=" * 60)
        logger.info("性能评估完成！")
        logger.info("=" * 60)
        
        return report


if __name__ == "__main__":
    # 示例用法
    logger.info("移动端性能评估工具示例")
    
    # 评估ONNX模型
    evaluator = MobilePerformanceEvaluator(
        model_path="./mobile/models/yolov8n.onnx",
        model_type='onnx'
    )
    
    report = evaluator.comprehensive_evaluation()
    print("\n评估报告:", report)

