"""
模型部署模块 - 使用Flask和FastAPI部署模型服务
"""
import torch
import numpy as np
from flask import Flask, request, jsonify
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import gradio as gr
from typing import Dict, List, Optional
import logging
from pathlib import Path
import io
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlaskModelServer:
    """基于Flask的模型服务"""
    
    def __init__(self, model: torch.nn.Module, model_name: str = "model"):
        self.app = Flask(__name__)
        self.model = model
        self.model.eval()
        self.model_name = model_name
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """健康检查"""
            return jsonify({'status': 'healthy', 'model': self.model_name})
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """预测接口"""
            try:
                data = request.get_json()
                
                # 处理输入数据
                input_data = self._preprocess_input(data)
                
                # 模型推理
                with torch.no_grad():
                    output = self.model(input_data)
                    predictions = torch.softmax(output, dim=1) if len(output.shape) > 1 else output
                
                # 后处理
                result = self._postprocess_output(predictions)
                
                return jsonify({
                    'success': True,
                    'predictions': result,
                    'model': self.model_name
                })
            
            except Exception as e:
                logger.error(f"预测错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/batch_predict', methods=['POST'])
        def batch_predict():
            """批量预测接口"""
            try:
                data = request.get_json()
                inputs = data.get('inputs', [])
                
                results = []
                for input_data in inputs:
                    processed_input = self._preprocess_input(input_data)
                    with torch.no_grad():
                        output = self.model(processed_input)
                        predictions = torch.softmax(output, dim=1) if len(output.shape) > 1 else output
                        result = self._postprocess_output(predictions)
                        results.append(result)
                
                return jsonify({
                    'success': True,
                    'results': results,
                    'count': len(results)
                })
            
            except Exception as e:
                logger.error(f"批量预测错误: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def _preprocess_input(self, data: Dict) -> torch.Tensor:
        """预处理输入数据"""
        # 根据实际需求实现预处理逻辑
        input_ids = data.get('input_ids', [])
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    
    def _postprocess_output(self, output: torch.Tensor) -> Dict:
        """后处理输出数据"""
        if len(output.shape) > 1:
            probs = output[0].cpu().numpy().tolist()
            predicted_class = int(torch.argmax(output[0]))
            return {
                'predicted_class': predicted_class,
                'probabilities': probs
            }
        else:
            return {
                'output': output.cpu().numpy().tolist()
            }
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """运行Flask服务"""
        logger.info(f"启动Flask服务: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


class FastAPIModelServer:
    """基于FastAPI的模型服务"""
    
    def __init__(self, model: torch.nn.Module, model_name: str = "model"):
        self.app = FastAPI(title=f"{model_name} API", version="1.0.0")
        self.model = model
        self.model.eval()
        self.model_name = model_name
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.get("/health")
        async def health():
            """健康检查"""
            return {"status": "healthy", "model": self.model_name}
        
        @self.app.post("/predict")
        async def predict(data: Dict):
            """预测接口"""
            try:
                input_data = self._preprocess_input(data)
                
                with torch.no_grad():
                    output = self.model(input_data)
                    predictions = torch.softmax(output, dim=1) if len(output.shape) > 1 else output
                
                result = self._postprocess_output(predictions)
                
                return JSONResponse({
                    "success": True,
                    "predictions": result,
                    "model": self.model_name
                })
            
            except Exception as e:
                logger.error(f"预测错误: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict_image")
        async def predict_image(file: UploadFile = File(...)):
            """图像预测接口"""
            try:
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                
                # 预处理图像
                input_tensor = self._preprocess_image(image)
                
                # 模型推理
                with torch.no_grad():
                    output = self.model(input_tensor)
                    predictions = torch.softmax(output, dim=1)
                
                result = self._postprocess_output(predictions)
                
                return JSONResponse({
                    "success": True,
                    "predictions": result,
                    "filename": file.filename
                })
            
            except Exception as e:
                logger.error(f"图像预测错误: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _preprocess_input(self, data: Dict) -> torch.Tensor:
        """预处理输入数据"""
        input_ids = data.get('input_ids', [])
        return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """预处理图像"""
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(image).unsqueeze(0)
    
    def _postprocess_output(self, output: torch.Tensor) -> Dict:
        """后处理输出数据"""
        if len(output.shape) > 1:
            probs = output[0].cpu().numpy().tolist()
            predicted_class = int(torch.argmax(output[0]))
            return {
                'predicted_class': predicted_class,
                'probabilities': probs
            }
        else:
            return {
                'output': output.cpu().numpy().tolist()
            }
    
    def run(self, host: str = '0.0.0.0', port: int = 8000, workers: int = 1):
        """运行FastAPI服务"""
        logger.info(f"启动FastAPI服务: http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port, workers=workers)


class GradioModelInterface:
    """基于Gradio的模型交互界面"""
    
    def __init__(self, model: torch.nn.Module, model_name: str = "model"):
        self.model = model
        self.model.eval()
        self.model_name = model_name
    
    def create_interface(self, input_type: str = "text"):
        """
        创建Gradio界面
        
        Args:
            input_type: 输入类型 ('text', 'image', 'audio')
        """
        if input_type == "text":
            interface = gr.Interface(
                fn=self._predict_text,
                inputs=gr.Textbox(lines=5, placeholder="输入文本..."),
                outputs="json",
                title=f"{self.model_name} - 文本分类",
                description="输入文本进行预测"
            )
        elif input_type == "image":
            interface = gr.Interface(
                fn=self._predict_image,
                inputs=gr.Image(type="pil"),
                outputs="json",
                title=f"{self.model_name} - 图像分类",
                description="上传图像进行预测"
            )
        elif input_type == "audio":
            interface = gr.Interface(
                fn=self._predict_audio,
                inputs=gr.Audio(type="filepath"),
                outputs="json",
                title=f"{self.model_name} - 音频分类",
                description="上传音频进行预测"
            )
        else:
            raise ValueError(f"不支持的输入类型: {input_type}")
        
        return interface
    
    def _predict_text(self, text: str) -> Dict:
        """文本预测"""
        # 实现文本预测逻辑
        return {"prediction": "示例结果", "confidence": 0.95}
    
    def _predict_image(self, image: Image.Image) -> Dict:
        """图像预测"""
        # 实现图像预测逻辑
        return {"prediction": "示例结果", "confidence": 0.95}
    
    def _predict_audio(self, audio_path: str) -> Dict:
        """音频预测"""
        # 实现音频预测逻辑
        return {"prediction": "示例结果", "confidence": 0.95}
    
    def launch(self, input_type: str = "text", 
              share: bool = False, 
              server_name: str = "0.0.0.0",
              server_port: int = 7860):
        """启动Gradio界面"""
        interface = self.create_interface(input_type)
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


class ModelDeploymentManager:
    """模型部署管理器"""
    
    def __init__(self, model: torch.nn.Module, model_name: str = "model"):
        self.model = model
        self.model_name = model_name
        self.flask_server = None
        self.fastapi_server = None
        self.gradio_interface = None
    
    def deploy_flask(self, host: str = '0.0.0.0', port: int = 5000):
        """部署Flask服务"""
        self.flask_server = FlaskModelServer(self.model, self.model_name)
        self.flask_server.run(host=host, port=port)
    
    def deploy_fastapi(self, host: str = '0.0.0.0', port: int = 8000, workers: int = 1):
        """部署FastAPI服务"""
        self.fastapi_server = FastAPIModelServer(self.model, self.model_name)
        self.fastapi_server.run(host=host, port=port, workers=workers)
    
    def deploy_gradio(self, input_type: str = "text", 
                     server_name: str = "0.0.0.0",
                     server_port: int = 7860):
        """部署Gradio界面"""
        self.gradio_interface = GradioModelInterface(self.model, self.model_name)
        self.gradio_interface.launch(
            input_type=input_type,
            server_name=server_name,
            server_port=server_port
        )
    
    def export_model(self, output_path: str, format: str = "torchscript"):
        """
        导出模型
        
        Args:
            output_path: 输出路径
            format: 导出格式 ('torchscript', 'onnx')
        """
        self.model.eval()
        
        if format == "torchscript":
            # 导出为TorchScript
            dummy_input = torch.randn(1, 128)  # 根据实际输入调整
            traced_model = torch.jit.trace(self.model, dummy_input)
            traced_model.save(output_path)
            logger.info(f"模型已导出为TorchScript: {output_path}")
        
        elif format == "onnx":
            # 导出为ONNX
            dummy_input = torch.randn(1, 128)  # 根据实际输入调整
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            logger.info(f"模型已导出为ONNX: {output_path}")


if __name__ == "__main__":
    # 示例用法
    logger.info("模型部署模块示例")

