"""
模型优化模块 - 实现量化、剪枝、蒸馏等优化技术
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Optional
import logging
from pathlib import Path
import onnx
import onnxruntime as ort
from onnxsim import simplify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """模型量化器"""
    
    @staticmethod
    def quantize_dynamic(model: nn.Module, 
                        dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        动态量化（Dynamic Quantization）

        动态量化在推理时对权重和部分激活进行量化，通常只支持如nn.Linear、nn.LSTM等模块。
        - 优点：无需校准数据，流程简单，对大部分线性层和RNN支持较好。
        - 只在推理时将权重从float32转换成int8，输入激活在执行算子前才量化，运算结束立刻反量化到float。
        - 常用于NLP类模型、部分全连接网络，其加速和压缩虽不如静态量化彻底，但操作和部署最为方便。
        - 无需模型内插入QObserver等量化观测点。

        Args:
            model: PyTorch模型
            dtype: 量化数据类型

        Returns:
            量化后的模型
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.Conv1d, nn.Conv2d}, 
            dtype=dtype
        )
        logger.info("模型动态量化完成（无需校准数据，适合线性层和部分卷积层）")
        return quantized_model

    @staticmethod
    def quantize_static(model: nn.Module, 
                       calibration_data: torch.Tensor,
                       dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        静态量化（Post-Training Static Quantization）

        静态量化通常被称为“离线量化”或“后训练量化”：
        - 流程需要插入量化观测器（Observer），并用校准数据多次前向推理以计算输入（激活）和权重的分布，生成合适的量化比例和零点。
        - 优点：对权重和所有激活同时量化，存储和加速效果最佳，适用场景广泛。
        - 缺点：流程稍复杂，要求准备一批代表性校准数据。
        - 对于部署端推理有最高的压缩和加速比，适合对资源要求高的场景。

        Args:
            model: PyTorch模型
            calibration_data: 校准数据（应与真实数据分布一致，足量）
            dtype: 量化数据类型

        Returns:
            量化后的模型
        """
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        # 使用校准数据运行一次前向推理以记录统计信息
        with torch.no_grad():
            model(calibration_data)
        quantized_model = torch.quantization.convert(model, inplace=False)
        logger.info("模型静态量化完成（需校准数据，权重与激活均量化）")
        return quantized_model
    
    @staticmethod
    def quantize_to_int8(model: nn.Module) -> nn.Module:
        """8位量化"""
        return ModelQuantizer.quantize_dynamic(model, dtype=torch.qint8)
    
    @staticmethod
    def quantize_to_int4(model: nn.Module) -> nn.Module:
        """4位量化（需要bitsandbytes库）"""
        try:
            import bitsandbytes as bnb
            # 使用bitsandbytes进行4位量化
            model = bnb.nn.Linear4bit.from_pretrained(model)
            logger.info("模型4位量化完成")
        except ImportError:
            logger.warning("bitsandbytes未安装，无法进行4位量化")
        return model


class ModelPruner:
    """模型剪枝器"""
    
    @staticmethod
    def prune_weights(model: nn.Module, 
                     pruning_ratio: float = 0.3,
                     method: str = 'l1_unstructured') -> nn.Module:
        """
        权重剪枝
        
        Args:
            model: PyTorch模型
            pruning_ratio: 剪枝比例
            method: 剪枝方法
                - 'l1_unstructured'：基于L1范数的非结构化剪枝，将绝对值最小的权重置零，能有效降低模型复杂度且影响较小。
                - 'l2_unstructured'：基于L2范数的非结构化剪枝，根据权重的L2范数大小筛选，类似L1但对大权重变化更敏感。
                - 'random_unstructured'：随机非结构化剪枝，随机选择部分权重进行剪除，作为基线方法参考。
            
        Returns:
            剪枝后的模型
        """
        pruned_model = model
        
        for module_name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                elif method == 'l2_unstructured':
                    prune.l2_unstructured(module, name='weight', amount=pruning_ratio)
                elif method == 'random_unstructured':
                    prune.random_unstructured(module, name='weight', amount=pruning_ratio)
                
                # 永久移除剪枝参数
                prune.remove(module, 'weight')
        
        logger.info(f"模型剪枝完成，剪枝比例: {pruning_ratio}")
        return pruned_model
    
    @staticmethod
    def get_pruning_statistics(model: nn.Module) -> Dict:
        """获取剪枝统计信息"""
        stats = {
            'total_params': 0,
            'pruned_params': 0,
            'remaining_params': 0
        }
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    total = mask.numel()
                    pruned = (mask == 0).sum().item()
                    stats['total_params'] += total
                    stats['pruned_params'] += pruned
                    stats['remaining_params'] += (total - pruned)
        
        return stats


class ModelDistiller:
    """模型蒸馏器"""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module):
        self.teacher_model = teacher_model
        self.student_model = student_model
    
    def distill(self, dataloader: torch.utils.data.DataLoader,
               temperature: float = 4.0,
               alpha: float = 0.5,
               num_epochs: int = 10,
               learning_rate: float = 0.001) -> nn.Module:
        """
        知识蒸馏
        
        Args:
            dataloader: 数据加载器
            temperature: 温度参数
            alpha: 蒸馏损失权重
            num_epochs: 训练轮数
            learning_rate: 学习率
            
        Returns:
            蒸馏后的学生模型
        """
        self.teacher_model.eval()
        self.student_model.train()
        
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)
        criterion_ce = nn.CrossEntropyLoss()
        # KL散度（Kullback-Leibler Divergence）损失的值越小，说明学生模型的输出分布越接近于教师模型的输出分布。
        # 在知识蒸馏中，我们的目标就是最小化KL散度，让学生模型更好地拟合教师模型的"软目标"概率分布。
        # 一般来说，KL散度损失越小越好。
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in dataloader:
                inputs = batch['input_ids'].to(next(self.teacher_model.parameters()).device)
                labels = batch['labels'].to(next(self.teacher_model.parameters()).device)
                
                # 教师模型预测
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(inputs)
                    teacher_probs = torch.softmax(teacher_outputs / temperature, dim=1)
                
                # 学生模型预测
                student_outputs = self.student_model(inputs)
                student_probs = torch.log_softmax(student_outputs / temperature, dim=1)
                
                # 计算损失
                loss_ce = criterion_ce(student_outputs, labels)
                loss_kl = criterion_kl(student_probs, teacher_probs) * (temperature ** 2)
                loss = alpha * loss_kl + (1 - alpha) * loss_ce
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        logger.info("模型蒸馏完成")
        return self.student_model


class ModelOptimizer:
    """模型优化器 - 整合各种优化技术"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.quantizer = ModelQuantizer()
        self.pruner = ModelPruner()
    
    def optimize(self, 
                quantization: bool = False,
                pruning: bool = False,
                pruning_ratio: float = 0.3,
                quantization_bits: int = 8) -> nn.Module:
        """
        综合优化模型
        
        Args:
            quantization: 是否量化
            pruning: 是否剪枝
            pruning_ratio: 剪枝比例
            quantization_bits: 量化位数
            
        Returns:
            优化后的模型
        """
        optimized_model = self.model
        
        # 剪枝
        if pruning:
            optimized_model = self.pruner.prune_weights(
                optimized_model, 
                pruning_ratio=pruning_ratio
            )
        
        # 量化
        if quantization:
            if quantization_bits == 8:
                optimized_model = self.quantizer.quantize_to_int8(optimized_model)
            elif quantization_bits == 4:
                optimized_model = self.quantizer.quantize_to_int4(optimized_model)
        
        return optimized_model
    
    def export_to_onnx(self, model: nn.Module, 
                      dummy_input: torch.Tensor,
                      output_path: str,
                      simplify_model: bool = True):
        """
        导出模型为ONNX格式
        
        Args:
            model: PyTorch模型
            dummy_input: 示例输入
            output_path: 输出路径
            simplify_model: 是否简化模型
        """
        model.eval()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 导出ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # 简化模型
        if simplify_model:
            onnx_model = onnx.load(str(output_path))
            # simplify函数用于对ONNX模型结构进行进一步简化和优化。它会移除冗余的节点、常量折叠、融合算子等，以提升模型推理速度并减少模型体积。
            # 返回值 simplified_model 为优化后的模型，check 为布尔值，指示简化后的模型是否与原模型语义等价。
            simplified_model, check = simplify(onnx_model)
            if check:
                onnx.save(simplified_model, str(output_path))
                logger.info(f"ONNX模型已简化并保存: {output_path}")
            else:
                logger.warning("ONNX模型简化失败")
        
        logger.info(f"模型已导出为ONNX: {output_path}")
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """获取模型大小信息"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        # model.buffers() 返回模型中不参与梯度更新的缓冲区张量（如 running_mean、running_var、某些 BatchNorm 层的统计信息等），
        # 它们通常在推理时用于辅助模型推理，但并不是模型参数（如权重和偏置），不会被优化器更新。
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        
        return {
            'parameters': sum(p.numel() for p in model.parameters()),
            'size_mb': total_size / (1024 ** 2),
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_size_mb': buffer_size / (1024 ** 2)
        }


if __name__ == "__main__":
    # 示例用法
    logger.info("模型优化模块示例")

