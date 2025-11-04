"""
示例脚本 - 展示如何使用各个模块
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 导入各个模块
from src.data_processing.processor import DataProcessor
from src.scraper.web_scraper import WebScraper
from src.model.trainer import SimpleCNN
from src.optimization.optimizer import ModelOptimizer
from src.multimodal.recognizer import MultimodalRecognizer
from src.deployment.server import ModelDeploymentManager


def example_data_processing():
    """数据处理示例"""
    print("\n" + "="*60)
    print("示例1: 数据处理模块")
    print("="*60)
    
    processor = DataProcessor()
    
    # 创建示例数据
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    print(f"原始数据形状: {sample_data.shape}")
    
    # 数据清洗
    cleaned_data = processor.data_cleaning(sample_data)
    print(f"清洗后数据形状: {cleaned_data.shape}")
    
    # 特征工程
    processed_data = processor.feature_engineering(cleaned_data)
    print(f"特征工程后数据形状: {processed_data.shape}")
    
    # 数据分析
    analysis = processor.analyze_data(processed_data, target_col='target')
    print(f"数据分析完成，特征数量: {len(analysis.get('numeric_stats', {}))}")
    
    return processed_data


def example_web_scraper():
    """Web爬虫示例"""
    print("\n" + "="*60)
    print("示例2: Web爬虫模块")
    print("="*60)
    
    scraper = WebScraper(delay=1.0, timeout=10)
    
    # 注意：实际使用时需要有效的URL
    print("Web爬虫模块已初始化")
    print("使用方法: scraper.scrape(url, extract_text=True, extract_links=True)")
    
    # 示例代码（不实际执行，避免网络请求）
    # data = scraper.scrape('https://example.com', extract_text=True)
    # print(f"爬取结果: {data}")
    
    return scraper


def example_model_training():
    """模型训练示例"""
    print("\n" + "="*60)
    print("示例3: 模型训练模块")
    print("="*60)
    
    # 创建模型
    model = SimpleCNN(vocab_size=10000, embed_dim=128, num_classes=2)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型创建完成")
    print(f"模型参数数量: {total_params:,}")
    
    # 示例前向传播
    dummy_input = torch.randint(0, 10000, (1, 128))
    output = model(dummy_input)
    print(f"模型输出形状: {output.shape}")
    
    return model


def example_model_optimization():
    """模型优化示例"""
    print("\n" + "="*60)
    print("示例4: 模型优化模块")
    print("="*60)
    
    # 创建模型
    model = SimpleCNN(vocab_size=10000, embed_dim=128, num_classes=2)
    
    # 创建优化器
    optimizer = ModelOptimizer(model)
    
    # 获取原始模型大小
    original_size = optimizer.get_model_size(model)
    print(f"原始模型大小: {original_size['size_mb']:.2f} MB")
    print(f"原始模型参数数量: {original_size['parameters']:,}")
    
    # 剪枝优化
    pruned_model = optimizer.pruner.prune_weights(model, pruning_ratio=0.3)
    pruned_size = optimizer.get_model_size(pruned_model)
    print(f"剪枝后模型大小: {pruned_size['size_mb']:.2f} MB")
    print(f"剪枝后参数数量: {pruned_size['parameters']:,}")
    
    return optimizer, model


def example_multimodal():
    """多模态识别示例"""
    print("\n" + "="*60)
    print("示例5: 多模态识别模块")
    print("="*60)
    
    recognizer = MultimodalRecognizer()
    
    print("多模态识别器已初始化")
    print("- 图像识别: recognizer.image_recognizer.extract_features(image_path)")
    print("- 视频识别: recognizer.video_recognizer.extract_features(video_path)")
    print("- 声纹识别: recognizer.voiceprint_recognizer.extract_features(audio_path)")
    
    return recognizer


def example_deployment():
    """模型部署示例"""
    print("\n" + "="*60)
    print("示例6: 模型部署模块")
    print("="*60)
    
    # 创建模型
    model = SimpleCNN(vocab_size=10000, embed_dim=128, num_classes=2)
    model.eval()
    
    # 创建部署管理器
    manager = ModelDeploymentManager(model, "example_model")
    
    print("模型部署管理器已创建")
    print("- Flask部署: manager.deploy_flask(host='0.0.0.0', port=5000)")
    print("- FastAPI部署: manager.deploy_fastapi(host='0.0.0.0', port=8000)")
    print("- Gradio部署: manager.deploy_gradio(input_type='text', server_port=7860)")
    
    # 导出模型示例
    print("\n导出模型为TorchScript格式...")
    try:
        Path("models").mkdir(exist_ok=True)
        dummy_input = torch.randint(0, 10000, (1, 128))
        manager.export_model("models/exported_model.pt", format="torchscript")
        print("模型导出成功!")
    except Exception as e:
        print(f"模型导出失败: {e}")
    
    return manager


def main():
    """主函数 - 运行所有示例"""
    print("\n" + "="*60)
    print("大模型算法工程师实践项目 - 示例脚本")
    print("="*60)
    
    try:
        # 示例1: 数据处理
        processed_data = example_data_processing()
        
        # 示例2: Web爬虫
        scraper = example_web_scraper()
        
        # 示例3: 模型训练
        model = example_model_training()
        
        # 示例4: 模型优化
        optimizer, optimized_model = example_model_optimization()
        
        # 示例5: 多模态识别
        multimodal_recognizer = example_multimodal()
        
        # 示例6: 模型部署
        deployment_manager = example_deployment()
        
        
        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)
        print("\n📚 更多详细信息请查看:")
        print("   - README.md: 项目文档")
        print("   - src/: 各模块源代码")
        print("   - config.yaml: 配置文件")
        print("\n🚀 开始使用:")
        print("   python main.py")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

