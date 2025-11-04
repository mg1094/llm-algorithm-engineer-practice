"""
数据处理模块 - 使用Pandas和NumPy进行数据采集、清洗、预处理和深度分析
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理类，提供数据采集、清洗、预处理和深度分析功能"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path: str, file_type: str = 'csv') -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: 文件路径
            file_type: 文件类型 ('csv', 'excel', 'json', 'parquet')
            
        Returns:
            DataFrame对象
        """
        file_path = Path(file_path)
        
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        elif file_type == 'json':
            df = pd.read_json(file_path)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
            
        logger.info(f"成功加载数据: {file_path}, 形状: {df.shape}")
        return df
    
    def data_cleaning(self, df: pd.DataFrame, 
                     drop_duplicates: bool = True,
                     handle_missing: str = 'fill') -> pd.DataFrame:
        """
        数据清洗
        
        Args:
            df: 输入DataFrame
            drop_duplicates: 是否删除重复行
            handle_missing: 缺失值处理方式 ('fill', 'drop', 'interpolate')
            
        Returns:
            清洗后的DataFrame
        """
        df_clean = df.copy()
        
        # 删除重复行
        if drop_duplicates:
            initial_shape = df_clean.shape[0]
            df_clean = df_clean.drop_duplicates()
            logger.info(f"删除重复行: {initial_shape - df_clean.shape[0]} 行")
        
        # 处理缺失值
        missing_count = df_clean.isnull().sum()
        if missing_count.sum() > 0:
            logger.info(f"缺失值统计:\n{missing_count[missing_count > 0]}")
            
            if handle_missing == 'fill':
                # 数值列用中位数填充，类别列用众数填充
                for col in df_clean.columns:
                    if df_clean[col].dtype in ['int64', 'float64']:
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else '', inplace=True)
            elif handle_missing == 'drop':
                df_clean = df_clean.dropna()
            elif handle_missing == 'interpolate':
                df_clean = df_clean.interpolate(method='linear')
        
        logger.info(f"数据清洗完成，最终形状: {df_clean.shape}")
        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame, 
                           numeric_cols: Optional[List[str]] = None,
                           categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        特征工程
        
        Args:
            df: 输入DataFrame
            numeric_cols: 数值特征列名列表
            categorical_cols: 类别特征列名列表
            
        Returns:
            特征工程后的DataFrame
        """
        df_fe = df.copy()
        
        # 自动识别数值列和类别列
        # 如果未指定数值特征列，则自动查找所有数值类型的列（如int、float）作为numeric_cols
        if numeric_cols is None:
            numeric_cols = df_fe.select_dtypes(include=[np.number]).columns.tolist()
        # 如果未指定类别特征列，则自动查找所有object类型的列作为categorical_cols
        if categorical_cols is None:
            categorical_cols = df_fe.select_dtypes(include=['object']).columns.tolist()
        
        # 数值特征标准化
        # fit_transform方法是sklearn中的常用方法，包含两个步骤：
        # 1. fit部分会根据提供的数值特征数据，计算出对应的标准化参数（如均值、方差），并保存到scaler对象中；
        # 2. transform部分会用fit中得到的参数，对传入的数据进行变换（如变为均值为0，方差为1的分布）。
        # 这样模型在训练集上fit后，可以在测试集或新数据上只用transform，保证一致。
        if numeric_cols:
            df_fe[numeric_cols] = self.scaler.fit_transform(df_fe[numeric_cols])
            logger.info(f"标准化数值特征: {len(numeric_cols)} 列")
        # 类别特征编码
        if categorical_cols:
            for col in categorical_cols:
                df_fe[col] = self.label_encoder.fit_transform(df_fe[col].astype(str))
            logger.info(f"编码类别特征: {len(categorical_cols)} 列")
        
        return df_fe
    
    def analyze_data(self, df: pd.DataFrame, 
                    target_col: Optional[str] = None) -> Dict:
        """
        深度数据分析
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名（用于相关性分析）
            
        Returns:
            分析结果字典
        """
        analysis = {}
        
        # 基本统计信息
        analysis['shape'] = df.shape
        analysis['dtypes'] = df.dtypes.to_dict()
        analysis['describe'] = df.describe().to_dict()
        
        # 缺失值统计
        analysis['missing_values'] = df.isnull().sum().to_dict()
        analysis['missing_percentage'] = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # 数值特征统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['numeric_stats'] = {
                col: {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median(),
                    'skewness': df[col].skew(),
                    'kurtosis': df[col].kurtosis()
                }
                for col in numeric_cols
            }
        
        # 相关性分析
        if target_col and target_col in df.columns:
            correlations = df.corr()[target_col].sort_values(ascending=False)
            analysis['correlations'] = correlations.to_dict()
        
        logger.info("数据分析完成")
        return analysis
    
    def train_test_split_data(self, df: pd.DataFrame, 
                             target_col: str,
                             test_size: float = 0.2,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        划分训练集和测试集
        
        Args:
            df: 输入DataFrame
            target_col: 目标列名
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"数据划分完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, df: pd.DataFrame, file_path: str):
        """保存处理后的数据"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix == '.csv':
            df.to_csv(file_path, index=False)
        elif file_path.suffix == '.parquet':
            df.to_parquet(file_path, index=False)
        else:
            df.to_csv(file_path.with_suffix('.csv'), index=False)
        
        logger.info(f"数据已保存: {file_path}")


if __name__ == "__main__":
    # 示例用法
    processor = DataProcessor()
    
    # 创建示例数据
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'target': np.random.randint(0, 2, 1000)
    })
    
    # 数据清洗
    cleaned_data = processor.data_cleaning(sample_data)
    
    # 特征工程
    processed_data = processor.feature_engineering(cleaned_data)
    
    # 数据分析
    analysis = processor.analyze_data(processed_data, target_col='target')
    print("数据分析结果:", analysis)

