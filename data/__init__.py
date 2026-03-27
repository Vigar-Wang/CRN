import importlib

DATASET_REGISTRY = {}

def register_dataset(name):
    """装饰器：注册数据集类"""
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

def get_dataset(name, **kwargs):
    """根据名称获取数据集类实例"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name](**kwargs)

# 自动导入默认数据集（可选）
from .dataset_mixed import DatasetMixed