import torch
import numpy as np
import random
import os

# 类别映射：确保顺序固定
CLASS_MAP = {'fire': 0, 'no_fire': 1, 'start_fire': 2}
IDX_TO_CLASS = {0: 'fire', 1: 'no_fire', 2: 'start_fire'}

def seed_everything(seed=42):
    """固定随机种子，保证结果可复现"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_class_weights(dataset):
    """
    自动计算类别权重，用于解决样本不平衡问题
    Weight = Total / (3 * Class_Count)
    """
    targets = dataset.targets
    class_counts = np.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)