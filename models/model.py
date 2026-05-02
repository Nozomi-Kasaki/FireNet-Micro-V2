import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional

class SqueezeExcitation(nn.Module):
    """SE注意力模块 - 动态增强重要通道特征"""
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class FireIRB(nn.Module):
    """火灾倒残差块 - 修正版本"""
    def __init__(self, in_channels: int, out_channels: int, 
                 expansion_ratio: float = 3.0, stride: int = 1,
                 use_se: bool = True):
        super().__init__()
        self.stride = stride
        self.use_se = use_se
        hidden_channels = int(in_channels * expansion_ratio)
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # 扩展阶段
        layers = []
        if in_channels != hidden_channels:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            ])
        
        # 深度卷积
        padding = 1 if stride == 1 else 0
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, 3, 
                     stride=stride, padding=padding, 
                     groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        ])
        
        if use_se and hidden_channels >= 16:  # 避免在通道数太少时使用SE
            layers.append(SqueezeExcitation(hidden_channels))
        
        # 投影阶段
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FireNetMicroV2(nn.Module):
    """FireNet-Micro (v2) - 修正版本"""
    def __init__(self, num_classes: int = 3, alpha: float = 0.25):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Stage 0: Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 八度卷积块
        self.stage1_conv1 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2: Fire-IRB块
        self.stage2 = nn.Sequential(
            FireIRB(24, 24, stride=1),
            FireIRB(24, 24, stride=1),
            FireIRB(24, 24, stride=1)
        )
        
        # Stage 3: Fire-IRB块
        self.stage3 = nn.Sequential(
            FireIRB(24, 48, stride=2),
            FireIRB(48, 48, stride=1),
            FireIRB(48, 48, stride=1),
            FireIRB(48, 48, stride=1)
        )
        
        # Stage 4: Fire-IRB块
        self.stage4 = nn.Sequential(
            FireIRB(48, 96, stride=2),
            FireIRB(96, 96, stride=1)
        )
        
        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        # Stem
        x = self.stem(x)  # [B, 32, 112, 112]
        
        # Stage 1: 简化版本
        x = self.stage1_conv1(x)  # [B, 24, 56, 56]
        
        # Stage 2
        x = self.stage2(x)  # [B, 24, 56, 56]
        
        # Stage 3
        x = self.stage3(x)  # [B, 48, 28, 28]
        
        # Stage 4
        x = self.stage4(x)  # [B, 96, 14, 14]
        
        # 分类头
        x = self.head(x)
        
        return x
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)