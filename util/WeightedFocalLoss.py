import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.4, gamma=2.0, class_weights=None, reduction='mean'):
        """
        Args:
            alpha (float): 类别权重调节因子（建议0.3-0.4）
            gamma (float): 困难样本聚焦因子（建议2.0）--> 越大越聚焦难分类样本
            class_weights (Tensor): class weight
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. 普通交叉熵（逐样本）+ 类别权重
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.class_weights,  # 不平衡数据的类别权重
            reduction='none'
        )
        
        # 2. focal系数
        pt = torch.exp(-ce_loss)  # pt = e^(-CE)
        focal_loss = self.alpha * (1.0 - pt)**self.gamma * ce_loss
    
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss