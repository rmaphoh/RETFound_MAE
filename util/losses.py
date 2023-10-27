import torch
from torch import nn 
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
class MixLoss(nn.Module):
    def __init__(self,mixup_fn,smoothing,class_loss_r):
        super(MixLoss,self).__init__()
        self.class_loss_r=class_loss_r
        self.segmentation_loss=nn.BCEWithLogitsLoss()
        if mixup_fn is not None:
            # smoothing is handled with mixup label transform
            self.class_loss = SoftTargetCrossEntropy()
        elif smoothing > 0.:
            self.class_loss = LabelSmoothingCrossEntropy(smoothing= smoothing)
        else:
            self.class_loss = torch.nn.CrossEntropyLoss()
    def __call__(self, x,targets):
        class_tar,seg_tar=targets
        class_x,seg_x=x
        return self.class_loss(class_x,class_tar)*self.class_loss_r + \
            (1-self.class_loss_r)*(self.segmentation_loss(seg_x,seg_tar))
    def __str__(self):
        return f"LOSS FUNCTION CONDITION:{self.class_loss_r}"
        