import torch.nn as nn 
from timm.loss import LabelSmoothingCrossEntropy
class CustomLoss(nn.Module):
    def __init__(self, smoothing, r=0.5):
        super().__init__()
        self.criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.r = r

    def forward(self, preds, targets):
        # preds shape: [batch_size, patches+1, num_class]
        # targets shape: [batch_size, num_patches+1]

        batch_size, num_patches_plus_one, num_class = preds.shape

        # Reshape preds and targets for the criterion
        preds_reshaped = preds.view(-1, num_class)
        targets_reshaped = targets.view(-1)

        # Calculate loss
        loss = self.criterion(preds_reshaped, targets_reshaped)

        # Reshape back to [batch_size, patches+1]
        loss = loss.view(batch_size, num_patches_plus_one)

        # Calculate final loss
        final_loss = self.r * loss[:, 0] + (1 - self.r) * loss[:, 1:].mean(dim=1)

        return final_loss.mean()