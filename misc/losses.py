import torch
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class Focal_loss(torch.nn.Module):
    def __init__(self,n_class,alpha=1,gamma=2,reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.n_class = n_class
        self.reduction = reduction

    def forward(self,inputs,targets):
        targets_one_hot = F.one_hot(targets, num_classes=self.n_class).float()

        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets_one_hot,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )