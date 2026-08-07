import torch
import torch.nn as nn
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

class Confusion_Loss(torch.nn.Module):
    '''
    Confusion loss built based on the paper 'Invesgating bias and fairness.....' 
    (https://www.repository.cam.ac.uk/bitstream/handle/1810/309834/XuEtAl-ECCV2020W.pdf?sequence=1&isAllowed=y)
    '''
    def __init__(self):
        super(Confusion_Loss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, output, label):
        # output (bs, out_size). label (bs)
        prediction = self.softmax(output) # (bs, out_size)
        log_prediction = torch.log(prediction)
        loss = -torch.mean(torch.mean(log_prediction, dim=1), dim=0)

        # loss = torch.mean(torch.mean(prediction*log_prediction, dim=1), dim=0)
        return loss

"""
For FairDisCo, code:  https://github.com/siyi-wind/FairDisCo
"""

class Supervised_Contrastive_Loss(torch.nn.Module):
    '''
    from https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
    https://blog.csdn.net/wf19971210/article/details/116715880
    Treat samples in the same labels as the positive samples, others as negative samples
    '''
    def __init__(self, temperature=0.1, device='cpu'):
        super(Supervised_Contrastive_Loss, self).__init__()
        self.temperature = temperature
        self.device = device
    
    def forward(self, projections, targets, attribute=None):
        # projections (bs, dim), targets (bs)
        # similarity matrix/T
        dot_product_tempered = F.cosine_similarity(projections.unsqueeze(1), projections.unsqueeze(0),dim=2)/self.temperature
        # print(dot_product_tempered)
        exp_dot_tempered = torch.exp(dot_product_tempered- torch.max(dot_product_tempered, dim=1, keepdim=True)[0])+ 1e-5
        # a matrix, same labels are true, others are false
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(self.device)
        # a matrix, diagonal are zeros, others are ones
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(self.device)
        mask_nonsimilar_class = ~mask_similar_class
        # a matrix, same labels are 1, others are 0, and diagonal are zeros
        mask_combined = mask_similar_class * mask_anchor_out
        # num of similar samples for sample
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # print(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr)
        # print(torch.sum(exp_dot_tempered * mask_nonsimilar_class* mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered)
        if attribute != None:
            mask_similar_attr = (attribute.unsqueeze(1).repeat(1, attribute.shape[0]) == attribute).to(self.device)
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class * mask_similar_attr, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
       
        else:
            log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_nonsimilar_class, dim=1, keepdim=True)+exp_dot_tempered+1e-5))
        supervised_contrastive_loss = torch.sum(log_prob * mask_combined)/(torch.sum(cardinality_per_samples)+1e-5)

        
        return supervised_contrastive_loss