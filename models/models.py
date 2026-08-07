
"""
FairDisCo code: https://github.com/siyi-wind/FairDisCo
"""

import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class FairDisCo(torch.nn.Module):
    def __init__(self, choice='vgg16', output_size=9, pretrained=True) :
        '''
        output_size: int  only one output
                     list  first is skin type, second is skin conditipon (used in disentangle, attribute_aware)
        '''
        super(FairDisCo, self).__init__()
        self.choice = choice
        bottle_neck = 256

        if self.choice == 'vgg16':
            self.feature_extractor = models.vgg16(pretrained=pretrained)
            num_ftrs = self.feature_extractor.classifier[6].in_features
            self.feature_extractor.classifier[6] = nn.Linear(num_ftrs, output_size)
        
        if self.choice == 'resnet18':
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            num_ftrs = self.feature_extractor.fc.in_features
            self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.classifier = nn.Linear(num_ftrs, output_size)
            self.project_head = nn.Sequential(
                 nn.Linear(num_ftrs, 512),
                 nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, 128),
            )
            
        if self.choice == 'disentangle':
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            num_ftrs = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
            # for contrastive loss
            self.project_head = nn.Sequential(
                 nn.Linear(bottle_neck, 512),
                 nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, 128),
            )
            # self.activation = torch.nn.ReLU()
            # branch 1
            self.branch_1 = nn.Linear(bottle_neck, output_size[0])
            # branch 2
            self.branch_2 = nn.Linear(bottle_neck, output_size[1])
        
        if self.choice == 'attribute_aware':
            # use sensitive information into the network to train
            bottle_neck = 256
            self.feature_extractor = models.resnet18(pretrained=pretrained)
            num_ftrs = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Linear(num_ftrs, bottle_neck)
            self.attribute_layer = nn.Linear(output_size[1], bottle_neck) 
            self.classifier = nn.Linear(bottle_neck, output_size[0])

    
    def forward(self, x, attribute=None):
        if self.choice == 'disentangle':
            feature_map = self.feature_extractor(x)  # (bs, bottle_neck)
            out_1 = self.branch_1(feature_map)
            out_2 = self.branch_2(feature_map)
            out_4 = self.project_head(feature_map)
            # detach feature map and pass though branch 2 again
            feature_map_detach = feature_map.detach()
            out_3 = self.branch_2(feature_map_detach)
            return [out_1, out_2, out_3, out_4]
            # return [out_1, out_2, out_3]
            
        elif self.choice == 'attribute_aware':
            feature_map = self.feature_extractor(x) # (bs, bottle_neck)
            attribute_upsample = self.attribute_layer(attribute) # (bs, bottle_neck)
            fused_feature = feature_map+attribute_upsample # (bs, bottle_neck)
            fused_feature = F.relu(fused_feature) # (bs, bottle_neck)
            out = self.classifier(fused_feature)
            return out

        else:
            output = self.feature_extractor(x)
            output = output.view(x.size(0), -1)
            out1 = self.classifier(output)
            out2 = self.project_head(output)
            return [out1, out2]