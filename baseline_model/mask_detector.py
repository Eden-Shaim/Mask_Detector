import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models

class MaskDetector(nn.Module):
    def __init__(self, in_channels=3, bbox_dim=4):
        super(MaskDetector, self).__init__()
        self.resnet50 = models.wide_resnet50_2(pretrained=False)
        self.bbox = nn.Linear(1000, bbox_dim)
        self.proper_mask = nn.Linear(1000 + bbox_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img_features = self.resnet50(img)
        bbox = self.bbox(img_features)
        proper_mask = self.proper_mask(torch.cat([img_features, bbox], dim=1))
        proper_mask = self.sigmoid(proper_mask).squeeze()
        return bbox, proper_mask

