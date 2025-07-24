import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
import math
from collections import OrderedDict
import pdb
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class TA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(TA, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.min_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        # self.relu1 = nn.GELU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid1 = nn.ReLU()
        # self.sigmoid1 = nn.Hardswish()
        self.fc3 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.GELU()
        self.fc4 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)  # 7,3     3,1
        # self.conv2 = nn.Conv2d(2, 1, 3, padding=1, bias=False)  # 7,3     3,1

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x.unsqueeze(-1)))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x.unsqueeze(-1)))))
        b, c, _ = x.size()
        # mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1).unsqueeze(-1)
        std = self.fc4(self.relu1(self.fc3(std)))
        # avg_out = self.fc2(self.relu1(self.fc1(mean.unsqueeze(-1))))
        # avg_out = self.fc2(self.relu1(self.fc1(std.unsqueeze(-1))))
        # x = torch.cat([avg_out, max_out,max_out,std], dim=-1).transpose(1,3)
        x = torch.cat([avg_out, std], dim=-1).transpose(1,3)
        x = self.conv1(x)
        return self.sigmoid1(x)

class AMTFNet(nn.Module):
    def __init__(self, num_classes: int = 18, dropout: float = 0.5) -> None:
        super().__init__()
        self.features1 = nn.Sequential(
            # nn.Conv1d(50, 100, kernel_size=3, stride=1, padding=1,groups=50),
            nn.Conv1d(50, 50, kernel_size=3, stride=1, padding=1,groups=50),
            nn.InstanceNorm1d(50, affine=True),
            # nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
        )
        self.features2 = nn.Sequential(
            nn.Conv1d(50, 50, kernel_size=5, stride=1, padding=2,groups=50),
            nn.InstanceNorm1d(50, affine=True),
            nn.ReLU(inplace=True),
        )
        #
        self.features3 = nn.Sequential(
            nn.Conv1d(50, 50, kernel_size=7, stride=1, padding=3,groups=50),
            nn.InstanceNorm1d(50, affine=True),
            nn.ReLU(inplace=True),
        )
        self.features4 = nn.Sequential(
            nn.Conv1d(50, 50, kernel_size=9, stride=1, padding=4,groups=50),
            nn.InstanceNorm1d(50, affine=True),
            nn.ReLU(inplace=True),
        )
        self.gru1 = nn.GRU(input_size=200, hidden_size=100, batch_first=True)
        # self.lstm = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.att3=TA(64, ratio=8)
        self.classifier = nn.Sequential(
            # nn.Linear(100, 100),
            nn.Dropout(p=dropout),
            # nn.Linear(3 * 22 * 128, 300),
            nn.Linear(100, num_classes),
        )


    def forward(self, x: torch.Tensor):
        # x, (h_n, c_n) = self.lstm(x.squeeze().transpose(1, 2))
        x1 = self.features1(x.squeeze())
        x2 = self.features2(x.squeeze())
        x3 = self.features3(x.squeeze())
        x4 = self.features4(x.squeeze())
        x=torch.cat([x1,x2,x3,x4],dim=1)
        x, h_n = self.gru1(x.squeeze().transpose(1, 2))
        h_n=self.att3(x).squeeze()# 
        h_n=torch.sum(h_n.unsqueeze(-1)*x,dim=-2)
        feature = h_n.squeeze()
        x = self.classifier(feature)
        return feature, x
