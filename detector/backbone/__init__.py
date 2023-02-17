import torch.nn as nn
from detector.backbone.CondConv import CondConv
from detector.backbone.Involution import Involution

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.F1 = CondConv(in_planes=3, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.involution1 = Involution(kernel_size=3, in_channel=64, stride=1)

        self.F2 = CondConv(in_planes=3, out_planes=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.involution2 = Involution(kernel_size=3, in_channel=128, stride=2)

        self.F3 = CondConv(in_planes=3, out_planes=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.involution3 = Involution(kernel_size=3, in_channel=256, stride=4)
    def forward(self, x):
        x1 = self.F1(x)
        x1 = self.involution1(x1)

        x2 = self.F2(x)
        x2 = self.involution2(x2)

        x3 = self.F3(x)
        x3 = self.involution3(x3)

        return (x1, x2, x3)
