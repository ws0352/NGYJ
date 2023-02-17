import torch.nn as nn
from detector.backbone import Backbone
from detector.neck.rep_pan import RepPAN
from detector.head.BetterHead import GFocalHeadV2



class Block(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = Backbone()
        self.neck = RepPAN(in_channels=[64, 128, 256])
        self.head = GFocalHeadV2(num_classes, [128, 256, 512])
    def forward(self, x, boxes, labels):
        f1, f2, f3 = self.backbone(x)
        x = self.neck((f1, f2, f3))
        return self.head(x,  boxes, labels)
        # for i in range(len(x)):
        #     print(x[i].shape)
        # print()
        # print(self.head)
