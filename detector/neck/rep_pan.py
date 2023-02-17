import torch
from torch import nn


class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        return self.upsample_transpose(x)

class SimConv(nn.Module):
    '''Normal Conv with SiLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Intialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, e=None):
        super().__init__()
        self.conv1 = RepVGGBlock(in_channels, out_channels)
        self.block = nn.Sequential(*(RepVGGBlock(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

class RepPAN(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """
    def __init__(self, in_channels=[256, 512, 1024], mid_channels = [128, 128, 256], out_channels=[128, 256, 512], layers=[12, 12, 12, 12], depth_mul=1.0, width_mul=1.0):
        super().__init__()
        assert in_channels is not None
        assert layers is not None

        layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))
        in_channels = list(map(lambda x: int(x * width_mul), in_channels))
        out_channels = list(map(lambda x: int(x * width_mul), out_channels))
        mid_channels = list(map(lambda x: int(x * width_mul), mid_channels))

        self.reduce_layer0 = SimConv(in_channels=in_channels[2], out_channels=mid_channels[2], kernel_size=1, stride=1)
        self.upsample0 = Transpose(in_channels=mid_channels[2], out_channels=mid_channels[2])
        self.Rep_p4 = RepBlock(in_channels=in_channels[1] + mid_channels[2], out_channels=mid_channels[2], n=layers[0])

        self.reduce_layer1 = SimConv(in_channels=mid_channels[2], out_channels=mid_channels[1], kernel_size=1, stride=1)
        self.upsample1 = Transpose(in_channels=mid_channels[1], out_channels=mid_channels[1])
        self.Rep_p3 = RepBlock(in_channels=in_channels[0] + mid_channels[1], out_channels=out_channels[0], n=layers[1])

        self.downsample2 = SimConv(in_channels=out_channels[0], out_channels=mid_channels[0], kernel_size=3, stride=2)
        self.Rep_n3 = RepBlock(in_channels=mid_channels[1] + mid_channels[0], out_channels=out_channels[1], n=layers[2])

        self.downsample1 = SimConv(in_channels=out_channels[1], out_channels=mid_channels[2], kernel_size=3, stride=2)
        self.Rep_n4 = RepBlock(in_channels=mid_channels[2] + mid_channels[2], out_channels=out_channels[2], n=layers[3])


    def forward(self, input):
        x2, x1, x0 = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs


if __name__ == '__main__':

    x1 = torch.randn(2, 256, 48, 48)
    x2 = torch.randn(2, 512, 24, 24)
    x3 = torch.randn(2, 1024, 12, 12)
    model = RepPAN()
    print(model((x1, x2, x3)))