# @Author  : chengpeng.chen
# @Email   : chencp@live.com
"""
RepGhostNetV2: When RepGhost meets MobileNetV4

refer to
https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act_layer='relu', **kwargs):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=kwargs.get('groups', 1), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        if not act_layer:
            self.act1 = nn.Identity()
        elif act_layer == 'relu':
            self.act1 = nn.ReLU(inplace=True)
        else:
            assert False, f"{act_layer}: act_layer not support yet"

    def forward(self, x):
        return self.act1(self.bn1(self.conv(x)))


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 end_dw_kernel_size=0,
                 stride=1,
                 expand_ratio=3,
                 act_layer='relu',
                 use_residual=True,
                 drop_path_rate=0,
                 **kwargs
                 ):
        super(UniversalInvertedBottleneckBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_residual = use_residual
        self.shortcut = self.use_residual and self.in_channels == self.out_channels and stride == 1

        self.stride = stride

        assert 0 <= drop_path_rate < 1
        self.stochastic_depth = torchvision.ops.StochasticDepth(drop_path_rate, mode='row') if drop_path_rate else None

        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self.start_dw_ = ConvBnAct(in_channels, in_channels, kernel_size=start_dw_kernel_size, stride=stride_, groups=in_channels, act_layer=None)

        # Expansion with 1x1 convs.
        expand_filters = _make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = ConvBnAct(in_channels, expand_filters, kernel_size=1, act_layer=act_layer)

        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self.middle_dw = ConvBnAct(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                        groups=expand_filters, act_layer=act_layer)
        # Projection with 1x1 convs.
        self.proj_conv = ConvBnAct(expand_filters, out_channels, kernel_size=1, act_layer=None)

        # Ending depthwise conv.
        self.end_dw_kernel_size = end_dw_kernel_size
        if self.end_dw_kernel_size:
            self.end_dw = ConvBnAct(out_channels, out_channels, kernel_size=self.end_dw_kernel_size, groups=out_channels, act_layer=None)

    def forward(self, input):
        shortcut = input
        x = input
        if self.start_dw_kernel_size:
            x = self.start_dw_(x)
        x = self.expand_conv(x)
        if self.middle_dw_kernel_size:
            x = self.middle_dw(x)
        x = self.proj_conv(x)
        if self.end_dw_kernel_size:
            x = self.end_dw(x)
        if self.shortcut:
            if self.stochastic_depth:
                x = self.stochastic_depth(x)
            x = x + shortcut
        return x

block_fn_map = {
    'convbn': ConvBnAct,
    'uib': UniversalInvertedBottleneckBlock
}

class UniversalNet(nn.Module):
    def __init__(
        self,
        block_spec,
        block_spec_schema,
        in_channels=3,
        num_classes=1000,
        dropout=0.0,
        **kwargs
    ):
        super(UniversalNet, self).__init__()
        # setting of inverted residual blocks
        self.block_spec = block_spec
        self.block_spec_schema = block_spec_schema
        self.num_classes = num_classes
        self.dropout = dropout

        layers = []
        for i, block_param in enumerate(block_spec):
            block_fn = block_fn_map[block_param[0]]
            block_kwargs = {k: v for k, v in zip(block_spec_schema[1:], block_param[1:])}
            layers.append(block_fn(in_channels=in_channels, **block_kwargs))
            in_channels = block_kwargs['out_channels']

        self.layers = nn.Sequential(*layers)

        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(
            in_channels, output_channel, 1, 1, 0, bias=True,
        )
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)

        return x
