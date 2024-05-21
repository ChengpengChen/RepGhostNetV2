# @Author  : chengpeng.chen
# @Email   : chencp@live.com
"""
RepGhostNetV2: When RepGhost meets MobileNetV4
"""

from univnet import UniversalNet


__all__ = [
    'repghostnetv2_small',
    'repghostnetv2_medium',
    'repghostnetv2_large'
]


def repghostnetv2_small(**kwargs):
    """
    Constructs a RepGhostNetV2Small
    """
    block_spec_schema = [
        'block_fn',
        'act_layer',
        'kernel_size',
        'start_dw_kernel_size',
        'middle_dw_kernel_size',
        'middle_dw_downsample',
        'end_dw_kernel_size',
        'stride',
        'out_channels',
        'expand_ratio',
    ]
    block_specs = [
        # 112px after stride 2.
        ('convbn', 'relu', 3, None, None, False, None, 2, 32, None),
        # 56px.
        ('convbn', 'relu', 3, None, None, False, None, 2, 32, None),
        ('convbn', None, 1, None, None, False, None, 1, 32, None),
        # 28px.
        ('convbn', 'relu', 3, None, None, False, None, 2, 96, None),
        ('convbn', None, 1, None, None, False, None, 1, 64, None),
        # 14px.
        ('uib', 'relu', None, 5, 5, True, 0, 2, 96, 3.0),
        ('uib', 'relu', None, 0, 3, True, 0, 1, 96, 2.0),
        ('uib', 'relu', None, 0, 3, True, 0, 1, 96, 2.0),
        ('uib', 'relu', None, 0, 3, True, 0, 1, 96, 2.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 96, 2.0),
        ('uib', 'relu', None, 0, 0, True, 3, 1, 96, 4.0),
        # 7px
        ('uib', 'relu', None, 0, 3, True, 5, 2, 128, 6.0),
        ('uib', 'relu', None, 0, 5, True, 0, 1, 128, 4.0),
        ('uib', 'relu', None, 0, 5, True, 0, 1, 128, 4.0),
        ('uib', 'relu', None, 0, 5, True, 0, 1, 128, 3.0),
        ('uib', 'relu', None, 0, 3, True, 0, 1, 128, 4.0),
        ('uib', 'relu', None, 0, 3, True, 0, 1, 128, 4.0),
        ('convbn', 'relu', 1, None, None, False, 0, 1, 960, None)
    ]

    return UniversalNet(block_specs, block_spec_schema, **kwargs)


def repghostnetv2_medium(**kwargs):
    """
    Constructs a RepGhostNetV2Medium
    """
    block_spec_schema = [
        'block_fn',
        'act_layer',
        'kernel_size',
        'start_dw_kernel_size',
        'middle_dw_kernel_size',
        'middle_dw_downsample',
        'end_dw_kernel_size',
        'stride',
        'out_channels',
        'expand_ratio',
    ]
    block_specs = [
        # 128px after stride 2.
        ('convbn', 'relu', 3, None, None, False, 0, 2, 32, None),
        # 64px.
        ('convbn', 'relu', 3, None, None, False, 0, 2, 128, None),
        ('convbn', None, 1, None, None, False, 0, 1, 48, None),
        # 32px.
        ('uib', 'relu', None, 3, 5, True, 3, 2, 80, 4.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 80, 2.0),
        # 16px.
        ('uib', 'relu', None, 0, 5, True, 3, 2, 160, 6.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 160, 4.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 160, 4.0),
        ('uib', 'relu', None, 0, 5, True, 3, 1, 160, 4.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 160, 4.0),
        ('uib', 'relu', None, 0, 0, True, 0, 1, 160, 4.0),
        ('uib', 'relu', None, 0, 0, True, 3, 1, 160, 2.0),
        ('uib', 'relu', None, 0, 0, True, 5, 1, 160, 4.0),
        # 8px
        ('uib', 'relu', None, 0, 5, True, 5, 2, 256, 6.0),
        ('uib', 'relu', None, 0, 5, True, 3, 1, 256, 4.0),
        ('uib', 'relu', None, 0, 5, True, 3, 1, 256, 4.0),
        ('uib', 'relu', None, 0, 5, True, 0, 1, 256, 4.0),
        ('uib', 'relu', None, 0, 0, True, 3, 1, 256, 4.0),
        ('uib', 'relu', None, 0, 0, True, 3, 1, 256, 4.0),
        ('uib', 'relu', None, 0, 5, True, 5, 1, 256, 2.0),
        ('uib', 'relu', None, 0, 5, True, 0, 1, 256, 4.0),
        ('uib', 'relu', None, 0, 0, True, 0, 1, 256, 4.0),
        ('uib', 'relu', None, 0, 0, True, 5, 1, 256, 4.0),
        ('uib', 'relu', None, 0, 0, True, 0, 1, 256, 2.0),
        ('convbn', 'relu', 1, None, None, False, 0, 1, 960, None)
    ]

    return UniversalNet(block_specs, block_spec_schema, **kwargs)


def repghostnetv2_large(**kwargs):
    """
    Constructs a RepGhostNetV2Large
    """
    block_spec_schema = [
        'block_fn',
        'act_layer',
        'kernel_size',
        'start_dw_kernel_size',
        'middle_dw_kernel_size',
        'middle_dw_downsample',
        'end_dw_kernel_size',
        'stride',
        'out_channels',
        'expand_ratio',
    ]
    block_specs = [
        # 192px after stride 2.
        ('convbn', 'relu', 3, None, None, False, 0, 2, 24, None),
        # 96px.
        ('convbn', 'relu', 3, None, None, False, 0, 2, 96, None),
        ('convbn', None, 1, None, None, False, 0, 1, 48, None),
        # 48px.
        ('uib', 'relu', None, 3, 5, True, 3, 2, 96, 4.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 96, 4.0),
        # 24px.
        ('uib', 'relu', None, 0, 5, True, 3, 2, 192, 4.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 5, True, 5, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 3, True, 5, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 3, True, 5, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 3, True, 5, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 3, True, 5, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 3, True, 3, 1, 192, 4.0),
        ('uib', 'relu', None, 0, 0, True, 5, 1, 192, 4.0),
        # 12px
        ('uib', 'relu', None, 0, 5, True, 5, 2, 512, 4.0),
        ('uib', 'relu', None, 0, 5, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 5, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 5, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 0, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 3, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 0, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 0, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 3, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 5, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 0, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 0, True, 5, 1, 512, 4.0),
        ('uib', 'relu', None, 0, 0, True, 0, 1, 512, 4.0),
        ('convbn', 'relu', 1, None, None, False, 0, 1, 960, None)
    ]

    return UniversalNet(block_specs, block_spec_schema, **kwargs)


if __name__ == "__main__":
    import torch
    model = repghostnetv2_small().eval()
    print(model)
    input = torch.randn(1, 3, 224, 224)
    # y = model(input)
    # print(y.size())

    import sys

    sys.path.append("../")
    from tools import cal_flops_params

    flops, params = cal_flops_params(model, input_size=input.shape)
