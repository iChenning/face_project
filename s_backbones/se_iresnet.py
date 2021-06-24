import torch
from torch import nn

__all__ = ['se_iresnet18', 'se_iresnet34', 'se_iresnet50', 'se_iresnet100', 'se_iresnet200']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, sub_cfg, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, reduction=16):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(sub_cfg[0], eps=1e-05, )
        self.conv1 = conv3x3(sub_cfg[0], sub_cfg[1])
        self.bn2 = nn.BatchNorm2d(sub_cfg[1], eps=1e-05, )
        self.prelu = nn.PReLU(sub_cfg[1])
        self.conv2 = conv3x3(sub_cfg[1], sub_cfg[2], stride)
        self.bn3 = nn.BatchNorm2d(sub_cfg[2], eps=1e-05, )
        self.downsample = downsample
        self.stride = stride
        self.se_module = SEModule(sub_cfg[2], reduction=reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.se_module(out) + identity
        out = self.relu(out)
        # out += identity
        return out


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout=0, embedding_size=512, zero_init_residual=False, cfg=None,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        if cfg is None:
            cfg = [64] + [64] * layers[0] * 2 + [128] * layers[1] * 2 + [256] * layers[2] * 2 + [512] * layers[3] * 2
        else:
            cfg_ = [64] + [64] * layers[0] * 2 + [128] * layers[1] * 2 + [256] * layers[2] * 2 + [512] * layers[3] * 2
            for i_, v_ in enumerate(cfg):
                cfg_[i_ * 2 + 1] = v_
            cfg = cfg_[:]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0], eps=1e-05)
        self.prelu = nn.PReLU(cfg[0])
        self.layer1 = self._make_layer(block, layers[0], cfg[0: sum(layers[:1]) * 2 + 1], stride=2)
        self.layer2 = self._make_layer(block,
                                       layers[1],
                                       cfg[sum(layers[:1]) * 2: sum(layers[:2]) * 2 + 1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       layers[2],
                                       cfg[sum(layers[:2]) * 2: sum(layers[:3]) * 2 + 1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       layers[3],
                                       cfg[sum(layers[:3]) * 2: sum(layers[:]) * 2 + 1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(cfg[-1], eps=1e-05, )
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(cfg[-1] * block.expansion * self.fc_scale, embedding_size)
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, blocks, sub_cfg, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(sub_cfg[0], sub_cfg[2], stride),
                nn.BatchNorm2d(sub_cfg[2], eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(sub_cfg[0: 3], stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        for idx in range(1, blocks):
            # downsample = nn.Sequential(
            #     conv1x1(sub_cfg[idx * 2: idx * 2 + 3][0], sub_cfg[idx * 2: idx * 2 + 3][2], stride=1),
            #     nn.BatchNorm2d(sub_cfg[idx * 2: idx * 2 + 3][2], eps=1e-05, ),
            # )
            layers.append(
                block(sub_cfg[idx * 2: idx * 2 + 3],
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, **kwargs):
    model = IResNet(block, layers, **kwargs)
    return model


def se_iresnet18(**kwargs):
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], **kwargs)


def se_iresnet34(**kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], **kwargs)


def se_iresnet50(**kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], **kwargs)


def se_iresnet100(**kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], **kwargs)


def se_iresnet200(**kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], **kwargs)


if __name__ == '__main__':
    f_ = open(r'E:pruned_info\glint360k-se_iresnet100-0.3.txt')
    cfg_ = [int(x) for x in f_.read().split()]
    f_.close()
    net = se_iresnet100(cfg=cfg_)
    print(net)

    # macs-params
    from thop import profile

    macs, params = profile(net, inputs=(torch.rand(1, 3, 112, 112),))
    print('macs:', round(macs / 1e9, 2), 'G, params:', round(params / 1e6, 2), 'M')
