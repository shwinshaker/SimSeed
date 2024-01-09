'''Pre-activation ResNet in PyTorch.
    with Fixup initialization

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['FixupPreActResNet18']

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bias1_relu = nn.Parameter(torch.zeros(1))
        self.bias1_conv = nn.Parameter(torch.zeros(1))
        self.bias2_relu = nn.Parameter(torch.zeros(1))
        self.bias2_conv = nn.Parameter(torch.zeros(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

        if stride != 1 or in_planes != self.expansion*planes:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            # )
            self.shortcut = nn.AvgPool2d(1, stride=stride)

    def forward(self, x):
        # out = F.relu(self.bn1(x))
        out = F.relu(x + self.bias1_relu)
        if hasattr(self, 'shortcut'):
            # shortcut = self.shortcut(out + self.bias1_relu)
            shortcut = self.shortcut(x + self.bias1_relu)
            # Fast version
            # shortcut = self.shortcut(x) 
            shortcut = torch.cat((shortcut, torch.zeros_like(shortcut)), 1)
            # double the depth - should be motivated from augment ODE
        else:
            shortcut = x
        out = self.conv1(out + self.bias1_conv)

        # out = F.relu(self.bn2(out))
        out = F.relu(out + self.bias2_relu) 
        out = self.conv2(out + self.bias2_conv)
        out = out * self.scale + self.bias

        out += shortcut
        return out


class FixupPreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, gain=1.0):
        super(FixupPreActResNet, self).__init__()
        self.in_planes = 64

        self.num_layers = sum(num_blocks)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, PreActBlock):
                nn.init.normal_(m.conv1.weight, mean=0,
                                std=gain * np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # out = F.relu(self.bn(out))
        out = F.relu(out + self.bias1)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out + self.bias2)
        return out


def FixupPreActResNet18(num_classes=10):
    return FixupPreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def FixupPreActResNet34():
    return FixupPreActResNet(PreActBlock, [3,4,6,3])

# def FixupPreActResNet50():
#     return PreActResNet(PreActBottleneck, [3,4,6,3])
# 
# def FixupPreActResNet101():
#     return PreActResNet(PreActBottleneck, [3,4,23,3])
# 
# def FixupPreActResNet152():
#     return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = FixupPreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

test()
