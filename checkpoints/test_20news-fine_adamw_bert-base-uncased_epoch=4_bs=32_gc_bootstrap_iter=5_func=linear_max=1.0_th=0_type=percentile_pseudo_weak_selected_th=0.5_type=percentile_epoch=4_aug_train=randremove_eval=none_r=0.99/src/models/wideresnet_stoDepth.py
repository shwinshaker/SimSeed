import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['wrn_stoDepth']

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, prob=1.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))

    def forward(self, x):
        if self.training:

            if torch.equal(self.m.sample(), torch.ones(1)):

                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                if not self.equalInOut: # this is not exactly the same as typical resnet
                    x = self.relu1(self.bn1(x))
                else:
                    out = self.relu1(self.bn1(x))
                out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
                if self.droprate > 0:
                    out = F.dropout(out, p=self.droprate, training=self.training)
                out = self.conv2(out)
                out = torch.add(x if self.equalInOut else self.convShortcut(x), out)

            else:

                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False

                out = x if self.equalInOut else self.convShortcut(x)

        else:

            if not self.equalInOut:
                x = self.relu1(self.bn1(x))
            else:
                out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            out = torch.add(x if self.equalInOut else self.convShortcut(x), out)

        return out


class WideResNet(nn.Module):

    __num_layers = 3

    def __init__(self, depth, widen_factor=1, num_classes=10, n_channel=3, dropRate=0.0, prob_0_L=[1.0, 0.5]):
        super(WideResNet, self).__init__()

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        nb_layers = (depth - 4) / 6

        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0] - prob_0_L[1]
        self.prob_step = self.prob_delta/(nb_layers * self.__num_layers - 1)

        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(n_channel, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # 1st block
        self.block1 = self._make_layer(block, nChannels[0], nChannels[1], nb_layers, stride=1, dropRate=dropRate)
        # 2nd block
        self.block2 = self._make_layer(block, nChannels[1], nChannels[2], nb_layers, stride=2, dropRate=dropRate)
        # 3rd block
        self.block3 = self._make_layer(block, nChannels[2], nChannels[3], nb_layers, stride=2, dropRate=dropRate)

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride=1, dropRate=0.0):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes,
                                out_planes,
                                i == 0 and stride or 1,
                                dropRate=dropRate,
                                prob=self.prob_now))
            self.prob_now = self.prob_now - self.prob_step
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wrn_stoDepth(**kwargs):
    return WideResNet(**kwargs)


def test():
    net = wrn_stoDepth(depth=28)
    net.eval()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

test()
