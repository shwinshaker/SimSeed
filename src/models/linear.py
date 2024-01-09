#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['get_linear']

class DLN(nn.Module):
    """
        Deep linear network, for illustration
    """

    def __init__(self, features, num_classes=1000, gain=1.0, out=3072): # 3 * 32 * 32 = 3072
        super(DLN, self).__init__()
        self.gain = gain
        self.features = features
        self.classifier = nn.Linear(out, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                assert(n == fan_out)
                m.weight.data.normal_(0, self.gain * math.sqrt(0.5 / n)) # !!!
                # m.weight.data.normal_(0, self.gain * math.sqrt(1. / (fan_in + fan_out))) # !!!
                # nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Affine(nn.Module):

    def __init__(self):
        super(Affine, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # return self.weight * x # + self.bias
        return x + self.bias


def make_layers(model, cfg, n_channel=3, batch_norm=False):
    layers = []
    # for logisitc bn, add bn before final layer
    if model in ['logistic'] and batch_norm:
        layers.append(nn.BatchNorm2d(n_channel, affine=False))
        # layers.append(nn.GroupNorm(n_channel, n_channel)) # equivalent to instance normalization
        # layers.append(Affine())

    in_channels = n_channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            # layers += [nn.AdaptiveAvgPool2d((1, 1))]
        elif v == 'C':
            layers += [nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, bias=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                # layers += [conv2d, nn.BatchNorm2d(v)]
                layers += [conv2d, nn.GroupNorm(v, v)] # equivalent to instance normalization
            else:
                layers += [conv2d]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {'logistic': {'mnist': {'arch': [], 'out': 784},
                    'cifar10': {'arch': [], 'out': 3072},
                    'cifar100': {'arch': [], 'out': 3072}},
       'dln11': {'mnist': {'arch': [8, 'A', 16, 'A', 32, 32, 'A', 64, 64, 'A'], 'out': 64}},
       # 'dln': {'mnist': {'arch': [8, 'C', 16, 16, 16, 16, 16, 16], 'out': 3136}}, # pool = 14
       'dln': {'mnist': {'arch': [8, 'C', 16, 16, 16, 16, 16, 16, 'A'], 'out': 784}}, # pool = 7
       # 'dln': {'mnist': {'arch': [8, 'C', 16, 16, 16, 16, 16, 16, 'A'], 'out': 144}}, # pool = 3
       # 'dln': {'mnist': {'arch': [8, 'C', 16, 16, 16, 16, 16, 16, 'A'], 'out': 16}}, # pool = 1
       # 'dln11': {'mnist': {'arch': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M'], 'out': 64}},
       'A_mnist_wopool': {'arch': [8, 16, 32, 32, 64, 64], 'out': 1024},
       # 'A': [64, 128, 256, 256, 512, 512, 512, 512],
       # 'AM': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       # 'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       # 'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       # 'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
      }


# def logistic(**kwargs):
#     model = DLN(make_layers(cfg['Logistic'][kwargs['dataset']], n_channel=kwargs['n_channel'], batch_norm=kwargs['batch_norm']),
#                 out=cfg['Logistic'][kwargs['dataset']], **kwargs)
#     return model
# 
# 
# def dln11(**kwargs):
#     model = DLN(make_layers(cfg['dln11'][kwargs['dataset']], n_channel=kwargs['n_channel'], batch_norm=kwargs['batch_norm']),
#                 out=cfg['dln11'][kwargs['dataset']], **kwargs)
#     # model = DLN(make_layers(cfg['A_mnist_wopool']['arch'], n_channel=n_channel, batch_norm=True),
#     #             out=cfg['A_mnist_wopool']['out'], **kwargs)
#     return model
# 

def get_linear(model, **kwargs):
    return DLN(make_layers(model, cfg[model][kwargs['dataset']]['arch'], n_channel=kwargs['n_channel'], batch_norm=kwargs['batch_norm']),
               out=cfg[model][kwargs['dataset']]['out'], gain=kwargs['gain'], num_classes=kwargs['num_classes'])
    # if model == 'dln11':
    #     return dln11(**kwargs)

    # if model == 'dln11_bn':
    #     return dln11_bn(**kwargs)

    # if model == 'dln11_lite':
    #     return dln11_lite(**kwargs)

    # if model == 'dln19':
    #     return dln19(**kwargs)

    # if model == 'dln19_bn':
    #     return dln19_bn(**kwargs)

    # raise KeyError(model)

