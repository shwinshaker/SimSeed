#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['ffn']

class FFN(nn.Module):
    """
        Deep feed-forward network, for illustration
    """

    def __init__(self, features=[], dim0=3, n_class=1):
        super(FFN, self).__init__()
        self.features = nn.Sequential()

        dim = dim0
        for idx, l in enumerate(features):
            layer = nn.Sequential()
            if l['width']:
                layer.add_module('linear', nn.Linear(dim, l['width']))
                dim = l['width']
            if l['act']:
                layer.add_module(l['act'], self.__get_activation(l['act']))
            if l['bn']:
                layer.add_module('bn', nn.BatchNorm1d(dim, affine=False))
            self.features.add_module('layer%i' % idx, layer)
        # self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(dim, n_class)
        # self.activation = nn.Sigmoid()

        self.__initialize()

    def __get_activation(self, act):
        if act == 'tanh':
            return nn.Tanh()
        if act == 'relu':
            return nn.ReLU()
        raise KeyError(act)

    def __initialize(self):
        for m in self.features:
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight.data)
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.orthogonal_(m.weight.data)
                m.bias.data.zero_()
        # nn.init.xavier_uniform_(self.classifier.weight.data)
        nn.init.xavier_normal_(self.classifier.weight.data)
        # nn.init.orthogonal_(self.classifier.weight.data)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.features:
            x = self.features(x)
        x = self.classifier(x)
        return x 


def make_layer(width=[], act='tanh', depth=5, bn=False):
    if not depth:
        # logisitc
        return [{'width': None, 'act': None, 'bn': bn}]

    if not width:
        return [{'width': None, 'act': act, 'bn': bn}]

    if isinstance(width, int):
        return [{'width': width, 'act': act, 'bn': bn}] * depth
    elif isinstance(width, list):
        return [{'width': w, 'act': act, 'bn': bn} for w in width]
    raise TypeError(width)


def ffn(**kwargs):
    print(kwargs)
    if kwargs['dataset'] == 'mnist':
        dim0 = 784
    else:
        raise KeyError(kwargs['dataset'])
    return FFN(make_layer(width=kwargs['width'], depth=kwargs['depth'], bn=kwargs['bn']), dim0=dim0, n_class=kwargs['num_classes']) 

