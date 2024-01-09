#!./env python

import numpy as np
import torch
import torch.nn as nn

__all__ = ['Mixupper']

class Mixupper:

    # if add extra mixed inputs, make double calculation, which is same as gradients
    #   Therefore just use the interpolated inputs

    def __init__(self, net, criterion, accuracy, alpha=0.2, multi=False, randomize=False, device=None):
        """
            alpha: beta distribution parameter
            randomize: if false, same lambda for all inputs (original version)
                       else, different lambdas for inputs
        """
        self.alpha = alpha
        self.multi= multi
        self.randomize = randomize
        self.device = device

        """
            need to modify the criterion and accuracy calculation because of soft labels in mixup
        """
        self.net = net
        self.criterion = criterion
        self.accuracy = accuracy

        if self.randomize:
            # Non-reduced version
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.accuracy = self.__accuracy

        if self.multi:
            self.evaluate = self.evaluate_multi
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            self.accuracy = self.__accuracy

    def evaluate(self, inputs, labels):
        lamb, inputs, labels_a, labels_b = self.mixup(inputs, labels)
        outputs = self.net(inputs)
        loss = self.mixup_criterion(lamb, outputs, labels_a, labels_b)
        prec1 = self.mixup_accuracy(lamb, outputs, labels_a, labels_b)
        return loss, prec1, inputs

    def evaluate_multi(self, inputs, labels):
        lamb, inputs = self.mixup_multi(inputs)
        outputs = self.net(inputs)
        loss = self.mixup_criterion_multi(lamb, outputs, labels)
        prec1 = self.mixup_accuracy_multi(lamb, outputs, labels)
        return loss, prec1, inputs

    def __get_lamb(self, alpha, size=1):
        
        def to_torch(arr):
            return torch.as_tensor(arr).float().to(self.device)

        if size == 1:
            lamb = np.random.beta(alpha, alpha)
            # Out-of-hull - not working
            # lamb += np.array(lamb < 0.5).astype(int) * 2. - 1.
            return to_torch(lamb)

        assert isinstance(size, tuple)
        # raise NotImplementedError # Not implemented for out-of-hull
        return to_torch(np.random.beta(alpha, alpha, size))

    def mixup(self, inputs, labels):
        if self.alpha == 0:
            return inputs, labels

        batch_size = inputs.size(0)

        indices = torch.randperm(batch_size)
        if self.randomize:
            lamb = self.__get_lamb(self.alpha, (batch_size, ))
            inputs_mix = lamb.view(batch_size, 1, 1, 1) * inputs + (1 - lamb.view(batch_size, 1, 1, 1)) * inputs[indices, :]
        else:
            lamb = self.__get_lamb(self.alpha)
            inputs_mix = lamb * inputs + (1 - lamb) * inputs[indices, :]

        # cannot just mix the labels because pytorch crossentropy doesn't support soft labels
        # labels_mix = lamb * labels + (1 - lamb) * labels[indices] 
        return lamb, inputs_mix, labels, labels[indices]

    def mixup_multi(self, inputs):
        if self.alpha == 0:
            return inputs, labels
        batch_size = inputs.size(0)

        lambs = self.__get_lamb(self.alpha, (batch_size, batch_size))
        lambs /= torch.sum(lambs, dim=1, keepdim=True)
        inputs_mix = torch.einsum('mb, bijk-> mijk', lambs, inputs)

        return lambs, inputs_mix

    def mixup_criterion(self, lamb, outputs, labels_a, labels_b):
        """
            if randomize, this criterion cannot be used, should multiplide by lambda before reduction
        """
        if self.randomize:
            # reduction - sum
            return (lamb * self.criterion(outputs, labels_a) + (1 - lamb) * self.criterion(outputs, labels_b)).sum() / outputs.size(0)
        return lamb * self.criterion(outputs, labels_a) + (1 - lamb) * self.criterion(outputs, labels_b)

    def mixup_criterion_multi(self, lambs, outputs, labels):
        batch_size = labels.size(0)
        losses = torch.stack([self.criterion(outputs, labels[i].repeat(batch_size)) for i in range(batch_size)])
        return torch.diagonal(torch.einsum('mb, bn-> mn', lambs, losses)).sum() / batch_size

    def mixup_accuracy(self, lamb, outputs, labels_a, labels_b):
        if self.randomize:
            # reduction - sum
            return (lamb * self.accuracy(outputs.data, labels_a.data) + (1 - lamb) * self.accuracy(outputs.data, labels_b.data)).sum() / outputs.size(0)
        return lamb.item() * self.accuracy(outputs.data, labels_a.data)[0] + (1 - lamb.item()) * self.accuracy(outputs.data, labels_b.data)[0]

    def mixup_accuracy_multi(self, lambs, outputs, labels):
        batch_size = labels.size(0)
        corrects = torch.stack([self.accuracy(outputs.data, labels[i].data.repeat(batch_size)) for i in range(batch_size)])
        return torch.diagonal(torch.einsum('mb, bn-> mn', lambs, corrects)).sum() / batch_size

    def __accuracy(self, outputs, labels):
        # accuracy without reduction
        _, pred = outputs.topk(1, 1, True, True)
        correct = pred.t().eq(labels.view(1, -1)).view(-1).float()
        return correct



