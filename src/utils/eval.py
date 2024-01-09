#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from torch.cuda.amp import autocast
from . import safe_divide

__all__ = ['F1Meter', 'AverageMeter', 'GroupMeter', 'accuracy', 'alignment', 'criterion_r', 'mse_one_hot', 'ce_soft']

class F1Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = {}
        self.fp = {}
        self.fn = {}
        self.count = 0
        self.f1 = 0
    
    def update(self, predictions, labels):
        self.count += len(predictions)
        unique_labels = np.unique(np.concatenate([np.unique(labels),
                                                  np.unique(predictions)]))
        for label in unique_labels:
            if label not in self.tp:
                self.tp[label] = 0
            if label not in self.fp:
                self.fp[label] = 0
            if label not in self.fn:
                self.fn[label] = 0
            self.tp[label] += np.sum((labels==label) & (predictions==label))
            self.fp[label] += np.sum((labels!=label) & (predictions==label))
            self.fn[label] += np.sum((predictions!=label) & (labels==label))

    @property
    def macro_f1(self):
        f1s = []
        for label in self.tp:
            precision = safe_divide(self.tp[label], (self.tp[label]+self.fp[label]))
            recall = safe_divide(self.tp[label], (self.tp[label]+self.fn[label]))
            f1s.append(safe_divide(2 * (precision * recall), (precision + recall)))
        return np.mean(f1s) * 100.0
    
    @property
    def micro_f1(self):
        tp = np.sum([self.tp[label] for label in self.tp])
        fp = np.sum([self.fp[label] for label in self.fp])
        fn = np.sum([self.fn[label] for label in self.fn])
        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        return 2 * (precision * recall) / (precision + recall)  * 100.0


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GroupMeter:
    """
        measure the accuracy of each class
    """
    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.meters = [AverageMeter() for _ in range(self.num_classes)]

    def update(self, out, las):
        _, preds = out.topk(1, 1, True, True)
        preds = preds.squeeze()
        for c in range(self.num_classes):
            num_c = (las == c).sum().item()
            if num_c == 0:
                continue
            acc = ((preds == las) & (las == c)).sum() * 100. / num_c
            self.meters[c].update(acc.item(), num_c)

    def output(self):
        return np.mean(self.output_group())

    def output_group(self):
        return [self.meters[i].avg for i in range(self.num_classes)]

    def pprint(self):
        print('')
        for i in range(self.num_classes):
            print('%10s: %.4f' % (self.classes[i], self.meters[i].avg))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def alignment(output1, output2):
    batch_size = output1.size(0)

    _, pred1 = output1.topk(1, dim=1, largest=True)
    pred1 = pred1.t()
    _, pred2 = output2.topk(1, dim=1, largest=True)
    pred2 = pred2.t()
    correct = pred1.eq(pred2)
    return correct.view(-1).float().sum(0).mul_(100.0 / batch_size)


def ce_soft(reduction='mean', num_classes=10, soft_label=True, temperature=1.0):
    def ce(outputs, labels):
        assert(outputs.size(1) == num_classes), (outputs.size(), num_classes)
        if not soft_label:
            if len(labels.size()) > 1:
                _, labels = torch.max(labels, 1)
            return torch.nn.functional.cross_entropy(outputs, labels, reduction=reduction)
        
        assert(len(labels.size()) == 2), ('soft labels required, got size: ', labels.size())
        log_probas = torch.nn.functional.log_softmax(outputs / temperature, dim=1)
        nll = -(log_probas * labels) * temperature**2 # normalize
        loss = nll.sum(dim=1)
        if reduction == 'mean':
            return loss.mean()
        return loss
    return ce


def mse_one_hot(reduction='mean', num_classes=10, soft_label=False, temperature=1.0):
    def mse(outputs, labels):
        assert(outputs.size(1) == num_classes), (outputs.size(), num_classes)
        probas = torch.nn.functional.softmax(outputs / temperature, dim=1)
        if not soft_label:
            if len(labels.size()) == 1:
                # provided labels are hard label
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            else:
                # provided labels are soft label
                _, labels = torch.max(labels, 1)
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        if reduction == 'mean':
            return nn.MSELoss(reduction=reduction)(probas, labels) * num_classes # mse_loss in pytorch average over dimension too
        return nn.MSELoss(reduction='none')(probas, labels).sum(dim=1) # reduce class dimension, keep batch dimension
    return mse


def criterion_r(output1, output2, c=None):

    if isinstance(c, nn.CrossEntropyLoss):
        # https://discuss.pytorch.org/t/how-should-i-implement-cross-entropy-loss-with-continuous-target-outputs/10720/17
        def cross_entropy(pred, soft_targets):
            logsoftmax = nn.functional.log_softmax
            return torch.mean(torch.sum(- soft_targets * logsoftmax(pred, dim=1), 1))
        return cross_entropy(output1, output2)
        
    raise NotImplementedError(type(c))

