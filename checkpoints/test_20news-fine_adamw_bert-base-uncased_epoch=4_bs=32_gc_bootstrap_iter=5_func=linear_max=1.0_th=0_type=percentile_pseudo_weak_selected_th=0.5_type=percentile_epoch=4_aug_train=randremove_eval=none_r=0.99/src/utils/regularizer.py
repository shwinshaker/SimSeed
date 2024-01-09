#!./env python

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import torch

from . import AverageMeter, F1Meter, accuracy, Logger, nan_filter, load_log
from . import random_label_exclude_batch
from . import Tracker

# __all__ = ['AntiBackdoor', 'AdversarialRemove', 'RandomRemove', 'ContrastiveBackdoor',
        #    'LabelSmoothingCrossEntropy', 'LossFloodingCrossEntropy']


class ContrastiveBackdoor:
    # L(f(x), f(x / seed word))
    # TODO
    def __init__(self):
        pass


class GroupDRO:
    # mix max_{seed} L(f(x with seed), y)
    def __init__(self):
        pass

class Regularizer(Tracker):
    def __init__(self, model, criterion, time_start=None, config=None, weight_func=None, log_name='log_train_reg'):
        super().__init__(model, criterion, time_start=time_start, config=config)

        self.weight_func = weight_func
        assert(weight_func() <= 1), f'invalid loss weight {weight_func():g}'
        self.reset_batch()
        self.set_log(log_name=log_name,
                     entry_dict={'Alpha': self.weight_func,
                                 'Train-Loss': lambda: self.loss_meter.avg,
                                 'Train-Acc': lambda: self.acc_meter.avg})

    def criterion_descent(self, inputs, labels):
        raise NotImplementedError

    def update_batch(self, inputs, labels):
        loss, acc = self.criterion_descent(inputs, labels)
        self.loss_meter.update(loss.item(), labels.size(0))
        self.acc_meter.update(acc.item(), labels.size(0))
        return self.weight_func() * loss

        # For backup, if needs to repeat augmentation
        # if hasattr(self.config, 'antibackdoor_repeat') and self.config.antibackdoor_repeat > 1:
        #     losses, acces = [], []
        #     for _ in range(self.config.antibackdoor_repeat):
        #         loss, acc = self.criterion_descent(inputs, labels)
        #         losses.append(loss)
        #         acces.append(acc)
        #     loss = torch.tensor(losses).mean()
        #     acc = torch.tensor(acces).mean()
        # else:
        #     loss, acc = self.criterion_descent(inputs, labels)

        # self.loss_meter.update(loss.item(), labels.size(0))
        # self.acc_meter.update(acc.item(), labels.size(0))
        # return self.weight_func() * loss

    def reset_batch(self):
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()

    def update_epoch(self, epoch):
        self.log(epoch)
        self.reset_batch()


class RandomRemove(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.randremove_alpha
        log_name = 'log_train_rand'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

    def criterion_descent(self, inputs, labels):
        inputs_rand = self.augmentor.remove_random_words_batch(inputs, self.config.randremove_num)
        outputs = self.model(**inputs_rand)
        loss = self.criterion(outputs.logits, labels)
        acc, = accuracy(outputs.logits.data, labels.data)
        return loss, acc

class MLMReplace(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.mlmreplace_alpha
        log_name = 'log_train_mlm'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

    def criterion_descent(self, inputs, labels):
        inputs_rand = self.augmentor.replace_random_words_with_mlm_batch(inputs, self.config.randremove_num)
        outputs = self.model(**inputs_rand)
        loss = self.criterion(outputs.logits, labels)
        acc, = accuracy(outputs.logits.data, labels.data)
        return loss, acc



class RandomExceptSeedRemove(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.randnotadremove_alpha
        log_name = 'log_train_rand_not_ad'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

    def criterion_descent(self, inputs, labels):
        inputs_rand = self.augmentor.remove_random_words_except_seed_words_batch(inputs, labels, self.config.randremove_num)
        outputs = self.model(**inputs_rand)
        loss = self.criterion(outputs.logits, labels)
        acc, = accuracy(outputs.logits.data, labels.data)
        return loss, acc

class RandomAndSeedRemove(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.randadremove_alpha
        log_name = 'log_train_rand_and_ad'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

    def criterion_descent(self, inputs, labels):
        inputs_ad, _ = self.augmentor.remove_seed_words_batch(inputs, labels)
        inputs_rand = self.augmentor.remove_random_words_batch(inputs_ad, self.config.randremove_num)
        outputs = self.model(**inputs_rand)
        loss = self.criterion(outputs.logits, labels)
        acc, = accuracy(outputs.logits.data, labels.data)
        return loss, acc

class RandomAndRandomSeedRemove(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.randrandadremove_alpha
        log_name = 'log_train_rand_and_rand_ad'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

    def criterion_descent(self, inputs, labels):
        inputs_ad = self.augmentor.remove_random_seed_words_batch(inputs, labels, self.config.randadremove_num)
        inputs_rand = self.augmentor.remove_random_words_except_seed_words_batch(inputs_ad, labels, self.config.randremove_num)
        # self.augmentor.check_aug(inputs, inputs_rand, labels=labels)
        outputs = self.model(**inputs_rand)
        loss = self.criterion(outputs.logits, labels)
        acc, = accuracy(outputs.logits.data, labels.data)
        return loss, acc

class ConsistencyRemove(Regularizer):
    def __init__(self, model, augmentor, time_start=None, config=None, num_classes=None):
        weight_func = lambda: config.consistremove_alpha
        criterion = None
        log_name = 'log_train_consistency'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

        from . import ce_soft, mse_one_hot
        self.consist_criterion = ce_soft(temperature=self.config.consistremove_T, num_classes=num_classes, reduction='mean')
        # self.consist_criterion = mse_one_hot(temperature=self.config.consistremove_T, num_classes=num_classes, soft_label=True, reduction='mean')

    def criterion_descent(self, inputs, labels):
        outputs = self.model(**inputs)
        inputs_aug = self.augmentor.remove_seed_words_batch(inputs, labels)
        outputs_aug = self.model(**inputs_aug)

        # TODO: Need detach here? Only need backprop in one branch? Check temporal ensemble / fixmatch implementation
        # And if detach, detach which one here? Try both.
        # soft_labels = F.softmax(outputs_aug.logits.detach() / self.config.consistremove_T, dim=1)
        # loss = self.consist_criterion(outputs.logits, soft_labels)
        soft_labels = F.softmax(outputs.logits.detach() / self.config.consistremove_T, dim=1)
        loss = self.consist_criterion(outputs_aug.logits, soft_labels)
        acc, = accuracy(outputs_aug.logits.data, labels.data)
        return loss, acc

class AugmentationParaphrase(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.augparaphrase_alpha
        log_name = 'log_train_paraphrase'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

    def criterion_descent(self, inputs, labels):
        inputs_aug = self.augmentor.paraphrase_batch(inputs, temperature=self.config.augparaphrase_temperature)
        outputs = self.model(**inputs_aug)
        loss = self.criterion(outputs.logits, labels)
        acc, = accuracy(outputs.logits.data, labels.data)
        return loss, acc

class AdversarialRemove(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.adremove_alpha
        log_name = 'log_train_ad'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

        self.set_log(log_name=log_name,
                     entry_dict={'Alpha': self.weight_func,
                                 'Train-Loss': lambda: self.loss_meter.avg,
                                 'Train-Acc': lambda: self.acc_meter.avg,
                                 'Mask-Avg': lambda: self.mask_meter.avg})

    def criterion_descent(self, inputs, labels):
        inputs_ad, offsets = self.augmentor.remove_seed_words_batch(inputs, labels)
        outputs = self.model(**inputs_ad)
        # loss = self.criterion(outputs.logits, labels)
        loss = F.cross_entropy(outputs.logits, labels, reduction='none')
        # loss *= mask
        # print(loss)
        # loss = loss[offsets > 0] # only compute loss on sentences with seed words removed
        # print(loss)
        loss = loss.mean()
        acc, = accuracy(outputs.logits.data, labels.data)
        mask = (offsets > 0).float()
        return loss, acc, mask.mean()

    def update_batch(self, inputs, labels):
        loss, acc, mask_avg = self.criterion_descent(inputs, labels)
        self.loss_meter.update(loss.item(), labels.size(0))
        self.acc_meter.update(acc.item(), labels.size(0))
        self.mask_meter.update(mask_avg.item(), labels.size(0))
        # print(f'valid removed fraction: {self.mask_meter.avg:g}', end='\r')
        return self.weight_func() * loss

    def reset_batch(self):
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.mask_meter = AverageMeter()


class AdversarialReplace(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.adreplace_alpha
        log_name = 'log_train_ad'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

        self.set_log(log_name=log_name,
                     entry_dict={'Alpha': self.weight_func,
                                 'Train-Loss': lambda: self.loss_meter.avg,
                                 'Train-Acc': lambda: self.acc_meter.avg,
                                 'Mask-Avg': lambda: self.mask_meter.avg})

    def criterion_descent(self, inputs, labels):
        inputs_ad, offsets = self.augmentor.replace_seed_words_batch(inputs, labels)
        outputs = self.model(**inputs_ad)
        # loss = self.criterion(outputs.logits, labels)
        loss = F.cross_entropy(outputs.logits, labels, reduction='none')
        # loss *= mask
        loss = loss[offsets > 0]
        loss = loss.mean()
        acc, = accuracy(outputs.logits.data, labels.data)
        mask = (offsets > 0).float()
        return loss, acc, mask.mean()

    def update_batch(self, inputs, labels):
        loss, acc, mask_avg = self.criterion_descent(inputs, labels)
        self.loss_meter.update(loss.item(), labels.size(0))
        self.acc_meter.update(acc.item(), labels.size(0))
        self.mask_meter.update(mask_avg.item(), labels.size(0))
        return self.weight_func() * loss

    def reset_batch(self):
        self.loss_meter = AverageMeter()
        self.acc_meter = AverageMeter()
        self.mask_meter = AverageMeter()


class AntiBackdoor(Regularizer):
    def __init__(self, model, criterion, augmentor, time_start=None, config=None):
        weight_func = lambda: config.antibackdoor_alpha
        log_name = 'log_train_backdoor'

        super().__init__(model, criterion, time_start=time_start, config=config, weight_func=weight_func, log_name=log_name)
        self.augmentor = augmentor

    def backdoor_attack(self, inputs, labels, labels_backdoor):
        # batch inputs
        inputs = self.augmentor.remove_seed_words_batch(inputs, labels)
        inputs = self.augmentor.insert_seed_word_batch(inputs, labels_backdoor)
        return inputs

    def criterion_descent(self, inputs, labels):
        labels_random = random_label_exclude_batch(labels, self.augmentor.classes) # use external random exclude function
        inputs_backdoor = self.backdoor_attack(inputs, labels, labels_random)
        labels_anti_backdoor = random_label_exclude_batch(labels_random, self.augmentor.classes)
        outputs = self.model(**inputs_backdoor)
        loss = self.criterion(outputs.logits, labels_anti_backdoor)
        acc, = accuracy(outputs.logits.data, labels_anti_backdoor.data)
        return loss, acc

    def criterion_ascent(self, inputs, labels):
        labels_random = random_label_exclude_batch(labels, self.augmentor.classes) # use external random exclude function
        inputs_backdoor = self.backdoor_attack(inputs, labels, labels_random)
        outputs = self.model(**inputs_backdoor)
        loss = self.criterion(outputs.logits, labels_random)
        acc, = accuracy(outputs.logits.data, labels_random.data)
        return -loss, acc



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, reduction='mean', smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.reduction = reduction
        self.smoothing_ = smoothing

    def forward(self, x, target, weights=None): # named 'weights' to allow overloading
        smoothing = weights
        if smoothing is None:
            smoothing = self.smoothing_
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise KeyError('Not supported reduction: %s' % self.reduction)

class LossFloodingCrossEntropy(nn.Module):
    def __init__(self, reduction='mean', flooding=0.1):
        super(LossFloodingCrossEntropy, self).__init__()
        self.reduction = reduction
        self.criterion = nn.CrossEntropyLoss(reduction='none') # take care of reduction in forwarding
        self.flooding_ = flooding

    def forward(self, x, target, weights=None):
        flooding = weights
        if flooding is None:
            flooding = self.flooding_
        loss = self.criterion(x, target)
        loss = (loss - flooding).abs() + flooding
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise KeyError('Not supported reduction: %s' % self.reduction)
