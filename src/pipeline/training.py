#!./env python

import torch
import torch.nn as nn
import numpy as np
import time
import os

from ..utils import AverageMeter, F1Meter, accuracy, Logger, nan_filter, load_log
from ..utils import save_checkpoint, save_model
from ..utils import print
from ..utils import shuffle_dict_of_tensors

from ..utils import random_label_exclude_batch
from ..utils import BackdoorTracker, SeedwordAdTracker, LossTracker

__all__ = ['Trainer']

class Trainer:
    def __init__(self, config):
        self.config = config

        # init
        self.time_start = time.time()
        self.last_end = 0.
        if config.resume:
            self.last_end = self.get_last_time() # min

        # read best acc
        self.best_acc = 0.
        self.best_loss = np.inf
        if config.resume:
            self.best_acc = self.get_best(phase='Acc')
            self.best_loss = self.get_best(phase='Loss')
            print('> Best Acc: %.2f Best Loss: %.2f' % (self.best_acc, self.best_loss))

        # logger
        base_names = ['Epoch', 'lr', 'Time-elapse(Min)']
        self.logger = Logger(os.path.join(config.save_dir, 'log.txt'), title='log', resume=config.resume)
        metrics = ['Train-Loss', 'Test-Loss',
                   'Train-Acc', 'Test-Acc',
                   'Macro-F1', 'Micro-F1']
        self.logger.set_names(base_names + metrics)

    def evaluate(self, model, loader, criterion):
        model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        f1_meter = F1Meter()
        for inputs, labels, _ in loader:
            with torch.no_grad():
                outputs = model(**inputs)
                loss = criterion(outputs.logits, labels)

            acc, = accuracy(outputs.logits.data, labels.data)
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc.item(), labels.size(0))
            f1_meter.update(outputs.logits.max(1)[1].data.cpu().numpy(), labels.data.cpu().numpy())
        return loss_meter.avg, acc_meter.avg, f1_meter.macro_f1, f1_meter.micro_f1

    def train(self, model, loader, criterion, optimizer, scheduler=None):
        model.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        optimizer.zero_grad()
        for batch_idx, (inputs, labels, _) in enumerate(loader):
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            std_loss = loss.item()

            if self.regularizers:
                loss *= sum([1 - regularizer.weight_func() for regularizer in self.regularizers])
            reg_losses = []
            for regularizer in self.regularizers:
                # loss += regularizer.update_batch(inputs, labels)
                reg_loss = regularizer.update_batch(inputs, labels)
                loss += reg_loss
                reg_losses.append(reg_loss.item())

            (loss / self.config.update_freq).backward()
            if (batch_idx + 1) % self.config.update_freq == 0:
                # - this gradient accumulation may be problematic when the dataset is small,
                # .  as the last big batch will be skipped if # batches is not divisible to # update freq
                # - should clipping only once when backprop, otherwise previous gradients are clipped multiple times
                #    see https://github.com/huggingface/transformers/issues/986
                if hasattr(self.config, 'gradient_clipping') and self.config.gradient_clipping:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

            if np.isnan(loss.item()):
                # print(batch_idx, inputs, labels, outputs.logits)
                print(std_loss, reg_losses, loss.item())
                # raise RuntimeError('NaN loss encountered')

            acc, = accuracy(outputs.logits.data, labels.data)
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc.item(), labels.size(0))

        return loss_meter.avg, acc_meter.avg


    def __call__(self, model, loaders, criterion, optimizer, scheduler=None):

        # tracker
        trackers = []
        if hasattr(self.config, 'backdoorTrack') and self.config.backdoorTrack:
            trackers.append(BackdoorTracker(model, criterion, loaders.augmentor, loaders.testloader, time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'seedwordAdTrack') and self.config.seedwordAdTrack:
            trackers.append(SeedwordAdTracker(model, criterion, loaders.augmentor, loaders.testloader, time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'lossTrack') and self.config.lossTrack:
            ## evaluate training set again if wish to record statistics in training (e.g., train loss)
            trackers.append(LossTracker(model, loaders.trainloader, trainsize=loaders.train_size_orig, time_start=self.time_start, config=self.config))

        # regularizer
        self.regularizers = []
        if hasattr(self.config, 'antibackdoor') and self.config.antibackdoor:
            from ..utils import AntiBackdoor
            self.regularizers.append(AntiBackdoor(model, criterion, loaders.augmentor, time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'adremove') and self.config.adremove:
            from ..utils import AdversarialRemove
            self.regularizers.append(AdversarialRemove(model, criterion, loaders.augmentor,
                                                       time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'adreplace') and self.config.adreplace:
            from ..utils import AdversarialReplace
            self.regularizers.append(AdversarialReplace(model, criterion, loaders.augmentor,
                                                        time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'randremove') and self.config.randremove:
            print(' ------------------------ Using RandomRemove ------------------------')
            from ..utils import RandomRemove
            self.regularizers.append(RandomRemove(model, criterion, loaders.augmentor, time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'randadremove') and self.config.randadremove:
            from ..utils import RandomAndSeedRemove
            self.regularizers.append(RandomAndSeedRemove(model, criterion, loaders.augmentor, time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'randnotadremove') and self.config.randnotadremove:
            from ..utils import RandomExceptSeedRemove
            self.regularizers.append(RandomExceptSeedRemove(model, criterion, loaders.augmentor, time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'randrandadremove') and self.config.randrandadremove:
            from ..utils import RandomAndRandomSeedRemove
            self.regularizers.append(RandomAndRandomSeedRemove(model, criterion, loaders.augmentor, time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'mlmreplace') and self.config.mlmreplace:
            from ..utils import MLMReplace
            self.regularizers.append(MLMReplace(model, criterion, loaders.augmentor, time_start=self.time_start, config=self.config))
        if hasattr(self.config, 'consistremove') and self.config.consistremove:
            from ..utils import ConsistencyRemove
            self.regularizers.append(ConsistencyRemove(model, loaders.augmentor, time_start=self.time_start, config=self.config, num_classes=loaders.num_classes))
        if hasattr(self.config, 'augparaphrase') and self.config.augparaphrase:
            from ..utils import AugmentationParaphrase
            self.regularizers.append(AugmentationParaphrase(model, criterion, loaders.augmentor, time_start=self.time_start, config=self.config))
 
        # start training
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train(model, loaders.trainloader, criterion, optimizer, scheduler)
            test_loss, test_acc, macro_f1, micro_f1 = self.evaluate(model, loaders.testloader, criterion)

            self.__update_best(epoch, test_acc, test_loss, model)

            # regularizer
            for regularizer in self.regularizers:
                regularizer.update_epoch(epoch)

            # log
            time_elapse = (time.time() - self.time_start)/60 + self.last_end
            logs = [epoch, self.__get_lr(optimizer), time_elapse]
            logs += [train_loss, test_loss, train_acc, test_acc, macro_f1, micro_f1]
            self.logger.append(logs)

            for tracker in trackers:
                tracker.update_epoch(epoch)

            # save checkpoint in case break
            save_checkpoint(epoch, model, optimizer, scheduler, config=self.config)

            if hasattr(self.config, 'save_model_interval') and self.config.save_model_interval:
                if (epoch + 1) % self.config.save_model_interval == 0:
                    save_model(model, basename=f'model-{epoch+1}', config=self.config)

            # save pseudo-labels
            if hasattr(self.config, 'save_pseudo_label_epoch') and self.config.save_pseudo_label_epoch:
                save_model(model, config=self.config)
                print('\n=====> Save pseudo labels..')
                from script.get_pseudo_label import pseudo_label
                import os
                print('\n> --------------- Start pseudo labeling using current model ----------------')
                pseudo_label(self.config,
                             dataset=self.config.dataset,
                             path_unlabeled_idx=self.config.save_pseudo_label_unlabeled_idx_path,
                             model_path=os.path.join('/home/chengyu/bert_classification', self.config.checkpoint_dir, self.config.checkpoint),
                             gpu_id=self.config.gpu_id, epoch=epoch)

        # save last model
        save_model(model, config=self.config)

        for tracker in trackers:
            tracker.close()

    def __update_best(self, epoch, acc, loss, model):
        if acc > self.best_acc:
            print('> Best acc got at epoch %i. Best: %.2f Current: %.2f' % (epoch, acc, self.best_acc))
            self.best_acc = acc
            # save_model(model, basename='best_model', config=self.config)

        if loss < self.best_loss:
            print('> Best loss got at epoch %i. Best: %.2f Current: %.2f' % (epoch, loss, self.best_loss))
            self.best_loss = loss
            # save_model(model, basename='best_model_loss', config=self.config)

    def __get_lr(self, optimizer):
        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]

    def get_last_time(self):
        return load_log(os.path.join(self.config.save_dir, 'log.txt'))['Time-elapse(Min)'][-1]

    def get_best(self, phase='Acc'):
        stats = load_log(os.path.join(self.config.save_dir, 'log.txt'), window=1)
        if phase == 'Acc':
            extrema = np.max
        elif phase == 'Loss':
            extrema = np.min
        else:
            raise KeyError(phase)
        return extrema(nan_filter(stats['Test-%s' % phase]))

        raise KeyError(option)

    def close(self):
        self.logger.close()
