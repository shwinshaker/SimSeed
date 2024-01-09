#!./env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import Logger, AverageMeter, accuracy
from ..utils import ExampleTracker, AdTracker, AvgTracker
from . import attack, scale_step
from .loss import trades_loss, llr_loss, mart_loss, bce_loss, fat_loss, gairat_loss
from .loss import kd_loss

from ..utils import mse_one_hot

import time
import copy

__all__ = ['AdTrainer']

class AdTrainer:
    def __init__(self, loaders, net, optimizer, criterion=None, config=None, time_start=None, swaer=None):
        self.loaders = loaders
        self.net = net
        self.optimizer = optimizer # only used for getting current learning rate
        self.criterion = criterion
        self.config = config
        self.device = self.config.device
        self.time_start = time_start

        if hasattr(self.config, 'sa') and self.config.sa == 'swa':
            assert(swaer is not None)
            self.swaer = swaer

        if hasattr(self.config, 'bootstrap') and self.config.bootstrap == 'swa':
            assert(swaer is not None)
            self.swaer = swaer

        # target
        self.target = None
        if config.target is not None:
            self.target = loaders.class_to_idx[config.target]

        # scale epsilon (each channel is different because different range)
        ## save unscaled eps for auto attack
        config.eps_ = config.eps
        config.pgd_alpha_ = config.pgd_alpha
        config.eps = scale_step(config.eps, config.dataset, device=config.device)
        config.pgd_alpha = scale_step(config.pgd_alpha, config.dataset, device=config.device)
        print('scaled eps [train]:', config.eps, config.pgd_alpha)

        # external model
        if config.ext_model:
            self.net_ext = copy.deepcopy(self.net)
            state_dict = torch.load(config.ext_model, map_location=config.device)
            self.net_ext.load_state_dict(state_dict)
            self.net_ext.eval()

        # sanity check and setup loss function
        self.epoch = config.epoch_start
        self.__ad_setup()

    def update(self, epoch, i):
        # make some logs

        if self.extra_metrics:
            self.extraLog.step(epoch, i)

        if self.config.ext_model:
            self.extmodelLog.step(epoch, i)

        if self.config.exTrack:
            self.exLog.step(epoch)

        if self.config.adTrack:
            self.adLog.step(epoch)

        ## epoch is current epoch + 1
        self.epoch = epoch + 1

    def reset(self, epoch):
        assert(epoch == self.epoch - 1), 'reset is not called after update!'

        ## reset some logger
        if self.extra_metrics:
            self.extraLog.reset()

        if self.config.ext_model:
            self.extmodelLog.reset()

        if self.config.exTrack:
            self.exLog.reset()

        if self.config.adTrack:
            self.adLog.reset()

    def close(self):
        if self.extra_metrics:
            self.extraLog.close()

        if self.config.ext_model:
            self.extmodelLog.close()

        if self.config.adTrack:
            self.adLog.close()

    def _loss(self, inputs, labels, weights, epoch=None):
        # template
        pass

    def __bootstrap_setup(self):
        if self.config.bootstrap == 'average':
            self.bs_count = 0 # add steps count for average in a sequential manner

            # initialize soft labels
            if self.config.dataset in ['mnist', 'cifar10', 'cifar100']:
                __trainsize = 50000
            elif self.config.dataset in ['tiny-imagenet']:
                __trainsize = 100000
            else:
                raise KeyError(self.config.dataset)
            self.probas_average = torch.zeros(__trainsize, len(self.loaders.classes)).to(self.device)

        self.bs_get_coeff = lambda epoch: self.config.bootstrap_coeff
        if hasattr(self.config, 'bootstrap_annealing') and self.config.bootstrap_annealing == 'exp':
            def get_coeff_exp(epoch,
                              epoch_start=self.config.bootstrap_start_at,
                              c0=self.config.bootstrap_coeff,
                              cmax=self.config.bootstrap_coeff_max,
                              slope=self.config.bootstrap_annealing_slope):
                if epoch < epoch_start:
                    raise RuntimeError('Bootstrap should not have been called at epoch %i' % epoch)
                return cmax - (cmax - c0)**((epoch - epoch_start + 1) * slope)
            self.bs_get_coeff = get_coeff_exp

    def __ad_setup(self):

        self.extra_metrics = []

        if not self.config.adversary:
            self._loss = self._clean_loss

            if hasattr(self.config, 'mixup_aug_target') and self.config.mixup_aug_target == 'true':
                self._loss = self._clean_loss_dual_targets

            if hasattr(self.config, 'kd') and self.config.kd:
                self._loss = self._clean_loss_kd

                assert(self.config.kd_coeff_st > 0)
                if isinstance(self.config.kd_teacher_st, str):
                    self.teacher_st = copy.deepcopy(self.net)
                    self.teacher_st.load_state_dict(torch.load(self.config.kd_teacher_st, map_location=self.config.device))
                    self.teacher_st.eval()
                elif isinstance(self.config.kd_teacher_st, list):
                    self.teacher_st = []
                    for path in self.config.kd_teacher_st:
                        net_teacher = copy.deepcopy(self.net)
                        net_teacher.load_state_dict(torch.load(path, map_location=self.config.device))
                        net_teacher.eval()
                        self.teacher_st.append(net_teacher)
                else:
                    raise KeyError('Missing kd teacher.')

            if hasattr(self.config, 'sa') and self.config.sa:
                self._loss = self._clean_loss_sa

                # initialize soft labels
                if self.config.dataset in ['mnist', 'cifar10', 'cifar100']:
                    __trainsize = 50000
                elif self.config.dataset in ['tiny-imagenet']:
                    __trainsize = 100000
                else:
                    raise KeyError(self.config.dataset)
                self.sa_soft_labels = torch.zeros(__trainsize, len(self.loaders.classes)).to(self.device)
                for inputs, labels, weights in self.loaders.trainloader:
                    indices = weights['index']
                    self.sa_soft_labels[indices] = F.one_hot(labels.to(self.device), num_classes=len(self.loaders.classes)).float()
             
            if hasattr(self.config, 'bootstrap') and self.config.bootstrap:
                self._loss = self._clean_loss_bootstrap
                self.__bootstrap_setup()

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss', 'Train-Acc']
            self.extraLog = AvgTracker('log_extra',
                                       self.optimizer,
                                       metrics=self.extra_metrics,
                                       time_start=self.time_start,
                                       config=self.config)

            # log for sample robust correctness
            if self.config.exTrack:
                self.exLog = ExampleTracker(self.loaders, options=self.config.exTrackOptions, resume=self.config.resume, config=self.config)

            if self.config.adTrack:
                self.adLog = AdTracker(self.loaders, self.net, self.config, options=self.config.adTrackOptions, resume=self.config.resume)

            return


        if self.config.adversary in ['gaussian', 'fgsm', 'pgd', 'fgsm_manifold', 'pgd_manifold', 'aa']:
            self._loss = self._ad_loss
            if hasattr(self.config, 'kd') and self.config.kd:
                self._loss = self._ad_loss_kd

                if self.config.kd_teacher_st:
                    assert(self.config.kd_coeff_st > 0)
                    self.teacher_st = copy.deepcopy(self.net)
                    self.teacher_st.load_state_dict(torch.load(self.config.kd_teacher_st, map_location=self.config.device))
                    self.teacher_st.eval()
                else:
                    assert(self.config.kd_coeff_st == 0)

                if self.config.kd_teacher_rb:
                    assert(self.config.kd_coeff_rb > 0)
                    if isinstance(self.config.kd_teacher_rb, str):
                        self.teacher_rb = copy.deepcopy(self.net)
                        self.teacher_rb.load_state_dict(torch.load(self.config.kd_teacher_rb, map_location=self.config.device))
                        self.teacher_rb.eval()
                    elif isinstance(self.config.kd_teacher_rb, list):
                        self.teacher_rb = []
                        for path in self.config.kd_teacher_rb:
                            net_teacher = copy.deepcopy(self.net)
                            net_teacher.load_state_dict(torch.load(path, map_location=self.config.device))
                            net_teacher.eval()
                            self.teacher_rb.append(net_teacher)
                else:
                    assert(self.config.kd_coeff_rb == 0)

            if hasattr(self.config, 'sa') and self.config.sa:
                self._loss = self._ad_loss_sa

                # initialize soft labels
                if self.config.dataset in ['mnist', 'cifar10', 'cifar100']:
                    __trainsize = 50000
                elif self.config.dataset in ['tiny-imagenet']:
                    __trainsize = 100000
                else:
                    raise KeyError(self.config.dataset)
                self.sa_soft_labels = torch.zeros(__trainsize, len(self.loaders.classes)).to(self.device)
                for inputs, labels, weights in self.loaders.trainloader:
                    indices = weights['index']
                    self.sa_soft_labels[indices] = F.one_hot(labels.to(self.device), num_classes=len(self.loaders.classes)).float()

            if hasattr(self.config, 'bootstrap') and self.config.bootstrap:
                self._loss = self._ad_loss_bootstrap
                self.__bootstrap_setup()

            # log for 'false' training loss and acc, aligned with previous work
            self.extra_metrics = ['Train-Loss', 'Train-Acc', 'Train-Loss-Ad', 'Train-Acc-Ad']
            self.extraLog = AvgTracker('log_extra',
                                       self.optimizer,
                                       metrics=self.extra_metrics,
                                       time_start=self.time_start,
                                       config=self.config)

            # log for sample robust correctness
            if self.config.exTrack:
                self.exLog = ExampleTracker(self.loaders, options=self.config.exTrackOptions, resume=self.config.resume, config=self.config)

            # log for perturbation and local linearity
            if self.config.adTrack:
                self.adLog = AdTracker(self.loaders, self.net, self.config, options=self.config.adTrackOptions, resume=self.config.resume)

            # log for external model evaluation
            if self.config.ext_model:
                self.extmodelLog = AvgTracker('log_ext_model',
                                              self.optimizer,
                                              metrics=['Train-Loss-Ad', 'Train-Acc-Ad'],
                                              time_start=self.time_start,
                                              config=self.config)

            return

        if hasattr(self.config, 'kd') and self.config.kd:
            raise NotImplementedError('Knowledge distillation not supported for this ad method! TODO..')

        # No extra logger: 'false' training accuracy not recorded for integrated loss - needs to code within specific loss function
        # other things not supported currectly..
        if self.config.target:
            raise NotImplementedError('Targeted attack not supported! TODO..')
        # if hasattr(self.config, 'alpha_sample_path') and self.config.alpha_sample_path:
        #     # for trades, this was implemented, but try incorporate into the loss function
        #     raise NotImplementedError('Sample-wise trading not supported! TODO..')
        if hasattr(self.config, 'reg_sample_path') and self.config.reg_sample_path:
            raise NotImplementedError('Sample-wise regularization Not supported!')


        if self.config.adversary in ['trades', 'mart', 'fat', 'gairat']:
            self._loss = getattr(self, '_%s_loss' % self.config.adversary)

            self.extra_metrics = ['Train-Loss-Ad', 'Train-Acc-Ad']
            self.extraLog = AvgTracker('log_extra',
                                       self.optimizer,
                                       metrics=self.extra_metrics,
                                       time_start=self.time_start,
                                       config=self.config)

            if self.config.exTrack:
                self.exLog = ExampleTracker(self.loaders, options=self.config.exTrackOptions, resume=self.config.resume, config=self.config)

            if self.config.adTrack:
                raise NotImplementedError('Ad tracker not implemented for adversary %s' % self.config.adversary)

            return


        if self.config.adversary in ['llr']:
            self._loss = self._llr_loss

            if self.config.exTrack:
                raise NotImplementedError('Example tracking not supported! TODO..')
            if self.config.adTrack:
                raise NotImplementedError('Ad tracking not supported! TODO..')
            if self.config.ext_model:
                raise NotImplementedError('External model not supported! TODO..')
            return

        raise KeyError('Unexpected adversary %s' % self.config.adversary)


    def _clean_loss_sa(self, inputs, labels, weights, epoch=None):
        outputs = self.net(inputs)

        ## -- weighted labels
        if epoch < self.config.sa_start_at:
            ## -- burn-in
            loss = self.criterion(outputs, labels).mean()
        else:
            ## -- weighted labels
            indices = weights['index']
            if self.config.sa == 'output':
                probas = F.softmax(outputs.detach() / self.config.sa_temperature, dim=1)
            elif self.config.sa == 'swa':
                outputs_swa = self.swaer.swa_net(inputs)
                probas = F.softmax(outputs_swa.detach() / self.config.sa_temperature, dim=1)
            else:
                raise KeyError(self.config.sa)

            # update soft label
            self.sa_soft_labels[indices] *= (1. - self.config.sa_coeff)
            self.sa_soft_labels[indices] += self.config.sa_coeff * probas
            soft_labels = self.sa_soft_labels[indices]

            if hasattr(self.config, 'sa_weighted') and self.config.sa_weighted:
                sample_weights, _ = soft_labels.max(dim=1)
                sample_weights *= outputs.size(0) / sample_weights.sum()
                loss = self._soft_ce(outputs / self.config.sa_temperature, soft_labels, weights=sample_weights) * self.config.sa_temperature**2
            else:
                loss = self._soft_ce(outputs / self.config.sa_temperature, soft_labels) * self.config.sa_temperature**2

        # -------- extra logs
        if self.extra_metrics:
            prec1, = accuracy(outputs.data, labels.data)
            self.extraLog.update({'Train-Loss': loss.mean().item(),
                                  'Train-Acc': prec1.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def _clean_loss_dual_targets(self, inputs, labels, weights, epoch=None):
        outputs = self.net(inputs)

        ## -- combine two targets
        soft_labels = F.one_hot(labels, num_classes=outputs.size(1)).float() * 0.5
        soft_labels += F.one_hot(weights['targets2'].to(self.device), num_classes=outputs.size(1)).float() * 0.5

        loss = self._soft_ce(outputs, soft_labels)

        # -------- extra logs
        if self.extra_metrics:
            prec1, = accuracy(outputs.data, labels.data)
            self.extraLog.update({'Train-Loss': loss.mean().item(),
                                  'Train-Acc': prec1.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def _bootstrap_loss(self, inputs, outputs, labels, weights, epoch=None):
        def proba_to_label(probas):
            if self.config.bootstrap_type == 'soft':
                return probas
            elif self.config.bootstrap_type == 'hard':
                _, preds = probas.max(1)
                return F.one_hot(preds, num_classes=outputs.size(1)).float() 
            else:
                raise KeyError(self.config.bootstrap_type)

        if epoch < self.config.bootstrap_start_at:
            ## -- burn-in
            loss = self.criterion(outputs, labels).mean()
        else:
            ## -- bootstrapping - weighted labels
            coeff = self.bs_get_coeff(epoch)
            soft_labels = F.one_hot(labels, num_classes=outputs.size(1)).float() * (1. - coeff)
            if self.config.bootstrap == 'self':
                probas = F.softmax(outputs.detach() / self.config.bootstrap_temperature, dim=1)
                soft_labels += proba_to_label(probas) * coeff
            elif self.config.bootstrap == 'swa':
                outputs_swa = self.swaer.swa_net(inputs)
                probas = F.softmax(outputs_swa.detach() / self.config.bootstrap_temperature, dim=1)
                soft_labels += proba_to_label(probas) * coeff
            elif self.config.bootstrap == 'average':
                self.bs_count += 1
                indices = weights['index']
                # update averaged probas
                probas = F.softmax(outputs.detach() / self.config.bootstrap_temperature, dim=1)
                self.probas_average[indices] *= (self.bs_count - 1) / self.bs_count
                self.probas_average[indices] += probas / self.bs_count
                soft_labels += proba_to_label(self.probas_average[indices]) * coeff
            else:
                raise KeyError(self.config.bootstrap)
            # loss = self._soft_ce(outputs, soft_label)
            loss = self._soft_ce(outputs / self.config.bootstrap_temperature, soft_labels) * self.config.bootstrap_temperature**2

        return loss

    def _clean_loss_bootstrap(self, inputs, labels, weights, epoch=None):
        outputs = self.net(inputs)

        # -------- bootstrapping
        loss = self._bootstrap_loss(inputs, outputs, labels, weights, epoch=epoch)

        # -------- extra logs
        if self.extra_metrics:
            prec1, = accuracy(outputs.data, labels.data)
            self.extraLog.update({'Train-Loss': loss.mean().item(),
                                  'Train-Acc': prec1.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def _clean_loss_kd(self, inputs, labels, weights, epoch=None):
        outputs = self.net(inputs)
        with torch.no_grad():
            if isinstance(self.config.kd_teacher_st, str):
                outputs_st = self.teacher_st(inputs)
                probas_st = F.softmax(outputs_st / self.config.kd_temperature, dim=1)
            elif isinstance(self.config.kd_teacher_st, list):
                probas_st = torch.zeros_like(outputs)
                for teacher in self.teacher_st:
                    outputs_st = teacher(inputs)
                    probas_st += F.softmax(outputs_st / self.config.kd_temperature, dim=1)
                probas_st /= len(self.teacher_st)
            else:
                pass

        ## -- conventional kd
        loss = self.criterion(outputs, labels)
        loss_kd_st = kd_loss(outputs, probas_st, T=self.config.kd_temperature)

        loss = loss * (1. - self.config.kd_coeff_st)
        loss += loss_kd_st * self.config.kd_coeff_st

        # -------- extra logs
        if self.extra_metrics:
            prec1, = accuracy(outputs.data, labels.data)
            self.extraLog.update({'Train-Loss': loss.mean().item(),
                                  'Train-Acc': prec1.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss.mean()


    def _clean_loss(self, inputs, labels, weights, epoch=None, update=True):
        outputs = self.net(inputs)
        if 'reg' in weights:
            loss = self.criterion(outputs,
                                  labels,
                                  weights=weights['reg'].to(self.device))
        else:
            loss = self.criterion(outputs, labels)

        if not update:
            # Only functional call 
            return loss
    
        # extra logs
        if self.extra_metrics:
            prec1, = accuracy(outputs.data, labels.data)
            self.extraLog.update({'Train-Loss': loss.mean().item(),
                                  'Train-Acc': prec1.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss.mean()

    def _soft_ce(self, outputs, scores, weights=None):
        log_probas = F.log_softmax(outputs, dim=1)
        loss = -(scores * log_probas)
        loss = loss.sum(dim=1) # sum over classes
        if weights is not None:
            loss *= weights
        return loss.mean()

    def _ad_loss_sa(self, inputs, labels, weights, epoch=None):
        self.net.eval()
        ctr = nn.CrossEntropyLoss() # Don't change the criterion in adversary generation part -- maybe change it later
        get_steps = False
        if self.config.exTrack and 'count_iters' in self.config.exTrackOptions:
            get_steps = True
        get_minimum = False
        if self.config.exTrack and 'min_perturbs' in self.config.exTrackOptions:
            get_minimum = True
        inputs_ad, stats = attack(self.net, ctr, inputs, labels, weight=None,
                                  adversary=self.config.adversary,
                                  eps=self.config.eps,
                                  pgd_alpha=self.config.pgd_alpha,
                                  pgd_iter=self.config.pgd_iter,
                                  randomize=self.config.rand_init,
                                  target=self.target,
                                  get_steps=get_steps,
                                  get_minimum=get_minimum,
                                  config=self.config)
        self.net.train()
        outputs_ad = self.net(inputs_ad)

        ## -- weighted labels
        if epoch < self.config.sa_start_at:
            ## -- burn-in
            loss = self.criterion(outputs_ad, labels).mean()
        else:
            if self.config.sa == 'output':
                ## -- bootstrapping - weighted labels
                indices = weights['index']

                probas = F.softmax(outputs_ad.detach() / self.config.sa_temperature, dim=1)
                # update soft label
                self.sa_soft_labels[indices] *= (1. - self.config.sa_coeff)
                self.sa_soft_labels[indices] += self.config.sa_coeff * probas
                soft_labels = self.sa_soft_labels[indices]
            elif self.config.sa == 'swa':
                outputs_swa = self.swaer.swa_net(inputs_ad)
                probas = F.softmax(outputs_swa.detach() / self.config.sa_temperature, dim=1)
                soft_labels = F.one_hot(labels, num_classes=outputs_swa.size(1)).float() * (1. - self.config.sa_coeff)
                soft_labels += self.config.sa_coeff * probas
            else:
                raise KeyError(self.config.sa)


            if hasattr(self.config, 'sa_weighted') and self.config.sa_weighted:
                sample_weights, _ = soft_labels.max(dim=1)
                sample_weights *= outputs_ad.size(0) / sample_weights.sum()
                loss = self._soft_ce(outputs_ad / self.config.sa_temperature, soft_labels, weights=sample_weights) * self.config.sa_temperature**2
            else:
                loss = self._soft_ce(outputs_ad / self.config.sa_temperature, soft_labels) * self.config.sa_temperature**2
                    
        # -------- recording
        loss_ad = loss # for log
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.ext_model:
            with torch.no_grad():
                outputs_ad_ext = self.net_ext(inputs_ad)
                loss_ad = self.criterion(outputs_ad_ext, labels)
            prec1_ad, = accuracy(outputs_ad_ext.data, labels.data)
            self.extmodelLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                     'Train-Acc-Ad': prec1_ad.item()},
                                    inputs.size(0))

        if self.config.exTrack:
            count_iter = None
            if 'count_iters' in self.config.exTrackOptions:
                count_iter = stats['step_counts']
            min_perturb = None
            if 'min_perturbs' in self.config.exTrackOptions:
                min_perturb = stats['min_perturb']
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch, count_iter=count_iter, min_perturb=min_perturb)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), inputs_ad=inputs_ad, epoch=self.epoch)

        return loss

    def _ad_loss_bootstrap(self, inputs, labels, weights, epoch=None):
        self.net.eval()
        ctr = nn.CrossEntropyLoss() # Don't change the criterion in adversary generation part -- maybe change it later
        get_steps = False
        if self.config.exTrack and 'count_iters' in self.config.exTrackOptions:
            get_steps = True
        get_minimum = False
        if self.config.exTrack and 'min_perturbs' in self.config.exTrackOptions:
            get_minimum = True
        inputs_ad, stats = attack(self.net, ctr, inputs, labels, weight=None,
                                  adversary=self.config.adversary,
                                  eps=self.config.eps,
                                  pgd_alpha=self.config.pgd_alpha,
                                  pgd_iter=self.config.pgd_iter,
                                  randomize=self.config.rand_init,
                                  target=self.target,
                                  get_steps=get_steps,
                                  get_minimum=get_minimum,
                                  config=self.config)
        self.net.train()
        outputs_ad = self.net(inputs_ad)

        ## ------ bootstrap loss
        loss = self._bootstrap_loss(inputs_ad, outputs_ad, labels, weights, epoch=epoch)

        # -------- recording
        loss_ad = loss
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.ext_model:
            with torch.no_grad():
                outputs_ad_ext = self.net_ext(inputs_ad)
                loss_ad = self.criterion(outputs_ad_ext, labels)
            prec1_ad, = accuracy(outputs_ad_ext.data, labels.data)
            self.extmodelLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                     'Train-Acc-Ad': prec1_ad.item()},
                                    inputs.size(0))

        if self.config.exTrack:
            count_iter = None
            if 'count_iters' in self.config.exTrackOptions:
                count_iter = stats['step_counts']
            min_perturb = None
            if 'min_perturbs' in self.config.exTrackOptions:
                min_perturb = stats['min_perturb']
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch, count_iter=count_iter, min_perturb=min_perturb)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), inputs_ad=inputs_ad, epoch=self.epoch)

        return loss

    def _ad_loss_kd(self, inputs, labels, weights, epoch=None):

        self.net.eval()
        ctr = nn.CrossEntropyLoss() # Don't change the criterion in adversary generation part -- maybe change it later
        if hasattr(self.config, 'loss'):
            if self.config.loss == 'ce':
                pass
            elif self.config.loss == 'mse':
                ctr = mse_one_hot() # nn.MSELoss()
            else:
                raise NotImplementedError()
        get_steps = False
        if self.config.exTrack and 'count_iters' in self.config.exTrackOptions:
            get_steps = True
        get_minimum = False
        if self.config.exTrack and 'min_perturbs' in self.config.exTrackOptions:
            get_minimum = True
        inputs_ad, stats = attack(self.net, ctr, inputs, labels, weight=None,
                                  adversary=self.config.adversary,
                                  eps=self.config.eps,
                                  pgd_alpha=self.config.pgd_alpha,
                                  pgd_iter=self.config.pgd_iter,
                                  randomize=self.config.rand_init,
                                  target=self.target,
                                  get_steps=get_steps,
                                  get_minimum=get_minimum,
                                  config=self.config)
        self.net.train()
        outputs_ad = self.net(inputs_ad)
        with torch.no_grad():
            if self.config.kd_teacher_st:
                outputs_st = self.teacher_st(inputs_ad)
                probas_st = F.softmax(outputs_st / self.config.kd_temperature, dim=1)
            if isinstance(self.config.kd_teacher_rb, str):
                outputs_rb = self.teacher_rb(inputs_ad)
                probas_rb = F.softmax(outputs_rb / self.config.kd_temperature, dim=1)
            elif isinstance(self.config.kd_teacher_rb, list):
                probas_rb = torch.zeros_like(outputs_ad)
                for teacher in self.teacher_rb:
                    outputs_rb = teacher(inputs_ad)
                    probas_rb += F.softmax(outputs_rb / self.config.kd_temperature, dim=1)
                probas_rb /= len(self.teacher_rb)
            else:
                # no rb teacher
                pass

        ## -- weighted probas
        # if self.config.kd_teacher_st:
        #     if hasattr(self.config, 'kd_temperature_st'):
        #         probas_st = F.softmax(outputs_st / self.config.kd_temperature_st, dim=1)
        #     else:
        #         probas_st = F.softmax(outputs_st / self.config.kd_temperature, dim=1)
        # if self.config.kd_teacher_rb:
        #     probas_rb = F.softmax(outputs_rb / self.config.kd_temperature, dim=1)
        # probas = F.one_hot(labels, num_classes=outputs_ad.size(1)).float() * (1. - self.config.kd_coeff_st - self.config.kd_coeff_rb)
        # if self.config.kd_teacher_st:
        #     probas += probas_st * self.config.kd_coeff_st
        # if self.config.kd_teacher_rb:
        #     probas += probas_rb * self.config.kd_coeff_rb
        # # loss = self._soft_ce(outputs_ad, probas)
        # loss = self._soft_ce(outputs_ad / self.config.kd_temperature, probas) * self.config.kd_temperature**2

        # loss_ad = loss # for log

        ## -- conventional kd
        loss_ad = self.criterion(outputs_ad, labels)
        if self.config.kd_teacher_st:
            loss_kd_st = kd_loss(outputs_ad, probas_st, T=self.config.kd_temperature)
        if self.config.kd_teacher_rb:
            loss_kd_rb = kd_loss(outputs_ad, probas_rb, T=self.config.kd_temperature)

        loss = loss_ad * (1. - self.config.kd_coeff_st - self.config.kd_coeff_rb)
        if self.config.kd_teacher_st:
            loss += loss_kd_st * self.config.kd_coeff_st
        if self.config.kd_teacher_rb:
            loss += loss_kd_rb * self.config.kd_coeff_rb
        loss = loss.mean()


        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.ext_model:
            with torch.no_grad():
                outputs_ad_ext = self.net_ext(inputs_ad)
                loss_ad = self.criterion(outputs_ad_ext, labels)
            prec1_ad, = accuracy(outputs_ad_ext.data, labels.data)
            self.extmodelLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                     'Train-Acc-Ad': prec1_ad.item()},
                                    inputs.size(0))

        if self.config.exTrack:
            count_iter = None
            if 'count_iters' in self.config.exTrackOptions:
                count_iter = stats['step_counts']
            min_perturb = None
            if 'min_perturbs' in self.config.exTrackOptions:
                min_perturb = stats['min_perturb']
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch, count_iter=count_iter, min_perturb=min_perturb)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), inputs_ad=inputs_ad, epoch=self.epoch)

        return loss

    def __get_pgd_alpha(self, pgd_iters, acc_radius=20.):
        pgd_alphas = acc_radius / pgd_iters.float()
        pgd_alphas = pgd_alphas.view(pgd_alphas.size(0), 1, 1, 1).to(self.device)
        pgd_alphas = scale_step(pgd_alphas, dataset=self.config.dataset, device=self.device)
        return pgd_alphas

    def _ad_loss(self, inputs, labels, weights, epoch=None):

        # -------- clean loss
        loss = 0.
        # if pure ad loss and sample-wise alpha not enabled, don't have to do this part
        if self.config.alpha < 1.0 or 'alpha' in weights:
            # do we need to enable model training for clean training when doing adversarial training? Test it.
            # self.net.eval() 
            loss = self._clean_loss(inputs, labels, weights, update=False)

        # ------- ad loss
        eps_weight = None
        if 'weps' in weights:
            eps_weight = weights['weps']

        pgd_alpha = self.config.pgd_alpha
        pgd_iter = self.config.pgd_iter
        adversary = self.config.adversary
        if 'num_iter' in weights:
            assert(self.config.adversary == 'pgd'), 'adversary %s not supported in instance-wise iteration mode!'
            adversary = 'pgd_custom'
            pgd_iter = weights['num_iter']
            pgd_alpha = self.__get_pgd_alpha(pgd_iter)

        self.net.eval()
        ctr = nn.CrossEntropyLoss() # Don't change the criterion in adversary generation part -- maybe change it later
        if hasattr(self.config, 'loss'):
            if self.config.loss == 'ce':
                pass
            elif self.config.loss == 'mse':
                ctr = mse_one_hot() # nn.MSELoss()
            else:
                raise NotImplementedError()
        get_steps = False
        if self.config.exTrack and 'count_iters' in self.config.exTrackOptions:
            get_steps = True
        get_minimum = False
        if self.config.exTrack and 'min_perturbs' in self.config.exTrackOptions:
            get_minimum = True
        inputs_ad, stats = attack(self.net, ctr, inputs, labels, weight=eps_weight,
                                  adversary=adversary,
                                  eps=self.config.eps,
                                  pgd_alpha=pgd_alpha,
                                  pgd_iter=pgd_iter,
                                  randomize=self.config.rand_init,
                                  target=self.target,
                                  get_steps=get_steps,
                                  get_minimum=get_minimum,
                                  config=self.config)
        self.net.train()

        outputs_ad = self.net(inputs_ad)

        if 'reg' in weights:
            loss_ad = self.criterion(outputs_ad,
                                     labels,
                                     weights=weights['reg'].to(self.device))
        else:
            loss_ad = self.criterion(outputs_ad, labels)
            # print('bce!')
            # loss_ad = bce_loss(outputs_ad, labels, reduction='none')

        # -------- combine two loss
        if 'alpha' in weights:
            # sample-wise weighting
            assert(loss.size(0) == inputs.size(0)), (loss.size(0), inputs.size(0))
            alpha = weights['alpha'].to(self.device)
            assert(loss.size() == loss_ad.size() == alpha.size()), (loss.size(), loss_ad.size(), alpha.size())
        else:
            alpha = self.config.alpha

        if 'lambda' in weights:
            lmbd = weights['lambda'].to(self.device)
        else:
            lmbd = torch.ones(inputs.size(0)).to(self.device)

        assert(loss_ad.size() == lmbd.size()), (loss_ad.size(), lmbd.size())
        loss *= (1 - alpha)
        loss += alpha * loss_ad 
        loss *= lmbd / lmbd.sum() # per-sample weight
        loss = loss.sum()
        # print(loss)

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.ext_model:
            with torch.no_grad():
                outputs_ad_ext = self.net_ext(inputs_ad)
                loss_ad = self.criterion(outputs_ad_ext, labels)
            prec1_ad, = accuracy(outputs_ad_ext.data, labels.data)
            self.extmodelLog.update({'Train-Loss-Ad': loss_ad.mean().item(),
                                     'Train-Acc-Ad': prec1_ad.item()},
                                    inputs.size(0))

        if self.config.exTrack:
            count_iter = None
            if 'count_iters' in self.config.exTrackOptions:
                count_iter = stats['step_counts']
            min_perturb = None
            if 'min_perturbs' in self.config.exTrackOptions:
                min_perturb = stats['min_perturb']
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch, count_iter=count_iter, min_perturb=min_perturb)

        if self.config.adTrack:
            self.adLog.update(inputs, labels, weights['index'].to(self.device), inputs_ad=inputs_ad, epoch=self.epoch)

        return loss

    def __get_pgd_ad(self, inputs, labels):
        # don't affect the training stats, do this in eval mode
        self.net.eval()
        # Don't change the criterion in adversary generation part -- maybe change it later
        ctr = nn.CrossEntropyLoss() 
        if hasattr(self.config, 'loss'):
            if self.config.loss == 'ce':
                pass
            elif self.config.loss == 'mse':
                # ctr = nn.MSELoss()
                ctr = mse_one_hot()
            else:
                raise NotImplementedError()

        # ensusre a consistent attack to evaluate training accuracy
        eps =  scale_step(8, self.config.dataset, device=self.config.device)
        pgd_alpha = scale_step(2, self.config.dataset, device=self.config.device)
        pgd_iter = 10
        inputs_ad, _ = attack(self.net, ctr, inputs, labels, weight=None,
                              adversary='pgd',
                              eps=eps,
                              pgd_alpha=pgd_alpha,
                              pgd_iter=pgd_iter,
                              randomize=self.config.rand_init,
                              target=self.target,
                              config=self.config)
        outputs_ad = self.net(inputs_ad)
        self.net.train()
        return outputs_ad

    def _trades_loss(self, inputs, labels, weights, epoch=None):
        # note: The official implementation use CE + KL * beta - amounts to alpha~= 0.85
        #       Previously we use (1-alpha) * CE + alpha * KL
        # integrate clean loss in trades loss
        # sample-weighting in trades loss - later
        loss, outputs_ad = trades_loss(self.net, inputs, labels, weights,
                                      eps=self.config.eps,
                                      alpha=self.config.pgd_alpha,
                                      num_iter=self.config.pgd_iter,
                                      norm='linf',
                                      rand_init=self.config.rand_init,
                                      config=self.config)

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            # Generate ad examples using PGD, otherwise not fair!
            outputs_ad = self.__get_pgd_ad(inputs, labels)
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def _mart_loss(self, inputs, labels, weights, epoch=None):
        loss, outputs_ad = mart_loss(self.net, inputs, labels, weights,
                                     eps=self.config.eps,
                                     alpha=self.config.pgd_alpha,
                                     num_iter=self.config.pgd_iter,
                                     norm='linf',
                                     rand_init=self.config.rand_init,
                                     config=self.config)

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch)

        return loss

    def __get_tau(self, epoch):
        tau = self.config.fat_taus[0]
        if epoch > self.config.fat_milestones[0]:
            tau = self.config.fat_taus[1]
        if epoch > self.config.fat_milestones[1]:
            tau = self.config.fat_taus[2]
        return tau

    def _fat_loss(self, inputs, labels, weights, epoch=None):
        tau = self.__get_tau(epoch)
        loss, outputs_ad, count_iter = fat_loss(self.net, inputs, labels, weights,
                                                eps=self.config.eps,
                                                alpha=self.config.pgd_alpha,
                                                num_iter=self.config.pgd_iter,
                                                tau=tau,
                                                norm='linf',
                                                rand_init=self.config.rand_init,
                                                config=self.config)

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            # Generate ad examples using PGD, otherwise not fair!
            outputs_ad = self.__get_pgd_ad(inputs, labels)
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch, count_iter=count_iter)

        return loss

    def _gairat_loss(self, inputs, labels, weights, epoch=None):
        tau = self.__get_tau(epoch)
        loss, outputs_ad, count_iter = gairat_loss(self.net, inputs, labels, weights,
                                                   eps=self.config.eps,
                                                   alpha=self.config.pgd_alpha,
                                                   num_iter=self.config.pgd_iter,
                                                   tau=tau,
                                                   norm='linf',
                                                   rand_init=self.config.rand_init,
                                                   config=self.config)

        # -------- recording
        if self.extra_metrics:
            prec1_ad, = accuracy(outputs_ad.data, labels.data)
            self.extraLog.update({'Train-Loss-Ad': loss.mean().item(),
                                  'Train-Acc-Ad': prec1_ad.item()},
                                 inputs.size(0))

        if self.config.exTrack:
            self.exLog.update(outputs_ad, labels, weights['index'].to(self.device), epoch=self.epoch, count_iter=count_iter)

        return loss

    def _llr_loss(self, inputs, labels, weights, epoch=None):
        loss = llr_loss(self.net, inputs, labels,
                        eps=self.config.eps,
                        alpha=self.config.pgd_alpha,
                        num_iter=self.config.pgd_iter,
                        norm='linf',
                        rand_init=self.config.rand_init,
                        config=self.config)
        return loss

    def __get_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]

