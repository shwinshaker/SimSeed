#!./env python

from . import ravel_parameters, Logger, o_dedup, Hooker, OutputHooker, AverageMeter, F1Meter, accuracy, load_log
from . import get_distance_matrix, get_equivalence_matrix
from . import mse_one_hot
from . import shuffle_dict_of_tensors, random_label_exclude_batch
import torch
import torch.nn as nn
import numpy as np

import os
import time
from copy import deepcopy
import re
import warnings

__all__ = ['Tracker',
           'BackdoorTracker',
           'SeedwordAdTracker',
           'LossTracker']
        #    'ParameterTracker',
        #    'LipTracker',
        #    'LrTracker',
        #    'FeatureTracker',
        #    'ResidualTracker',
        #    'RobustTracker',
        #    'ManifoldTracker',
        #    'ExampleTracker',
        #    'AvgTracker',
        #    'AdTracker']

def get_layers(d):
    return o_dedup([k.rstrip('.weight').rstrip('.bias') for k in d.keys()])


class Tracker:
    def __init__(self, model, criterion, time_start=None, config=None):
        # init
        self.model = model
        self.criterion = criterion
        self.config = config

        self.time_start = time_start
        if time_start is None:
            self.time_start = time.time()
        self.last_end = 0.
        if config.resume:
            self.last_end = self.get_last_time() # min

    def set_log(self, log_name='log_regularization', entry_dict={}):
        # logger
        self.logger = Logger(os.path.join(self.config.save_dir, f'{log_name}.txt'), title=log_name, resume=self.config.resume)
        base_names = ['Epoch', 'Time-elapse(Min)']
        entry_names = list(entry_dict.keys())
        self.logger.set_names(base_names + entry_names)

        # logging function
        def _log(epoch):
            time_elapse = (time.time() - self.time_start)/60 + self.last_end
            logs = [epoch, time_elapse]
            logs += [entry_dict[name]() for name in entry_names]
            self.logger.append(logs)

        self.log = _log

    def get_last_time(self):
        return load_log(os.path.join(self.config.save_dir, 'log.txt'))['Time-elapse(Min)'][-1]

    def close(self):
        self.logger.close()


class LossTracker(Tracker):
    def __init__(self, model, loader, trainsize, time_start=None, config=None):
        super().__init__(model, None, time_start=time_start, config=config)
        self.set_log(log_name='log_train_loss',
                     entry_dict={'Train-Loss': lambda: self.train_loss,
                                 'Train-Acc': lambda: self.train_acc,
                                 'n_examples': lambda: self.n_examples})
        self.loader = loader
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.norm_loss_avg = np.zeros(trainsize)

    def update_epoch(self, epoch):
        self.train_loss, self.train_acc, self.n_examples = self.evaluate(self.loader, epoch)
        self.log(epoch)
        with open('./norm_loss_avg.npy', 'wb') as f:
            np.save(f, self.norm_loss_avg, allow_pickle=True)

    def evaluate(self, loader, epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        n_examples = 0
        for inputs, labels, weights in loader:
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = self.criterion(outputs.logits, labels)
                acc, = accuracy(outputs.logits.data, labels.data)
            self.norm_loss_avg[weights['index'].cpu().numpy()] *= epoch / (epoch + 1)
            self.norm_loss_avg[weights['index'].cpu().numpy()] += (loss - loss.mean()).detach().cpu().numpy() * 1.0 / (epoch + 1)
            loss_meter.update(loss.mean().item(), labels.size(0))
            acc_meter.update(acc.item(), labels.size(0))
            n_examples += labels.size(0)
        return loss_meter.avg, acc_meter.avg, n_examples


class BackdoorTracker(Tracker):
    def __init__(self, model, criterion, augmentor, loader, time_start=None, config=None):
        super().__init__(model, criterion, time_start=time_start, config=config)

        self.set_log(log_name='log_test_backdoor',
                     entry_dict={'Test-Loss-BD': lambda: self.test_loss_bd,
                                 'Test-Acc-BD': lambda: self.test_acc_bd,
                                 'Macro-F1-BD': lambda: self.macro_f1_bd,
                                 'Micro-F1-BD': lambda: self.micro_f1_bd})

        self.augmentor = augmentor
        self.loader = loader

    def update_epoch(self, epoch):
        self.test_loss_bd, self.test_acc_bd, self.macro_f1_bd, self.micro_f1_bd = self.evaluate(self.loader)
        self.log(epoch)

    def backdoor_attack(self, inputs, labels, labels_backdoor):
        # batch inputs
        inputs, _ = self.augmentor.remove_seed_words_batch(inputs, labels)
        inputs = self.augmentor.insert_seed_word_batch(inputs, labels_backdoor)
        return inputs

    def evaluate(self, loader):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        f1_meter = F1Meter()
        for inputs, labels, _ in loader:
            with torch.no_grad():
                # this labels_backdoor is for anti-backdoor training
                # inputs_backdoor, labels_backdoor = backdoor(shuffle_dict_of_tensors(inputs), labels)
                labels_random = random_label_exclude_batch(labels, self.augmentor.classes)
                inputs_backdoor = self.backdoor_attack(inputs, labels, labels_random)
                outputs = self.model(**inputs_backdoor)
                loss = self.criterion(outputs.logits, labels_random)

            acc, = accuracy(outputs.logits.data, labels_random.data)
            loss_meter.update(loss.item(), labels_random.size(0))
            acc_meter.update(acc.item(), labels_random.size(0))
            f1_meter.update(outputs.logits.max(1)[1].data.cpu().numpy(), labels_random.data.cpu().numpy())
        return loss_meter.avg, acc_meter.avg, f1_meter.macro_f1, f1_meter.micro_f1


class SeedwordAdTracker(Tracker):
    def __init__(self, model, criterion, augmentor, loader, time_start=None, config=None):
        super().__init__(model, criterion, time_start=time_start, config=config)

        self.set_log(log_name='log_test_ad',
                     entry_dict={'Test-Loss-AD': lambda: self.test_loss_ad,
                                 'Test-Acc-AD': lambda: self.test_acc_ad,
                                 'Macro-F1-AD': lambda: self.macro_f1_ad,
                                 'Micro-F1-AD': lambda: self.micro_f1_ad})

        self.loader = loader
        self.augmentor = augmentor

    def update_epoch(self, epoch):
        self.test_loss_ad, self.test_acc_ad, self.macro_f1_ad, self.micro_f1_ad = self.evaluate(self.loader)
        self.log(epoch)

    def evaluate(self, loader):
        self.model.eval()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        f1_meter = F1Meter()
        for inputs, labels, _ in loader:
            with torch.no_grad():
                inputs_ad, _ = self.augmentor.remove_seed_words_batch(inputs, labels)
                outputs = self.model(**inputs_ad)
                loss = self.criterion(outputs.logits, labels)

            acc, = accuracy(outputs.logits.data, labels.data)
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc.item(), labels.size(0))
            f1_meter.update(outputs.logits.max(1)[1].data.cpu().numpy(), labels.data.cpu().numpy())
        return loss_meter.avg, acc_meter.avg, f1_meter.macro_f1, f1_meter.micro_f1


class ParameterTracker:
    def __init__(self, model):
        model_name = model._get_name().lower()
        if not 'vgg' in model_name:
            warnings.warn('%s may not be supported by parameterTracker!' % model_name)

        self.init_paras = deepcopy(dict(model.named_parameters()))
        self.last_paras = deepcopy(self.init_paras)

        self.logger = Logger('para_log.txt', title='Parameter Tracker')
        names = ['Epoch', 'Mini-batch', 'Std', 'Tot-Update', 'Update']
        num_layers = len(get_layers(self.init_paras))
        names += ['Std%i' % i for i in range(num_layers)]
        names += ['Tot-Update%i' % i for i in range(num_layers)]
        names += ['Update%i' % i for i in range(num_layers)]
        self.logger.set_names(names)

    def update(self, model, epoch, ib):
        paras = dict(model.named_parameters())

        std = torch.std(ravel_parameters(paras))
        tot_delta = self._distance(paras, self.init_paras)
        delta = self._distance(paras, self.last_paras)

        logs = [epoch, ib, std, tot_delta, delta]
        logs += self._layer_std(paras)
        logs += self._layer_distance(paras, self.init_paras)
        logs += self._layer_distance(paras, self.last_paras)

        self.logger.append(logs)
        self.last_paras = deepcopy(paras)

    def close(self):
        self.logger.close()

    def _distance(self, dict1, dict2):
        return torch.norm(ravel_parameters(dict1) - ravel_parameters(dict2))
    
    def _layer_distance(self, dict1, dict2):
        assert(dict1.keys() == dict2.keys())
        layers = get_layers(dict1)
        l_norms = []
        for l in layers:
            w = l + '.weight'
            b = l + '.bias'
    
            # should carefully re-examine the norm used here
            dw = (dict1[w] - dict2[w]).view(-1)
            db = (dict1[b] - dict2[b]).view(-1)
            dp = torch.cat([dw, db])
            l_norms.append(torch.norm(dp))
            # print(torch.std(p)) # if centered, stds is about the norm / \sqrt{fan_in}
        return l_norms
    
    def _layer_std(self, d):
        layers = get_layers(d)
        l_stds = []
        for l in layers:
            w = l + '.weight'
            b = l + '.bias'
            p = torch.cat([d[w].view(-1), d[b].view(-1)])
            l_stds.append(torch.std(p))
        return l_stds


class LrTracker:
    """
        - Adagrad only
        - Track the effective learning rate in each layer
    """
    def __init__(self, model):
        if not 'vgg' in model._get_name().lower():
            warnings.warn('%s may not be supported by LrTracker!' % model_name)

        para_dict = dict(model.named_parameters())

        self.pids = []
        for l in get_layers(para_dict):
            tmp = []
            for suf in ['.weight', '.bias']:
                pid = id(para_dict[l + suf])
                tmp.append(pid)
            self.pids.append(tuple(tmp))   

        self.logger = Logger('lr_log.txt', title='Lr Tracker')
        names = ['Epoch', 'Mini-batch', 'Mean', 'Std']
        num_layers = len(get_layers(para_dict))
        names += ['Mean%i' % i for i in range(num_layers)]
        names += ['Std%i' % i for i in range(num_layers)]
        self.logger.set_names(names)

    def update(self, epoch, i, optimizer):
        means, stds = self._layer_lr_avg(optimizer)

        # Caveat! overall std may not appears to be what you think. #TODO
        logs = [epoch, i, np.mean(means), np.std(means)]
        logs += means
        logs += stds
        self.logger.append(logs)

    def close(self):
        self.logger.close()

    def _layer_lr_avg(self, optimizer):
        means = []
        stds = []
        lr = [g['lr'] for g in optimizer.state_dict()['param_groups']]
        print(lr)
        eps = [g['eps'] for g in optimizer.state_dict()['param_groups']]
        print(eps)
        print(optimizer.state_dict()['state'][self.pids[0][0]]['sum'].mean())
        print(optimizer.state_dict()['state'][self.pids[0][0]]['sum'].min())
        print(optimizer.state_dict()['state'][self.pids[0][0]]['sum'].max())
        assert(len(lr) == 1)
        assert(len(eps) == 1)

        for pid_ in self.pids:
            lrs = []
            for pid in pid_:
                lrs.append((lr[0] / (optimizer.state_dict()['state'][pid]['sum'].sqrt() + eps[0])).view(-1))
            means.append(torch.median(torch.cat(lrs)).item())
            stds.append(torch.std(torch.cat(lrs)).item())
        return means, stds


class LipTracker:
    def __init__(self, model, device):
        self.hookers = []
        names = []
        for n, m in model.named_modules():
            if type(m) in [nn.Conv2d, nn.BatchNorm2d, nn.Linear]:
                self.hookers.append(Hooker(n, m, device=device))
                names.append(n)

        self.logger = Logger('lip_log.txt', title='Lipschitz Tracker')
        names = ['Epoch', 'Mini-Batch'] + names
        self.logger.set_names(names)

    def update(self, epoch, ib):
        lips = [h.lip() for h in self.hookers]
        self.logger.append([epoch, ib] + lips)

    def close(self):
        self.logger.close()


class FeatureTracker:
    def __init__(self, model, device):
        names = []
        hookers = []
        for n, m in model.named_modules():
            if type(m) in [nn.AdaptiveAvgPool2d]:
                hookers.append(OutputHooker(n, m, device=device))
                names.append(n)
        assert(len(names) == 1), names
        self.hooker = hookers[0]

    def get_feature(self):
        return self.hooker.output


class ManifoldTracker:
    def __init__(self, model, device=None, config=None):
        self.model = model
        self.device = config.device
        self.featTrack = FeatureTracker(model, device=config.device)
        self.diag_mask = (torch.ones(config.batch_size) - torch.eye(config.batch_size)).to(config.device)

        self.base_names = ['Epoch', 'Mini-Batch']
        self.reg_metrics = ['L2-Reg', 'LMR', 'Manifold-Reg']
        self.mr_meter = AverageMeter()
        self.logger = Logger('log_reg.txt', title='log for regularization terms')
        self.logger.set_names(self.base_names + self.reg_metrics)

    def update(self, inputs, labels=None):
        # Calculate manifold regularization - and also update the record
        batch_size = inputs.size(0)

        ## A little bit weird here
        # inputs are given, but features are implicitly captured in the last propagation
        feats = self.featTrack.get_feature()
        assert(feats.size()[0] == batch_size)

        # mask diagnonal zeros, and other weights
        mask = self.diag_mask[:batch_size, :batch_size]
        if labels is not None:
            weights = 1. - get_equivalence_matrix(labels) * 0.8
            mask *= weights
        mr = self.get_manifold_regularization(feats.view(batch_size, -1),
                                              inputs.view(batch_size, -1),
                                              mask)

        ## regularize outputs not going to work 
        # mr = get_manifold_regularization(outputs.view(batch_size, -1), inputs.view(batch_size, -1), diag_mask)
        self.mr_meter.update(mr.item(), batch_size**2)

        return mr

    def record(self, epoch, i, lmr):
        logs = [epoch, i]
        # for comparison of regularization magnitude
        logs.append(self.get_l2_regularization())
        logs.append(lmr)
        logs.append(self.mr_meter.avg)
        self.logger.append(logs)

        self.mr_meter.reset()

    def get_manifold_regularization(self, tensor1, tensor2, mask, eps=1e-7):
        assert(tensor1.size(0) == tensor2.size(0))
        assert(tensor1.size(0) == mask.size(0))
        n = tensor1.size(0)
        dm1 = get_distance_matrix(tensor1)
        dm2 = get_distance_matrix(tensor2)
        reg = dm1 / (dm2 + eps) * mask # [:n,:n]
        return reg.sum() / n**2
        # return torch.var(dm1 / (dm2 + eps) * mask[:n,:n])

    def get_l2_regularization(self):
        """
            to check the magnitude of regularization
        """
        l2_reg = 0
        for w in self.model.parameters():
            if w.requires_grad:
                l2_reg = l2_reg + w.norm(2)
        return l2_reg.item()

    def close(self):
        self.logger.close()


class ResidualTracker:
    def __init__(self, model, device):
        assert('resnet' in str(model.__class__).lower()) # currently only support preResnet
        self.names = []
        self.block_hookers = []
        # self.residual_hookers = []
        for n, m in model.named_modules():
            if self._is_block(n, m):
                self.block_hookers.append(FeatureHooker(n, m, device=device))
                self.names.append(n)
            # if self._is_residual(n, m):
            #     self.residual_hookers.append(FeatureHooker(n, m, device=device))
            
        # assert(len(self.block_hookers) == len(self.residual_hookers))

        self.logger = Logger('res_ratio_log.txt', title='Residual Tracker')
        self.base_names = ['Epoch', 'Mini-Batch']
        self.is_name_set = False
        # names = ['Epoch', 'Mini-Batch'] + names
        # self.logger.set_names(names)

    def update(self, epoch, ib):
        xs = [h.output for h in self.block_hookers]
        # residuals = [h.output for h in self.residual_hookers]
        # residuals = [h.input for h in self.residual_hookers]
        # assert(xs[0].size() == residuals[0].size())
        # assert(not all([torch.all(r == x) for r, x in zip(residuals, xs)]))
        ratios = []
        inds = []
        for i in range(len(xs)-1):
            if xs[i].size() == xs[i+1].size():
                residual = xs[i+1] - xs[i]
                ratios.append(torch.norm(residual) / torch.norm(xs[i+1]))
                inds.append(i+1)

        if not self.is_name_set:
            self.logger.set_names(self.base_names + [self.names[i] for i in inds])
            self.is_name_set = True

        # ratios = [torch.norm(r) / torch.norm(x) for r, x in zip(residuals, xs)]
        self.logger.append([epoch, ib] + ratios)

    def _is_block(self, name, module):
        return bool(re.match(r'^layer\d+.\d+$', name))

    # def _is_residual(self, name, module):
    #     # return bool(re.match(r'^layer\d+.\d+.conv2$', name))
    #     return bool(re.match(r'^layer\d+.\d+.bn2$', name))
    #     # return bool(re.match(r'^layer\d+.\d+.id$', name))

    def close(self):
        self.logger.close()



import time

class RobustTracker:

    available_metrics = ['CW', 'FGSM', 'CLEVER', 'DeepFool']

    def __init__(self, net, loaders, config, time_start):
        assert(all([m in self.available_metrics for m in config.rbTrack]))
        self.config = config

        if hasattr(config, 'rbTrackSavebest') and config.rbTrackSavebest:
            self.net = net
            self.best_rb = 1. # initialization to a big number
            self.best_metric ='FGSM'

            if hasattr(config, 'rbTrackSubsize') and config.rbTrackSubsize > 0:
                self.best_n_prob = config.rbTrackSubsize // 5
            else:
                if config.rbTrackPhase == 'train':
                    self.best_n_prob = len(loaders.trainset) // 5
                else:
                    self.best_n_prob = len(loaders.testset) // 5
            self.best_burn_in = 10

            if hasattr(config, 'rbTrackSavebestRank'):
                self.prob_rank = np.load(config.rbTrackSavebestRank)
            else:
                self.prob_rank = np.load('data_subsets/mean_pl_proba.npy')

        self.metrics = config.rbTrack
        if hasattr(config, 'rbTrackSubsize') and config.rbTrackSubsize > 0:
            if config.rbTrackPhase == 'train':
                dataset = loaders.trainset
            else:
                dataset = loaders.testset
            assert(config.rbTrackSubsize <= len(dataset))
            self.subsize = config.rbTrackSubsize

            # sample a subset
            subids = np.random.choice(len(dataset), self.subsize, replace=False)
            targets = [dataset.targets[i] for i in subids]
            subset = torch.utils.data.Subset(dataset, subids)
            subset.targets = targets

            ## build a loader 
            self.loader = torch.utils.data.DataLoader(subset, batch_size=config.batch_size,
                                                      shuffle=False, num_workers=4)

            ## Save id-index look-up table
            self.rbTrackIds = np.array([subset[i][2]['index'] for i in range(len(subset))])
            with open('rb_ids.npy', 'wb') as f:
                np.save(f, self.rbTrackIds, allow_pickle=True)
        else:
            if config.rbTrackPhase == 'train':
                self.loader = loaders.trainloader
            else:
                self.loader = loaders.testloader

        self.device = config.device
        self.time_start = time_start

        from src.preprocess import dataset_stats
        mean = np.array(dataset_stats[config.dataset]['mean'])
        std = np.array(dataset_stats[config.dataset]['std'])

        from . import DeNormalizer
        self.denormalize = DeNormalizer(mean, std, loaders.n_channel, config.device)

        # create an ART instance wrapper
        from art.estimators.classification import PyTorchClassifier
        ctr = nn.CrossEntropyLoss()
        if hasattr(config, 'loss'):
            if config.loss == 'ce':
                pass
            elif config.loss == 'mse':
                ctr = mse_one_hot() # nn.MSELoss()
            else:
                raise NotImplementedError()
        self.classifier = PyTorchClassifier(model=net,
                                            loss=ctr,
                                            input_shape=tuple(loaders.shape),
                                            nb_classes=loaders.num_classes,
                                            channels_first=True,
                                            preprocessing=(mean.reshape(loaders.n_channel, 1, 1),
                                                           std.reshape(loaders.n_channel, 1, 1)),
                                            clip_values=(0, 1),
                                            device_type='gpu')

        # create methods
        self.methods = dict()
        if 'FGSM' in self.metrics:
            from art.attacks.evasion import FastGradientMethod
            self.FGSM = FastGradientMethod(self.classifier, norm=np.inf, targeted=False, eps_step=1/255, minimal=True)
            self.methods['FGSM'] = self.__get_FGSM
        if 'DeepFool' in self.metrics:
            raise NotImplementedError()
        if 'CW' in self.metrics:
            from art.attacks.evasion import CarliniLInfMethod
            self.CW = CarliniLInfMethod(self.classifier, targeted=False, confidence=0)
            self.methods['CW'] = self.__get_CW
        if 'CLEVER' in self.metrics:
            from art.metrics import clever_u
            self.methods['CLEVER'] = self.__get_CLEVER

        # logger
        self.logger = Logger('log_robust.txt', title='fair robust metrics')
        names = ['Epoch', 'Time-elapse(Min)'] + self.metrics
        self.logger.set_names(names)

        # self.logger_distr = 

    def update(self, epoch):
        perturbs = dict([(n, []) for n in self.metrics])
        # mean_perturbs = dict([(m, AverageMeter()) for m in self.metrics])
        for _, (inputs, _, _) in enumerate(self.loader, 0):
            inputs = inputs.to(self.device)
            inputs = self.__to_art(inputs)
            for name in self.methods:
                perturbs[name].extend(self.methods[name](inputs))
                # mean_perturbs[name].update(np.mean(self.methods[name](inputs)), inputs.size(0))

        self.logger.append([epoch, (time.time() - self.time_start)/60] + [np.mean(perturbs[n]) for n in self.metrics])
        for n in self.metrics:
            self.__push(perturbs[n], file_name='rb_%s.npy' % n)
        # self.logger.append([epoch, (time.time() - self.time_start)/60] + [mean_perturbs[m].avg for m in self.metrics])

        # TODO: record distribution
        # TODO: record distribution of closest class

        if hasattr(self.config, 'rbTrackSavebest') and self.config.rbTrackSavebest:
            self.__update_best(epoch)

    def __update_best(self, epoch):
        """
            save best model based on problematic set indicator
        """

        rb = self.__get_rb(epoch, file_name='rb_%s.npy' % self.best_metric)
        print('[RobustTrack]', rb.shape, self.prob_rank.shape, self.rbTrackIds.shape)

        id_prob = self.prob_rank[self.rbTrackIds].argsort()[-self.best_n_prob:]
        rb_index = np.mean(rb[id_prob]) 
        print('[RobustTrack] Index rb: %.4f' % rb_index)

        if epoch >= self.best_burn_in and rb_index < self.best_rb:
            print('> [RobustTrack] Best got at epoch %i. Best: %.3f Current: %.3f' % (epoch, rb_index, self.best_rb))
            self.best_rb = rb_index
            torch.save(self.net.state_dict(), 'best_model_rb_%s.pt' % self.best_metric)


    def __get_FGSM(self, inputs):
        inputs_ad = self.__from_art(self.FGSM.generate(inputs))
        # TODO: output label
        return self.__distance(inputs, inputs_ad)

    def __get_CW(self, inputs):
        inputs_ad = self.__from_art(self.CW.generate(inputs, verbose=False))
        return self.__distance(inputs, inputs_ad)

    def __get_CLEVER(self, inputs):
        li = []
        for i, inputt in enumerate(inputs):
            li.append(clever_u(self.classifier, inputt.numpy(), nb_batches=2, batch_size=10, radius=0.3, norm=np.inf))
        return li
        
    def __distance(self, inputs, inputs_ad):
        """
            l-inf
        """
        inputs = inputs.view(inputs.size(0), -1)
        inputs_ad = inputs_ad.view(inputs_ad.size(0), -1)
        return torch.norm(inputs_ad - inputs, float('inf'), dim=1).tolist()

    def __to_art(self, tensor):
        # from torch type (cuda tensor) to art type (cpu tensor)
        return self.denormalize(tensor).cpu().detach()

    def __from_art(self, array):
        # from art type (cpu tensor) to torch type (cuda tensor) 
        return torch.Tensor(array)

    def __push(self, array, file_name='record.npy'):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                record = np.load(f, allow_pickle=True)
            record = np.vstack([record, array])
        else:
            record = np.array(array)
        with open(file_name, 'wb') as f:
            np.save(f, record, allow_pickle=True)

    def __get_rb(self, epoch, file_name='record.npy'):
        with open(file_name, 'rb') as f:
            record = np.load(f, allow_pickle=True)
        if len(record.shape) == 1:
            assert(epoch == 0)
            return record
        return record[epoch]


    def close(self):
        self.logger.close()


class ExampleTracker:
    """
        Track which examples are wrong or correct
    """

    __options = ['count_wrong', 'epoch_first', 'count_iters',
                 'min_perturbs', 'record_correct']

    def __init__(self, loaders, resume=False, options=['count_wrong', 'epoch_first', 'record_correct'], config=None):
        self.indices = []
        for option in options:
            assert(option in self.__options), 'Option %s not supported!' % option
        if 'min_perturbs' in options:
            assert(config.adversary == 'pgd'), 'min_perturbs only allowed when using pgd training!'
        self.options = options

        # for sanity check: only allow outputs when all the examples are scanned
        self.count = 0

        if config.dataset in ['mnist', 'cifar10', 'cifar100']:
            self.__trainsize = 50000
        elif config.dataset in ['tiny-imagenet']:
            self.__trainsize = 100000
        else:
            raise KeyError(config.dataset)

        if resume:
            self.trainsize = self.__trainsize
            if 'count_wrong' in options:
                self.count_wrong = np.load('count_wrong.npy')
            if 'epoch_first' in options:
                self.epoch_first = np.load('epoch_first.npy')
            if 'count_iters' in options:
                self.count_iters = np.zeros(self.__trainsize).astype(np.int8)
            if 'min_perturbs' in options:
                self.min_perturbs = np.zeros(self.__trainsize)
            if 'record_correct' in options:
                self.record_correct = np.zeros(self.__trainsize).astype(np.int8) # save some space

            if loaders.trainids is not None:
                self.trainsubids = loaders.trainids
                self.trainsize = len(self.trainsubids)
                # mask out
                ids_ = np.setdiff1d(np.arange(self.__trainsize), self.trainsubids)
                if 'count_wrong' in options:
                    assert(np.all(self.count_wrong[ids_] == -1)), 'masked id mismatch in saved count_wrong'
                if 'epoch_first' in options:
                    assert(np.all(self.epoch_first[ids_] == -2)), 'masked id mismatch in saved epoch_first'
                if 'count_iters' in options:
                    self.count_iters[ids_] = -1
                if 'min_perturbs' in options:
                    self.min_perturbs[ids_] = -1
                if 'record_correct' in options:
                    self.record_correct[ids_] = -1

        else:
            self.trainsize = self.__trainsize
            if 'count_wrong' in options:
                self.count_wrong = np.zeros(self.__trainsize) # count the number of epochs that an example is correct during training
            if 'epoch_first' in options:
                self.epoch_first = -np.ones(self.__trainsize)
            if 'count_iters' in options:
                self.count_iters = np.zeros(self.__trainsize).astype(np.int8)
            if 'min_perturbs' in options:
                self.min_perturbs = np.zeros(self.__trainsize)
            if 'record_correct' in options:
                self.record_correct = np.zeros(self.__trainsize).astype(np.int8) # save some space

            if loaders.trainids is not None:
                self.trainsubids = loaders.trainids
                self.trainsize = len(self.trainsubids)
                # mask out
                ids_ = np.setdiff1d(np.arange(self.__trainsize), self.trainsubids)
                if 'count_wrong' in options:
                    self.count_wrong[ids_] = -1
                if 'epoch_first' in options:
                    self.epoch_first[ids_] = -2
                if 'count_iters' in options:
                    self.count_iters[ids_] = -1
                if 'min_perturbs' in options:
                    self.min_perturbs[ids_] = -1
                if 'record_correct' in options:
                    self.record_correct[ids_] = -1

    def get_indices(self):
        assert(self.count == self.trainsize), 'num of ex scanned are short! Current: %i Expected: %i' % (self.count, self.trainsize)
        return np.hstack(self.indices)

    def update(self, outputs, labels, ids, epoch=None, count_iter=None, min_perturb=None):
        # sanity check
        if hasattr(self, 'trainsubids'):
            assert(np.all(np.in1d(ids.cpu().numpy(), self.trainsubids)))

        _, preds = outputs.topk(1, 1, True, True)
        self.count += outputs.size(0) # For sanity check
        correct_ids = ids[preds.squeeze().eq(labels)].cpu().numpy()
        wrong_ids = ids[~preds.squeeze().eq(labels)].cpu().numpy()

        # For early stopping usage
        self.indices.append(wrong_ids)
        
        # Record number of wrong epochs throughout the training
        if 'count_wrong' in self.options:
            self.count_wrong[wrong_ids] += 1

        # Record the epoch that an example is first learned
        if 'epoch_first' in self.options:
            for i in correct_ids:
                if self.epoch_first[i] == -1:
                    self.epoch_first[i] = epoch

        # Record all correct learning events in this epoch
        if 'record_correct' in self.options:
            self.record_correct[correct_ids] = 1
            self.record_correct[wrong_ids] = 0

        # Record number of necessary attacks (for early stopping in early stopping)
        if 'count_iters' in self.options:
            assert(count_iter is not None)
            self.count_iters[ids.cpu().numpy()] = count_iter.cpu().numpy()
    
        # Record minimum perturbation during attacking
        if 'min_perturbs' in self.options:
            assert(min_perturb is not None)
            self.min_perturbs[ids.cpu().numpy()] = min_perturb.cpu().numpy()

    def step(self, epoch):
        # For log
        print('[%i] %i out of %i examples are wrong' % (epoch,
                                                        len(self.get_indices()),
                                                        self.trainsize))
        print('[%i] max count: %i, min_count: %i' % (epoch, np.max(self.count_wrong), np.min(self.count_wrong)))
        if 'count_iters' in self.options:
            print('[%i] max iters: %i, min_iters: %i' % (epoch, np.max(self.count_iters), np.min(self.count_iters)))

        # Save
        if 'count_wrong' in self.options:
            with open('./count_wrong.npy', 'wb') as f:
                np.save(f, self.count_wrong, allow_pickle=True)
        if 'epoch_first' in self.options:
            with open('./epoch_first.npy', 'wb') as f:
                np.save(f, self.epoch_first, allow_pickle=True)
        # with open('./count_iters.npy', 'wb') as f:
        #     np.save(f, self.count_iters, allow_pickle=True)
        if 'count_iters' in self.options:
            self.__push(self.count_iters, file_name='count_iters.npy')
        if 'min_perturbs' in self.options:
            self.__push(self.min_perturbs, file_name='min_perturbs.npy')
        if 'record_correct' in self.options:
            self.__push(self.record_correct, file_name='record_correct.npy')

    def __push(self, array, file_name='record_correct.npy'):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                record = np.load(f, allow_pickle=True)
            record = np.vstack([record, array])
        else:
            record = np.array(array)
        with open(file_name, 'wb') as f:
            np.save(f, record, allow_pickle=True)

    # def __norm(self, delta):
    #     return torch.norm(delta.view(delta.size(0), -1), float('inf'), dim=1)

    def reset(self):
        # Clear
        self.indices = []
        self.count = 0


class AvgTracker:
    """
        Track epoch-wise average stats of various metrics
    """

    def __init__(self, name, optimizer, metrics=[], time_start=None, config=None):
        self.optimizer = optimizer
        self.metrics = metrics
        self.config = config
        if time_start is not None:
            self.time_start = time_start
        else:
            self.time_start = time.time()

        base_names = ['Epoch', 'Mini-batch', 'lr', 'Time-elapse(Min)']
        self.logger = Logger('%s.txt' % name,
                             title='log for deprecated metrics',
                             resume=self.config.resume)
        self.logger.set_names(base_names + self.metrics)
        self.meters = dict([(m, AverageMeter()) for m in self.metrics])

    def update(self, dic, size):
        for key in dic:
            self.meters[key].update(dic[key], size)

    def step(self, epoch, i):
        time_elapse = (time.time() - self.time_start)/60
        logs = [epoch, i, self.__get_lr(), time_elapse]
        logs.extend([self.meters[m].avg for m in self.metrics]) # keep the order
        self.logger.append(logs)

    def reset(self):
        for m in self.meters:
            self.meters[m].reset()

    def __get_lr(self):
        lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
        assert(len(lrs) == 1)
        return lrs[0]

    def close(self):
        self.logger.close()


# from .hessian import hessian
class AdTracker:
    """
        Track the stats of adversarial perturbation
    """

    __options = ['alignment', 'linearity', 'hessian']

    __nan = torch.tensor(np.nan)

    def __init__(self, loaders, net, config, options=['alignment', 'linearity', 'hessian'], resume=False):
        # for sanity check: only allow outputs when all the examples are scanned
        self.count = 0
        self.epoch = config.epoch_start
    
        self.config = config
        self.options = options
        assert(options), 'options is null!'
        assert(all([opt in self.__options for opt in options])), options
        self.net = net
        self.criterion = lambda reduction: nn.CrossEntropyLoss(reduction=reduction)
        if hasattr(config, 'loss'):
            if config.loss == 'ce':
                pass
            elif config.loss == 'mse':
                self.criterion = lambda reduction: mse_one_hot(reduction) # nn.MSELoss(reduction)
            else:
                raise NotImplementedError()
        self.eps = self.config.eps

        if self.config.dataset in ['mnist', 'cifar10', 'cifar100']:
            self.__trainsize = 50000
        elif self.config.dataset in ['tiny-imagenet']:
            self.__trainsize = 100000
        else:
            raise KeyError(self.config.dataset)
        self.trainsize = self.__trainsize
        self.__shape = torch.prod(torch.tensor(loaders.trainset[0][0].size())).item()

        if 'linearity' in self.options:
            self.norm_local = np.zeros(self.__trainsize, dtype=np.float16) # col 0: x; col 1: rand perturb
            self.norm_local_rand = np.zeros(self.__trainsize, dtype=np.float16) # col 0: x; col 1: rand perturb
            self.cosine_local = np.zeros(self.__trainsize, dtype=np.float16) # save some space

        if 'hessian' in self.options:
            self.ex_ids = [2319, 13687] # pre-determined - random sampled
            self.n_ex = len(self.ex_ids)
            self.losses = {'loss': [np.nan] * self.n_ex,
                           'loss_ad': [np.nan] * self.n_ex,
                           'loss_1st': [np.nan] * self.n_ex,
                           'loss_2nd': [np.nan] * self.n_ex,
                           'ghhg': [np.nan] * self.n_ex,
                           'tr_hh': [np.nan] * self.n_ex,
                           'lm_gg': [np.nan] * self.n_ex,
                           }

        if 'alignment' in self.options:
            if resume:
                self.__running_avg = torch.load('ad_running_avg.pt').to(self.config.device)
            else:
                self.__running_avg = torch.zeros(self.__trainsize, self.__shape, device=self.config.device)
            self.cosine = np.zeros(self.__trainsize, dtype=np.float16) # save some space

        # mask ids not in the training subset
        if loaders.trainids is not None:
            self.trainsubids = loaders.trainids
            self.trainsize = len(self.trainsubids)
            ids_ = np.setdiff1d(np.arange(self.__trainsize), self.trainsubids)
            if 'linearity' in self.options:
                self.norm_local[ids_] = np.nan
                self.norm_local_rand[ids_] = np.nan
                self.cosine_local[ids_] = np.nan

            if 'hessian' in self.options:
                assert(all([idx in self.trainsubids for idx in self.ex_ids])), 'bad idx choices for hessian tracker!'

            if 'alignment' in self.options:
                if resume:
                    # sanity check
                    assert(torch.all(torch.isnan(self.__running_avg[ids_]))), 'masked id mismatch in saved runnning avg!'
                self.__running_avg[ids_] = self.__nan.to(self.config.device)
                self.cosine[ids_] = np.nan


    def update(self, inputs, labels, ids, inputs_ad=None, epoch=None):
        # There might be some logics issue with this implementation,
        #       as the gradients of all examples in one epoch are not calculated by the same model
        #       model is changed after a batch

        ids = ids.cpu().numpy()
        # sanity check
        if hasattr(self, 'trainsubids'):
            assert(np.all(np.in1d(ids, self.trainsubids)))
        self.count += inputs.size(0) # For sanity check

        if 'linearity' in self.options:
            # calculate local linearity (cosine simi between x and x + gaussian)
            delta = torch.zeros_like(inputs, requires_grad=True)
            delta_rand = self.__rand_sphere(inputs.size())
            grad = self.get_gradient(inputs, delta, labels)
            grad_rand = self.get_gradient(inputs, delta_rand, labels)
            self.norm_local[ids] = self.norm(grad)
            self.norm_local_rand[ids] = self.norm(grad_rand)
            self.cosine_local[ids] = self.cosine_simi(grad, grad_rand)

        if 'hessian' in self.options:
            for ei, idx in enumerate(self.ex_ids):
                if idx in ids:
                    i = np.argwhere(ids == idx).item()
                    delta = torch.zeros_like(inputs[i].unsqueeze(0), requires_grad=True)
                    grad = self.get_gradient(inputs[i], delta, labels[i])
                    hess = self.get_hessian(inputs[i], labels[i])
                    perturb = inputs_ad[i] - inputs[i]
                    
                    self.losses['loss'][ei] = self.get_loss(inputs[i], labels[i]).item()
                    self.losses['loss_ad'][ei] = self.get_loss(inputs_ad[i], labels[i]).item()
                    self.losses['loss_1st'][ei] = (grad.view(-1) @ perturb.view(-1, 1)).item()
                    self.losses['loss_2nd'][ei] = ((perturb.view(-1) @ hess @ perturb.view(-1, 1)) / 2.).item()
                    
                    self.losses['ghhg'][ei] = (grad.view(-1) @ hess @ hess @ grad.view(-1, 1)).item()
                    self.losses['tr_hh'][ei] = torch.trace(hess @ hess).item()
                    self.losses['lm_gg'][ei] = (grad.view(-1) @ grad.view(-1, 1)).item()

        if 'alignment' in self.options:
                # calculate cosine simi between adjacent epochs, update running avg
                perturb = inputs_ad - inputs
                perturb = perturb.view(perturb.size(0), -1).detach()
                self.cosine[ids] = self.cosine_simi(self.__running_avg[ids], perturb)
                self.update_running_avg(perturb, ids)

    def step(self, epoch):
        assert(self.count == self.trainsize), 'num of ex scanned are short! Current: %i Expected: %i' % (self.count, self.trainsize)
        assert(self.epoch == epoch), 'epoch mismatch! current: %i, expected: %i' % (self.epoch, epoch)

        if 'linearity' in self.options:
            print('[%i] max cosine local: %.4f, min cosine local: %.4f' % (epoch, np.nan_to_num(np.nanmax(self.cosine_local)), np.nan_to_num(np.nanmin(self.cosine_local))))
            print('[%i] max norm local: %.4f, min norm local: %.4f' % (epoch, np.nan_to_num(np.nanmax(self.norm_local)), np.nan_to_num(np.nanmin(self.norm_local))))
            self.__push(self.norm_local, file_name='ad_norm_local.npy')
            self.__push(self.norm_local_rand, file_name='ad_norm_local_rand.npy')
            self.__push(self.cosine_local, file_name='ad_cosine_local.npy')

        if 'hessian' in self.options:
            print('[%i] MEAN loss: %.4f, loss_ad: %.4f, loss_1st: %.4f, loss_2nd: %.4f' % (epoch,
                                                                                            np.nanmean(self.losses['loss']),
                                                                                            np.nanmean(self.losses['loss_ad']),
                                                                                            np.nanmean(self.losses['loss_1st']),
                                                                                            np.nanmean(self.losses['loss_2nd']),))
            print('[%i] 0th ghhg: %.4e, tr_hh: %.4e, lm_gg: %.4e' % (epoch,
                                                                     self.losses['ghhg'][0],
                                                                     self.losses['tr_hh'][0],
                                                                     self.losses['lm_gg'][0],))
            self.__push_pt(self.losses, file_name='ad_losses.pt')

        if 'alignment' in self.options:
            print('[%i] max cosine: %.4f, min cosine: %.4f' % (epoch, np.nan_to_num(np.nanmax(self.cosine)), np.nan_to_num(np.nanmin(self.cosine))))
            self.__push(self.cosine)
            torch.save(self.__running_avg, './ad_running_avg.pt') # save running avg at this epoch in case an interrupt

        # iter
        self.epoch = epoch + 1

    def norm(self, g):
        if len(g.shape) == 1:
            # return np.linalg.norm(g)
            return torch.norm(g).cpu().detach().numpy()

        if len(g.shape) == 2:
            # return np.linalg.norm(g, axis=1)
            return torch.norm(g, dim=1).cpu().detach().numpy()

        raise TypeError('Shape not implemented..', g.size())

    def cosine_simi(self, a, b):
        # raise TypeError('Shape not implemented..', a.shape)
        assert(a.size() == b.size())
        if len(a.size()) == 1:
            cos = nn.CosineSimilarity(dim=0)
            return cos(a, b).cpu().detach().numpy()

        if len(a.size()) == 2:
            cos = nn.CosineSimilarity(dim=1)
            return cos(a, b).cpu().detach().numpy()

        raise TypeError('Shape not implemented..', a.size())

    def get_loss(self, inputs, labels, reduction='mean'):
        if len(inputs.size()) == 3:
            inputs = inputs.unsqueeze(0)
        if len(labels.size()) == 0:
            labels = labels.unsqueeze(0)
        with torch.no_grad():
            # loss = nn.CrossEntropyLoss(reduction=reduction)(self.net(inputs), labels).detach()
            loss = self.criterion(reduction=reduction)(self.net(inputs), labels).detach()
        return loss

    def get_gradient(self, inputs, delta, labels, reduction='mean'):
        if len(inputs.size()) == 3:
            inputs = inputs.unsqueeze(0)
        if len(labels.size()) == 0:
            labels = labels.unsqueeze(0)
        self.net.zero_grad() # not necesssarily, just in case
        # criterion = nn.CrossEntropyLoss(reduction=reduction)
        criterion = self.criterion(reduction=reduction)
        loss = criterion(self.net(inputs + delta), labels)
        loss.backward()
        return delta.grad.view(delta.size(0), -1).detach()

    def get_hessian(self, inpt, label):
        # only allow one example a time, too time-consuming
        assert(len(inpt.size()) == 3)
        assert(len(label.size()) == 0)
        inpt = inpt.unsqueeze(0)
        label = label.unsqueeze(0)
        delta = torch.zeros_like(inpt, requires_grad=True, device=self.config.device)
        self.net.zero_grad() # not necesssarily, just in case
        # criterion = nn.CrossEntropyLoss(reduction='sum') # because only one example, doesn't matter sum or mean
        criterion = self.criterion(reduction='sum') # because only one example, doesn't matter sum or mean
        loss = criterion(self.net(inpt + delta), label)
        loss.backward(retain_graph=True)
        
        h = torch.zeros(delta.numel(), delta.numel(), device=self.config.device)
        hessian(loss, delta, out=h)
        return h.detach()

    def __rand_sphere(self, size):
        delta = torch.rand(size, requires_grad=True, device=self.config.device)
        delta.data = delta.data * 2 * self.eps - self.eps
        return delta

    def update_running_avg(self, perturb, ids):
        # self.__running_avg[ids] = self.__running_avg[ids] * (self.epoch / (self.epoch + 1.))
        # self.__running_avg[ids] += perturb * (1. / (self.epoch + 1.))
        self.__running_avg[ids] = perturb

    def __push_pt(self, record, file_name='ad.pt'):
        if os.path.exists(file_name):
            records = torch.load(file_name)
            records.append(record)
        else:
            records = [record]
        torch.save(records, file_name)

    def __push(self, array, file_name='ad_cosine.npy'):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                record = np.load(f, allow_pickle=True)
            record = np.vstack([record, array])
        else:
            record = np.array(array)
        with open(file_name, 'wb') as f:
            np.save(f, record, allow_pickle=True)

    def reset(self):
        # Clear
        self.count = 0

    def close(self):
        # clear buffs
        if 'alignment' in self.options:
            os.remove('./ad_running_avg.pt')

