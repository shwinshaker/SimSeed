#!./env python

import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from src.models import get_model
from src.pipeline import Trainer # , train_weighted, train_reg_weighted
from src.preprocess import get_loaders
from src.utils import Dict2Obj
from src.utils import LabelSmoothingCrossEntropy, LossFloodingCrossEntropy
import random
import datetime
import json
import os
import time
from copy import deepcopy
import re

# - suppress huggingface warning
from transformers import logging
logging.set_verbosity_error() # warning()

#TODO: organize this into src.pipeline:
#       bootstrap.py: train_bootstrap(), save_pseudo_labels()
#       probe_train.py: train_probe(), aggregate_pseudo_labels(), save_weak_labels()

def save_weak_labels(config):
    # because selecting weak labels will determine the unlabeled subset, we have to select weak labels before model's pseudo-labels are generated
    print('\n=====> Save pseudo labels on weakly labeled subset..')

    print('        Get saved weak pseudo-labels..')
    # since pseudo_weak_sup_select is turned on, it must has the pseudo_weak.pt file
    # tar_weak = torch.load(os.path.join(os.path.join(config.data_dir, config.dataset), "pseudo_weak.pt")) 
    # assert(np.all(np.load(f'{config.save_dir}/id_train_weak_sup.npy') == tar_weak['index']))
    
    # We only need confidence with labels already specified.
    from script.get_pseudo_label import pseudo_label
    if config.pseudo_weak_sup_select_eval_aug == 'adremove':
        assert(config.pseudo_weak_sup_select_metric in ['confidence', 'margin']), \
               'augmentation in probe training only works for confidence or margin-based metric'

    path_name = 'weak_sup'
    if hasattr(config, 'pseudo_weak_sup_eval_split'):
        path_name += f'_split={config.pseudo_weak_sup_eval_split}'
    tar_unlabeled = torch.load(f'{config.save_dir}/unlabeled={path_name}.pt')
    # path_name = re.findall(r'unlabeled=(.*)\.pt', save_weak_labels_path_unlabeled)[0]
    # np.save(f'{config.save_dir}/{path_name}.npy', tar_unlabeled['index'])

    print('        Get confidence on weak pseudo-labels..')
    conf_type = config.pseudo_weak_sup_select_metric
    if hasattr(config, 'pseudo_weak_sup_select_metric_after_agg') and config.pseudo_weak_sup_select_metric_after_agg:
        assert(config.pseudo_weak_sup_select_metric in ['confidence', 'margin']), 'only confidence or margin-based metric can be built after aggregation'
        # save softmax for metric-based selection after aggregating multiple runs
        conf_type = 'softmax'
    # if epoch is not None:
        # save_name += '_epoch=%i' % (epoch + 1)
    if config.pseudo_weak_sup_select_train_aug != 'none':
        path_name += f'_aug_train={config.pseudo_weak_sup_select_train_aug}'
    if config.pseudo_weak_sup_select_eval_aug != 'none':
        path_name += f'_aug_eval={config.pseudo_weak_sup_select_eval_aug}'
    if hasattr(config, 'pseudo_weak_sup_select_use_prediction') and config.pseudo_weak_sup_select_use_prediction:
        use_prediction = True
    else:
        use_prediction = False
    tar = pseudo_label(config,
                       dataset=config.dataset,
                       path_unlabeled_idx=tar_unlabeled['index'], # f'{config.save_dir}/{path_name}.npy',
                       path_name=path_name,
                       model_path=config.save_dir,
                       gpu_id=config.gpu_id,
                       save_dir=config.save_dir,
                       data_dir=config.data_dir,
                       pseudo_label=tar_unlabeled['pseudo_label'],
                       true_label=tar_unlabeled['true_label'],
                       conf_type=conf_type,
                       aug_type=config.pseudo_weak_sup_select_eval_aug, # true labels can be used to double check
                       dropout=config.get('pseudo_weak_sup_select_mc_dropout'),
                       dropout_passes=config.get('pseudo_weak_sup_select_mc_dropout_passes'),
                       use_prediction=use_prediction,
                       randremove_num=config.randremove_num,
                       randadremove_num=config.randadremove_num if hasattr(config, 'randadremove_num') else None,
                       augparaphrase_temperature=config.augparaphrase_temperature) 

    # -- sanity check: `pseudo_label` function here only adds confidence to the tar since labels are already given
    # assert(np.all(np.load(f'{config.save_dir}/id_eval_weak_sup.npy') == tar_unlabeled['index']))
    # assert(np.all(np.load(f'{config.save_dir}/id_eval_weak_sup.npy') == tar['index']))
    assert(np.all(tar_unlabeled['index'] == tar['index']))
    assert(np.all(tar_unlabeled['true_label'] == tar['true_label']))
    if not use_prediction:
        assert(np.all(tar_unlabeled['pseudo_label'] == tar['pseudo_label']))
    # print('   Accuracy on weak pseudo-labels: %.2f' % (np.sum(tar_weak['pseudo_label'] == tar['pseudo_label']) / len(tar['pseudo_label'])))
        
def save_pseudo_labels(config):
    print('\n=====> Save pseudo labels..')
    from script.get_pseudo_label import pseudo_label
    print(' --------------- Start pseudo labeling using current model ----------------')
    if config.save_pseudo_label_unlabeled_idx_path is None:
        config.save_pseudo_label_unlabeled_idx_path = f"{config.save_dir}/id_unlabeled.npy"
        print(f"save_pseudo_label_unlabeled_idx_path changed to {config.save_pseudo_label_unlabeled_idx_path}")
    pseudo_label(config,
                 dataset=config.dataset,
                 path_unlabeled_idx=config.save_pseudo_label_unlabeled_idx_path,
                 model_path=config.save_dir,
                 gpu_id=config.gpu_id,
                 save_dir=config.save_dir)


def train_single(config):
    print('=====> Init training..')
    config, model, loaders, criterion, optimizer, scheduler = init_train(config)

    print('=====> Training..')
    trainer = Trainer(config)
    trainer(model, loaders, criterion, optimizer, scheduler=scheduler)
    trainer.close()

def clean(config):
    print('\n=====> Clean saved checkpoints..')
    if not config.save_checkpoint:
        os.remove(os.path.join(config.save_dir, 'checkpoint.pth.tar'))
    if not config.save_model:
        if os.path.exists(os.path.join(config.save_dir, 'best_model.pt')):
            os.remove(os.path.join(config.save_dir, 'best_model.pt'))
        if os.path.exists(os.path.join(config.save_dir, 'best_model_loss.pt')):
            os.remove(os.path.join(config.save_dir, 'best_model_loss.pt'))
        os.remove(os.path.join(config.save_dir, 'model.pt'))


def print_level1(string):
    print()
    print(f'          ----------------------------------------------------------------------------')
    print(f'                                    {string}                               ')
    print(f'          ----------------------------------------------------------------------------')
    print()

def print_level2(string):
    print()
    print(f'                                    {string}                               ')
    print()


from src.utils import get_tensor_max_except, get_files_regex_match
def aggregate_pseudo_labels(config):
    print('\n==> Search saved pseudo-labels..')
    tar_names = get_files_regex_match(config.weak_model_path, regex_str='pseudo_unlabeled=weak_sup.*\.pt')
    tar_paths = ['%s/%s' % (config.weak_model_path, tar_name) for tar_name in tar_names]
    print('     Found saved tar paths: ', tar_paths)
    loaded_tars = [torch.load(tar_path) for tar_path in tar_paths]

    if hasattr(config, 'pseudo_weak_sup_select_cross_fitting') and config.pseudo_weak_sup_select_cross_fitting:
        print('\n==> Merge splits from cross-fitting')
        # - find all root names without splitting
        root_names = list(set([re.findall(r'split=\d+(.*)\.pt', tar_name)[0] for tar_name in tar_names]))
        print('     Found unique root names: ', root_names)

        # # - find aug types
        # aug_types = []
        # if any(['aug' not in tar_name for tar_name in tar_names]):
        #     aug_types.append('none')
        # aug_types.extend(list(set([re.findall(r'aug=(.*)\.pt', tar_name)[0] for tar_name in tar_names if 'aug' in tar_name])))
        # print('     Found augment types: ', aug_types)

        # - merge splits for each aug type
        merged_tars = []
        # for aug_type in aug_types:
        #     print(f'     ---- Merging split (aug_type={aug_type}).. ----')
        #     if aug_type == 'none':
        #         regex_match = r'pseudo_unlabeled=weak_sup_split=\d+\.pt'
        #     else:
        #         regex_match = r'pseudo_unlabeled=weak_sup_split=\d+_aug=%s\.pt' % aug_type
        for root_name in root_names:
            print(f'     ---- Merging split (root={root_name}).. ----')
            regex_match = r'pseudo_unlabeled=weak_sup_split=\d+%s\.pt' % root_name
            tar_splits = [loaded_tars[i] for i, tar_name in enumerate(tar_names) if re.match(regex_match, tar_name)]
            print('     Found splits: ', [tar_name for tar_name in tar_names if re.match(regex_match, tar_name)])

            # -- sanity check
            print('     Validating splits...') 
            import itertools
            assert(not any(len(np.intersect1d(split1['index'], split2['index'])) for split1, split2 in itertools.combinations(tar_splits, 2)))
            # from functools import reduce
            # assert(np.all(reduce(np.union1d, (split['index'] for split in tar_splits)) == tar['index']))
            # will check this later in data loader
            assert(all(split1['model_path'] == split2['model_path'] for split1, split2 in itertools.combinations(tar_splits, 2)))
            assert(all(split1['model_state'] == split2['model_state'] for split1, split2 in itertools.combinations(tar_splits, 2)))
            path_names_strip = [re.findall(r'(.*)_split=.*', split['path_name'])[0] for split in tar_splits]
            assert(len(set(path_names_strip)) == 1)

            # -- merge
            print('     Merging splits...') 
            print('       Array keys:', [key for key in tar_splits[0].keys() if isinstance(tar_splits[0][key], np.ndarray)])
            merged_tar = {key: np.concatenate([split[key] for split in tar_splits]) for key in tar_splits[0].keys() \
                            if isinstance(tar_splits[0][key], np.ndarray) and key != 'mis_index'}
            print('       Concated array length (except mis_index): ', [len(merged_tar[key]) for key in merged_tar])
            merged_tar['model_path'] = tar_splits[0]['model_path']
            merged_tar['model_state'] = tar_splits[0]['model_state']
            # merged_tar['path_unlabeled_idx'] = 
            # shouldn't load(tar['path_unlabeled_idx']) be the same as tar['index']? Yes, see get_pseudo_label
            #   if so, just remove it, or save the array directly in the dict
            merged_tar['path_name'] = path_names_strip[0]
            print('       Other keys: ', {k: v for k, v in merged_tar.items() if not isinstance(v, np.ndarray)})
            print('     Done.') 
            merged_tars.append(merged_tar)
        loaded_tars = merged_tars

    if len(loaded_tars) == 1: 
        # sort tar index to align with original index order, only to pass sanity check in later dataloader
        sort_idx = np.argsort(loaded_tars[0]['index'])
        tar = {k: v[sort_idx] if isinstance(v, np.ndarray) and k != 'mis_index' else v for k, v in loaded_tars[0].items()}
        torch.save(tar, '%s/%s.pt' % (config.weak_model_path, 'pseudo_unlabeled=weak_sup')) # fixed name
        return

    print('\n==> Aggregate confidences from multiple augmentations..')
    print('       Align list of tars..')
    sorted_tars = []
    for tar in loaded_tars:
        sort_idx = np.argsort(tar['index'])
        sorted_tars.append({k: v[sort_idx] if isinstance(v, np.ndarray) and k != 'mis_index' else v for k, v in tar.items()})
    loaded_tars = sorted_tars

    print('       Sanity check list of tars..')
    for tar in loaded_tars[1:]:
        assert(np.array_equal(loaded_tars[0]['index'], tar['index'])), 'index unaligned!'
        assert(np.array_equal(loaded_tars[0]['pseudo_label'], tar['pseudo_label'])), 'pseudo_label unaligned!'
        assert(np.array_equal(loaded_tars[0]['true_label'], tar['true_label'])), 'true_label unaligned!'

    print(f'       Aggregate confidence from list of tars (rule = {config.pseudo_weak_sup_select_confidence_agg})..')
    if config.pseudo_weak_sup_select_confidence_agg == 'average':
        confidences = [tar['confidence'] for tar in loaded_tars]
        confidence = np.stack(confidences).mean(axis=0) # to be selected, ranks in both confidences have to be high
    elif config.pseudo_weak_sup_select_confidence_agg == 'rank_min':
        confidences = [rank(tar['confidence']) for tar in loaded_tars]
        confidence = np.stack(confidences).min(axis=0) # to be selected, ranks in both confidences have to be high
    elif config.pseudo_weak_sup_select_confidence_agg == 'rank_average':
        confidences = [rank(tar['confidence']) for tar in loaded_tars]
        confidence = np.stack(confidences).mean(axis=0) # to be selected, ranks in both confidences have to be high
    else:
        raise KeyError('confidence aggregrate rule %s not recognized!' % config.pseudo_weak_sup_select_confidence_agg)

    if hasattr(config, 'pseudo_weak_sup_select_metric_after_agg') and config.pseudo_weak_sup_select_metric_after_agg:
        print(f'       Build confidence after aggregation (metric = {config.pseudo_weak_sup_select_metric})..')
        assert(config.pseudo_weak_sup_select_confidence_agg == 'average'), 'build confidence metric after agg only works with confidence_agg=average'
        assert(config.pseudo_weak_sup_select_metric in ['confidence', 'margin']), 'only confidence or margin-based metric can be built after aggregation'
        assert(len(confidence.shape) == 2), 'invalid confidence shape! Should resemble softmax, likely called from wrong place'
        softmaxs, labels = torch.from_numpy(confidence), torch.from_numpy(loaded_tars[0]['pseudo_label'])
        if config.pseudo_weak_sup_select_metric == 'confidence':
            confidence = softmaxs.gather(1, labels.view(-1, 1)).squeeze().numpy()
        elif config.pseudo_weak_sup_select_metric == 'margin':
            confidence = (softmaxs.gather(1, labels.view(-1, 1)).squeeze() - get_tensor_max_except(softmaxs, labels)[0]).numpy()

    tar = deepcopy(loaded_tars[0])
    tar['confidence'] = confidence
    assert(len(set([tar_['path_name'] for tar_ in loaded_tars])) == len([tar_['path_name'] for tar_ in loaded_tars])), 'path_name not unique!'
    tar['path_name'] = [tar_['path_name'] for tar_ in loaded_tars]
    tar['mis_index'] = {tar_['path_name']: tar_['mis_index'] for tar_ in loaded_tars}
    torch.save(tar, '%s/%s.pt' % (config.weak_model_path, 'pseudo_unlabeled=weak_sup')) # fixed name
    return


def probe_train_single(config_probe, config):

    ## -- process config
    if isinstance(config.pseudo_weak_sup_select_train_aug, list):
        train_augs = deepcopy(config.pseudo_weak_sup_select_train_aug)
    else:
        train_augs = [config.pseudo_weak_sup_select_train_aug]
    if isinstance(config.pseudo_weak_sup_select_eval_aug, list):
        eval_augs = deepcopy(config.pseudo_weak_sup_select_eval_aug)
    else:
        eval_augs = [config.pseudo_weak_sup_select_eval_aug]
    same_eval_as_train = False
    if 'same_as_train' in eval_augs:
        assert(len(eval_augs) == 1), 'same_as_train should be the only aug type'
        eval_augs = deepcopy(train_augs)
        same_eval_as_train = True

    ## -- probe training with augments
    for aug_type_train in train_augs:
        if aug_type_train == 'adremove':
            config_probe.adremove = True
            config_probe.adremove_alpha = 1.0 # train on context only, no regular loss
        elif aug_type_train == 'randremove':
            config_probe.randremove = True
            config_probe.randremove_alpha = 1.0
            if hasattr(config, 'pseudo_weak_sup_select_aug_randremove_ratio'):
                config_probe.randremove_num = config.pseudo_weak_sup_select_aug_randremove_ratio
            else:
                config_probe.randremove_num = 0.1
        elif aug_type_train == 'randadremove':
            config_probe.randadremove = True
            config_probe.randadremove_alpha = 1.0
            if hasattr(config, 'pseudo_weak_sup_select_aug_randremove_ratio'):
                config_probe.randremove_num = config.pseudo_weak_sup_select_aug_randremove_ratio
            else:
                config_probe.randremove_num = 0.1
        elif aug_type_train == 'randnotadremove':
            config_probe.randnotadremove = True
            config_probe.randnotadremove_alpha = 1.0
            if hasattr(config, 'pseudo_weak_sup_select_aug_randremove_ratio'):
                config_probe.randremove_num = config.pseudo_weak_sup_select_aug_randremove_ratio
            else:
                config_probe.randremove_num = 0.1
        elif aug_type_train == 'randrandadremove':
            config_probe.randrandadremove = True
            config_probe.randrandadremove_alpha = 1.0
            if hasattr(config, 'pseudo_weak_sup_select_aug_randremove_ratio'):
                config_probe.randremove_num = config.pseudo_weak_sup_select_aug_randremove_ratio
            else:
                config_probe.randremove_num = 0.1
            if hasattr(config, 'pseudo_weak_sup_select_aug_randadremove_ratio'):
                config_probe.randadremove_num = config.pseudo_weak_sup_select_aug_randadremove_ratio
            else:
                config_probe.randadremove_num = 0.1
        elif aug_type_train == 'paraphrase':
            config_probe.augparaphrase = True
            config_probe.augparaphrase_alpha = 1.0
            if hasattr(config, 'pseudo_weak_sup_select_aug_paraphrase_temperature') and config.pseudo_weak_sup_select_aug_paraphrase_temperature:
                config_probe.augparaphrase_temperature = config.pseudo_weak_sup_select_aug_paraphrase_temperature
            else:
                config_probe.augparaphrase_temperature = 2
        elif aug_type_train == 'mlmreplace':
            config_probe.mlmreplace = True
            config_probe.mlmreplace_alpha = 1.0
            if hasattr(config, 'pseudo_weak_sup_select_aug_mlmreplace_ratio'):
                config_probe.randremove_num = config.pseudo_weak_sup_select_aug_mlmreplace_ratio
            else:
                config_probe.randremove_num = 0.1
        else:
            config_probe.adremove = False
            config_probe.randremove = False
            config_probe.randadremove = False
            config_probe.augparaphrase = False
            config_probe.mlmreplace = False

        config_probe.pseudo_weak_sup_select_train_aug = aug_type_train # only for naming the saved pseudo-label tar
        print_level1(f'Aug type train: {aug_type_train}')
        train_single(config_probe)

        for aug_type_eval in eval_augs:
            if same_eval_as_train:
                if aug_type_eval != aug_type_train:
                    continue
            print_level2(f'Aug type eval: {aug_type_eval}')
            config_probe.pseudo_weak_sup_select_eval_aug = aug_type_eval
            save_weak_labels(config_probe)

        clean(config_probe)


def probe_train(config):

    # TODO: ~~Add configs into a list in a tree order~~ (too complicated)
    #       base configs  x-> aug configs  x-> cross-fitting config1
    #                      -> aug config2  x-> cross-fitting config1
    #       simply define aug_probe_train(config):
    #          for ... in train_augs; eval_aug = ... base_tmp_configs = ...; Run

    print('\n=====> Probe training')
    # Probe training: first traing once to get confidence on weak labels, then train on selected pseudo-labels
    config_probe = deepcopy(config)
    config_probe.pseudo_weak_sup_select = False
    config_probe.save_dir = 'weak_label_filter'
    if not os.path.isdir(config_probe.save_dir):
        os.mkdir(config_probe.save_dir)
    config_probe.epochs = config.pseudo_weak_sup_select_train_epochs # more epochs damage confidence
    if hasattr(config, 'pseudo_weak_sup_select_channel_dropout') and config.pseudo_weak_sup_select_channel_dropout:
        config_probe.channel_dropout = config.pseudo_weak_sup_select_channel_dropout
        config_probe.channel_dropout_type = config.pseudo_weak_sup_select_channel_dropout_type
    if config.pseudo_weak_sup_select_metric == 'norm_loss_avg':
        config_probe.lossTrack = True

    if hasattr(config, 'pseudo_weak_sup_select_cross_fitting') and config.pseudo_weak_sup_select_cross_fitting:
        print(f'=====> Cross-fitting with {config.pseudo_weak_sup_select_cross_fitting_n_splits} splits..')
        for i_split in range(config.pseudo_weak_sup_select_cross_fitting_n_splits):
            print_level1(f'Eval split [{i_split+1}/{config.pseudo_weak_sup_select_cross_fitting_n_splits}]')
            config_probe.pseudo_weak_sup_n_splits = config.pseudo_weak_sup_select_cross_fitting_n_splits
            config_probe.pseudo_weak_sup_eval_split = i_split

            probe_train_single(config_probe, config)
    else:
        probe_train_single(config_probe, config)

    config.weak_model_path = 'weak_label_filter'


def train_wrap(**config):
    config = Dict2Obj(config)

    start = time.time()

    # time for log
    print('\n=====> Current time..')
    print(datetime.datetime.now())

    # -------------------------- Environment ------------------------------
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ------ start training ----------
    config.save_dir = '.'

    if (hasattr(config, 'pseudo_weak_sup') and config.pseudo_weak_sup) and \
       (hasattr(config, 'pseudo_weak_sup_select') and config.pseudo_weak_sup_select):
        print_level1('Probe training starts..')
        probe_train(config)
        print_level1('Aggregate pseudo-labels (& confidence)..')
        aggregate_pseudo_labels(config)
        print_level1('Probe training ended.')

    # config.randremove = True
    # config.randremove_alpha = 1.0
    # config.randremove_num = 0.9
    train_single(config)
    if hasattr(config, 'save_pseudo_label') and config.save_pseudo_label:
        save_pseudo_labels(config)
    clean(config)
    print('  Finished.. %.3f\n' % ((time.time() - start) / 60.0))

    if hasattr(config, 'bootstrap') and config.bootstrap and config.bootstrap_n_iteration:
        # ------ start boostrapping ----------
        train_bootstrap(config, time_start=start)

    # time for log
    print('\n=====> Current time..')
    print(datetime.datetime.now())


def train_bootstrap(config, time_start=None):

    print_level1('Bootstrapping starts..')

    # default settings for training subset in bootstrapping
    if config.trainsize is not None:
        # Needs to ensure the same randomly selected set is used in training
        print('====> Hyperparameter adapted..')
        assert config.train_subset_path is None
        config.train_subset_path = f"{config.save_dir}/id_train_{config.dataset}_size={config.trainsize}.npy"
        config.trainsize = None # use already randomly selected subset 
        print('    trainsize: %s' % config.trainsize)
        print('    train_subset_path: %s' % config.train_subset_path)

    ## start training
    iter_start = 0
    config_init = deepcopy(config)
    for iter_ in range(iter_start, config.bootstrap_n_iteration):
        print_level1(f'Boostrap Iter: [{iter_+1}/{config.bootstrap_n_iteration}]')
        config.bootstrapping = True # Set training state to bootstrapping
        config.save_dir = 'iteration-%i' % iter_
        os.mkdir(config.save_dir)
        if iter_ == 0:
            config.pseudo_model_path = '.'
            config.pseudo_threshold = get_bootstrap_threshold(config_init.pseudo_threshold, iter_, config=config)
            if hasattr(config, 'bootstrap_threshold_init') and config.bootstrap_threshold_init:
                config.pseudo_threshold = config_init.bootstrap_threshold_init # in the first iteration, use the best threshold for the model trained only on the labeled dataset
            config.epochs = get_bootstrap_epochs(config_init.epochs, iter_, config=config)

            # config.randremove = True
            # config.randremove_alpha = 1.0
            # config.randremove_num = 0.9

        elif iter_ == config.bootstrap_n_iteration - 1:
            # in the final iteration, using the default training setting for fair comparison
            config.pseudo_model_path = 'iteration-%i' % (iter_ - 1)
            config.pseudo_threshold = get_bootstrap_threshold(config_init.pseudo_threshold, iter_, config=config)
            config.epochs = get_bootstrap_epochs(config_init.epochs, iter_, config=config)
            if hasattr(config, 'bootstrap_epoch_final') and config.bootstrap_epoch_final:
                config.epochs = config_init.bootstrap_epoch_final
            # config.adremove = True
            if (hasattr(config, 'adremove') and config.adremove) and \
               (hasattr(config, 'bootstrap_reg_weight_func') and config.bootstrap_reg_weight_func):
                config.adremove_alpha = get_bootstrap_reg_weight(config_init.adremove_alpha, iter_, config=config)
            # config.adremove_alpha = 0.5

            # config.randremove = False

        else:
            config.pseudo_model_path = 'iteration-%i' % (iter_ - 1)
            config.pseudo_threshold = get_bootstrap_threshold(config_init.pseudo_threshold, iter_, config=config)
            config.epochs = get_bootstrap_epochs(config_init.epochs, iter_, config=config)
            if (hasattr(config, 'adremove') and config.adremove) and \
               (hasattr(config, 'bootstrap_reg_weight_func') and config.bootstrap_reg_weight_func):
                config.adremove_alpha = get_bootstrap_reg_weight(config_init.adremove_alpha, iter_, config=config)

            # config.randremove = True
            # config.randremove_alpha = 1.0
            # config.randremove_num = 0.9

        print('====> Hyperparameter adapted..')
        print('    epochs: %i' % config.epochs)

        train_single(config)
        if hasattr(config, 'save_pseudo_label') and config.save_pseudo_label:
            save_pseudo_labels(config)
        clean(config)
        print('  Iter %i   Finished.. %.3f\n' % (iter_, (time.time() - time_start) / 60.0))


def get_bootstrap_reg_weight(weight, iter_=None, config=None):
    if config.bootstrap_reg_weight_func == 'constant':
        return weight
    elif config.bootstrap_reg_weight_func.startswith('linear_min'):
        weight_min = float(re.findall(r'min=(\d\.\d+)', config.bootstrap_reg_weight_func)[0])
        return weight - (iter_ + 1) / config.bootstrap_n_iteration * (weight - weight_min)
    elif config.bootstrap_reg_weight_func.startswith('linear_max'):
        weight_max = float(re.findall(r'max=(\d\.\d+)', config.bootstrap_reg_weight_func)[0])
        return weight + (iter_ + 1) / config.bootstrap_n_iteration * (weight_max - weight)
    elif config.bootstrap_reg_weight_func.startswith('step_iter'):
        step_iter = float(re.findall(r'iter=(\d+)', config.bootstrap_reg_weight_func)[0])
        step_value = float(re.findall(r'value=(\d\.\d+)', config.bootstrap_reg_weight_func)[0])
        if iter_ < step_iter:
            return weight
        else:
            return step_value
    else:
        raise NotImplementedError(config.bootstrap_reg_weight_func)


def get_bootstrap_threshold(th, iter_=None, config=None):
    if config.bootstrap_threshold_func == 'constant':
        return th
    elif config.bootstrap_threshold_func.startswith('increment'):
        inc = float(config.bootstrap_threshold_func.split('_')[1])
        return th + iter_ * inc 
    elif config.bootstrap_threshold_func.startswith('decrement'):
        dec = float(config.bootstrap_threshold_func.split('_')[1])
        return th - iter_ * dec
    elif config.bootstrap_threshold_func.startswith('linear_min'):
        th_min = float(re.findall(r'min=(\d\.\d+)', config.bootstrap_threshold_func)[0])
        return th - (iter_ + 1) / config.bootstrap_n_iteration * (th - th_min)
    elif config.bootstrap_threshold_func.startswith('linear_max'):
        th_max = float(re.findall(r'max=(\d\.\d+)', config.bootstrap_threshold_func)[0])
        return th + (iter_ + 1) / config.bootstrap_n_iteration * (th_max - th)
    elif config.bootstrap_threshold_func.startswith('loglinear'):
        th_min = float(re.findall(r'min=(\d\.\d+)', config.bootstrap_threshold_func)[0])
        def to_neg_log(th): return -math.log(1 - th)
        def from_neg_log(th_log): return 1 - math.exp(-th_log)
        return from_neg_log(to_neg_log(th) - (iter_ + 1) / config.bootstrap_n_iteration * (to_neg_log(th) - to_neg_log(th_min)))
    else:
        raise NotImplementedError(config.bootstrap_threshold_func)

def get_bootstrap_epochs(epochs, iter_, config=None):
    if config.bootstrap_epoch_func == 'constant':
        return epochs
    elif config.bootstrap_epoch_func.startswith('linear'):
        epochs_min = int(re.findall(r'min=(\d+)', config.bootstrap_epoch_func)[0])
        return int(epochs_min + iter_ / config.bootstrap_n_iteration * (epochs - epochs_min))
    else:
        raise NotImplementedError(config.bootstrap_epoch_func)

def init_train(config):
    #TODO: Merge to trainer's init
    # Random seed
    if config.manual_seed is not None:
        random.seed(config.manual_seed)
        torch.manual_seed(config.manual_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.manual_seed)
        ## TODO: Hugging face seed

    # -------------------------- Dataset ------------------------------
    print('\n=====> Loading data..')
    loaders = get_loaders(dataset=config.dataset, test_ratio=config.test_ratio, batch_size=config.batch_size,
                          trainsize=config.get('trainsize'), train_subset_path=config.get('train_subset_path'),
                          data_dir=config.data_dir,
                          config=config)

    # --------------------------------- criterion ------------------------------- 
    config.reduction = 'mean' # 'none' # Prevent reduction if wish to do instance reweighting
    if hasattr(config, 'label_smoothing') and config.label_smoothing:
        criterion = LabelSmoothingCrossEntropy(reduction=config.reduction, smoothing=config.label_smoothing)
    elif hasattr(config, 'loss_flooding') and config.loss_flooding:
        criterion = LossFloodingCrossEntropy(reduction=config.reduction, flooding=config.loss_flooding)
    else:
        criterion = nn.CrossEntropyLoss(reduction=config.reduction)

    # -------------------------- Model ------------------------------
    print('\n=====> Initializing model..')
    model = get_model(config, loaders)

    ## -- Load weights
    if config.state_path:
        print('=====> Loading pre-trained weights..')
        assert(not config.resume), 'pre-trained weights will be overriden by resume checkpoint! Resolve this later!'
        state_dict = torch.load(config.state_path, map_location=config.device)
        model.load_state_dict(state_dict)
    
    # -------------------------- Optimizer ------------------------------
    print('\n=====> Initializing optimizer..')
    if config.opt.lower() == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    else:
        raise KeyError(config.opt)

    # -------------------------- Scheduler ------------------------------
    if config.scheduler.lower() == 'linear':
        # TODO: len(loaders.trainloader) // config.update_freq might not yield the exact number of training steps with larger batch size
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=len(loaders.trainloader) // config.update_freq * config.epochs)
    else:
        scheduler = None

    return config, model, loaders, criterion, optimizer, scheduler

if __name__ == '__main__':

    with open('para.json') as json_file:
        config = json.load(json_file)
        print(config)
    train_wrap(**config)
