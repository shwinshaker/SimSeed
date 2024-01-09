#!./env python

import argparse
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import shutil
import yaml
import json
import torch

from src.utils import check_path, check_path_remote
from fractions import Fraction

def check_num(num):
    if type(num) in [float, int]:
        return num

    if isinstance(num, str):
        return float(Fraction(num))

    raise TypeError(num)


def read_config(config_file='config.yaml', remote=False, server=None): # , remote_dir=''):

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -- hyperparas massage --
    # TODO: tracker not implemented for resnet
    for key in ['lr', 'wd', 'momentum', 'gamma',  'label_smoothing', 'loss_flooding']:
        if key in config and config[key] is not None:
            config[key] = check_num(config[key])

    if config['state_path']:
        config['state_path'] = os.path.join(os.getcwd(), config['checkpoint_dir'], config['state_path'])

    # -- checkpoint set --
    config['checkpoint'] = '%s' % config['opt']
    config['checkpoint'] += '_%s' % config['model']

    if 'train_module' in config and config['train_module']:
        config['checkpoint'] += '_train=%s' % '-'.join(config['train_module'])

    if config['dataset'] != 'imdb':
        config['checkpoint'] = config['dataset'] + '_' + config['checkpoint']

    config['checkpoint'] += '_epoch=%i' % config['epochs']

    # if config['scheduler'] != 'multistep':
    #     config['checkpoint'] += '_%s' % config['scheduler']
    #     if 'cyclic' in config['scheduler']:
    #         config['checkpoint'] += '_%g_%g' % (config['lr'], config['lr_max'])
    #     if config['scheduler'] == 'cosine':
    #         config['checkpoint'] += '_lr=%g' % config['lr'] # max lr, also initial lr
    #     if config['scheduler'] == 'cosine_restart':
    #         config['checkpoint'] += '_cycle=%g' % config['epoch_cycle']
    # else:
    #     if config['lr'] != 0.1:
    #         config['checkpoint'] += ('_lr=%.e' % config['lr']).replace('.', '_')

    if 'update_freq' not in config:
        config['update_freq'] = 1
    if config['batch_size'] * config['update_freq'] != 16:
        config['checkpoint'] += '_bs=%i' % (config['batch_size'] * config['update_freq'])
    if config['wd'] is not None and config['wd'] > 0:
        config['checkpoint'] += '_wd=%g' % config['wd']
    if config['momentum'] is not None and config['momentum'] > 0:
        config['checkpoint'] += '_mom=%g' % config['momentum']
    if 'trainsize' in config and config['trainsize']:
        config['checkpoint'] += '_ntrain=%i' % config['trainsize']
    if 'train_subset_path' in config and config['train_subset_path']:
        config['checkpoint'] += '_sub=%s' % config['train_subset_path'].split('/')[-1].split('.')[0]
    if 'trainnoisyratio' in config and config['trainnoisyratio']:
        config['checkpoint'] += '_trainnoise=%g' % config['trainnoisyratio']
    if 'trainnoisyconfusionmatrix' in config and config['trainnoisyconfusionmatrix']:
        config['checkpoint'] += '_trainnoise_confusion=%s' % os.path.split(config['trainnoisyconfusionmatrix'])[-1].rstrip('.npy')

    if 'label_smoothing' in config and config['label_smoothing']:
        config['checkpoint'] += '_ls=%g' % config['label_smoothing']
    if 'loss_flooding' in config and config['loss_flooding']:
        config['checkpoint'] += '_lf=%g' % config['loss_flooding']
    if 'gradient_clipping' in config and config['gradient_clipping']:
        config['checkpoint'] += '_gc'

    if 'antibackdoor' in config and config['antibackdoor']:
        config['checkpoint'] += '_abd=%g' % config['antibackdoor_alpha']
        if 'antibackdoor_repeat' in config and config['antibackdoor_repeat'] > 1:
            config['checkpoint'] += '_repeat=%i' % config['antibackdoor_repeat']
        # config['checkpoint'] += '_ascent'

    if 'adremove' in config and config['adremove']:
        config['checkpoint'] += '_adr=%g' % config['adremove_alpha']
    if 'randremove' in config and config['randremove']:
        config['checkpoint'] += '_rar=%g' % config['randremove_alpha']
        if config['randremove_num'] != 1:
            config['checkpoint'] += '_num=%g' % config['randremove_num']
    if 'consistremove' in config and config['consistremove']:
        config['checkpoint'] += '_cstr=%g' % config['consistremove_alpha']
        if config['consistremove_T'] != 1:
            config['checkpoint'] += '_T=%g' % config['consistremove_T']

    if 'bootstrap' in config and config['bootstrap']:
        config['checkpoint'] += '_bootstrap'
        if config['bootstrap_n_iteration'] > 1:
            config['checkpoint'] += '_iter=%i' % config['bootstrap_n_iteration']
        if config['bootstrap_n_iteration'] > 1:
            if 'bootstrap_threshold_func' in config and config['bootstrap_threshold_func'] != 'constant':
                config['checkpoint'] += '_func=%s' % config['bootstrap_threshold_func']
            if 'bootstrap_epoch_func' in config and config['bootstrap_epoch_func'] != 'constant':
                config['checkpoint'] += '_epoch_func=%s' % config['bootstrap_epoch_func']
            if 'bootstrap_reg_weight_func' in config and config['bootstrap_reg_weight_func'] != 'constant':
                config['checkpoint'] += '_reg_weight_func=%s' % config['bootstrap_reg_weight_func']

        # defaul peudo-label setting
        config['checkpoint'] += '_th=%g_type=%s' % (config['pseudo_threshold'], config['pseudo_threshold_type'])
        if 'pseudo_class_balance' in config and (not config['pseudo_class_balance']):
            config['checkpoint'] += '_nonbalanced'

    def is_aug_exist(aug_args, aug_type):
        if isinstance(aug_args, str):
            return True if aug_args == aug_type else False
        return True if aug_type in aug_args else False

    if 'pseudo_weak_sup' in config and config['pseudo_weak_sup']:
        config['checkpoint'] += '_pseudo_weak'
        if 'pseudo_weak_type' in config and config['pseudo_weak_type'] != 'seed-match':
            config['checkpoint'] += '_%s' % config['pseudo_weak_type']
        if 'pseudo_weak_sup_select' in config and config['pseudo_weak_sup_select']:
            # if 'bootstrap' in config and config['bootstrap'] and config['bootstrap_n_iteration']:
            config['checkpoint'] += '_selected_th=%g_type=%s' % (config['pseudo_weak_sup_select_threshold'], config['pseudo_weak_sup_select_threshold_type'])
            if 'pseudo_weak_sup_select_class_balance' in config and (not config['pseudo_weak_sup_select_class_balance']):
                config['checkpoint'] += '_nonbalanced'
            if 'pseudo_weak_sup_select_cross_fitting' in config and config['pseudo_weak_sup_select_cross_fitting']:
                config['checkpoint'] += '_crossfit=%i' % config['pseudo_weak_sup_select_cross_fitting_n_splits']
            if 'pseudo_weak_sup_select_metric' in config and config['pseudo_weak_sup_select_metric'] != 'confidence':
                config['checkpoint'] += '_metric=%s' % config['pseudo_weak_sup_select_metric']
            if 'pseudo_weak_sup_select_train_epochs' in config and config['pseudo_weak_sup_select_train_epochs'] != 1:
                config['checkpoint'] += '_epoch=%g' % config['pseudo_weak_sup_select_train_epochs']
            if 'pseudo_weak_sup_select_channel_dropout' in config and config['pseudo_weak_sup_select_channel_dropout']:
                config['checkpoint'] += '_chdropout=%g' % config['pseudo_weak_sup_select_channel_dropout']
                config['checkpoint'] += '_%s' % config['pseudo_weak_sup_select_channel_dropout_type']
            if 'pseudo_weak_sup_select_aug' in config and config['pseudo_weak_sup_select_aug']:
                raise NotImplementedError('pseudo_weak_sup_select_aug is deprecated. Please use pseudo_weak_sup_select_train_aug and pseudo_weak_sup_select_eval_aug instead.')
            if (not isinstance(config['pseudo_weak_sup_select_train_aug'], list)) and (not isinstance(config['pseudo_weak_sup_select_eval_aug'], list)):
                if config['pseudo_weak_sup_select_train_aug'] == config['pseudo_weak_sup_select_eval_aug'] or config['pseudo_weak_sup_select_eval_aug'] == 'same_as_train':
                    if config['pseudo_weak_sup_select_train_aug'] == 'none':
                        pass
                    else:
                        config['checkpoint'] += '_aug=%s' % config['pseudo_weak_sup_select_train_aug']
                else:
                    config['checkpoint'] += '_aug_train=%s' % config['pseudo_weak_sup_select_train_aug']
                    config['checkpoint'] += '_eval=%s' % config['pseudo_weak_sup_select_eval_aug']
            else:
                # Multiple confidence
                if isinstance(config['pseudo_weak_sup_select_train_aug'], list) and config['pseudo_weak_sup_select_eval_aug'] == 'same_as_train':
                    config['checkpoint'] += '_aug=%s' % '+'.join(config['pseudo_weak_sup_select_train_aug'])
                else:
                    if isinstance(config['pseudo_weak_sup_select_train_aug'], list):
                        config['checkpoint'] += '_aug_train=%s' % '+'.join(config['pseudo_weak_sup_select_train_aug'])
                    else:
                        config['checkpoint'] += '_aug_train=%s' % config['pseudo_weak_sup_select_train_aug']
                    if isinstance(config['pseudo_weak_sup_select_eval_aug'], list):
                        config['checkpoint'] += '_eval=%s' % '+'.join(config['pseudo_weak_sup_select_eval_aug'])
                    else:
                        config['checkpoint'] += '_eval=%s' % config['pseudo_weak_sup_select_eval_aug']
                # Multiple confidence, needs to be aggregated
                if config['pseudo_weak_sup_select_confidence_agg'] != 'average':
                    config['checkpoint'] += '_conf_agg=%s' % config['pseudo_weak_sup_select_confidence_agg']
                if 'pseudo_weak_sup_select_metric_after_agg' in config and config['pseudo_weak_sup_select_metric_after_agg']:
                    config['checkpoint'] += '_conf_after_agg'

            if is_aug_exist(config['pseudo_weak_sup_select_train_aug'], 'randremove') or \
               is_aug_exist(config['pseudo_weak_sup_select_eval_aug'], 'randremove') or \
               is_aug_exist(config['pseudo_weak_sup_select_train_aug'], 'randadremove') or \
               is_aug_exist(config['pseudo_weak_sup_select_eval_aug'], 'randadremove') or \
               is_aug_exist(config['pseudo_weak_sup_select_train_aug'], 'randnotadremove') or \
               is_aug_exist(config['pseudo_weak_sup_select_eval_aug'], 'randnotadremove'):
                if 'pseudo_weak_sup_select_aug_randremove_ratio' in config and config['pseudo_weak_sup_select_aug_randremove_ratio'] != 0.1:
                    config['checkpoint'] += '_r=%g' % config['pseudo_weak_sup_select_aug_randremove_ratio']

            if is_aug_exist(config['pseudo_weak_sup_select_train_aug'], 'randrandadremove') or \
               is_aug_exist(config['pseudo_weak_sup_select_eval_aug'], 'randrandadremove'):
                if 'pseudo_weak_sup_select_aug_randremove_ratio' in config and config['pseudo_weak_sup_select_aug_randremove_ratio'] != 0.1:
                    config['checkpoint'] += '_r=%g' % config['pseudo_weak_sup_select_aug_randremove_ratio']
                if 'pseudo_weak_sup_select_aug_randadremove_ratio' in config and config['pseudo_weak_sup_select_aug_randadremove_ratio'] != 0.1:
                    config['checkpoint'] += '_rad=%g' % config['pseudo_weak_sup_select_aug_randadremove_ratio']

            if is_aug_exist(config['pseudo_weak_sup_select_train_aug'], 'mlmreplace') or \
               is_aug_exist(config['pseudo_weak_sup_select_eval_aug'], 'mlmreplace'):
                if 'pseudo_weak_sup_select_aug_mlmreplace_ratio' in config and config['pseudo_weak_sup_select_aug_mlmreplace_ratio'] != 0.1:
                    config['checkpoint'] += '_r=%g' % config['pseudo_weak_sup_select_aug_mlmreplace_ratio']

            if is_aug_exist(config['pseudo_weak_sup_select_train_aug'], 'paraphrase') or \
               is_aug_exist(config['pseudo_weak_sup_select_eval_aug'], 'paraphrase'):
                if 'pseudo_weak_sup_select_aug_paraphrase_temperature' in config and config['pseudo_weak_sup_select_aug_paraphrase_temperature'] != 2:
                    config['checkpoint'] += '_T=%g' % config['pseudo_weak_sup_select_aug_paraphrase_temperature']

            if 'pseudo_weak_sup_select_mc_dropout' in config and config['pseudo_weak_sup_select_mc_dropout']:
                config['checkpoint'] += '_dropout'
                if 'pseudo_weak_sup_select_mc_dropout_passes' in config and config['pseudo_weak_sup_select_mc_dropout_passes'] != 10:
                    config['checkpoint'] += '_%d' % config['pseudo_weak_sup_select_mc_dropout_passes']
            if 'pseudo_weak_sup_select_use_prediction' in config and config['pseudo_weak_sup_select_use_prediction']:
                config['checkpoint'] += '_use_prob_pred'

    if 'pseudo_model_path' in config and config['pseudo_model_path'] is not None:
        teacher_spec = re.findall(r'sgd_(.+)_lr', config['pseudo_model_path'])[0]
        config['checkpoint'] += '_pseudo_path=%s' % teacher_spec
        config['checkpoint'] += '_th=%g_type=%s' % (config['pseudo_threshold'], config['pseudo_threshold_type'])
        if 'pseudo_class_balance' in config and (not config['pseudo_class_balance']):
            config['checkpoint'] += '_nonbalanced'
        ths = re.findall(r'th=(\d\.\d+)', config['pseudo_model_path'])
        if ths:
            # too many bootstrapping rounds, use abbr, otherwise filename too long
            config['checkpoint'] += '_model=pseudolabel_th=%s' % '-'.join(ths)

    if config['suffix']:
        config['checkpoint'] += '_%s' % config['suffix']
    del config['suffix']

    if 'trial' in config and config['trial']:
        config['checkpoint'] += '-%i' % config['trial']
        del config['trial']

    if config['test']:
        config['checkpoint'] = 'test_' + config['checkpoint']
        config['encoding_max_length'] = 64 # speed up
        config['batch_size'] = 32
        config['update_freq'] = 1
    del config['test']

    path = os.path.join(config['checkpoint_dir'], config['checkpoint'])
    ## path is an unique identifier for the experiment
    ## Therefore we check path across all servers
    print('local: ', path)
    path = check_path(path, config)
    _, checkpoint = os.path.split(path)
    config['checkpoint'] = checkpoint

    if config['resume']:
        config['resume_checkpoint'] = 'checkpoint.pth.tar'
        assert(os.path.isfile(os.path.join(path, config['resume_checkpoint']))), 'checkpoint %s not exists!' % config['resume_checkpoint']

    print("\n--------------------------- %s ----------------------------------" % config_file)
    for k, v in config.items():
        print('%s:'%k, v, type(v))
    print("---------------------------------------------------------------------\n")

    return config

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='sequence-classification')
    parser.add_argument('--config', '-c', default='config.yaml', type=str, metavar='C', help='config file')
    parser.add_argument('--remote', action='store_true', help='run on remote server?')
    parser.add_argument('--server', '-s', default=None, type=str, help='remote server name')
    # parser.add_argument('--remote_dir', default='/data', type=str, help='remote working dir')
    parser.add_argument('--queue', action='store_true', help='queueing jobs?')
    args = parser.parse_args()

    config = read_config(args.config, args.remote, args.server) # , args.remote_dir)
    with open('tmp/para.json', 'w') as f:
        json.dump(config, f)

    # reveal the path to bash
    with open('tmp/path.tmp', 'w') as f:
        f.write(config['checkpoint'])


    
