#!./env python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from collections import Counter

from . import get_dataset
from ..utils import Dict2Obj
from . import EncodingDataset, WeightedDataset, summary, class_balanced_choice, random_split, kfold_random_split
from . import add_label, add_label_noise, add_label_noise_confusion_matrix, get_subset
from .pseudo_label import pseudo_label_selection
from .weak_supervision import get_weak_supervision, get_weak_supervision_xclass, get_weak_supervision_prompt, get_weak_supervision_random_flip
from ..utils import print, save_array
import warnings
import os
import re

__all__ = ['get_loaders']

def add_trainsubids(trainsubids, ids):
    assert(not np.any(np.intersect1d(trainsubids, ids))), 'Pseudo-label indices intersected with labeled set.'
    return np.concatenate([trainsubids, ids])

def get_loaders(dataset='imdb', test_ratio=0.15, batch_size=16,
                shuffle_train_loader=True,
                trainsize=None, train_subset_path=None,
                data_dir='/home/chengyu/bert_classification/data',
                show_examples=False,
                config=None):

    data_dict = get_dataset(dataset, data_dir=data_dir, config=config)

    print('==> Label names')
    print(data_dict['label_names'])
    # save encode / decode setting for later use, need to be consistent with the dataset generation setup
    def _decode(inputs):
        return data_dict['tokenizer'].decode(inputs['input_ids'], skip_special_tokens=True)
    def _encode(inputs):
        return data_dict['tokenizer'](inputs,  # Sentence to encode.
                                      add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                      max_length=config.encoding_max_length,  #
                                      padding='max_length', # Pad short sentence
                                      truncation=True, # Truncate long sentence
                                      return_attention_mask=True,  # Construct attn. masks.
                                      return_tensors='pt')  # Return pytorch tensors.
    data_dict['tokenizer'].input_decode = _decode
    data_dict['tokenizer'].input_encode = _encode
    testset = EncodingDataset(data_dict['encodings'],
                              data_dict['labels'],
                              tokenizer=data_dict['tokenizer'],
                              label_decode=lambda label: data_dict['label_names'][label],
                              classes=data_dict['labels'].unique().tolist(),
                              device=config.device)

    trainset = EncodingDataset(data_dict['encodings'],
                               None,
                               tokenizer=data_dict['tokenizer'],
                               label_decode=lambda label: data_dict['label_names'][label],
                               classes=data_dict['labels'].unique().tolist(),
                               device=config.device)

    __train_size_orig = len(trainset) # for info

    # --- randomly select training subsets ------
    trainsubids = np.array([], dtype=int)
    if trainsize is not None:
        assert train_subset_path is None, 'selected based on ids is prohibited when size is enabled'
        assert trainsize < len(trainset), 'training set has only %i examples' % len(trainset)

        _trainsubids, _trainevalids = random_split(trainset, trainsize, seed=None, class_balanced=config.class_balanced_sampling, classes=allset.classes)
        # -- note that it is possible that  trainsize != len(trainsubids) because of rounding up
        save_array('id_train_%s_size=%i.npy' % (dataset, trainsize), _trainsubids, config=config)
        if hasattr(config, 'save_pseudo_label') and config.save_pseudo_label:
            save_array('id_unlabeled.npy', _trainevalids, config=config)

        add_label(trainset, ids=_trainsubids, labels=data_dict['labels'][_trainsubids])
        warnings.warn('True labels are added to the training set!!')
        trainsubids = add_trainsubids(trainsubids, _trainsubids)

    if train_subset_path is not None:
        assert trainsize is None, 'selected based on ids is prohibited when size is enabled'
        if isinstance(train_subset_path, str):
            with open(train_subset_path, 'rb') as f:
                _trainsubids = np.load(f)
        else:
            assert(isinstance(train_subset_path, np.ndarray))
            _trainsubids = train_subset_path
        if hasattr(config, 'save_pseudo_label') and config.save_pseudo_label:
            save_array('id_unlabeled.npy', np.setdiff1d(np.arange(len(trainset)), _trainsubids), config=config)

        add_label(trainset, ids=_trainsubids, labels=data_dict['labels'][_trainsubids])
        warnings.warn('True labels are added to the training set!!')
        trainsubids = add_trainsubids(trainsubids, _trainsubids)

    # -- inject label noise --
    if hasattr(config, 'trainnoisyratio') and config.trainnoisyratio:
        # assert(not (len(labelnoisyids) > 0)), 'double noise standards set!'
        seed = None
        if hasattr(config, 'noise_seed') and config.noise_seed:
            seed = config.noise_seed
        add_label_noise(trainset, ids=None, ratio=config.trainnoisyratio, num_classes=len(trainset.classes), seed=seed)

    if hasattr(config, 'trainnoisyconfusionmatrix') and config.trainnoisyconfusionmatrix:
        seed = None
        if hasattr(config, 'noise_seed') and config.noise_seed:
            seed = config.noise_seed
        add_label_noise_confusion_matrix(trainset, ids=trainsubids, matrix_path=config.trainnoisyconfusionmatrix, classes=trainset.classes, seed=seed)

    # --- pseudo label from weak supervision ---
    if hasattr(config, 'pseudo_weak_sup') and config.pseudo_weak_sup:
        print('\n==> Load weak pseudo-labels..')
        if hasattr(config, 'pseudo_weak_type') and 'x-class' in config.pseudo_weak_type:
            print('==> Using x-class..')
            tar = get_weak_supervision_xclass(dataset, data_dir=data_dir, method=config.pseudo_weak_type)
        elif hasattr(config, 'pseudo_weak_type') and config.pseudo_weak_type == 'prompt':
            print('==> Using prompt..')
            tar = get_weak_supervision_prompt(dataset, data_dir=data_dir)
        elif hasattr(config, 'pseudo_weak_type') and config.pseudo_weak_type == 'random-flip':
            print('==> Using synthesized random flipping noise..')
            tar = get_weak_supervision_random_flip(dataset, data_dir=data_dir)
        else:
            tar = get_weak_supervision(np.arange(len(trainset)), data_dir, dataset)
        if hasattr(config, 'pseudo_weak_sup_n_splits') and config.pseudo_weak_sup_n_splits:
            print(f'\n==> Load {config.pseudo_weak_sup_eval_split}-th split among {config.pseudo_weak_sup_n_splits} splits..')
            assert(not (hasattr(config, 'pseudo_weak_sup_select') and config.pseudo_weak_sup_select))
            sub_idx_eval, sub_idx_train = kfold_random_split(len(tar['index']),
                                                             config.pseudo_weak_sup_n_splits,
                                                             select_split=config.pseudo_weak_sup_eval_split,
                                                             seed=42)
            print('    Loaded train size: {}, pseudo-label eval size: {}'.format(len(sub_idx_train), len(sub_idx_eval)))
            
            # save_array(f'id_train_weak_sup_split={config.pseudo_weak_sup_eval_split}.npy',
                    #    tar['index'][sub_idx_eval], config=config)
            torch.save({'index': tar['index'][sub_idx_eval],
                        'pseudo_label': tar['pseudo_label'][sub_idx_eval],
                        'true_label': tar['true_label'][sub_idx_eval],},
                       f'{config.save_dir}/unlabeled=weak_sup_split={config.pseudo_weak_sup_eval_split}.pt')
            # save_array('id_unlabeled.npy', np.setdiff1d(np.arange(len(trainset)), tar['index']), config=config)
            # - if not report any error then neglect this
            add_label(trainset, ids=tar['index'][sub_idx_train], labels=tar['pseudo_label'][sub_idx_train])
            trainsubids = add_trainsubids(trainsubids, tar['index'][sub_idx_train])

        else:
            # save all ids having weak labels for later confidence evaluation, and then later selection of weak labels
            # save_array('id_train_weak_sup.npy',  tar['index'], config=config) 
            torch.save({'index': tar['index'],
                        'pseudo_label': tar['pseudo_label'],
                        'true_label': tar['true_label'],},
                       f'{config.save_dir}/unlabeled=weak_sup.pt')

            if hasattr(config, 'pseudo_weak_sup_select') and config.pseudo_weak_sup_select:
                # tar_path = '%s/pseudo_unlabeled=%s.pt' % (config.weak_model_path, 'id_train_weak_sup')
                print('\n==> Select weak pseudo-labels based on confidence..')
                # tar_names = get_files_regex_match(config.weak_model_path, regex_str='pseudo_unlabeled=weak_sup.*\.pt')
                # tar_paths = ['%s/%s' % (config.weak_model_path, tar_name) for tar_name in tar_names]
                # print('==> Found saved tar paths: ', tar_paths)
                # loaded_tars = [torch.load(tar_path) for tar_path in tar_paths]
                # -- fixed tar name. Multiple confidence sources should be aggregated before this.
                loaded_tar = torch.load(f'{config.weak_model_path}/pseudo_unlabeled=weak_sup.pt')
                assert(np.all(np.sort(loaded_tar['index']) == np.sort(tar['index']))), 'fatal, mismatched index!'
                assert(np.all(loaded_tar['index'] == tar['index'])), 'mismatched order, might because order of aggregated tar is changed'

                # assert(len(loaded_tars) == 1), 'multiple confidence is deprecated!'
                # assert(all([np.all(loaded_tar['index'] == tar['index']) for loaded_tar in loaded_tars]))

                tar = pseudo_label_selection(loaded_tar,
                                             classes=trainset.classes,
                                             threshold=config.pseudo_weak_sup_select_threshold,
                                             threshold_type=config.pseudo_weak_sup_select_threshold_type,
                                             class_balance=config.pseudo_weak_sup_select_class_balance,
                                             save_dir=config.save_dir)

            # assert(np.all(np.sort(saved_weak_sup_idx) == np.sort(tar['index']))), 'Weak sup ids not aligned!'
            # assert(np.all(np.isin(tar['index'], tar_weak['index']))), 'Weak sup ids in saved pseudo-labels not aligned!'
            save_array('id_unlabeled.npy', np.setdiff1d(np.arange(len(trainset)), tar['index']), config=config)

            add_label(trainset, ids=tar['index'], labels=tar['pseudo_label'])
            trainsubids = add_trainsubids(trainsubids, tar['index'])

    # --- pseudo label from model path ---
    if hasattr(config, 'pseudo_model_path') and config.pseudo_model_path is not None:
        if config.pseudo_unlabeled_idx_path is None:
            config.pseudo_unlabeled_idx_path = f"./id_unlabeled.npy" # use all the rest of training set as the unlabeled set
        print('\n==> Select pseudo-labels..')
        tar_path = '%s/pseudo_unlabeled=%s.pt' % (config.pseudo_model_path, os.path.split(config.pseudo_unlabeled_idx_path)[1].replace('.npy', ''))
        tar = pseudo_label_selection(torch.load(tar_path), classes=trainset.classes,
                                     threshold=config.pseudo_threshold,
                                     threshold_type=config.pseudo_threshold_type,
                                     class_balance=config.pseudo_class_balance,
                                     save_dir=config.save_dir)
        add_label(trainset, ids=tar['index'], labels=tar['pseudo_label'])
        trainsubids = add_trainsubids(trainsubids, tar['index'])

    # --- apply weights ---
    trainset = WeightedDataset(trainset)
    testset = WeightedDataset(testset)

    # --- apply training subsets ------
    if len(trainsubids) > 0:
        trainset = get_subset(trainset, trainsubids)
    else:
        raise RuntimeError('No train subids selected!')
        # warnings.warn('No train subids selected. Use all training data and true labels')

    # --- select test subsets ------
    if test_ratio is not None:
        rng = np.random.default_rng(42) # fixed seed
        testsubids = class_balanced_choice(rng, testset.labels, int(len(testset) * test_ratio), classes=testset.classes)
        testset = get_subset(testset, testsubids)

    # --- summary ---
    print('\n==> training set')
    summary(trainset, show_examples=show_examples)
    print('\n==> test set')
    summary(testset, show_examples=show_examples)

    # --- integrate ---
    loaders = {'trainset': trainset,
               'testset': testset,
               'trainloader': DataLoader(trainset, batch_size=batch_size, shuffle=shuffle_train_loader),
               'testloader': DataLoader(testset, batch_size=batch_size, shuffle=False)}
    loaders = Dict2Obj(loaders)
    loaders.classes = trainset.classes
    loaders.num_classes = len(loaders.classes)
    loaders.input_shape = trainset[0][0]['input_ids'].size()
    loaders.trainsubids = trainsubids
    loaders.train_size_orig = __train_size_orig

    # backdoor
    from .augmentation import Augmentor
    loaders.augmentor = Augmentor(data_dict['tokenizer'],
                                  list(data_dict['label_names'].keys()),
                                  dataset=dataset,
                                  data_dir=data_dir,
                                  label_decode=loaders.trainset.label_decode,
                                  device=loaders.trainset.device)
    if hasattr(config, 'augparaphrase') and config.augparaphrase:
        ## - setup paraphraser before training, because loading model is slow
        loaders.augmentor.setup_paraphrase(max_length=config.encoding_max_length)
    if hasattr(config, 'mlmreplace') and config.mlmreplace:
        loaders.augmentor.setup_mlm()

    return loaders


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = get_loaders(config=Dict2Obj({'device': device}))
