import torch
import numpy as np
import os
import warnings

import re
from ..utils import print, safe_divide_numpy

def rank(arr):
    return arr.argsort().argsort()

def pseudo_label_selection(tar, classes, threshold, threshold_type, class_balance, save_dir=None, verbose=True):

    # compose (multiple) confidence sources
    print('\n==> Pseudo-label selection..')
    if isinstance(tar, list):
        raise RuntimeError('Multiple confidence sources are deprecated!')

    # get selected
    eids = get_ids(tar['confidence'],
                   threshold=threshold,
                   threshold_type=threshold_type,
                   class_balance=class_balance,
                   classes=classes, labels=tar['pseudo_label'], verbose=verbose)
    assert(len(np.unique(eids)) == len(eids))

    # log
    selected_tar = {'index': tar['index'][eids],
                    'pseudo_label': tar['pseudo_label'][eids],
                    'true_label': tar['true_label'][eids],
                    'confidence': tar['confidence'][eids],
                    'model_path': tar['model_path'],
                    # 'path_unlabeled_idx': tar['path_unlabeled_idx'],
                    'path_name': tar['path_name'],
                    'class_balanced': class_balance,
                    'threshold': threshold,
                    'threshold_type': threshold_type,
                   }

    if verbose:
        print('[SSL] ----------------- Pseudo-labeling Info -----------------')
        print('[SSL] Pseudo-labeled subset size: %i' % len(selected_tar['index']))
        print('[SSL] Class balanced: %s' % selected_tar['class_balanced'])
        print('[SSL] Class count (True label): ', [(i, c) for i, c in zip(*np.unique(selected_tar['true_label'], return_counts=True))])
        print('[SSL] Class count (Pseudo label): ', [(i, c) for i, c in zip(*np.unique(selected_tar['pseudo_label'], return_counts=True))])
        print('[SSL] Model path: %s' % selected_tar['model_path'])
        # print('[SSL] Unlabeled index path: %s' % selected_tar['path_unlabeled_idx'])
        print('[SSL] Unlabeled path name: %s' % selected_tar['path_name'])
        print('[SSL] Threshold: %g  Type: %s' % (selected_tar['threshold'], selected_tar['threshold_type']))
        print('[SSL] Minimum confidence: %g Maximum confidence: %g' % (np.min(selected_tar['confidence']), np.max(selected_tar['confidence'])))
        noise_ratio = 1 - (selected_tar['pseudo_label'] == selected_tar['true_label']).sum() / len(selected_tar['true_label'])
        print('[SSL] Noise ratio: %.2f%%' % (noise_ratio * 100))
        # coverage = len(selected_tar['index']) / len(np.load(selected_tar['path_unlabeled_idx']))
        coverage = len(selected_tar['index']) / len(tar['index'])
        print('[SSL] Coverage: %.2f%%' % (coverage * 100))
        print('[SSL] --------------------------------------------------------')

    # save
    if save_dir is not None:
        # save_path = '%s/pseudo_selected_unlabeled=%s.pt' % (save_dir, os.path.split(selected_tar['path_unlabeled_idx'])[1].replace('.npy', ''))
        save_path = '%s/pseudo_selected_unlabeled=%s.pt' % (save_dir, selected_tar['path_name'])
        torch.save(selected_tar, save_path)
    return selected_tar


def get_top_k_max(confidence, k=0):
    indices = confidence.argsort()[::-1]
    return confidence[indices[k]]
    
def warn_none_count(counts, classes, count_type=None):
    none_classes = ', '.join(np.array(classes)[np.argwhere(counts == 0).ravel()].astype(str))
    if count_type == 'selected':
        warnings.warn(f'Warning! None in pseudo-labels selected for class {none_classes}!')
    else:
        warnings.warn(f'Warning! None in pseudo-labels found for class {none_classes}!')

def get_ids(confidence, threshold, threshold_type='percentile', class_balance=True, classes=None, labels=None, min_select_per_class=1, verbose=True):
    
    assert(threshold_type in ['percentile', 'value']), 'Threshold type %s not recognized!' % threshold_type
    assert(threshold >= 0 and threshold <= 1), 'Invalid threshold value: %.4f' % threshold

    confidence = np.array(confidence)
    if not class_balance:
        if threshold_type == 'percentile':
            indices = confidence.argsort()[::-1]
            return indices[:round(threshold * len(indices))]
        elif threshold_type == 'value':
            return np.where(confidence >= threshold)[0]

    labels = np.array(labels)
    # Determine the selection size for each class
    if threshold_type == 'percentile':
        # - Selection based on the pseudo-labels distribution, better for imbalanced dataset
        counts = np.array([(labels == label).sum() for label in classes])
        if np.any(counts == 0):
            warn_none_count(counts, classes)
        if verbose:
            print('Distribution of all pseudo-labels: ', dict(zip(classes, counts)))
        class_size = (threshold * counts).round().astype(int)
        class_size = dict(zip(classes, class_size))

    elif threshold_type == 'value':
        counts = np.array([(labels == label).sum() for label in classes])
        if np.any(counts == 0):
            warn_none_count(counts, classes)
        counts_conf = np.array([(labels[confidence >= threshold] == label).sum() for label in classes])
        if np.any(counts_conf == 0):
            warn_none_count(counts_conf, classes, count_type='selected')
        if verbose:
            print('Distribution of all pseudo-labels: ', dict(zip(classes, counts)))
            print('Distribution of pseudo-labels with confidence >= %.2f: ' % threshold, dict(zip(classes, counts_conf)))
        # -- Select the minimum available ratio of pseudo-labels across all classes
        #      Here our assumption is the distribution of pseudo-labels can reasonably approximate the distribution of true labels
        #      if pseudo-labels are strictly balanced then this degenerates to selecting the minium number of availabel labels in each classes
        counts_conf = np.where(counts_conf == 0, min_select_per_class, counts_conf) 
        # - at least select `min_select_per_class` in terms of ratio,
        #    because simply selecting `min_select_per_class` may will cause `min_select_per_class` selected in all classes 
        min_ratio = np.min(safe_divide_numpy(counts_conf, counts))
        class_size = (min_ratio * counts).round().astype(int)
        class_size = dict(zip(classes, class_size))

    # perform selection
    indices = confidence.argsort()[::-1]
    labels = labels[indices]
    ids_ = []
    for label in classes:
        ids_.append(indices[labels == label][:max(class_size[label], min_select_per_class)])
        # if verbose:
            # print('PL selection: ', label, class_size[label], len(indices[labels == label][:max(class_size[label], min_select_per_class)]))
    return np.concatenate(ids_)

        
        
