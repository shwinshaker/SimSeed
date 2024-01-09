#!./env python

import torch

import numpy as np
import random
import os

from collections import Counter
from collections.abc import Iterable
from ..utils import print, save_array
import warnings

def find_indices(b, a):
    """
        Find the indices of all elements from array 'a' within array 'b', in the order of their appearance in array 'a'
    """
    sorter = np.argsort(b)
    return sorter[np.searchsorted(b, a, sorter=sorter)]

def add_label(dataset, ids=None, labels=None):
    for ei, i in enumerate(ids):
        dataset.labels[i] = labels[ei]

## label noise
def rand_target_exclude(rng, classes, target):
    classes_ = [c for c in classes if c != target]
    return rng.choice(classes_)

def add_label_noise(dataset, ids=None, ratio=None, num_classes=10, note=None, seed=None):
    classes = list(range(num_classes))
    if seed is not None:
        rng = np.random.default_rng(seed) # fixed seed
    else:
        rng = np.random.default_rng()
    if ids is None:
        assert(ratio is not None)
        ids = rng.choice(len(dataset), int(ratio * len(dataset)), replace=False)
        save_path = 'id_label_noise_rand_ratio=%g' % ratio
        if note is not None:
            save_path += '_%s' % note
        save_path += '.npy'
        save_array(save_path, ids)
        
    for i in ids:
        dataset.labels[i] = rand_target_exclude(rng, classes, dataset.labels[i])

def add_label_noise_confusion_matrix(dataset, ids=None, matrix_path=None, classes=None, seed=None):
    confusion_matrix = np.load(matrix_path)
    print('confusion_matrix: ', confusion_matrix)
    if seed is not None:
        rng = np.random.default_rng(seed) # fixed seed
    else:
        rng = np.random.default_rng()
    if ids is None:
        warnings.warn('All ids used for label noise generation by a confusion matrix!')
        # add label noise to all data
        ids = np.arange(len(dataset))
        
    class_to_index = {c: i for i, c in enumerate(classes)}
    for i in ids:
        dataset.labels[i] = rng.choice(classes, p=confusion_matrix[class_to_index[dataset.labels[i].item()]])


def class_balanced_choice(rng, labels, n_choice, classes=None):
    if classes is None:
        classes = np.arange(len(labels.unique()))
    # if n_choice % n_class != 0:
    #     warnings.warn('sample size %i is not divisible by the number of classes %i!' % (n_choice, n_class))

    unique_labels, counts = labels.unique(return_counts=True)
    n_choice_per_class = (counts.numpy() / len(labels) * n_choice).round().astype(int)
    n_choice_per_class = dict(zip(unique_labels.numpy(), n_choice_per_class))
    idx_choice = []
    for c in classes:
        idx = torch.where(labels == c)[0].numpy()
        rng.shuffle(idx)
        idx_choice.extend(idx[:n_choice_per_class[c]])
    return np.array(idx_choice)

def random_split(dataset, size1, seed=None, class_balanced=True, classes=None):
    if seed is not None:
        rng = np.random.default_rng(seed) # fixed seed
    else:
        rng = np.random.default_rng()
    if class_balanced:
        ids1 = class_balanced_choice(rng, dataset.labels, size1, classes=classes)
    else:
        ids1 = rng.choice(len(dataset), size1, replace=False)
    ids2 = np.setdiff1d(np.arange(len(dataset)), ids1)
    return ids1, ids2

def kfold_random_split(size, n_splits, select_split=None, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed) # fixed seed
    else:
        rng = np.random.default_rng()
    idx = np.arange(size)
    rng.shuffle(idx)
    idx_splits = np.array_split(idx, n_splits)
    if select_split is None:
        return idx_splits
    idx_select = idx_splits[select_split]
    idx_others = np.concatenate([idx_splits[i] for i in range(n_splits) if i != select_split])
    return idx_select, idx_others

def get_subset(dataset, ids):
    assert(isinstance(ids, np.ndarray))
    dataset_ = torch.utils.data.Subset(dataset, ids)
    dataset_.labels = dataset.labels[ids]
    dataset_.classes = dataset.classes
    dataset_ .label_decode = dataset.label_decode
    dataset_.tokenizer = dataset.tokenizer
    dataset_.device = dataset.device
    return dataset_

class EncodingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, tokenizer=None, label_decode=None, encode=None, classes=None, device=None):
        self.encodings = encodings
        if labels is None:
            any_key = list(self.encodings.keys())[0]
            self.labels = torch.empty(len(self.encodings[any_key]), dtype=torch.int64)
        else:
            self.labels = labels
        self.device = device
        
        self.tokenizer = tokenizer
        self.label_decode = label_decode
        if classes is None:
            self.classes = labels.unique().tolist()
        else:
            self.classes = classes

    def __getitem__(self, idx):
        inputs = {key: val[idx].to(self.device) for key, val in self.encodings.items()}
        labels = self.labels[idx].to(self.device)
        return inputs, labels

    def __len__(self):
        return len(self.labels)


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, weights=None):
        assert(isinstance(dataset, torch.utils.data.Dataset))
        if weights is None:
            self.weights = {}
        else:
            for key in weights:
                assert(len(weights[key]) == len(dataset)), (key, len(weights[key]), len(dataset))
            self.weights = weights
        self.dataset = dataset

        # save attributes
        self.labels = dataset.labels
        self.classes = dataset.classes
        self.tokenizer = dataset.tokenizer
        self.label_decode = dataset.label_decode
        self.device = dataset.device

    def __getitem__(self, index):
        data, target = self.dataset[index]
        weight = dict([(key, self.weights[key][index]) for key in self.weights])
        weight['index'] = index

        return data, target, weight

    def __len__(self):
        return len(self.dataset)



def summary(dataset, show_examples=True):
    print('---------- Basic info ----------------')
    print('dataset size: %i' % len(dataset))
    print('input shape: ', dataset[0][0]['input_ids'].size())
    print('num classes: %i' % len(dataset.classes))
    print('---------- Frequency count -----------------')
    if len(dataset[0]) == 2:
        unique_labels, counts = np.unique([label.item() for _, label in dataset], return_counts=True)
        # d = dict(Counter([label.item() for _, label in dataset]).most_common())
    else:
        unique_labels, counts = np.unique([label.item() for _, label, _ in dataset], return_counts=True)
        # d = dict(Counter([label.item() for _, label, _ in dataset]).most_common())
    counts_dict = dict(zip(unique_labels, counts))
    for c in dataset.classes:
        if c not in counts_dict:
            counts_dict[c] = 0
    print(counts_dict)
    if show_examples:
        print('---------- Example Decoding-----------------')
        rand_idx = np.random.choice(len(dataset), 2) # 5
        for idx in rand_idx:
            input_, label_, _ = dataset[idx]
            print(' ----------------------------------------------')
            print(dataset.label_decode(label_.item()))
            print(dataset.tokenizer.input_decode(input_))