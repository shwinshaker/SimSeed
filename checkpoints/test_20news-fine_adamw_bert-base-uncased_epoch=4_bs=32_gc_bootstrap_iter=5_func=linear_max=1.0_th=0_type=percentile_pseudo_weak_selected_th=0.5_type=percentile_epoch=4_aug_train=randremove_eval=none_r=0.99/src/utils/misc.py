#!./env python
import torch
import numpy as np

def get_tensor_max_except(tensor, mask_idx):
    # tensor: row x column
    # mask_idx: column index at each row
    tensor = tensor.clone()
    tensor[torch.arange(tensor.size(0)).unsqueeze(1), mask_idx.unsqueeze(1)] = -torch.inf
    return tensor.max(1)

def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator

def safe_divide_numpy(a, b):
    a, b = a.astype(float), b.astype(float)
    return np.divide(a, b, out=np.zeros_like(b), where=b!=0)

def random_label_exclude(labels, label):
    return np.random.choice([c for c in labels if c != label])

def random_label_exclude_batch(labels, classes):
    return torch.tensor([random_label_exclude(classes, label) for label in labels], device=labels.device)

def shuffle_dict_of_tensors(inputs):
    key0 = list(inputs.keys())[0]
    length = inputs[key0].size(0)
    rand_idx = torch.randperm(length)
    return {key: inputs[key].detach().clone()[rand_idx] for key in inputs}
    
def nan_filter(arr):
    arr = np.array(arr)
    arr[np.isnan(arr)] = 0
    return arr

class Dict2Obj:
    """
        Turns a dictionary into a class
    """

    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])

    def get(self, key, default=None):
        return getattr(self, key, default)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def iterable(obj):
    """
        check if an object is iterable
    """
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True
    
def is_in(labels, classes):
    """
        return the element-wise belonging of a tensor to a list
    """
    assert(len(labels.size())) == 1
    if not classes:
        return torch.ones(labels.size(0), dtype=torch.bool).to(labels.device)
    
    if not iterable(classes):
        return labels == classes
    
    id_in = torch.zeros(labels.size(0), dtype=torch.bool).to(labels.device)
    for c in classes:
        id_in |= (labels == c)
    return id_in

def ravel_state_dict(state_dict):
    """ 
        state_dict: all variables, incluing parameters and buffers
    """
    li = []
    for _, paras in state_dict.items():
        li.append(paras.view(-1))
    return torch.cat(li)

# def ravel(paras_tup):
#     li = []
#     for paras in paras_tup:
#         li.append(paras.view(-1))
#     return torch.cat(li)

def ravel_parameters(para_dict):
    """
        parameters: learnable variables only
        para_dict = dict(model.named_parameters())
    """
    return torch.cat([p.detach().view(-1) for n, p in para_dict.items() if p.requires_grad])

def o_dedup(seq):
    """
        remove duplicates while preserving order
        https://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-whilst-preserving-order
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_distance_matrix(tensor):
    """
        poor precision for diagonal terms - otherwise same as sklearn.metrics.pairwise_distances
    """
    assert(len(tensor.size()) == 2)
    n = tensor.size()[0]
    ss = (tensor**2).sum(axis=1)
    return ss.view(n, -1) + ss.view(-1, n) - 2.0 * torch.matmul(tensor, tensor.T)

def get_equivalence_matrix(tensor):
    assert(len(tensor.size()) == 1)
    return (tensor.view(tensor.size(0), -1) == tensor)


class DeNormalizer:
    def __init__(self, mean, std, n_channel, device):
        self.mean = torch.Tensor(mean).view(1, n_channel, 1, 1).to(device)
        self.std = torch.Tensor(std).view(1, n_channel, 1, 1).to(device)

    def __call__(self, tensor):
        return tensor * self.std + self.mean

class Normalizer:
    def __init__(self, mean, std, n_channel, device):
        self.mean = torch.Tensor(mean).view(1, n_channel, 1, 1).to(device)
        self.std = torch.Tensor(std).view(1, n_channel, 1, 1).to(device)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std
