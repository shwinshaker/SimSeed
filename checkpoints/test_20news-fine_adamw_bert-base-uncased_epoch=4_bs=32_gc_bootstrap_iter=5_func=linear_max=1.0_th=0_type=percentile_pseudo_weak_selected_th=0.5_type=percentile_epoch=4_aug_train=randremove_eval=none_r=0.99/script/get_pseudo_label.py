
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from src.models import get_model
from src.preprocess import get_loaders
from src.utils import accuracy, Dict2Obj, get_tensor_max_except

from transformers import BertForSequenceClassification
import warnings

import sys
def print(*objects, sep=' ', end='\n', file=sys.stdout, flush=True, verbose=True):
    if verbose:
        import builtins
        # flushed print
        builtins.print(*objects, sep=sep, end=end, file=file, flush=flush)
    else:
        pass

# Reference: https://stackoverflow.com/questions/63285197/measuring-uncertainty-using-mc-dropout-on-pytorch
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_aug_inputs(inputs, aug_type=None, labels=None, augmentor=None, config=None):
    if aug_type == 'adremove':
        inputs, _ = augmentor.remove_seed_words_batch(inputs, labels)
    elif aug_type == 'randremove':
        inputs = augmentor.remove_random_words_batch(inputs, config.randremove_num)
    elif aug_type == 'randadremove':
        inputs, _ = augmentor.remove_seed_words_batch(inputs, labels)
        inputs = augmentor.remove_random_words_batch(inputs, config.randremove_num)
    elif aug_type == 'randnotadremove':
        inputs = augmentor.remove_random_words_except_seed_words_batch(inputs, labels, config.randremove_num)
    elif aug_type == 'randrandadremove':
        # old_inputs = inputs
        inputs = augmentor.remove_random_seed_words_batch(inputs, labels, config.randadremove_num)
        inputs = augmentor.remove_random_words_except_seed_words_batch(inputs, labels, config.randremove_num)
        # augmentor.check_aug(old_inputs, inputs, labels=labels)
    elif aug_type == 'paraphrase':
        inputs = augmentor.paraphrase_batch(inputs, config.augparaphrase_temperature)
    elif aug_type == 'mlmreplace':
        # old_inputs = inputs
        inputs = augmentor.replace_random_words_with_mlm_batch(inputs, config.randremove_num)
        # augmentor.check_aug(old_inputs, inputs, labels=labels)
    return inputs

def get_softmax(model, inputs, dropout=False, n_passes=10, reduction=True):
    if not dropout:
        return F.softmax(model(**inputs).logits, dim=1).detach().cpu()

    print('> ------ dropout')
    model.eval()
    enable_dropout(model)
    dropout_softmaxs = []
    for _ in range(n_passes):
        with torch.no_grad():
            dropout_softmaxs.append(F.softmax(model(**inputs).logits, dim=1).detach().cpu())
    dropout_softmaxs = torch.stack(dropout_softmaxs)
    model.eval()

    if reduction:
        return dropout_softmaxs.mean(axis=0) #, dropout_softmaxs.std(axis=0)
    return dropout_softmaxs

def get_confidence(inputs, outputs, labels=None, model=None, augmentor=None, config=None):
    
    if config.conf_type == 'confidence':
        inputs = get_aug_inputs(inputs, aug_type=config.aug_type, labels=labels, augmentor=augmentor, config=config)
        softmaxs = get_softmax(model, inputs, dropout=config.dropout, n_passes=config.dropout_passes)
        if labels is not None:
            return softmaxs.gather(1, labels.view(-1, 1)).squeeze().numpy()
        else:
            return softmaxs.max(1)[0].numpy()

    elif config.conf_type == 'dropout_stability':
        inputs = get_aug_inputs(inputs, aug_type=config.aug_type, labels=labels, augmentor=augmentor, config=config)
        softmaxs = get_softmax(model, inputs, dropout=config.dropout, reduction=False, n_passes=config.dropout_passes)

        dropout_mean, dropout_std = dropout_softmaxs.mean(axis=0), dropout_softmaxs.std(axis=0)
        if labels is None:
            _, labels = dropout_mean.max(dim=1)
        return dropout_std.gather(1, labels.view(-1, 1)).squeeze().numpy()

    elif config.conf_type == 'seed_stability' or config.conf_type == 'seed_stability_label':
        if config.conf_type == 'seed_stability_label':
            warnings.warn('conf_type seed_stability_label is deprecated. Now label-based stability will be used by default when label is not None.')
        softmaxs = get_softmax(model, inputs, dropout=config.dropout, n_passes=config.dropout_passes)
        inputs_ad = get_aug_inputs(inputs, aug_type='adremove', labels=labels, augmentor=augmentor, config=config)
        softmaxs_ad = get_softmax(model, inputs_ad, dropout=config.dropout, n_passes=config.dropout_passes)
        if labels is not None:
            # -- reason: regular 1d distance (abs)
            # return -1 * (softmaxs_ad.gather(1, labels.view(-1, 1)) - softmaxs.gather(1, labels.view(-1, 1))).abs().squeeze().numpy()
            # -- reason: probability at label decrease more, meaning more likely a wrong label
            return (softmaxs_ad.gather(1, labels.view(-1, 1)) - softmaxs.gather(1, labels.view(-1, 1))).squeeze().numpy()
        else:
            return -1 * nn.MSELoss(reduction='none')(softmaxs_ad, softmaxs).sum(dim=1)

    elif config.conf_type == 'margin':
        inputs = get_aug_inputs(inputs, aug_type=config.aug_type, labels=labels, augmentor=augmentor, config=config)
        softmaxs = get_softmax(model, inputs, dropout=config.dropout, n_passes=config.dropout_passes)
        if labels is not None:
            return (softmaxs.gather(1, labels.view(-1, 1)).squeeze() - get_tensor_max_except(softmaxs, labels)[0]).numpy()
        else:
            return (softmaxs.max(1)[0] - get_tensor_max_except(softmaxs, softmaxs.max(1)[1])[0]).numpy()

    elif config.conf_type == 'softmax':
        inputs = get_aug_inputs(inputs, aug_type=config.aug_type, labels=labels, augmentor=augmentor, config=config)
        softmaxs = get_softmax(model, inputs, dropout=config.dropout, n_passes=config.dropout_passes)
        return softmaxs.numpy()

    elif config.conf_type == 'entropy':
        raise NotImplementedError
        # assert(labels is None), f'metric [{metric}] not applicable in this case!'
        # epsilon = sys.float_info.min
        # return - (dropout_mean*np.log(dropout_mean + epsilon)).sum(axis=-1).numpy()

    raise KeyError(config.conf_type)


def get_pseudo_labels(net, loader, pseudo_label=None, true_label=None, augmentor=None, config=None):
    # pseudo_label: if pseudo-labels are already provided, return confidence associated with these labels 
    ids_list = []
    preds_list = []
    confs_list = []
    labels_list = []
    mis_ids_list = []

    # get batch size
    for inputs, labels, weights in loader:
        batch_size = len(labels)
        break
    
    if config.conf_type == 'norm_loss_avg':
        norm_loss_avg = np.load('./norm_loss_avg.npy')

    n_correct = 0
    n_total = 0
    net.eval()
    for e, (inputs, labels, weights) in enumerate(loader):
        with torch.no_grad():
            outputs = net(**inputs)
            acc, = accuracy(outputs.logits.data, labels.data)
        
            preds = outputs.logits.max(1)[1]
            preds_list.append(preds.cpu().numpy())

            pseudo_label_batch = None
            if pseudo_label is not None:
                batch_idx = np.arange(len(labels)) + e * batch_size
                assert(np.all(labels.cpu().numpy() == true_label[batch_idx])) # sanity check
                pseudo_label_batch = torch.from_numpy(pseudo_label[batch_idx])
            if config.conf_type == 'norm_loss_avg':
                confs_list.append(norm_loss_avg[weights['index'].cpu().numpy()])
            else:
                confs_list.append(get_confidence(inputs, outputs, labels=pseudo_label_batch, model=net, augmentor=augmentor, config=config))

            ids = weights['index'].to(config.device)
            ids_list.append(ids.cpu().numpy())
            mis_ids_list.append(ids[~preds.squeeze().eq(labels)].cpu().numpy())
            labels_list.append(labels.cpu())
        
            n_correct += (preds.squeeze().eq(labels)).sum()
            n_total += labels.size(0)
            
        # print('----------- [%i/%i] --- # correct: %i -- Acc (Batch): %.3f ------' % \
            #   (e, len(loader), n_correct, acc.item()), end='\r')

    print()
    print('----------- # correct: %i -- Acc: %.3f -- Noise: %.3f ---' % (n_correct, n_correct/n_total, 1-n_correct/n_total))
    
    results = {'pseudo_label': np.hstack(preds_list) if config.use_prediction or pseudo_label is None else pseudo_label,
               'true_label': np.hstack(labels_list),
               'confidence': np.concatenate(confs_list), # Use `concatenate` instead of `hstack` to allow stacking of both 1d and 2d arrays
               'index': np.hstack(ids_list),
               'mis_index': np.hstack(mis_ids_list),
              }    
    return results


def pseudo_label(config, dataset, path_unlabeled_idx, model_path, gpu_id, model_state='model.pt',
                 save_dir='.', data_dir='/home/chengyu/bert_classification/data', epoch=None, pseudo_label=None, true_label=None,
                 conf_type='confidence', aug_type=None, path_name=None, use_prediction=False, dropout=False, dropout_passes=10,
                 randremove_num=None, randadremove_num=None, augparaphrase_temperature=None):

    # --- massage hypers
    if aug_type == 'none':
        aug_type = None

    assert(epoch is None), 'Not implemented yet! Will interfere with confidence aggregation.'
    assert(type(path_unlabeled_idx) in [str, np.ndarray])

    if path_name is None:
        assert(isinstance(path_unlabeled_idx, str))
        path_name = os.path.split(path_unlabeled_idx)[1].replace('.npy', '')

    print('---- Set environment..')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('---- Get unlabeled dataloader..')
    # Do we need random augment when pseudo-labeling?
    # -- Not used in `In defense of pseudo-labeling...`
    print(' --- dataset: %s' % dataset)
    print(' --- path name: %s' % path_name)
    loaders_unlabeled = get_loaders(dataset=dataset, batch_size=128,
                                    train_subset_path=path_unlabeled_idx,
                                    shuffle_train_loader=False,
                                    data_dir=data_dir,
                                    show_examples=False,
                                    config=Dict2Obj({'device': device,
                                                     'model': config.model,
                                                     'encoding_max_length': config.encoding_max_length,
                                                     'class_balanced_sampling': config.class_balanced_sampling,
                                                     'augparaphrase': True if aug_type == 'paraphrase' else False, 
                                                     'mlmreplace': True if aug_type == 'mlmreplace' else False,
                                                     'save_dir': save_dir}))


    print('---- Get model..')
    print(' --- model path: %s' % model_path)
    print(' --- model: %s' % (config.model))
    print(' --- model state: %s' % model_state)
    model = get_model(config, loaders_unlabeled)
    # model = BertForSequenceClassification.from_pretrained(
    #     config.model,
    #     num_labels=loaders_unlabeled.num_classes,  # The number of output labels -- 2 for binary classification.
    #     output_attentions=False,  # Whether the model returns attentions weights.
    #     output_hidden_states=False,  # Whether the model returns all hidden-states.
    # )
    model.to(device)
    state_dict = torch.load('%s/%s' % (model_path, model_state), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f'---- Get predictions (confidence={conf_type}, augmentation={aug_type}, dropout={dropout} (n_passes={dropout_passes}))..')
    results = get_pseudo_labels(model, loaders_unlabeled.trainloader,
                                pseudo_label=pseudo_label, true_label=true_label, augmentor=loaders_unlabeled.augmentor,
                                config=Dict2Obj({'conf_type': conf_type,
                                                 'aug_type': aug_type,
                                                 'use_prediction': use_prediction,
                                                 'dropout': dropout,
                                                 'dropout_passes': dropout_passes,
                                                 'randremove_num': randremove_num,
                                                 'randadremove_num': randadremove_num,
                                                 'augparaphrase_temperature': augparaphrase_temperature,
                                                 'device': device,}))
    assert(np.all(results['true_label'] == np.array(loaders_unlabeled.trainset.labels)))
    # provided subset index and obtained index should be the same
    if isinstance(path_unlabeled_idx, str):
        assert(np.all(results['index'] == np.load(path_unlabeled_idx))) 
    else:
        assert(np.all(results['index'] == path_unlabeled_idx)) 

    print('---- Save pseudo labels..')
    print(' --- save path: %s' % save_dir)
    results['model_path'] = os.path.split(model_path)[1]
    results['model_state'] = model_state
    # results['path_unlabeled_idx'] = path_unlabeled_idx
    results['path_name'] = path_name

    if save_dir is not None:
        save_name = 'pseudo_unlabeled=%s' % path_name
        torch.save(results, '%s/%s.pt' % (save_dir, save_name))
    return results

if __name__ == '__main__':


    config = Dict2Obj({
        'model': 'bert-large-uncased',
        'dataset': 'imdb',
        'gpu_id': 6,
        'encoding_max_length': 64,
    })

    # model_path = 'checkpoints/adamw_bert-large-uncased_ntrain=100'
    model_path = 'checkpoints/adamw_bert-large-uncased_ntrain=1000'
    path_unlabeled_idx = '%s/id_unlabeled_imdb_size=41500.npy' % model_path

    pseudo_label(config, config.dataset, path_unlabeled_idx, model_path, gpu_id=config.gpu_id, save_dir=model_path)

    


