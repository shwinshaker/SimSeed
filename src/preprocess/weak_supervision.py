#!./env python
import pickle
import json
import os
import numpy as np
import torch
from ..utils import print

__all__ = ['get_weak_supervision', 'get_weak_supervision_xclass', 'get_weak_supervision_prompt', 'get_weak_supervision_random_flip']

# def get_noise_supervision():
# TODO

def print_pseudo_label_info(tar, train_size):
    print('[WSL] ----------------- Pseudo-labeling Info -----------------')
    print('[WSL] Pseudo-labeled subset size: %i' % len(tar['index']))
    print('[WSL] Class count (True label): ', [(i, c) for i, c in zip(*np.unique(tar['true_label'], return_counts=True))])
    print('[WSL] Class count (Pseudo label): ', [(i, c) for i, c in zip(*np.unique(tar['pseudo_label'], return_counts=True))])
    noise_ratio = 1 - (tar['pseudo_label'] == tar['true_label']).sum() / len(tar['true_label'])
    print('[WSL] Noise ratio: %.2f%%' % (noise_ratio * 100))
    coverage = len(tar['index']) / train_size
    print('[WSL] Coverage: %.2f%%' % (coverage * 100))
    print('[WSL] --------------------------------------------------------')


## ------ synthesized random flipping noise ------
def get_weak_supervision_random_flip(dataset, data_dir='/home/chengyu/bert_classification/data'):
    data_path = os.path.join(data_dir, dataset)
    pseudo_label_path = os.path.join(data_path, "pseudo_weak_random_flip.pt")
    if os.path.exists(pseudo_label_path):
        tar = torch.load(pseudo_label_path)
        print_pseudo_label_info(tar, len(tar['index']))
        return tar
    raise FileNotFoundError(f"Pseudo-label file not found at {pseudo_label_path}!")


## ------ prompt ------
def get_weak_supervision_prompt(dataset,
                                data_dir='/home/chengyu/bert_classification/data',
                                weak_dir='/home/chengyu/bert_classification/data_prompt',
                                file_name = "labelled_data.jsonl",
                                label_dic_name="classes_orig.txt",
                                label_name_dic_name = "classes.txt",
                                tar_name = 'pseudo_weak_prompt.pt'):
    
    pseudo_label_path = os.path.join(f'{data_dir}/{dataset}', tar_name)
    if os.path.exists(pseudo_label_path):
        print(f' -- pseudo_label tar exists! Load existing at {pseudo_label_path}.. ')
        tar = torch.load(pseudo_label_path)
        print_pseudo_label_info(tar, len(tar['index']))
        return tar
    
    print(' -- pseudo_label tar doesn\'t exist! Generating.. ')
    ## ----- original dataset ------
    df = pickle.load(open(os.path.join(f'{data_dir}/{dataset}', 'df.pkl'), "rb"))
    labels = df["label"].unique().tolist()
    label_to_index = {label: i for i, label in enumerate(labels)}

    ## ----- weak supervision ------
    # label_to_index_weak = {l.strip('\n'): i for i, l in enumerate(open(f'{weak_dir}/{dataset}/{label_dic_name}', "r").readlines())}
    index_to_label_weak = {i: l.strip('\n') for i, l in enumerate(open(f'{weak_dir}/{dataset}/{label_dic_name}', "r").readlines())}
    name_to_index_weak = {l.strip('\n'): i for i, l in enumerate(open(f'{weak_dir}/{dataset}/{label_name_dic_name}', "r").readlines())}

    tar = {'index': [], 'pseudo_label': [], 'true_label': []}
    for i, line in enumerate(open(f'{weak_dir}/{dataset}/{file_name}', "r").readlines()):
        dic = json.loads(line)
    #     print(df.iloc[i]['label'], index_to_label_weak[dic["gpt2-medium_correct"]])
    #     print('---', df.iloc[i]['text'])
    #     print('>>>', dic['prompt'])
    #     assert(df.iloc[i]['label'] == index_to_label_weak[dic["gpt2-medium_correct"]])
        assert(df.iloc[i]['label'] == index_to_label_weak[dic["answer_index"]])


        tar['index'].append(i)
        tar['pseudo_label'].append(label_to_index[index_to_label_weak[name_to_index_weak[dic["gpt2-medium_predicted"].strip()]]])
        tar['true_label'].append(label_to_index[index_to_label_weak[dic["answer_index"]]])

    for k in tar:
        tar[k] = np.array(tar[k])

    torch.save(tar, pseudo_label_path)
    print_pseudo_label_info(tar, len(tar['index']))
        
    return tar

## ------ x-class ------
def get_weak_supervision_xclass(dataset,
                                data_dir='/home/chengyu/bert_classification/data',
                                method='x-class',
                                weak_dir='/home/zihan/projects/partial_class/datasets',
                                label_dic_name="classes_orig.txt"):
    if method == 'x-class':
        file_name = "selection.json"
        tar_name = 'pseudo_weak_xclass.pt'
        print(f'     Using selected pseudo-labels.. {file_name}  {tar_name}')
    elif method == 'x-class-all':
        file_name = "cluster.json"
        tar_name = 'pseudo_weak_xclass_all.pt'
        print(f'     Using all pseudo-labels.. {file_name}  {tar_name}')
    elif method == 'x-class-repr':
        file_name = "repr.json"
        tar_name = 'pseudo_weak_xclass_repr.pt'
        print(f'     Using repr-based pseudo-labels.. {file_name}  {tar_name}')
    else:
        raise KeyError(f'method {method} not supported!')
    pseudo_label_path = os.path.join(f'{data_dir}/{dataset}', tar_name)
    if os.path.exists(pseudo_label_path):
        print(f' -- pseudo_label tar exists! Load existing at {pseudo_label_path}.. ')
        tar = torch.load(pseudo_label_path)
        print_pseudo_label_info(tar, len(tar['index']))
        return tar
    
    print(' -- pseudo_label tar doesn\'t exist! Generating.. ')
    ## ----- original dataset ------
    df = pickle.load(open(os.path.join(f'{data_dir}/{dataset}', 'df.pkl'), "rb"))
    labels = df["label"].unique().tolist()
    label_to_index = {label: i for i, label in enumerate(labels)}

    ## ----- weak supervision ------
    label_to_index_weak = {l.strip('\n'): i for i, l in enumerate(open(f'{weak_dir}/{dataset}/{label_dic_name}', "r").readlines())}
    index_to_label_weak = {i: l.strip('\n') for i, l in enumerate(open(f'{weak_dir}/{dataset}/{label_dic_name}', "r").readlines())}
    # print(len(open(f'{path}/selection.json', "r").readlines()))
    tar = {'index': [], 'pseudo_label': [], 'true_label': []}
    for i, line in enumerate(open(f'{weak_dir}/{dataset}/{file_name}', "r").readlines()):
        dic = json.loads(line)
        assert(df.iloc[dic['index']]['label'] in label_to_index)
        assert(label_to_index_weak[df.iloc[dic['index']]['label']] == dic['gt_label'])
        tar['index'].append(dic['index'])
        tar['pseudo_label'].append(label_to_index[index_to_label_weak[dic['label']]])
        tar['true_label'].append(label_to_index[index_to_label_weak[dic['gt_label']]])

    for k in tar:
        tar[k] = np.array(tar[k])
        
    torch.save(tar, pseudo_label_path)
    print_pseudo_label_info(tar, len(tar['index']))
        
    return tar


## --------- seed word matching ---------
def get_weak_supervision(trainids, data_dir, dataset, seed_file_name="seedwords.json"):
    data_path = os.path.join(data_dir, dataset)
    pseudo_label_path = os.path.join(data_path, "pseudo_weak.pt")
    if os.path.exists(pseudo_label_path):
        tar = torch.load(pseudo_label_path)
        print_pseudo_label_info(tar, train_size=len(trainids))
        return tar
    
    print('=====> Load raw text..')
    df = pickle.load(open(os.path.join(data_path, "df.pkl"), "rb"))
    df = df.iloc[trainids]
    df = df.reset_index(drop=True)
    
    print('=====> Preprocess text..')
    stored_file = os.path.join(data_path, "df_preprocessed.pkl")
    if os.path.exists(stored_file):
        df_preprocessed = pickle.load(open(stored_file, "rb"))
    else:
        df_preprocessed = preprocess(df)
        pickle.dump(df_preprocessed, open(stored_file, 'wb'))
    
    print('=====> Load seed words..')
    labels = df["label"].unique().tolist()
    with open(os.path.join(data_path, seed_file_name)) as fp:
        label_term_dict = json.load(fp)
    
    print('=====> Init tokensizer..')
    tokenizer = fit_get_tokenizer(df_preprocessed['text'], max_words=150000)
    
    print('=====> Get pseudo-labels..')
    index, pseudo_label, true_label = generate_pseudo_labels(df_preprocessed, labels, label_term_dict, tokenizer)
    
    print('=====> Save pseudo-labels..')
    # #TODO: problematic! label_to_index may not be consistent with the label_to_index in main dataloader, because trainids can be a subset
    ## use the unique list in the entire dataframe instead
    df = pickle.load(open(os.path.join(f'{data_dir}/{dataset}', 'df.pkl'), "rb"))
    labels = df["label"].unique().tolist()
    label_to_index = {label: i for i, label in enumerate(labels)}
    print('weak_supervision: ', label_to_index)
    # label_to_index = {label: i for i, label in enumerate(labels)}
    tar = {'index': np.array(index),
           'pseudo_label': np.array([label_to_index[l] for l in pseudo_label]),
           'true_label': np.array([label_to_index[l] for l in true_label]),
          }
    torch.save(tar, pseudo_label_path)
    print_pseudo_label_info(tar, train_size=len(trainids))
    return tar


from nltk.corpus import stopwords
import string
def preprocess(df):
    print("Preprocessing data..", flush=True)
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    for index, row in df.iterrows():
        if index % 100 == 0:
            print("Finished rows: " + str(index) + " out of " + str(len(df)), flush=True)
        line = row["text"]
        words = line.strip().split()
        new_words = []
        for word in words:
            word_clean = word.translate(str.maketrans('', '', string.punctuation))
            if len(word_clean) == 0 or word_clean in stop_words:
                continue
            new_words.append(word_clean)
        df["text"][index] = " ".join(new_words)
    return df

from tensorflow.keras.preprocessing.text import Tokenizer
def fit_get_tokenizer(data, max_words):
    tokenizer = Tokenizer(num_words=max_words, filters='!"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(data)
    return tokenizer

def generate_pseudo_labels(df, labels, label_term_dict, tokenizer):
    def argmax_label(count_dict):
        maxi = 0
        max_label = None
        keys = sorted(count_dict.keys())
        for l in keys:
            count = 0
            for t in count_dict[l]:
                count += count_dict[l][t]
            if count > maxi:
                maxi = count
                max_label = l
        return max_label

    y = []
    X = []
    y_true = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        print("[%i/%i]" % (index, len(df)), end='\r')
        line = row["text"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set()
            for w in label_term_dict[l]:
                seed_words.add(w)
            int_labels = list(set(words).intersection(seed_words))
            if len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += 1
                    except:
                        count_dict[l][word] = 1
        if flag:
            lbl = argmax_label(count_dict)
            if not lbl:
                continue
            y.append(lbl)
            X.append(index)
            y_true.append(label)
    return X, y, y_true

