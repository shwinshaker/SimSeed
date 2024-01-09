#!./env python
import torch
import os
import pandas as pd
import pickle
from transformers import BertTokenizer
from ..utils import print


def get_dataset(dataset, data_dir='/home/chengyu/bert_classification/data', config=None):
    if dataset == 'imdb':
        return get_imdb(path=data_dir, config=config)
    elif dataset.lower() in ['agnews', 'nyt-fine', 'nyt-coarse', '20news-fine', '20news-coarse', 'rotten_tomatoes']:
        return get_encoding_dataset(dataset.lower(), path=data_dir, config=config)
    else:
        raise KeyError('dataset: %s' % dataset)

def get_encoding_dataset(dataset, path='/home/chengyu/bert_classification/data', config=None):

    path = os.path.join(path, dataset)
    if config.encoding_max_length == 64:
        file_name = 'df_preprocessed'
    else:
        file_name = f'df_preprocessed_length={config.encoding_max_length}'
    if config.model != 'bert-base-uncased':
        file_name +=  f'_{config.model}'
    file_name += '.pt'
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        return torch.load(file_path)

    ## preprocess
    print(' -- tokenizing..')
    df = pickle.load(open(os.path.join(path, 'df.pkl'), "rb"))
    labels_to_index = {l: i for i, l in enumerate(df['label'].unique())}
    print('dataset loader: ', labels_to_index)
    label_names = {i: l for i, l in enumerate(df['label'].unique())}
    df['label'] = df['label'].apply(lambda label: labels_to_index.get(label))

    tokenizer = BertTokenizer.from_pretrained(config.model, do_lower_case=True)
    encodings = tokenizer(df['text'].tolist(),  # Sentence to encode.
                          add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                          max_length=config.encoding_max_length,  # 
                          padding='max_length', # Pad short sentence
                          truncation=True, # Truncate long sentence
                          return_attention_mask=True,  # Construct attn. masks.
                          return_tensors='pt')  # Return pytorch tensors.
    labels = torch.tensor(df['label'].tolist())

    print(' -- save tokenized data..')
    data_dict = {'encodings': encodings,
                 'labels': labels,
                 'tokenizer': tokenizer,
                 'label_names': label_names}
    torch.save(data_dict, file_path)
    return data_dict

def get_imdb(path='/home/chengyu/bert_classification/data', filename='imdb_encoded.pt', config=None):

    path = os.path.join(path, 'imdb')
    if os.path.exists(os.path.join(path, filename)):
        return torch.load(os.path.join(path, filename))

    ## preprocess
    def map_review(text):
        return " ".join(text.strip().split("<br />"))

    def map_sentiment(text):
        return {"positive": 1, "negative": 0}.get(text)

    df = pd.read_csv('/data3/zichao/backdoor2/data/imdb/IMDB.csv')
    label_names = {i: l for i, l in enumerate(df['sentiment'].unique())}
    df['review'] = df['review'].apply(map_review)
    df['sentiment'] = df['sentiment'].apply(map_sentiment)

    encodings = tokenizer(df['review'].tolist(),  # Sentence to encode.
                          add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                          max_length=config.encoding_max_length,  # 
                          padding='max_length', # Pad short sentence
                          truncation=True, # Truncate long sentence
                          return_attention_mask=True,  # Construct attn. masks.
                          return_tensors='pt')  # Return pytorch tensors.
    labels = torch.tensor(df['sentiment'].tolist())

    data_dict = {'encodings': encodings,
                 'labels': labels,
                 'tokenizer': tokenizer}
    torch.save(data_dict, os.path.join(path, filename))
    
    return data_dict
