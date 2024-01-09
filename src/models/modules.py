import torch
import torch.nn as nn
import torch.nn.functional as F

# To try, preserve the original dropout layer instead of replacing it
# model.bert.embeddings.dropout = ChannelDropout(p=config.channel_dropout)
# model.bert.embeddings.dropout = nn.Sequential(model.bert.embeddings.dropout, ChannelDropout(p=config.channel_dropout))

## embedding dropout
def add_dropout1d_word_embedding(model, config):
    # from .modules import Dropout
    model.bert.embeddings.word_embeddings = nn.Sequential(model.bert.embeddings.word_embeddings,
                                                          Dropout(p=config.channel_dropout))

def add_dropout2d_word_embedding(model, config):
    # model.bert.embeddings.word_embeddings = nn.Sequential(model.bert.embeddings.word_embeddings,
    #                                                       ChannelDropout(p=config.channel_dropout))
    model.bert.embeddings.word_embeddings = nn.Sequential(model.bert.embeddings.word_embeddings,
                                                          FixedRateChannelDropout(p=config.channel_dropout))

def add_embed_dropout_word_embedding(model, config):
    model.bert.embeddings.word_embeddings = nn.Sequential(model.bert.embeddings.word_embeddings,
                                                          EmbeddingDropout(p=config.channel_dropout))


def replace_dropout_in_embedding(model, config):
    model.bert.embeddings.dropout.p = config.channel_dropout

def replace_dropout_in_embedding_with_dropout2d(model, config):
    model.bert.embeddings.dropout = ChannelDropout(p=config.channel_dropout)

## attention dropout
def add_dropout_in_1st_attention_with_dropout2d(model, config):
    # should be equivalent to token deletion if the dropout is applied to the column dimension in each attention score matrix
    class CustomDropout(nn.Module):
        """
            apply dropout to 2nd dimension
        """
        def __init__(self, p):
            super().__init__()
            self.dropout = nn.Dropout2d(p=p)
            self.p = p
        
        def forward(self, inputs):
            # Expect input size to be (batch size, n_head, encoding_length, encoding_length)
            assert(len(inputs.size()) == 4), inputs.size()
            # dropout2d only drops the 2nd dimension, so we need to permute the tensor
            #   We cannot simply set the 2nd dimension to 0,
            #   because in this case each sentence in the batch will be have the same positions dropped
            return self.dropout(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    model.bert.encoder.layer[0].attention.self.dropout = nn.Sequential(model.bert.encoder.layer[0].attention.self.dropout,
                                                                       CustomDropout(p=config.channel_dropout))

def replace_dropout_in_1st_attention_with_dropout2d(model, config):
    model.bert.encoder.layer[0].attention.self.dropout = ChannelDropout4D(p=config.channel_dropout)

def replace_dropout_in_1st_attention(model, config):
    # is not equivalent to token deletion, as the attetion score is a 768x768 matrix?
    model.bert.encoder.layer[0].attention.self.dropout.p = config.channel_dropout

def replace_dropout_all_attention(model, config):
    for layer in model.bert.encoder.layer:
        layer.attention.self.dropout.p = config.channel_dropout

def replace_dropout_all_attention_with_dropout2d(model, config):
    for layer in model.bert.encoder.layer:
        layer.attention.self.dropout = ChannelDropout4D(p=config.channel_dropout)

## all layer output (3d)
def replace_dropout_all_output_with_dropout2d(model, config):
    for layer in model.bert.encoder.layer:
        layer.output.dropout = ChannelDropout(p=config.channel_dropout)

def replace_dropout_1st_output_with_dropout2d(model, config):
    model.bert.encoder.layer[0].output.dropout = ChannelDropout(p=config.channel_dropout)

def replace_dropout_4th_output_with_dropout2d(model, config):
    model.bert.encoder.layer[3].output.dropout = ChannelDropout(p=config.channel_dropout)

def replace_dropout_8th_output_with_dropout2d(model, config):
    model.bert.encoder.layer[7].output.dropout = ChannelDropout(p=config.channel_dropout)

def replace_dropout_12th_output_with_dropout2d(model, config):
    model.bert.encoder.layer[11].output.dropout = ChannelDropout(p=config.channel_dropout)


def replace_dropout_4th_output_with_embed_dropout(model, config):
    model.bert.encoder.layer[3].output.dropout = EmbeddingDropout(p=config.channel_dropout)

def replace_dropout_8th_output_with_embed_dropout(model, config):
    model.bert.encoder.layer[7].output.dropout = EmbeddingDropout(p=config.channel_dropout)

## hidden layer dropout
def replace_dropout_all(model, config):
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = config.channel_dropout


## implementation of custom dropout layers
class EmbeddingDropout(nn.Module):
    ## --- scale at training time, no scale at test time (pytorch original dropout)
    def __init__(self, p):
        super().__init__()
        # original dropout, scale at training time, no scale at test time (inverted dropout)
        self.dropout = nn.Dropout2d(p=p)
        self.p = p
    
    def forward(self, inputs):
        ## expect inputs to be (N, C, H)
        ## - N: batch size, i.e., number of sentences
        ## - C: number of channels, e.g. number of tokens in a sentence
        ## - H: embedding size
        assert(len(inputs.size()) == 3)
        return self.dropout(inputs.unsqueeze(1).permute(0, 3, 2, 1)).permute(0, 2, 1, 3).squeeze(-1)


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.p = p
    
    def forward(self, inputs):
        ## dropout in pytorch is scaled, output *= 1/(1-p), so we need to scale it back
        return self.dropout(inputs) # * (1-self.p)


class ChannelDropout4D(nn.Module):
    """
        z ~ Bern(batch_size, encoding_length)
        z.repeat(multi_head, encoding length)
    """
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout2d(p=p)
        self.p = p
    
    def forward(self, inputs):
        # Expect input size to be (batch size, n_head, encoding_length, encoding_length)
        assert(len(inputs.size()) == 4), inputs.size()
        # dropout2d only drops the 2nd dimension, so we need to permute the tensor
        #   We cannot simply set the 2nd dimension to 0,
        #   because in this case each sentence in the batch will be have the same positions dropped
        return self.dropout(inputs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class FixedRateChannelDropout(nn.Module):
    ## -- select exactly p fraction of channels to drop out, instead of dropping out a channel with probability p
    def __init__(self, p, at_least_one=False, at_least_left_one=False):
        super().__init__()
        assert(type(p) in [float, int]), f'dropout rate must be float or int, but got {type(p)}'
        assert(p <= 1.0 and p >= 0.0), f'illegal dropout rate: {p:g}'
        self.p = p
        self.at_least_one = at_least_one
        self.at_least_left_one = at_least_left_one

    def forward(self, inputs):
        assert(len(inputs.size()) == 3)
        if self.training:
            drop_num = round(self.p * inputs.shape[1])
            if self.at_least_one:
                drop_num = max(drop_num, 1) # at least choose 1
            if self.at_least_left_one:
                drop_num = min(drop_num, inputs.shape[1] - 1) # at most choose total - 1
                
            ## randomly mask dim 1. Mask is different for dim 0
            indices = torch.argsort(torch.rand(*inputs.shape[:2]), dim=1)
            inputs = inputs.clone()
            inputs[torch.arange(inputs.shape[0]).unsqueeze(-1), indices[:, :drop_num]] = 0
            inputs *= 1.0/(1 - self.p)
        return inputs
    

class ChannelDropout(nn.Module):
    ## --- scale at training time, no scale at test time (pytorch original dropout)
    def __init__(self, p):
        super().__init__()
        # original dropout, scale at training time, no scale at test time (inverted dropout)
        self.dropout = nn.Dropout2d(p=p)
        self.p = p
    
    def forward(self, inputs):
        ## expect inputs to be (N, C, H)
        ## - N: batch size, i.e., number of sentences
        ## - C: number of channels, e.g. number of tokens in a sentence
        ## - H: embedding size
        assert(len(inputs.size()) == 3)
        ## Dropout2d only works for the second dimension (C) in a 4D tensor (N,C,H,W), so we need to unsqueeze and squeeze
        ## - The official documentation https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html is ambiguous, see my own experiments
        # return self.dropout(inputs.unsqueeze(-1)).squeeze(-1)
        return self.dropout(inputs.unsqueeze(-1)).squeeze(-1)

# class ChannelDropout(nn.Module):
#     ## --- no scale at training time, scale at test time if uncomment the scaling below
#     def __init__(self, p):
#         super().__init__()
#         # original dropout, scale at training time, no scale at test time (inverted dropout)
#         self.dropout = nn.Dropout2d(p=p)
#         self.p = p
    
#     def forward(self, inputs):
#         ## expect inputs to be (N, C, H)
#         ## - N: batch size, i.e., number of sentences
#         ## - C: number of channels, e.g. number of tokens in a sentence
#         ## - H: embedding size
#         assert(len(inputs.size()) == 3)
#         ## Dropout2d only works for the second dimension (C) in a 4D tensor (N,C,H,W), so we need to unsqueeze and squeeze
#         ## - The official documentation https://pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html is ambiguous, see my own experiments
#         # return self.dropout(inputs.unsqueeze(-1)).squeeze(-1)
#         ## dropout in pytorch is scaled, output *= 1/(1-p), so we need to scale it back
#         return self.dropout(inputs.unsqueeze(-1)).squeeze(-1) * (1-self.p)

# class ChannelDropout(nn.Module):
#     ## --- no scaling at training or test time
#     def __init__(self, p):
#         super().__init__()
#         self.dropout = nn.Dropout2d(p=p)
#         self.p = p
    
#     def forward(self, inputs):
#         assert(len(inputs.size()) == 3)
#         dropped_out = self.dropout(inputs.unsqueeze(-1)).squeeze(-1) 
#         if self.training:
#             dropped_out *= 1 - self.p
#         return dropped_out
