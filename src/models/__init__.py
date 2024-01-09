import torch
import torch.nn as nn

from transformers import BertForSequenceClassification, AdamW

def get_model(config, loaders):
    # from pretrained: replace the pretraining head with a randomly initialized classification head
    model = BertForSequenceClassification.from_pretrained(
        config.model,
        num_labels=loaders.num_classes,  # The number of output labels -- 2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.to(config.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    print("     Total params: %.2fM" % (sum(p.numel() for p in model.parameters())/1000000.0))

    if hasattr(config, 'channel_dropout') and config.channel_dropout:
        from . import modules
        getattr(modules, config.channel_dropout_type)(model, config)
        # from .modules import replace_dropout_in_embedding_with_dropout2d
        # replace_dropout_in_embedding_with_dropout2d(model, config)
        # from .modules import add_dropout2d_word_embedding
        # add_dropout2d_word_embedding(model, config)
        # from .modules import add_dropout1d_word_embedding
        # add_dropout1d_word_embedding(model, config)
        # from .modules import replace_dropout_all
        # replace_dropout_all(model, config)

        model.eval()
        print("     Channel dropout enabled: %g Type: %s" % (config.channel_dropout, config.channel_dropout_type))
        print(model)

    if hasattr(config, 'train_module') and config.train_module:
        for name, param in model.named_parameters():
            if all([m not in name for m in config.train_module]):
                param.requires_grad = False

    return model