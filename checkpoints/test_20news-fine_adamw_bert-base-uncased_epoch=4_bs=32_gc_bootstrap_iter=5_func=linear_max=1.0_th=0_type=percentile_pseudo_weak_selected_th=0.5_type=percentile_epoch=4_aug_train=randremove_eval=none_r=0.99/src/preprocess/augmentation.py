import torch
import numpy as np
import json
import os
import warnings

def copy(tensor):
    return tensor.detach().clone()

def zeros(length, device):
    return torch.zeros(length, device=device, dtype=int)
    
class Augmentor:
    def __init__(self, tokenizer, classes, dataset, data_dir, label_decode=None, device=None):
        self.tokenizer = tokenizer
        self.classes = classes
        self.input_decode = tokenizer.input_decode
        self.label_decode = label_decode
        self.device = device
        
        # load seed words dict
        with open(os.path.join(data_dir, dataset, "seedwords.json")) as fp:
            self.seed_words = json.load(fp)

        # encoded seed word look up dict
        self.seed_words_encoded = {}
        for c in self.classes:
            self.seed_words_encoded[c] = []
            for seed_word in self.seed_words[self.label_decode(c)]:
                self.seed_words_encoded[c].append(self.tokenizer(seed_word)['input_ids'][1])

    def random_seed_word(self, label):
        return np.random.choice(self.seed_words_encoded[label])

    def replace_elements_random(self, tensor, elements, new_element):
        return
        # is_in = torch.prod(torch.stack([(tensor == e) for e in elements]), dim=0)
        # for idx in is_in.nonzero().flatten():
        # tensor[idx] = random_seed_word(other_label)
        # return tensor[is_in.nonzero().flatten()]

    def replace_seed_word_batch(self):
        return

    def insert_element_random(self, tensor, element):
        nonzero_idx = tensor.nonzero().flatten() # [-1]
        insert_id = torch.randint(nonzero_idx[-1], (1,)).squeeze()
        new_tensor = torch.cat([tensor[:insert_id], torch.tensor([element], device=tensor.device), tensor[insert_id:]])
        return new_tensor[:len(tensor)]
        
    def insert_seed_word_batch(self, inputs, labels):
        input_ids = []
        for i, input_id in enumerate(inputs['input_ids']):
            seed_word = self.random_seed_word(labels[i].item())
            input_ids.append(self.insert_element_random(copy(input_id), seed_word))
        input_ids = torch.stack(input_ids)
        
        new_inputs = {}
        new_inputs['input_ids'] = input_ids
        new_inputs['token_type_ids'] = copy(inputs['token_type_ids'])
        # because insert at places where encoding is non-zero, attention mask should be the same
        new_inputs['attention_mask'] = copy(inputs['attention_mask'])
        return new_inputs

    # def remove_elements_from_tensor(self, tensor, elements):
    #     return torch.tensor([e.item() for e in tensor if e not in elements], device=tensor.device)


    # --------------------- remove seed words ---------------------
    # def remove_elements_from_tensor(self, tensor, elements):
    #     is_in = torch.prod(torch.stack([(tensor != e) for e in elements]), dim=0)
    #     return tensor[is_in.nonzero().flatten()]

    def remove_seed_words_batch(self, inputs, labels):
        device = inputs['input_ids'][0].device
        new_inputs = []
        offsets = []
        for i, _ in enumerate(inputs['input_ids']):
            input_ = {k: v[i] for k, v in inputs.items()}
            seed_words = self.seed_words[self.label_decode(labels[i].item())].copy()
            seed_words_tokenized = [part for seed_word in seed_words for part in self.tokenizer.tokenize(seed_word)]
            seed_words = list(set(seed_words).union(seed_words_tokenized))
            tokens_list = self.tokenizer.convert_ids_to_tokens(input_['input_ids'], skip_special_tokens=True)
            tokens_list_removed = [t for t in tokens_list if t not in seed_words]
            removed_string = self.tokenizer.convert_tokens_to_string(tokens_list_removed)
            new_inputs.append(self.tokenizer.input_encode(removed_string))
            offsets.append(len(tokens_list) - len(tokens_list_removed))
        new_inputs = {k: torch.cat([v[k] for v in new_inputs]).to(device) for k in new_inputs[0].keys()}
        offsets = torch.tensor(offsets, device=device)
        # self.check_aug(inputs, new_inputs, labels=labels, seed_words=seed_words) #, offsets=offsets)
        return new_inputs, offsets

    # def remove_seed_words_batch(self, inputs, labels):
    #     device = inputs['input_ids'][0].device
    #     input_ids = []
    #     attention_masks = []
    #     offsets = []
    #     for i, input_id in enumerate(inputs['input_ids']):
    #         seed_words = self.seed_words_encoded[labels[i].item()]
    #         new_input_id = self.remove_elements_from_tensor(copy(input_id), seed_words)
    #         offset = len(input_id) - len(new_input_id)
    #         new_input_id = torch.cat([new_input_id, zeros(offset, device)]) # append zeros
    #         new_attention_mask = copy(inputs['attention_mask'][i])[offset:] # remove ones from the beginning
    #         new_attention_mask = torch.cat([new_attention_mask, zeros(offset, device)]) # append zeros
    #         offsets.append(offset)
    #         input_ids.append(new_input_id)
    #         attention_masks.append(new_attention_mask)
    #     input_ids = torch.stack(input_ids)
    #     attention_masks = torch.stack(attention_masks)
    #     offsets = torch.tensor(offsets, device=device)

    #     new_inputs = {}
    #     new_inputs['input_ids'] = input_ids
    #     new_inputs['attention_mask'] = attention_masks
    #     new_inputs['token_type_ids'] = copy(inputs['token_type_ids'])

    #     # self.check_aug(inputs, new_inputs, labels)
    #     return new_inputs, offsets

    def remove_random_seed_words_batch(self, inputs, labels, num_remove):
        # remove only a random fraction of seed words in the sequence
        device = inputs['input_ids'][0].device
        new_inputs = []
        for i, _ in enumerate(inputs['input_ids']):
            input_ = {k: v[i] for k, v in inputs.items()}
            seed_words = self.seed_words[self.label_decode(labels[i].item())].copy()
            seed_words_tokenized = [part for seed_word in seed_words for part in self.tokenizer.tokenize(seed_word)]
            seed_words = list(set(seed_words).union(seed_words_tokenized))
            tokens_list = self.tokenizer.convert_ids_to_tokens(input_['input_ids'], skip_special_tokens=True)
            idx_seed = [it for it, t in enumerate(tokens_list) if t in seed_words]
            idx_remove = self.random_choice(idx_seed, num_remove, with_replace=False)
            tokens_list_removed = [t for it, t in enumerate(tokens_list) if it not in idx_remove]
            removed_string = self.tokenizer.convert_tokens_to_string(tokens_list_removed)
            new_inputs.append(self.tokenizer.input_encode(removed_string))
        new_inputs = {k: torch.cat([v[k] for v in new_inputs]).to(device) for k in new_inputs[0].keys()}
        # self.check_aug(inputs, new_inputs, labels=labels, seed_words=seed_words) #, offsets=offsets)
        return new_inputs

    def remove_random_words_except_seed_words_batch(self, inputs, labels, num_remove):
        ## num_remove: fraction of words to remove among the words that are not seed words
        device = inputs['input_ids'][0].device
        new_inputs = []
        for i, _ in enumerate(inputs['input_ids']):
            input_ = {k: v[i] for k, v in inputs.items()}
            seed_words = self.seed_words[self.label_decode(labels[i].item())].copy()
            seed_words_tokenized = [part for seed_word in seed_words for part in self.tokenizer.tokenize(seed_word)]
            seed_words = list(set(seed_words).union(seed_words_tokenized))
            tokens_list = self.tokenizer.convert_ids_to_tokens(input_['input_ids'], skip_special_tokens=True)
            idx_except_seed = [it for it, t in enumerate(tokens_list) if t not in seed_words]
            idx_remove = self.random_choice(idx_except_seed, num_remove, with_replace=False)
            tokens_list = [t for it, t in enumerate(tokens_list) if it not in idx_remove]
            removed_string = self.tokenizer.convert_tokens_to_string(tokens_list)
            new_inputs.append(self.tokenizer.input_encode(removed_string))
        new_inputs = {k: torch.cat([v[k] for v in new_inputs]).to(device) for k in new_inputs[0].keys()}
        # self.check_aug(inputs, new_inputs, labels=labels)
        return new_inputs


    # --------------------- random remove ---------------------
    # def remove_random_elements_from_tensor(self, tensor, num_remove):
    #     nonzero_idx = tensor.nonzero().flatten()
    #     remove_idx = self.random_choice(nonzero_idx[-1], num_remove, with_replace=False)
    #     rest_idx = self.remove_elements_from_tensor(nonzero_idx, remove_idx)
    #     return tensor[rest_idx]

    def remove_random_words_batch(self, inputs, num_remove):
        device = inputs['input_ids'][0].device
        new_inputs = []
        for i, _ in enumerate(inputs['input_ids']):
            input_ = {k: v[i] for k, v in inputs.items()}
            tokens_list = self.tokenizer.convert_ids_to_tokens(input_['input_ids'], skip_special_tokens=True)
            ## at least one word should be left, otherwise the augmented sentence is empty
            remove_idx = self.random_choice(len(tokens_list), num_remove, with_replace=False, at_least_left_one=True)
            tokens_list = [t for it, t in enumerate(tokens_list) if it not in remove_idx]
            removed_string = self.tokenizer.convert_tokens_to_string(tokens_list)
            new_inputs.append(self.tokenizer.input_encode(removed_string))
        new_inputs = {k: torch.cat([v[k] for v in new_inputs]).to(device) for k in new_inputs[0].keys()}

        # self.check_aug(inputs, new_inputs)

        # device = inputs['input_ids'][0].device
        # input_ids = []
        # attention_masks = []
        # for i, input_id in enumerate(inputs['input_ids']):
        #     new_input_id = self.remove_random_elements_from_tensor(copy(input_id), num_remove=num_remove)
        #     offset = len(input_id) - len(new_input_id)
        #     new_input_id = torch.cat([new_input_id, zeros(offset, device)]) # append zeros
        #     new_attention_mask = copy(inputs['attention_mask'][i])[offset:] # remove ones from the beginning
        #     new_attention_mask = torch.cat([new_attention_mask, zeros(offset, device)]) # append zeros
        #     input_ids.append(new_input_id)
        #     attention_masks.append(new_attention_mask)
        # input_ids = torch.stack(input_ids)
        # attention_masks = torch.stack(attention_masks)

        # new_inputs = {}
        # new_inputs['input_ids'] = input_ids
        # new_inputs['attention_mask'] = attention_masks
        # new_inputs['token_type_ids'] = copy(inputs['token_type_ids'])

        # self.check_aug(inputs, new_inputs)
        return new_inputs

    # --------------------- paraphrase ---------------------
    def setup_paraphrase(self, max_length=512):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.paraphrase_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        self.paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to(self.device)
        self.paraphrase_max_length = max_length

    def paraphrase_batch(self, inputs, temperature):
        temperature = float(temperature)
        device = inputs['input_ids'][0].device

        new_inputs = []
        for i, _ in enumerate(inputs['input_ids']):
            input_ = {k: v[i] for k, v in inputs.items()}
            input_string = self.input_decode(input_)
            para_string = self.__paraphrase_t5(input_string, temperature)
            new_inputs.append(self.tokenizer.input_encode(para_string))
        new_inputs = {k: torch.cat([v[k] for v in new_inputs]).to(device) for k in new_inputs[0].keys()}

        # self.check_aug(inputs, new_inputs)
        return new_inputs

    def __paraphrase_t5(self, input_string, temperature):
        """
            Reference: https://github.com/Vamsi995/Paraphrase-Generator
        """
        # sentence = "eu set to clear oracle bn bid for peoplesoft . the european commission is set to clear the billion hostile takeover bid by oracle for rival software group peoplesoft in a move that is likely to pave the way for the creation of a new us software giant"
        text =  "paraphrase: " + input_string + " </s>"
        encoding = self.paraphrase_tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        outputs = self.paraphrase_model.generate(
                input_ids=input_ids,
                attention_mask=attention_masks,
                max_length=self.paraphrase_max_length,
                do_sample=True,
                temperature=temperature,
                top_k=120,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=1,
        )
        return self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


    def replace_random_words_with_synonym_batch(self, inputs, labels):
        raise NotImplementedError()

    def replace_seed_words_with_mlm_batch(self, inputs, labels):
        raise NotImplementedError()

    def crop_random_segment_batch(self, inputs, num_remove):
        raise NotImplementedError()


    def replace_seed_words_with_synonym_batch(self, inputs, labels):
        raise NotImplementedError()

    def setup_mlm(self):
        # from transformers import BertTokenizer,
        from transformers import BertForMaskedLM
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.maskedLM_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(self.device)

    def replace_random_words_with_mlm_batch(self, inputs, num_mask):
        # TODO: Sample from distribution of masked token instead of argmax

        device = inputs['input_ids'][0].device

        ## -- random masking
        masked_sentences = []
        for i, _ in enumerate(inputs['input_ids']):
            input_ = {k: v[i] for k, v in inputs.items()}
            tokens_list = self.tokenizer.convert_ids_to_tokens(input_['input_ids'], skip_special_tokens=True)
            mask_idx = self.random_choice(len(tokens_list), num_mask, with_replace=False)
            # print(mask_idx)
            tokens_list = [t if it not in mask_idx else self.tokenizer.mask_token for it, t in enumerate(tokens_list)]
            masked_sentence = self.tokenizer.convert_tokens_to_string(tokens_list)
            # print('\n', masked_sentence)
            masked_sentences.append(masked_sentence)

        ## -- predict masked token
        # new_inputs = []
        masked_inputs = self.tokenizer.input_encode(masked_sentences).to(device)
        with torch.no_grad():
            logits = self.maskedLM_model(**masked_inputs).logits
        for i, _ in enumerate(masked_inputs['input_ids']):
            mask_token_index = (masked_inputs.input_ids[i] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0] # retrieve index of [MASK]
            predicted_token_id = logits[i, mask_token_index].argmax(axis=-1)
            masked_inputs.input_ids[i, mask_token_index] = predicted_token_id
            # new_inputs.append(masked_inputs)
        # new_inputs = {k: torch.cat([v[k] for v in new_inputs]).to(device) for k in new_inputs[0].keys()}
        new_inputs = masked_inputs.to(device)

        # self.check_aug(inputs, new_inputs)
        return new_inputs

    # --------------------- utils ---------------------
    def random_choice(self, total, num, with_replace=False, at_least_one=False, at_least_left_one=False):
        if isinstance(total, list):
            idx = self.random_choice(len(total), num, with_replace=with_replace)
            return [total[i] for i in idx]

        assert(type(num) in [float, int]), f'num must be float or int, but got {type(num)}'
        if isinstance(num, int):
            warnings.warn(f'int num {num} converted to float! int num will cause confusion (e.g. 1.0 vs. 1), disabled at this moment')
            num = float(num)

        assert(num <= 1.0 and num >= 0.0), f'illegal choice num: {num:g}'
        # num = int(num * total)
        num = round(num * total)
        if at_least_one:
            num = max(num, 1) # at least choose 1
        if at_least_left_one:
            num = min(num, total - 1) # at most choose total - 1

        if with_replace:
            return torch.randint(total, (num,))
        return torch.randperm(total)[:num]

    def check_aug(self, inputs, new_inputs, labels=None, offsets=None, print_tokens=True, seed_words=None):
        if not hasattr(self, 'print_count'):
            self.print_count = 0
        for i in range(inputs['input_ids'].size(0)):
            input_ = {k: v[i] for k, v in inputs.items()}
            new_input_ = {k: v[i] for k, v in new_inputs.items()}
            if offsets is not None:
                if offsets[i] > 0:
                    continue
            print('\n')
            if labels is not None:
                print('Label: ', self.label_decode(labels[i].item()))
                if seed_words is not None:
                    print('Seed words: ', seed_words)
                else:
                    print('Seed words: ', self.seed_words[self.label_decode(labels[i].item())])
            print('Original input: ')
            print(self.input_decode(input_))
            if print_tokens:
                print(self.tokenizer.convert_ids_to_tokens(input_['input_ids'], skip_special_tokens=True))
            print('Augment input: ')
            print(self.input_decode(new_input_))
            if print_tokens:
                print(self.tokenizer.convert_ids_to_tokens(new_input_['input_ids'], skip_special_tokens=True))
            self.print_count += 1
            if self.print_count > 100:
                raise RuntimeError()