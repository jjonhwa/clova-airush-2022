import pandas as pd
import torch

from torch.utils.data import Dataset


class ClfDataset(Dataset):
    def __init__(self, data, tokenizer, args, is_training=True):
        self.dataset = data
        self.context = self.dataset['contents'].values
        if is_training:
            self.label = torch.tensor(self.dataset['label'].values)
        else:
            self.label = torch.tensor([-2] * len(self.dataset))
        self.max_length = args.maxlen
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        context = self.context[idx]
        label = self.label[idx]

        if self.args.backbone == 'klue-roberta':
            encoded_dict = self.tokenizer(
                context,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False
            )
            
            encoded_dict['input_ids'] = encoded_dict['input_ids'].squeeze(0)
            encoded_dict['attention_mask'] = encoded_dict['attention_mask'].squeeze(0)
            encoded_dict['label'] = label
        else:
            encoded_dict = self.tokenizer(
                context,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
            )
            
            encoded_dict['input_ids'] = encoded_dict['input_ids'].squeeze(0)
            encoded_dict['attention_mask'] = encoded_dict['attention_mask'].squeeze(0)
            encoded_dict['token_type_ids'] = encoded_dict['token_type_ids'].squeeze(0)
            encoded_dict['label'] = label
            
        return encoded_dict
