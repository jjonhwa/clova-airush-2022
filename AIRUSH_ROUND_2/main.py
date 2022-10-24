import argparse
import os
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    DistilBertModel,
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    BertTokenizerFast,
    BertForSequenceClassification
)

import nsml
from nsml import DATASET_PATH

from dataset import ClfDataset
from train import train, test
from preprocess import *

def bind_nsml(model, args=None):
    def save(dir_name, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))

    def load(dir_name, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'), map_location=args.device)
        model.load_state_dict(state, strict=False)

    def infer(file_path, **kwargs):
        print('start inference')
        if args.backbone == 'kpf':
            model_path = 'jinmang2/kpfbert'
            tokenizer = BertTokenizerFast.from_pretrained(model_path)
        elif args.backbone == 'klue-bert':
            model_path = 'klue/bert-base'
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif args.backbone == 'klue-roberta':
            model_path = 'klue/roberta-large'
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif args.backbone == 'kobig':
            model_path = "monologg/kobigbird-bert-base"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif args.backbone == 'electra':
            model_path = 'tunib/electra-ko-base'
            tokenizer = AutoTokenizer.from_pretrained(model_path) 
        elif args.backbone == 'kcelectra':
            model_path = 'beomi/KcELECTRA-base'
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif args.backbone == 'kobig-non-head':
            model_path = "monologg/kobigbird-bert-base"
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        test_data = pd.read_csv(file_path)
        if args.preprocess == 'all':
            for i in range(len(test_data)):
                ctxt = test_data['contents'][i]
                ctxt = preprocessing(ctxt)
                test_data['contents'][i] = ctxt
        elif args.preprocess == 'hanspell': # Hanspell
            contents = test_data['contents']
            new_text = spell_check_using_hanspell(list(contents), max_length=args.maxlen, is_training=False)
            test_data['contents'] = new_text
            
        elif args.preprocess == 'hanspell_all':
            for i in range(len(test_data)):
                ctxt = test_data['contents'][i]
                ctxt = preprocessing(ctxt)
                test_data['contents'][i] = ctxt
            
            contents = test_data['contents']
            new_text = spell_check_using_hanspell(list(contents), max_length=args.maxlen, is_training=False)
            test_data['contents'] = new_text

        test_dataset = ClfDataset(test_data, tokenizer, args, is_training=False)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )
        results = test(model, args, test_dataloader)
        return results

    nsml.bind(save, load, infer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train/train_data")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--preprocess", type=str, default='hanspell')
    parser.add_argument("--backbone", type=str, default='electra')
    parser.add_argument("--loss", type=str, default="crossentropy")
    args = parser.parse_args()

    print(args)
    
    # model load
    if args.backbone == 'kpf':
        model_path = 'jinmang2/kpfbert'
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=6).to(args.device)
    elif args.backbone == 'klue-bert':
        model_path = 'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6).to(args.device)
    elif args.backbone == 'klue-roberta':
        model_path = 'klue/roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6).to(args.device)
    elif args.backbone == 'kobig':
        model_path = "monologg/kobigbird-bert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6).to(args.device)
    elif args.backbone == 'electra':
        model_path = 'tunib/electra-ko-base'
        tokenizer = AutoTokenizer.from_pretrained(model_path) 
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6).to(args.device)
    elif args.backbone == 'kcelectra':
        model_path = 'beomi/KcELECTRA-base'
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6).to(args.device)
    elif args.backbone == 'kobig-non-head':
        model_path = "monologg/kobigbird-bert-base"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(args.device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)

    print('model is loaded')

    bind_nsml(model, args=args)

    # test mode
    if args.pause:
        nsml.paused(scope=locals())

    # train mode
    if args.mode == "train":
        # initialize args
        args.train_path = os.path.join(DATASET_PATH, args.train_path)
        train_data = pd.read_csv(args.train_path)

        # Preprocessing
        ## Deduplicate Data
        if args.preprocess == 'dup':
            print('preprocessing_dup')
            train_data = remove_dup(train_data)

        ## Preprocessing Data from preprocessing.py
        elif args.preprocess == 'all':
            print('preprocessing_all')
            # train_data = remove_dup(train_data)

            for i in range(len(train_data)):
                ctxt = train_data['contents'][i]
                ctxt = preprocessing(ctxt)
                train_data['contents'][i] = ctxt

        ## Spell check with hanspell
        elif args.preprocess == 'hanspell': # Hanspell
            print('preprocessing_hanspell')
            # train_data = remove_dup(train_data)
            contents = train_data['contents']
            new_text = spell_check_using_hanspell(list(contents), max_length=args.maxlen)
            new_label = list(train_data['label'])

            train_data = pd.DataFrame({'contents': new_text,
                                        'label': new_label})

        ## Spell check + Preprocessing Data
        elif args.preprocess == 'hanspell_all':
            print('preprocessing_hanspell_all')
            # train_data = remove_dup(train_data) # 성능 하락

            for i in range(len(train_data)):
                ctxt = train_data['contents'][i]
                ctxt = preprocessing(ctxt)
                train_data['contents'][i] = ctxt
            
            contents = train_data['contents']
            new_text = spell_check_using_hanspell(list(contents), max_length=args.maxlen)
            new_label = list(train_data['label'])

            train_data = pd.DataFrame({'contents': new_text,
                                        'label': new_label})
        
        train_dataset = ClfDataset(train_data, tokenizer, args)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True
        )

        train(model, args, train_dataloader)
