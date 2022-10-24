import argparse
import os

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import (
    TensorDataset,
    DataLoader, 
    Dataset,
    RandomSampler, 
    SequentialSampler
)
from model import Ensemble
from transformers import (
    DistilBertModel,
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    BertTokenizerFast,
    BertForSequenceClassification
)
from tokenization_kobert import KoBertTokenizer

import nsml
from nsml import DATASET_PATH, DATASET_NAME

from dataset import ClfDataset
from train import ensemble_test
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

        test_data = pd.read_csv(file_path)
        original_data = pd.DataFrame.copy(test_data)

        # Preprocessing
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

        # Ensemble Data Upload

        model_path = 'jinmang2/kpfbert'
        tokenizer_7 = BertTokenizerFast.from_pretrained(model_path)
        args.backbone = 'kpf'
        test_dataset_7 = ClfDataset(test_data, tokenizer_7, args, is_training=False)
        test_dataloader_7 = DataLoader(
            test_dataset_7,
            batch_size=32,
            shuffle=False
        )
        
        # model_path = 'klue/bert-base'
        # klue_tokenizer = AutoTokenizer.from_pretrained(model_path)
        # klue_test_dataset = ClfDataset(test_data, klue_tokenizer, args, is_training=False)
        # klue_test_dataloader = DataLoader(
        #     klue_test_dataset,
        #     batch_size=32,
        #     shuffle=False
        # )

        model_path = 'jinmang2/kpfbert'
        tokenizer_2 = BertTokenizerFast.from_pretrained(model_path)
        args.backbone = 'kpf'
        test_dataset_2 = ClfDataset(original_data, tokenizer_2, args, is_training=False)
        test_dataloader_2 = DataLoader(
            test_dataset_2,
            batch_size=32,
            shuffle=False
        )

        tokenizer_3 = AutoTokenizer.from_pretrained('klue/roberta-large')
        args.backbone = 'klue-roberta'
        test_dataset_3 = ClfDataset(original_data, tokenizer_3, args, is_training=False)
        test_dataloader_3 = DataLoader(
            test_dataset_3,
            batch_size=32,
            shuffle=False
        )

        tokenizer_4 = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')
        args.backbone = 'kobig'
        test_dataset_4 = ClfDataset(original_data, tokenizer_4, args, is_training=False)
        test_dataloader_4 = DataLoader(
            test_dataset_4,
            batch_size=32,
            shuffle=False
        )

        tokenizer_5 = AutoTokenizer.from_pretrained('tunib/electra-ko-base')
        args.backbone = 'electra'
        test_dataset_5 = ClfDataset(original_data, tokenizer_5, args, is_training=False)
        test_dataloader_5 = DataLoader(
            test_dataset_5,
            batch_size=32,
            shuffle=False
        )
        results = ensemble_test(model, args, test_dataloader_7, test_dataloader_2, test_dataloader_3, test_dataloader_4, test_dataloader_5)
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
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--preprocess", type=str, default='hanspell')
    args = parser.parse_args()

    print(args)

    # Best Solo Model => 0.978461538
    model_path = 'jinmang2/kpfbert'
    model0 = BertForSequenceClassification.from_pretrained(model_path, num_labels=6).to(args.device)
    bind_nsml(model0, args=args)
    nsml.load(checkpoint='0', session="KR96413/airush2022-2-7/971") # Session명 변경
    
    # # 9번: KLUE/BERT-base => 0.97051282
    # model_path = 'klue/bert-base'
    # model1 = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6).to(args.device)
    # bind_nsml(model1, args=args)
    # nsml.load(checkpoint='1', session="KR96413/airush2022-2-7/1019")

    kpf_model = BertForSequenceClassification.from_pretrained('jinmang2/kpfbert', num_labels=6).to(args.device)
    # kpf_tokenizer = BertTokenizerFast.from_pretrained('jinmang2/kpfbert')
    bind_nsml(kpf_model, args=args)
    nsml.load(checkpoint='0', session="KR96413/airush2022-2-7/302")

    klue_model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-large', num_labels=6).to(args.device)
    # klue_tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
    bind_nsml(klue_model, args=args)
    nsml.load(checkpoint='1', session="KR96413/airush2022-2-7/466")

    kobig_model = AutoModel.from_pretrained('monologg/kobigbird-bert-base')
    # kobig_tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')
    bind_nsml(kobig_model, args=args)
    nsml.load(checkpoint='2', session='KR96413/airush2022-2-7/689')
    # kobig_state = kobig_model.state_dict()

    electra_model = AutoModelForSequenceClassification.from_pretrained('tunib/electra-ko-base', num_labels=6).to(args.device)
    bind_nsml(electra_model, args=args)
    nsml.load(checkpoint='1', session='KR96413/airush2022-2-7/535')
    print('model is loaded')

    model = Ensemble(model0, kpf_model, klue_model, kobig_model, electra_model)
    model.to(args.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    bind_nsml(model, args=args)

    # test mode
    if args.pause:
        nsml.paused(scope=locals())

    # train mode
    if args.mode == "train":

        nsml.save('0')
        exit()