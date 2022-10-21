import os
import nsml
import argparse
import torch

from nsml import DATASET_PATH
from transformers import (
    AutoConfig, 
    AutoTokenizer,
    BertConfig,
    AutoModel
)
from torch.utils.data import DataLoader

from dataset import ClfDataset
from model import RobertaForClassification, KPFSummationClassification
from utils import seed_everything
from train import train, predict, test

def bind_nsml(model, args=None):
    def save(dir_name, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))

    def load(dir_name, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'), map_location=args.device)
        model.load_state_dict(state, strict=False)
        print('model is loaded')

    def infer(file_path, **kwargs):
        print('start inference')
        if args.model_name == "roberta":
            tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
        elif args.model_name == "kpf":
            tokenizer = AutoTokenizer.from_pretrained('jinmang2/kpfbert')
        elif args.model_name == "kobig":
            tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')

        if args.model_name == "roberta" or args.model_name("kpf"):
            results, _ = test(model, args, file_path, tokenizer)
        else: # kobig
            test_dataset = ClfDataset(file_path, tokenizer, args, is_training=False)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch,
                shuffle=False
            )
            results, _ = predict(model, args, test_dataloader)
        return results

    nsml.bind(save, load, infer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train/train_data/train_data")
    parser.add_argument("--valid_path", type=str, default="train/train_data/valid_data")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--maxlen", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pause", type=int, default=0)
    parser.add_argument("--underfit", type=int, default=4000)
    parser.add_argument("--model_name", type=str, default="roberta")
    args = parser.parse_args()

    # initialize args
    args.train_path = os.path.join(DATASET_PATH, args.train_path)
    args.valid_path = os.path.join(DATASET_PATH, args.valid_path)

    print(args)

    # model load
    if args.model_name == "roberta":
        model_path = 'klue/roberta-large'
        config = AutoConfig.from_pretrained(model_path)
        model = RobertaForClassification.from_pretrained(model_path, config=config)
    elif args.model_name == "kpf":
        model_path = 'jinmang2/kpfbert'
        config = BertConfig.from_pretrained(model_path)
        model = KPFSummationClassification.from_pretrained(model_path, config=config)
    elif args.model_name == "kobig":
        model_path = "monologg/kobigbird-bert-base"
        model = AutoModel.from_pretrained(model_path, add_pooling_layer=False)

    model.to(args.device)

    seed_everything(42)

    bind_nsml(model, args=args)

    # test mode
    if args.pause:
        nsml.paused(scope=locals())

    # train mode
    if args.mode == "train":
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        train_dataset = ClfDataset(args.train_path, tokenizer, args)
        validation_dataset = ClfDataset(args.valid_path, tokenizer, args)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            shuffle=True
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=args.batch,
            shuffle=False
        )

        train(model, args, train_dataloader, validation_dataloader)