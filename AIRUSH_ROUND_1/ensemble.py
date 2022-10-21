import os
import torch
import argparse
import nsml
from nsml import DATASET_PATH, DATASET_NAME

from transformers import (
    BertConfig,
    AutoConfig,
    AutoTokenizer,
    BertTokenizerFast,
    AutoModel
)

from train import ensemble_test
from model import RobertaForClassification, KPFSummationClassification, Ensemble
from utils import seed_everything


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
        results, _ = ensemble_test(model, file_path)
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
    args = parser.parse_args()

    # initialize args
    args.train_path = os.path.join(DATASET_PATH, args.train_path)
    args.valid_path = os.path.join(DATASET_PATH, args.valid_path)

    print(args)

    seed_everything(42)

    # model load
    model_path1 = 'klue/roberta-large'
    config1 = AutoConfig.from_pretrained(model_path1)
    model1 = RobertaForClassification.from_pretrained(model_path1, config=config1)
    model1.to(args.device)
    klue_tokenizer = AutoTokenizer.from_pretrained(model_path1)
    bind_nsml(model1, args=args)
    nsml.load(checkpoint='4', session='KR96413/airush2022-1-2a/574') # KLUE-RoBERTa Best Model

    model_path2 = 'jinmang2/kpfbert'
    config2 = BertConfig.from_pretrained(model_path2)
    model2 = KPFSummationClassification.from_pretrained(model_path2, config=config2)
    model2.to(args.device)
    kpf_tokenizer = BertTokenizerFast.from_pretrained(model_path2)
    bind_nsml(model2, args=args)
    nsml.load(checkpoint='4', session='KR96413/airush2022-1-2a/573') # KPF-bert Best Model

    model_path3 = "monologg/kobigbird-bert-base"
    model3 = AutoModel.from_pretrained(model_path3, add_pooling_layer=False)
    model3.to(args.device)
    kobig_tokenizer = AutoTokenizer.from_pretrained(model_path3) 
    bind_nsml(model3, args=args)
    nsml.load(checkpoint='2', session='KR96413/airush2022-1-2a/547') # KoBigBird Best Model

    model = Ensemble(model1, model2, model3,
                     klue_tokenizer, kpf_tokenizer, kobig_tokenizer) 
    model.to(args.device)
    bind_nsml(model, args=args)
    
    # test mode
    if args.pause:
        nsml.paused(scope=locals())

    # train mode
    if args.mode == "train":

        nsml.save('best')
        exit()

