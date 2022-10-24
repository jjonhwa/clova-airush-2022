import torch
import random
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

import nsml

from loss import LargeMarginSoftmaxV1, CrossEntropy_soft, one_hot_embedding, accuracy

def ensemble_test(model, args, loader0, loader1, loader2, loader3, loader4):
    model.eval()
    predict_labels = []
        
    for batch0, batch1, batch_2, batch_3, batch_4 in zip(loader0, loader1, loader2, loader3, loader4):
        with torch.no_grad():
            out = model(batch0, batch1, batch_2, batch_3, batch_4)

        score_max_softmax, pred_softmax_label = torch.max(out, axis=1)
        score_max_softmax = list(score_max_softmax.cpu().detach().numpy())
        pred_softmax_label = list(pred_softmax_label.cpu().detach().numpy())

        for score, pred in zip(score_max_softmax, pred_softmax_label):
            predict_labels.append([score, pred])
    
    sorted_labels = sorted(predict_labels, key = lambda x: x[0])
    threshold_score = sorted_labels[1600][0]

    answer = []
    for i in range(len(predict_labels)):
        if predict_labels[i][0] < threshold_score:
            answer.append(-1)
        else:
            answer.append(int(predict_labels[i][1]))
    
    return answer

def test(model, args, data_loader):
    model.eval()
    predict_labels = []
        
    if args.backbone == 'klue-roberta':
        for batch in data_loader:
            b_input_ids = batch['input_ids'].to(args.device)
            b_input_mask = batch['attention_mask'].to(args.device)
            # b_token_type_ids = batch['token_type_ids'].to(args.device)

            with torch.no_grad():
                out = model(b_input_ids,
                            # token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)

            score_max_softmax, pred_softmax_label = torch.max(F.softmax(out['logits'], dim=1), axis=1)
            score_max_softmax = list(score_max_softmax.cpu().detach().numpy())
            pred_softmax_label = list(pred_softmax_label.cpu().detach().numpy())

            for score, pred in zip(score_max_softmax, pred_softmax_label):
                predict_labels.append([score, pred])
    
    elif args.backbone == 'kobig-non-head':
        for batch in data_loader:
            b_input_ids = batch['input_ids'].to(args.device)
            b_input_mask = batch['attention_mask'].to(args.device)
            b_token_type_ids = batch['token_type_ids'].to(args.device)

            with torch.no_grad():
                out = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)
                num_labels = 6
                out = out['pooler_output'][:, :num_labels]
            score_max_softmax, pred_softmax_label = torch.max(F.softmax(out, dim=1), axis=1)
            score_max_softmax = list(score_max_softmax.cpu().detach().numpy())
            pred_softmax_label = list(pred_softmax_label.cpu().detach().numpy())

            for score, pred in zip(score_max_softmax, pred_softmax_label):
                predict_labels.append([score, pred])
    
    else: # Klue-bert, Kobig, electra, kcelectra
        for batch in data_loader:
            b_input_ids = batch['input_ids'].to(args.device)
            b_input_mask = batch['attention_mask'].to(args.device)
            b_token_type_ids = batch['token_type_ids'].to(args.device)

            with torch.no_grad():
                out = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)

            score_max_softmax, pred_softmax_label = torch.max(F.softmax(out['logits'], dim=1), axis=1)
            score_max_softmax = list(score_max_softmax.cpu().detach().numpy())
            pred_softmax_label = list(pred_softmax_label.cpu().detach().numpy())

            for score, pred in zip(score_max_softmax, pred_softmax_label):
                predict_labels.append([score, pred])

    sorted_labels = sorted(predict_labels, key = lambda x: x[0])
    threshold_score = sorted_labels[1600][0]

    answer = []
    for i in range(len(predict_labels)):
        if predict_labels[i][0] < threshold_score:
            answer.append(-1)
        else:
            answer.append(int(predict_labels[i][1]))
    
    return answer

def train(model, args, train_loader):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=100,
                                                num_training_steps=total_steps)
    
    model.train()

    # initialize
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print('start training')

    if args.loss == "crossentropy":
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
    elif args.loss == "largemargin":
        loss_fn = LargeMarginSoftmaxV1()
    elif args.loss == "relax":
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        crossentropy_soft = partial(CrossEntropy_soft, reduction='none')
        softmax = nn.Softmax(dim=1)

        # alpha, upper는 HyperParameter
        alpha = 0.01
        upper = 1

    for epoch in range(args.epochs):
        train_loss = []

        for step, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            b_input_ids = batch['input_ids'].to(args.device)
            b_input_mask = batch['attention_mask'].to(args.device)

            if args.backbone != 'klue-roberta':
                b_token_type_ids = batch['token_type_ids'].to(args.device)

            b_labels = batch['label'].to(args.device)


            if args.backbone == 'klue-roberta':
                out = model(b_input_ids,
                            attention_mask=b_input_mask)
                loss = loss_fn(out['logits'], b_labels)
            elif args.backbone == 'kobig-non-head':
                out = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)
                num_labels = 6
                out = out['pooler_output'][:, :num_labels]

                loss = loss_fn(out, b_labels)
            else:
                out = model(b_input_ids,
                            token_type_ids=b_token_type_ids,
                            attention_mask=b_input_mask)
                loss = loss_fn(out['logits'], b_labels)
            

            if args.loss == "crossentropy" or args.loss == "largemargin":
                loss.backward()
                train_loss.append(loss.item())
                optimizer.step()
                scheduler.step()
            
            elif args.loss == "relax":
                loss_ce_full = loss # args.loss의 차이에 따라서 변수명을 새로 설정하기 위해 재설정
                loss_ce = torch.mean(loss_ce_full)

                if step <= 300 : # step을 Hyperparameter로서 활용
                    loss = (loss_ce -  alpha).abs()
                else:
                    if loss_ce > alpha:
                        loss = loss_ce
                    else:
                        pred = torch.argmax(out['logits'], dim=1)
                        correct = torch.eq(pred, b_labels).float()

                        # Label에 대한 Logit 값
                        confidence_target = softmax(out['logits'])[torch.arange(b_labels.size(0)), b_labels]

                        # 최소값을 고정시켜준다.
                        confidence_target = torch.clamp(confidence_target, min=0., max=upper)

                        ## ????
                        num_classes = 6
                        confidence_else = (1.0 - confidence_target) / (num_classes-1)

                        onehot = one_hot_embedding(b_labels, num_classes=num_classes)
                        soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, num_classes) + \
                            (1-onehot) * confidence_else.unsqueeze(-1).repeat(1, num_classes)

                        loss = (1-correct) * crossentropy_soft(out['logits'], soft_targets) - 1. * loss_ce_full
                        loss = torch.mean(loss)
                
                prec1, prec5 = accuracy(out['logits'].data, b_labels.data, topk=(1, 5))
            
            loss.backward()
            train_loss.append(loss.item())
            optimizer.step()
            scheduler.step()

        avg_train_loss = np.mean(train_loss)

        if args.loss == "relax":
            print('Epoch %d, Average training loss: %f, Train acc: %f'  % (epoch, avg_train_loss, prec1))
        else:
            print("Epoch {0},  Average training loss: {1:.2f}".format(epoch, avg_train_loss))

        # save model
        nsml.save(epoch)
    
    return model