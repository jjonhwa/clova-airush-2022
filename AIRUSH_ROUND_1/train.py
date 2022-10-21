import torch
import numpy as np
import pandas as pd

from transformers import AdamW, get_linear_schedule_with_warmup
import nsml

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def ensemble_test(model, file_path):
    print('start predict')
    
    model.eval()

    dataset = pd.read_csv(file_path)
    context = dataset['contents'].values
    encoded_label = torch.tensor([-1] * len(context))

    eval_accuracy = []
    logits = []
    for i in range(len(dataset)):
        b_labels = encoded_label[i].unsqueeze(-1).unsqueeze(dim=0)
        logit = model(context[i])
        logits.append(logit)
        label = b_labels.cpu().numpy()

        pred_flat = np.argmax(logit, axis=1).flatten()
        labels_flat = label.flatten()
        accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

        eval_accuracy.append(accuracy)

    logits = np.vstack(logits)
    predict_labels = np.argmax(logits, axis=1)
    return predict_labels, np.mean(eval_accuracy)

    
def test(model, args, file_path, tokenizer):
    print('start predict')
    
    model.eval()

    dataset = pd.read_csv(file_path)
    context = dataset['contents'].values
    encoded_label = torch.tensor([-1] * len(context))

    eval_accuracy = []
    logits = []
    for i in range(len(dataset)):
        
        # Overflow 적용 => 하나라도 "엉터리 문서"라고 판정할 경우 => 최종 "엉터리"로 Labeling
        if args.model_name == "roberta":
            token_type = False
        else:
            token_type = True

        encoded_dict = tokenizer(
                          context[i],
                          return_tensors='pt',
                          padding='max_length',
                          stride=128,
                          truncation=True,
                          max_length=args.maxlen,
                          return_token_type_ids=token_type,
                          return_overflowing_tokens=True
                       )

        b_labels = encoded_label[i].unsqueeze(-1).unsqueeze(dim=0)
        item_logit = []
        for k in range(len(encoded_dict['input_ids'])):
            
            if args.model_name == "roberta":
                b_input_ids = encoded_dict['input_ids'][k].unsqueeze(dim=0).to(args.device)
                b_input_mask = encoded_dict['attention_mask'][k].unsqueeze(dim=0).to(args.device)
                
                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask)
            
            else:
                b_input_ids = encoded_dict['input_ids'][k].unsqueeze(dim=0).to(args.device)
                b_input_mask = encoded_dict['attention_mask'][k].unsqueeze(dim=0).to(args.device)
                b_token_type_ids = encoded_dict['token_type_ids'][k].unsqueeze(dim=0).to(args.device)
                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    token_type_ids=b_token_type_ids,
                                    attention_mask=b_input_mask)

            # logit: (0으로 labeling될 확률, 1로 labeling될 확률)
            if args.model_name == "kobig":
                logit = outputs[0][:,0,:][:2]
            else:
                logit = outputs[0] 

            item_logit.append(logit.detach().cpu().numpy())
        
        nonsense_index = 0
        for i, value in enumerate(item_logit):
            sense, nonsense = value.squeeze()
            if nonsense > sense: #  Nonsense가 Sense보다 클 경우를 기준으로 선택
                nonsense_index = i


        logit = item_logit[nonsense_index]
        logits.append(logit)
        label = b_labels.cpu().numpy()

        pred_flat = np.argmax(logit, axis=1).flatten()
        labels_flat = label.flatten()
        accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

        eval_accuracy.append(accuracy)

    logits = np.vstack(logits)
    predict_labels = np.argmax(logits, axis=1)
    return predict_labels, np.mean(eval_accuracy)

def predict(model, args, data_loader):
    print('start predict')
    model.eval()

    eval_accuracy = []
    logits = []
    
    for step, batch in enumerate(data_loader):

        if args.model_name == "roberta":
            b_input_ids = batch['input_ids'].to(args.device)
            b_input_mask = batch['attention_mask'].to(args.device)
            b_labels = batch['label'].to(args.device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)
        
        else:
            b_input_ids = batch['input_ids'].to(args.device)
            b_input_mask = batch['attention_mask'].to(args.device)
            b_token_type_ids = batch['token_type_ids'].to(args.device)
            b_labels = batch['label'].to(args.device)
            
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=b_token_type_ids,
                                attention_mask=b_input_mask)

        if args.model_name == "kobig":
            logit = outputs[0][:,0,:]
        else:
            logit = outputs[0]

        logit = logit.detach().cpu().numpy()
        label = b_labels.cpu().numpy()

        logits.append(logit)

        accuracy = flat_accuracy(logit, label)
        eval_accuracy.append(accuracy)

    logits = np.vstack(logits)
    predict_labels = np.argmax(logits, axis=1)
    return predict_labels, np.mean(eval_accuracy)


def train(model, args, train_loader, valid_loader):
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      eps=args.eps
                      )
    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    print('start training')
    for epoch in range(args.epochs):
        model.train()
        train_loss = []
        for step, batch in enumerate(train_loader):
            model.zero_grad()

            if args.model_name == "roberta":
                b_input_ids = batch['input_ids'].to(args.device)
                b_input_mask = batch['attention_mask'].to(args.device)
                b_labels = batch['label'].to(args.device)
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs[0]

            else:
                b_input_ids = batch['input_ids'].to(args.device)
                b_input_mask = batch['attention_mask'].to(args.device)
                b_token_type_ids = batch['token_type_ids'].to(args.device)
                b_labels = batch['label'].to(args.device)
                outputs = model(b_input_ids,
                                token_type_ids=b_token_type_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs[0]

            if args.model_name == "kobig":
                loss = loss[:,0,:]

            train_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = np.mean(train_loss)
        _, avg_val_accuracy = predict(model, args, valid_loader)
        print("Epoch {0},  Average training loss: {1:.6f} , Validation accuracy : {2:.6f}"\
              .format(epoch, avg_train_loss, avg_val_accuracy))

        nsml.save(epoch)
    return model
