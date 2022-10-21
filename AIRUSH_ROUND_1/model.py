import torch

from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, BertPreTrainedModel, RobertaModel

class RobertaForClassification(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.network = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1][-1][:,0] + outputs[1][-2][:,0] + outputs[1][-3][:,0] + outputs[1][-4][:,0]
        pooled_output = self.network(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        outputs.hidden_states = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class KPFSummationClassification(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1][-1][:,0] + outputs[1][-2][:,0] + outputs[1][-3][:,0] + outputs[1][-4][:,0]
        logits = self.classifier(pooled_output)

        loss = None
        outputs.hidden_states = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Ensemble(nn.Module):
    
    def __init__(self, model1, model2, model3, tokenizer1, tokenizer2, tokenizer3):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.tokenizer3 = tokenizer3

    def forward(self, context=None):
        
        # KLUE-RoBERTa large: Test
        encoded_dict_1 = self.tokenizer1(
            context,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
            stride=128,
            return_overflowing_tokens=True
        )

        item_logit = []
        for k in range(len(encoded_dict_1['input_ids'])):
            b_input_ids = encoded_dict_1['input_ids'][k].unsqueeze(dim=0).to('cuda')
            b_input_mask = encoded_dict_1['attention_mask'][k].unsqueeze(dim=0).to('cuda')

            with torch.no_grad():
                outputs_1 = self.model1(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=None
                )
            
            logit = outputs_1[0]
            item_logit.append(logit.detach().cpu().numpy())

        nonsense_index = 0
        for idx, value in enumerate(item_logit):
            sense, nonsense = value.squeeze()
            if nonsense > sense: #  Nonsense가 Sense보다 클 경우를 기준으로 선택
                nonsense_index = idx

        logit_1 = item_logit[nonsense_index]

        # KPF-bert: Test
        encoded_dict_2 = self.tokenizer2(
            context,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512,
            # return_token_type_ids=False,
            stride=128,
            return_overflowing_tokens=True
        )

        item_logit = []
        for k in range(len(encoded_dict_2['input_ids'])):
            b_input_ids = encoded_dict_2['input_ids'][k].unsqueeze(dim=0).to('cuda')
            b_input_mask = encoded_dict_2['attention_mask'][k].unsqueeze(dim=0).to('cuda')
            b_token_type_ids = encoded_dict_2['token_type_ids'][k].unsqueeze(dim=0).to('cuda')
            with torch.no_grad():
                outputs_2 = self.model2(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=b_token_type_ids
                )
            
            logit = outputs_2[0]
            item_logit.append(logit.detach().cpu().numpy())

        nonsense_index = 0
        for idx, value in enumerate(item_logit):
            sense, nonsense = value.squeeze()
            if nonsense > sense: #  Nonsense가 Sense보다 클 경우를 기준으로 선택
                nonsense_index = idx

        logit_2 = item_logit[nonsense_index]

        # KoBigBird: Test
        encoded_dict_3 = self.tokenizer3(
            context,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=1024,
            # return_token_type_ids=False,
            stride=256,
            return_overflowing_tokens=True
        )

        item_logit = []
        for k in range(len(encoded_dict_3['input_ids'])):
            b_input_ids = encoded_dict_3['input_ids'][k].unsqueeze(dim=0).to('cuda')
            b_input_mask = encoded_dict_3['attention_mask'][k].unsqueeze(dim=0).to('cuda')
            b_token_type_ids = encoded_dict_3['token_type_ids'][k].unsqueeze(dim=0).to('cuda')
            with torch.no_grad():
                outputs_3 = self.model3(
                    b_input_ids,
                    attention_mask=b_input_mask,
                    token_type_ids=b_token_type_ids
                )
            
            logit = outputs_3[0][:,0,:][:,:2]
            item_logit.append(logit.detach().cpu().numpy())

        nonsense_index = 0
        for idx, value in enumerate(item_logit):
            sense, nonsense = value.squeeze()
            if nonsense > sense: #  Nonsense가 Sense보다 클 경우를 기준으로 선택
                nonsense_index = idx

        logit_3 = item_logit[0]
        

        # Logit Ensemble 수행
        
        # output_logit = (logit_1 + logit_2 + logit_3) / 3
        # output_logit = (logit_1 + logit_3) / 2
        # output_logit = (logit_2 + logit_3) / 2
        # output_logit = (logit_1 + logit_2) / 2
        # output_logit = (logit_1*0.3 + logit_2*0.2 + logit_3*0.5)
        # output_logit = (logit_1*0.4 + logit_3*0.6)
        # output_logit = (logit_1*0.3 + logit_3*0.7)
        # output_logit = (logit_1*0.45 + logit_3*0.55)
        # output_logit = (logit_1*2 + logit_2*1 + logit_3*3)
        # output_logit = (logit_1*1 + logit_2*1 + logit_3*2)
        output_logit = (logit_1*1 + logit_2*1 + logit_3*1) / 3
        return output_logit