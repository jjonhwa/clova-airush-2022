from torch import nn

import torch
import torch.nn.functional as F

class Ensemble(nn.Module):
    
    def __init__(self, model0, model1, model2, model3, model4):
        super().__init__()
        self.model0 = model0
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4

    def forward(self, batch0, batch1, batch2, batch3, batch4):
        
        batches = [batch0, batch1, batch2, batch3, batch4]
        models = [self.model0, self.model1, self.model2, self.model3, self.model4]

        for i, (batch, model) in enumerate(zip(batches, models)):

            if i == 2: # KLUE/RoBERTa-large
                b_input_ids = batch['input_ids'].to('cuda')
                b_input_mask = batch['attention_mask'].to('cuda')

                with torch.no_grad():
                    out = model(b_input_ids,
                                # token_type_ids=b_token_type_ids,
                                attention_mask=b_input_mask)
                    out = F.softmax(out['logits'], dim=1)
            else:
                b_input_ids = batch['input_ids'].to('cuda')
                b_input_mask = batch['attention_mask'].to('cuda')
                b_token_type_ids = batch['token_type_ids'].to('cuda')

                with torch.no_grad():
                    out = model(b_input_ids,
                                token_type_ids=b_token_type_ids,
                                attention_mask=b_input_mask)
                    if i == 3:
                        out = F.softmax(out['pooler_output'][:, :6], dim=1)
                    else:
                        out = F.softmax(out['logits'], dim=1)
            
            # Model별 성능에 따라 비율을 조정하여 Ensemble  수행
            if i == 0:
                logit1 = out
            elif i == 1:
                logit2 = out
            else:
                logit2 += out
        
        # "KPF(LargeMargine)" + "(KPF(Relax) + Klue-RoBERTa + KoBigBird + Tunib-Electra)"
        logit2 /= 4
        logit = (logit1 + logit2) / 2
        return logit