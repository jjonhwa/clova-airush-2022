from torch import nn
import torch.nn.functional as F

import torch


class LargeMarginSoftmaxV1(nn.Module):

    def __init__(self, lam=0.3, reduction='mean', ignore_index=255):
        super(LargeMarginSoftmaxV1, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lam = lam
        self.ce_crit = nn.CrossEntropyLoss(
                reduction='none', ignore_index=ignore_index)


    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LargeMarginSoftmaxV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()
        logits.retain_grad()
        logits.register_hook(lambda grad: grad)
        with torch.no_grad():
            num_classes = logits.size(1)
            coeff = 1. / (num_classes - 1.)
            lb = label.clone().detach()
            mask = label == self.ignore_index
            lb[mask] = 0
            idx = torch.zeros_like(logits).scatter_(1, lb.unsqueeze(1), 1.)

        lgts = logits - idx * 1.e6
        q = lgts.softmax(dim=1)
        q = q * (1. - idx)

        log_q = lgts.log_softmax(dim=1)
        log_q = log_q * (1. - idx)
        mg_loss = ((q - coeff) * log_q) * (self.lam / 2)
        mg_loss = mg_loss * (1. - idx)
        mg_loss = mg_loss.sum(dim=1)

        ce_loss = self.ce_crit(logits, label)
        loss = ce_loss + mg_loss
        loss = loss[mask == 0]

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)

def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # topk: 가장 높은 점수 순서대로 Label 값 정렬
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() # to tensor
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.reshape(correct[:k], (-1,)).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res