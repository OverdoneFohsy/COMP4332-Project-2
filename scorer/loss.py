import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score

class CEloss(nn.Module):
    def forward(self,pos_score,neg_score):
        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss
    
def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)