import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch


class CEloss(nn.Module):
    def forward(self,pos_score,neg_score):
        score = torch.cat([pos_score, neg_score])
        label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss