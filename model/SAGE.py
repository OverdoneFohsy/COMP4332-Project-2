from dgl.nn import SAGEConv
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
import numpy as np


class GraphSAGE(nn.Module):
    def __init__(self, word2id, in_feats=1024, h_feats=1024):
        super(GraphSAGE, self).__init__()
        
        length = len(word2id)
        self.length = length
        self.embedding = nn.Embedding(num_embeddings=length, embedding_dim=1024)
        
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, node):
        in_feat = self.embedding(node)
        
        # print(in_feat.shape)
        # g = dgl.graph((u, v))
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
