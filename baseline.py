import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import pandas as pd
from dataset.word2id import word2id
from dataset.data_load import load_train_data
import os
from dataset.Mydataset import Mydataset
from scorer.pred import DotPredictor
from model.SAGE import GraphSAGE
from sklearn.metrics import roc_auc_score
from scorer.loss import CEloss
from tqdm import tqdm


if __name__ == '__main__':
    
    #   default value
    epochs = 100
    cuda_device = 0
    save_pth = './model_saved'
    
    
    #   load data
    path = './data'
    pth = path + '/' + 'train.csv'
    pth2 = path + '/' + 'valid.csv'
    pth3 = path + '/' + 'test.csv'
    df = pd.read_csv(pth)
    df2 = pd.read_csv(pth2)
    df3 = pd.read_csv(pth3)
    ids = df['user_id']
    ids2 = df2['user_id']
    ids3 = df3["src"]
    
    #   word to id convert
    if not os.path.exists('./data/word2id.npy'):
        word2id = word2id().item()
    else: 
        word2id = np.load('./data/word2id.npy', allow_pickle=True).item()
    length = len(word2id)
    # print(len(word2id))
    
    
    
    # convert the edges
    edges = load_train_data(path)
    id_u = []
    id_v = []
    for i in range(len(edges)):
        u = edges[i][0]
        v = edges[i][1]
        id_u.append(word2id[u])
        id_v.append(word2id[v])
    dataset = Mydataset(id_u,id_v)
    g = dataset[0]
    g = g.cuda(cuda_device)
    u, v = g.edges()
    u = u.cuda(cuda_device)
    v = v.cuda(cuda_device)
    
    node = np.arange(0,len(g.nodes()),1)
    node = torch.from_numpy(node)
    node = node.cuda(cuda_device)
    
    #   generate false walk
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(),g.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())
    neg_g = neg_g.cuda(cuda_device)

    #   tools preparation
    pred = DotPredictor()
    model = GraphSAGE(word2id,1024,1024)
    model = model.cuda(cuda_device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    loss_fn = CEloss()
    
    
    all_logits = []
    for epoch in tqdm(range(epochs)):
        # forward
        h = model(g, node)
        pos_score = pred(g, h)
        neg_score = pred(g, h)
        loss = loss_fn(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print('In epoch {}, loss: {}'.format(epoch, loss))
            print('AUC', roc_auc_score(pos_score, neg_score))
            torch.save(model,save_pth)