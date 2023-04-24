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
from model.SAGE import ASCIISAGE
from sklearn.metrics import roc_auc_score
from scorer.loss import CEloss, compute_auc
from tqdm import tqdm


if __name__ == '__main__':
    
    #   default value
    epochs = 500
    cuda_device = 0
    save_pth = './ASCII_saved'
    
    
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
    u, v = g.edges()
    
    
    id2word = dict(zip(word2id.values(), word2id.keys()))
    vec_u = []
    for node in g.nodes():
        char_u = []
        for j in range(len(id2word[node.item()])):
            # print(nodes[i][0])
            char_u.append(ord(id2word[node.item()][j]))
        vec_u.append(char_u)
    
    in_feat = np.asarray(vec_u)
    in_feat = torch.from_numpy(in_feat)
    in_feat = in_feat.type(torch.FloatTensor)
    print(in_feat.shape)
    
    #   generate false walk
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(g.number_of_nodes(),g.number_of_nodes()))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_g = dgl.graph((neg_u, neg_v), num_nodes=g.number_of_nodes())
    
    g = g.to('cuda:0') 
    neg_g = neg_g.to('cuda:0') 
    in_feat = in_feat.cuda(cuda_device)

    #   tools preparation
    pred = DotPredictor()
    model = ASCIISAGE(word2id,22,1024)
    model = model.cuda(cuda_device)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    loss_fn = CEloss()
    
    
    all_logits = []
    for epoch in tqdm(range(epochs)):
        # forward
        h = model(g, in_feat)
        pos_score = pred(g, h)
        neg_score = pred(g, h)
        
        loss = loss_fn(pos_score, neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print('In epoch {}, loss: {}'.format(epoch+1, loss))
            torch.save(model,save_pth+'/'+str(epoch+1)+'_saved.pth')
    
    with torch.no_grad():
        h = model(g, in_feat)
        pos_score = pred(g, h)
        neg_score = pred(g, h)
        print('AUC', compute_auc(pos_score.cpu(), neg_score.cpu()))