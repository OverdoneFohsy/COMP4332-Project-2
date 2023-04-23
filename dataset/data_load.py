import pandas as pd
import numpy as np
import os


def edgeMaker(ids,friends):
    edges = list()
    for idx in range(len(ids)):
        id = ids[idx]
        friend = friends[idx]
        for p in eval(friend):
            edges.append((id,p))
    
    return edges

def load_train_data(path):
    pth = path + '/' + 'train.csv'
    df = pd.read_csv(pth)
    edges = edgeMaker(df['user_id'],df['friends'])
    
    return edges

def load_valid_data(path):
    pth = path + '/' + 'valid.csv'
    df = pd.read_csv(pth)
    edges = edgeMaker(df['user_id'],df['friends'])
    
    return edges

def load_test_data(path):
    edges = list()
    scores = list()
    pth = path + '/' + 'test.csv'
    df = pd.read_csv(pth)
    for idx, row in df.iterrows():
        edges.append((row["src"], row["dst"]))
        scores.append(row["score"])
    
    return edges,scores

# if __name__ == '__main__':
#     path = './data'
#     pth = path + '/' + 'train.csv'
#     pth2 = path + '/' + 'valid.csv'
#     pth3 = path + '/' + 'test.csv'
#     df = pd.read_csv(pth)
#     df2 = pd.read_csv(pth2)
#     df3 = pd.read_csv(pth3)
#     ids = df['user_id']
#     ids2 = df2['user_id']
#     ids3 = df3["src"]
#     print(len(set(ids3)))
#     print(len(set(ids2) | set(ids) | set(ids3)))