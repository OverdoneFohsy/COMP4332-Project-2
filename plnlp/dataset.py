import pandas as pd
import torch
import torch_geometric as pyg
import networkx as nx
from collections import defaultdict
import torch_geometric.transforms as T
import sys

def gen_user_index(train_path,valid_path):
    id=set()


    df=pd.read_csv(train_path)
    for idx, row in df.iterrows():
        user_id, friends = row["user_id"], eval(row["friends"])
    #   print(f'{user_id}, {friends}')
        id.update([user_id]+friends)
    df=pd.read_csv(valid_path)
    val_id=[]
    for idx, row in df.iterrows():
        user_id, friends = row["user_id"], eval(row["friends"])
    #   print(f'{user_id}, {friends}')
        for i in [user_id]+friends:
            if i not in id:
                val_id.append(i)
    
    id=sorted(id)
    id+=sorted(val_id)
    # print(len(id))
    d = {'user_id':id,'id':list(range(len(id)))}
    result=pd.DataFrame.from_dict(d)
    result.to_csv('data/id.csv',index=False)
    dd={d['user_id'][i]:d['id'][i] for i in range(len(d['id']))}
    return dd

def get_user_index(path):
    try:
        df=pd.read_csv(path+'/id.csv',header=0)
    except FileNotFoundError():
        return gen_user_index(path+'/train.csv', path+'/valid.csv')
    else:
        id={row['user_id']:row['id'] for idx, row in df.iterrows()}
        return id
    

def load_data(file_name,id_dict):
    """
    read edges from an edge file
    """
    edges = list()
    df = pd.read_csv(file_name)
    for idx, row in df.iterrows():
        user_id, friends = row["user_id"], eval(row["friends"])
        for friend in friends:
            # add each friend relation as an edge
            edges.append((id_dict[user_id], id_dict[friend]))
    edges = sorted(edges)
    
    return edges

def load_test_data(file_name,id_dict):
    """
    read edges from an edge file
    """
    edges = list()
    
    df = pd.read_csv(file_name)
    for idx, row in df.iterrows():
        edges.append((id_dict.get(row["src"],-1), id_dict.get(row["dst"],-1)))
        # scores.append(row["score"])
    src = df['src'].to_list()
    dst= df['dst'].to_list()
    #edges = sorted(edges)
    
    return edges,src,dst

def construct_graph_from_edges(edges):
    """
    generate a directed graph object given true edges
    DiGraph documentation: https://networkx.github.io/documentation/stable/reference/classes/digraph.html
    """
    # convert a list of edges {(u, v)} to a list of edges with weights {(u, v, w)}
    edge_weight = defaultdict(float)
    for e in edges:
        edge_weight[e] += 1.0
    weighed_edge_list = list()
    for e in sorted(edge_weight.keys()):
        weighed_edge_list.append((e[0], e[1], edge_weight[e]))
        
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(weighed_edge_list)
    
    print("number of nodes:", graph.number_of_nodes())
    print("number of edges:", graph.number_of_edges())
    
    return graph

def get_data(dir):
    id_dict=get_user_index(dir)
    train_edges=load_data(dir+'/train.csv', id_dict)
    valid_edges=load_data(dir+'/valid.csv', id_dict)
    test_edges,src,dst=load_test_data(dir+'/test.csv',id_dict)
    g=construct_graph_from_edges(train_edges)
    g.add_weighted_edges_from([[k[0],k[1],1.0] for k in valid_edges])
    data=pyg.utils.from_networkx(g)
    # print(data)
    # print([list(e) for e in g.edges()])
    split={'train':{},'valid':{},'test':{}}
    split['train']['edge']=torch.tensor([list(tup) for tup in train_edges],dtype=torch.long)
    split['train']['weight']=torch.ones(split['train']['edge'].size(0))
    split['valid']['edge']=torch.tensor([list(tup) for tup in valid_edges],dtype=torch.long)
    split['test']['edge']=torch.tensor([list(tup) for tup in test_edges],dtype=torch.long)
    split['test']['src']=src
    split['test']['dst']=dst
    return id_dict,data,split
    
    
    
    # id_dict=get_user_index(id)
    # edges=load_data(train, id_dict)
    # g=construct_graph_from_edges(edges)
    # data=pyg.utils.from_networkx(g)
    # return data

if __name__=="__main__":
    # id_dict=gen_user_index('data/train.csv', 'data/valid.csv')
    # # id_dict=get_user_index('data/id.csv')
    # print(list(id_dict.keys())[8333:8337])
    # train_edges=load_data('data/train.csv', id_dict)
    # val_edges=load_data('data/valid.csv', id_dict)

    # g=construct_graph_from_edges(train_edges)
    # print(g)
    # g.add_weighted_edges_from([[k[0],k[1],1.0] for k in val_edges])
    # print(g)
    # data=pyg.utils.from_networkx(g)
    # print(data)
    # # print([list(e) for e in g.edges()])
    # split={'train':{},'valid':{},'test':{}}
    # split['train']['edges']=torch.tensor([list(tup) for tup in train_edges],dtype=torch.int)
    # split['train']['weight']=torch.ones(split['train']['edges'].size(0))
    # print(split['train'])
    a,b,c=get_data('data')
    print(b.adj_t)
    b = T.ToSparseTensor()(b)
    row, col, _ = b.adj_t.coo()
    d = torch.stack([col, row], dim=0)
    print(b.edge_index)
    print(sys.getsizeof(d))