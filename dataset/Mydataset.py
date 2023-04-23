import pandas as pd
import dgl
from dgl.data import DGLDataset
import torch
import os
import numpy as np


class Mydataset(DGLDataset):
    def __init__(self, id_u, id_v):

        u = torch.from_numpy(np.asarray(id_u))
        v = torch.from_numpy(np.asarray(id_v))
        self.u = u
        self.v = v
        self.graph = dgl.graph((u, v))
        # self.graph.ndata['feat'] = np.array(1024)
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1