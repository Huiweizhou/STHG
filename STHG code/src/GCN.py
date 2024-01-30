import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import numpy as np


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, dropout_p):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, n_hidden))
        self.layers.append(dglnn.GraphConv(n_hidden, n_hidden))
        self.dropout = nn.Dropout(dropout_p)
        self.n_hidden = n_hidden

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = h.view(h.shape[0], -1)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h