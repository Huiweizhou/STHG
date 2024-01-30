import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 heads,
                 activation=F.elu,
                 out_drop=0.0,
                 feat_drop=.5,
                 attn_drop=.0,
                 negative_slope=0.2,
                 residual=True,
                 num_layers=1,
                 ):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.drop = nn.Dropout(out_drop)
        if num_layers > 1:
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    num_hidden * heads[l - 1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                num_hidden * heads[-2], num_hidden, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATConv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l, block in enumerate(g):
            h = self.gat_layers[l](block, h)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
            # if l != self.num_layers - 1:
                # h = F.relu(h)
                # h = self.drop(h)
        return h
