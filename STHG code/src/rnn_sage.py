import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import time
import numpy as np
from Sage import SAGE


class RnnSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, time_steps, aggr, dropout_p):
        super(RnnSage, self).__init__()
        self.time_setps = time_steps
        self.R = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

        self.Sage = SAGE(input_dim, hidden_dim, aggr, dropout_p)
        self.rnn = nn.RNN(hidden_dim*2, hidden_dim, nonlinearity='relu')###tpair-rnn

        self.act = nn.ReLU()

    def forward(self, data_list_i, data_list_j):
        states = []
        for t in range(self.time_setps):
            x_i = data_list_i[t][2][0].srcdata['feats']
            x_j = data_list_j[t][2][0].srcdata['feats']
            sage_o_i = self.Sage(data_list_i[t][2], x_i)
            sage_o_j = self.Sage(data_list_j[t][2], x_j)
            sage_o_i = F.normalize(sage_o_i, p=2, dim=1)
            sage_o_j = F.normalize(sage_o_j, p=2, dim=1)
            
            _, idx_i = torch.unique(data_list_i[t][1], return_inverse=True)
            _, idx_j = torch.unique(data_list_j[t][1], return_inverse=True)

            states.append(torch.cat((sage_o_i[idx_i], sage_o_j[idx_j]), -1))
        outs, _ = self.rnn(torch.stack(states))

        #
        return self.out(outs).squeeze(-1)

