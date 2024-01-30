import torch
import torch.nn as nn
import torch.nn.functional as F
from Sage import SAGE
import math


class RnnAtt(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, time_steps, aggr, attn_wnd, dropout_p, device):
        super(RnnAtt, self).__init__()
        self.time_steps = time_steps
        self.attn_wnd = attn_wnd if attn_wnd != -1 else time_steps
        #self.gcn=GCN(input_dim,hidden_dim,dropout_p)
        self.Sage = SAGE(input_dim, hidden_dim, aggr, dropout_p)
        # RNN LSTM
        self.rnn = nn.RNN(hidden_dim * 2, hidden_dim, nonlinearity='relu')
        ##self.rnn = nn.LSTM(hidden_dim * 2, hidden_size=hidden_dim)
        self.act = nn.ReLU()#
        #
        nhead=1
        self.att=nn.MultiheadAttention(hidden_dim,nhead,dropout=dropout_p)
        #
        self.masks = torch.triu(torch.full((self.time_steps, self.time_steps), float('-inf')), diagonal=1).to(device)
        re_masks = torch.triu(torch.full((self.time_steps, self.time_steps), float('-inf')), diagonal=self.attn_wnd)
        self.masks += re_masks.transpose(0, 1).to(device)
        #
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, data_list_i, data_list_j):
        # print(np.random.randint(5, 10))
        # time.sleep(1)
        states = []
        for t in range(self.time_steps):
            x_i = data_list_i[t][2][0].srcdata['feats']
            x_j = data_list_j[t][2][0].srcdata['feats']
            sage_o_i = self.Sage(data_list_i[t][2], x_i)
            sage_o_j = self.Sage(data_list_j[t][2], x_j)
            sage_o_i = F.normalize(sage_o_i, p=2, dim=1)
            sage_o_j = F.normalize(sage_o_j, p=2, dim=1)

            _, idx_i = torch.unique(data_list_i[t][1], return_inverse=True)
            _, idx_j = torch.unique(data_list_j[t][1], return_inverse=True)

            states.append(torch.cat((sage_o_i[idx_i], sage_o_j[idx_j]), -1))

        states_t=torch.stack(states)
        output_rnn, _ = self.rnn(states_t)
        #print("output_rnn",output_rnn.shape)

        ##output_rnn = self.act(output_rnn)

        att_in = output_rnn

        ''', attn_mask=self.masks'''
        outs, attn_weights = self.att(output_rnn, output_rnn, output_rnn, attn_mask=self.masks)
        #print("attn_output", attn_output.shape)


        return self.out(outs).squeeze(-1)


