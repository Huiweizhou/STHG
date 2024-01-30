import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import time
import numpy as np
from Sage import SAGE
from torch import Tensor
from typing import Optional, Any
from time2vec import SineActivation, CosineActivation
import pdb

class TransformerEncoderLayer_qk(nn.TransformerEncoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_qk, self).__init__(d_model=d_model, nhead=nhead)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.modules.transformer._get_activation_fn(activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_qk, self).__setstate__(state)

    def forward(self, src: Tensor, new_v: Tensor, src_mask:Optional[Tensor] =  None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        src2 = self.self_attn(src, src, new_v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]#QK,V
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder_qk(nn.TransformerEncoder):

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder_qk, self).__init__(encoder_layer=encoder_layer, num_layers=num_layers)
        self.layers = nn.modules.transformer._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, new_v: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        output = src

        for mod in self.layers:
            output = mod(output, new_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class STHG(nn.Module):
    def __init__(
            self, input_dim, hidden_dim, out_dim,
            attn_wnd, time_steps, aggr, dropout_p,
            nhead, num_layers, device,batch_size
    ):
        super(STHG, self).__init__()
        self.time_steps = time_steps
        self.attn_wnd = attn_wnd if attn_wnd!=-1 else time_steps
        self.device = device

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=nhead, dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.masks = torch.triu(torch.full((self.time_steps, self.time_steps), float('-inf')), diagonal=1).to(device)
        re_masks = torch.triu(torch.full((self.time_steps, self.time_steps), float('-inf')), diagonal=self.attn_wnd)
        self.masks += re_masks.transpose(0, 1).to(device)
        self.out = nn.Linear(hidden_dim * 2, out_dim, bias=False)
        self.Sage = SAGE(input_dim, hidden_dim, aggr, 0.5)
        self.act = nn.ReLU()

        self.multihead_attn = nn.MultiheadAttention(hidden_dim * 2, nhead)

        encoder_layer_qk = TransformerEncoderLayer_qk(d_model=hidden_dim * 2, nhead=nhead, dropout=dropout_p)
        self.transformer_encoder_qk = TransformerEncoder_qk(encoder_layer_qk, num_layers=num_layers)

        self.time2vec = SineActivation(1, hidden_dim * 2)

        self.W_half = nn.Linear(hidden_dim * 4, hidden_dim * 2)

        self.batch_size=batch_size



    def forward(self, data_list_i, data_list_j):
        temporal_pairs_embeddings = []
        for t in range(self.time_steps):
            x_i = data_list_i[t][2][0].srcdata['feats']
            x_j = data_list_j[t][2][0].srcdata['feats']
            sage_o_i = self.Sage(data_list_i[t][2], x_i)
            sage_o_j = self.Sage(data_list_j[t][2], x_j)
            sage_o_i = F.normalize(sage_o_i, p=2, dim=1)
            sage_o_j = F.normalize(sage_o_j, p=2, dim=1)
            _, idx_i = torch.unique(data_list_i[t][1], return_inverse=True)
            _, idx_j = torch.unique(data_list_j[t][1], return_inverse=True)
            temporal_pairs_embeddings.append(torch.cat((sage_o_i[idx_i], sage_o_j[idx_j]), -1))

        temporal_pairs_embeddings = torch.stack(temporal_pairs_embeddings)

        temporal_pairs_embeddings_xs = temporal_pairs_embeddings.transpose(0, 1)
        outs_zs = self.transformer_encoder(temporal_pairs_embeddings_xs).transpose(0, 1)
        temporal_pairs_embeddings_t2v = temporal_pairs_embeddings
        t2v = []
        for i in range(self.time_steps):
            t2v.append(self.time2vec(torch.Tensor([[i + 1]]).to(self.device)))
        t2v = torch.stack(t2v)
        t2v_b = torch.broadcast_to(t2v, temporal_pairs_embeddings_t2v.shape)
        temporal_pairs_embeddings_t2v=temporal_pairs_embeddings_t2v+t2v_b
        outs = self.transformer_encoder_qk(temporal_pairs_embeddings_t2v, outs_zs, mask=self.masks)

        return self.out(outs).squeeze(-1)


