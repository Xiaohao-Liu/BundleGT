
from typing import OrderedDict
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

eps = 1e-9


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(
        indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


class SublayerConnection(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = {
            "dim": 64,
            "residual": False,
            "layernorm": False,
            "dropout": 0,
            "device": None
        }

        for i in conf:
            self.conf[i] = conf[i]
        self.device = self.conf['device']
        self.ln = nn.LayerNorm(
            self.conf["dim"], eps=eps, elementwise_affine=False).to(self.device)
        self.dropout = nn.Dropout(self.conf['dropout'])

    def forward(self, x, sublayer):
        y = self.dropout(sublayer(x))
        return y


class SelfAttention(nn.Module):
    def __init__(self, conf, data={}):
        super().__init__()
        self.conf = {
            "dim": 64,
            "n_head": 1,
            "device": None,
            "dropout_ratio": 0,
        }

        for i in conf:
            self.conf[i] = conf[i]
        self.device = self.conf["device"]

        self.w_q = nn.Linear(
            self.conf['dim'], self.conf['dim'], bias=False).to(self.device)
        self.w_k = nn.Linear(
            self.conf['dim'], self.conf['dim'], bias=False).to(self.device)

        self.dropout = nn.Dropout(self.conf['dropout_ratio'])

    def multiHeadAttention(self, x, mask, dims):
        bs, N_bs, heads, N_max, d = dims

        q = self.w_q(x)
        k = self.w_k(x)

        q = q.view(-1, N_bs, N_max, heads, d//heads).transpose(2, 3)
        k = k.view(-1, N_bs, N_max, heads, d//heads).transpose(2, 3)
        A = q.mul(d ** -0.5) @ k.transpose(-2, -1)

        att = A.masked_fill(mask.view(-1, N_bs, 1, 1, N_max), -1e9)
        att = att.softmax(dim=-1)

        v = x.view(-1, N_bs, N_max, heads, d//heads).transpose(2, 3)
        y = att @ v  # [bs, N_bundle, heads, N_max, d']

        y = y.transpose(2, 3).contiguous().view(-1, N_bs, N_max, d)

        return y

    def forward(self, x, mask, dims):
        bs, N_bs, heads, N_max, d = dims
        assert tuple(x.shape) == (bs, N_bs, N_max, d)
        assert tuple(mask.shape) == (bs, N_bs, 1, N_max)

        x = self.dropout(self.multiHeadAttention(x, mask, dims))

        return x


class LiT(nn.Module):
    def __init__(self, conf, data):
        super().__init__()
        self.conf = {
            "n_layer": 3,
            "dim": 64,
            "num_sequence": None,
            "num_token": 70,
            "n_head": 1,
            "dropout_ratio": 0,
            "data_path": None,
            "dataset": None,
            "device": None,
            "tag": ""
        }
        self.data = {
            "sp_graph": None,
        }
        for i in conf:
            self.conf[i] = conf[i]
        self.device = self.conf["device"]
        for i in data:
            self.data[i] = data[i]

        self.sp_graph = self.data["sp_graph"]
        self.num_sequence = self.conf["num_sequence"]
        self.graph = to_tensor(self.sp_graph).to(self.device)

        self.__generate_sequences()

        self.attn_encode = [SelfAttention(
            conf=self.conf
        ) for _ in range(self.conf["n_layer"])]

    def __generate_sequences(self):
        sequence = [i.coalesce().indices().squeeze(0) for i in self.graph]

        pad_token = self.graph.shape[1] + 1
        sequence = pad_sequence(
            sequence,
            batch_first=True,
            padding_value=pad_token)

        if self.conf["num_token"] > 0 and self.conf["num_token"] < sequence.shape[1]:
            indices = torch.arange(
                self.conf["num_token"]).to(self.device).long()
            sequence = torch.index_select(sequence, 1, indices)

        self.mask = (sequence == pad_token).detach()  # [ #bundles, N_max]
        self.sequence = sequence.masked_fill(self.mask, 0)

        self.len = torch.sum(1-self.mask.float(), dim=-1)
        self.N_max = self.mask.shape[1]

    def forward(self, e, e_residual, alpha, idx, layer=0):
        N_bs = idx.shape[1]
        dims = (
            idx.shape[0],
            idx.shape[1],
            self.conf["n_head"],
            self.N_max,
            self.conf["dim"]
        )

        seq = self.sequence[idx]  # [bs, #bs, N_max]
        mask = self.mask[idx].unsqueeze(2)  # [bs, #bs, 1, N_max]
        len = self.len[idx].unsqueeze(2)  # [bs, #bs, 1]
        N_max = self.N_max

        x = e[seq]  # [bs, #bs, N_max, d]

        x = self.attn_encode[layer](
            x,
            mask=mask,
            dims=dims
        )

        x = x.masked_fill(mask.view(-1, N_bs, N_max, 1), 0)

        y = x.sum(-2) / (len + eps)  # mean-pooling, [bs, N_bs, N_max, d]

        y = alpha * y + (1-alpha) * e_residual  # residual

        return y
