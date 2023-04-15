import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from copy import copy

from .LiT import LiT

eps = 1e-9


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(
        indices), torch.FloatTensor(values), torch.Size(graph.shape))
    return graph


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + eps))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + eps))
    graph = rowsum_sqrt @ graph @ colsum_sqrt
    return graph


class LGCN(nn.Module):
    def __init__(self, conf, data):
        super().__init__()
        self.conf = {
            "dim": 64,
            "device": None,
            "n_user": None,
            "n_item": None,
            "layer_alpha": None,
            "gcn_norm": False,
        }

        self.data = {
            "graph": None,
            "user_embedding": None,
            "item_embedding": None
        }

        for i in conf:
            self.conf[i] = conf[i]
        self.device = self.conf["device"]
        for i in data:
            self.data[i] = data[i]

        self.conf["n_user"] = self.data['graph'].shape[0]
        self.conf["n_item"] = self.data['graph'].shape[1]

        if self.data["user_embedding"] is None:
            self.user_embedding = nn.Parameter(
                torch.FloatTensor(self.conf["n_user"], self.conf["dim"]).to(self.device))
            nn.init.xavier_uniform_(self.user_embedding, gain=1)
        if self.data["item_embedding"] is None:
            self.item_embedding = nn.Parameter(
                torch.FloatTensor(self.conf["n_item"], self.conf["dim"]).to(self.device))
            nn.init.xavier_uniform_(self.user_embedding, gain=1)

        assert self.conf["layer_alpha"] is None or len(
            self.conf["layer_alpha"]) == self.conf["n_layer"]
        if self.conf["layer_alpha"] is None:
            self.conf["layer_alpha"] = [1 for _ in range(self.conf["n_layer"])]

        self.graph = self.__get_laplace_graph(self.data["graph"])

    def __get_laplace_graph(self, graph):
        graph2 = sp.bmat([
            [None, graph],
            [graph.T, None]
        ])

        return to_tensor(laplace_transform(graph2)).to(self.device)

    def forward(self, user_embedding=None, item_embedding=None, layer=None):
        if user_embedding is None:
            user_embedding = self.user_embedding
        if item_embedding is None:
            item_embedding = self.item_embedding
        features = torch.cat((
            user_embedding,
            item_embedding,
        ), 0)

        if layer is None:
            all_features = [features]
            for l in range(self.conf["n_layer"]):
                features = torch.spmm(self.graph, features)
                if self.conf["gcn_norm"]:
                    features = features / (l+1)
                    all_features.append(F.normalize(features, p=2, dim=-1))
                else:
                    all_features.append(features)
            features = torch.sum(torch.stack(all_features, dim=1), dim=1)
        else:
            features = torch.spmm(self.graph, features)

        return torch.split(
            features, (user_embedding.shape[0], item_embedding.shape[0]), 0)
