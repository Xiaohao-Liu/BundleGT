#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .HGT import HGT

eps = 1e-9


class BundleGT(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        # Recommendation Basic <<<
        self.embedding_size = conf["embedding_size"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]

        self.eval_bundles = torch.arange(self.num_bundles).to(
            self.device).long().detach().view(1, -1)
        self.eval_users = torch.arange(self.num_users).to(
            self.device).long().detach().view(1, -1)
        # Recommendation Basic <<<

        self.num_ui_layers = conf["num_ui_layers"]  # [1,2,3]
        self.num_trans_layers = conf["num_trans_layers"]  # [1,2,3]

        # GCN Configuration >>>
        self.gcn_norm = self.conf["gcn_norm"]
        self.layer_alpha = self.conf['layer_alpha']  # None
        # GCN Configuration <<<

        # ML Basic >>>
        self.embed_L2_norm = conf["l2_reg"]
        # ML Basic <<<

        # Attention part >>>
        # 0 means keep the default
        self.num_token = conf["num_token"] if "num_token" in conf else 0

        self.dropout_ratio = conf["dropout_ratio"]
        # <<< Attention part

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        self.user_bundle_cf_count = self.ui_graph.sum(axis=1)

        self.HGT = HGT(conf={
            "n_user": self.num_users,
            "n_item": self.num_items,
            "n_bundle": self.num_bundles,
            "dim": self.embedding_size,
            "n_ui_layer": self.num_ui_layers,
            "n_trans_layer": self.num_trans_layers,
            "gcn_norm": self.gcn_norm,
            "layer_alpha": self.layer_alpha,
            "num_token": self.num_token,
            "device": self.device,
            "data_path": self.conf["data_path"],
            "dataset": self.conf["dataset"],
            "head_token": False,
            "dropout_ratio": self.dropout_ratio,
            "ub_alpha": self.conf["ub_alpha"],
            "bi_alpha": self.conf["bi_alpha"],
        },
            data={
                "graph_ui": self.ui_graph,
                "graph_ub": self.ub_graph,
                "graph_bi": self.bi_graph,
        }
        )

        self.MLConf = {}
        for k in ['lr', 'l2_reg', 'embedding_size', 'batch_size_train', 'batch_size_test', 'early_stopping']:
            self.MLConf[k] = self.conf[k]
        print("[ML Configuration]", self.MLConf)
        print("[HGT Configuration]", self.HGT.conf)

    def propagate(self):
        return self.HGT()

    def forward(self, batch):
        losses = {}
        users, bundles = batch

        users_feature, _, bundles_features = self.propagate()
        self.batch_size = users.shape[0]

        i_u = users_feature[users]
        i_b = bundles_features[bundles]

        score = torch.sum(torch.mul(i_u, i_b), dim=-1)

        loss = torch.mean(torch.nn.functional.softplus(
            score[:, 1] - score[:, 0]))

        l2_loss = self.embed_L2_norm * self.HGT.reg_loss()
        loss = loss + l2_loss
        losses["l2"] = l2_loss.detach()

        losses["loss"] = loss

        return losses

    def evaluate(self, propagate_result, users):

        users_feature, _, bundles_feature = propagate_result
        users_embedding = users_feature[users]
        scores = torch.mm(users_embedding, bundles_feature.t())

        return scores
