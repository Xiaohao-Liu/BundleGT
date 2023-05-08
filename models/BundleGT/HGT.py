import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import copy
from .LiT import LiT
from .LGCN import LGCN


eps = 1e-9


class HGT(nn.Module):
    def __init__(self, conf, data):
        super().__init__()
        self.conf = {
            # basic
            "n_user": None,
            "n_item": None,
            "n_bundle": None,
            "dim": 64,
            "n_ui_layer": 3,
            "n_trans_layer": 3,
            "device": None,
            # gcn
            "gcn_norm": False,
            "layer_alpha": None,
            # transformer
            "num_token": 5,
            "n_head": 1,
            "data_path": None,
            "dataset": None,
            "dropout_ratio": 0.0,
            "ub_alpha": 0.5,
            "bi_alpha": 0.5,
        }
        self.data = {
            "graph_ui": None,
            "graph_ub": None,
            "graph_bi": None,
        }

        for i in conf:
            self.conf[i] = conf[i]
        for i in data:
            self.data[i] = data[i]
        self.device = self.conf["device"]

        self.init_emb()

        LiT_bi_conf = copy(self.conf)
        LiT_bi_conf["n_layer"] = self.conf["n_trans_layer"]
        LiT_bi_conf["num_sequence"] = LiT_bi_conf["n_bundle"]
        self.LiT_bi = LiT(
            conf=LiT_bi_conf,
            data={
                "sp_graph": self.data["graph_bi"],
            }
        )

        avg_ub_seq = {"Youshu": 70, "NetEase": 30, "iFashion": 5}
        LiT_ub_conf = copy(self.conf)
        LiT_ub_conf["n_layer"] = self.conf["n_trans_layer"]
        LiT_ub_conf["num_sequence"] = LiT_ub_conf["n_user"]
        LiT_ub_conf["tag"] = 'ub'
        LiT_ub_conf["num_token"] = avg_ub_seq[LiT_ub_conf['dataset']]
        self.LiT_ub = LiT(
            conf=LiT_ub_conf,
            data={
                "sp_graph": self.data["graph_ub"],
            }
        )

        LGN_ui_conf = copy(self.conf)
        LGN_ui_conf["n_layer"] = self.conf["n_ui_layer"]
        self.LGN_ui = LGCN(
            conf=LGN_ui_conf, data={
                "graph": self.data["graph_ui"]
            }
        )

        Agg_ub_conf = copy(self.conf)
        Agg_ub_conf["n_layer"] = self.conf["n_trans_layer"]
        self.Agg_ub = LGCN(
            conf=Agg_ub_conf, data={
                "graph": self.data["graph_ub"]
            }
        )

        Agg_bi_conf = copy(self.conf)
        Agg_bi_conf["n_layer"] = self.conf["n_trans_layer"]
        self.Agg_bi = LGCN(
            conf=Agg_bi_conf, data={
                "graph": self.data["graph_bi"]
            }
        )

    def init_emb(self):
        self.user_embedding = nn.Parameter(
            torch.FloatTensor(self.conf["n_user"], self.conf["dim"]).to(self.device))
        nn.init.xavier_uniform_(self.user_embedding, gain=1)

        self.item_embedding = nn.Parameter(
            torch.FloatTensor(self.conf["n_item"], self.conf["dim"]).to(self.device))
        nn.init.xavier_uniform_(self.item_embedding, gain=1)

        self.bundle_embedding = nn.Parameter(
            torch.FloatTensor(self.conf["n_bundle"], self.conf["dim"]).to(self.device))
        nn.init.xavier_uniform_(self.bundle_embedding, gain=1)

        self.eval_bundles = torch.arange(self.conf["n_bundle"]).to(
            self.device).long().detach().view(1, -1)
        self.eval_users = torch.arange(self.conf["n_user"]).to(
            self.device).long().detach().view(1, -1)

    def forward(self):
        # return self.forwar_ub()

        u_feature = self.user_embedding
        i_feature = self.item_embedding
        b_feature = self.bundle_embedding

        u_feature_cf, i_feature_cf = self.LGN_ui(
            u_feature, i_feature, layer=None)

        for layer in range(self.conf["n_trans_layer"]):

            u_b2u, b_u2b = self.Agg_ub(u_feature, b_feature, layer=layer)
            b_i2b, i_b2i = self.Agg_bi(b_feature, i_feature, layer=layer)

            i_feature = i_feature_cf + i_b2i

            b_feature_T = self.LiT_bi(
                i_feature, b_i2b, self.conf["bi_alpha"], self.eval_bundles, layer=layer).squeeze(0)
            b_feature = b_feature_T + b_u2b

            u_feature_T = self.LiT_ub(
                b_feature, u_b2u, self.conf["ub_alpha"], self.eval_users, layer=layer).squeeze(0)
            u_feature = u_feature_T + u_feature_cf

        return u_feature, i_feature, b_feature

    def reg_loss(self):
        u_rl = (self.user_embedding ** 2).sum()
        i_rl = (self.item_embedding ** 2).sum()
        b_rl = (self.bundle_embedding ** 2).sum()
        return u_rl + i_rl + b_rl
