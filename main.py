#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# cancle storing 

import os
import yaml
import json
import argparse
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime
import time
import time
import torch
import torch.optim as optim
from utility import Datasets
from models.BundleGT import BundleGT

#  fix seed

def setup_seed():
    seed = 2022
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed()

def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="Youshu", type=str, help="which dataset to use, options: NetEase, Youshu, iFashion")
    parser.add_argument("-m", "--model", default="CrossCBR", type=str, help="which model to use, options: CrossCBR")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    parser.add_argument( "--folder", default="", type=str, help="take logs into a floder")

    # ML Basic >>>
    parser.add_argument( "--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument( "--l2_reg", default=2e-6, type=float, help="L2 regularization")
    parser.add_argument( "--embedding_size", default=64, type=int, help="Embedding size")
    parser.add_argument( "--batch_size_train", default=2048, type=int, help="the size of train batch")
    parser.add_argument( "--batch_size_test", default=2048, type=int, help="the size of test batch")
    parser.add_argument( "--early_stopping", default=40, type=int, help="early stopping strategy")
    # ML Basic <<<

    parser.add_argument( "--num_ui_layers", default=4, type=int, help="the number of ui_layers")
    parser.add_argument( "--num_trans_layers", default=1, type=int, help="the number of transformer layers")

    # GCN >>>
    parser.add_argument( "--gcn_norm", default=False, type=lambda x: int(x)!=0, help="")
    parser.add_argument( "--layer_alpha", default=None, type=lambda x: eval(x), help="[1,1,1]")
    # GCN <<<

    # Attention >>>
    parser.add_argument( "--dropout_ratio", default=0.0, type=float, help="dropout_ratio")
    parser.add_argument( "--num_token", default=0, type=int, help="(0, Maximum], 0 means keep the whole sequence")
    parser.add_argument( "--ub_alpha", default=0.5, type=float, help="")
    parser.add_argument( "--bi_alpha", default=0.5, type=float, help="")
    # Attention <<<

    args = parser.parse_args()

    return args


def main():
    conf_ori = yaml.safe_load(open("./config.yaml"))
    print("load config file done!")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    if "_" in dataset_name:
        conf = conf_ori[dataset_name.split("_")[0]]
    else:
        conf = conf_ori[dataset_name]
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    
    # model's defualt configures
    if conf["model"] in conf_ori:
      for i in conf_ori[conf["model"]]:
        conf[i] = conf_ori[conf["model"]][i]

    # manual configures
    for k in paras:
        conf[k] = paras[k]
    
    dataset = Datasets(conf)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("cuda available:",torch.cuda.is_available())
    conf["device"] = device

    
    log_path = "./log/%s/%s" %(conf["dataset"], conf["model"])
    if conf["folder"] != "":
        log_path = log_path + "/%s" %(conf["folder"])
    checkpoint_model_path = "./checkpoints/%s/%s/model" %(conf["dataset"], conf["model"])
    checkpoint_conf_path = "./checkpoints/%s/%s/conf" %(conf["dataset"], conf["model"])
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    if not os.path.isdir(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)
    if not os.path.isdir(checkpoint_conf_path):
        os.makedirs(checkpoint_conf_path)
    
    conf["lr"] = eval(conf["lr"]) if type(conf["lr"]) is str else conf["lr"]
    conf["l2_reg"] = eval(conf["l2_reg"]) if type(conf["l2_reg"]) is str else conf["l2_reg"]

    settings = []
    if conf["info"] != "":
        settings += [conf["info"]]

    settings += [
        "Neg_%d" %(conf["neg_num"]), 
        "BS_", str(conf["batch_size_train"]), 
        "lr_", str(conf["lr"]), 
        "l2_", str(conf["l2_reg"]), 
        "emb_", str(conf["embedding_size"]),
        ]

    setting = "_".join(settings)
    log_path = log_path + "/" + setting
    base_checkpoint_model_path = checkpoint_model_path
    checkpoint_model_path = checkpoint_model_path + "/" + setting
    checkpoint_conf_path = checkpoint_conf_path + "/" + setting
        
    if conf['model'] == 'BundleGT':
        model = BundleGT(conf, dataset.graphs).to(device)
    else:
        raise ValueError("Unimplemented model %s" %(conf["model"]))
    
    with open(log_path, "a") as log:
        log.write("%s\n" %(conf))
    
    optimizer = optim.Adam(model.parameters(), lr=conf["lr"])

    batch_cnt = len(dataset.train_loader)
    test_interval_bs = int(batch_cnt * conf["test_interval"])

    best_metrics, best_perform = init_best_metrics(conf)
    best_epoch = 0
    previous_loss = 1e8
    setup_seed()
    if conf['early_stopping'] > 0 :
        conf['epochs'] = 10000
    for epoch in range(conf['epochs']):
        epoch_anchor = epoch * batch_cnt
        model.train(True)
        pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))
        avg_loss = {}
        s_time = time.time()
        for batch_i, batch in pbar:
            model.train(True)
            optimizer.zero_grad()
            batch = [x.to(device) for x in batch]
            batch_anchor = epoch_anchor + batch_i

            losses = model(batch)
            loss = losses["loss"]
            loss.backward()
            optimizer.step()

            loss_scalar = loss.detach().cpu()
            for l in losses:
                if not l in avg_loss: avg_loss[l] = []
                avg_loss[l].append(losses[l].detach().cpu())

            pbar.set_description(
                "\033[1;41m[%.4f]\033[0m"%best_metrics["test"]["recall"][20] +\
                " epoch: %d, %s" %(epoch, 
                                    ", ".join(["%s: %.4f"%(l, losses[l]) for l in losses]))
                )
            
            if (batch_anchor+1) % test_interval_bs == 0:  
                metrics = {}
                metrics["val"] = test(model, dataset.val_loader, conf)
                metrics["test"] = test(model, dataset.test_loader, conf)
                best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)
                    
        if conf['early_stopping'] > 0 and (epoch - best_epoch) >= conf['early_stopping']:
            with open(log_path, "a") as f:
                str_ = "early stopping!"
                f.write(str_)
            break
        
        with open(log_path, "a") as f:

            loss_str = "EPOCH: {}, {}, Time: {:.5f}\n".format(epoch,
                                                        ', '.join(['avg.%s: %.5f'%(l, np.mean(avg_loss[l])) for l in avg_loss])
                                                        , time.time() - s_time)
            avg_loss = []
            f.write(loss_str)
            print(loss_str)

def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform

def write_log( log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()

    print(val_str)
    print(test_str)

def log_metrics(conf, model, metrics, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    topk_ = 20 
    print("top%d as the final evaluation standard" %(topk_))
    if metrics["val"]["recall"][topk_] > best_metrics["val"]["recall"][topk_] and metrics["val"]["ndcg"][topk_] > best_metrics["val"]["ndcg"][topk_]:
        # torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            print(best_perform["val"][topk])
            if topk == topk_:
                
                print("\033[1;31m" + best_perform["test"][topk] + "\033[0m") 
            else:
                print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")
    
    print("[!]\033[1;41m", best_perform["test"][topk_], "\033[0m")
    log.write("[!]" + best_perform["test"][topk_]  + "\n")

    log.close()

    return best_metrics, best_perform, best_epoch


def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate()
    t_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_i, batch in t_bar:
        s_time = time.time()
        users, ground_truth_u_b, train_mask_u_b = batch
        pred_b = model.evaluate(rs, users.to(device)).to('cpu')
        e_time = time.time() - s_time
        pred_b = pred_b - 1e8 * train_mask_u_b
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])
        d_time = time.time()-  s_time
        t_bar.set_description("e_time: %.5f d_time: %.5f" %(e_time, d_time))

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics


def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)
        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
