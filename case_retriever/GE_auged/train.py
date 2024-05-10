from transformers import AutoModel,AutoTokenizer, AutoConfig,BertTokenizer,BertConfig,RobertaTokenizer,RobertaConfig
from omegaconf import OmegaConf
import wandb
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk, Dataset
from torch.optim.lr_scheduler import _LRScheduler,CosineAnnealingWarmRestarts
from torch import optim
from torch.nn.utils import clip_grad_norm_

from tqdm.auto import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import random
import logging
import os
import torch.nn.functional as F
import json

from config_bienc import parameters as conf

from utils import get_dev_sample,get_sample,Similarity,Contrastive_Loss,seed_everything,AverageMeter,myDataLoader
from model import Bert_model


def calculate_metric(doc_similarity, doc_labels,topk,top3):
    
    '''
    doc_similarity : torch.tensor
    doc_labels : list of label inds
    topk: how many topk
    '''

    all_recall_topk = 0.0
    all_recall_top3 = 0.0
    all_precision_topk = 0.0
    all_precision_top3 = 0.0
    data_size= len(doc_similarity)

    res_size = 0
    for i in tqdm(range(data_size),desc = f'calc_score_top{topk}'):
        n_trues = len(doc_labels[i])
        if n_trues == 0:
            continue

        top_num = min(topk, len(doc_similarity[i])) # 항상 10가 된다.
        _, topk_indices = doc_similarity[i].topk(top_num)
        n_covered_top_k = sum([ids in doc_labels[i] for ids in topk_indices])
        
        recall_topk = n_covered_top_k / n_trues
        precision_topk = n_covered_top_k / topk

        recall_topk = n_covered_top_k / n_trues
        precision_topk = n_covered_top_k / topk
        
        all_recall_topk += recall_topk
        all_precision_topk += precision_topk


        ############## TOP3 ######################
        _, top3_indices = doc_similarity[i].topk(top3)
        n_covered_top_3 = sum([ids in doc_labels[i] for ids in top3_indices])
        
        recall_top3 = n_covered_top_3 / n_trues
        precision_top3 = n_covered_top_3 / top3
        
        all_recall_top3 += recall_top3
        all_precision_top3 += precision_top3
        res_size += 1

    return float(all_recall_topk/res_size),float(all_precision_topk/res_size), float(all_recall_top3/res_size),float(all_precision_top3/res_size)

def evaluate(model,d_loader,topk):
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    all_pred = []
    all_cand = []
    gold_inds = []
    sim_calculator = Similarity(1)
    model.eval()
    d_loader.reset()
    with torch.no_grad():
        for batch in tqdm(d_loader):
            input_ids, attention_mask,gold = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['gold_inds']
            # cand_input_ids, cand_attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            
            hidden = model(input_ids,attention_mask)
            # cand_hidden = model(cand_input_ids,cand_attention_mask)

            all_pred.append(hidden)
            all_cand.append(hidden)
            for gd in gold:
                if gd:
                    gold_inds.append(gd)
                else:
                    gold_inds.append([])
            del input_ids, attention_mask,hidden #  cand_input_ids, cand_attention_mask, cand_hidden
            
    all_pred = torch.cat(all_pred,0)
    all_cand = torch.cat(all_cand,0)

    all_pred = sim_calculator(all_pred.unsqueeze(1), all_cand.unsqueeze(0)) # (sample hidden) @ (hidden sample) -> (sample, sample)
    
    all_pred.diagonal().fill_(0)
    all_pred = all_pred.detach().cpu()
    recall_topk, precision_topk, recall_top3, precision_top3 = calculate_metric(all_pred,gold_inds,topk,3)

    # recall_top3, precision_top3 = calculate_metric(all_pred,gold_inds,3)
    model.train()
    
    return model, recall_topk, precision_topk,recall_top3, precision_top3

def get_positive_weight(label):     # compute positive-negative ratio in batch
    num_positives = sum(label)
    num_negatives = len(label)-num_positives
    if num_positives == 0:          # when batch is composed of all negatives or all positives, apply constant positive weight to loss function (assumption)
        return conf.neg_ratio
    else:
        return num_negatives/num_positives
    
def train():
    ## Device
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(device)
    print("conf.train.seed",conf.seed)

    #####################  data prepare   #####################
    with open(f'{conf.train_file}','r') as f:
        data = json.load(f)
    train_example = get_sample(data,tokenizer)
    train_dataloader = myDataLoader(True,train_example,conf.batch_size)
    
    with open(f'{conf.valid_file}','r') as f:
        dev = json.load(f)
    print('data keys', dev[0].keys())
    dev_example = get_dev_sample(dev,tokenizer)
    dev_dataloader = myDataLoader(False,dev_example,conf.batch_size_test)
    
    #####################  model prepare   #####################
    model = Bert_model(conf.model_type,conf.bert_size,conf.cache_dir,model_config.hidden_size,tokenizer)
    if conf.checkpoint:
        model.load_state_dict(torch.load(conf.backbone))
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },

        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            },
    ]

    #####################  Other operator   #####################
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=conf.learning_rate,weight_decay=0.01,eps = 1e-8)
    
    # loss_func = Contrastive_Loss(use_margin = False,temperature = 0.05)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-8)
    calc_sim = Similarity(0.05)
    model = nn.DataParallel(model)
    model.to(device)
    model.train()
    losses = AverageMeter()
    best_precision = 0.0

    #####################  TRAIN START   #####################
    for ep in range(conf.epoch):
        tbar1 = tqdm(train_dataloader)

        for idx, batch in enumerate(tbar1):
            batch = {k: v.to(device) for k, v in batch.items() if k not in  ['gold_inds','origin_index']}
            org,cand = model(**batch)
            logit =calc_sim(org,cand) 
            pos_weight = get_positive_weight(batch['label'])
            loss_func = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)

            loss = loss_func(logit,batch['label'])
            
            clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.update(loss.item(), len(batch))
            del batch,loss

            if idx % 5 == 4:
                tb_x = ep * len(tbar1) + idx + 1
                # tbar1.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)
                wandb.log({f'train/loss':losses.avg,f'train/step' :tb_x,'learning_rate' :optimizer.param_groups[0]['lr']})

        print(f'EPOCH: {ep + 1}, avg loss: {losses.avg}')

        model,recall_topk, precision_topk,recall_top3, precision_top3 = evaluate(model,dev_dataloader,conf.topk)
        
        wandb.log({f'val/recall@{conf.topk}' : recall_topk, f'val/precision@{conf.topk}' : precision_topk,'val/epoch' : ep + 1,
                    f'val/recall@3': recall_top3 , f"val_precision@3" : precision_top3})
                
        print(f'val/recall@{conf.topk}:  {recall_topk}, val/precision@{conf.topk} : {precision_topk} val/epoch :{ep + 1} val/recall@3 : {recall_top3} , val/precision@3 :{precision_top3}')
        
        if precision_topk > best_precision:
            best_precision = precision_topk
            model_name = f'E_{ep+1}_{conf.exp_name}_precision@{conf.topk}_{round(best_precision,3)}.pt'
            torch.save(model.module.state_dict(), conf.output_path + model_name)
        train_dataloader.reset()



if __name__ == '__main__':
    if conf.model_type == 'bert':
        tokenizer = BertTokenizer,BertConfig.from_pretrained(conf.bert_size)
        model_config = BertConfig.from_pretrained(conf.bert_size)
    elif conf.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(conf.bert_size)
        model_config = RobertaConfig.from_pretrained(conf.bert_size)
    wandb.login()
    wandb.init(project='CBR', entity='hanseong_1201', name='sampling with GE')
    # wandb.config.update(OmegaConf.to_container(args))
    train()