#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script
"""
from tqdm import tqdm
import json
import os
from datetime import datetime
import time
import logging
from utils import *
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim
from transformers.optimization import get_cosine_schedule_with_warmup

from Model_new import bert_model

import wandb
""" added for CBR """
# from convert import *

if conf.pretrained_model == "bert":
    print("Using bert")
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "roberta":
    print("Using roberta")
    from transformers import RobertaTokenizer
    from transformers import RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "finbert":
    print("Using finbert")
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "longformer":
    print("Using longformer")
    from transformers import LongformerTokenizer, LongformerConfig
    tokenizer = LongformerTokenizer.from_pretrained(conf.model_size)
    model_config = LongformerConfig.from_pretrained(conf.model_size)


# create output paths
if conf.mode == "train":
    # model_dir_name = conf.model_save_name + "_" + \
    #     datetime.now().strftime("%Y%m%d%H%M%S")         # for restart
    model_dir_name = conf.model_save_name
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=True)       # for restart 
    os.makedirs(results_path, exist_ok=True)
    log_file = os.path.join(results_path, 'log.txt')

else:
    saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(
        conf.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, "results")
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

op_list = read_txt(conf.op_list_file, log_file)
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)


train_data, train_examples, op_list, const_list = \
    read_examples(input_path=conf.train_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)

valid_data, valid_examples, op_list, const_list = \
    read_examples(input_path=conf.valid_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file)


kwargs = {"examples": train_examples,
          "tokenizer": tokenizer,
          "max_seq_length": conf.max_seq_length,
          "is_training": True,
          "const_list": const_list}

train_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = valid_examples
kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)


def train():
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    const_size = len(const_list)
    model = bert_model(model_config.hidden_size, const_size)
    model = nn.DataParallel(model)
    model.to(conf.device)
    model.train()

    train_iterator = DataLoader(
        is_training=True, data=train_features, batch_size=conf.batch_size, reserved_token_size=reserved_token_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    total_steps = train_iterator.num_batches * conf.epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = total_steps*conf.warm_up_prop, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(reduction='none')

    record_loss_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0
    record_step = 0

    ep_global = 0   
    if conf.resume:
        checkpoint = torch.load(conf.resume_model)
        model.load_state_dict(checkpoint)
        ep_global = 0

    for ep in range(ep_global, conf.epoch):                     
        train_iterator.reset()
        write_log(log_file, "Epoch %d starts" % (ep))
        for step, x in enumerate(train_iterator):

            input_ids = torch.tensor(x['input_ids']).to(conf.device)
            input_mask = torch.tensor(x['input_mask']).to(conf.device)
            segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
            option_mask = torch.tensor(x['option_mask']).to(conf.device)

            argument_ids = torch.tensor(x['argument_ids']).to(conf.device)
            argument_mask = torch.tensor(x['argument_mask']).to(conf.device)

            model.zero_grad()
            optimizer.zero_grad()

            this_logits = model(input_ids, input_mask, segment_ids, option_mask)

            loss = criterion(this_logits.view(-1, this_logits.shape[-1]), argument_ids.view(-1))
            # loss = loss * argument_mask.view(-1)
            # loss = loss.sum() / argument_mask.sum()

            record_loss += loss.item()*100
            record_step += 1

            wandb.log({"loss/train_loss": loss.item()*100, "params/batch": step})

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step > 1 and step % conf.report_loss == 0:
                write_log(log_file, "%d : loss = %.3f" %
                          (step, record_loss / record_step))
                record_loss = 0.0
                record_step = 0

        # measure training time for every epoch
        cost_time = time.time() - start_time
        write_log(log_file, "----------------------Epoch %d time = %.3f" % (ep, cost_time))
        start_time = time.time()

        model.eval()
        results_path_cnt = os.path.join(results_path, 'loads', str(ep))
        os.makedirs(results_path_cnt, exist_ok=True)
        write_log(log_file, "----------------------Epoch %d Model Evaluation" % (ep))
        evaluate(valid_examples, valid_features, model, results_path_cnt, 'valid')
        
        saved_model_path_cnt = os.path.join(saved_model_path, 'model_{}.pt'.format(ep))
        # os.makedirs(saved_model_path_cnt, exist_ok=True)
        torch.save(model.state_dict(), saved_model_path_cnt)
        # keep_recent_models(saved_model_path+'/', 3) 

        model.train()


def evaluate(data_ori, data, model, ksave_dir, mode='valid'):

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, reserved_token_size=reserved_token_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.CrossEntropyLoss(reduction='none')

    k = 0
    all_results = []
    with torch.no_grad():
        for step, x in enumerate(tqdm(data_iterator)):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            option_mask = x['option_mask']

            argument_ids = x['argument_ids']
            argument_mask = x['argument_mask']

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids, option_mask, argument_ids, argument_mask]:
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)
            option_mask = torch.tensor(option_mask).to(conf.device)

            argument_ids = torch.tensor(argument_ids).to(conf.device)
            argument_mask = torch.tensor(argument_mask).to(conf.device)

            this_logits = model(input_ids, input_mask, segment_ids, option_mask)

            loss = criterion(this_logits.view(-1, this_logits.shape[-1]), argument_ids.view(-1))
            # loss = loss * argument_mask.view(-1)
            # loss = loss.sum() / argument_mask.sum()

            wandb.log({"loss/valid_loss": loss.item()*100, "params/batch": step})

            for this_logit, this_id in zip(this_logits.tolist(), x["unique_id"]):
                all_results.append(
                    RawResult(
                        unique_id=int(this_id),
                        logits=this_logit,
                        loss=None
                    ))

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions.json")
    output_nbest_file = os.path.join(ksave_dir_mode,
                                     "nbest_predictions.json")
    output_eval_file = os.path.join(ksave_dir_mode, "full_results.json")
    output_error_file = os.path.join(ksave_dir_mode, "full_results_error.json")

    all_predictions, all_nbest = compute_predictions(
        data_ori,
        data,
        all_results,
        n_best_size=conf.n_best_size,
        const_list=const_list)
    write_predictions(all_predictions, output_prediction_file)
    write_predictions(all_nbest, output_nbest_file)

    if mode == "valid":
        original_file = conf.valid_file
    else:
        original_file = conf.test_file

    exe_acc, prog_acc, op_acc = evaluate_result(
        output_nbest_file, original_file, output_eval_file, output_error_file, program_mode=conf.program_mode)

    prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc) + " operator acc: " + str(op_acc)
    write_log(log_file, prog_res)

    wandb.log({
        "evaluate/exec_acc": exe_acc,
        "evaluate/prog_acc": prog_acc,
        "evaluate/op_acc": op_acc})

    return



if __name__ == '__main__':
    wandb.init(project="argument_generator")
#                ,resume="must", id="2ow9tdts"
# )   #resume
 

    if conf.mode == "train":
        train()