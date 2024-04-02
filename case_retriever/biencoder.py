import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score, precision_score, average_precision_score, ndcg_score

from tqdm import tqdm, trange
from datetime import datetime
import time
import os
import re
import random
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import pickle
import wandb

from config_bienc import parameters as conf



""" Uility functions """
def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')

def keep_recent_models(model_path, max_files):
    os.chdir(model_path)
    files = filter(os.path.isfile, os.listdir(model_path))
    files = [file for file in files]
    files.sort(key=lambda x: os.path.getmtime(x))
    if len(files) > max_files:
        del_list = files[0:(len(files)-max_files)]
        for del_file in del_list:
            os.remove(model_path + del_file)

def save_archive(archive_path, data, output_name):
    pickle.dump(data, open(archive_path+output_name, 'wb')) 

""" Read datasets and converting to features """
class QuestionExample(
    collections.namedtuple(
    "QuestionExample",
    "org_index, question, program, positives, negatives")):
    def convert_example(self, *args, **kwargs):
        return convert_qa_example(self, *args, **kwargs)


def tokenize(tokenizer, text):
    _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
    tokenize_fn = tokenizer.tokenize
    tokens=[]
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))
    return tokens

def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
#     program.append('EOF')
    return program

def levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return levenshtein(s2, s1, debug)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        if debug:
            print(current_row[1:])
        previous_row = current_row

    return previous_row[-1]

def operators_in_program(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    # print(program)

    operators = ""
    i=0
    while i < len(program):
        if i%4==0:
            operators += (program[i] + " ")
        i+=1
    operators = operators.replace("(", ",")
    operators = operators[:-2]
            
    return operators


def get_model_input(tokenizer, question, max_seq_len, cls_token, sep_token):
    question_token = tokenize(tokenizer, question)
    tokens = [cls_token]+question_token+[sep_token]
    seg_ids = [0]*len(tokens)
    
    if len(tokens) > max_seq_len:
        print('too long')
        tokens = tokens[:max_seq_len-1]
        tokens += [sep_token]
        seg_ids = seg_ids[:max_seq_len]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1]*len(input_ids)

    padding = [0]*(max_seq_len-len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    seg_ids.extend(padding)

    assert len(input_ids)==max_seq_len
    assert len(input_mask)==max_seq_len
    assert len(seg_ids)==max_seq_len  

    return input_ids, input_mask, seg_ids


def wrap_pair(tokenizer, question, candidate, label, max_seq_len, cls_token, sep_token):

    input_ids_q, input_mask_q, seg_ids_q = get_model_input(tokenizer, question, max_seq_len, cls_token, sep_token)
    input_ids_c, input_mask_c, seg_ids_c = get_model_input(tokenizer, candidate, max_seq_len, cls_token, sep_token)

    features = {
        'input_ids_q': input_ids_q,
        'input_mask_q': input_mask_q,
        'seg_ids_q': seg_ids_q,
        'input_ids_c': input_ids_c,
        'input_mask_c': input_mask_c,
        'seg_ids_c': seg_ids_c,
        'label': label
    }
    return features


def convert_qa_example(example, tokenizer, max_seq_len, cls_token, sep_token, input_concat, program_type):
    pos_features=[]
    neg_features=[]
    question=example.question
    index=example.org_index

    for positive in example.positives:
        gold_ques = positive['question']
        if program_type == "prog":
            gold_prog = positive['program'][:-5]
        elif program_type == "ops":
            gold_prog = operators_in_program(positive['program'])

        if input_concat == "qandp":
            gold_cand = gold_ques + " " + '[QNP]' + " " + gold_prog
        elif input_concat == "ponly":
            gold_cand = gold_prog
        elif input_concat == "qonly":
            gold_cand = gold_ques

        features = wrap_pair(tokenizer, question, gold_cand, 1, max_seq_len, cls_token, sep_token)
        features['query_index']=index
        features['cand_index']=positive['index']
        features['cand_question']=positive['question']
        features['cand_program']=positive['program']
        pos_features.append(features)
    
    for negative in example.negatives:
        neg_ques = negative['question']                                   
        if program_type == "prog":
            neg_prog = negative['program'][:-5]
        elif program_type == "ops":
            neg_prog = operators_in_program(negative['program'])

        if input_concat == "qandp":
            neg_cand = neg_ques + " " + '[QNP]' + " " + neg_prog
        elif input_concat == "ponly":
            neg_cand = neg_prog
        elif input_concat == "qonly":
            neg_cand = neg_ques
        
        features = wrap_pair(tokenizer, question, neg_cand, 0, max_seq_len, cls_token, sep_token)
        features['query_index']=index
        features['cand_index']=negative['index']
        features['cand_question']=negative['question']
        features['cand_program']=negative['program']
        neg_features.append(features)

    return pos_features, neg_features


def convert_to_features(examples, tokenizer):
    pos_features=[]
    neg_features=[]
    for (index, example) in enumerate(examples):
        pos, neg = example.convert_example(
            tokenizer=tokenizer,
            max_seq_len=conf.max_seq_len,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            input_concat=conf.input_concat,
            program_type=conf.program_type
        )
        pos_features.extend(pos)
        neg_features.extend(neg)
    return pos_features, neg_features



class ExampleDataset(Dataset):
    def __init__(self, example, tokenizer):
        self.data=[]
        self.label=[]
        self.cand_index=[]
        self.cand_question=[]
        self.cand_program=[]
        for positive in example.positives:
            cand_q = positive['question']
            cand_p = positive['program'][:-5]
            cand = cand_q + " " + '[QNP]' + " " + cand_p
            self.data.append(cand)
            self.label.append(1)
            self.cand_index.append(positive['index'])
            self.cand_question.append(positive['question'])
            self.cand_program.append(positive['program'])

        for negative in example.negatives:
            cand_q = negative['question']
            cand_p = negative['program'][:-5]
            cand = cand_q + " " + '[QNP]' + " " + cand_p
            self.data.append(cand)
            self.label.append(0)
            self.cand_index.append(negative['index'])
            self.cand_question.append(negative['question'])
            self.cand_program.append(negative['program'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.cand_index[idx], self.cand_question[idx], self.cand_program[idx]

def convert_to_features_all(examples, tokenizer):
    features=[]
    for i, example in enumerate(examples):
        index = example.org_index
        # encode given question
        question = example.question
        encoded_q = tokenizer.encode_plus(question, padding="max_length", truncation=True, max_length=conf.max_seq_len, return_tensors="pt")
        input_ids_q = encoded_q['input_ids'].tolist()[0]
        input_mask_q = encoded_q['attention_mask'].tolist()[0]
        seg_ids_q = [0]*len(input_ids_q)
        # input_ids_q, input_mask_q, seg_ids_q = get_model_input(tokenizer, question, conf.max_seq_len, tokenizer.cls_token, tokenizer.sep_token)

        # encode candidates by batch
        example_dataset = ExampleDataset(example, tokenizer)
        example_loader = DataLoader(example_dataset, batch_size=64, shuffle=False)
        for j, (data, label, cand_index, cand_question, cand_program) in enumerate(example_loader):
            encoded_c = tokenizer.batch_encode_plus(data, padding="max_length", truncation=True, max_length=conf.max_seq_len, return_tensors="pt")        
            input_ids_c = encoded_c['input_ids'].tolist()
            input_mask_c = encoded_c['attention_mask'].tolist()

            # create feature for each candidates
            for k in range(len(data)):
                features.append({
                    'input_ids_q': input_ids_q,
                    'input_mask_q': input_mask_q,
                    'seg_ids_q': seg_ids_q,
                    'input_ids_c': input_ids_c[k],
                    'input_mask_c': input_mask_c[k],
                    'seg_ids_c': [0]*len(input_ids_c[k]),
                    'label': label[k].item(),
                    'query_index': index,
                    'cand_index': cand_index[k].item(),
                    'cand_question': cand_question[k],
                    'cand_program': cand_program[k]
                })

        # # encode candidates one by one
        # for j, positive in enumerate(example.positives):
        #     index_c = positive['index']
        #     cand_q = positive['question']
        #     cand_p = positive['program'][:-5]
        #     cand = cand_q + " " + '[QNP]' + " " + cand_p
        #     input_ids_c, input_mask_c, seg_ids_c = get_model_input(tokenizer, cand, conf.max_seq_len, tokenizer.cls_token, tokenizer.sep_token)
        #     features.append({
        #         'input_ids_q': input_ids_q,
        #         'input_mask_q': input_mask_q,
        #         'seg_ids_q': seg_ids_q,
        #         'input_ids_c': input_ids_c,
        #         'input_mask_c': input_mask_c,
        #         'seg_ids_c': seg_ids_c,
        #         'label': 1,
        #         'query_index': index,
        #         'cand_index': index_c,
        #         'cand_question': cand_q,
        #         'cand_program': cand_p
        #     })

        # for j, negative in enumerate(example.negatives):
        #     index_c = negative['index']
        #     cand_q = negative['question']
        #     cand_p = negative['program'][:-5]
        #     cand = cand_q + " " + '[QNP]' + " " + cand_p
        #     input_ids_c, input_mask_c, seg_ids_c = get_model_input(tokenizer, cand, conf.max_seq_len, tokenizer.cls_token, tokenizer.sep_token)
        #     features.append({
        #         'input_ids_q': input_ids_q,
        #         'input_mask_q': input_mask_q,
        #         'seg_ids_q': seg_ids_q,
        #         'input_ids_c': input_ids_c,
        #         'input_mask_c': input_mask_c,
        #         'seg_ids_c': seg_ids_c,
        #         'label': 0,
        #         'query_index': index,
        #         'cand_index': index_c,
        #         'cand_question': cand_q,
        #         'cand_program': cand_p
        #     })
 
    
    return features




def get_negatives_by_score(gold_program, negative_candidates):
    
    gold_program = program_tokenization(gold_program)
    if conf.program_type == 'ops':
        i=0
        gold_ops=[]
        while i < len(gold_program):
            gold_ops.append(gold_program[i])
            i+=4

    elif conf.program_type == 'prog':
        i=0
        gold_ops=[]
        gold_args=[]
        while i < len(gold_program):
            if (i%4==1) or (i%4==2):
                gold_args.append(gold_program[i])
            if (i%4==0):
                if gold_program[i]!='EOF':
                    gold_ops.append(gold_program[i])
            i+=1
    
    
    """ calculate distance score for negative candidates """
    negative_candidates_scores=[]
    for negative in negative_candidates:
        neg_program = program_tokenization(negative['program'])
        i=0
        neg_ops=[]
        neg_args=[]
        if conf.program_type == 'ops':
            while i < len(neg_program):
                neg_ops.append(neg_program[i])
                i+=4
            ops_len = len(gold_ops)
            ops_distance = levenshtein(gold_ops, neg_ops)
            score = (ops_len - ops_distance)/ops_len

        elif conf.program_type == 'prog':
            while i < len(neg_program):
                if (i%4==1) or (i%4==2):
                    neg_args.append(neg_program[i])
                if (i%4==0):
                    if neg_program[i]!='EOF':
                        neg_ops.append(neg_program[i])
                i+=1
            ops_len = len(gold_ops)
            ops_distance = levenshtein(gold_ops, neg_ops)
            ops_score = (ops_len - ops_distance)/ops_len
            arg_len = len(gold_args)
            arg_distance = levenshtein(gold_args, neg_args)
            arg_score = (arg_len - arg_distance)/arg_len
            score = 0.8 * ops_score + 0.2 * arg_score
        negative_candidates_scores.append([negative, score])
        
    negative_candidates_scores.sort(key=lambda x:x[1])
    return negative_candidates_scores


def get_positives_and_negatives(data, mode, seed):
    random.seed(seed)
    gold_program = data['program']
    gold_index = [gold['index'] for gold in data['gold_index']]
    golds = data['gold_index']
    num_golds = len(golds)

    candidates = data['candidate_questions']
    negative_candidates = []
    for candidate in candidates:
        if candidate['index'] not in gold_index:
            negative_candidates.append(candidate)
    num_neg_cands = len(negative_candidates)

    k_pos = round(conf.num_cand / (conf.neg_ratio+1))
    if num_golds < k_pos:
        k_pos = num_golds
    k_neg = k_pos * conf.neg_ratio

    if mode == 'train':
        # if number of negatives are less than k_neg, use all negatives
        if len(negative_candidates) < k_neg:
            random.shuffle(golds)
            positives = golds[:k_pos]
            negatives = negative_candidates
            return positives, negatives

        if conf.sampling == 'random':           
            random.shuffle(golds)
            random.shuffle(negative_candidates)
            positives = golds[:k_pos]
            negatives = negative_candidates[:k_neg]

        elif conf.sampling == 'hard':
            # golds.sort(key=lambda x:x['question_score'])
            # negative_candidates.sort(key=lambda x:x['program_score']) 
            # num_hard = round(k_neg * conf.hard_ratio)
            # num_easy = k_neg - num_hard
            negative_candidates_scores = get_negatives_by_score(gold_program, negative_candidates)
            num_hard = round(k_neg * conf.hard_ratio)
            num_easy = k_neg - num_hard
            hard_index = len(negative_candidates_scores)-num_hard
            hard_negatives = [negative[0] for negative in negative_candidates_scores[hard_index:]]
            easy_negatives = [negative[0] for negative in negative_candidates_scores[:num_easy]]
            random.shuffle(golds)
            positives = golds[:k_pos]
            negatives = hard_negatives + easy_negatives     
        
        elif conf.sampling == 'adjusted_hard':
            negative_candidates_scores = get_negatives_by_score(gold_program, negative_candidates)
            num_hard = round(k_neg * conf.hard_ratio)
            num_easy = k_neg - num_hard
            fix_hard_index = len(negative_candidates_scores)-round(num_hard * conf.fix_ratio)
            fix_easy_index = round(num_easy * conf.fix_ratio)
            fix_hard_negatives = [negative[0] for negative in negative_candidates_scores[fix_hard_index:]]
            fix_easy_negatives = [negative[0] for negative in negative_candidates_scores[:fix_easy_index]]
            
            rand_neg_num = k_neg - len(fix_hard_negatives) - len(fix_easy_negatives)
            random_negatives = [negative[0] for negative in negative_candidates_scores[fix_easy_index:fix_hard_index]]
            random_negatives = random.sample(random_negatives, rand_neg_num)
            random.shuffle(golds)
            positives = golds[:k_pos]
            negatives = fix_hard_negatives + fix_easy_negatives + random_negatives
    else:
        positives = golds
        negatives = negative_candidates

    # # randomly get positive and negative candidates, satisfying the neg_ratio
    # if mode == "train":
    #     K_neg = conf.neg_ratio * conf.K_pos   # neg_ratio = num_of_neg / num_of_pos
    #     if num_golds >= conf.K_pos:
    #         if num_neg_cands >= K_neg:
    #             random.shuffle(golds)
    #             random.shuffle(negative_candidates)
    #             positives = golds[:conf.K_pos]
    #             negatives = negative_candidates[:K_neg]
    #         else:
    #             num_pos = int((1/conf.neg_ratio) * num_neg_cands)            
    #             num_neg = int(conf.neg_ratio * num_pos)                      
    #             random.shuffle(golds)
    #             random.shuffle(negative_candidates)
    #             positives = golds[:num_pos]
    #             negatives = negative_candidates[:num_neg]
    #     elif 0 < num_golds < conf.K_pos:
    #         num_neg = int(conf.neg_ratio * num_golds)    
    #         random.shuffle(negative_candidates)
    #         positives = golds
    #         negatives = negative_candidates[:num_neg]
    #     else:
    #         positives = []
    #         negatives = []
    # # test on all data
    # else:
    #     positives = golds
    #     negatives = negative_candidates

    return positives, negatives


def read_example(data, mode, seed):
    org_index=data['original_index']       
    question=data['question']
    program=data['program']
    candidates = get_positives_and_negatives(data, mode, seed)
    positives=candidates[0]
    negatives=candidates[1]
    return QuestionExample(
        org_index=org_index, 
        question=question, 
        program=program, 
        positives=positives,
        negatives=negatives)
    

# def read_examples(input_path, log_path, mode):
#     write_log(log_path, "Readings "+input_path)
#     with open(input_path) as file:
#         input_data = json.load(file)
#     examples=[]
#     for data in input_data:
#         examples.append(read_example(data, mode))
#     return input_data, examples


def read_examples(input_data, mode, seed):
    examples=[]
    for data in input_data:
        examples.append(read_example(data, mode, seed))
    return examples

def load_dataset(input_path, log_path):
    write_log(log_path, "Readings "+input_path)
    with open(input_path) as file:
        input_data = json.load(file)
    return input_data




""" Data Loader """
class myDataLoader:
    def __init__(self, is_training, data, batch_size):
        if conf.use_all_cands:
            self.data = data
        else:
            self.data_pos = data[0]
            self.data_neg = data[1]
            self.data = self.data_pos + self.data_neg              
        self.batch_size = batch_size
        self.is_training = is_training
        self.data_size = len(self.data)
        self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 \
            else int(self.data_size / batch_size) + 1
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        # drop last batch
        if self.is_training:
            bound = self.num_batches - 1
        else:
            bound = self.num_batches
        if self.count < bound:
            return self.get_batch()
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        if conf.use_all_cands:
            self.data = self.data
        else:
            self.data = self.data_pos + self.data_neg
        random.shuffle(self.data)
        return

    def get_batch(self):
        start_index = self.count * self.batch_size
        end_index = min((self.count + 1) * self.batch_size, self.data_size)

        self.count += 1
        # print (self.count)

        batch_data = {"input_ids_q": [],
                      "input_mask_q": [],
                      "seg_ids_q": [],
                      "input_ids_c": [],
                      "input_mask_c": [],
                      "seg_ids_c": [],
                      "label": [],
                      "query_index": [],
                      "cand_index": [],
                      "cand_question": [],
                      "cand_program": []
                      }
        for each_data in self.data[start_index: end_index]:
            batch_data["input_ids_q"].append(each_data["input_ids_q"])
            batch_data["input_mask_q"].append(each_data["input_mask_q"])
            batch_data["seg_ids_q"].append(each_data["seg_ids_q"])
            batch_data["input_ids_c"].append(each_data["input_ids_c"])
            batch_data["input_mask_c"].append(each_data["input_mask_c"])
            batch_data["seg_ids_c"].append(each_data["seg_ids_c"])
            batch_data["label"].append(each_data["label"])
            batch_data["query_index"].append(each_data["query_index"])
            batch_data["cand_index"].append(each_data["cand_index"])
            batch_data["cand_question"].append(each_data["cand_question"])
            batch_data["cand_program"].append(each_data["cand_program"])
        return batch_data



""" Model """
class Bert_model(nn.Module):
    def __init__(self, hidden_size, tokenizer):
        super(Bert_model, self).__init__()

        if conf.model == 'bert':
            self.bert = BertModel.from_pretrained(conf.bert_size, cache_dir=conf.cache_dir)
        elif conf.model == 'roberta':
            self.bert = RobertaModel.from_pretrained(conf.bert_size, cache_dir=conf.cache_dir)

        self.bert.resize_token_embeddings(len(tokenizer))                                       

        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, input_ids, input_mask, segment_ids):
        bert_outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=input_mask, 
            token_type_ids=segment_ids)

        embeddings = bert_outputs.last_hidden_state
        bert_output = embeddings[:, 0, :]

        output = self.linear(bert_output)
        # output = bert_output
        return output


# ## dual encoder
# class Biencoder_module(nn.Module):
#     def __init__(self, hidden_size, tokenizer):
#         super(Biencoder_module, self).__init__()
#         self.query_encoder = Bert_model(hidden_size, tokenizer)
#         self.cand_encoder = Bert_model(hidden_size, tokenizer)
    
#     def forward(self, input_ids_q, input_mask_q, segment_ids_q, input_ids_c, input_mask_c, segment_ids_c):
#         embedding_query = None
#         if input_ids_q is not None:
#             embedding_query = self.query_encoder(input_ids_q, input_mask_q, segment_ids_q)

#         embedding_cand = None
#         if input_ids_c is not None:
#             embedding_cand = self.cand_encoder(input_ids_c, input_mask_c, segment_ids_c)
        
#         return embedding_query, embedding_cand


# class Biencoder(nn.Module):
#     def __init__(self, hidden_size, tokenizer):
#         super(Biencoder, self).__init__()
#         self.model = Biencoder_module(hidden_size, tokenizer)
#         self.model = self.model.to(conf.device)
#         self.model = nn.DataParallel(self.model)
    
#     def compute_score(self, embed_q, embed_c):
#         embed_q = embed_q.unsqueeze(1)
#         embed_c = embed_c.unsqueeze(2)
#         score = torch.bmm(embed_q, embed_c)
#         score = torch.squeeze(score)
#         return score

#     def forward(self, input_ids_q, input_mask_q, segment_ids_q, input_ids_c, input_mask_c, segment_ids_c, label, pos_weight):
#         embed_q, _ = self.model(input_ids_q, input_mask_q, segment_ids_q, None, None, None)
#         _, embed_c = self.model(None, None, None, input_ids_c, input_mask_c, segment_ids_c)
#         score = self.compute_score(embed_q, embed_c)

#         loss_function = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
#         loss = loss_function(score, label)
#         return score, loss
        

# single encoder
class Biencoder(nn.Module):
    def __init__(self, hidden_size, tokenizer):
        super(Biencoder, self).__init__()
        self.model = Bert_model(hidden_size, tokenizer)
        self.model = self.model.to(conf.device)
        self.model = nn.DataParallel(self.model)
    
    def compute_score(self, embed_q, embed_c):
        embed_q = embed_q.unsqueeze(1)
        embed_c = embed_c.unsqueeze(2)
        score = torch.bmm(embed_q, embed_c)
        score = torch.squeeze(score)
        return score

    def forward(self, input_ids_q, input_mask_q, segment_ids_q, input_ids_c, input_mask_c, segment_ids_c, label, pos_weight):
        embed_q = self.model(input_ids_q, input_mask_q, segment_ids_q)
        embed_c = self.model(input_ids_c, input_mask_c, segment_ids_c)
        score = self.compute_score(embed_q, embed_c)

        loss_function = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)
        loss = loss_function(score, label)
        return score, loss


def get_positive_weight(label, mode):     # compute positive-negative ratio in batch
    num_positives = sum(label)
    num_negatives = len(label)-num_positives
    if num_positives == 0:          # when batch is composed of all negatives or all positives, apply constant positive weight to loss function (assumption)
        return conf.neg_ratio
        # if mode == 'train':
        #     return conf.neg_ratio
        # elif mode == 'valid':
        #     return valid_ratio
        # elif mode == 'test':
        #     return inf_ratio
    else:
        return num_negatives/num_positives
    

class EarlyStopping:
    def __init__(self, patience):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience
        
    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1
    
    def is_stop(self):
        return self.patience >= self.patience_limit
    

""" Model training """
def train(train_data, valid_data):
    mode = 'train'  
    model = Biencoder(model_config.hidden_size, tokenizer)
    model.train()

    # load train data
    train_examples = read_examples(train_data, 'train', 0)
    kwargs={'examples': train_examples, 'tokenizer': tokenizer}
    write_log(log_path, "Starts converting training data to features")
    if conf.use_all_cands:
        train_features = convert_to_features_all(**kwargs)
    else:
        train_features = convert_to_features(**kwargs)
    train_loader = myDataLoader(is_training=True, data=train_features, batch_size=conf.batch_size)
    total_steps = train_loader.num_batches * conf.epoch
    print("total number of batches: ", train_loader.num_batches)

    # load valid data
    valid_examples = read_examples(valid_data, 'valid', 0)
    kwargs={'examples': valid_examples, 'tokenizer': tokenizer}
    write_log(log_path, "Starts converting validation data to features")
    if conf.use_all_cands:
        valid_features = convert_to_features_all(**kwargs)
    else:
        valid_features = convert_to_features(**kwargs)

    # for optimization
    early_stop = EarlyStopping(patience=conf.patience)
    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    if not conf.resume:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = total_steps*conf.warm_up_prop, num_training_steps=total_steps)

    # for records
    record_step = 0
    record_loss = 0.0
    start_time = time.time()
    ep_global = 0   
    if conf.resume:
        checkpoint = torch.load(conf.resume_model)
        ep_global = checkpoint['epoch']+1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = total_steps*conf.warm_up_prop, num_training_steps=total_steps, last_epoch=checkpoint['epoch'])
    
    # starts training
    for ep in trange(ep_global, conf.epoch, desc="epoch"):
        write_log(log_path, "Epoch %d starts" % (ep))
        # sampling every epoch
        if (conf.resume==False and ep > 0) or (conf.resume==True and ep > ep_global):
            train_examples = read_examples(train_data, 'train', ep)
            kwargs={'examples': train_examples, 'tokenizer': tokenizer}
            write_log(log_path, "Starts converting training data to features")
            if conf.use_all_cands:
                train_features = convert_to_features_all(**kwargs)
            else:
                train_features = convert_to_features(**kwargs)
            train_loader = myDataLoader(is_training=True, data=train_features, batch_size=conf.batch_size)
            total_steps = train_loader.num_batches * conf.epoch
            print("total number of batches: ", train_loader.num_batches)

        train_loader.reset()
        for step, batch in enumerate(train_loader):
            input_ids_q = torch.tensor(batch['input_ids_q']).to(conf.device)
            input_mask_q = torch.tensor(batch['input_mask_q']).to(conf.device)
            seg_ids_q = torch.tensor(batch['seg_ids_q']).to(conf.device)
            input_ids_c = torch.tensor(batch['input_ids_c']).to(conf.device)
            input_mask_c = torch.tensor(batch['input_mask_c']).to(conf.device)
            seg_ids_c = torch.tensor(batch['seg_ids_c']).to(conf.device)
            label = torch.tensor(batch['label'], dtype=torch.float32).to(conf.device)
            pos_weight = torch.tensor([get_positive_weight(label, mode)], dtype=torch.float32).to(conf.device)

            model.zero_grad()
            optimizer.zero_grad()

            score, loss = model(input_ids_q, input_mask_q, seg_ids_q, input_ids_c, input_mask_c, seg_ids_c, label, pos_weight)            

            record_loss += loss.item()*100
            record_step+=1

            wandb.log({"loss/train_loss": (loss.item()*100), "params/batch": step})
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # record loss for every 100 batches
            if step > 1 and step % conf.report_loss == 0:
                write_log(log_path, "%d : loss = %.3f" % (step, record_loss / record_step))
                record_loss = 0.0
                record_step = 0

        # save model every epoch
        saved_model_path_dict = os.path.join(model_path, 'epoch_{}'.format(ep))
        torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, saved_model_path_dict)
        # keep_recent_models(model_path+'/', (conf.patience+1))

        # measure training time for every epoch
        cost_time = time.time() - start_time
        write_log(log_path, "----------------------Epoch %d time = %.3f" % (ep, cost_time))
        start_time = time.time()
        
        # record model evaluation and save prediciton file every epoch
        model.eval()
        results_path_cnt = os.path.join(results_path, 'prediction', str(ep))
        os.makedirs(results_path_cnt, exist_ok=True)
        write_log(log_path, "----------------------Epoch %d Model Evaluation" % (ep))
        val_loss = evaluate(valid_features, model, results_path_cnt, "valid")
        wandb.log({"loss/valid_loss": val_loss})

        # # check early stopping
        # early_stop.step(val_loss)
        # if early_stop.is_stop():
        #     write_log(log_path, "Early Stopped at epoch: %d" % (ep))
        #     break

        model.train()


""" Model Evaluation """
def evaluate(features, model, results_path_cnt, mode):

    results_path_cnt_mode = os.path.join(results_path_cnt, mode)
    os.makedirs(results_path_cnt_mode, exist_ok=True)

    data_iterator = myDataLoader(is_training=False, data=features, batch_size=conf.batch_size_test)

    total_loss = 0.0
    scores = []
    query_index = []
    cand_index = []
    cand_question = []
    cand_program = []

    with torch.no_grad():
        data_iterator.reset()   # shuffle
        for x in data_iterator:
            input_ids_q = torch.tensor(x['input_ids_q']).to(conf.device)
            input_mask_q = torch.tensor(x['input_mask_q']).to(conf.device)
            seg_ids_q = torch.tensor(x['seg_ids_q']).to(conf.device)
            input_ids_c = torch.tensor(x['input_ids_c']).to(conf.device)
            input_mask_c = torch.tensor(x['input_mask_c']).to(conf.device)
            seg_ids_c = torch.tensor(x['seg_ids_c']).to(conf.device)
            label = torch.tensor(x['label'], dtype=torch.float32).to(conf.device)
            pos_weight = torch.tensor([get_positive_weight(label, mode)], dtype=torch.float32).to(conf.device)

            q_index = x['query_index']
            c_index = x['cand_index']
            c_question = x['cand_question']
            c_program = x['cand_program']

            score, loss = model(input_ids_q, input_mask_q, seg_ids_q, input_ids_c, input_mask_c, seg_ids_c, label, pos_weight)            

            total_loss += loss.item()*100
            scores.extend(score.tolist())
            query_index.extend(q_index)
            cand_index.extend(c_index)
            cand_question.extend(c_question)
            cand_program.extend(c_program)

    output_prediction_file = os.path.join(results_path_cnt_mode, "predictions.json")
    if mode == "valid":
        metrics = retrieve_evaluate(scores, query_index, cand_index, cand_question, cand_program, output_prediction_file, conf.valid_file, topk=conf.topk)

    val_loss = total_loss/data_iterator.num_batches
    write_log(log_path, "validation loss = %.3f" % (val_loss))
    write_log(log_path, metrics)
    return val_loss


def compute_recall_precision(gold_pred_index, gold_index):
    tp_fn=len(gold_index)
    tp_fp=len(gold_pred_index)

    tp=0
    for index in gold_pred_index:
        if index in gold_index:
            tp+=1

    recall = tp / tp_fn
    precision = tp / tp_fp
    
    return recall, precision 


def retrieve_evaluate(scores, query_index, cand_index, cand_question, cand_program, output_prediction_file, original_file, topk):

    # save predicted results
    results = {}
    check_index = {}
    for score, q_index, c_index, c_question, c_program in zip(scores, query_index, cand_index, cand_question, cand_program):
        if q_index not in results:
            results[q_index] = []
            check_index[q_index] = []
        if c_index not in check_index[q_index]:
            results[q_index].append({
                "question": c_question,
                "program": c_program,
                "score": score,
                "index": c_index
            })
            check_index[q_index].append(c_index)
        
    with open(original_file) as file:
        all_data = json.load(file)

    len_data = 0
    recall_sum_k = 0.0
    recall_sum_3 = 0.0
    precision_sum_k = 0.0
    precision_sum_3 = 0.0
    avg_precision_sum = 0.0
    ndcg_sum = 0.0

    for data in all_data:
        index = data['original_index']

        # predicted results
        pred = results[index]
        pred_index = [pred_data['index'] for pred_data in pred]

        # true golds in original data
        gold_true_index = [gold['index'] for gold in data['gold_index']]

        # predicted golds in predicted results - topk
        sorted_pred = sorted(pred, key=lambda kv: kv['score'], reverse=True)
        gold_pred_k = sorted_pred[:topk]
        gold_pred_index_k = [data['index'] for data in gold_pred_k]

        # predicted golds in predicted results - top3
        gold_pred_3 = sorted_pred[:3]
        gold_pred_index_3 = [data['index'] for data in gold_pred_3]
    
        # add reranked results to output json file.
        data['reranked'] = gold_pred_k  


        """ Calculate evaluation metrics"""
        if len(gold_true_index)==0: # if no golds, don't compute metrics (if no_golds, error occurs in average precision)
            continue
        len_data+=1

        # create y_true, y_pred, y_score, y_true_score
        y_true=[]
        for index in pred_index:
            if index in gold_true_index:
                y_true.append(1)
            else:
                y_true.append(0)
        
        y_pred_score = [data['score'] for data in pred]  # predicted score
        
        # y_true_score=[] # program score
        # for index in pred_index:
        #     for candidate in data['candidate_questions']:
        #         if candidate['index']==index:
        #             y_true_score.append(candidate['program_score'])

        metrics_k = compute_recall_precision(gold_pred_index_k, gold_true_index)
        recall_sum_k += metrics_k[0]
        precision_sum_k += metrics_k[1]

        metrics_3 = compute_recall_precision(gold_pred_index_3, gold_true_index)
        recall_sum_3  += metrics_3[0]
        precision_sum_3 += metrics_3[1]

        avg_precision_sum += average_precision_score(y_true=y_true, y_score=y_pred_score, average=conf.average)
        # ndcg_sum += ndcg_score(y_true=[y_true_score], y_score=[y_pred_score])
    
    with open(output_prediction_file, "w") as file:
        json.dump(all_data, file, indent=4)

    recall_k = recall_sum_k / len_data
    recall_3 = recall_sum_3 / len_data
    precision_k = precision_sum_k / len_data
    precision_3 = precision_sum_3 / len_data
    avg_prec = avg_precision_sum / len_data
    # ndcg = ndcg_sum / len_data

    metrics = "Recall Top" + str(topk) + ": " + str(recall_k) + "\n" + "Recall Top 3: " + str(recall_3) + "\n" + "Precision Top" + str(topk) + ": " + str(precision_k)+ "\n" + "Precision Top 3: " + str(precision_3) + "\n" + "Mean Average Precision: " + str(avg_prec) + "\n" 
    # + "NDCG: " + str(ndcg) + "\n"

    if conf.mode == "train":
        wandb.log({
            "evaluate/recall_for_top3": recall_3,
            "evaluate/recall_for_top10": recall_k,
            "evaluate/precision_for_top3": precision_3,
            "evaluate/precision_for_top10": precision_k,
            "evaluate/average_precision": avg_prec
            # "evaluate/ndcg": ndcg
        })
    
    return metrics


""" Inference """
def test(test_data):

    mode = 'test'  
    model = Biencoder(model_config.hidden_size, tokenizer)
    
    checkpoint = torch.load(conf.saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    # load test data
    test_examples = read_examples(test_data, 'test', 0)
    kwargs={'examples': test_examples, 'tokenizer': tokenizer}
    write_log(log_path, "Starts converting test data to features")
    if conf.use_all_cands:
        test_features = convert_to_features_all(**kwargs)
    else:
        test_features = convert_to_features(**kwargs)
    data_iterator = myDataLoader(is_training=False, data=test_features, batch_size=conf.batch_size_test)
    
    scores = []
    query_index = []
    cand_index = []
    cand_question = []
    cand_program = []

    with torch.no_grad():
        data_iterator.reset()   # shuffle 
        for x in tqdm(data_iterator):
            input_ids_q = torch.tensor(x['input_ids_q']).to(conf.device)
            input_mask_q = torch.tensor(x['input_mask_q']).to(conf.device)
            seg_ids_q = torch.tensor(x['seg_ids_q']).to(conf.device)
            input_ids_c = torch.tensor(x['input_ids_c']).to(conf.device)
            input_mask_c = torch.tensor(x['input_mask_c']).to(conf.device)
            seg_ids_c = torch.tensor(x['seg_ids_c']).to(conf.device)
            label = torch.tensor(x['label'], dtype=torch.float32).to(conf.device)
            pos_weight = torch.tensor([get_positive_weight(label, mode)], dtype=torch.float32).to(conf.device)

            q_index = x['query_index']
            c_index = x['cand_index']
            c_question = x['cand_question']
            c_program = x['cand_program']

            score, loss = model(input_ids_q, input_mask_q, seg_ids_q, input_ids_c, input_mask_c, seg_ids_c, label, pos_weight)            

            scores.extend(score.tolist())
            query_index.extend(q_index)
            cand_index.extend(c_index)
            cand_question.extend(c_question)
            cand_program.extend(c_program)

    output_prediction_file = os.path.join(results_path, "predictions.json")
    if mode == 'test':
        metrics = retrieve_evaluate(scores, query_index, cand_index, cand_question, cand_program, output_prediction_file, conf.inference_file, topk=conf.topk)
    
    write_log(log_path, metrics)






if __name__ == '__main__':

    """Import tokenizer and model config"""
    if conf.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained(conf.bert_size)
        model_config = BertConfig.from_pretrained(conf.bert_size)
    elif conf.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(conf.bert_size)
        model_config = RobertaConfig.from_pretrained(conf.bert_size)

    special_token = {'additional_special_tokens': [ '[QNP]']}          # token between question and program
    num_added_toks = tokenizer.add_special_tokens(special_token)

    if conf.mode == 'train':
        """Set output path"""
        dir_model = os.path.join(conf.output_path, conf.dir_name)
        results_path = os.path.join(dir_model, "results")
        model_path = os.path.join(dir_model, "model")
        log_path = os.path.join(results_path, 'log.txt')
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

        write_log(log_path, "####################INPUT PARAMETERS###################")   
        for attr in conf.__dict__:
            value = conf.__dict__[attr]
            write_log(log_path, attr + " = " + str(value))
        write_log(log_path, "#######################################################")

        """Import dataset and convert info features"""
        train_data = load_dataset(conf.train_file, log_path)
        valid_data = load_dataset(conf.valid_file, log_path)

        # train_data, train_examples = read_examples(conf.train_file, log_path, 'train')
        # kwargs={'examples': train_examples, 'tokenizer': tokenizer}
        # write_log(log_path, "Starts converting training data to features")
        # train_features = convert_to_features(**kwargs)

        # valid_data, valid_examples = read_examples(conf.valid_file, log_path, 'valid')
        # kwargs['examples'] = valid_examples
        # write_log(log_path, "Starts converting validation data to features")
        # valid_features = convert_to_features(**kwargs)
  

        # if not conf.use_all_cands:
        #     valid_ratio = len(valid_features[1])/len(valid_features[0])     
        # else:
        #     num_pos=0
        #     for feature in valid_features:
        #         num_pos+=feature['label']
        #     num_neg = len(valid_features)-num_pos
        #     valid_ratio = num_neg/num_pos

        if not conf.resume:
            wandb.init(project="case_retriever", notes=conf.dir_name)
        elif conf.resume:
            wandb.init(project="case_retriever", notes=conf.dir_name, resume="must", id=conf.wandb_id)

        write_log(log_path, "Starts training...")
        train(train_data, valid_data)

    else:
        """Set path"""
        dir_model = os.path.join(conf.output_path, conf.dir_name)
        results_path = os.path.join(dir_model, "results")
        log_path = os.path.join(results_path, "log.txt")
        os.makedirs(results_path, exist_ok = True)

        write_log(log_path, "####################INPUT PARAMETERS###################")   
        for attr in conf.__dict__:
            value = conf.__dict__[attr]
            write_log(log_path, attr + " = " + str(value))
        write_log(log_path, "#######################################################")

        """Import dataset and convert info features"""
        test_data = load_dataset(conf.inference_file, log_path)
        # inf_data, inf_examples = read_examples(conf.inference_file, log_path, 'test')
        # kwargs={'examples': inf_examples, 'tokenizer': tokenizer}
        # write_log(log_path, "Starts converting data to features")
        # if not conf.use_all_cands:
        #     inf_features = convert_to_features(**kwargs)
        # else:
        #     inf_features = convert_to_features_all(**kwargs)

        # if not conf.use_all_cands:
        #     inf_ratio = len(inf_features[1])/len(inf_features[0])
        # else:
        #     num_pos=0
        #     for feature in inf_features:
        #         num_pos+=feature['label']
        #     num_neg = len(inf_features)-num_pos
        #     inf_ratio = num_neg/num_pos
      
        write_log(log_path, "Inference starts...")
        test(test_data)


   