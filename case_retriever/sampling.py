import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer

import numpy as np
import json
import pickle
from tqdm import tqdm
import collections
import random


# Read a txt file into a list
def read_txt(input_path):
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        item = line.strip().lower()
        items.append(item)
    return items

def save_archive(archive_path, data, output_name):
    pickle.dump(data, open(archive_path+output_name, 'wb')) 


" ------- functions for questions --------"
def get_embedding(questions):
    # bert_size = "bert-base-uncased"
    # tokenizer = BertTokenizer.from_pretrained(bert_size)
    # model = BertModel.from_pretrained(bert_size)

    bert_size = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(bert_size)
    model = BertModel.from_pretrained(bert_size)

    model = nn.DataParallel(model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    encodings = tokenizer(questions, padding=True, return_tensors='pt')
    encodings = encodings.to(device)

    print('Starts getting question embedding')
    with torch.no_grad():
        embeds = model(**encodings)
    
    embedding = embeds.last_hidden_state[:,0,:]
    return embedding

def cos_sim(a,b):
    # Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    # return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))


def question_score(questions, embedding):

    cosine_scores = cos_sim(embedding, embedding)
    scores={}
    for i in tqdm(range(len(questions))):
        scores[i]=[]
        for j in range(len(questions)):
            if i==j:
                continue
            scores[i].append((j, cosine_scores[i][j].item()))
    return scores

def sort_questions(scores, num_cand):
    # if use_all_cands:
    #     print('Starts sort questions by similarity')
    #     sorted_scores={}
    #     for i in tqdm(range(len(scores))):
    #         sorted_q_score = sorted(scores[i], key=lambda x:x[1], reverse=True)
    #         sorted_scores[i] = sorted_q_score
    # else:
    print('Starts getting top '+str(num_cand)+' similar questions')
    sorted_scores={}
    for i in tqdm(range(len(scores))):
        top100 = sorted(scores[i], key=lambda x:x[1], reverse=True)[:num_cand]
        sorted_scores[i] = top100
    return sorted_scores


" ------- functions for programs --------"
# tokenize program
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
    program.append('EOF')
    return program

# compute levenshtein distance
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

# get operators in program
def operator(program):
    ops=[]
    i=0
    while i < len(program):
        if program[i]!='EOF':
            ops.append(program[i])
        i+=4
    return ops

# get arguments in program
def arguments(program, constants):
    args=[]
    i=0
    while i < len(program):
        if i%4==1:
            if program[i] not in constants:
                args.append('arg1')
            else:
                # args.append(program[i])
                if 'const' in program[i]:
                    args.append('const')
                else:
                    args.append(program[i])
        elif i%4==2:
            if program[i] not in constants:
                args.append('arg2')
            else:
                # args.append(program[i])
                if 'const' in program[i]:
                    args.append('const')
                else:
                    args.append(program[i])
        i+=1
    return args

# compute program score
def distance_score(query, cand, constants, weight):
    query_program = program_tokenization(query)
    cand_program = program_tokenization(cand)

    ops_q = operator(query_program)
    ops_c = operator(cand_program)
    ops_len = len(ops_q)
    ops_distance = levenshtein(ops_q, ops_c)
    ops_score = (ops_len - ops_distance)/ops_len

    arg_q = arguments(query_program, constants)
    arg_c = arguments(cand_program, constants)
    arg_len = len(arg_q)
    arg_distance = levenshtein(arg_q, arg_c)
    arg_score = (arg_len - arg_distance)/arg_len
    
    return weight * ops_score + (1-weight) * arg_score


def program_score(programs, constants, ops_weight, threshold):
    scores={}
    golds={}
    for i in tqdm(range(len(programs))):
        scores[i]=[]
        golds[i]=[]
        query = programs[i]
        for j in range(len(programs)):
            if i==j:
                continue
            cand = programs[j]
            score = distance_score(query, cand, constants, ops_weight)
            scores[i].append((j, score))
            if score >= threshold:
                golds[i].append(j)
    return scores, golds




"------- Functions to build dataset-----"
# convert original program to masked program (mask arguments)
def masked_program(program, constants):
    program = program_tokenization(program)
    args = arguments(program, constants)
    i=1
    j=0
    while i < len(program):
        if (i%4)==1 or (i%4)==2:
            program[i]=args[j]
            j+=1
        i+=1
    new_program = ""
    for i in range(len(program)):
        if (i%4)==1 or (i%4)==3:
            new_program+=program[i]
            new_program+=', '
        else:
            new_program+=program[i]
    return new_program




"""Added for random sampling"""

def load_dataset(finqa_dataset_path, constants_path, archive_path, 
                 mode, q_score_available, p_score_available):

    # get questions, programs, and constants
    data = json.load(open(finqa_dataset_path))
    questions = [data[i]['qa']['question'] for i in range(len(data))]
    programs = [data[i]['qa']['program'] for i in range(len(data))]
    constants = read_txt(constants_path)

    # get question embedding and compute similarity score
    print('starts getting question score')
    if q_score_available:
        q_scores = pickle.load(open(archive_path+mode+'_scores_question', 'rb'))
    else:
        embedding = get_embedding(questions)
        q_scores = question_score(questions, embedding)
        save_archive(archive_path, q_scores, mode+'_scores_question')

    # compute program score
    print('starts getting program score and gold indices')
    ops_weight = 0.8
    threshold = 0.9
    if p_score_available:
        p_scores = pickle.load(open(archive_path+mode+'_scores_program', 'rb'))                
        gold_indices = pickle.load(open(archive_path+mode+'_gold_indices', 'rb'))                
    else:
        p_scores, gold_indices = program_score(programs, constants, ops_weight, threshold)
        save_archive(archive_path, p_scores, mode+'_scores_program')
        save_archive(archive_path, gold_indices, mode+'_gold_indices')

    return data, q_scores, p_scores, gold_indices, constants


def get_random_samples(data, constants, seed, index, q_scores, p_scores, gold_index, k_pos, k_neg):

    if k_pos==0:
        return [], []

    non_gold_index = []
    for j in range(len(data)):
        if j!=index and j not in gold_index:
            non_gold_index.append(j)

    random.seed(seed)
    random.shuffle(gold_index)
    random.shuffle(non_gold_index)
    positive_index = gold_index[:k_pos]
    negative_index = non_gold_index[:k_neg]

    # print('seed: ', seed)
    # print('positive_index: ', positive_index)
    # print('negative_index: ', negative_index)

    positives=[]
    for c_index in positive_index:
        candidate={}
        candidate['index']=c_index
        candidate['question']=data[c_index]['qa']['question']
        candidate['program']=masked_program(data[c_index]['qa']['program'], constants)
        if c_index > index:
            s_index = c_index-1
        else:
            s_index = c_index
        candidate['question_score']=q_scores[index][s_index][1]
        candidate['program_score']=p_scores[index][s_index][1]
        positives.append(candidate)
    
    negatives=[]
    for c_index in negative_index:
        candidate={}
        candidate['index']=c_index
        candidate['question']=data[c_index]['qa']['question']
        candidate['program']=masked_program(data[c_index]['qa']['program'], constants)
        if c_index > index:
            s_index = c_index-1
        else:
            s_index = c_index
        candidate['question_score']=q_scores[index][s_index][1]
        candidate['program_score']=p_scores[index][s_index][1]
        negatives.append(candidate)
    
    # print('positives: ', positives)
    # print('negatives: ', negatives)
    
    return positives, negatives



class QuestionExample(
    collections.namedtuple(
    "QuestionExample",
    "org_index, question, program, positives, negatives")):
    def convert_example(self,):
        return 


def get_examples(finqa_dataset_path, constants_path, archive_path, 
                 mode, seed, q_score_available, p_score_available,
                 num_cand, neg_ratio):
    
    data, q_scores, p_scores, gold_indices, constants = load_dataset(finqa_dataset_path, constants_path, archive_path, mode, q_score_available, p_score_available)

    print('starts random sampling and get examples')
    examples=[]
    for i in tqdm(range(len(data))):
        org_index = i
        question = data[i]['qa']['question']
        program = masked_program(data[i]['qa']['program'], constants)
        gold_index = gold_indices[i]
        k_pos = round(num_cand / (neg_ratio+1))
        if len(gold_index) < k_pos:
            k_pos = len(gold_index)
        k_neg = k_pos * neg_ratio

        # print('org_index: ', org_index)
        # print('question: ', question)
        # print('program: ', program)
        # print('gold_index: ', gold_index)
        # print('k_pos: ', k_pos)
        # print('k_neg: ', k_neg)

        positives, negatives = get_random_samples(data, constants, seed, i, q_scores, p_scores, gold_index, k_pos, k_neg)

        example = QuestionExample(
            org_index = org_index,
            question = question,
            program = program,
            positives = positives,
            negatives = negatives)
        examples.append(example)

    return examples

        


# if __name__ == '__main__':
#     finqa_dataset_path = '/shared/s3/lab07/yikyung/cbr/dataset/finqa_original/train.json'
#     constants_path = '/shared/s3/lab07/yikyung/cbr/dataset/finqa_original/constant_list.txt'
#     archive_path = '/shared/s3/lab07/yikyung/cbr/dataset/archives/'
#     mode = 'train'
#     seed = 0
#     q_score_available = True
#     p_score_available = True
#     num_cand = 10
#     neg_ratio = 2
#     examples = get_examples(finqa_dataset_path, constants_path, archive_path, 
#                  mode, seed, q_score_available, p_score_available,
#                  num_cand, neg_ratio)
    
#     print(examples)