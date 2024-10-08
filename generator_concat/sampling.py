import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer

import json
import pickle
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
    bert_size = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_size)
    model = BertModel.from_pretrained(bert_size)

    # bert_size = "roberta-base"
    # tokenizer = RobertaTokenizer.from_pretrained(bert_size)
    # model = BertModel.from_pretrained(bert_size)

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
    for i in range(len(questions)):
        scores[i]=[]
        for j in range(len(questions)):
            if i==j:
                continue
            scores[i].append((j, cosine_scores[i][j].item()))
    return scores


def question_score_test(train_size, test_size, train_embedding, test_embedding):

    cosine_scores = cos_sim(test_embedding, train_embedding)
    scores={}
    for i in range(test_size):
        scores[i]=[]
        for j in range(train_size):
            scores[i].append((j, cosine_scores[i][j].item()))
    return scores




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
    for i in range(len(programs)):
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


def program_score_test(train_programs, test_programs, constants, ops_weight, threshold):
    scores={}
    golds={}
    for i in range(len(test_programs)):
        scores[i]=[]
        golds[i]=[]
        query = test_programs[i]
        for j in range(len(train_programs)):
            cand = train_programs[j]
            score = distance_score(query, cand, constants, ops_weight)
            scores[i].append((j, score))
            if score >= threshold:
                golds[i].append(j)
    return scores, golds


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




"------- Functions for sampling training set -----"

def load_dataset(finqa_dataset_path, constants_path, archive_path, 
                 mode, q_score_available, p_score_available, candidates_available, 
                 pos_pool, neg_pool):

    data = json.load(open(finqa_dataset_path))
    constants = read_txt(constants_path)

    # get question embedding and compute question similarity score
    if q_score_available:
        print('starts loading question score')
        q_scores = pickle.load(open(archive_path + mode + '_scores_question', 'rb'))
    else:
        print('starts getting question score')
        questions = [data[i]['qa']['question'] for i in range(len(data))]
        embedding = get_embedding(questions)
        q_scores = question_score(questions, embedding)
        # save_archive(archive_path, q_scores, mode + '_scores_question_random')

    # compute program score
    if p_score_available:
        print('starts loading program score and gold indices')
        p_scores = pickle.load(open(archive_path + mode + '_scores_program', 'rb'))                
        gold_indices = pickle.load(open(archive_path + mode + '_gold_indices', 'rb'))                
    else:
        print('starts getting program score and gold indices')
        ops_weight = 0.8
        threshold = 0.9
        programs = [data[i]['qa']['program'] for i in range(len(data))]
        p_scores, gold_indices = program_score(programs, constants, ops_weight, threshold)
        # save_archive(archive_path, p_scores, mode + '_scores_program_random')
        # save_archive(archive_path, gold_indices, mode + '_gold_indices_random')

    # sort by question score and get case pool
    if candidates_available:
        print('starts loading question similar candidates')
        gold_cands = pickle.load(open(archive_path + mode + '_' + str(pos_pool) + '_gold_candidates', 'rb'))
        non_gold_cands = pickle.load(open(archive_path + mode + '_' + str(neg_pool) + '_non_gold_candidates', 'rb'))
    else:
        print('starts getting question similar candidates')
        gold_cands, non_gold_cands = get_question_similar_candidates(data, q_scores, gold_indices, pos_pool, neg_pool)
        # save_archive(archive_path, gold_cands, mode + '_' + str(pos_pool) + '_gold_candidates_random')
        # save_archive(archive_path, non_gold_cands, mode + '_' + str(neg_pool) + '_non_gold_candidates_random')

    return data, q_scores, p_scores, gold_indices, constants, gold_cands, non_gold_cands


def get_question_similar_candidates(data, q_scores, gold_indices, pos_pool, neg_pool):

    gold_cands={}
    non_gold_cands={}
    for i in range(len(data)):
        gold_cands[i]=[]
        non_gold_cands[i]=[]
        q_score = q_scores[i]
        gold_index = gold_indices[i]
        if len(gold_index)==0:
            gold_candidates=[]
            non_gold_candidates_pair = sorted(q_score, key=lambda x:x[1], reverse=True)[:neg_pool]
            non_gold_candidates = [index for (index, score) in non_gold_candidates_pair]
            
        else:
            gold_pair=[]                
            non_gold_pair=[]
            for (c_index, score) in q_score:
                if c_index in gold_index:
                    gold_pair.append((c_index, score))
                else:
                    non_gold_pair.append((c_index, score))
    
            # sort by question score and get top-(pool) candidates
            gold_candidates_pair = sorted(gold_pair, key=lambda x:x[1], reverse=True)[:pos_pool]
            non_gold_candidates_pair = sorted(non_gold_pair, key=lambda x:x[1], reverse=True)[:neg_pool]

            gold_candidates = [index for (index, score) in gold_candidates_pair]
            non_gold_candidates = [index for (index, score) in non_gold_candidates_pair]
            
        gold_cands[i]=gold_candidates
        non_gold_cands[i]=non_gold_candidates
    
    return gold_cands, non_gold_cands


        
def get_random_cases(seed, input_data, p_scores, gold_cands, non_gold_cands, num_case, top3_precision):

    random.seed(seed)

    for index, data in enumerate(input_data):
        p_score = p_scores[index]
        gold_index = gold_cands[index]
        non_gold_index = non_gold_cands[index]

        # program score for non-golds
        non_gold_score=[]
        min_p_score = min([score for (c_index, score) in p_score])      
        for c_index in non_gold_index:
            if c_index > index:     
                s_index = c_index-1         # if i==j, the score was skipped when we created q_score and p_score
            else:
                s_index = c_index
            score = p_score[s_index][1]
            non_gold_score.append(score - min_p_score)                  # to make all scores greater than 0 (scaling)

        if len(gold_index)!=0:
            input_cases=[]
            for i in range(num_case):
                case={}
                gold_case = random.choices([0, 1], weights=[(1-top3_precision), top3_precision])
                if gold_case[0]==1:
                    case_index = random.sample(gold_index, k=1)
                else:
                    case_index = random.choices(non_gold_index, weights=non_gold_score, k=1)
                case['question']=input_data[case_index[0]]['qa']['question']
                case['program']=input_data[case_index[0]]['qa']['program']+", EOF"
                input_cases.append(case)

        else:   # if there is no gold, get non gold cases with weights
            input_cases=[]
            case_index = random.choices(non_gold_index, weights=non_gold_score, k=num_case)
            for i in range(num_case):
                case={}
                case['question']=input_data[case_index[i]]['qa']['question']
                case['program']=input_data[case_index[i]]['qa']['program']+", EOF"
                input_cases.append(case)        

        data['case_retrieved']=input_cases

    return input_data
    