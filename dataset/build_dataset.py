import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer

import numpy as np
import json
import pickle
from tqdm import tqdm


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
    print('Starts storing question score')
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


def program_score(programs, constants, ops_weight):
    print('Starts computing program score')
    scores={}
    for i in tqdm(range(len(programs))):
        scores[i]=[]
        query = programs[i]
        for j in range(len(programs)):
            if i==j:
                continue
            cand = programs[j]
            score = distance_score(query, cand, constants, ops_weight)
            scores[i].append((j, score))
    return scores




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


def find_program_score(program_score, index):
    for score in program_score:
        if score[0]==index:
            return score[1]
        
def build_dataset_score(data, q_scores, p_scores, constants, threshold, use_all_cands):
    print('Starts building dataset')
    new_data=[]
    for i in tqdm(range(len(data))):
        contents={}
        contents['original_index']=i
        contents['question']=data[i]['qa']['question']
        contents['program']=masked_program(data[i]['qa']['program'], constants)
        contents['candidate_questions']=[]
        candidates=[]
        golds=[]
        for j in range(len(q_scores[i])):
            index = q_scores[i][j][0]
            candidate={}
            candidate['index']=index
            candidate['question']=data[index]['qa']['question']
            candidate['program']=masked_program(data[index]['qa']['program'], constants)
            candidate['question_score']=q_scores[i][j][1]
            if use_all_cands:
                p_score = p_scores[i][j][1]
            else:
                p_score = find_program_score(p_scores[i], index)
            candidate['program_score']=p_score
            candidates.append(candidate)
            if p_score >= threshold:
                gold={}
                gold['index']=index
                gold['question']=data[index]['qa']['question']
                gold['program']=masked_program(data[index]['qa']['program'], constants)
                gold['question_score']=q_scores[i][j][1]
                gold['program_score']=p_score
                golds.append(gold)    
        # when there is no golds; take top 1 program score >= 0.5
        # if len(golds)==0:
        #     silver=sorted(candidates, key=lambda x:x['program_score'], reverse=True)[:1][0]
        #     if silver['program_score'] > 0.5:
        #         gold={}
        #         gold['index']=silver['index']
        #         gold['question']=silver['question']
        #         gold['program']=silver['program']
        #         golds.append(gold)
        contents['candidate_questions']=candidates
        contents['gold_index']=golds
        new_data.append(contents)
    return new_data


def build_dataset_EM(data, q_scores, constants):
    print('Starts building dataset')
    new_data=[]
    for i in tqdm(range(len(data))):
        contents={}
        contents['original_index']=i
        contents['question']=data[i]['qa']['question']
        contents['program']=masked_program(data[i]['qa']['program'], constants)
        query_program = program_tokenization(data[i]['qa']['program'])
        query_ops = operator(query_program)
        candidates=[]
        golds=[]
        for j in range(len(q_scores[i])):
            index = q_scores[i][j][0]
            candidate={}
            candidate['index']=index
            candidate['question']=data[index]['qa']['question']
            candidate['program']=masked_program(data[index]['qa']['program'], constants)
            candidate['question_score']=q_scores[i][j][1]
            candidates.append(candidate)
            cand_program = program_tokenization(data[index]['qa']['program'])
            cand_ops = operator(cand_program)
            if query_ops == cand_ops:
                gold={}
                gold['index']=index
                gold['question']=data[index]['qa']['question']
                gold['program']=data[index]['qa']['program']
                gold['question_score']=q_scores[i][j][1]
                golds.append(gold)    
        contents['candidate_questions']=candidates
        contents['gold_index']=golds
        new_data.append(contents)
    return new_data



def main():
    "-----Which dataset to build?----"
    base_dataset = 'train'  # train / dev / test
    q_score_available = False
    p_score_available = False

    gold_standard = 'score' # score / EM (exact match)
    use_all_cands = False    # True: use all candidates, False: use top (num_cand) candidates
    num_cand = 200

    ops_weight = 0.8
    threshold = 0.9


    """Set path""" 
    original_dataset_path = '/shared/s3/lab07/yikyung/cbr/dataset/finqa_original/'
    constants_path='/shared/s3/lab07/yikyung/cbr/dataset/finqa_original/constant_list.txt'
    archive_path = '/shared/s3/lab07/yikyung/cbr/dataset/archives/'
    output_path = '/shared/s3/lab07/yikyung/cbr/dataset/case_retriever/'

    data = json.load(open(original_dataset_path+base_dataset+'.json'))    


    # get questions, programs, and constants
    questions = [data[i]['qa']['question'] for i in range(len(data))]
    programs = [data[i]['qa']['program'] for i in range(len(data))]
    constants=read_txt(constants_path)


    # get question embedding and compute similarity score
    if q_score_available:
        if not use_all_cands:
            q_scores = pickle.load(open(archive_path+base_dataset+str(num_cand)+'_scores_question', 'rb'))
        else:
            q_scores = pickle.load(open(archive_path+base_dataset+'_scores_question', 'rb'))
    else:
        embedding = get_embedding(questions)
        q_scores = question_score(questions, embedding)
        if not use_all_cands:
            q_scores = sort_questions(q_scores, num_cand)
            save_archive(archive_path, q_scores, base_dataset+ str(num_cand) +'_scores_question')
        else:
            save_archive(archive_path, q_scores, base_dataset+'_scores_question')


    # build dataset
    if gold_standard == 'score':
        # get programs scores
        if p_score_available:
            if not use_all_cands:
                p_scores = pickle.load(open(archive_path+base_dataset+str(num_cand)+'_scores_program', 'rb'))
            else:
                p_scores = pickle.load(open(archive_path+base_dataset+'_scores_program', 'rb'))                
        else:
            p_scores = program_score(programs, constants, ops_weight)
            if not use_all_cands:
                save_archive(archive_path, p_scores, base_dataset+str(num_cand)+'_scores_program')
            else:
                save_archive(archive_path, p_scores, base_dataset+'_scores_program')
        new_dataset = build_dataset_score(data, q_scores, p_scores, constants, threshold, use_all_cands)

    elif gold_standard == 'EM':
        new_dataset = build_dataset_EM(data, q_scores, constants)

    # save dataset
    if use_all_cands:
        output_dataset_path = output_path + base_dataset + '_' + gold_standard + '_all.json'
    else:
        output_dataset_path = output_path + base_dataset + '_' + gold_standard + '_' +str(num_cand) + '.json'
    
    with open(output_dataset_path, 'w') as file:
            json.dump(new_dataset, file, indent='\t')


if __name__ == '__main__':
    main()