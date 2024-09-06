import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer, BertConfig
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import recall_score, precision_score, average_precision_score, ndcg_score

from tqdm import tqdm
from datetime import datetime
import time
import os
import json
import numpy as np
import pickle
import wandb

from config_bienc import parameters as conf
import biencoder as biencoder
import sampling as sampling


def load_dataset_test(finqa_train, finqa_test, constants_path, archive_path, 
                      mode, q_score_available, p_score_available, candidates_available, num_test):

    train_data = json.load(open(finqa_train))
    test_data  = json.load(open(finqa_test))
    constants = sampling.read_txt(constants_path)
    train_size = len(train_data)
    test_size = len(test_data)

    # get question embedding and compute similarity score
    if q_score_available:
        print('starts loading question score')
        q_scores = pickle.load(open(archive_path + mode + '_scores_question', 'rb'))
    else:
        print('starts getting question score')
        train_questions = [data['qa']['question'] for data in train_data]
        test_questions = [data['qa']['question'] for data in test_data]
        train_embedding = sampling.get_embedding(train_questions)
        test_embedding = sampling.get_embedding(test_questions)
        q_scores = sampling.question_score_test(train_size, test_size, train_embedding, test_embedding)
        # sampling.save_archive(archive_path, q_scores, mode + '_scores_question')

    # compute program score
    if p_score_available:
        print('starts loading program score and gold indices')
        p_scores = pickle.load(open(archive_path + mode + '_scores_program', 'rb'))                
        gold_indices = pickle.load(open(archive_path + mode + '_gold_indices', 'rb'))                
    else:
        print('starts getting program score and gold indices')
        ops_weight = 0.8
        threshold = 0.9
        train_programs = [data['qa']['program'] for data in train_data]
        test_programs = [data['qa']['program'] for data in test_data]
        p_scores, gold_indices = sampling.program_score_test(train_programs, test_programs, constants, ops_weight, threshold)
        # sampling.save_archive(archive_path, p_scores, mode + '_scores_program')
        # sampling.save_archive(archive_path, gold_indices, mode + '_gold_indices')

    print('get question similar candidates')
    if candidates_available:
        candidates = pickle.load(open(archive_path + mode + '_' + str(num_test) + '_candidates', 'rb'))
    else:
        candidates={}
        for i in range(test_size):
            q_score = q_scores[i]
            candidates_pair = sorted(q_score, key=lambda x:x[1], reverse=True)[:num_test]            
            candidates[i]=[index for (index, score) in candidates_pair]
        # sampling.save_archive(archive_path, candidates, mode + '_' + str(num_test) + '_candidates')

    return train_data, test_data, q_scores, p_scores, gold_indices, constants, candidates


def get_examples_test(train_data, test_data, q_scores, p_scores, constants, candidates):

    examples=[]
    for i, data in enumerate(test_data):
        org_index = i
        question = data['qa']['question']
        program = sampling.masked_program(data['qa']['program'], constants)
        q_score = q_scores[i]
        p_score = p_scores[i]
        candidate = candidates[i]

        candidate_cases=[]
        for c_index, c_data in enumerate(train_data):
            if c_index in candidate:
                case={}
                case['index']=c_index
                case['question']=c_data['qa']['question']
                case['program']=sampling.masked_program(c_data['qa']['program'], constants)
                case['question_score']=q_score[c_index][1]
                case['program_score']=p_score[c_index][1]
                candidate_cases.append(case)

        example = sampling.QuestionExample(
            org_index = org_index,
            question = question,
            program = program,
            positives = candidate_cases,
            negatives = [])
        examples.append(example)

    return examples


def retrieve_evaluate_test(test_data, gold_indices, scores, query_index, cand_index, cand_question, cand_program, output_prediction_file, topk):

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
        
    len_data = 0
    recall_sum_k = 0.0
    recall_sum_3 = 0.0
    precision_sum_k = 0.0
    precision_sum_3 = 0.0
    avg_precision_sum = 0.0

    for i, data in enumerate(test_data):
        index = i

        # predicted results
        pred = results[index]
        pred_index = [pred_data['index'] for pred_data in pred]

        # true golds in original data
        gold_true_index = gold_indices[index]

        # predicted golds in predicted results - topk
        sorted_pred = sorted(pred, key=lambda kv: kv['score'], reverse=True)
        gold_pred_k = sorted_pred[:topk]
        gold_pred_index_k = [data['index'] for data in gold_pred_k]

        # predicted golds in predicted results - top3
        gold_pred_3 = sorted_pred[:3]
        gold_pred_index_3 = [data['index'] for data in gold_pred_3]
    
        # add reranked results to output json file.
        data['case_retrieved'] = gold_pred_k  


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

        metrics_k = biencoder.compute_recall_precision(gold_pred_index_k, gold_true_index)
        recall_sum_k += metrics_k[0]
        precision_sum_k += metrics_k[1]

        metrics_3 = biencoder.compute_recall_precision(gold_pred_index_3, gold_true_index)
        recall_sum_3  += metrics_3[0]
        precision_sum_3 += metrics_3[1]

        avg_precision_sum += average_precision_score(y_true=y_true, y_score=y_pred_score, average=conf.average)
        # ndcg_sum += ndcg_score(y_true=[y_true_score], y_score=[y_pred_score])
    
    with open(output_prediction_file, "w") as file:
        json.dump(test_data, file, indent=4)

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
def test():

    mode = 'test'  
    model = biencoder.Biencoder(model_config.hidden_size, tokenizer)
    
    checkpoint = torch.load(conf.saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # load data (use cases from test set)
    # write_log(log_path, "Readings "+ conf.inference_file)
    # inf_examples = biencoder.read_examples(biencoder.load_data(conf.inference_file), 'test', 0)
    # kwargs={'examples': inf_examples, 'tokenizer': tokenizer}
    # write_log(log_path, "Starts converting data to features")
    # inf_features = convert_to_features(**kwargs)

    # new load data (use cases from training set)
    biencoder.write_log(log_path, "Readings "+ conf.inference_file + " and " + conf.train_original)
    record_start = time.time()
    kwargs_load = {
        'finqa_train': conf.train_original,
        'finqa_test': conf.inference_file,
        'constants_path': conf.constant_file,
        'archive_path': conf.archive_path,
        'mode': mode,
        'q_score_available': conf.q_score_avail_test,
        'p_score_available': conf.p_score_avail_test,
        'candidates_available': conf.candidates_avail_test,
        'num_test': conf.num_test
    }
    train_data, test_data, q_scores, p_scores, gold_indices, constants, candidates = load_dataset_test(**kwargs_load)
    record_time = time.time() - record_start
    biencoder.write_log(log_path, "Time for loading data: %.3f" %record_time)

    biencoder.write_log(log_path, "Get examples...")
    record_start = time.time()
    inf_examples = get_examples_test(train_data, test_data, q_scores, p_scores, constants, candidates)
    record_time = time.time() - record_start
    biencoder.write_log(log_path, "Time for getting examples: %.3f" %record_time)
    
    if conf.test_feature_available:
        biencoder.write_log(log_path, "Loading inference features")
        inf_features = pickle.load(open(conf.archive_path+ mode + '_' + str(conf.num_test) + '_features', 'rb'))
    
    else:
        kwargs={'examples': inf_examples, 'tokenizer': tokenizer}
        biencoder.write_log(log_path, "Starts converting inference data to features")
        record_start = time.time()
        inf_features, neg_features = biencoder.convert_to_features(**kwargs)
        record_time = time.time() - record_start
        biencoder.write_log(log_path, "Time for converting inference data to input features: %.3f" %record_time)
        # sampling.save_archive(conf.archive_path, inf_features, mode+'_'+str(conf.num_test)+'_features')

    data_iterator = biencoder.myDataLoader(is_training=False, data=inf_features, batch_size=conf.batch_size_test)
    
    scores = []
    query_index = []
    cand_index = []
    cand_question = []
    cand_program = []

    biencoder.write_log(log_path, "Inference starts...")
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
            pos_weight = torch.tensor([biencoder.get_positive_weight(label, mode)], dtype=torch.float32).to(conf.device)

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
        # metrics = biencoder.retrieve_evaluate(scores, query_index, cand_index, cand_question, cand_program, output_prediction_file, conf.inference_file, topk=conf.topk)
        metrics = retrieve_evaluate_test(test_data, gold_indices, scores, query_index, cand_index, cand_question, cand_program, output_prediction_file, topk=conf.topk)
    biencoder.write_log(log_path, metrics)





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


    """Set path"""
    dir_model = os.path.join(conf.output_path, conf.dir_name)
    results_path = os.path.join(dir_model, "results")
    log_path = os.path.join(results_path, "log.txt")
    os.makedirs(results_path, exist_ok = True)

    biencoder.write_log(log_path, "####################INPUT PARAMETERS###################")   
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        biencoder.write_log(log_path, attr + " = " + str(value))
    biencoder.write_log(log_path, "#######################################################")

    test()


   