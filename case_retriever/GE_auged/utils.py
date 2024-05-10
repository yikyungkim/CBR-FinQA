import torch
from torch import nn
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import os
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
import random
from transformers import AutoTokenizer
from multiprocessing import Pool
from functools import partial
class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        # self.IP = torch.matmul
    def forward(self, x, y):
        # x,y = F.normalize(x), F.normalize(y)
        # return self.cos(x, y) / self.temp
        return (x@y.T).diagonal() # inner product

class Contrastive_Loss():
    def __init__(self,use_margin, temperature, margin=0.3):
        self.margin = margin
        self.sim = Similarity(temperature)
        self.loss_fn = nn.CrossEntropyLoss()
        self.use_margin = use_margin
        self.cross_batch = None
    def __call__(self, claim,pos,neg=None, do_normalize=True):
        calc_sim = self.sim(claim.unsqueeze(1),pos.unsqueeze(0))
        # calc_sim = self.sim(claim,pos)
        # batch,1,hidden
        # 1 batch hiddne
        labels = torch.arange(calc_sim.size(0)).long().to(calc_sim.device)
        # plus margin
        if self.use_margin:
            tn = torch.zeros(calc_sim.shape)
            tn.diagonal().fill_(self.margin)
            calc_sim -= tn.to(calc_sim.device)
        if neg is not None:
            neg_sim = self.sim(claim.unsqueeze(1),neg.unsqueeze(0))
            calc_sim = torch.cat([calc_sim,neg_sim],1) # B 2B
        loss = self.loss_fn(calc_sim, labels)
        return loss
    

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)               
    random.seed(seed)
    print('lock_all_seed')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def get_dev_sample(data,tokenizer):
    output = []
    for i in tqdm(range(len(data)), desc= 'processing'):
        origin = tokenizer(data[i]['question'],max_length=256, padding=True, truncation=True, return_tensors='pt')
        for key, val in origin.items():
            origin[key] = val.squeeze()
        output.append(origin)
        # question으로 정렬 먼저 진행
        gold = sorted(data[i]['gold_index'],key = lambda x : x['question_score'],reverse=True)
        output[i]['gold_inds'] = deepcopy([item['index'] for item in gold])
        output[i]['origin_index'] = i
    return output
def get_sample(data,tokenizer):
    output = []
    for i in tqdm(range(len(data)), desc= 'processing'):
        origin = tokenizer(data[i]['question'],max_length=256, padding=True, truncation=True, return_tensors='pt')
        for key, val in origin.items():
            origin[key] = val.squeeze()
        output.append(origin)
        # question으로 정렬 먼저 진행
        gold = data[i]['gold_index']
        output[i]['gold_inds'] = deepcopy(gold)
        output[i]['candidates'] = deepcopy(data[i]['candidates'])
        output[i]['origin_index'] = i
    return output
class myDataLoader:
    def __init__(self, is_training, data, batch_size):
        self.data = data
        self.visited = [False] * len(self.data)
        self.batch_size = batch_size
        self.is_training = is_training
        self.data_size = len(self.data)
        self.count = 0
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.train_size = 1000
        self.normal_term = 256
        if self.is_training:
            self.batch_sampled = self.preprocessing()
            self.num_batches = int(len(self.batch_sampled)//self.batch_size)
            random.shuffle(self.batch_sampled)
            if self.data_size % batch_size != 0:
                self.num_batches += 1
        else:
            self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 else int(self.data_size / batch_size) + 1


    def __iter__(self):
        return self

    def __next__(self):
        if self.is_training:
            if self.count < self.num_batches:
                output = self.batch_sampled[self.batch_size * self.count:self.batch_size * (self.count+1)]
                self.count += 1
                return self.dynamic_padding(output)
            else:
                raise StopIteration
        else:
            if self.count < self.num_batches:
                return self.eval_batch()
            else:
                raise StopIteration

    def __len__(self):
        return self.num_batches

    def reset(self):
        if self.is_training:
            self.batch_sampled = self.preprocessing()
            random.shuffle(self.batch_sampled)
        self.count = 0

    def eval_batch(self):
        output = {
            "input_ids" : [], 
            "attention_mask" : [], 
            'gold_inds': []
        }
        cnt = 0
        for i in range(self.count * self.batch_size , min((self.count + 1)* self.batch_size,self.data_size)):
            output['input_ids'].append(self.data[i]['input_ids'])
            output['attention_mask'].append(self.data[i]['attention_mask'])
            output['gold_inds'].append(self.data[i]['gold_inds'])
            self.visited[i] = True
        output['input_ids'] = pad_sequence(output['input_ids'],batch_first=True,padding_value = 1)
        output['attention_mask'] = pad_sequence(output['attention_mask'],batch_first=True,padding_value = 0)
        self.count += 1
        return output
    def preprocessing(self):
        for ind, item in enumerate(tqdm(self.data,desc='pre-sampling',total = len(self.data))):
            index = item['origin_index']
            k_pos = self.train_size//3
            if len(item['gold_inds']) < k_pos: # 존재하는 개수 보다 크면 안되니까
                k_pos = len(item['gold_inds'])

            gold_inds = [gold for gold in  item['gold_inds'] if 'index' in gold]
            gold_index = sorted(gold_inds,key = lambda x: x['question_score'],reverse=True)
            gold_inds = [int(gold['index']) for gold in  gold_index]

            neg_inds = [i for i in range(len(self.data)) if i not in gold_inds and i != index]
            neg_min = min([item['candidates'][str(i)]['program_score'] for i in neg_inds])
            negative_scores_scaled = [item['candidates'][str(i)]['program_score']- neg_min for i in neg_inds]

            positives = random.sample(item['gold_inds'], k_pos//2)
            positives += gold_index[:k_pos//2]
            
            hard_negatives = random.choices(neg_inds, weights=negative_scores_scaled, k=k_pos)
            neg_inds = [i for i in neg_inds if i not in hard_negatives]
            negatives = random.choices(neg_inds, k=k_pos)

            output = []
            for pos in positives:
                if 'original_index' in pos: #HJ
                    cand_token = self.tokenizer(pos['question'],max_length=256, padding=True, truncation=True, return_tensors='pt')
                    for key, val in cand_token.items():
                        cand_token[key] = val.squeeze()
                else:
                    cand_token = self.data[pos['index']]
                    # cand_token = self.data[index]
                tmp = {'input_ids':item['input_ids'],'attention_mask': item['attention_mask'], 
                    'cand_input_ids':cand_token['input_ids'],'cand_attention_mask': cand_token['attention_mask'],
                    'label': 1}
                output.append(deepcopy(tmp))
            for neg in negatives:
                cand_token = self.data[neg]
                tmp = {'input_ids':item['input_ids'],'attention_mask': item['attention_mask'], 
                    'cand_input_ids':cand_token['input_ids'],'cand_attention_mask': cand_token['attention_mask'],
                    'label': 0}
                output.append(deepcopy(tmp))
            for neg in hard_negatives:
                cand_token = self.data[neg]
                tmp = {'input_ids':item['input_ids'],'attention_mask': item['attention_mask'], 
                    'cand_input_ids':cand_token['input_ids'],'cand_attention_mask': cand_token['attention_mask'],
                    'label': 0}
                output.append(deepcopy(tmp))
        return output

    def dynamic_padding(self,batch):
        output = {
            "input_ids" : [], 
            "attention_mask" : [], 
            "cand_input_ids" : [], 
            "cand_attention_mask" : [], 
            'label': []
        }
        for item in batch:
            output['input_ids'].append(item['input_ids']) 
            output['attention_mask'].append(item['attention_mask']) 
            
            output['cand_input_ids'].append(item['cand_input_ids']) 
            output['cand_attention_mask'].append(item['cand_attention_mask']) 
            output['label'].append(item['label'])
        output['input_ids'] = pad_sequence(output['input_ids'],batch_first=True,padding_value = 1)
        output['attention_mask'] = pad_sequence(output['attention_mask'],batch_first=True,padding_value = 0)
        output['cand_input_ids'] = pad_sequence(output['cand_input_ids'],batch_first=True,padding_value = 1)
        output['cand_attention_mask'] = pad_sequence(output['cand_attention_mask'],batch_first=True,padding_value = 0)
        output['label'] = torch.LongTensor(output['label'])
        # batch siz의 index가 다 들어옴 -> 합쳐진 pos_gold가 주어진 batch에도 없어야 함.
        return output
    def divide_data(self,data, num_workers):
        chunk_size = len(data) // num_workers
        if isinstance(data, list):
            chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        else:
            chunks = [data.iloc[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        return chunks
    # 작업자 프로세스에서 실행할 함수
    def make_data(self,result):
        if isinstance(result[0], list):
            new_df = []
            for tmp in result:
                new_df.extend(tmp)
        elif isinstance(result[0], dict):
            new_df = {}
            for tmp in result:
                new_df.update(tmp)
            return new_df
    def multi_process(self,func,data,num_workers=16):
        p = Pool(num_workers)
        result = p.map(func,self.divide_data(data,num_workers))
        p.close()
        p.join()
        new_df = self.make_data(result)
        return new_df