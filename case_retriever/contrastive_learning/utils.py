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
        return self.cos(x, y) / self.temp
        # return x@y.T # inner product

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



def get_sample(data,tokenizer):
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

def get_sample_with_program(data,tokenizer):
    output = []
    for i in tqdm(range(len(data)), desc= 'processing'):
        origin = tokenizer(data[i]['question'],max_length=256, padding=True, truncation=True, return_tensors='pt')
        candidates = tokenizer(data[i]['question'] + '</s>' + data[i]['program'],max_length=256, padding=True, truncation=True, return_tensors='pt')
        for key, val in origin.items():
            origin[key] = val.squeeze()
        for key, val in candidates.items():
            origin['cand_' + key] = val.squeeze()
        output.append(origin)
        # question으로 정렬 먼저 진행
        gold = sorted(data[i]['gold_index'],key = lambda x : x['question_score'],reverse=True)
        output[i]['gold_inds'] = deepcopy([item['index'] for item in gold])
        output[i]['origin_index'] = i
    return output

""" Data Loader """
class WithProgramDataLoader:
    def __init__(self, is_training, data, batch_size):
        self.data = data
        self.visited = [False] * len(self.data)
        self.batch_size = batch_size
        self.is_training = is_training
        self.data_size = len(self.data)
        self.count = 0
        self.normal_term = 256
        if self.is_training:
            self.num_batches = 0# int(self.data_size / batch_size) if self.data_size % batch_size == 0 else int(self.data_size / batch_size) + 1
            self.batch_sampled = self.preprocessing()
        else:
            self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 else int(self.data_size / batch_size) + 1


    def __iter__(self):
        return self

    def __next__(self):
        if self.is_training:
            self.count += 1
            if self.count < self.num_batches:
                current_batch = deepcopy(self.batch_sampled[self.count])
                random.shuffle(current_batch)
                output = self.gen_output(current_batch)
                self.count += 1
                return output
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
        self.count = 0
        self.shuffle_all_data()

    def shuffle_all_data(self):
        self.visited = [False] * self.data_size
        return 

    def eval_batch(self):
        output = {
            "input_ids" : [], 
            "attention_mask" : [], 
            "cand_input_ids" : [], 
            "cand_attention_mask" : [], 
            'gold_inds': []
        }
        for i in range(self.count * self.batch_size , min((self.count + 1)* self.batch_size,self.data_size)):
            output['input_ids'].append(self.data[i]['input_ids'])
            output['attention_mask'].append(self.data[i]['attention_mask'])
            output['cand_input_ids'].append(self.data[i]['cand_input_ids'])
            output['cand_attention_mask'].append(self.data[i]['cand_attention_mask'])
            output['gold_inds'].append(self.data[i]['gold_inds'])
            self.visited[i] = True
        output['input_ids'] = pad_sequence(output['input_ids'],batch_first=True,padding_value = 1)
        output['attention_mask'] = pad_sequence(output['attention_mask'],batch_first=True,padding_value = 0)

        output['cand_input_ids'] = pad_sequence(output['cand_input_ids'],batch_first=True,padding_value = 1)
        output['cand_attention_mask'] = pad_sequence(output['cand_attention_mask'],batch_first=True,padding_value = 0)
        self.count += 1
        return output
    
    def preprocessing(self):
        sampled_inds = []
        # for _ in tqdm(range(self.num_batches),desc = 'data sampling'):
        batch_size = 0
        batch = []
        with tqdm(total= self.data_size) as pbar:
            while not all(self.visited):
                cur_batch = self.get_batching()
                if len(cur_batch) < 3:
                    break
                if len(batch) +len(cur_batch) < self.batch_size:
                    batch += cur_batch
                elif len(cur_batch) > self.batch_size:
                    for i in range(0,len(cur_batch), self.batch_size):
                        sampled_inds.append(cur_batch[i: i + self.batch_size])
                        batch_size += 1
                    pbar.update(len(cur_batch))
                elif len(batch) +len(cur_batch) >= self.batch_size:
                    for i in range(0,len(batch), self.batch_size):
                        sampled_inds.append(batch[i: i + self.batch_size])
                        batch_size += 1
                    pbar.update(len(batch))
                    batch = []
                    batch += cur_batch
                else:
                    if batch:
                        for i in range(0,len(batch), self.batch_size):
                            sampled_inds.append(batch[i: i + self.batch_size])
                            batch_size += 1
                        pbar.update(len(batch))
        self.num_batches = batch_size
        return sampled_inds
    
    def get_batching(self,trial = 0):
        batch_gold = []
        # while True:
            # sample = random.choice([idx for idx,item in enumerate(self.visited) if item == False]) # random에서 데이터 하나 뽑고
            # sample = [idx for idx,item in enumerate(self.visited) if item == False]
        org_cands = [self.data[idx]['origin_index'] for idx in range(len(self.data)) if not self.visited[idx]]
        sample = sorted(org_cands,key = lambda x: -len(self.data[x]['gold_inds']))[0] # max 에서  추출중 현재

        # if not self.visited[sample]: # 무조건 visited False 상황임 근데 ㅋㅋ
        self.visited[sample] = True
        batch = [sample]
        pos_gold = self.data[sample]['gold_inds']
        size = 1
        candidates = [self.data[i]['origin_index'] for i in org_cands if i not in pos_gold] # negative candidates
        candidates = sorted(candidates,key = lambda x: len(self.data[x]['gold_inds'])) # max len should be back for pop

        
        batch_gold.extend(pos_gold)
        while size < self.normal_term:
            # 여기에 buffer를 추가해 주자.
            if len(candidates) > 4: # buffer
                cand = candidates.pop(0) # max: (), min: (0) # middle: len(candidates)//2)
            else:
                if trial < 2:
                    trial += 1
                    for i in batch:
                        self.visited[i] = False
                    self.get_batching(trial = trial)
                break

            if not self.visited[cand]: # 방문한적이 없고 -> 학습한 적이 없고
                cand_pos = self.data[cand]['gold_inds']  # candidates 의 pos를 구해오고
                if not bool(set(batch_gold + batch) & set(cand_pos + [cand])): # 겹치는게 없다면 #해 gold랑 gold끼리만 비교하고 있구나
                    batch.append(cand)
                    self.visited[cand] = True
                    batch_gold.extend(cand_pos)
                    size += 1
                else:
                    continue
                # 최종 batch에 append
        for item in batch:
            if item in batch_gold:
                raise Exception(f"positive gold in batch {item},\n gold batch : {batch_gold}")
        return batch

    def gen_output(self,batch):        
        output = {
            "input_ids" : [], 
            "attention_mask" : [], 
            "pos_input_ids" : [], 
            "pos_attention_mask" : [], 
            'gold_inds': [],
            'origin_index': []
        }

        for i in batch: # list of index
            output['input_ids'].append(self.data[i]['input_ids'])
            output['attention_mask'].append(self.data[i]['attention_mask'])
            output['gold_inds'].append(self.data[i]['gold_inds'])
            output['origin_index'].append(self.data[i]['origin_index'])
            if self.data[i]['gold_inds']:
                pos = self.data[i]['gold_inds'][0]
                self.data[i]['gold_inds'] = self.data[i]['gold_inds'][1:] + [self.data[i]['gold_inds'][0]]
            else:
                pos = i # origin index
            output['pos_input_ids'].append(self.data[pos]['cand_input_ids'])
            output['pos_attention_mask'].append(self.data[pos]['cand_attention_mask'])
            
        output['input_ids'] = pad_sequence(output['input_ids'],batch_first=True,padding_value = 1)
        output['attention_mask'] = pad_sequence(output['attention_mask'],batch_first=True,padding_value = 0)
        output['pos_input_ids'] = pad_sequence(output['pos_input_ids'],batch_first=True,padding_value = 1)
        output['pos_attention_mask'] = pad_sequence(output['pos_attention_mask'],batch_first=True,padding_value = 0)
        
        return output
""" Data Loader """
class myDataLoader:
    def __init__(self, is_training, data, batch_size):
        self.data = data
        self.visited = [False] * len(self.data)
        self.batch_size = batch_size
        self.is_training = is_training
        self.data_size = len(self.data)
        self.count = 0
        self.normal_term = 256
        if self.is_training:
            self.num_batches = 0# int(self.data_size / batch_size) if self.data_size % batch_size == 0 else int(self.data_size / batch_size) + 1
            self.batch_sampled = self.preprocessing()
        else:
            self.num_batches = int(self.data_size / batch_size) if self.data_size % batch_size == 0 else int(self.data_size / batch_size) + 1


    def __iter__(self):
        return self

    def __next__(self):
        if self.is_training:
            self.count += 1
            if self.count < self.num_batches:
                current_batch = deepcopy(self.batch_sampled[self.count])
                random.shuffle(current_batch)
                output = self.gen_output(current_batch)
                self.count += 1
                return output
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
        self.count = -1
        self.shuffle_all_data()

    def shuffle_all_data(self):
        self.visited = [False] * len(self.data)
        return

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
        sampled_inds = []
        # for _ in tqdm(range(self.num_batches),desc = 'data sampling'):
        batch_size = 0
        batch = []
        with tqdm(total= self.data_size) as pbar:
            while not all(self.visited):
                cur_batch = self.get_batching()
                if len(cur_batch) < 3:
                    break
                if len(batch) +len(cur_batch) < self.batch_size:
                    batch += cur_batch
                elif len(cur_batch) > self.batch_size:
                    for i in range(0,len(cur_batch), self.batch_size):
                        sampled_inds.append(cur_batch[i: i + self.batch_size])
                        batch_size += 1
                    pbar.update(len(cur_batch))
                elif len(batch) +len(cur_batch) >= self.batch_size:
                    for i in range(0,len(batch), self.batch_size):
                        sampled_inds.append(batch[i: i + self.batch_size])
                        batch_size += 1
                    pbar.update(len(batch))
                    batch = []
                    batch += cur_batch
                else:
                    if batch:
                        for i in range(0,len(batch), self.batch_size):
                            sampled_inds.append(batch[i: i + self.batch_size])
                            batch_size += 1
                        pbar.update(len(batch))
        self.num_batches = batch_size
        return sampled_inds
    
    def get_batching(self,trial = 0):
        batch_gold = []
        # while True:
            # sample = random.choice([idx for idx,item in enumerate(self.visited) if item == False]) # random에서 데이터 하나 뽑고
            # sample = [idx for idx,item in enumerate(self.visited) if item == False]
        org_cands = [self.data[idx]['origin_index'] for idx in range(len(self.data)) if not self.visited[idx]]
        sample = sorted(org_cands,key = lambda x: -len(self.data[x]['gold_inds']))[0] # max 에서  추출중 현재

        # if not self.visited[sample]: # 무조건 visited False 상황임 근데 ㅋㅋ
        self.visited[sample] = True
        batch = [sample]
        pos_gold = self.data[sample]['gold_inds']
        size = 1
        candidates = [self.data[i]['origin_index'] for i in org_cands if i not in pos_gold] # negative candidates
        candidates = sorted(candidates,key = lambda x: len(self.data[x]['gold_inds'])) # max len should be back for pop

        
        batch_gold.extend(pos_gold)
        while size < self.normal_term:
            # 여기에 buffer를 추가해 주자.
            if len(candidates) > 4: # buffer
                cand = candidates.pop(0) # max: (), min: (0) # middle: len(candidates)//2)
            else:
                if trial < 2:
                    trial += 1
                    for i in batch:
                        self.visited[i] = False
                    self.get_batching(trial = trial)
                break

            if not self.visited[cand]: # 방문한적이 없고 -> 학습한 적이 없고
                cand_pos = self.data[cand]['gold_inds']  # candidates 의 pos를 구해오고
                if not bool(set(batch_gold + batch) & set(cand_pos + [cand])): # 겹치는게 없다면 #해 gold랑 gold끼리만 비교하고 있구나
                    batch.append(cand)
                    self.visited[cand] = True
                    batch_gold.extend(cand_pos)
                    size += 1
                else:
                    continue
                # 최종 batch에 append
        for item in batch:
            if item in batch_gold:
                raise Exception(f"positive gold in batch {item},\n gold batch : {batch_gold}")
        return batch
        # batch siz의 index가 다 들어옴 -> 합쳐진 pos_gold가 주어진 batch에도 없어야 함.
    def gen_output(self,batch):
        output = {
            "input_ids" : [], 
            "attention_mask" : [], 
            "pos_input_ids" : [], 
            "pos_attention_mask" : [], 
            'gold_inds': []
        }
        for i in batch: # list of index
            output['input_ids'].append(self.data[i]['input_ids'])
            output['attention_mask'].append(self.data[i]['attention_mask'])
            output['gold_inds'].append(self.data[i]['gold_inds'])
            if self.data[i]['gold_inds']:
                pos = self.data[i]['gold_inds'][0]
                self.data[i]['gold_inds'] = self.data[i]['gold_inds'][1:] + [self.data[i]['gold_inds'][0]]
            else:
                pos = i # origin index
            output['pos_input_ids'].append(self.data[pos]['input_ids'])
            output['pos_attention_mask'].append(self.data[pos]['attention_mask'])
        output['input_ids'] = pad_sequence(output['input_ids'],batch_first=True,padding_value = 1)
        output['attention_mask'] = pad_sequence(output['attention_mask'],batch_first=True,padding_value = 0)
        output['pos_input_ids'] = pad_sequence(output['pos_input_ids'],batch_first=True,padding_value = 1)
        output['pos_attention_mask'] = pad_sequence(output['pos_attention_mask'],batch_first=True,padding_value = 0)
        return output