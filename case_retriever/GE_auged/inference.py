from transformers import AutoModel,AutoTokenizer, AutoConfig,BertTokenizer,BertConfig,RobertaTokenizer,RobertaConfig
from train import evaluate
from model import Bert_model
import wandb
from config_bienc import parameters as conf
from utils import myDataLoader,get_sample, get_sample_with_program,WithProgramDataLoader
from glob import glob
import torch
from torch import nn
import json


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    if conf.model_type == 'bert':
        tokenizer = BertTokenizer,BertConfig.from_pretrained(conf.bert_size)
        model_config = BertConfig.from_pretrained(conf.bert_size)
    elif conf.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(conf.bert_size)
        model_config = RobertaConfig.from_pretrained(conf.bert_size)

    wandb.login()
    wandb.init(project='CBR_test', entity='hanseong_1201', name='param update q_only')
    #### LOAD TRAINED MDOEL ####
    def get_EPOCH(x):  return int(x.split('/')[-1].split('_')[1])
    ckpt_ls = sorted(glob(conf.output_path + '*.pt'),key=get_EPOCH,reverse=True)
    check_num = min(10,len(ckpt_ls))
    
    ######## DATA LOAD #########
    with open(f'{conf.test_file}','r') as f:
        data = json.load(f)
    
    test_example = get_sample_with_program(data,tokenizer)
    test_dataloader = WithProgramDataLoader(False,test_example,conf.batch_size)

    for ckpt in ckpt_ls[:check_num]:
        ep = get_EPOCH(ckpt)
        model = Bert_model(conf.model_type,conf.bert_size,conf.cache_dir,model_config.hidden_size,tokenizer)
        model.load_state_dict(torch.load(ckpt))
        model = nn.DataParallel(model)
        model.to(device)

        model,recall_topk, precision_topk,recall_top3, precision_top3 = evaluate(model,test_dataloader,conf.topk)
        wandb.log({f'val/recall@{conf.topk}' : recall_topk, f'val/precision@{conf.topk}' : precision_topk,'val/epoch' : ep,
                    f'val/recall@3': recall_top3 , f"val/precision@3" : precision_top3})
        
        print({f'val/recall@{conf.topk}' , recall_topk, f'val/precision@{conf.topk}' , precision_topk,'val/epoch' , ep,
            f'val/recall@3', recall_top3 , f"val/precision@3" , precision_top3})
        del model
    
    