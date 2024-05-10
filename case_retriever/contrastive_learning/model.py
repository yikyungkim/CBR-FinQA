import torch
from torch import nn
from transformers import BertModel, RobertaModel

""" Model """
class Bert_model(nn.Module):
    def __init__(self,model_type, model_size, cache_dir, hidden_size, tokenizer):
        super(Bert_model, self).__init__()

        if model_type == 'bert':
            self.bert = BertModel.from_pretrained(model_size, cache_dir=cache_dir)
        elif model_type == 'roberta':
            self.bert = RobertaModel.from_pretrained(model_size, cache_dir=cache_dir)

        self.bert.resize_token_embeddings(len(tokenizer))                                       

        self.linear = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, input_ids, attention_mask,pos_input_ids = None, pos_attention_mask= None,gold_inds = None):
        bert_outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask)
        origin_output = bert_outputs.last_hidden_state[:, 0, :]
        origin_output = self.linear(origin_output)
        if pos_input_ids is not None:
            pos_outputs = self.bert(
                input_ids=pos_input_ids, 
                attention_mask=pos_attention_mask)
            pos_output = pos_outputs.last_hidden_state[:, 0, :]
            pos_output = self.linear(pos_output)
            # output = bert_output
            return origin_output, pos_output
        return origin_output
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
        # origin (pos, neg)
        # label 0,1
        # logit label
        # origin sample
        # 
        loss = loss_function(score, label)
        return score, loss
