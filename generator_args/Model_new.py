import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from config import parameters as conf

if conf.pretrained_model == "bert":
    from transformers import BertModel
elif conf.pretrained_model == "roberta":
    from transformers import RobertaModel
elif conf.pretrained_model == "finbert":
    from transformers import BertModel
elif conf.pretrained_model == "longformer":
    from transformers import LongformerModel


class bert_model(nn.Module):
    def __init__(self, hidden_size, const_size):
        super(bert_model, self).__init__()

        self.bert = RobertaModel.from_pretrained(conf.model_size, cache_dir=conf.cache_dir)

        self.seq_prj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.const_ind = nn.Parameter(torch.arange(0, const_size), requires_grad=False)
        self.const_embedding = nn.Embedding(const_size, hidden_size)

        self.option_embed_prj = nn.Linear(hidden_size, hidden_size, bias=True)
        

    def forward(self, input_ids, input_mask, segment_ids, option_mask):

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        bert_seq_output = bert_outputs.last_hidden_state               
        batch_size, seq_len, dim = list(bert_seq_output.size())

        seq_output = self.seq_prj(bert_seq_output)                # [B, seq_len, dim]

        const_embed = self.const_embedding(self.const_ind)        # [const_size, dim]
        const_embed = const_embed.repeat(batch_size, 1, 1)        # [B, const_size, dim]

        option_embed = torch.cat([const_embed, seq_output], dim=1)  # [B, (seq_len+const_size), dim]
        option_embed = self.option_embed_prj(option_embed)          # 추가

        option_logits = torch.matmul(seq_output, torch.transpose(option_embed, 1, 2))   # [B, seq_len, (seq_len+const_size)]
        option_mask = torch.unsqueeze(option_mask, dim=1)       # [B, 1, (seq_len+const_size)]
        logits = option_logits - 1e6 * (1 - option_mask)        # [B, seq_len, (seq_len+const_size)]
        # logits = torch.mul(option_logits, option_mask)      # [B, seq_len, (seq_len+const_size)]
        # print("logits shape: ", logits.size())

        return logits

