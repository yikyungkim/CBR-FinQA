from tqdm import tqdm
import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,TrainingArguments,AutoConfig
from datasets import Dataset
import torch
import logging
import os
from peft import LoraConfig, TaskType,get_peft_model,prepare_model_for_kbit_training, PeftModel
import pandas as pd
import math
import bitsandbytes as bnb
import transformers
from typing import Dict
from typing import List, Optional
from accelerate import Accelerator
import numpy as np
import random
import json
import time


def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        cache_dir = "/data2/yikyungkim/cache"
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        cache_dir = "/data2/yikyungkim/cache"
    )
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",cache_dir = "/data2/yikyungkim/cache")

    special_tokens_dict = dict()
    special_tokens_dict["unk_token"] = "<unk>"
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer

if __name__ == "__main__":

    case = False

    model, tokenizer = create_model_and_tokenizer()
    if case:
        output_dir = "/data2/yikyungkim/generator_llama/llama3_case/checkpoint-390"
    else:
        output_dir = "/data2/yikyungkim/generator_llama/llama3_noCase/checkpoint-360"
    model = PeftModel.from_pretrained(model, output_dir)

    model.eval()

    train = pd.read_csv("cbr_train.csv")
    dev = pd.read_csv("cbr_dev.csv")
    if case:
        test = pd.read_csv("cbr_test.csv")
    else:
        test = pd.read_csv("cbr_test_noCase.csv")

    def generate_description(model, text: str):
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        inputs_length = len(inputs["input_ids"][0])

        record_start = time.time()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=1024, early_stopping=True)
        record_time = time.time() - record_start
        print("Time for generating output: %.3f" %record_time)

        return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    if __name__ == '__main__':
        idx = 0
        # for df in [train, dev, test]:
        for df in [test]:
            pred_list = []
            gold_list = []
            for i in tqdm(range(len(df))):
                data = df.iloc[i]['text'].split('Output')
                input_data = data[0]
                gold = data[1].replace('\n','').replace(':','').replace('</s>','').strip()
                result = generate_description(model, input_data)
                if 'Output' in result:
                    result = result.split('Output')[1]
                if '   ' in result:
                    result = result.split('   ')[0]
                result = result.replace('\n','').replace(':','').replace('</s>','').strip()
                pred_list.append(result)
                gold_list.append(gold)
                print(f"Predict: {result}")
                print(f"Gold: {gold}")
                print('------------------------------')

            dic = {"pred":pred_list, "gold":gold_list}

            if case:
                with open("result_llama3_case.json", 'w') as file:
                    json.dump(dic, file, indent=4)
            else:
                with open("result_llama3_noCase_master.json", 'w') as file:
                    json.dump(dic, file, indent=4)


        