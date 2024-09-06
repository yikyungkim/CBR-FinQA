# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
from tqdm import tqdm
import json
import collections
import os

from config import parameters as conf
from utils import *
from finqa_utils import *

import markdown

from typing import List, Optional
import fire
from llama import Dialog, Llama

from transformers import AutoTokenizer
import transformers
import torch

# os.environ["CUDA_VISIBLE_DEVICES"]="2"


def write_log(log_file, s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')

def write_predictions(output, output_file):
    with open(output_file, "w") as writer:
        writer.write(json.dumps(output, indent=4) + "\n")

def convert_to_markdown(table):
    # Generate the header line
    header = "| " + " | ".join(table[0]) + " |"
    # Generate the separator line
    separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
    # Initialize the markdown table with header and separator
    markdown_table = [header, separator]
    
    # Loop over the table skipping the first row (since it's the header)
    for row in table[1:]:
        # Generate each row line
        row_line = "| " + " | ".join(row) + " |"
        markdown_table.append(row_line)
    
    return "\n".join(markdown_table)

def markdown_to_html(markdown_text):
    # Convert markdown text to HTML
    html_output = markdown.markdown(markdown_text, extensions=['tables'])
    return html_output

def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)

    return " ".join(res)

def table_row_to_text(header, row):
    '''
    use templates to convert table row to text
    '''
    res = ""
    
    if header[0]:
        res += (header[0] + " ")

    for head, cell in zip(header[1:], row[1:]):
        res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")
    
    res = remove_space(res)
    return res.strip()

QAExample = collections.namedtuple(
        "QAExample",
        "id question program answer context_input case_input numbers prompt"
    )

# def read_examples(input_path, case_path, log_file, i, j):
def read_examples(input_path, case_path, log_file):

    write_log(log_file, "Reading " + input_path)
    with open(input_path) as input_file:
        input_data = json.load(input_file)

    write_log(log_file, "Reading " + case_path)
    with open(case_path) as case_file:
        case_data = json.load(case_file)

    input_data = input_data[:3]
    case_data = case_data[:3]

    # input_data = input_data[i:j]
    # case_data = case_data[i:j]

    examples = []
    for entry, case_entry in zip(input_data, case_data):
        examples.append(read_QA_example(entry, case_entry))
    return examples

def read_QA_example(entry, case_entry):
    example_id = entry['id']
    question = entry['qa']['question']
    program = entry['qa']['program']
    answer = entry['qa']['exe_ans']

    # full contexts
    table = entry['table']
    markdown_table = convert_to_markdown(table)
    html_table = markdown_to_html(markdown_table)

    pre_text=""
    for item in entry['pre_text']:
        pre_text+=item
        pre_text+="\n"

    post_text=""
    for item in entry['post_text']:
        post_text+=item
        post_text+="\n"
    
    table_text=""
    for i in range(len(table)):
        item = table_row_to_text(table[0], table[i])
        table_text+=item
        table_text+="\n"
    all_text=pre_text+post_text+table_text

    # argument pool    
    context_tokens = all_text.split(' ')
    numbers = []
    for tok in context_tokens:
        num = str_to_num(tok)
        if num is not None:
            numbers.append(tok)
            if tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))

    for row in entry['table']:
        tok = row[0]
        if tok and tok in all_text:
            numbers.append(tok)


    # retrieved cases
    if not conf.random_case:
        case_retrieved = case_entry['case_retrieved'][:conf.num_case]
        case_input=""
        for case in case_retrieved:
            case_question = case['question']
            case_program = case['program'][:-5]
            if conf.input_concat=="qandp":
                case = "Question: " + case_question + "\n" \
                    + "Program: " + case_program
            elif conf.input_concat=="ponly":
                case = "Program" + case_program 
            case_input+=case
            case_input+="\n"

    if conf.random_case:
        case_input=""
        case_question = "as a result of the sales of certain non-core towers and other assets what was the percent of the change in the recorded net losses from 2007 to 2008"
        case_program = "subtract(10.5, 7.1), divide(#0, 7.1)"
        if conf.input_concat=="qandp":
            case = "Question: " + case_question + "\n" \
                + "Program: " + case_program
        elif conf.input_concat=="ponly":
            case = "Program" + case_program 
        case_input+=case
        case_input+="\n"

        case_question = "what was the average initial health care trend rate for the three year period in%?"
        case_program = "table_average(initial health care trend rate, none)"
        if conf.input_concat=="qandp":
            case = "Question: " + case_question + "\n" \
                + "Program: " + case_program
        elif conf.input_concat=="ponly":
            case = "Program" + case_program 
        case_input+=case
        case_input+="\n"
    
        case_question = "hard assets were what percent of the brazilian purchase price , as finally determined?"
        case_program = "divide(83539, const_1000), divide(#0, 585.3)"
        if conf.input_concat=="qandp":
            case = "Question: " + case_question + "\n" \
                + "Program: " + case_program
        elif conf.input_concat=="ponly":
            case = "Program" + case_program 
        case_input+=case
        case_input+="\n"

    if conf.use_case:
        prompt = f'''
        """
        Similar questions and programs:

        {case_input}
        """

        """
        Contexts:


        {html_table}


        """

        """
        Operations:

        {op_list}
        """

        """
        Arguments:

        {numbers}
        {const_list}
        """

        Question: {question}

        '''

    return QAExample(
        id=example_id,
        question=question,
        program=program,
        answer=answer,
        context_input=all_text,
        case_input=case_input,
        numbers=numbers,
        prompt=prompt
    )



def generate_llama_answer_new(dialog):

    model = "meta-llama/Llama-2-13b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        dialog,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=4096,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")



def generate_program_new(examples, log_file):

    write_log(log_file, "Starts generating answer..")

    for example in tqdm(examples):
        dialog = example.prompt + "\n" + conf.guide_prompt
        generate_llama_answer_new(dialog)



    # generated_output=[]
    # for dialog, result in zip(dialogs, results):
    #     for msg in dialog:
    #         write_log(log_file, f"{msg['role'].capitalize()}: {msg['content']}\n")

    #     write_log(log_file,
    #         f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    #     )
    #     generated_output.append(result['generation']['content'])

    #     write_log(log_file, "\n==================================\n")


    # return generated_output



def generate_llama_answer(dialogs):
    # """
    # Examples to run with the models finetuned for chat. Prompts correspond of chat
    # turns between the user and assistant with the final one always being the user.

    # An optional system prompt at the beginning to control how the model should respond
    # is also supported.

    # The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    # `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    # """
    generator = Llama.build(
        ckpt_dir=conf.ckpt_dir,
        tokenizer_path=conf.tokenizer_path,
        max_seq_len=conf.max_seq_len,
        max_batch_size=conf.max_batch_size,
    )

    results = generator.chat_completion(
        dialogs,
        max_gen_len=conf.max_gen_len,
        temperature=conf.temperature,
        top_p=conf.top_p,
    )

    return results



def generate_program(examples, log_file):

    write_log(log_file, "Starts generating answer..")

    dialogs=[]
    for example in tqdm(examples):
        dialog = [
            {"role": "system", "content": conf.guide_prompt},
            {"role": "user", "content": example.prompt}
        ]
        dialogs.append(dialog)

    results = generate_llama_answer(dialogs)


    # dialogs=[]
    # results=[]
    # for example in tqdm(examples):
    #     test_dialog=[]
    #     dialog = [
    #         {"role": "system", "content": conf.guide_prompt},
    #         {"role": "user", "content": example.prompt}
    #     ]
    #     test_dialog.append(dialog)
    #     result = generate_llama_answer(test_dialog)
    #     dialogs.append(dialog)
    #     results.append(result)

    generated_output=[]
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            write_log(log_file, f"{msg['role'].capitalize()}: {msg['content']}\n")

        write_log(log_file,
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        generated_output.append(result['generation']['content'])

        write_log(log_file, "\n==================================\n")


    return generated_output


def generate_output_file(examples, generated_output):

    all_predictions = collections.OrderedDict()
    all_predictions["pred_programs"] = collections.OrderedDict()
    all_predictions["ref_programs"] = collections.OrderedDict()
    all_nbest = collections.OrderedDict()

    for index, example in enumerate(examples):
        nbest_json=[]
        output = collections.OrderedDict()
        output['id']=example.id
        output['ref_prog']=program_tokenization(example.program)
        output['ref_answer']=example.answer
        output['pred_prog']=program_tokenization(generated_output[index])
        nbest_json.append(output)

        all_predictions["pred_programs"][str(index)] = nbest_json[0]["pred_prog"]
        all_predictions["ref_programs"][str(index)] = nbest_json[0]["ref_prog"]
        all_nbest[str(index)] = nbest_json
    
    return all_predictions, all_nbest


if __name__ == "__main__":

    ## set file path
    dir_model = os.path.join(conf.output_path, conf.dir_name)
    results_path = os.path.join(dir_model, "results")
    log_path = os.path.join(dir_model, "log.txt")
    os.makedirs(results_path, exist_ok = True)

    # read operation and constant lists
    op_list = read_txt(conf.op_list_file, log_path)
    const_list = read_txt(conf.const_list_file, log_path)
    const_list = [const.lower().replace('.', '_') for const in const_list]

    write_log(log_path, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_path, attr + " = " + str(value))
    write_log(log_path, "#######################################################")

    examples = read_examples(conf.test_file, conf.test_case, log_path)
    # examples = read_examples(conf.test_file, conf.test_case, log_path, conf.i, conf.j)
    generate_program_new(examples, log_path)
    # generated_output = generate_program(examples, log_path)

    # generate output files
    # prediction_file = os.path.join(results_path, "predictions.json")
    # nbest_file = os.path.join(results_path, "nbest_predictions.json")
    # eval_file = os.path.join(results_path, "full_results.json")
    # error_file = os.path.join(results_path, "full_results_error.json")
    # all_predictions, all_nbest = generate_output_file(examples, generated_output)
    # write_predictions(all_predictions, prediction_file)
    # write_predictions(all_nbest, nbest_file)

    # exe_acc, prog_acc, op_acc = evaluate_result(nbest_file, conf.test_file, conf.test_case, eval_file, error_file, conf.program_mode)
    # prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc) + " operator acc: " + str(op_acc)
    # write_log(log_path, prog_res)
    