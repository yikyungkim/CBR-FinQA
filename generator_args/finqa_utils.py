"""MathQA utils.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip

from config import parameters as conf


sys.path.insert(0, '../utils/')
from general_utils import table_row_to_text


def str_to_num(text):
    text = text.replace(",", "")
    try:
        num = int(text)
    except ValueError:
        try:
            num = float(text)
        except ValueError:
            if text and text[-1] == "%":
                num = text
            else:
                num = None
    return num



def arguments_to_indices(arguments, numbers, number_indices, const_list):
    argument_indices=[]
    for arg in arguments:
        if arg in const_list:
            argument_indices.append(const_list.index(arg))
        else:
            if arg in numbers:
                num_ind = numbers.index(arg)
            else:
                num_ind=-100  # test할때 gold arg가 numbers에 없는 경우들 있음. -100으로 하면 Index error
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num)==str_to_num(arg):
                        num_ind = num_idx
                        break
            num_ind_in_token = number_indices[num_ind]
            argument_indices.append(len(const_list)+num_ind_in_token)
    assert len(arguments)==len(argument_indices)
    return argument_indices


def argument_indices_to_label(argument_indices, max_seq_len, const_list_size):
    argument_labels=[]
    for ind in argument_indices:
        label = [0] * (const_list_size + max_seq_len)
        label[ind] = 1
        argument_labels.append(label)
    return argument_labels


def fill_masked_program(feature, pred_args):
    masked_program = feature.masked_program
    j=0
    for i in range(len(masked_program)):
        if (i%4==1) or (i%4==2):
            masked_program[i]=pred_args[j]
            j+=1
    return masked_program



class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "id original_question question_tokens options answer \
            numbers number_indices original_program program masked_program arguments"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 question,
                 input_ids,
                 input_mask,
                 option_mask,
                 segment_ids,
                 options,
                 answer=None,
                 program=None,
                 masked_program=None,
                 arguments=None,
                 argument_ids=None,
                 argument_mask=None,
                 mask_indices=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.question = question
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.option_mask = option_mask
        self.segment_ids = segment_ids
        self.options = options
        self.answer = answer
        self.program = program
        self.masked_program = masked_program
        self.arguments = arguments
        self.argument_ids = argument_ids
        self.argument_mask = argument_mask
        self.mask_indices = mask_indices


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    if conf.pretrained_model in ["bert", "finbert"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
    elif conf.pretrained_model in ["roberta", "longformer"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))

    return tokens


def _detokenize(tokens):
    text = " ".join(tokens)

    text = text.replace(" ##", "")
    text = text.replace("##", "")

    text = text.strip()
    text = " ".join(text.split())
    return text


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


""" Added for CBR arguments """
def mask_program(is_training, original_program, tokenizer):
    if is_training==True:
        program = program_tokenization(original_program)[:-1]
    else:
        program = original_program[:-1]

    gold_args=[]
    for i in range(len(program)):
        if (i%4==1) or (i%4==2):
            gold_args.append(program[i])
            program[i]=tokenizer.mask_token

    masked_program=""
    for i in range(len(program)):
        if (i%4==1) or (i%4==3):
            masked_program+=program[i]
            masked_program+=', '
        else:
            masked_program+=program[i]
    
    return masked_program, gold_args


def convert_single_mathqa_example(example, is_training, tokenizer, max_seq_length,
                                  const_list, cls_token, sep_token):
    """Converts a single MathQAExample into an InputFeature."""
    features = []

    """ input_ids, input_mask, segment_id for input tokens (original question + contexts + masked program) """
    """ option_mask for vocab output space (numbers + constants) """ 

    question_tokens = example.question_tokens
    if len(question_tokens) >  max_seq_length - 2:
        print("too long")
        question_tokens = question_tokens[:max_seq_length - 2]
    tokens = [cls_token] + question_tokens + [sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(tokens)

    # """ use different segment id for masked program (0 for question & context. 1 for masked_program)"""
    # sep_index =[]
    # for i, token in enumerate(input_ids):
    #     if token == tokenizer.sep_token_id:
    #         sep_index.append(i)
    
    # prog_idx_start = sep_index[1]+1
    # prog_idx_end = sep_index[2]+1
    # segment_ids = [0] * prog_idx_start + [1] * (prog_idx_end - prog_idx_start)

    for ind, offset in enumerate(example.number_indices):
        if offset < len(input_mask):
            input_mask[offset] = 2                        # input_mask: 숫자=2, 나머지 문장=1
        else:
            if is_training == True:
                print("number not in input")
                # print(example.original_question)
                # print(tokens)
                # print(example.numbers[ind])
                return features

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
      
    number_mask = [tmp - 1 for tmp in input_mask]       
    for ind in range(len(number_mask)):
        if number_mask[ind] < 0:
            number_mask[ind] = 0                        # number_mask: 숫자=1, 나머지 문장=0, padding=0
    
    option_mask = [1] * len(const_list)                 
    option_mask = option_mask + number_mask
    option_mask = [float(tmp) for tmp in option_mask]   # option_mask: constants & numbers=1, 나머지 문장=0 

    assert len(option_mask) == (len(const_list) + max_seq_length)

    for ind in range(len(input_mask)):                  # input_mask 원복 (문장=1,padding=0)
        if input_mask[ind] > 1:
            input_mask[ind] = 1


    """ argument_labels for target arguments """
    program = example.program
    masked_program = program_tokenization(example.masked_program)

    arguments = example.arguments
    numbers = example.numbers
    number_indices = example.number_indices

    """ If training, """
    # if arguments is not None:
    #     argument_id = arguments_to_indices(arguments, numbers, number_indices, const_list)
    # else:
    #     arguments = ""
    #     argument_id = []

    # arg=0
    # argument_ids=[]
    # argument_mask=[0] * max_seq_length     
    # mask_indices=[]
    # for i, token in enumerate(tokens):
    #     if token != tokenizer.mask_token:
    #         argument_ids.append(-100)
    #     elif token == tokenizer.mask_token:
    #         argument_ids.append(argument_id[arg])   # argument_ids: 각 토큰별 class label [512]
    #         argument_mask[i]=1                      # argument_mask: argument(mask) 위치=1, 나머지=-100 [542]
    #         mask_indices.append(i)                  # mask_indices: argument(mask)의 위치       
    #         arg+=1
    

    # if len(mask_indices)!=len(argument_id):
    #     if len(example.question_tokens) > max_seq_length-2:
    #         print("too long. can't get all mask token indices")
    #     else:
    #         print("example question: ", example.original_question)
    #         print("target indices: ", argument_id)
    #         print("mask_indices: ", mask_indices)

    # padding_arg = [-100] * (max_seq_length - len(argument_ids))
    # argument_ids.extend(padding_arg)

    # assert len(argument_ids) == max_seq_length
    # assert len(argument_mask) == max_seq_length

    """ If test, """
    argument_ids=[]
    argument_mask=[]
    mask_indices=[]
    for i, token in enumerate(tokens):
        if token == tokenizer.mask_token:
            mask_indices.append(i)                  # mask_indices: argument(mask)의 위치       


    features.append(
        InputFeatures(
            unique_id=-1,
            example_index=-1,
            tokens=tokens,
            question=example.original_question,
            input_ids=input_ids,
            input_mask=input_mask,
            option_mask=option_mask,
            segment_ids=segment_ids,
            options=example.options,
            answer=example.answer,
            program=program,
            masked_program=masked_program,
            arguments=arguments,
            argument_ids=argument_ids,
            argument_mask=argument_mask,
            mask_indices = mask_indices
            ))
    return features


def read_mathqa_entry(entry, tokenizer):
    
    question = entry["qa"]["question"]
    this_id = entry["id"]
    context = ""

    if conf.retrieve_mode == "single":
        for ind, each_sent in entry["qa"]["model_input"]:
            context += each_sent
            context += " "

    elif conf.retrieve_mode == "slide":
        if len(entry["qa"]["pos_windows"]) > 0:
            context = random.choice(entry["qa"]["pos_windows"])[0]
        else:
            context = entry["qa"]["neg_windows"][0][0]
            
    elif conf.retrieve_mode == "gold":
        for each_con in entry["qa"]["gold_inds"]:
            context += entry["qa"]["gold_inds"][each_con]
            context += " "

    elif conf.retrieve_mode == "none":
        # no retriever, use longformer
        table = entry["table"]
        table_text = ""
        for row in table[1:]:
            this_sent = table_row_to_text(table[0], row)
            table_text += this_sent
        context = " ".join(entry["pre_text"]) + " " + " ".join(entry["post_text"]) + " " + table_text

    context = context.strip()
    # process "." and "*" in text
    context = context.replace(". . . . . .", "")
    context = context.replace("* * * * * *", "")

    original_question = question + " " + tokenizer.sep_token + " " + context.strip()
    
    """ added for CBR arguments """
    if conf.program_mode == "seq":
        if 'program' in entry["qa"]:
            original_program = entry["qa"]['program']
            program = program_tokenization(original_program)
            masked_program, gold_args = mask_program(True, original_program, tokenizer)
        else:
            program = None
            original_program = None
    original_question += (" " + tokenizer.sep_token + " " + masked_program)


    if "exe_ans" in entry["qa"]:
        options = entry["qa"]["exe_ans"]
    else:
        options = None

    original_question_tokens = original_question.split(' ')

    numbers = []
    number_indices = []
    question_tokens = []
    for i, tok in enumerate(original_question_tokens):
        num = str_to_num(tok)
        if num is not None:
            numbers.append(tok)
            number_indices.append(len(question_tokens))
            if tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))
                number_indices.append(len(question_tokens) + 1)
        tok_proc = tokenize(tokenizer, tok)
        question_tokens.extend(tok_proc)

    if "exe_ans" in entry["qa"]:
        answer = entry["qa"]["exe_ans"]
    else:
        answer = None

    # table headers
    for row in entry["table"]:
        tok = row[0]
        if tok and tok in original_question:
            numbers.append(tok)
            tok_index = original_question.index(tok)
            prev_tokens = original_question[:tok_index]
            number_indices.append(len(tokenize(tokenizer, prev_tokens)) + 1)


    return MathQAExample(
        id=this_id,
        original_question=original_question,
        question_tokens=question_tokens,
        options=options,
        answer=answer,
        numbers=numbers,
        number_indices=number_indices,
        original_program=original_program,
        program=program,
        masked_program=masked_program,
        arguments=gold_args)


def read_mathqa_entry_test(entry, tokenizer):
    
    question = entry["qa"]["question"]
    this_id = entry["id"]
    context = ""

    for ind, each_sent in entry["qa"]["model_input"]:
        context += each_sent
        context += " "

    context = context.strip()
    # process "." and "*" in text
    context = context.replace(". . . . . .", "")
    context = context.replace("* * * * * *", "")

    original_question = question + " " + tokenizer.sep_token + " " + context.strip()
    
    """ added for CBR arguments """
    if conf.program_mode == "seq":
        if 'predicted' in entry["qa"]:
            original_program = entry["qa"]['program']
            program = program_tokenization(original_program)
            cbr_program = entry["qa"]['predicted']
            masked_program, gold_args = mask_program(False, cbr_program, tokenizer)     
        else:
            cbr_program = None
            original_program = None
            program = None
            masked_program, gold_args = None, None

    original_question += (" " + tokenizer.sep_token + " " + masked_program)

    # gold_args는 original program에서 가져와야함 (다시 정의)
    gold_args=[]
    for i in range(len(program[:-1])):
        if (i%4==1) or (i%4==2):
            gold_args.append(program[i])

    if "exe_ans" in entry["qa"]:
        options = entry["qa"]["exe_ans"]
        answer = entry["qa"]["exe_ans"]
    else:
        options = None
        answer = None

    original_question_tokens = original_question.split(' ')

    numbers = []
    number_indices = []
    question_tokens = []
    for i, tok in enumerate(original_question_tokens):
        num = str_to_num(tok)
        if num is not None:
            numbers.append(tok)
            number_indices.append(len(question_tokens))
            if tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))
                number_indices.append(len(question_tokens) + 1)
        tok_proc = tokenize(tokenizer, tok)
        question_tokens.extend(tok_proc)


    # table headers
    for row in entry["table"]:
        tok = row[0]
        if tok and tok in original_question:
            numbers.append(tok)
            tok_index = original_question.index(tok)
            prev_tokens = original_question[:tok_index]
            number_indices.append(len(tokenize(tokenizer, prev_tokens)) + 1)


    return MathQAExample(
        id=this_id,
        original_question=original_question,
        question_tokens=question_tokens,
        options=options,
        answer=answer,
        numbers=numbers,
        number_indices=number_indices,
        original_program=original_program,
        program=program,
        masked_program=masked_program,
        arguments=gold_args)