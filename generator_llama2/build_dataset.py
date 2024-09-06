from tqdm import tqdm
import json
import csv
import pandas as pd



def read_txt(input_path):

    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items

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

def create_prompt(entry, case_entry):
    question = entry['qa']['question']
    program = entry['qa']['program']
    answer = entry['qa']['exe_ans']

    # retrieved contexts
    context = entry['qa']['model_input']
    context_input=""
    for item in context:
        context_input+=item[1]
        context_input+="\n"

    table = entry['table']

    # full contexts
    # pre_text=""
    # for item in entry['pre_text']:
    #     pre_text+=item
    #     pre_text+="\n"

    # post_text=""
    # for item in entry['post_text']:
    #     post_text+=item
    #     post_text+="\n"
    
    table_text=""
    for i in range(len(table)):
        item = table_row_to_text(table[0], table[i])
        table_text+=item
        table_text+="\n"

    all_text=context_input+table_text
    # all_text=pre_text+post_text+table_text

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

    numbers = list(set(numbers))

    # retrieved cases
    case_retrieved = case_entry['case_retrieved'][:3]
    case_input=""
    for case in case_retrieved:
        case_question = case['question']
        case_program = case['program'][:-5]
        case = "Similar Question: " + case_question + "\n" \
            + "Similar Program: " + case_program
        case_input+=case
        case_input+="\n"

    # guide_prompt = '''
    # 1. You are given similar questions and their answer programs to solve the questions.
    # 2. Please answer the question based on the given context.
    # 3. Your response should be in the 'program' format. For example, "divide(number, number), divide(number, number)".  
    # 4. The operations should be selected from the given operation list, and more than one operation can be used.
    # 5. The arguments in the program should be selected from the given argument list.
    # 6. Do not include any other words besides the program in your response. 
    # '''

    # input_prompt = f'''
    # {case_input}

    # Contexts:
    # {context_input}

    # Table: {table}
    

    # Possible Operations: {op_list}

    # Possible Arguments: {numbers} {const_list}
    
    # Question: {question}
    # '''

    guide_prompt = '''
    1. Please answer the question based on the given context.
    2. Your response should be in the 'program' format. For example, "divide(number, number), divide(number, number)".  
    3. The operations should be selected from the given operation list, and more than one operation can be used.
    4. The arguments in the program should be selected from the given argument list.
    5. Do not include any other words besides the program in your response. 
    '''

    input_prompt = f'''
    Contexts:
    {context_input}

    Table: {table}
    

    Possible Operations: {op_list}

    Possible Arguments: {numbers} {const_list}
    
    Question: {question}
    '''


    full_prompt = f'''<s> {guide_prompt}

    Input: {input_prompt}

    Output: {program} </s>

    '''

    return full_prompt


def read_examples(input_path, case_path):

    with open(input_path) as input_file:
        input_data = json.load(input_file)

    with open(case_path) as case_file:
        case_data = json.load(case_file)

    # input_data = input_data[:100]
    # case_data = case_data[:100]

    prompts = ["text"]
    for entry, case_entry in zip(input_data, case_data):
        prompts.append(create_prompt(entry, case_entry))
    return prompts


def save_csv(csv_path, prompts):
    df=pd.DataFrame(prompts)
    df.to_csv(csv_path, index=False,header=None)


if __name__ == "__main__":
    op_list_file = "/data2/yikyungkim/dataset/finqa_original/operation_list.txt"
    const_list_file = "/data2/yikyungkim/dataset/finqa_original/constant_list.txt"
    op_list = read_txt(op_list_file)
    const_list = read_txt(const_list_file)
    const_list = [const.lower().replace('.', '_') for const in const_list]

    # input_data = '/data2/yikyungkim/dataset/finqa_original/train.json'
    input_data = '/data2/yikyungkim/dataset/finqa_retriever_output/test_retrieve.json'

    # case_data = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300/generator_input/valid_100/results/predictions.json'
    case_data = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300/training_100/results/predictions.json'

    prompts = read_examples(input_data, case_data)

    csv_file = '/home/yikyungkim/CBR-FinQA/generator_llama2/Finetune_LLMs/finetuning_repo/cbr_test_noCase.csv'
    save_csv(csv_file, prompts)
