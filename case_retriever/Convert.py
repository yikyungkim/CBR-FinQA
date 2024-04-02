
import json
import random

""" case retriever output을 generator의 input으로 바꿔주기 위함 """
def convert_case_retrieved(json_in, json_out, mode, k):

    with open(json_in) as f_in:
        inf_data = json.load(f_in)

    # noise: case input으로 case retriever의 top-k 사용
    if mode == 'noise':
        for data in inf_data:
            retrieved = data['reranked'][:k]
            # retrieved = data['reranked_cross'][:k]
            data['model_input']=retrieved
    # gold: case input으로 gold 사용.
    # gold 갯수가 k보다 작은 경우 gold & top-(k-gold) 사용.
    # gold가 없는 경우 case input 없음.
    else:
        for data in inf_data:
            golds = data['gold_index']
            gold_index = [gold['index'] for gold in golds]
            gold_num = len(gold_index)
            retrieved = data['reranked_cross']

            model_input=[]
            if gold_num >= k:
                random.shuffle(golds)
                model_input.extend(golds[:k])
            elif 0 < gold_num < k:
                model_input.extend(golds)
                for retrieve in retrieved:
                    if len(model_input)==k:
                        break
                    if retrieve['index'] not in gold_index:
                        model_input.append(retrieve)            
            data['model_input']=model_input

    with open(json_out, "w") as f_out:
        json.dump(inf_data, f_out, indent=4)


""" argument가 masking되지 않은 output에 대해 argument를 masking해주어 model input에 넣어줌 (기존 generator 사용목적) """
def new_convert_case_retrieved(json_in, json_out, mode, k, constants):

    with open(json_in) as f_in:
        inf_data = json.load(f_in)

    for data in inf_data:
        retrieved = data['reranked_cross'][:k]
        
        for case in retrieved:
            program = program_tokenization(case['program'])

            i=0
            while i < len(program):
                if i%4==1:
                    if program[i] not in constants:
                        program[i]='arg1, '
                    else:
                        if 'const' in program[i]:
                            program[i]='const, '
                        else:
                            program[i]=(program[i]+', ')
                elif i%4==2:
                    if program[i] not in constants:
                        program[i]='arg2'
                    else:
                        if 'const' in program[i]:
                            program[i]='const'
                elif i%4==3:
                    if i != (len(program)-1):
                        program[i]='), '
                i+=1
            
            mask_program = ""
            for i in range(len(program)):
                mask_program+=program[i]
        
            case['program']=mask_program
        
        data['model_input']=retrieved
 
    with open(json_out, "w") as f_out:
        json.dump(inf_data, f_out, indent=4)



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



def read_txt(input_path):
    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        item = line.strip().lower()
        items.append(item)
    return items

if __name__ == '__main__':

    mode = 'noise'
    json_in = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/inference/cross_roberta-base_q+p/on_bienc_results_2/10_cands_test/results/predictions.json'

    output_path = '/shared/s3/lab07/yikyung/cbr/dataset/case_retriever_output/cross_encoder/cross_30/on_bienc_results_2_10/'
    json_out = output_path + 'test_retrieved_noise3_program.json'

    constants_path='/shared/s3/lab07/yikyung/cbr/dataset/finqa_original/constant_list.txt'
    constants=read_txt(constants_path)

    convert_case_retrieved(json_in, json_out, mode, k=3)
    # new_convert_case_retrieved(json_in, json_out, mode, 3, constants)


