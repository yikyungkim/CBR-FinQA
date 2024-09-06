import json
from gpt import *
from config import parameters as conf

def load_data(input_path):
    with open(input_path) as input_file:
        input_data = json.load(input_file)
    return input_data

def write_data(output, output_file):
    with open(output_file, "w") as writer:
        writer.write(json.dumps(output, indent=4) + "\n")

def refine_further(input_data):
    for i in range(len(input_data)):
        data = input_data[str(i)][0]
        predicted = data['pred_prog']
        refined=[]
        for tok in predicted:
            tok = tok.replace(" ","")
            tok = tok.replace("\n","")
            refined.append(tok)
        data['pred_prog']=refined
    return input_data


if __name__ == "__main__":
    nbest_predictions_path = '/data2/yikyungkim/generator_gpt/inference/gpt-4o_cbr_fulltable/bienc_11_small_10percent/results/nbest_predictions.json'
    nbest_predictions_output = '/data2/yikyungkim/generator_gpt/inference/gpt-4o_cbr_fulltable/bienc_11_small_10percent/results/nbest_predictions_new.json'
    nbest_predictions = load_data(nbest_predictions_path)
    refined_predictions = refine_further(nbest_predictions)
    write_data(refined_predictions, nbest_predictions_output)

    eval_file = '/data2/yikyungkim/generator_gpt/inference/gpt-4o_cbr_fulltable/bienc_11_small_10percent/results/full_results_new.json'
    error_file = '/data2/yikyungkim/generator_gpt/inference/gpt-4o_cbr_fulltable/bienc_11_small_10percent/results/full_results_error_new.json'
    exe_acc, prog_acc, op_acc = evaluate_result(nbest_predictions_output, conf.test_file, conf.test_case, eval_file, error_file, conf.program_mode)
    prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc) + " operator acc: " + str(op_acc)
    print(prog_res)

