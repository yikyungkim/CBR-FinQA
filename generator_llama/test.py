
log_file = '/data2/yikyungkim/generator_llama/inference/question_similarity/log.txt'


def read_txt(input_path):

    with open(input_path) as input_file:
        input_data = input_file.readlines()
    items = []
    for line in input_data:
        items.append(line.strip())
    return items

logs = read_txt(log_file)

exe_results=0.0
prog_results=0.0
op_results=0.0

for log in logs:
    if "exe acc" in log:
        log_split = log.split(' ')
        exe_results+=float(log_split[2])
        prog_results+=float(log_split[5])
        op_results+=float(log_split[8])

print(exe_results/13)
print(prog_results/13)
print(op_results/13)