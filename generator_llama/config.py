class parameters():

    prog_name = "generator"

    # set up your own path here
    # root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = '/data2/yikyungkim/generator_llama/inference/'
    cache_dir = "/data2/yikyungkim/cache/"
    dir_name = 'bienc_11_llama2'

    program_type = "prog"    # ops, prog
    input_concat = "qandp"  # qandp, ponly
    num_case = 3    # 1, 3
    use_case = True
    random_case = False

    ### files from the retriever results
    # train_file = "/data2/yikyungkim/dataset/finqa_retriever_output/train_retrieve.json"
    # valid_file = "/data2/yikyungkim/dataset/finqa_retriever_output/dev_retrieve.json"
    # test_file = "/data2/yikyungkim/dataset/finqa_retriever_output/test_retrieve.json"
    test_file = '/data2/yikyungkim/dataset/finqa_original/test_random1.json'

    ### files from case retriever
    # train_case = "/data2/yikyungkim/dataset/finqa_original/train.json"
    # valid_case = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300/generator_input/valid_100/results/predictions.json'
    # test_case = "/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300/training_100/results/predictions.json"        #bienc_11
    # test_case = '/data2/yikyungkim/case_retriever/inference/cross_roberta-base_q+p_100200/training_300/results/predictions.json'           # cross_41
    test_case = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300/training_100_random/results/predictions.json'     #bienc_11_small testset
    # test_case = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300_random/test_random/results/predictions.json'  #bienc_11_small testset random1%
    # test_case = '/data2/yikyungkim/case_retriever/inference/not_trained/test_random_train_random/predictions.json'        # not trained case retriever (inference only)
    # test_case = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300_random10/test_random/results/predictions.json'   #bienc_11_small testset random10%

    op_list_file = "/data2/yikyungkim/dataset/finqa_original/operation_list.txt"
    const_list_file = "/data2/yikyungkim/dataset/finqa_original/constant_list.txt"

    archive_path = "/data2/yikyungkim/generator_concat/archives/"


    ### Set model
    # ckpt_dir = '/data2/yikyungkim/meta-llama/llama3/Meta-Llama-3-8B-Instruct/'
    # tokenizer_path = '/data2/yikyungkim/meta-llama/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model'
    # temperature = 0.6
    # top_p = 0.9
    # max_seq_len = 4096
    # max_batch_size = 4
    # max_gen_len = None

    # i = 46
    # j = 50

    # guide_prompt = '''
    # 1. Please answer the question based on the given context.
    # 2.  Your response should be in the 'program' format. For example, "divide(number, number), divide(number, number)".  
    # 3. The operations should be selected from the given operation list, and more than one operation can be used.
    # 4. The arguments in the program should be selected from the given argument list.
    # 5. Do not include any other words besides the program in your response. 
    # '''

    # guide_prompt = '''
    # 1. You are given sample questions and their answer programs to solve the questions.
    # 2. Please answer the question based on the given context.
    # 3. Your response should be in the 'program' format. For example, "divide(number, number), divide(number, number)".  
    # 4. The operations should be selected from the given operation list, and more than one operation can be used.
    # 5. The arguments in the program should be selected from the given argument list.
    # 6. Do not include any other words besides the program in your response. 
    # '''

    guide_prompt = '''
    You are an expert in finance industry.
    1. You are given similar questions and their answer programs to solve the questions.
    2. Please answer the question based on the given context.
    3. Your response should be in the 'program' format. For example, "divide(number, number), divide(number, number)".  
    4. The operations should be selected from the given operation list, and more than one operation can be used.
    5. The arguments in the program should be selected from the given argument list.
    6. Do not include any other words besides the program in your response. 
    '''
    
    # single sent or sliding window
    # single, slide, gold, none
    retrieve_mode = "single"

    # use seq program or nested program
    program_mode = "seq" 
