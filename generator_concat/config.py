class parameters():

    prog_name = "generator"

    # set up your own path here
    root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = '/shared/s3/lab07/yikyung/cbr/generator_concat/inference/bienc_9/'
    cache_dir = "/home/s3/yikyungkim/research/cbr/cache/"
    model_save_name = "qandp_3_train_all_ponly"    # directory name

    program_type = "prog"    # ops, prog
    input_concat = "ponly"  # qandp, ponly
    num_case = 3    # 1, 3

    # train_file = root_path + "dataset/train.json"
    # valid_file = root_path + "dataset/dev.json"
    # test_file = root_path + "dataset/test.json"

    ### files from the retriever results
    train_file = root_path + "dataset/finqa_retriever_output/train_retrieve.json"
    valid_file = root_path + "dataset/finqa_retriever_output/dev_retrieve.json"
    test_file = root_path + "dataset/finqa_retriever_output/test_retrieve.json"

    # test_file = '/shared/s3/lab07/yikyung/cbr/generator_concat/output_int/test_human_cont_10_yk.json'

    ### files from case retriever
    train_case = '/shared/s3/lab07/yikyung/cbr/dataset/case_retriever_output/cross_encoder/cross_30/on_bienc_results_2/train_golds.json'
    valid_case = '/shared/s3/lab07/yikyung/cbr/dataset/case_retriever_output/cross_encoder/cross_30/on_bienc_results_2/dev_noise.json'
    # test_case = '/shared/s3/lab07/yikyung/cbr/dataset/case_retriever_output/cross_encoder/cross_30/on_bienc_results_2_10/test_retrieved_noise3_program.json'
    test_case = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/inference/bi_bert-base_q+p_mixed_100300-29/training_100/results/predictions.json'
    # test_case = '/shared/s3/lab07/yikyung/cbr/generator_concat/output_int/test_human_case_10_yk.json'

    op_list_file = root_path + "dataset/finqa_original/operation_list.txt"
    const_list_file = root_path + "dataset/finqa_original/constant_list.txt"

    # # model choice: bert, roberta, albert
    # pretrained_model = "bert"
    # model_size = "bert-base-uncased"

    # model choice: bert, roberta, albert
    pretrained_model = "roberta"
    # model_size = "roberta-large"
    model_size = "roberta-base"

    # # finbert
    # pretrained_model = "finbert"
    # model_size = root_path + "pre-trained-models/finbert/"

    # pretrained_model = "longformer"
    # model_size = "allenai/longformer-base-4096"

    # single sent or sliding window
    # single, slide, gold, none
    retrieve_mode = "single"

    # use seq program or nested program
    program_mode = "seq"

    # train, test, or private
    # private: for testing private test data
    device = 'cuda'
    mode = "test"
    resume = False
    # resume_model = '/home/ubuntu/yikyung/generator_concat/output/roberta-large-case3-noise/saved_model/model_5.pt'
    # wandb_id = 'j14sp6ow'

    saved_model_path = "/shared/s3/lab07/yikyung/cbr/generator_concat/train/roberta-base-case3-noise_20230609071454/saved_model/model_99.pt"
    # threshold = 4.6
    
    build_summary = False

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512 # 2k for longformer, 512 for others
    max_program_length = 30
    n_best_size = 20
    dropout_rate = 0.1

    batch_size = 16
    batch_size_test = 16 
    # epoch = 300
    epoch = 100
    learning_rate = 2e-5

    report = 300
    report_loss = 100

    max_step_ind = 11

