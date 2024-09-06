class parameters():

    prog_name = "generator"

    # set up your own path here
    root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = "/shared/s3/lab07/yikyung/cbr/generator_sep/output/"
    cache_dir = "/home/s3/yikyungkim/research/cbr/cache/"

    model_save_name = "roberta-large-case1"    # directory name

    # train_file = root_path + "dataset/train.json"
    # valid_file = root_path + "dataset/dev.json"
    # test_file = root_path + "dataset/test.json"

    ### files from the retriever results
    train_file = root_path + "dataset/finqa_retriever_output/train_retrieve.json"
    valid_file = root_path + "dataset/finqa_retriever_output/dev_retrieve.json"
    test_file = root_path + "dataset/finqa_retriever_output/test_retrieve.json"

    ### files from case retriever
    train_case = root_path + "dataset/case_retriever_output/cross_encoder/train_retrieved_noise1_L.json"
    valid_case = root_path + "dataset/case_retriever_output/cross_encoder/dev_retrieved_noise1_L.json"
    test_case = "/home/ubuntu/yikyung/dataset/case_retriever_output/cross_encoder/test_retrieved_noise1_L.json"

    # infer table-only text-only
    # test_file = root_path + "dataset/test_retrieve_7k_text_only.json"

    op_list_file = root_path + "dataset/finqa_original/operation_list.txt"
    const_list_file = root_path + "dataset/finqa_original/constant_list.txt"

    # # model choice: bert, roberta, albert
    # pretrained_model = "bert"
    # model_size = "bert-base-uncased"

    # model choice: bert, roberta, albert
    pretrained_model = "roberta"
    model_size = "roberta-large"
    # model_size = "roberta-base"

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
    mode = "train"
    resume = False
    # resume_model = '/home/ubuntu/yikyung/generator_sep/output/roberta-large-case1/saved_model/model_1.pt'
    # wandb_id = 'frx77y9v'
    # saved_model_path = "/home/ubuntu/yikyung/generator_sep/output/roberta-base-case1-noise_20230609152751/saved_model/model_99.pt"
    # threshold = 4.1
    
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
