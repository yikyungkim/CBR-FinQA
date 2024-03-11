class parameters():

    prog_name = "generator_args"

    # set up your own path here
    root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = "/shared/s3/lab07/yikyung/cbr/generator_args/output/"
    cache_dir = "/home/s3/yikyungkim/research/cbr/cache/"

    model_save_name = "roberta-base-bs32-noise-logits-order-layer-gold_cases"    # directory name

    ### files from the retriever results
    train_file = root_path + "dataset/finqa_retriever_output/train_retrieve.json"
    valid_file = root_path + "dataset/finqa_retriever_output/dev_retrieve.json"

    # roberta-base (full)
    # test_file = "/shared/s3/lab07/yikyung/cbr/generator_concat/output/inference_only_20230613153256_roberta-base-case3-noise-program/results/test/full_results.json"    
    
    # roberta-base (error)
    # test_file = "/shared/s3/lab07/yikyung/cbr/generator_concat/output/inference_only_20230613153256_roberta-base-case3-noise-program/results/test/full_results_error.json"  
    
    # roberta-base (w. retrieved contexts & gold cases)
    test_file = '/shared/s3/lab07/yikyung/cbr/generator_concat/output/inference_only_20230605183446_roberta-base-case1-rc_gc-test-cross/results/test/full_results.json'

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

    # saved_model_path = "/shared/s3/lab07/yikyung/cbr/generator_args/output/roberta-base-bs32-noise-logits-order/saved_model/model_80.pt"
    saved_model_path = '/shared/s3/lab07/yikyung/cbr/generator_args/output/roberta-base-bs32-noise-logits-order-layer/saved_model/model_99.pt'

    build_summary = False

    sep_attention = True
    layer_norm = True
    num_decoder_layers = 1

    max_seq_length = 512 # 2k for longformer, 512 for others

    n_best_size = 20
    dropout_rate = 0.1

    batch_size = 32
    batch_size_test = 32
    epoch = 100
    learning_rate = 2e-5
    warm_up_prop = 0.2  # scheduler 

    report = 300
    report_loss = 100

    max_step_ind = 11

