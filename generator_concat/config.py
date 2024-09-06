class parameters():

    prog_name = "generator"

    # set up your own path here
    # root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = '/data2/yikyungkim/generator_concat/inference/'
    cache_dir = "/data2/yikyungkim/cache/"
    model_save_name = "roberta-base-case3_ponly-noise_85_sampling100100"    # directory name

    program_type = "prog"    # ops, prog
    input_concat = "ponly"  # qandp, ponly
    num_case = 3    # 1, 3


    # train_file = root_path + "dataset/train.json"
    # valid_file = root_path + "dataset/dev.json"
    # test_file = root_path + "dataset/test.json"

    ### files from the retriever results
    # train_file = "/data2/yikyungkim/dataset/finqa_retriever_output/train_retrieved_random10.json"
    # valid_file = "/data2/yikyungkim/dataset/finqa_retriever_output/dev_retrieved_random10.json"
    # test_file = "/data2/yikyungkim/dataset/finqa_retriever_output/test_retrieved_random10.json"
    train_file = "/data2/yikyungkim/dataset/finqa_retriever_output/train_retrieve.json"
    valid_file = "/data2/yikyungkim/dataset/finqa_retriever_output/dev_retrieve.json"
    test_file = "/data2/yikyungkim/dataset/finqa_retriever_output/test_retrieve.json"

    # test_file = '/shared/s3/lab07/yikyung/cbr/generator_concat/output_int/test_human_cont_10_yk.json'

    ### files from case retriever
    # train_case = "/data2/yikyungkim/dataset/finqa_original/train_random10.json"
    # valid_case = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300_random10/dev/results/predictions.json'
    # test_case = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300_random10/test/results/predictions.json'          
    test_case = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300/training_100/results/predictions.json'

    op_list_file = "/data2/yikyungkim/dataset/finqa_original/operation_list.txt"
    const_list_file = "/data2/yikyungkim/dataset/finqa_original/constant_list.txt"

    archive_path = "/data2/yikyungkim/generator_concat/archives/"


    # # model choice: bert, roberta, albert
    # pretrained_model = "bert"
    # model_size = "bert-base-uncased"

    # model choice: bert, roberta, albert
    pretrained_model = "roberta"
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
    # resume_model = '/data2/yikyungkim/generator_concat/train/roberta-large-case3_ponly-noise_75_sampling100100/saved_model/model_63.pt'
    # wandb_id = '8i4cuy36'

    saved_model_path = "/data2/yikyungkim/generator_concat/train/roberta-base-case3_ponly-noise_85_sampling100100/saved_model/model_92.pt"
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


    """For training set"""              
    use_retrieved_cases = False            # sampling from all candiates or not (True: get samples from all 6250 candidates, False: get samples from top-100 question similar candidates)
    top3_precision_val = 0.75

    pos_pool = 20             # number of gold candidates (postive case pool)  
    neg_pool = 20            # number of non-gold candidates (negative case pool)  

    q_score_available = False        # if question similarity score is already computed or not (similarity between training question <-> training question)
    p_score_available = False        # if program score is already computed or not 
    candidates_available = False     # if top-(pool) question similar candidates are saved or not


    """For validation & test set"""              
    # use retrieved cases from valid_case, test_case
