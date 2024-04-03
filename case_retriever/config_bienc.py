class parameters():

    """Set path"""
    root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/'         # train, inference
    cache_dir = "/home/s3/yikyungkim/research/cbr/cache"

    """Set dataset path"""
    train_file = root_path + "dataset/case_retriever/train_score_100.json"    
    valid_file = root_path + "dataset/case_retriever/dev_score_100.json"

    train_original = root_path + "dataset/finqa_original/train.json"
    constant_file = root_path + "dataset/finqa_original/constant_list.txt"
    archive_path = root_path + 'dataset/archives/'

    """ For inference """
    # we need inference file for train, valid, and test 
    # saved_model_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/bi_bert-base_q+p/model/epoch_18'    
    # inference_file = root_path + "dataset/case_retriever/new_train.json"    
    # inference_file = root_path + "dataset/case_retriever/new_dev.json"    
    # inference_file = root_path + "dataset/case_retriever/new_test.json"    

    """ Resume training """
    resume = False
    resume_model = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/bi_roberta-base_q+p_rand_200_single/model/epoch_2' 
    wandb_id = '149k1iyy'           


    """Set model"""  
    mode = 'train'                   # train, test
    dir_name = 'bi_roberta-base_q+p_ep'        

    # model = 'bert'       
    # bert_size = "bert-base-uncased"         
    model = 'roberta'        
    bert_size = "roberta-base"     


    """Set model config"""              
    input_concat = 'qandp'          # qandp, qonly, ponly              
    program_type = 'prog'           # prog, ops

    data_type = 'base'
    # num_test = 60


    device = "cuda"
    epoch = 30
    max_seq_len = 128       
    batch_size = 64                 
    batch_size_test = 64

    learning_rate = 2e-5       
    warm_up_prop = 0.2  # scheduler 
    patience = 10       # early stopping
    dropout_rate = 0.1  # used for cross-encoder

    num_cand = 100 
    # K_pos = 60            # number of positives examples
    sampling='random'       # how to get samples. random, hard, adjusted_hard
    neg_ratio = 2         # ratio of negative examples to positive examples

    use_all_cands = False    # sampling from all candiates or not (True: get samples from all candidates, False: get samples from top-100)
    q_score_available = True
    p_score_available = True

    hard_ratio = 0.7      # ratio of hard negatives in negatives
    fix_ratio = 0.5       # ratio of fixed hard and easy negatives in hard and easy negatives (rest = random)

    average = "macro"   # for evaluation metrics
    topk = 10           # get top-k re-ranked result
    report_loss = 100   # record loss in log_file for every n batches       






    # """for inference (set same as best performance config from sweep results)""" # for bi_bert-base_q
    # epoch = 100
    # max_seq_len = 64    # (number of questions greater than max_seq_len(64) = 6 in training data)
    # batch_size = 64
    # batch_size_test = 64

    # learning_rate = 0.0005534172464058902
    # warm_up_prop = 0.05  # scheduler 
    # patience = 10       # early stopping

    # K_pos = 40          # number of positives examples
    # neg_ratio = 6       # ratio of negative examples to positive examples
    # average = "macro"   # for evaluation metrics
    # topk = 100           # get top-10 re-ranked result
    # report_loss = 100   # record loss in log_file for every n batches   