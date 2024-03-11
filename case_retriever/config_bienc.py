class parameters():

    """Set path"""
    root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/'         # train, inference

    cache_dir = "/home/s3/yikyungkim/research/cbr/cache"

    """Set dataset path"""
    train_file = root_path + "dataset/case_retriever/new_train_sample.json"    
    valid_file = root_path + "dataset/case_retriever/new_dev_sample.json"
    
    # train_file = root_path + "dataset/case_retriever/new_train_ops.json"    
    # valid_file = root_path + "dataset/case_retriever/new_dev_ops.json"


    """ For inference """
    # we need inference file for train, valid, and test 
    # saved_model_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/bi_bert-base_q+p/model/epoch_18'    
    # saved_model_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/bi_bert-base-ops_q+p/model/epoch_18' #qandp
    # saved_model_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/bi_bert-base-ops_p/model/epoch_15' # ponly

    # inference_file = root_path + "dataset/case_retriever/new_train_ops.json"    
    # inference_file = root_path + "dataset/case_retriever/new_dev_ops.json"
    # inference_file = root_path + "dataset/case_retriever/new_test_ops.json"    
    # inference_file = root_path + "dataset/case_retriever/new_train.json"    
    # inference_file = root_path + "dataset/case_retriever/new_dev.json"    
    # inference_file = root_path + "dataset/case_retriever/new_test.json"    

    resume = False
    resume_model = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/bi_bert-base_q+p_sep_nosampling/model/epoch_4'   #qandp
    wandb_id = 'hogklrr3'           # qandp


    """Set model"""  
    mode = 'train'                          # train, test
    dir_name = 'bi_bert-base_q+p_loss'        # qandp

    model = 'bert'       
    bert_size = "bert-base-uncased"         
    # model = 'roberta'        
    # bert_size = "roberta-base"     


    """Set model config"""              
    input_concat = 'qandp'          # qandp, qonly, ponly               (qandp -> model resize)
    program_type = 'prog'           # prog, ops
    negative_type = 'random'          # random, hard, adjusted_hard

    data_type = 'base'
    # num_test = 60

    loss_adjust = True

    device = "cuda"
    epoch = 30
    max_seq_len = 128       
    batch_size = 1                 # 64 (loss_adjust=False), 1 (loss_adjust=True)
    batch_size_test = 1            # 64 (loss_adjust=False), 1 (loss_adjust=True)

    learning_rate = 2e-5       
    warm_up_prop = 0.2  # scheduler 
    patience = 10       # early stopping
    dropout_rate = 0.1  # used for cross-encoder

    num_cand = 100
    K_pos = 40          # number of positives examples
    neg_ratio = 2         # ratio of negative examples to positive examples
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