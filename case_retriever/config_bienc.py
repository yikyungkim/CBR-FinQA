class parameters():

    """Set path"""
    # root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300_random10/'         # train, inference
    cache_dir = "/data2/yikyungkim/cache"

    """Set dataset path"""
    train_file = "/data2/yikyungkim/dataset/case_retriever/train_score_100.json" 
    valid_file = "/data2/yikyungkim/dataset/case_retriever/dev_score_100.json"

    train_original = "/data2/yikyungkim/dataset/finqa_original/train_random10.json"
    valid_original = "/data2/yikyungkim/dataset/finqa_original/dev_random.json"
    constant_file = "/data2/yikyungkim/dataset/finqa_original/constant_list.txt"
    archive_path = "/data2/yikyungkim/case_retriever/archives/"


    """ For inference """
    # we will need inference file for train, valid, and test for program generator
    saved_model_path = '/data2/yikyungkim/case_retriever/train/bi_bert-base_q+p_mixed_100300_random10/model/epoch_7'    
    # inference_file = root_path + "dataset/case_retriever/test_score_100.json"    
    inference_file = "/data2/yikyungkim/dataset/finqa_original/test_random1.json"


    """ Resume training """
    resume = False
    resume_model = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/bi_bert-base_q+p_mixed_cand200/model/epoch_4' 
    wandb_id = '1602imr7'           


    """Set model"""  
    mode = 'test'                   # train, test
    dir_name = 'test_random'        

    model = 'bert'       
    bert_size = "bert-base-uncased"         
    # model = 'roberta'        
    # bert_size = "roberta-base"     


    """Set model config"""              
    input_concat = 'qandp'          # qandp, qonly, ponly              
    program_type = 'prog'           # prog, ops

    device = "cuda"
    epoch = 30
    max_seq_len = 128       
    batch_size = 128                 
    batch_size_test = 128

    learning_rate = 2e-5       
    warm_up_prop = 0.2      # scheduler 
    patience = 10           # early stopping

    average = "macro"       # for evaluation metrics
    topk = 10               # get top-k re-ranked result
    report_loss = 100       # record loss in log_file for every n batches       


    """For training set"""              
    train_size = 100          # size of training set
    neg_ratio = 2             # ratio of negative examples to positive examples
    pos_pool = 100             # number of gold candidates (postive case pool)  
    neg_pool = 300            # number of non-gold candidates (negative case pool)  
    sampling = 'mixed'        # how to get samples. random, hard, mixed

    use_all_cands = True            # sampling from all candiates or not (True: get samples from all 6250 candidates, False: get samples from top-100 question similar candidates)
    q_score_available = False        # if question similarity score is already computed or not (similarity between training question <-> training question)
    p_score_available = False        # if program score is already computed or not 
    candidates_available = False     # if top-(pool) question similar candidates are saved or not


    """For validation & test set"""              
    q_score_avail_test = False          # if question similarity score is already computed or not (similarity between inference question <-> training question)
    p_score_avail_test = False          # if program score is already computed or not 
    num_test = 300                      # number of question similar cases to use for inference (100, 300, 1000, 6251)
    candidates_avail_test = False       # if top-(pool) question similar candidates are saved or not
    test_feature_available = False      # if converted features are saved or not

