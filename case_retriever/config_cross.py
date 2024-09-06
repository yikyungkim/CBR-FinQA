class parameters():

    """Set path"""
    # root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = '/data2/yikyungkim/case_retriever/train/'
    cache_dir = "/data2/yikyungkim/cache"


    """Set dataset path"""
    train_file = "/data2/yikyungkim/dataset/case_retriever/train_score_100.json" 
    valid_file = "/data2/yikyungkim/dataset/case_retriever/dev_score_100.json"

    train_original = "/data2/yikyungkim/dataset/finqa_original/train.json"
    valid_original = "/data2/yikyungkim/dataset/finqa_original/dev.json"
    constant_file = "/data2/yikyungkim/dataset/finqa_original/constant_list.txt"
    archive_path = "/data2/yikyungkim/case_retriever/archives/"

    """ For inference """
    # we need inference file for train, valid, and test 
    #  cross roberta q+p 이용시 !!!
    # Bert_model: if 문 삭제 (init & forward), def __init__ tokenizer 삭제
    # def test(): model = tokenizer 없는거로

    saved_model_path = '/data2/yikyungkim/case_retriever/train/cross_roberta-base_q+p/epoch_27'  # operator
    inference_file = "/data2/yikyungkim/dataset/finqa_original/train.json"               
    # inference_file = '/data2/yikyungkim/case_retriever/inference/bi_bert-base_q+p_mixed_100300/training_100_top100/results/predictions.json'

    data_type = 'base'                      # inference data: base, biencoder
    bienc_num = 100


    """ Resume training """
    resume = False
    resume_model = '/data2/yikyungkim/case_retriever/train/cross_roberta-base_q+p_100300/model/epoch_27' 
    # wandb_id = 'p2llocjs'        #100100
    # wandb_id = 'ywhh7j9g'       #100200
    wandb_id = 'opmg59hr'       #100300
    # wandb_id = 'cyza9gif'       #66300


    """Set model"""  
    mode = 'train'                          # train, test
    dir_name = 'cross_roberta-large_q+p_100200'         

    # model = 'bert'       
    # bert_size = "bert-base-uncased"             
    model = 'roberta'        
    # bert_size = "roberta-base"     
    bert_size = "roberta-large"    


    """Set model config"""              
    input_concat = 'qandp'                  # qandp, ponly
    program_type = 'prog'                    # prog, ops

    loss = 'cross'                    # bce, cross
    sort = 'softmax'                  # softmax, score

    device = "cuda"
    epoch = 30
    max_seq_len = 128       
    batch_size = 128         
    batch_size_test = 128   

    learning_rate = 2e-5       
    warm_up_prop = 0.2  # scheduler 
    patience = 10       # early stopping
    dropout_rate = 0.1  # used for cross-encoder

    average = "macro"   # for evaluation metrics
    topk = 10           # get top-k re-ranked result
    report_loss = 100   # record loss in log_file for every n batches       


    """For training set"""              
    # K_pos = 40          # number of positives examples
    train_size = 100          # size of training set
    neg_ratio = 2             # ratio of negative examples to positive examples
    pos_pool = 100             # number of gold candidates (postive case pool)  
    neg_pool = 200            # number of non-gold candidates (negative case pool)  
    sampling = 'mixed'        # how to get samples. random, hard, mixed

    use_all_cands = True            # sampling from all candiates or not (True: get samples from all 6250 candidates, False: get samples from top-100 question similar candidates)
    q_score_available = True        # if question similarity score is already computed or not (similarity between training question <-> training question)
    p_score_available = True        # if program score is already computed or not 
    candidates_available = True     # if top-(pool) question similar candidates are saved or not


    """For validation & test set"""              
    q_score_avail_test = True          # if question similarity score is already computed or not (similarity between inference question <-> training question)
    p_score_avail_test = True          # if program score is already computed or not 
    num_test = 300                      # number of question similar cases to use for inference (100, 300, 1000, 6251)
    candidates_avail_test = True       # if top-(pool) question similar candidates are saved or not
    test_feature_available = False      # if converted features are saved or not
