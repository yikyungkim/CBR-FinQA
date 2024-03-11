class parameters():

    """Set model path"""
    root_path = "/shared/s3/lab07/yikyung/cbr/"
    output_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/inference/cross_roberta-base_q+p/on_case_dataset/'                 # train/inference
    # output_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/inference/cross_roberta-base-ops-only_p-16/on_bienc_results_11/'

    cache_dir = "/home/s3/yikyungkim/research/cbr/cache"

    """Set dataset path"""
    train_file = root_path + "dataset/case_retriever/new_train.json"    
    valid_file = root_path + "dataset/case_retriever/new_dev.json"

    # train_file = '/shared/s3/lab07/yikyung/cbr/dataset/case_retriever/new_train_sample.json'
    # valid_file = '/shared/s3/lab07/yikyung/cbr/dataset/case_retriever/new_dev_sample.json'

    # train_file = root_path + "dataset/case_retriever/new_train_ops.json"    
    # valid_file = root_path + "dataset/case_retriever/new_dev_ops.json"


    """ For inference """
    # we need inference file for train, valid, and test 

    """ cross roberta q+p 이용시"""
    # Bert_model: if 문 삭제 (init, forward), def __init__ tokenizer 삭제
    # def test(): model = tokenizer 없는거로

    saved_model_path = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/cross_roberta-base_q+p/model/epoch_27'  # operator

    # inference_file = '/home/ubuntu/yikyung/dataset/case_retriever/new_train.json'
    # inference_file = '/home/ubuntu/yikyung/dataset/case_retriever/new_dev.json'

    # inference_file = root_path + "dataset/case_retriever/new_test_ops.json"           # operators      
    inference_file = root_path + "dataset/case_retriever/new_test.json"                 # programs  
    
    # inference_file = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/inference/bi_bert-base-ops_p-15/results_100/predictions.json'  # Biencoder_11
    # inference_file = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/inference/bi_bert-base-ops_q+p-18/results_100/predictions.json'     # Biencoder_12
    # inference_file = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/inference/bi_bert-base-ops_q+p-18/results_100/predictions_masked.json'     # Biencoder_12_masked
    

    """Set model"""  
    mode = 'test'                          # train, test
    # dir_name = 'cross_roberta-base_q+p_bce-29'  
    dir_name = '100_cands_softmax'         

    input_concat = 'qandp'                  # qandp, ponly
    program_type = 'prog'                    # prog, ops
    negative_type = 'random'                  # random, hard, adjusted_hard

    data_type = 'base'                      # inference data: base, biencoder
    num_test = 60

    resume = False
    resume_model = '/shared/s3/lab07/yikyung/cbr/case_retriever/output/train/cross_roberta-base_q+p_neg/model/epoch_6'
    wandb_id = 'cxzf79t9'     


    """Set model config"""              
    # model = 'bert'       
    # bert_size = "bert-base-uncased"   
          
    model = 'roberta'        
    bert_size = "roberta-base"     
    # bert_size = "roberta-large"    
    # 
    loss = 'cross'                    # bce, cross
    sort = 'softmax'                    # softmax, score 

    device = "cuda"
    epoch = 30
    max_seq_len = 512       
    # batch_size = 32         
    batch_size = 64         
    # batch_size_test = 64   
    batch_size_test = 128   

    learning_rate = 2e-5       
    warm_up_prop = 0.2  # scheduler 
    patience = 10       # early stopping
    dropout_rate = 0.1  # used for cross-encoder

    # K_pos = 40          # number of positives examples
    # K_pos = 37          # number of positives examples
    neg_ratio = 2         # ratio of negative examples to positive examples
    hard_ratio = 0.5      # ratio of hard negatives in negatives
    fix_ratio = 0.5       # ratio of fixed hard and easy negatives in hard and easy negatives (rest = random)


    average = "macro"   # for evaluation metrics
    topk = 10           # get top-k re-ranked result
    report_loss = 100   # record loss in log_file for every n batches       
