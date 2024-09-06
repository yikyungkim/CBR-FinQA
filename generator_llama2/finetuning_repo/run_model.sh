#!/usr/bin/python3

#SBATCH --job-name=llama_desc
#SBATCH --nodelist=n01
#SBATCH --gres=gpu:2
#SBATCH --time=0-48:00:00
#SBATCH --mem=40000MB
#SBATCH --cpus-per-task=4


python trl_finetune.py --block_size 1024 --eval_steps 368 --save_steps 368 --log_steps 368 -tf cbr_train.csv -vf cbr_dev.csv -m meta-llama/Llama-2-13b-chat-hf -o ./checkpoints2 --split_model -b 1 -lr 2e-4 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18610 --use_int4 --token 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
python trl_finetune.py --block_size 1024 --eval_steps 368 --save_steps 368 --log_steps 368 -tf cbr_train_noCase.csv -vf cbr_dev_noCase.csv -m meta-llama/Llama-2-13b-chat-hf -o ./checkpoints3 --split_model -b 1 -lr 2e-4 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18610 --use_int4 --token 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


python trl_finetune.py --block_size 1024 --eval_steps 10 --save_steps 10 --log_steps 10 -tf cbr_train.csv -vf cbr_dev.csv -m meta-llama/Llama-2-13b-chat-hf -o ./checkpoints_llama2_loss --split_model -b 1 -lr 2e-4 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18610 --use_int4 --token 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


python trl_finetune.py --block_size 1024 --eval_steps 10 --save_steps 10 --log_steps 10 -tf cbr_train.csv -vf cbr_dev.csv -m meta-llama/Meta-Llama-3-8B -o /data2/yikyungkim/generator_llama/llama3_case --split_model -b 1 -lr 2e-4 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18610 --use_int4 --token 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
python trl_finetune.py --block_size 1024 --eval_steps 10 --save_steps 10 --log_steps 10 -tf cbr_train_noCase.csv -vf cbr_dev_noCase.csv -m meta-llama/Meta-Llama-3-8B -o /data2/yikyungkim/generator_llama/llama3_noCase --split_model -b 1 -lr 2e-4 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18610 --use_int4 --token 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






#!/bin/bash
  
#SBATCH --job-name=llama
#SBATCH --nodelist=n02
#SBATCH --gres=gpu:4
#SBATCH --time=0-48:00:00

#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --output=/home/yikyungkim/CBR-FinQA/generator_llama2/Finetune_LLMs/finetuning_repo/slurm/%j.out 

source /home/${USER}/.bashrc
source /data2/yikyungkim/anaconda3/etc/profile.d/conda.sh
conda activate qlora

conda install git pip
pip install -U git+https://github.com/huggingface/accelerate

srun python trl_finetune.py --block_size 1024 --eval_steps 10 --save_steps 10 --log_steps 10 -tf cbr_train.csv -vf cbr_dev.csv -m meta-llama/Meta-Llama-3-8B -o ./checkpoints_llama3 --split_model -b 1 -lr 2e-4 -e 1 --gradient_accumulation_steps 16 --pad_token_id=18610 --use_int4 --token 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
