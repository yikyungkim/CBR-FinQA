#!/bin/bash
  
#SBATCH --job-name=llama3
#SBATCH --nodelist=master
#SBATCH --gres=gpu:2
#SBATCH --time=0-24:00:00

#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=/home/yikyungkim/CBR-FinQA/generator_llama2/Finetune_LLMs/finetuning_repo/slurm/%j.out 

source /home/${USER}/.bashrc
source /data2/yikyungkim/anaconda3/etc/profile.d/conda.sh
conda activate qlora


python trl_inference.py