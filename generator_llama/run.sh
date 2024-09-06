#!/bin/bash
  
#SBATCH --job-name=llama
#SBATCH --nodelist=master
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=24GB
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=/home/yikyungkim/CBR-FinQA/generator_llama/slurm_output/%j.out 

source /home/${USER}/.bashrc
source /data2/yikyungkim/anaconda3/etc/profile.d/conda.sh

PYTHONPATH=.

torchrun --nproc_per_node 1 llama.py