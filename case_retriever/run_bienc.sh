#!/bin/bash
  
#SBATCH --job-name=biencoder
#SBATCH --nodelist=master
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00

#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --output=/home/yikyungkim/CBR-FinQA/case_retriever/slurm_output/%j.out 

source /home/${USER}/.bashrc
source /data2/yikyungkim/anaconda3/etc/profile.d/conda.sh

PYTHONPATH=.
srun python biencoder_test.py