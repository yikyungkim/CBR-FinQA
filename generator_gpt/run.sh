#!/bin/bash
  
#SBATCH --job-name=gpt
#SBATCH --nodelist=master
#SBATCH --gres=gpu:0
#SBATCH --time=0-12:00:00
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=/home/yikyungkim/CBR-FinQA/generator_gpt/slurm_output/%j.out 

source /home/${USER}/.bashrc
source /data2/yikyungkim/anaconda3/etc/profile.d/conda.sh

PYTHONPATH=.

srun python gpt.py