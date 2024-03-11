#!/bin/bash
  
#SBATCH --job-name=cross_enc
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=P2
#SBATCH --output=/home/s3/yikyungkim/research/cbr/case_retriever/slurm_output/%j.out 
source /home/s3/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
PYTHONPATH=.

srun python cross_encoder.py