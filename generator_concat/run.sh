#!/bin/bash
  
#SBATCH --job-name=generator
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=32
#SBATCH --partition=P2
#SBATCH --output=/home/s3/yikyungkim/research/cbr/generator_concat/slurm_output/%j.out 

source /home/s3/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
PYTHONPATH=.

srun python Test.py