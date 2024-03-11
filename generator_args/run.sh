#!/bin/bash
  
#SBATCH --job-name=generator_args
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=32
#SBATCH --partition=P2
#SBATCH --output=/home/s3/yikyungkim/research/cbr/generator_args/slurm_output/%j.out 
source /home/${USER}/.bashrc
source ~/anaconda/etc/profile.d/conda.sh
PYTHONPATH=.

srun python main.py