#!/bin/bash
  
#SBATCH --job-name=generator
#SBATCH --nodelist=master
#SBATCH --gres=gpu:1
#SBATCH --time=0-48:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=/home/yikyungkim/CBR-FinQA/generator_concat/slurm_output/%j.out 

source /home/${USER}/.bashrc
source /data2/yikyungkim/anaconda3/etc/profile.d/conda.sh

PYTHONPATH=.

srun python Test.py