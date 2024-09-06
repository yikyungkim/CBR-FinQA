#!/bin/bash 
# 
#SBATCH --job-name="gen_sep"
#SBATCH --partition=LocalQ
#SBATCH --mem=10gb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output="/home/ubuntu/yikyung/generator_sep/slurm_output/%j.out"
#SBATCH --nodelist=gpu-1

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh

CUDA_VISIBLE_DEVICES= 0,1,2,3
python "Main.py"
