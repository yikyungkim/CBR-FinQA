#!/bin/bash

#SBATCH --job-name=CBR_CL
#SBATCH --nodes=1
#SBATCH --nodelist=n01
#SBATCH --gres=gpu:2
#SBATCH --time=0-48:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=6



srun python3 train.py
srun python3 inference.py