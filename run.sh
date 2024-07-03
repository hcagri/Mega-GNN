#!/bin/bash

#SBATCH --job-name="multi-gnn"
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu-a100
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16
#SBATCH --account=research-eemcs-st
#SBATCH --output=./%j.out # standard output of the job will be printed here
#SBATCH --error=./%j.err # standard error of the job will be printed here

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate multignn

srun python /home/$USER/projects/Multi-GNN/main.py --data Small_HI --model gin --emlps --reverse_mp --ego --flatten_edges

conda deactivate

nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"