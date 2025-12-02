#!/bin/bash
#SBATCH --job-name=train_verifier
#SBATCH --output=/scratch/dq2024/doc-verifier/out_files/lora.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256GB
#SBATCH --account=torch_pr_152_courant


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity exec ${nv} \
    --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/env.sh
conda activate dr-env-3-9

export HF_TOKEN='${HF_TOKEN}'


cd /scratch/dq2024/doc-verifier

torchrun --nproc_per_node=2 --master_port=29501 train_lora.py
"