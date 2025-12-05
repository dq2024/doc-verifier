#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=merge
#SBATCH --output=/scratch/dq2024/doc-verifier/out_files/merge.out
#SBATCH --account=torch_pr_152_courant


if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi


singularity exec ${nv} \
            --overlay /scratch/$USER/dr-env-3-9.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate dr-env-3-9;

python merge_json.py
"
