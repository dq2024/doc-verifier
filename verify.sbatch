#!/bin/bash
#SBATCH --job-name=filter_docs
#SBATCH --output=/scratch/dq2024/doc-verifier/out_files/verify.out
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
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

python verify.py \
    --input /scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100.json \
    --output /scratch/dq2024/doc-verifier/verified_output.json \
    --model-dir /scratch/dq2024/doc-verifier/models/llama-3.2-1b-verifier-classifier-query_split \
    --base-model meta-llama/Llama-3.2-1B \
    --batch-size 16 \
    --max-length 1024 \
    --threshold 0.0
"