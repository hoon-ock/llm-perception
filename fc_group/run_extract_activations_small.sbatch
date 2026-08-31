#!/bin/bash
#SBATCH --job-name=extract_activations_small
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR"

export HF_HOME=/work/ock/hoon/hf_cache

# 7-8B models: fit comfortably on one GPU, run full precision (no quantization).
MODELS=(
  "meta-llama/Llama-3.1-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "Qwen/Qwen3-8B"
  "Qwen/Qwen2.5-Math-7B"
)
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

python fc_group/extract_activations_subset.py \
  --config fc_group/config_extract_activation.yaml \
  --model-name "$MODEL_NAME"
