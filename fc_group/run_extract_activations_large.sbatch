#!/bin/bash
#SBATCH --job-name=extract_activations_large
#SBATCH --partition=ock
#SBATCH --gres=gpu:4
#SBATCH --array=0-4
#SBATCH --time=1-00:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR"

export HF_HOME=/work/ock/hoon/hf_cache

# 32-70B-scale models: don't fit unquantized, run 4-bit via device_map="auto"
# so accelerate shards layers across the allocated GPUs.
MODELS=(
  "meta-llama/Llama-3.1-70B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
  "Qwen/Qwen3-32B"
  "Qwen/Qwen2.5-Math-72B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

python fc_group/extract_activations_subset.py \
  --config fc_group/config_extract_activation.yaml \
  --model-name "$MODEL_NAME" \
  --load-in-4bit
