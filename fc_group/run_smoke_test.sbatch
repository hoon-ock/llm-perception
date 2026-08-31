#!/bin/bash
#SBATCH --job-name=extract_activations_smoke
#SBATCH --partition=ock
#SBATCH --gres=gpu:4
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --mem=200G
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR"

export HF_HOME=/work/ock/hoon/hf_cache

# Smoke test: one small model (full precision, single GPU) and one large
# model (4-bit, sharded across the allocated GPUs) to validate both code
# paths end-to-end before running the full model lists.
MODELS=(
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-70B"
)
LOAD_4BIT=(0 1)

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"
EXTRA_ARGS=()
if [ "${LOAD_4BIT[$SLURM_ARRAY_TASK_ID]}" = "1" ]; then
  EXTRA_ARGS+=(--load-in-4bit)
fi

python fc_group/extract_activations_subset.py \
  --config fc_group/config_extract_activation.yaml \
  --model-name "$MODEL_NAME" \
  "${EXTRA_ARGS[@]}"
