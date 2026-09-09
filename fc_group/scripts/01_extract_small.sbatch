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
# Model list comes from fc_group/scripts/models.sh, which derives the names
# from fc_group/model_registry.py and checks the small/large split against it.
# Four sbatch files used to carry their own copy of this list; that is what
# let the analyses quietly cover 2 models while extraction covered 10.
source fc_group/scripts/models.sh || exit 1
MODELS=("${SMALL_MODELS[@]}")

# --array is static but MODELS is derived at runtime, so this is the one thing
# that cannot self-update. Catch the mismatch here rather than silently running
# a narrower sweep than intended.
ARRAY_SIZE=5   # must equal the span of '#SBATCH --array=0-4' above
if [ "${#MODELS[@]}" -ne "$ARRAY_SIZE" ]; then
  echo "ERROR: ${#MODELS[@]} models but --array declares $ARRAY_SIZE."
  echo "       Update the --array line and ARRAY_SIZE together."
  exit 1
fi
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

python fc_group/extract_activations_subset.py \
  --config fc_group/config_extract_activation.yaml \
  --model-name "$MODEL_NAME"
