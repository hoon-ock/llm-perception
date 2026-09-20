#!/bin/bash
#SBATCH --job-name=chem_extract_activations
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
# 1 core was the default. This path is not quantized, so the CPU cost is
# safetensors deserialization before .to(device); 8 cores x 2 array tasks is 16
# of the node's 32, leaving headroom.
#SBATCH --cpus-per-task=8
#SBATCH --output=fc_group/logs/slurm-chem-extract-%A_%a.out

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

export HF_HOME=/work/ock/hoon/hf_cache

# ---- Preflight: fail here rather than inside the model load ----

# Every path in this script is relative to the repo root.
[ -f fc_group/extract_activations_subset.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

python -c "import yaml, torch, transformers" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

# 7-8B models: fit comfortably on one GPU, run full precision (no quantization).
# Model list comes from fc_group/scripts/chem/chem_models.sh, which checks each
# name against fc_group/model_registry.py. This track is deliberately separate
# from the five-model sweep in ../: those jobs size their '#SBATCH --array' spans
# to their own lists, so widening the shared sweep would mean editing every one
# of them and re-running work that is already collected.
source fc_group/scripts/chem/chem_models.sh || exit 1
MODELS=("${CHEM_MODELS[@]}")

# --array is static but MODELS is derived at runtime, so this is the one thing
# that cannot self-update. Catch the mismatch here rather than silently running
# a narrower sweep than intended.
ARRAY_SIZE=2   # must equal the span of '#SBATCH --array=0-1' above
if [ "${#MODELS[@]}" -ne "$ARRAY_SIZE" ]; then
  echo "ERROR: ${#MODELS[@]} models but --array declares $ARRAY_SIZE."
  echo "       Update the --array line and ARRAY_SIZE together."
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_NAME="${MODELS[$TASK_ID]}"
echo "=== task=$TASK_ID model=$MODEL_NAME ==="

# SUMMARY line so an extraction sweep is triageable with the same grep as every
# other stage, rather than being the one job whose outcome has to be read from sacct.
if python fc_group/extract_activations_subset.py \
  --config fc_group/config_extract_activation.yaml \
  --model-name "$MODEL_NAME" \
  --entity-types functional_group; then
  echo "SUMMARY: ok  $MODEL_NAME"
else
  echo "SUMMARY: failed  $MODEL_NAME"
  exit 1
fi
