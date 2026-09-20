#!/bin/bash
#SBATCH --job-name=chem_extract_smoke
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
# stderr folded into stdout so the "=== model=... ===" marker stays attached to
# any traceback below it -- same reason 02_probe.sbatch does it.
#SBATCH --output=fc_group/logs/slurm-chem-smoke-%A_%a.out

set -o pipefail

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

export HF_HOME=/work/ock/hoon/hf_cache

# Smoke test, not an extraction. One template and three layers over the 92
# molecules is enough to exercise every step that can fail per model -- auth,
# download, tokenizer, model load, forward pass, hidden-state indexing and the
# file write -- while finishing in minutes instead of hours.
#
# Both models here are 8B and run at full precision on one GPU, the same as the
# <=8B half of the shared sweep.
# Model list comes from fc_group/scripts/chem/chem_models.sh, which checks each
# name against fc_group/model_registry.py. This track is deliberately separate
# from the five-model sweep in ../: those jobs size their '#SBATCH --array' spans
# to their own lists, so widening the shared sweep would mean editing every one
# of them and re-running work that is already collected.
source fc_group/scripts/chem/chem_models.sh || exit 1
MODELS=("${CHEM_MODELS[@]}")

ARRAY_SIZE=2   # must equal the span of '#SBATCH --array=0-1' above
if [ "${#MODELS[@]}" -ne "$ARRAY_SIZE" ]; then
  echo "ERROR: ${#MODELS[@]} models but --array declares $ARRAY_SIZE."
  echo "       Update the --array line and ARRAY_SIZE together."
  exit 1
fi

[ -f fc_group/extract_activations_subset.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

python -c "import yaml, torch, transformers" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$TASK_ID]}"
echo "=== task=$TASK_ID model=$MODEL ==="

if python fc_group/extract_activations_subset.py \
     --config fc_group/config_extract_activation.yaml \
     --model-name "$MODEL" \
     --entity-types functional_group \
     --max-templates 1 \
     --layers 0 middle top; then
  # Loading and writing without error is necessary but not sufficient: the file
  # also has to have the width the registry claims, since every downstream
  # analysis resolves depth and dimension through get_model_config.
  python fc_group/verify_smoke_output.py --model-name "$MODEL" \
      && echo "SUMMARY: ok  $MODEL" \
      || { echo "SUMMARY: bad-output  $MODEL"; exit 1; }
else
  echo "SUMMARY: failed  $MODEL"
  exit 1
fi
