#!/bin/bash
#SBATCH --job-name=extract_smoke_large
#SBATCH --partition=ock
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --mem=200G
# stderr folded into stdout so the "=== model=... ===" marker stays attached to
# any traceback below it -- same reason run_functional_group_probe.sbatch does it.
#SBATCH --output=slurm-smoke-large-%A_%a.out

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
# The five models here are the >=32B half of the registry, loaded in 4-bit and
# sharded across the four allocated GPUs. The <=8B half runs at full precision on
# a single GPU in run_smoke_test.sbatch; --gres cannot vary per array task, which
# is why these are two files rather than one array of ten.
# Model list comes from fc_group/scripts/models.sh, which derives the names
# from fc_group/model_registry.py and checks the small/large split against it.
# Four sbatch files used to carry their own copy of this list; that is what
# let the analyses quietly cover 2 models while extraction covered 10.
source fc_group/scripts/models.sh || exit 1
MODELS=("${LARGE_MODELS[@]}")

ARRAY_SIZE=5   # must equal the span of '#SBATCH --array=0-4' above
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
     --layers 0 middle top \
     --load-in-4bit; then
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
