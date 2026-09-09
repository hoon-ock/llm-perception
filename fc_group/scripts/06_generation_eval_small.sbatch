#!/bin/bash
#SBATCH --job-name=generation_eval_small
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
#SBATCH --time=03:00:00
#SBATCH --mem=32G
# stderr folded into stdout so the "=== model=... ===" marker stays attached to
# any traceback below it -- same reason run_functional_group_probe.sbatch does it.
#SBATCH --output=slurm-geneval-small-%A_%a.out

set -o pipefail

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

export HF_HOME=/work/ock/hoon/hf_cache

# Behavioural readout across the whole registry: score every functional-group
# name as a continuation of the declarative prompt and take the argmax, then
# also log what the model freely generates. No activations involved.
#
# Same model split and allocation as the smoke test, for the same reason:
# --gres cannot vary per array task, so the <=8B half runs full precision on
# one GPU and the >=32B half runs 4-bit across four.
#
# Run fc_group/check_model_registry.py first -- this is one of the jobs that
# would pull every model in the registry (~480 GB not yet cached).
# Model list comes from fc_group/scripts/models.sh, which derives the names
# from fc_group/model_registry.py and checks the small/large split against it.
# Four sbatch files used to carry their own copy of this list; that is what
# let the analyses quietly cover 2 models while extraction covered 10.
source fc_group/scripts/models.sh || exit 1
MODELS=("${SMALL_MODELS[@]}")

ARRAY_SIZE=3   # must equal the span of '#SBATCH --array=0-2' above
if [ "${#MODELS[@]}" -ne "$ARRAY_SIZE" ]; then
  echo "ERROR: ${#MODELS[@]} models but --array declares $ARRAY_SIZE."
  echo "       Update the --array line and ARRAY_SIZE together."
  exit 1
fi

[ -f fc_group/generation_eval.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

python -c "import yaml, torch, transformers" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$TASK_ID]}"
echo "=== task=$TASK_ID model=$MODEL ==="

if python fc_group/generation_eval.py \
     --config fc_group/config_extract_activation.yaml \
     --model-name "$MODEL" \
     --entity-types functional_group \
     --max-new-tokens 10; then
  echo "SUMMARY: ok  $MODEL"
else
  echo "SUMMARY: failed  $MODEL"
  exit 1
fi
