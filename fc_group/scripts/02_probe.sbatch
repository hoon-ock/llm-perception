#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
# Measured peak RSS at 70B width (920 x 8192) is 910 MB; 4G is ~4x headroom.
#SBATCH --mem=4G
# 5 models x 24 entity types. Append '%N' (--array=0-119%20) to cap concurrency
# if the partition is busy; each task is a single CPU core.
#SBATCH --array=0-119
#SBATCH --output=slurm-probe-%A_%a.out

set -o pipefail

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

# The compute node caps per-user threads (RLIMIT_NPROC) and numpy's OpenBLAS
# backend blows past it. Load-bearing, not decoration.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Configuration, overridable at submit time ----
#
# This replaces the old run_functional_group_probe_fine_molecule.sbatch, which
# was a near-duplicate of this file differing only in these two settings:
#
#   sbatch --export=ALL,PROBE_TARGET=fine,PROBE_SPLIT=molecule fc_group/scripts/02_probe.sbatch
#
# An empty PROBE_SPLIT leaves the probe on its own default (leave-one-group-out),
# which is the only split with dynamic range.
PROBE_TARGET="${PROBE_TARGET:-all}"
PROBE_SPLIT="${PROBE_SPLIT:-}"

# ---- Preflight ----

[ -f fc_group/functional_group_probe.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

python -c "import yaml, torch, scipy, seaborn, sklearn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

source fc_group/scripts/models.sh || exit 1

ENTITIES=()
while IFS= read -r entity; do
  [ -n "$entity" ] && ENTITIES+=("$entity")
done < <(read_entity_types)
[ ${#ENTITIES[@]} -gt 0 ] || { echo "ERROR: no entity types parsed from config"; exit 1; }

ARRAY_SIZE=120   # must equal the span of '#SBATCH --array=0-119' above
N_EXPECTED=$(( ${#ALL_MODELS[@]} * ${#ENTITIES[@]} ))
if [ "$N_EXPECTED" -ne "$ARRAY_SIZE" ]; then
  echo "ERROR: ${#ALL_MODELS[@]} models x ${#ENTITIES[@]} entity types = $N_EXPECTED runs,"
  echo "       but '#SBATCH --array' declares $ARRAY_SIZE."
  echo "       Update the --array line and ARRAY_SIZE together."
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${ALL_MODELS[$(( TASK_ID / ${#ENTITIES[@]} ))]}"
ENTITY="${ENTITIES[$(( TASK_ID % ${#ENTITIES[@]} ))]}"

echo "=== task=$TASK_ID model=$MODEL entity_type=$ENTITY target=$PROBE_TARGET split=${PROBE_SPLIT:-<default>} ==="

# --num-null-samples 0 is deliberate. The permutation null is ~91% of this job's
# runtime and does not depend on the activations: shuffled labels leave no signal
# regardless of what X holds, so it is a property of the fold structure, class
# distribution, classifier and metric -- identical across every task here.
# Measured, it sits flat at chance. Run it separately once per model and quote
# that as the reference band; see "Reference nulls" in fc_group/README.md.
#
# --surface-baseline, by contrast, is now ON. It costs ~0.23 s per task and it is
# the control that separates recalled chemistry from reading "-ol" out of the
# prompt -- character n-grams alone score 0.975-1.00 on four of the five splits.
# It was off in the previous sweep, which is why the existing results have no
# orthography reference for the 70B half at all.
SPLIT_ARGS=()
[ -n "$PROBE_SPLIT" ] && SPLIT_ARGS+=(--split "$PROBE_SPLIT")

RUN_LOG=$(mktemp)
if python fc_group/functional_group_probe.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     --target "$PROBE_TARGET" \
     "${SPLIT_ARGS[@]}" \
     --surface-baseline \
     --num-null-samples 0 2>&1 | tee "$RUN_LOG"; then
  if grep -q "No layer files found" "$RUN_LOG"; then
    echo "SUMMARY: no activation data  $MODEL | $ENTITY"
    rm -f "$RUN_LOG"; exit 1
  fi
  echo "SUMMARY: ok  $MODEL | $ENTITY"
  rm -f "$RUN_LOG"
else
  echo "SUMMARY: crashed  $MODEL | $ENTITY"
  rm -f "$RUN_LOG"; exit 1
fi
