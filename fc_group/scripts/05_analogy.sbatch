#!/bin/bash
#SBATCH --job-name=analogy
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
# 10 models x 24 entity types. Append '%N' (--array=0-239%20) to cap concurrency
# if the partition is busy; each task is a single CPU core, so there is otherwise
# no reason to throttle.
#SBATCH --array=0-239
# stderr folded into stdout on purpose: the "=== task=... ===" marker is on
# stdout, so splitting the streams makes tracebacks unattributable.
#SBATCH --output=slurm-analogy-%A_%a.out

# Needed so `python ... | tee` reports python's status rather than tee's.
set -o pipefail

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

# The login/compute node caps per-user threads (RLIMIT_NPROC); numpy's OpenBLAS
# backend defaults to one thread per core and blows past it, segfaulting. These
# are load-bearing, not decoration.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Preflight: fail here rather than 240 identical failures below ----

# Every path in this script is relative to the repo root.
[ -f fc_group/functional_group_analogy_carbon_matched.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

python -c "import yaml, torch, scipy, seaborn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

# One shared model list and one shared entity list, so this job cannot drift
# from the others or from what was actually extracted.
source fc_group/scripts/models.sh || exit 1

ENTITIES=()
while IFS= read -r entity; do
  [ -n "$entity" ] && ENTITIES+=("$entity")
done < <(read_entity_types)
[ ${#ENTITIES[@]} -gt 0 ] || { echo "ERROR: no entity types parsed from config"; exit 1; }

# --array is static but the lists it indexes are derived at runtime, so this is
# the one thing that cannot self-update. Catch the mismatch here.
ARRAY_SIZE=240   # must equal the span of '#SBATCH --array=0-239' above
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

echo "=== task=$TASK_ID model=$MODEL entity_type=$ENTITY ==="

RUN_LOG=$(mktemp)
if python fc_group/functional_group_analogy_carbon_matched.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
      \     2>&1 | tee "$RUN_LOG"; then
  # Exit 0 still covers "the activations for this entity were never extracted",
  # which is expected for models the extraction sweep has not reached yet and a
  # real problem for the ones it has.
  if grep -qE "No layer files found|no activation data" "$RUN_LOG"; then
    echo "SUMMARY: no activation data  $MODEL | $ENTITY"
    rm -f "$RUN_LOG"; exit 1
  fi
  echo "SUMMARY: ok  $MODEL | $ENTITY"
  rm -f "$RUN_LOG"
else
  echo "SUMMARY: crashed  $MODEL | $ENTITY"
  rm -f "$RUN_LOG"; exit 1
fi
