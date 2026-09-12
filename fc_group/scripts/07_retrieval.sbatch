#!/bin/bash
#SBATCH --job-name=retrieval
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
# 8G, not the 32G that 04/05 request: this job reads a ~2 MB npz per layer and draws
# matplotlib figures. It never loads a model or an activation tensor.
#SBATCH --mem=8G
# 5 models x 24 entity types. Append '%N' (--array=0-119%20) to cap concurrency if the
# partition is busy; each task is a single CPU core, so there is otherwise no reason to
# throttle.
#SBATCH --array=0-119
# stderr folded into stdout on purpose: the "=== task=... ===" marker is on stdout, so
# splitting the streams makes tracebacks unattributable.
#SBATCH --output=slurm-retrieval-%A_%a.out

# Needed so `python ... | tee` reports python's status rather than tee's.
set -o pipefail

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

# The login/compute node caps per-user threads (RLIMIT_NPROC); numpy's OpenBLAS backend
# defaults to one thread per core and blows past it, segfaulting. These are load-bearing,
# not decoration.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Preflight: fail here rather than 120 identical failures below ----

# Every path in this script is relative to the repo root.
[ -f fc_group/analogy_retrieval.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

python -c "import yaml, torch, scipy, seaborn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

# This job reads what 05_analogy wrote, not the activations themselves. If that tree is
# absent the whole array is pointless, so say so once here.
[ -d fc_group/Results_HCC/functional_group_analogy ] || {
  echo "ERROR: fc_group/Results_HCC/functional_group_analogy/ missing -- run 05_analogy"
  echo "       first; this job reads its diff_vectors_layer_*.npz files."
  exit 1; }

# One shared model list and one shared entity list, so this job cannot drift from the
# others or from what was actually extracted.
source fc_group/scripts/models.sh || exit 1

ENTITIES=()
while IFS= read -r entity; do
  [ -n "$entity" ] && ENTITIES+=("$entity")
done < <(read_entity_types)
[ ${#ENTITIES[@]} -gt 0 ] || { echo "ERROR: no entity types parsed from config"; exit 1; }

# --array is static but the lists it indexes are derived at runtime, so this is the one
# thing that cannot self-update. Catch the mismatch here.
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
echo "=== task=$TASK_ID model=$MODEL entity_type=$ENTITY ==="

RUN_LOG=$(mktemp)
# Both scripts take the slash form models.sh provides and normalise it themselves.
if python fc_group/analogy_retrieval.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" 2>&1 | tee "$RUN_LOG"; then
  # analogy_retrieval.py self-checks before it writes, so reaching here means the
  # trials, the summaries and each other all agreed. No postflight step needed.
  echo "SUMMARY: ok  $MODEL | $ENTITY"
  rm -f "$RUN_LOG"
else
  # Three different non-zero exits, and telling them apart matters across 120 tasks.
  # A model whose activations 05_analogy has not reached yet has no diff-vector npz to
  # read -- expected for part of the sweep, not a crash. A failed self-check means the
  # analysis ran but its output does not hang together, which is a bug in the code, not
  # in the environment. Anything else is an ordinary traceback.
  if grep -qE "no diff_vectors_layer_|nothing to report" "$RUN_LOG"; then
    echo "SUMMARY: no retrieval input  $MODEL | $ENTITY"
  elif grep -q "FAILED SELF-CHECK" "$RUN_LOG"; then
    echo "SUMMARY: bad-output  $MODEL | $ENTITY"
  else
    echo "SUMMARY: crashed  $MODEL | $ENTITY"
  fi
  rm -f "$RUN_LOG"; exit 1
fi
