#!/bin/bash
#SBATCH --job-name=chem_retrieval
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
# 8G, not the 32G that 04/05 request: this job reads a ~2 MB npz per layer and draws
# matplotlib figures. It never loads a model or an activation tensor.
#SBATCH --mem=8G
# 2 models x 1 entity type: the declarative 'functional_group' arm only.
# Both lists live in fc_group/scripts/chem/chem_models.sh.
#SBATCH --array=0-1
# stderr folded into stdout on purpose: the "=== task=... ===" marker is on stdout, so
# splitting the streams makes tracebacks unattributable.
#SBATCH --output=fc_group/logs/slurm-chem-retrieval-%A_%a.out

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

# ---- Preflight: fail here rather than twice below ----

# Every path in this script is relative to the repo root.
[ -f fc_group/analogy_retrieval.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

python -c "import yaml, torch, scipy, seaborn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

# This job reads what 05_analogy wrote, not the activations themselves. If that tree is
# absent the whole array is pointless, so say so once here.
# Passed explicitly below rather than left to the script's default, so what this job
# consumes sits next to the check that it exists.
ANALOGY_DIR=fc_group/Results/functional_group_analogy
[ -d "$ANALOGY_DIR" ] || {
  echo "ERROR: $ANALOGY_DIR/ missing -- run 05_analogy first; this job reads its"
  echo "       diff_vectors_layer_*.npz files."
  exit 1; }

# One shared model list and one shared entity list, so this job cannot drift from the
# others or from what was actually extracted.
source fc_group/scripts/chem/chem_models.sh || exit 1

ENTITIES=("${CHEM_ENTITIES[@]}")

# --array is static but the lists it indexes are derived at runtime, so this is the one
# thing that cannot self-update. Catch the mismatch here.
ARRAY_SIZE=2   # must equal the span of '#SBATCH --array=0-1' above
N_EXPECTED=$(( ${#CHEM_MODELS[@]} * ${#ENTITIES[@]} ))
if [ "$N_EXPECTED" -ne "$ARRAY_SIZE" ]; then
  echo "ERROR: ${#CHEM_MODELS[@]} models x ${#ENTITIES[@]} entity types = $N_EXPECTED runs,"
  echo "       but '#SBATCH --array' declares $ARRAY_SIZE."
  echo "       Update the --array line and ARRAY_SIZE together."
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${CHEM_MODELS[$(( TASK_ID / ${#ENTITIES[@]} ))]}"
ENTITY="${ENTITIES[$(( TASK_ID % ${#ENTITIES[@]} ))]}"
echo "=== task=$TASK_ID model=$MODEL entity_type=$ENTITY ==="

# All three quadruple sets in one task: cross-family (inter), within-family down the
# halogen column (halide), and within-group up the chain length (carbon). The job reads
# a ~2 MB npz and draws figures, so the extra runs are nearly free -- and running them
# together is what guarantees all three always come from the same diff vectors.
# Each writes to its own tree; see --quadruple-set in analogy_retrieval.py.
STATUS=0
for QSET in inter halide carbon; do
  RUN_LOG=$(mktemp)
  # The script takes the slash form models.sh provides and normalises it itself.
  if python fc_group/analogy_retrieval.py \
       --analogy-dir "$ANALOGY_DIR" \
       --model-name "$MODEL" \
       --entity-type "$ENTITY" \
       --quadruple-set "$QSET" 2>&1 | tee "$RUN_LOG"; then
    # analogy_retrieval.py self-checks before it writes, so reaching here means the
    # trials, the summaries and each other all agreed. No postflight step needed.
    echo "SUMMARY: ok  $MODEL | $ENTITY | $QSET"
  else
    # Three different non-zero exits, and telling them apart matters across the track.
    # A model whose activations 05_analogy has not reached yet has no diff-vector npz to
    # read -- expected for part of the sweep, not a crash. A failed self-check means the
    # analysis ran but its output does not hang together, which is a bug in the code, not
    # in the environment. Anything else is an ordinary traceback.
    if grep -qE "no diff_vectors_layer_|nothing to report" "$RUN_LOG"; then
      echo "SUMMARY: no retrieval input  $MODEL | $ENTITY | $QSET"
    elif grep -q "FAILED SELF-CHECK" "$RUN_LOG"; then
      echo "SUMMARY: bad-output  $MODEL | $ENTITY | $QSET"
    else
      echo "SUMMARY: crashed  $MODEL | $ENTITY | $QSET"
    fi
    STATUS=1
  fi
  rm -f "$RUN_LOG"
done
# One non-zero exit if either set failed, so a partially-failed task is never reported
# as a success by sacct.
exit $STATUS
