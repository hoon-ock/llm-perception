#!/bin/bash
#SBATCH --job-name=chem_ambiguity
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
# Measured: the default path is ~2 s per task and --full ~1 s per layer on top, so 80
# layers at 70B depth is a couple of minutes plus the activation reads --full does. An
# hour is 20-40x headroom and still nothing on a CPU partition.
#SBATCH --time=01:00:00
# 8G, matching 07: this job reads a ~1 MB npz, and under AMBIGUITY_FULL one activation
# file at a time (~15 MB at 70B width). It never loads a model.
#SBATCH --mem=8G
# 2 models x 1 entity type: the declarative 'functional_group' arm only.
# Both lists live in fc_group/scripts/chem/chem_models.sh.
#SBATCH --array=0-1
# stderr folded into stdout on purpose: the "=== task=... ===" marker is on stdout, so
# splitting the streams makes tracebacks unattributable.
#SBATCH --output=fc_group/logs/slurm-chem-ambiguity-%A_%a.out

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

# ---- Configuration, overridable at submit time ----
#
# Empty (the default) runs the plain table: mean probability on each family per held-out
# group, which is the number worth quoting and costs seconds. Set to 1 for the derived
# analysis on top -- temperature-calibrated restricted posteriors, the ambiguous-vs-control
# contrast with bootstrap CIs and a permutation test, and the centroid-axis projection:
#
#   sbatch --export=ALL,AMBIGUITY_FULL=1 fc_group/scripts/08_ambiguity.sbatch
#
# That is what answers "could this just be how hard select_C regularized each model", and
# it is the half that reads activations. Same pattern as 02_probe's PROBE_TARGET.
AMBIGUITY_FULL="${AMBIGUITY_FULL:-}"

# ---- Preflight: fail here rather than twice below ----

# Every path in this script is relative to the repo root.
[ -f fc_group/ambiguity_metric.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

# sklearn and seaborn come in via `from functional_group_probe import ...`; torch via the
# shared activation loaders that --full reaches. Same list as 02_probe for that reason.
python -c "import yaml, torch, scipy, seaborn, sklearn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

# This job reads what 02_probe wrote, not the activations themselves (except under
# AMBIGUITY_FULL). If that tree is absent the whole array is pointless, so say so once.
# The script resolves the per-model path itself from its own DEFAULT_OUTPUT_DIR, so only
# the root is checked here -- passing --probe-dir would be a second place to get it wrong.
PROBE_DIR=fc_group/Results/functional_group_probe
[ -d "$PROBE_DIR" ] || {
  echo "ERROR: $PROBE_DIR/ missing -- run 02_probe first; this job reads the"
  echo "       oof_proba_*.npz files it writes."
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
echo "=== task=$TASK_ID model=$MODEL entity_type=$ENTITY full=${AMBIGUITY_FULL:-<no>} ==="

FULL_ARGS=()
[ -n "$AMBIGUITY_FULL" ] && FULL_ARGS+=(--full)

RUN_LOG=$(mktemp)
# The script takes the slash form models.sh provides and normalises it itself.
if python fc_group/ambiguity_metric.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     "${FULL_ARGS[@]}" 2>&1 | tee "$RUN_LOG"; then
  # ambiguity_metric.py self-checks before it writes, so reaching here means the saved
  # probabilities agreed with the predictions the probe wrote alongside them.
  echo "SUMMARY: ok  $MODEL | $ENTITY"
  rm -f "$RUN_LOG"
else
  # Three different non-zero exits, and telling them apart matters across the track.
  # A model/entity_type whose probe results predate oof_proba_*.npz has nothing to read
  # -- expected for part of the sweep until 02_probe has been rerun, not a crash. A
  # failed self-check means the analysis ran but its input does not hang together, which
  # is a bug rather than a missing file. Anything else is an ordinary traceback.
  if grep -q "Rerun the probe" "$RUN_LOG"; then
    echo "SUMMARY: no probe output  $MODEL | $ENTITY"
  elif grep -q "FAILED SELF-CHECK" "$RUN_LOG"; then
    echo "SUMMARY: bad-output  $MODEL | $ENTITY"
  else
    echo "SUMMARY: crashed  $MODEL | $ENTITY"
  fi
  rm -f "$RUN_LOG"
  exit 1
fi
