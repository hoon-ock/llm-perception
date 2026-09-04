#!/bin/bash
#SBATCH --job-name=fc_probe
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
# 2 models x 23 entity types. Append '%8' (--array=0-45%8) to cap how many
# run at once if the partition is busy; there is no reason to throttle otherwise,
# since each task is a single CPU core.
#SBATCH --array=0-45
# stderr is folded into stdout on purpose: the "=== model=... ===" marker is on
# stdout, so splitting the streams makes tracebacks unattributable.
#SBATCH --output=slurm-%A_%a.out

# Needed so `python ... | tee` reports python's status rather than tee's.
set -o pipefail

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

# The login/compute node caps per-user threads/processes (RLIMIT_NPROC);
# numpy's OpenBLAS backend defaults to spawning one thread per core, which
# blows past that cap and segfaults. Force single-threaded BLAS/OpenMP.
#
# One core is not an oversight, and neither is the absence of --gres=gpu.
# The probe is entirely scikit-learn -- PCA, LogisticRegression, StandardScaler
# -- with torch used only for `torch.load(..., map_location="cpu")`, so a GPU
# would sit idle for the whole run while competing with the extraction jobs
# that actually need one. Extra cores do not help either: measured on a
# 920x8192 fold, one preprocess+fit takes 0.35s at 1 thread and 0.33s at 8.
# The matrices are too small for BLAS parallelism and lbfgs is serial.
# Throughput here comes from the array width, not from within a task.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Sweep definition ----
#
# One array task per (model, entity_type) pair, over every entity type in the
# extraction config.
#
# The probe's target is always the functional group, so the entity type selects
# which *prompt* the last-token state is read from -- `functional_group` asks
# for the group, `pka` asks about acidity, `molecule_name_bare` asks nothing at
# all. Sweeping all of them is the attribute-specificity matrix: is functional
# group identity present in every last-token state, or only when the prompt
# asks for it?
MODELS=(
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-70B"
)

# ---- Preflight: fail loudly here rather than 30 minutes into a run ----

# Every path in this script is relative to the repo root.
[ -f fc_group/functional_group_probe.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

# A failed `conda activate` is not fatal on its own -- catch it here, before
# `python` silently resolves to a system interpreter without the deps.
# scipy/seaborn come in transitively via functional_group_analogy_carbon_matched,
# which functional_group_probe.py imports its loaders from; sklearn is the probe
# itself; yaml is used both below and by --surface-baseline.
python -c "import yaml, torch, scipy, seaborn, sklearn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

# Derive the entity list from the config rather than hand-listing it, so the two
# cannot drift -- the same approach as run_anisotropy_diagnostic.sbatch. A
# read-loop rather than `mapfile`, which needs bash 4+ and so would not run on
# the macOS bash used for local dry-run checks. Splitting on newlines keeps
# names containing a space ("functional_group question") as single elements.
ENTITIES=()
while IFS= read -r entity; do
  [ -n "$entity" ] && ENTITIES+=("$entity")
done < <(python -c "
import yaml
with open('fc_group/config_extract_activation.yaml') as f:
    cfg = yaml.safe_load(f)
for e in cfg['extraction']['entities']:
    print(e['entity_type'])
")

# An empty list would otherwise still index into MODELS and run with entity="".
[ ${#ENTITIES[@]} -gt 0 ] || {
  echo "ERROR: no entity types parsed from fc_group/config_extract_activation.yaml"
  exit 1; }

# `--array` is a static range but the lists it indexes are derived at runtime,
# so it is the one thing that cannot self-update. Catch the mismatch here.
ARRAY_SIZE=46   # must equal the span of '#SBATCH --array=0-45' above
N_EXPECTED=$(( ${#MODELS[@]} * ${#ENTITIES[@]} ))
if [ "$N_EXPECTED" -ne "$ARRAY_SIZE" ]; then
  echo "ERROR: ${#MODELS[@]} models x ${#ENTITIES[@]} entity types = $N_EXPECTED runs,"
  echo "       but '#SBATCH --array' declares $ARRAY_SIZE."
  echo "       Update the --array line and ARRAY_SIZE together."
  exit 1
fi

echo "preflight: ${#MODELS[@]} models x ${#ENTITIES[@]} entity types = $N_EXPECTED tasks"

# ---- Resolve this task's pair ----
# Defaulted so the mapping can be exercised outside SLURM for a sanity check.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$(( TASK_ID / ${#ENTITIES[@]} ))]}"
ENTITY="${ENTITIES[$(( TASK_ID % ${#ENTITIES[@]} ))]}"

echo "=== task=$TASK_ID model=$MODEL entity_type=$ENTITY ==="

# ---- Run ----
# ~20 min for the 8B (32 layers), ~1h for the 70B (80 layers, 8192-dim).
RUN_LOG=$(mktemp)
if python fc_group/functional_group_probe.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     --target all \
     --num-null-samples 50 \
     --surface-baseline 2>&1 | tee "$RUN_LOG"; then
  # Exit 0 still covers "the activations for this entity were never extracted",
  # which is a real problem when they are supposed to exist.
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
