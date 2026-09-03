#!/bin/bash
#SBATCH --job-name=anisotropy_diagnostic
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
# stderr is folded into stdout on purpose: the per-run "=== model=... ===" markers
# are on stdout, so splitting the streams makes tracebacks unattributable.
#SBATCH --output=slurm-%j.out

# Needed so `python ... | tee` reports python's status rather than tee's.
set -o pipefail

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR" || { echo "ERROR: cannot cd to $SLURM_SUBMIT_DIR"; exit 1; }

# The login/compute node caps per-user threads/processes (RLIMIT_NPROC);
# numpy's OpenBLAS backend defaults to spawning one thread per core, which
# blows past that cap and segfaults. Force single-threaded BLAS/OpenMP.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Preflight: fail loudly here rather than 46 identical failures below ----

# Every path in this script is relative to the repo root.
[ -f fc_group/anisotropy_diagnostic.py ] || {
  echo "ERROR: must be submitted from the repo root (cwd=$PWD)"; exit 1; }

# A failed `conda activate` is not fatal on its own -- catch it here, before
# `python` silently resolves to a system interpreter without the deps.
# scipy/seaborn come in transitively: anisotropy_diagnostic.py imports its
# shared helpers from functional_group_analogy_carbon_matched.py, which needs
# both. No sklearn, unlike the t-SNE job.
python -c "import yaml, torch, scipy, seaborn, matplotlib, pandas" || {
  echo "ERROR: conda env not active or deps missing (python=$(command -v python))"
  exit 1; }

# CPU-only analysis (torch is used only for torch.load of the saved tensors) --
# no --gres requested, so this doesn't compete with GPU extraction jobs.
MODELS=(
  "meta-llama/Llama-3.1-8B"
  "meta-llama/Llama-3.1-70B"
)

# Derive the full entity-type list from the config instead of hardcoding it
# here, so this stays in sync with fc_group/config_extract_activation.yaml.
ENTITY_TYPES=$(python -c "
import yaml
with open('fc_group/config_extract_activation.yaml') as f:
    cfg = yaml.safe_load(f)
for e in cfg['extraction']['entities']:
    print(e['entity_type'])
") || { echo "ERROR: failed to read entity types from config"; exit 1; }

# An empty list would otherwise still drive one loop iteration with entity="".
[ -n "$ENTITY_TYPES" ] || { echo "ERROR: no entity types parsed from config"; exit 1; }

# ---- Sweep: keep going on failure, account for it at the end ----

FAILED=()
NODATA=()
RUN_LOG=$(mktemp)

for model in "${MODELS[@]}"; do
  while IFS= read -r entity; do
    [ -z "$entity" ] && continue
    echo "=== model=$model entity_type=$entity ==="
    if python fc_group/anisotropy_diagnostic.py \
         --entity-type "$entity" \
         --model-name "$model" 2>&1 | tee "$RUN_LOG"; then
      # Exit 0 still covers "the activations for this entity were never
      # extracted", which is a real problem when they're supposed to exist.
      grep -q "No layer files found" "$RUN_LOG" && NODATA+=("$model | $entity")
    else
      FAILED+=("$model | $entity")
    fi
  done <<< "$ENTITY_TYPES"
done

rm -f "$RUN_LOG"

echo "================ SUMMARY ================"
echo "crashed:        ${#FAILED[@]}"
[ ${#FAILED[@]} -gt 0 ] && printf '  %s\n' "${FAILED[@]}"
echo "no data:        ${#NODATA[@]}"
[ ${#NODATA[@]} -gt 0 ] && printf '  %s\n' "${NODATA[@]}"
echo "runs on disk:   $(find fc_group/Results/anisotropy_diagnostic -name 'summary_all_layers.json' 2>/dev/null | wc -l | tr -d ' ')"

# Make the job's exit status mean something, so sacct reflects reality.
[ ${#FAILED[@]} -eq 0 ] && [ ${#NODATA[@]} -eq 0 ]
