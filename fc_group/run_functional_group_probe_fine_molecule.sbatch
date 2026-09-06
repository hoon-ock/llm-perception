#!/bin/bash
#SBATCH --job-name=fc_probe_fine_mol
#SBATCH --partition=ock
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
# Measured peak RSS of a real run at 70B scale (920 x 8192) is 910 MB. 4G is
# ~4x headroom. The design matrix here is identical in size to the group-split
# sweep's -- only the fold structure differs -- so the same figure applies.
#SBATCH --mem=4G
# 2 models x 23 entity types. Append '%8' (--array=0-45%8) to cap how many
# run at once if the partition is busy; there is no reason to throttle otherwise,
# since each task is a single CPU core.
#SBATCH --array=0-45
# stderr is folded into stdout on purpose: the "=== model=... ===" marker is on
# stdout, so splitting the streams makes tracebacks unattributable.
#
# The 'fine-mol-' prefix is not cosmetic. This sweep and run_functional_group_probe.sbatch
# are both submitted from the repo root, so a shared 'slurm-%A_%a.out' pattern would
# interleave two sweeps in one directory listing with no way to tell them apart.
#SBATCH --output=slurm-fine-mol-%A_%a.out

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
# extraction config -- the same grid as run_functional_group_probe.sbatch.
#
# What differs is the experiment. That script runs `--target all` at the default
# `group` split, which expands to exactly two jobs: coarse/group and
# fine_to_coarse/group. fine/group is skipped by the probe itself, because
# holding a class out of a 20-way classifier removes its output unit and
# accuracy would be 0 by construction. So every number it reports is scored in
# the 5-family space, and the sweep produces no true 20-class result.
#
# This script is that missing result. Under the molecule split
# (StratifiedGroupKFold over molecules) all 20 fine classes appear on both sides
# of every fold, so all 20 output units exist and the held-out molecules are
# scorable in fine space. "Which of the 20 functional groups is this", not
# "which of the 5 heteroatom families".
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
#
# --surface-baseline is ON here, unlike in run_functional_group_probe.sbatch, and
# that is the single most important line in this file.
#
# The first time fine/molecule was run (8B, entity_type=functional_group) it
# scored 1.000 -- against a char n-gram baseline of 0.975. TF-IDF over the raw
# prompt string solves the 20-way task without the model at all, by reading
# "-ol" / "-amine" / "-oic acid" out of the name. A 1.000 quoted without that
# control next to it is a non-result, and this is the exact regime where that
# already happened once.
#
# It is not expected to be saturated everywhere: `functional_group` is the worst
# case because its prompt names the group outright, while the 21 name+formula
# prompts span 0.4934-0.5724 on the same baseline. Sweeping all 23 with the
# control attached is what separates recalled chemistry from orthography, per
# entity type. It costs ~0.23s per task (~0.2% of runtime).
#
# Read the output as (best_balanced_acc - surface_baseline_balanced_acc), NOT as
# best_balanced_acc against the 0.05 chance line. Both are in summary_fine_molecule.json.
#
# --num-null-samples 0 is deliberate, not corner-cutting. The permutation null is
# the dominant cost and does not depend on the activations: shuffled labels leave
# no signal regardless of what X holds, so the null is a property of the fold
# structure, class distribution, classifier and metric -- all identical across
# every task here. Measured across layers 0/16/31 it is flat and at chance
# (fine 0.037-0.047 against 0.05), so running it in all 46 tasks estimates one
# constant 46 times. It is still needed once, as the leak detector: if the
# molecule-level shuffle ever broke, the null would rise above chance. Run it
# separately per model -- see "Reference nulls" in fc_group/README.md.
#
# --n-splits is left at its default of 4, which is deliberate: 17 of the 20 fine
# classes have exactly 4 molecules, so 5 folds would guarantee folds containing
# zero instances of some classes. See iter_folds() in functional_group_probe.py.
RUN_LOG=$(mktemp)
if python fc_group/functional_group_probe.py \
     --model-name "$MODEL" \
     --entity-type "$ENTITY" \
     --target fine \
     --split molecule \
     --surface-baseline \
     --num-null-samples 0 2>&1 | tee "$RUN_LOG"; then
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
