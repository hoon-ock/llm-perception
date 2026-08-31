#!/bin/bash
#SBATCH --job-name=tsne_functional_groups
#SBATCH --partition=ock
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR"

# The login/compute node caps per-user threads/processes (RLIMIT_NPROC);
# numpy's OpenBLAS backend defaults to spawning one thread per core, which
# blows past that cap and segfaults. Force single-threaded BLAS/OpenMP.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# CPU-only analysis (no GPU model loading) -- no --gres requested, so this
# doesn't compete with GPU extraction jobs on the same node.
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
")

for model in "${MODELS[@]}"; do
  while IFS= read -r entity; do
    echo "=== model=$model entity_type=$entity ==="
    python fc_group/tsne_functional_groups.py \
      --entity-type "$entity" \
      --model-name "$model"
  done <<< "$ENTITY_TYPES"
done
