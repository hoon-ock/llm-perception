#!/bin/bash
#SBATCH --job-name=extract_activations
#SBATCH --partition=ock
#SBATCH --gres=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR"

python fc_group/extract_activations_subset.py \
  --config fc_group/config_extract_activation.yaml \
  --max-templates 5
