#!/bin/bash
#SBATCH --job-name=extract_activations_large
#SBATCH --partition=ock
# Two GPUs, not four. A 70B under NF4 with double-quant off needs ~44G:
# ~35G packed 4-bit weights, ~4.4G of fp32 absmax scales (blocksize 64), and
# ~4.2G of embed_tokens + lm_head, which bitsandbytes leaves in fp16 (128256 x
# 8192, untied at this scale). That does not safely fit one 48G L40S once
# activations land on it, but sits comfortably on two. It matters because ock is
# a single node with 4 GPUs: at gpu:4 each array task owned the whole partition
# and the two 70B models ran strictly back to back. At gpu:2 they run at once.
#SBATCH --gres=gpu:2
#SBATCH --array=0-1
#SBATCH --time=1-00:00:00
# Host RAM, not VRAM, and the load path barely touches it: device_map="auto"
# reads safetensors shard by shard and quantizes each onto the GPU, so the fp16
# checkpoint is never resident on the host and peak RSS is roughly one shard.
# This was 200G on a node that has 491G, so it was never actually blocking --
# just an order of magnitude more than the job can use. Confirm with
# `sacct -j <id> --format=ReqMem,MaxRSS`; raise this if MaxRSS approaches 64G.
#SBATCH --mem=64G
# 1 core was the default and the 70B load is CPU-bound on safetensors
# deserialization plus bitsandbytes quantization. Deliberately 8 rather than the
# 8-per-GPU ratio (16): at 16 the two concurrent tasks would need exactly all 32
# of the node's cores, so anything else running pushes one to pending.
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

module load anaconda
conda activate llm-perception

cd "$SLURM_SUBMIT_DIR"

export HF_HOME=/work/ock/hoon/hf_cache

# 32-70B-scale models: don't fit unquantized, run 4-bit via device_map="auto"
# so accelerate shards layers across the allocated GPUs.
# Model list comes from fc_group/scripts/models.sh, which derives the names
# from fc_group/model_registry.py and checks the small/large split against it.
# Four sbatch files used to carry their own copy of this list; that is what
# let the analyses quietly cover 2 models while extraction covered 10.
source fc_group/scripts/models.sh || exit 1
MODELS=("${LARGE_MODELS[@]}")

# --array is static but MODELS is derived at runtime, so this is the one thing
# that cannot self-update. Catch the mismatch here rather than silently running
# a narrower sweep than intended.
ARRAY_SIZE=2   # must equal the span of '#SBATCH --array=0-1' above
if [ "${#MODELS[@]}" -ne "$ARRAY_SIZE" ]; then
  echo "ERROR: ${#MODELS[@]} models but --array declares $ARRAY_SIZE."
  echo "       Update the --array line and ARRAY_SIZE together."
  exit 1
fi
MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

python fc_group/extract_activations_subset.py \
  --config fc_group/config_extract_activation.yaml \
  --model-name "$MODEL_NAME" \
  --load-in-4bit
