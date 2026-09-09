#!/bin/bash
# Shared model lists for every job in this directory. Source it, do not run it.
#
# Before this existed, four separate sbatch files each carried their own MODELS
# array: extraction listed all ten, while the probe, t-SNE, anisotropy and
# analogy jobs listed only Llama-8B and 70B. Adding a model meant editing four
# lists, and forgetting one produced a sweep that was quietly narrower than
# intended rather than an error.
#
# The names come from fc_group/model_registry.py, which is already the source of
# truth for hidden_dim/num_layers and already raises on an unregistered model.
# The size split does not live there because it is a scheduling decision, not a
# property of the model: it is what decides gpu:1 at full precision versus gpu:2
# in 4-bit. So it is declared here and then checked against the registry, which
# is the part that matters -- adding a model to the registry without giving it a
# size class fails loudly instead of dropping it from every sweep.

ALL_MODELS=()
while IFS= read -r line; do
  [ -n "$line" ] && ALL_MODELS+=("$line")
done < <(python -c "
import sys
sys.path.insert(0, 'fc_group')
from model_registry import MODEL_CONFIGS
print('\n'.join(MODEL_CONFIGS))
") || { echo "ERROR: could not read models from fc_group/model_registry.py"; exit 1; }

# <=8B: fits on one GPU at full precision.
SMALL_MODELS=(
  "meta-llama/Llama-3.1-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "phenixace/Chem-R-Faithful"
)

# >=32B: needs 4-bit sharded across the four allocated GPUs.
LARGE_MODELS=(
  "meta-llama/Llama-3.1-70B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
)

# The consistency check this file exists for.
_registry_sorted=$(printf '%s\n' "${ALL_MODELS[@]}" | sort)
_split_sorted=$(printf '%s\n' "${SMALL_MODELS[@]}" "${LARGE_MODELS[@]}" | sort)
if [ "$_registry_sorted" != "$_split_sorted" ]; then
  echo "ERROR: SMALL_MODELS + LARGE_MODELS does not match fc_group/model_registry.py"
  echo "  only in registry: $(comm -23 <(echo "$_registry_sorted") <(echo "$_split_sorted") | tr '\n' ' ')"
  echo "  only in split:    $(comm -13 <(echo "$_registry_sorted") <(echo "$_split_sorted") | tr '\n' ' ')"
  echo "  Add the model to SMALL_MODELS or LARGE_MODELS in fc_group/scripts/models.sh."
  exit 1
fi
unset _registry_sorted _split_sorted

# Entity types come from the extraction config for the same reason -- one source
# of truth, so this cannot drift from what was actually extracted.
read_entity_types() {
  python -c "
import yaml
with open('fc_group/config_extract_activation.yaml') as f:
    cfg = yaml.safe_load(f)
for e in cfg['extraction']['entities']:
    print(e['entity_type'])
"
}
