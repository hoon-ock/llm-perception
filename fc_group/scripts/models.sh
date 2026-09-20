#!/bin/bash
# Shared model lists for every job in this directory. Source it, do not run it.
#
# Before this existed, four separate sbatch files each carried their own MODELS
# array: extraction listed all ten, while the probe, t-SNE, anisotropy and
# analogy jobs listed only Llama-8B and 70B. Adding a model meant editing four
# lists, and forgetting one produced a sweep that was quietly narrower than
# intended rather than an error.
#
# The sweep is declared here; fc_group/model_registry.py is the catalogue it
# draws from. That direction matters, and it used to be the other way round:
# ALL_MODELS was generated from MODEL_CONFIGS and asserted equal to
# SMALL+LARGE, which made "registered" and "swept" the same thing. It no longer
# is -- a model can be registered so `get_model_config` resolves it (the probe,
# t-SNE, anisotropy and ambiguity scripts all raise KeyError otherwise) while
# being run by its own job outside this sweep, as fc_group/scripts/chem/ does.
# So the check below is a subset check rather than an equality one. The part
# that was ever load-bearing -- a typo'd or unregistered name failing loudly
# instead of silently narrowing the sweep -- is unchanged.
#
# The size split lives here and not in the registry because it is a scheduling
# decision, not a property of the model: it is what decides gpu:1 at full
# precision versus gpu:2 in 4-bit.

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

ALL_MODELS=("${SMALL_MODELS[@]}" "${LARGE_MODELS[@]}")

_REGISTRY_MODELS=()
while IFS= read -r line; do
  [ -n "$line" ] && _REGISTRY_MODELS+=("$line")
done < <(python -c "
import sys
sys.path.insert(0, 'fc_group')
from model_registry import MODEL_CONFIGS
print('\n'.join(MODEL_CONFIGS))
") || { echo "ERROR: could not read models from fc_group/model_registry.py"; exit 1; }

_registry_sorted=$(printf '%s\n' "${_REGISTRY_MODELS[@]}" | sort)
_sweep_sorted=$(printf '%s\n' "${ALL_MODELS[@]}" | sort)

# Every swept model must be registered, or its depth lookup fails mid-job.
_unregistered=$(comm -13 <(echo "$_registry_sorted") <(echo "$_sweep_sorted"))
if [ -n "$_unregistered" ]; then
  echo "ERROR: swept but not in fc_group/model_registry.py: $(echo "$_unregistered" | tr '\n' ' ')"
  echo "  Add it to MODEL_CONFIGS, or fix the spelling in SMALL_MODELS/LARGE_MODELS."
  exit 1
fi

# Registered but not swept is legal now, so say so rather than staying silent: a
# model dropping out of a sweep unnoticed is the failure the old equality check
# existed to prevent, and this keeps that warning without forbidding the case.
_unswept=$(comm -23 <(echo "$_registry_sorted") <(echo "$_sweep_sorted"))
if [ -n "$_unswept" ]; then
  echo "note: registered, not in this sweep: $(echo "$_unswept" | tr '\n' ' ')"
fi
unset _registry_sorted _sweep_sorted _unregistered _unswept _REGISTRY_MODELS

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
