#!/bin/bash
# Model and entity lists for the chemistry-comparison track. Source it, do not run it.
#
# This track exists so two newly added chemistry models can be run without
# touching the five-model sweep in fc_group/scripts/. Those jobs carry static
# '#SBATCH --array' spans sized to their own lists, so widening the shared sweep
# means editing every one of them; keeping this separate means the established
# sweep and its already-collected results stay exactly as they are.
#
# Scope is deliberately narrow: two models, one entity type. The declarative
# 'functional_group' arm only -- not 'functional_group question', not the SMILES
# or IUPAC arms, not the RAG arms.
#
# Registration is still required and still lives in fc_group/model_registry.py:
# functional_group_probe.py, tsne_functional_groups.py, anisotropy_diagnostic.py
# and ambiguity_metric.py all resolve depth through get_model_config, which
# raises KeyError on an unregistered name. What this file adds is the ability to
# be registered *without* being swept -- see the note in ../models.sh.

CHEM_MODELS=(
  "weidawang/Chem-R-8B"
  "OpenDFM/ChemDFM-v1.5-8B"
)

# Declarative only. Kept as an array so the matrix-shaped jobs below index it the
# same way the shared ones index read_entity_types().
CHEM_ENTITIES=(
  "functional_group"
)

# Fail on the login node rather than after a 16 GB download.
for _m in "${CHEM_MODELS[@]}"; do
  python -c "
import sys
sys.path.insert(0, 'fc_group')
from model_registry import MODEL_CONFIGS
sys.exit(0 if '$_m' in MODEL_CONFIGS else 1)
" || {
    echo "ERROR: $_m is not in fc_group/model_registry.py"
    echo "  Add its hidden_dim/num_layers/default_layers to MODEL_CONFIGS first."
    exit 1; }
done
unset _m

# Same check for the entity types, against the extraction config that defines them.
for _e in "${CHEM_ENTITIES[@]}"; do
  python -c "
import sys, yaml
with open('fc_group/config_extract_activation.yaml') as f:
    cfg = yaml.safe_load(f)
names = [e['entity_type'] for e in cfg['extraction']['entities']]
sys.exit(0 if '$_e' in names else 1)
" || {
    echo "ERROR: entity type '$_e' not found in fc_group/config_extract_activation.yaml"
    exit 1; }
done
unset _e
