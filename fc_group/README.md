# fc_group — Functional Group Geometry Analysis

This sub-project asks the same kind of question as the root repository's periodic-table
study, applied to organic chemistry: **do LLM hidden states encode functional groups as a
consistent geometric structure**, analogous to word2vec-style analogies
(`king - man + woman ~= queen`)?

For each non-alkane molecule we define:

```
diff_vec(group, C) = act(molecule with `group` at chain length C) - act(alkane backbone at the same chain length C)
```

e.g. `diff_vec(alcohol, 3) = act(propan-1-ol) - act(propane)`

The pipeline never generates text — every script does a single forward pass and pulls
per-layer **hidden states** via forward hooks, then analyzes the geometry of those
activation vectors (diff-vector consistency across chain lengths, within/between-class
similarity, anisotropy baselines, t-SNE/PCA visualization).

## Repository structure

| File | Role |
|---|---|
| `config_extract_activation.yaml` | Extraction config: model name, batch size, quantization, and the entity types / prompt templates to run |
| `model_registry.py` | Per-model architecture constants (`hidden_dim`, `num_layers`, `default_layers`) used to switch between LLMs |
| `extract_activations_subset.py` | Loads a model, runs one forward pass per batch, saves per-layer hidden-state activations as `.pt` files |
| `extract_properties.py` | Unrelated to the LLM — builds `functional_group_dataset.csv` via RDKit + PubChem PUG View |
| `functional_group_dataset.csv` | Curated dataset (IUPAC name, formula, functional group, pKa/pKaH, TPSA, HBD/HBA, boiling point, water solubility, ...) |
| `functional_group_analogy.py` | Diff-vector analogy geometry: within-class consistency, between-class distinctness, second-order analogies |
| `functional_group_analogy_carbon_matched.py` | Same analysis with carbon-count-matched controls |
| `anisotropy_diagnostic.py` | How much of the diff-vector geometry is generic transformer-hidden-state anisotropy vs. genuine chemistry-specific structure |
| `tsne_functional_groups.py` | t-SNE/PCA visualization of activations, colored by functional group / pKa / TPSA / etc. |

Generated at runtime, not committed (see `.gitignore`):
`activation_datasets_functional_groups/` (saved activation tensors, one subdirectory per
model), `Results/` (plots, CSVs, JSON summaries), `logs/`.

## Setup

Same environment as the repository root — see the top-level `README.md` for
`requirements.txt` install and hardware/quantization notes (bitsandbytes 4-bit
quantization is Linux+NVIDIA only; on macOS/CPU it falls back to plain fp16/bf16).

Gated models (Llama, Qwen, DeepSeek-distilled checkpoints) need Hugging Face auth — either
run `huggingface-cli login` once, or add an `HF_TOKEN` key to
`config_extract_activation.yaml` (it's read via `config_data.get("HF_TOKEN")`, `config.json`
at the repo root is a separate file used by other sub-projects and is *not* read here).

## Usage

**1. Extract activations**

```bash
python fc_group/extract_activations_subset.py --config fc_group/config_extract_activation.yaml
```

Useful flags:
- `--entity-types functional_group molecule_name_bare ...` — only process a subset of the entity types defined in the config
- `--layers bottom middle top` or `--layers 0 15 31` — only extract specific layers instead of every layer
- `--max-templates N` — use only the first N prompt templates per entity type (fast smoke test)
- `--model-name <hf_repo_id>` — override `extraction.model_name` from the config file without editing it

**2. Run the analysis scripts** against the saved activations:

```bash
python fc_group/functional_group_analogy.py --entity-type functional_group
python fc_group/functional_group_analogy_carbon_matched.py --entity-type functional_group
python fc_group/anisotropy_diagnostic.py --entity-type functional_group
python fc_group/tsne_functional_groups.py --entity-type functional_group
```

Common flags across the analysis scripts: `--entity-type`, `--model-name`, `--layers`
(comma-separated indices), `--output-dir`, `--num-null-samples`. Run any script with `--help`
for its full list.

Available `entity_type` values are defined in `config_extract_activation.yaml`:
`functional_group`, `functional_group_structure`, `pka`, `pkah`, `tpsa`,
`avg_carbon_oxidation_state`, `hbd`, `hba`, `boiling_point_c`, `water_solubility`,
`molecule`, `molecule_name_bare`, `molecule_formula_bare` — each also has a `... question`
variant using interrogative prompt templates instead of declarative ones.

## Multi-model support

`model_name` is a config/CLI parameter, not hardcoded — swapping models doesn't require
editing the extraction script, and since the pipeline only ever does a forward pass (no
generation, no chat template, no output parsing), there's no prompt-format work needed to
add a new model. It just needs an entry in `model_registry.py`:

```python
MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B": {"hidden_dim": 4096, "num_layers": 32, "default_layers": [0, 16, 31]},
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {"hidden_dim": 4096, "num_layers": 32, "default_layers": [0, 16, 31]},
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B":  {"hidden_dim": 3584, "num_layers": 28, "default_layers": [0, 14, 27]},
    "Qwen/Qwen3-8B":                            {"hidden_dim": 4096, "num_layers": 36, "default_layers": [0, 18, 35]},
    "Qwen/Qwen2.5-Math-7B":                     {"hidden_dim": 3584, "num_layers": 28, "default_layers": [0, 14, 27]},
}
```

The one architecture constraint: the model must expose `model.model.layers` (true for
Llama/Mistral/Qwen2/Qwen3-family causal LMs in `transformers`, not guaranteed for every
architecture). Verify `hidden_dim`/`num_layers` against the model's actual `config.json` on
Hugging Face before adding a new entry.

Extraction:
```bash
python fc_group/extract_activations_subset.py --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

Analysis (must match the model name used at extraction time, since it's used to locate the
saved activations under `activation_datasets_functional_groups/<model-name>/`):
```bash
python fc_group/functional_group_analogy.py --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B --entity-type functional_group
```

When `--model-name` is omitted everywhere, behavior is unchanged from before this option
existed — it defaults to `meta-llama/Llama-3.1-8B`.
