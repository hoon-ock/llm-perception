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
| `functional_group_probe.py` | Supervised counterpart to the above: layer-wise linear probe predicting functional group from the last-token residual stream |
| `run_functional_group_probe.sbatch` | SLURM job array running the probe for 2 models x every entity type in the config (46 tasks) |

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
python fc_group/functional_group_probe.py --entity-type functional_group
```

Common flags across the analysis scripts: `--entity-type`, `--model-name`, `--layers`
(comma-separated indices), `--output-dir`, `--num-null-samples`. Run any script with `--help`
for its full list.

Available `entity_type` values are defined in `config_extract_activation.yaml`:
`functional_group`, `functional_group_structure`, `pka`, `pkah`, `tpsa`,
`avg_carbon_oxidation_state`, `hbd`, `hba`, `boiling_point_c`, `water_solubility`,
`molecule`, `molecule_name_bare`, `molecule_formula_bare` — each also has a `... question`
variant using interrogative prompt templates instead of declarative ones.

## The linear probe

`functional_group_probe.py` is the one supervised analysis here, and the functional-group
counterpart of `Direct_recall/categorical_probe.py` in the root project. It asks whether
functional-group identity is *linearly decodable* from the last-token residual stream, and at
what depth decodability peaks -- the question the unsupervised geometry scripts cannot answer.

Three targets (`--target`, or `all` for every one of them):

| Target | Trained on | Scored on | Chance |
|---|---|---|---|
| `fine` | 20 `functional_group` classes | the same 20 | 5% |
| `coarse` | 5 heteroatom families -- hydrocarbon / halide / oxygen / nitrogen / sulfur | the same 5 | 20% |
| `fine_to_coarse` | all 20 classes | the 5 families | 20% |

`fine_to_coarse` is the one worth explaining. Under leave-one-group-out a plain `fine` probe is
impossible -- withholding a class removes its output unit -- but a probe *trained* on all 20
can still be asked whether a held-out group's molecules land in the right family. Withhold
`alkyl iodide`, and "it was called a bromide" is a stronger result than "it landed somewhere in
the halide blob". Because it is scored on the 5 families, it is directly comparable to `coarse`.

Three splits, in increasing order of strictness (`--split`, repeatable):

| Split | Folds | What it tests |
|---|---|---|
| `molecule` | 4 (`StratifiedGroupKFold`) | Decodability with all templates of a molecule kept on one side. 4 folds, not 5: 17 of the 20 fine classes have exactly 4 molecules. |
| `carbon` | 4 (leave-one-carbon-count-out) | Whether the functional-group direction is invariant to backbone length -- the probe analogue of `functional_group_analogy_carbon_matched.py`. |
| `group` | 19 (leave-one-fine-group-out) | Withhold `alkyl iodide` entirely; is it still placed with the halides having seen only F/Cl/Br? |

**`--split` defaults to `group` alone.** `molecule` and `carbon` are saturated by the molecule
name in the prompt -- character n-grams alone score 0.975-1.00 on both -- so running them by
default only produces impressive numbers that say nothing about the model. They remain one flag
away and become worth running again the moment the prompts stop naming the molecule. Pairing
`--target fine_to_coarse --split molecule` is also the natural ceiling for a leave-one-group-out
score: it measures how often the fine probe gets the family right when the group *was* seen.

`group` is skipped for the plain `fine` target on purpose: removing a class from an n-way
classifier removes its output unit, so held-out accuracy would be 0 by construction and measure
nothing. Use `fine_to_coarse` instead.

Reported per layer: balanced accuracy (headline), macro-F1, plain accuracy, and a per-molecule
score that sums class probabilities over a molecule's prompt templates before the argmax.
`--num-null-samples` runs a label-permutation null shuffled over the **92 molecules** (not the
expanded rows, which would leak templates of the same molecule across the split); it defaults
to five layers spread over depth. `--null-layers all` overrides that.

### Reference nulls

**The sweep runs with `--num-null-samples 0`, and that is deliberate.** The null is ~91% of the
job's runtime -- 19 folds x 50 permutations x ~0.2 s against ~19 s of real CV per layer/target
-- and it does not depend on the activations. Shuffled labels leave no signal regardless of what
`X` holds, so the null is a property of the fold structure, class distribution, classifier and
metric, all identical across every task in the sweep. Measured across layers 0/16/31 it is flat
and sits at chance:

| experiment | chance | by layer | spread |
|---|---|---|---|
| `coarse_molecule` | 0.20 | 0.189 0.193 0.193 | 0.004 |
| `coarse_group` | 0.20 | 0.199 0.184 0.189 | 0.014 |
| `fine_molecule` | 0.05 | 0.037 0.039 0.043 | 0.006 |
| `fine_to_coarse_group` | 0.20 | 0.200 0.185 0.190 | 0.015 |

Layer 0 and layer 31 are entirely different representations, so 46 tasks each computing this
would be estimating one constant 46 times. Measured on the sweep's own configuration, turning it
off is **8.3x faster** (168.4 s -> 20.2 s) and leaves every `probe_scores_*.csv` byte-identical.

The null still matters once, as the **leak detector**: if the molecule-level shuffle ever broke,
it would rise above chance. Run it separately per model -- the two differ in hidden dimension --
and quote those as the reference band for every entity type:

```bash
python fc_group/functional_group_probe.py --model-name meta-llama/Llama-3.1-8B \
  --entity-type functional_group --num-null-samples 50 --output-dir Results/null_ref_8b
python fc_group/functional_group_probe.py --model-name meta-llama/Llama-3.1-70B \
  --entity-type functional_group --num-null-samples 50 --output-dir Results/null_ref_70b
```

With the null off, `summary_*.json` carries `null_mean: null` and `p_value_best_layer: null`,
and the layer-trend plots drop the null band while keeping the chance line and the surface
baseline.

`--surface-baseline` fits the same probe on char n-grams of the raw prompt strings and draws
the result onto every layer-trend plot. Every template embeds `{iupac_name} ({formula})`, so
"Propan-1-ol (CH3CH2CH2OH)" hands a probe the substrings `-ol` and `OH` directly; this baseline
is how much accuracy is available without running the model at all, and anything the probe
scores below that line is not evidence of recalled chemistry. On a first run over Llama-3.1-8B
it already reorders which results mean anything:

| Split | Surface n-grams | Probe (layer 31) |
|---|---|---|
| fine / molecule | 0.98 | 1.00 |
| fine / carbon | 1.00 | 1.00 |
| coarse / molecule | 1.00 | 1.00 |
| coarse / carbon | 1.00 | 1.00 |
| **coarse / group** | **0.47** | **0.98** |

The molecule and carbon splits are saturated by orthography alone -- the functional group is
recoverable from the molecule's spelling, so a near-perfect probe there says little. The
leave-one-group-out split is the one that separates them, and it is also the only split where
the probe climbs with depth (0.41 -> 0.72 -> 0.98 across layers 0/16/31).

> These numbers come from the local single-template smoke activations (layers 0/16/31 only,
> one prompt template rather than ten), so treat them as a shape, not a result. Re-run against
> the full extraction before quoting them.

**The sweep does not run it.** That is a deliberate choice rather than a cost one: it is only
~0.23 s per task (~0.2% of runtime), and unlike the null it genuinely varies per entity type --
the 21 name+formula prompts span 0.4934-0.5724, and `molecule_formula_bare` sits apart at
0.5197 because its prompt carries no molecule name. So sweep results carry no orange line and
`summary_*.json` has `surface_baseline_balanced_acc: null`.

It does not scale with layer count, so one layer recovers it for any entity type in seconds
(measured: 6.8 s):

```bash
python fc_group/functional_group_probe.py --entity-type <name> \
  --layers 31 --num-null-samples 0 --surface-baseline --output-dir <dir>
```

Worth doing for any entity type whose probe score you intend to quote -- without it there is no
way to tell a real result from one that spelling already explains.

### Cost

Two things keep this affordable, both exact rather than approximations:

`--pca-components` defaults to a full-rank PCA rotation of each training fold. Under an L2
penalty the optimum lies in the span of the training data, and PCA is an orthonormal rotation
within that span, so the penalty is preserved and the raw-4096-dim solution is reproduced to
the digit -- while running ~10x faster. Pass `-1` to fit in the raw hidden-state basis and
confirm this on your own data.

The scaler and PCA are also fit once per fold and reused, rather than refit inside the loops
that vary only the labels (`run_null`) or only `C` (`select_C`). Neither transform is
supervised, so this cannot change a result; measured end to end it is ~1.5x, and on the wider
70B activations `select_C` is ~2.8x and `run_null` ~1.8x. The gain is bounded by the logistic
regression itself, which dominates on real activation data.

### The sweep

`run_functional_group_probe.sbatch` is a 46-task array: 2 models x every entity type in
`config_extract_activation.yaml`, running `--target all` at the default `group` split. The probe
target is always the functional group, so the entity type selects which *prompt* the last-token
state is read from -- `functional_group` asks for the group, `pka` asks about acidity,
`molecule_name_bare` asks nothing at all. Sweeping all of them is the attribute-specificity
matrix: is functional-group identity present in every last-token state, or only when the prompt
asks for it?

The entity list is derived from the config, as in `run_anisotropy_diagnostic.sbatch`, so the two
cannot drift. The one thing that cannot self-update is the static `#SBATCH --array` range, so a
preflight asserts `models x entities == ARRAY_SIZE` and fails at submission if they disagree.

**One CPU core, 4 GB, no GPU, and all three are deliberate.** Measured peak RSS at 70B scale
(920 x 8192) is 910 MB, so 4 GB is ~4x headroom; the earlier 32 GB request -- inherited from
`run_anisotropy_diagnostic.sbatch` -- pinned the array to 15 concurrent tasks on one node
(15 x 32 GB = 480 GB) with the remaining 31 stuck in `PD (Resources)`. The probe is entirely scikit-learn; `torch`
appears only as `torch.load(..., map_location="cpu")`. A GPU would idle for the whole run while
competing with the extraction jobs that need one. Extra cores do not help either -- measured on
a 920x8192 fold, one preprocess+fit takes 0.35s at 1 thread and 0.33s at 8, because the matrices
are too small for BLAS parallelism and lbfgs is serial. Throughput comes from the array width.

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
