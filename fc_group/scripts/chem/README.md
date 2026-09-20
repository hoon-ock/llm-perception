# The chemistry-comparison track

Two chemistry models, one entity type, nine stages. Separate from the five-model
sweep in `../` on purpose.

## What it runs

| Model | Base | Licence | Role |
|---|---|---|---|
| `weidawang/Chem-R-8B` | Llama-3.1-8B-Instruct | Apache-2.0 | controlled |
| `OpenDFM/ChemDFM-v1.5-8B` | LLaMA-3-8B | AGPL-3.0 | off-base |

Entity type: `functional_group` only — the declarative arm. Not
`functional_group question`, not the SMILES or IUPAC arms, not the RAG arms.

So every job here is `2 models x 1 entity type = 2 tasks`, `--array=0-1`.

## Why a separate track

The jobs in `../` carry static `#SBATCH --array` spans sized to
`SMALL_MODELS`/`LARGE_MODELS` and to the entity list. Adding two models to the
shared sweep means editing every `--array` line and its paired `ARRAY_SIZE`, and
re-running 5 models x 23 entity types of already-collected work to keep the tree
uniform. This track leaves all of that untouched.

The cost is that `../models.sh` no longer treats the registry as the sweep. It
used to assert `SMALL_MODELS + LARGE_MODELS == MODEL_CONFIGS`, which made
"registered" and "swept" the same thing. It is now a subset check — every swept
model must be registered, but a registered model may sit outside the sweep, and
`models.sh` prints a `note:` line naming any that do. Registration itself is
still mandatory: `functional_group_probe.py`, `tsne_functional_groups.py`,
`anisotropy_diagnostic.py` and `ambiguity_metric.py` all resolve depth through
`get_model_config`, which raises `KeyError` on an unknown name.

## Why these two models

The sweep's only chemistry model is `phenixace/Chem-R-Faithful`, so every
"chemistry-tuned vs base" result rests on one checkpoint, and two things are
confounded in it:

- **Chem-R-8B** is what Chem-R-Faithful was GRPO-trained *from*, with
  fabrication-gated rewards. Holding both separates domain tuning from
  faithfulness tuning. Same base as `meta-llama/Llama-3.1-8B`, so this is a
  controlled contrast.
- **ChemDFM-v1.5-8B** is an independently trained chemistry model of the same
  size and shape (4096/32). But it sits on Llama-3-8B, not 3.1
  (`max_position_embeddings` 8192, `rope_scaling` null, `vocab_size` 128264), so
  a ChemDFM-vs-base gap confounds domain training with the 3 -> 3.1 difference.
  It answers "does the effect survive a different lab's checkpoint" and nothing
  narrower. See the tier note in `fc_group/model_registry.py`.

## Order

Submit from the repo root, same as `../`.

```bash
sbatch fc_group/scripts/chem/chem_00_smoke_test.sbatch   # gate: auth, load, write
sbatch fc_group/scripts/chem/chem_01_extract.sbatch      # ~16 GB download each
sbatch fc_group/scripts/chem/chem_02_probe.sbatch
sbatch fc_group/scripts/chem/chem_06_generation_eval.sbatch
```

`00`/`01`/`02`/`06` are the minimum for a probe-plus-behaviour result. `03`
(t-SNE), `04` (anisotropy), `05` (analogy), `07` (retrieval) and `08`
(ambiguity) mirror the rest of the pipeline and are what the paper's geometry,
analogy and ambiguity claims need. `07` reads what `05` wrote; `08` reads what
`02` wrote.

## Before the first submit

`weidawang/Chem-R-8B` is **gated** — its `config.json` returns HTTP 401 without
credentials. Accept the licence on its model page and export `HF_TOKEN`, or the
download fails inside the job.

```bash
python fc_group/check_model_registry.py     # verifies dims against live configs
```

## Generated, not hand-written

These files were produced from their `../` counterparts by a scripted
transform — the same approach `../rag/` uses — so the preflight, thread-limit
exports, `SUMMARY:` lines and failure triage stay identical. The deltas are:
source `chem_models.sh` instead of `models.sh`, `ENTITIES=("${CHEM_ENTITIES[@]}")`
instead of `read_entity_types()`, `--array=0-1` / `ARRAY_SIZE=2`, and a `chem-`
infix on the log path so the two tracks do not interleave in `fc_group/logs/`.
