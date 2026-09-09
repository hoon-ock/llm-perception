# SLURM jobs

Submit from the **repo root** — every path inside these scripts is repo-root-relative:

```bash
sbatch fc_group/scripts/01_extract_small.sbatch
```

Numbered by dependency order. Nothing from `02` onward can run until `01` has produced
activations; a task whose activations are missing reports `SUMMARY: no activation data` and
exits non-zero rather than failing obscurely.

| job | resources | tasks | what it runs |
|---|---|---|---|
| `00_smoke_test_small` / `_large` | gpu:1 / gpu:4 | 5 each | one template, three layers — checks a model loads and emits right-shaped output before committing to a full sweep |
| `01_extract_small` / `_large` | gpu:1 / gpu:4 | 5 each | `extract_activations_subset.py` — the activations everything else reads |
| `02_probe` | cpu, 4G | 240 | `functional_group_probe.py` |
| `03_tsne` | cpu, 32G | 240 | `tsne_functional_groups.py` |
| `04_anisotropy` | cpu, 32G | 240 | `anisotropy_diagnostic.py` |
| `05_analogy` | cpu, 32G | 240 | `functional_group_analogy_carbon_matched.py` |
| `06_generation_eval_small` / `_large` | gpu:1 / gpu:4 | 5 each | `generation_eval.py` — behavioural readout, needs no activations |

240 = 10 models × 24 entity types.

## Why small/large are separate files

`--gres` cannot vary per array task. Models ≤8B run at full precision on one GPU; models ≥32B
need 4-bit sharded across four. One array cannot express both.

## `models.sh`

Sourced by every job. Model **names** come from `fc_group/model_registry.py`, which is already
the source of truth for `hidden_dim`/`num_layers`. The **small/large split** is declared in
`models.sh` because it is a scheduling decision rather than a property of the model — and is
then asserted to union exactly to the registry, so adding a model there without giving it a size
class fails loudly instead of dropping it from every sweep.

Entity types come from `config_extract_activation.yaml` the same way. Neither list is retyped
in any job.

Each array job also checks `len(models) × len(entities) == ARRAY_SIZE` against the literal in
the `#SBATCH --array` line, which is the one thing that cannot derive itself. Change the two
together.

## Checking a sweep

```bash
grep -h SUMMARY slurm-*.out | sort | uniq -c
```

## Variations

`02_probe` takes its target and split from the environment, which is what replaced the old
`run_functional_group_probe_fine_molecule.sbatch`:

```bash
sbatch --export=ALL,PROBE_TARGET=fine,PROBE_SPLIT=molecule fc_group/scripts/02_probe.sbatch
```

Cap concurrency on a busy partition by appending `%N` to the array, e.g. `--array=0-239%20`.

## `archive/`

Superseded runners, kept for reference. Nothing here is wired to `models.sh`.
