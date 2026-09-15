# SLURM jobs

Submit from the **repo root** — every path inside these scripts is repo-root-relative:

```bash
sbatch fc_group/scripts/01_extract_small.sbatch
```

Logs land in `fc_group/logs/`, not the submission directory. **SLURM does not create that
directory** — it resolves `--output` before the script runs, so a missing `fc_group/logs/`
loses the job's output. `.gitkeep` keeps it present through a clone; everything written into
it is ignored.

Numbered by dependency order. Nothing from `02` onward can run until `01` has produced
activations; a task whose activations are missing reports `SUMMARY: no activation data` and
exits non-zero rather than failing obscurely.

| job | resources | tasks | what it runs |
|---|---|---|---|
| `00_smoke_test_small` / `_large` | gpu:1 / gpu:4 | 3 / 2 | one template, three layers — checks a model loads and emits right-shaped output before committing to a full sweep |
| `01_extract_small` / `_large` | gpu:1 / gpu:2 | 3 / 2 | `extract_activations_subset.py` — the activations everything else reads |
| `02_probe` | cpu, 4G | 115 | `functional_group_probe.py`. Also emits `confusion_{tag}.csv` (where every prediction landed, every layer) and `oof_proba_{tag}.npz` (the class probabilities behind them) with **no flag change** — `08` reads the latter. `--no-save-proba` opts out of the npz |
| `03_tsne` | cpu, 32G | 115 | `tsne_functional_groups.py` |
| `04_anisotropy` | cpu, 32G | 115 | `anisotropy_diagnostic.py` |
| `05_analogy` | cpu, 32G | 115 | `functional_group_analogy_carbon_matched.py` |
| `06_generation_eval_small` / `_large` | gpu:1 / gpu:4 | 3 / 2 | `generation_eval.py` — behavioural readout, needs no activations |
| `07_retrieval` | cpu, 8G | 115 | `analogy_retrieval.py` — reads what `05_analogy` wrote, not the activations, so it must run after it. Runs **all three quadruple sets** per task (`inter`, `halide`, `carbon`), each into its own Results tree, and emits one `SUMMARY:` line per set. Self-checks before writing, so `SUMMARY: ok` means the output was validated |
| `08_ambiguity` | cpu, 8G | 115 | `ambiguity_metric.py` — reads the `oof_proba_*.npz` that `02_probe` wrote, so it must run after it. Writes the mean probability on each family per held-out group, which is how the four convention-labelled groups (`amide`, `nitro`, `sulfoxide`, `sulfone`) are read. `SUMMARY: no probe output` means that model/entity_type's probe results predate the npz — rerun `02_probe`, not a crash |

115 = 5 models × 23 entity types.

> **On this branch that product is currently 125, not 115.** `config_extract_activation.yaml` carries two extra entity types (`functional_group_rag`, `functional_group_rag_mismatched`) as an uncommitted change, and `read_entity_types` in `models.sh` returns every entity in the config with no filter. So `02`–`05`, `07` and `08` all abort on their own
> `len(models) × len(entities) == ARRAY_SIZE` guard until that is resolved — the guard working as designed, but the whole matrix is blocked meanwhile. The fix belongs in `models.sh` rather than in the array literals: `scripts/rag/README.md` states the shared scripts are meant to stay byte-identical to `main`, which means the RAG arms should be filtered out of `read_entity_types` and left to `scripts/rag/`, not folded into the main sweep.

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

`08_ambiguity` takes its depth from the environment the same way. Bare, it writes the plain probability table in seconds; `AMBIGUITY_FULL=1` adds the temperature-calibrated contrast with bootstrap CIs and the centroid-axis projection — the half that answers "could this just be how hard `select_C` regularized each model", and the only half that reads activations:

```bash
sbatch --export=ALL,AMBIGUITY_FULL=1 fc_group/scripts/08_ambiguity.sbatch
```

Cap concurrency on a busy partition by appending `%N` to the array, e.g. `--array=0-119%20`.

## `archive/`

Superseded runners, kept for reference. Nothing here is wired to `models.sh`.
