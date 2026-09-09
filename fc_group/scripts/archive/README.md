# Superseded job scripts

Kept for reference. These are **not** maintained and are not wired to `../models.sh`.

| file | replaced by |
|---|---|
| `run_functional_group_probe_fine_molecule.sbatch` | `../02_probe.sbatch` with `--export=ALL,PROBE_TARGET=fine,PROBE_SPLIT=molecule`. It was a near-duplicate of the main probe runner differing only in target, split and `--surface-baseline` — the last of which is now on for every probe task. |
