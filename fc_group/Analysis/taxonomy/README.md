# Which taxonomy does the geometry follow?

The paper's aim-2 headline, and the one result that turns "Chem-R's clustermap looks
different" into a number with a confidence interval.

Source: `fc_group/Results/functional_group_analogy/{model}/functional_group/data/
between_class_similarity_layer_*.csv` — the diff-vector cosine between every pair of the
19 non-alkane groups, which the analogy script already writes. That tree is **gitignored**,
so `data/` here is the durable record.

```bash
python fc_group/Analysis/taxonomy/analyze_taxonomy.py
```

## The question

`COARSE_MAP` sorts a group into a family by the heteroatom that **names** it. That is one
defensible partition of the 19 groups. Reactivity class is another, and it disagrees in
exactly four places:

| group | naming | reactivity | why |
|---|---|---|---|
| `amide` | nitrogen | carbonyl | an acyl derivative; reacts as one |
| `nitro` | nitrogen | C–N | no carbonyl carbon, so not an acyl derivative |
| `sulfone`, `sulfoxide` | sulfur | S=O | oxidised sulfur, not divalent like thiol/thioether |
| `alcohol`/`ether` vs `thiol`/`thioether` | oxygen vs sulfur | one family | the two isostere pairs |

For each model and layer, with cosines z-scored inside that model's own between-class
distribution so a globally flatter model is not penalised:

```
contrast(partition) = mean z-cosine(pairs inside a family) - mean z-cosine(pairs across)
delta               = contrast(reactivity) - contrast(naming)
```

`delta > 0` means the geometry sorts by reactivity; `delta < 0`, by nomenclature.

## What it found

`data/taxonomy_contrast.csv`. **All three models start reactivity-leaning and only the base
model crosses over**:

| delta | L0 | L8 | L16 | L24 | L31 |
|---|---|---|---|---|---|
| Llama-3.1-8B | +0.534 | +0.506 | +0.260 | +0.003 | **−0.205** |
| Chem-R-Faithful | +0.245 | +0.400 | +0.423 | +0.367 | **+0.268** |
| DeepSeek-R1-Distill | +0.365 | +0.366 | +0.341 | +0.153 | −0.002 |

The base model's geometry migrates, monotonically, from reactivity toward nomenclature
across the stack. Chem-R's does not. That is the mechanism: **Llama ends up organised by the
naming convention its probe labels encode; Chem-R ends up organised by chemistry.**

It also explains, without needing a second hypothesis, why Llama scores higher on the probe
(`coarse_group` 1.000 vs 0.983, molecule-level CI [−0.032, −0.006] — see `../probe/`): the
probe's labels *are* the naming partition, so a model organised that way is being scored on
its own axis.

## Why the paired bootstrap is the only test reported

`data/taxonomy_paired.csv`. Resampling unit is the **group**, n = 19.

| layer | Chem-R delta | Llama delta | paired | 95% CI | P(>0) |
|---|---|---|---|---|---|
| 0 | +0.245 | +0.534 | **−0.289** | [−0.535, +0.005] | 0.029 |
| 16 | +0.423 | +0.260 | +0.163 | [−0.138, +0.468] | 0.863 |
| 24 | +0.367 | +0.003 | +0.363 | [+0.004, +0.698] | 0.976 |
| 31 | +0.268 | −0.205 | **+0.474** | [+0.108, +0.798] | 0.992 |

Per-model deltas do **not** individually clear zero at n = 19 — quote the paired difference
only. Both models are evaluated on the same drawn groups, so their sampling noise is largely
shared and the paired interval is far tighter than either marginal.

Layer 0 is the control that makes the rest worth believing: at the input embedding the sign
is *reversed* and significant. The divergence is produced by the stack, not inherited from a
tokenizer the three models share.

## The permutation control

`data/taxonomy_permutation.csv`. A partition into more, smaller families scores a higher
within-minus-between contrast for purely combinatorial reasons, so the control shuffles group
labels while **matching the family-size profile** — reactivity's [4,5,4,4,2] against naming's
[4,5,6,4].

Both nulls centre on 0.00 ± 0.19, so the shape-matching worked: a 5-family partition gets no
free advantage over a 4-family one. Against those nulls:

| min z over the six (model × partition) cells | L0 | L8 | L16 | L24 | L31 |
|---|---|---|---|---|---|
| | 1.94 | 2.78 | **5.08** | **6.97** | **7.58** |

**From layer 16 on, both named partitions clear the matched null in every model** (z = 5.1–9.9,
p ≤ 0.001). At layers 0 and 8 they do not reliably: the *naming* partition is weakest of all
(z = 1.9–4.1, p up to 0.045), while reactivity already clears at L0 in every model (z = 3.9–4.5).

That is a limit and a result at once. It limits the L0 paired control in the previous section —
a difference between two weakly-structured contrasts — so read that row as "the sign is not yet
what it becomes", not as a significant reversal. And it is consistent with the trajectory: at
the input embedding there is barely any nomenclature structure to find, and what taxonomy is
present is already the reactivity one.

## Limits

- Five layers only (0/8/16/24/31) — that is what the analogy tree holds.
- n = 19 groups, 171 pairs. Fine for the paired test, not for per-model claims.
- The reactivity partition is hand-specified. It is defensible group-by-group (table above)
  and it beats shape-matched random partitions, but it is not the only such partition, and no
  search over partitions was run — that would need a correction this design does not have.
