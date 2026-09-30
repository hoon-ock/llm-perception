# Prompt wording or chemistry? A depth curve

```bash
python fc_group/Analysis/prompt_structure/analyze_prompt_structure.py
python fc_group/Analysis/paper/make_visuals_prompt_structure.py
```

For each of the 920 prompts, the share of its 10 nearest neighbours **in activation space**
carrying the same label — the prompt template that produced it, its heteroatom family, or its
functional group. One row per (model, layer), all 32 layers, all five checkpoints.

This promotes a result that `visuals/tsne/README.md` found at five layers on 2-D t-SNE
coordinates. That measure describes a panel's layout; this one is computed on the activations,
so it is a claim about the model. The two are companions, not substitutes.

## What it shows

**The first half of every stack is about the prompt, not the molecule.** Template agreement
climbs to ~0.99 by layer 3 and holds a plateau through roughly layer 15, against a chance floor
of 0.099. Whatever those layers encode, the nearest-neighbour structure is dominated by
wording.

**Family agreement overtakes template agreement at layer 19** — and within one layer of
that in every checkpoint (base 19, chem-r 19, chem-faithful 19, chemdfm 19, reason 20). This is *later* than the probe's accuracy step — the largest
single-layer jump is 15→16 in four of the five checkpoints and 17→18 in chem-faithful, and the
probe first clears 0.90 balanced accuracy at layer 16–18 (`probe/data/layer_curves.csv`). The
gap is the interesting part: a linear reader can recover the family two to three layers before
the representation's own neighbourhood structure reorganises around it. Decodability and
dominant structure are not the same event, and the probe sees the earlier one.

**The checkpoints separate only after that, and only on two of the three labels.** Layer 31:

| model | template | family | fine group |
|---|---|---|---|
| base | 0.759 | 0.934 | 0.593 |
| chem-r | 0.587 | 0.913 | 0.753 |
| chem-faithful | 0.489 | 0.942 | 0.843 |
| chemdfm | 0.568 | 0.934 | 0.789 |
| reason | 0.778 | 0.907 | 0.609 |

Family agreement is saturated for everyone (0.90–0.94) — chemistry training does not improve
the coarse partition, which base already has. The separation is entirely in *suppressing
wording* and *resolving within family*.

**The family column is released but not plotted.** `COARSE_MAP` files four groups by the atom
that *names* them rather than the one they contain — `amide` (−CONH₂) and `nitro` (−NO₂) under
nitrogen, `sulfoxide` and `sulfone` under sulfur. A point whose neighbours are chemically
sensible, an amide sitting among esters and ketones, therefore scores as *disagreement*, and
the label's ceiling is not 1.0 for reasons that have nothing to do with the model. That does
not bias the comparison across checkpoints — every model is scored against the same labels —
but it makes the panel easy to misread, so the figure draws only `template` and
`functional_group`, which carry no such convention. The columns stay in the CSV because the
crossover layer is measured against them and because a number worth computing is worth
releasing. The same caveat is the subject of `Analysis/ambiguity/`.

## What is robust, and what is not

**Robust.** At every k tested (5, 10, 20), all three chemistry fine-tunes sit **below** base on
template and **above** base on fine group. Three independent chemistry runs, one direction. The
crossover layer moves only from 18–19 at k=5 to 20–21 at k=20.

**Not robust: where `reason` sits relative to base.** Template agreement at layer 31:

| | k=5 | k=10 | k=20 |
|---|---|---|---|
| base | 0.824 | 0.759 | 0.709 |
| chem-r | 0.709 | 0.587 | 0.532 |
| chem-faithful | 0.641 | 0.489 | 0.404 |
| chemdfm | 0.715 | 0.568 | 0.481 |
| reason | 0.862 | 0.778 | 0.683 |

reason is above base at k=5 and k=10 and **below** it at k=20, and the gap is small at every k.
So the safe statement is that reason *tracks base closely* while the chemistry models pull away
from both — not that reason is above base. What the reason curve establishes is that the effect
is specific to **chemistry** training rather than to fine-tuning in general; its exact rank
against base is not evidence of anything.

The fine ordering *within* the three chemistry models is also not k-stable (chem-r and chemdfm
swap between k=5 and k=10). Quote the group, not the ranking.

## The centring choice, inspectable

Neighbours are ranked on **mean-centred** cosine. The residual stream is strongly anisotropic —
mean pairwise cosine 0.944 in base at layer 31 (`geometry/data/geometry_by_layer.csv`) — so on
raw vectors the shared mean direction contributes to every neighbour ranking, and models would
be scored partly on how collapsed their space is. `analyze_geometry.py` applies the same
correction before it will quote a between-class cosine at all.

Both readings are released. On this data they agree, so the finding does not rest on the choice
(layer 31, template):

| model | centred | raw |
|---|---|---|
| base | 0.759 | 0.751 |
| chem-r | 0.587 | 0.563 |
| chem-faithful | 0.489 | 0.460 |
| chemdfm | 0.568 | 0.559 |
| reason | 0.778 | 0.773 |

## Against the t-SNE measure

Layer 31 template agreement, activation space against the 2-D coordinates:

| model | activation | t-SNE 2-D |
|---|---|---|
| base | 0.759 | 0.744 |
| chem-r | 0.587 | 0.594 |
| chem-faithful | 0.489 | 0.455 |
| chemdfm | 0.568 | 0.520 |
| reason | 0.778 | 0.786 |

Close in value and identical in ordering, which is the check that the t-SNE panels were a
faithful picture rather than a projection artifact. They are still different quantities and
should not be quoted interchangeably.

## Caveats

**ChemDFM is built on Llama-3, not 3.1** — the same dagger Table 1 carries. A chemdfm–base gap
confounds chemistry training with the base-model change. Chem-R-8B is the controlled contrast
and moves the same way, so the result does not rest on ChemDFM.

**n = 3 chemistry checkpoints.** A direction, not an effect size. Nothing here carries an
interval.

**920 points are 92 molecules measured 10 ways.** A molecule's ten templates are near-duplicates
of each other, which is exactly why template agreement can reach 0.99 — and why the independent
sample size behind every number on this page is 92, not 920.

**Chance floors differ per label** and are on every row, because the labels are not otherwise
comparable to each other: template 0.099 / family 0.239 / group 0.054. Computed analytically as `sum_c n_c (n_c - 1) / n (n - 1)`,
and checked against a 200-draw label shuffle.

## Sources

- `fc_group/Analysis/prompt_structure/data/label_agreement_by_layer.csv` — 160 rows
  (5 models × 32 layers), `knn_*_centered`, `knn_*_raw` and `chance_*` for all three labels.
- generators — `analyze_prompt_structure.py` (reads activations),
  `fc_group/Analysis/paper/make_visuals_prompt_structure.py` (figure, reads only the CSV).
- `template_index` is not a dataset column: it is reconstructed from the molecule-major /
  template-minor row layout, the same assumption `Analysis/tsne/compute_tsne_coords.py` makes.
