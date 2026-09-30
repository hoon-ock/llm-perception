# `visuals/generation/` — per-functional-group generation against per-group representation

Exploratory PNGs, not paper figures. Written by
[`Analysis/paper/make_visuals_generation_vs_representation.py`](../../paper/make_visuals_generation_vs_representation.py),
which reads only `generation/data/group_*.csv` and never `Results/` — the same rule the
`make_visuals_retrieval*.py` family follows. Those CSVs are snapshotted by
[`Analysis/generation/analyze_generation_vs_representation.py`](../../generation/analyze_generation_vs_representation.py).

The reading of all three together is
[`narrative/group_level_representation_behaviour.md`](../../narrative/group_level_representation_behaviour.md).

Regenerate:

```
python fc_group/Analysis/generation/analyze_generation_vs_representation.py
python fc_group/Analysis/paper/make_visuals_generation_vs_representation.py
```

---

## `gen_vs_retrieval.png`

The question as posed: does a group the model retrieves badly get generated badly? Analogy
retrieval hit@1 (top row) and mean rank (bottom) at layer 31, against per-group free-generation
accuracy with superclass credit. One facet per checkpoint, ten facets.

Read the **left-hand tail**, not the pile. Most groups sit at hit@1 = 1.000, so the panel's whole
information content is the handful of points off that value — and they run the wrong way. Base:
`sulfone` at hit@1 0.000 generates at 0.875, `imine` at 0.667 generates at 0.275.

Three conventions worth knowing:

* **Not every point is labelled.** `label_subset()` annotates points off the modal predictor value
  plus the two y-extremes. Labelling all twenty turns the pile into overstrike and hides the tail,
  which is the only part that carries information.
* **The displayed rho is flipped on the bottom row.** `analyze_generation_vs_representation.py`
  scores rank predictors negated so "higher is better" holds uniformly in
  `group_correlations.csv`; these panels plot the raw rank, so the sign is flipped back to match
  the axis the reader sees. A rho quoted from the CSV will differ in sign from the one in the
  panel title, by design.
* **A cream panel background means the outcome is unreliable** for that checkpoint — split-half
  over templates below 0.5. chem-r and chem are cream. Their flat clouds mean the measurement
  could not resolve a relationship, not that there is none.

## `rho_by_layer.png`

Correlation against depth, six predictors, five checkpoints plus a pooled fit. This is the figure
that decides the question, because the one association that exists is layer-specific and any
single-layer view either misses it or overstates it.

* Shaded bands are cluster-bootstrap 95% CIs **over functional groups** — drawn for the two
  cohesion predictors only. Shading all six is unreadable, and those two are the ones carrying a
  claim. The width is the result: with n = 17–20 groups the bands run roughly ±0.45, so only
  rho beyond ~0.55 clears zero in a single model.
* The pooled panel ranks within model before pooling and still bootstraps over groups, so it
  does not manufacture power by counting the same twenty groups five times.
* Cream panels are underpowered checkpoints, as above, and the label carries the reliability.
* Everything here except the preregistered primary is exploratory; `group_correlations.csv`
  carries the BH-adjusted p-values and the `is_primary` flag.

## `group_response_composition.png`

What the per-group spread is actually made of. Each functional group is one stacked bar of the
five adjudicated response types from `free_generation_scoring.py`, one row per checkpoint, with
teacher-forced raw recall overlaid as a dark tick.

This is the panel that answers the question the correlation could not. Read base's four halides:
tall light-blue `superclass only` blocks under a tick near the top — `alkyl chloride` is 0.150
strict against 0.950 teacher-forced, because the model answers "alkyl halide". Then read `imine`
and `thioether`: no light blue at all, a tall orange `named no group` block, tick at 0.800. Then
`alkyl fluoride` and `alkyl iodide`, the only two groups where the tick itself is low.

Three different failure modes, pooled into one number by strict scoring, and only the third is the
kind of thing an embedding metric is about.
