# t-SNE of functional-group activations, by depth

```bash
python fc_group/Analysis/tsne/compute_tsne_coords.py      # coordinates, once
python fc_group/Analysis/paper/make_visuals_tsne.py       # both figures
```

- **`tsne_depth_coarse.png`** — rows are models, columns are layers 0/8/16/24/31, points are
  the 920 prompts, coloured by the five heteroatom families the probe classifies. The paper
  figure.
- **`tsne_depth_fine.png`** — the same panels coloured by all 20 functional groups. Hue still
  carries the family, so the two read together; the marker separates members within a family.
  Appendix.
- **`tsne_depth_template.png`** — the same panels coloured by which of the 10 prompt templates
  produced each point. The control figure, and the one that makes the early layers legible:
  through layer 16 each blob is a *single* colour, so the visible structure is wording. By
  layer 24 the colours interleave, and in chem-faithful at layer 31 they are thoroughly mixed.
  A sequential ramp is used rather than the categorical palette of the other two, so no hue
  here can be mistaken for a family, and because a template index is a bare label — nothing
  distinguishes template 3 from template 7 except which one it is.

`fc_group/Results/tsne_plots/` holds the original sweep — 50 PNGs, no coordinates, silhouette
annotated on every panel. Neither figure here is a re-render of those: the coordinates were
recomputed with the same parameters (PCA to 50 then t-SNE, perplexity 30, `random_state=42`)
and released, which is what makes these reproducible without `Results/`.

## What the figure shows

Structure appears with depth, and it is not the structure you would guess from layers 0–16.
Those early panels look organised, but the ten tight blobs are the **ten prompt templates**,
not chemistry. The layout reorganises around the molecule somewhere between layer 16 and 24 —
the same place the probe's accuracy steps up. This holds in all five checkpoints.

`data/tsne_neighbourhood.csv` puts a number on it. For each point, the share of its 10 nearest
neighbours **in the 2-D panel** carrying the same label — a description of the layout, at the
five layers drawn here:

> The same quantity measured **in activation space, at all 32 layers**, is in
> [`Analysis/prompt_structure/`](../../prompt_structure/README.md). That is the version to
> quote as a claim about the model; this one describes these panels. They agree in ordering at
> layer 31, and that agreement is itself the check that the panels are a faithful picture.
> The per-layer analysis also locates the crossover — layer 19 — which five layers cannot, and
> reports which parts of the result survive a change of `k` and which do not.

| | L0 | L8 | L16 | L24 | L31 |
|---|---|---|---|---|---|
| base — template | 0.657 | **0.989** | 0.910 | 0.680 | 0.744 |
| chem-r — template | 0.650 | **0.988** | 0.892 | 0.629 | 0.594 |
| chem-faithful — template | 0.653 | **0.987** | 0.879 | 0.494 | 0.455 |
| chemdfm — template | 0.666 | **1.000** | 0.906 | 0.526 | 0.520 |
| reason — template | 0.662 | **0.993** | 0.927 | 0.732 | 0.786 |
| base — family | 0.575 | 0.532 | 0.710 | **0.942** | **0.942** |
| chem-r — family | 0.563 | 0.575 | 0.664 | **0.878** | **0.903** |
| chem-faithful — family | 0.561 | 0.559 | 0.688 | **0.923** | **0.933** |
| chemdfm — family | 0.511 | 0.559 | 0.696 | **0.932** | **0.945** |
| reason — family | 0.598 | 0.511 | 0.721 | **0.885** | **0.877** |

`tsne_depth_template.png` is that table drawn. Two things fall out.

**At layer 8 the panel is a picture of the prompt, not the molecule** — template agreement
0.987–1.000 in all five models, chemdfm exactly 1.000, while family agreement sits near
0.51–0.58. Anyone reading
those panels as "the model has not learned chemistry yet" is reading a real fact, but the
visible clusters are wording. The template figure shows this without any statistic: at layers
8 and 16, each blob is one solid colour.

**Every chemistry fine-tune discards template identity; the reasoning distill does not.**
At layer 31 the five split cleanly either side of base (0.744). Template agreement:
chem-faithful **0.455**, chemdfm **0.520**, chem-r **0.594** — all below base — against reason
**0.786**, which is *above* it. Fine-group agreement runs the other way over the same five:
chem-faithful **0.831**, chemdfm **0.811**, chem-r **0.719**, base **0.591**, reason **0.584**. Chemistry
training both adds chemical structure and removes prompt-wording structure; reasoning
distillation adds neither and slightly increases the wording structure. That is the
unsupervised counterpart of "fine-tuning sharpened categorical commitment".

## Three things the figure does not say

**t-SNE coordinates are not comparable across panels.** Every panel is an independent
embedding with its own arbitrary orientation, scale and origin. Only within-panel grouping
carries information. This is why there are no axis ticks, no shared limits, and no claim
anywhere about a cluster "moving" between layers.

**No silhouette score is printed, deliberately.** `fc_group/Analysis/tsne/README.md` §3
establishes that the score the original sweep annotated was computed on the 2-D coordinates
rather than on the activations, and that on this sample "it does not rank plots by how well
they cluster — sometimes it ranks them backwards". A number that cannot be quoted does not
belong on a figure, so it is neither computed nor stored. Cluster quality is the probe's claim
(`probe/data/layer_curves.csv`), not this figure's.

The same caution applies to the table above, and it is why the column is called *neighbourhood
agreement* rather than a cluster score: it is computed on the 2-D coordinates, so it describes
**the panel's layout**, not the separability of the representation. It answers "what is this
picture organised by", which is a question about the picture.

**Visual separation is not linear decodability**, and here the two disagree on purpose. Base's
layer-31 panel looks tidy while its raw space is the most anisotropic of the three (pairwise
cosine $0.944 \pm 0.018$, Table 3). t-SNE will separate points no linear probe could, and does
not preserve density or between-cluster distance.

## Sources

Committed CSVs only, never `Results/`:

- coordinates — `fc_group/Analysis/tsne/data/tsne_coords.csv`
  (23000 rows = 5 models × 5 layers × 920 prompts; `x`, `y`, and the `functional_group`,
  `coarse_family`, `carbon_count`, `template_index` labels for each point).
  `coarse_family` is `functional_group_probe.COARSE_MAP`, the same map the probe classifies
  with, so the colours and Table 1 cannot disagree.
- neighbourhood agreement — `fc_group/Analysis/tsne/data/tsne_neighbourhood.csv`
- generators — `fc_group/Analysis/tsne/compute_tsne_coords.py` (coordinates, reads
  activations), `fc_group/Analysis/paper/make_visuals_tsne.py` (all three figures, reads only
  the CSV). `template_index` is not a dataset column: it is reconstructed from the row layout,
  which is molecule-major and template-minor — the order `generate_prompts()` in
  `extract_activations_subset.py` writes and that `functional_group_probe.py` already relies
  on. If that layout ever changes, the template figure is the first thing that breaks.

**All five checkpoints**, in `_common.BEHAVIOURAL` row order — the order Tables 1 and 3 use,
so a reader moving between them never re-sorts. A model with no activation tree is skipped
with a message and its row dropped rather than blanked, so the grid never carries an empty
band; `--models` takes any slug.

**ChemDFM sits on Llama-3, not 3.1.** The same caveat its Table 1 dagger carries: a
chemdfm–base gap confounds chemistry training with the base-model change. Chem-R-8B is the
controlled contrast, and it moves the same way (template 0.594, fine-group 0.719, against
base's 0.744 and 0.591), so the split above does not rest on ChemDFM.

A note on the unit, the same one the probe carries: a molecule's 10 templates are
near-duplicates, so 920 points are 92 molecules measured 10 ways. Each panel's apparent
density is 10× the independent sample size, which is exactly why the early layers can form
ten crisp blobs out of prompt wording alone.
