# t-SNE of functional-group activations — what the plots do and don't show

Companion to `fc_group/Analysis/probe/README.md`. That one covers the supervised probe; this
one covers the unsupervised geometry in `fc_group/Results_HCC/tsne_plots/`.

> **The headline is a caveat.** The silhouette score printed on every one of these plots is
> computed on the **2-D t-SNE coordinates**, not on the activations, and in this sample it does
> not rank plots by how well they cluster — sometimes it ranks them backwards. §3 has the
> evidence. Nothing in this tree should be quoted as a cluster-quality number.

## 1. What actually exists

`fc_group/Results_HCC/tsne_plots/{model}/{coloring}/{prompt_type}_layer_{L}.png`

- **2760 files, all PNG.** Zero CSV, zero JSON — no machine-readable output at all.
- 2 models × 12 colorings × 23 prompt types × 5 layers.
- Layers: 8B `0, 8, 16, 24, 31`; 70B `0, 20, 40, 60, 79`.
- Colorings present: `functional_group`, `functional_group_structure`, `carbon_count`, `mw`,
  `pka`, `pkah`, `tpsa`, `hba`, `hbd`, `boiling_point_c`, `water_solubility`,
  `avg_carbon_oxidation_state`.

Two things the current script writes that this tree does **not** contain, because the sweep
predates them:

| artifact | added in | sweep ran |
|---|---|---|
| `metrics/*_metrics.csv` (silhouette per layer × feature) | `a703522`, 2026-09-04 | **2026-09-01** |
| `template_index` coloring | `a703522`, 2026-09-04 | **2026-09-01** |

So every claim below rests on **reading the plots**, which is also all the source material
anyone else has. The absence of `template_index` matters specifically for the third claim.

## 2. Method

11 plots inspected directly, all under the `functional_group` coloring (recorded in
`sampled_plots.csv` with silhouette and verdict). "Clustered" means colour-coherent regions are
visible; "mixed" means every clump contains many colours. This is a sample, not a census — 5 of
23 prompt types at the 8B middle layer, plus a layer series for `pka` in both models.

## 3. The printed silhouette is not usable

Four plots at the same model and layer (8B, L16, `functional_group` coloring):

| prompt | silhouette | what the plot shows |
|---|---|---|
| `functional_group` | **−0.08** | fully mixed, no FG structure |
| `tpsa` | **−0.08** | fully mixed |
| `avg_carbon_oxidation_state` | **−0.08** | fully mixed |
| `pka` | **−0.21** | clear functional-group clusters |

Within that layer the metric is **inverted** — the three worst plots share the best score. And
across layers it is not even consistently inverted: 70B `pka` L60 is the cleanest plot in the
whole sample at **−0.06**, while 8B `pka` L31, visibly degraded, is **−0.28**.

The cause is at `fc_group/tsne_functional_groups.py:161`:

```python
silhouette_2d = float(silhouette_score(tsne_data, labels_numeric))
```

It scores the 2-D embedding, with 20 classes. t-SNE routinely splits one class across several
islands; when it does, mean intra-class distance is large while the nearest other class is
immediately adjacent, so silhouette goes strongly negative *even though local purity is
perfect*. Uniform mixing, by contrast, makes intra- and inter-class distances equal and drives
the score toward 0. The metric therefore rewards mixing and punishes genuine multi-island
clustering.

**Fix:** compute cluster quality in the original activation space, before PCA/t-SNE — silhouette
on the 4096-d (or 8192-d) vectors, or a k-NN label-purity score, which is robust to a class
occupying several regions. t-SNE stays a visualisation; it should not be the thing measured.

## 4. The three claims

### "The pKa prompt is the only one that produces valid clustering by functional group"

**Half right — `pkah` does it too, and the question variant does not.**

| prompt (8B, L16) | verdict |
|---|---|
| `pka` | clustered |
| `pkah` | clustered — indistinguishable in quality |
| `pka question` | **mixed** |
| `functional_group` | mixed |
| `tpsa` | mixed |
| `avg_carbon_oxidation_state` | mixed |

So it is not "the pKa prompt" but the **declarative acidity prompts**, plural. That
`pka question` fails while `pka` succeeds is the sharper result: the effect follows the
*declarative form*, not the acidity content — the same declarative/question axis that organises
the probe results (`Analysis/probe/README.md` §2).

The word **"only" remains unverified**: 5 of 23 prompt types were checked. Settling it means
scoring all 23, which needs §6.

Worth noting alongside: `functional_group` is the prompt the probe reads functional group from
*best* (balanced accuracy 0.997), yet its t-SNE is the most thoroughly mixed of the sample.
Linear decodability and cluster geometry are measuring different things, and this is a clean
example of them disagreeing — a linear boundary does not require compact clusters.

### "Most evident at the middle layers, then less distinct at later layers"

**Direction right, location wrong — the peak is at about three-quarters depth, not the middle.**

| model | layer | depth | verdict |
|---|---|---|---|
| 8B | 0 | 0.00 | mixed |
| 8B | 16 | 0.52 | clustered, fragmented |
| 8B | **24** | **0.77** | **cleanest — compact and colour-pure** |
| 8B | 31 | 1.00 | degraded, diffuse centre |
| 70B | 40 | 0.51 | clean |
| 70B | **60** | **0.76** | **cleanest plot in the sample** |

Both models peak near **d ≈ 0.76**, and the two agree closely. The decline is real but appears
confined to the final layer (8B L31); L24 → L31 is where it happens, not L16 → L24. Note this
sits *later* than the probe's saturation depth for declarative prompts, d ≈ 0.47–0.52
(`Analysis/probe/README.md` §1) — linear decodability arrives around half depth, visible cluster
geometry sharpens for another quarter of the stack.

Caveat: only 5 layers per model were rendered, so "peak at L24" means "best of {0, 8, 16, 24,
31}" — layers 20 and 28 were never plotted.

### "At initial layers, clustering reflects the prompt template rather than functional group"

**Plausible and mechanically expected, but not verifiable from this tree.**

The plot that would show it — the `template_index` coloring — was added three days after the
sweep ran and does not exist in `Results_HCC/`. What can be said:

- 8B `pka` L0 shows many small, tight, multi-coloured clumps (silhouette −0.11), which is the
  signature of a variable other than functional group organising the space.
- The mechanism is expected under `aggregation: "last"`: at layer 0 the state is dominated by
  the final token, which is a property of the template. `pka` has **11 templates** with 9
  distinct endings and roughly 5 distinct final tokens (` of`, ` is`, ` about`, ` pKa`,
  ` approximately`), so template-driven structure at L0 is what the design predicts.
- But the L0 plot shows more clumps than 5, 9, or 11, so the grouping variable is **not
  confirmed** to be template alone.

Re-running with `template_index` settles this in one pass.

## 5. Sample size and what would change these conclusions

Everything here comes from 11 plots. The most likely way any of it is wrong:

- **"Only pKa"** — 18 prompt types unchecked at the middle layer. If any of them clusters, the
  claim narrows further.
- **"Peak at d ≈ 0.76"** — 5 rendered layers out of 32/80. The true peak could be anywhere
  between L16 and L31 (8B).
- **Visual verdicts are judgments.** They are recorded in `sampled_plots.csv` so they can be
  disagreed with per-plot, but they are not measurements.

## 6. To settle this properly

Re-run the sweep with the current script — it now produces both missing artifacts:

```bash
sbatch fc_group/run_tsne_analysis.sbatch     # emits metrics/*_metrics.csv and template_index plots
```

That alone gives a 23 × 5 × 2 table of silhouettes and the template plots for §4.3. It does
**not** fix §3 — those metrics are still 2-D. Worth doing at the same time:

1. Add a **high-dimensional** cluster metric to `tsne_functional_groups.py` beside the 2-D one
   (silhouette on the PCA-50 or raw activations, plus k-NN label purity), and keep both columns
   so the 2-D value can be shown to be misleading rather than silently replaced.
2. Render **more layers**. Five points cannot locate a peak; the probe sweep runs every layer
   for a fraction of the cost, and t-SNE at 92–1012 points is cheap.
3. Score against `template_index` as a *label*, not just a coloring — "how much of the geometry
   is template" is the quantity §4.3 needs, and it is one silhouette call.

## Files

| file | contents |
|---|---|
| `sampled_plots.csv` | the 11 plots inspected: model, prompt, layer, silhouette, visual verdict |
