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

12 plots inspected directly, all under the `functional_group` coloring (recorded in
`sampled_plots.csv` with silhouette and verdict). "Clustered" means colour-coherent regions are
visible; "mixed" means every clump contains many colours. This is a sample, not a census — 6 of
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

There is a second, independent reason they cannot be compared **across prompts**: point count
and perplexity differ. `perplexity = min(30, max(5, n // 4))`, and the two bare prompts have one
template (92 points, perplexity 23) against 644-1012 points and perplexity 30 everywhere else.
Even a well-behaved metric would not be comparable across that boundary.

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
scoring all 23, which needs §7.

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

**Why this shape?** See §5 — the layer where each model is most
isotropic coincides exactly with its t-SNE peak, and the final-layer degradation is an
anisotropy spike.

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

## 5. Why the peak sits at d ≈ 0.76, and why the last layer degrades

Ordered by how well the existing data supports each. None of this required a re-run — the
`anisotropy_diagnostic` tree covers the same prompts and the same layers.

### Supported by measurement

**(a) The representation is most spread out exactly where the clusters are cleanest.**
Mean pairwise cosine among activations (lower = more spread) bottoms out at the t-SNE peak
layer in both models, then jumps back at the final layer:

| prompt | | | | | | most isotropic | t-SNE peak |
|---|---|---|---|---|---|---|---|
| 8B `pka` | 0.992 | 0.984 | 0.928 | **0.719** | 0.945 | L24 | L24 |
| 8B `functional_group` | 0.990 | 0.965 | 0.943 | **0.856** | 0.943 | L24 | — |
| 70B `pka` | 0.997 | 0.953 | 0.784 | **0.560** | 0.886 | L60 | L60 |
| 70B `functional_group` | 0.996 | 0.954 | 0.837 | **0.780** | 0.933 | L60 | — |

It also predicts *which prompts* cluster. At L16/L40, where most of the verdicts in
`sampled_plots.csv` were made, the two prompts that cluster are the two most isotropic, with a
clear gap to the rest — in both models:

| 8B L16 | anisotropy | diff-vector PC1 | verdict |
|---|---|---|---|
| `pka` | **0.928** | **0.478** | clustered |
| `pkah` | **0.931** | **0.483** | clustered |
| `functional_group` | 0.943 | 0.608 | mixed |
| `tpsa` | 0.969 | 0.612 | mixed |
| `avg_carbon_oxidation_state` | 0.972 | 0.604 | mixed |
| `pka question` | 0.980 | 0.529 | mixed |

The 70B at L40 gives the same order (`pka` 0.784, `pkah` 0.789, then 0.837–0.974).

The likely mechanism is **effective dimensionality**, not anisotropy as such. A caveat matters
here: `perform_pca` (`tsne_functional_groups.py:91`) uses sklearn `PCA`, which **centers**, so
t-SNE never sees the uncentered geometry and "the cloud is compressed" cannot be the
explanation directly. What high pairwise cosine indexes is variance concentrating in few
directions — and the independent measure agrees, since the clustering prompts also have the
lowest PC1 share. If 60–80% of variance sits in one to three directions, PCA-50 → t-SNE is
resolving 20 classes out of a near-degenerate subspace. Note the PC1 figure is computed over
*diff vectors*, not the raw activations t-SNE consumes, so it is indicative rather than
conclusive.

**(b) The final-layer degradation is the anisotropy spike.** `pka` goes 0.719 → 0.945 (8B) and
0.560 → 0.886 (70B) on the last step alone — most of the way back to the layer-0 value. This is
the familiar final-LayerNorm / rogue-dimension effect, and it is the same signature as the
probe's late-layer decline for declarative prompts (`Analysis/probe/README.md` §3).

**(c) Linear separability and metric clustering are different properties.** This dissolves the
apparent tension with the probe saturating earlier (d ≈ 0.47–0.52) rather than leaving it as a
puzzle. A probe needs only a separating hyperplane; t-SNE needs neighbourhoods dominated by one
class. The two come apart measurably here: `functional_group` has the **highest** centered
within−between gap of any prompt (0.875 / 0.884, `Analysis/analogy/`) and is simultaneously the
most visually mixed. Class information is present and linearly readable; it is simply not the
dominant source of variance.

### Plausible, not measurable from these outputs

**(d) Template variance washes out with depth.** At L0 the state is dominated by the final
token, which is a template property, and the L0 plots show many small multi-coloured clumps.
Prediction: silhouette against `template_index` decays toward 0 by d ≈ 0.76. Needs the
`template_index` coloring, which postdates the sweep (§1).

**(e) Next-token specialisation.** For `pka` the next token is a pKa *number*, not a group name,
so at the final layer points might reorganise by predicted pKa — merging groups that share a
pKa, splitting same-group molecules whose chain length shifts it. **Tested and inconclusive:**
the `pka`-coloured 8B L31 plot shows structure, but the colouring is continuous so no silhouette
is computed for it and there is no like-for-like comparison against the categorical plot. It is
not obviously more pKa-ordered than group-ordered. (Incidentally, the molecules with no pKa
value do form a coherent region.) Settling this needs a numeric measure, e.g. a k-NN regression
score on pKa per layer.

**(f) Depth-of-processing.** Abstract features peak in late-middle layers; the last layers
belong to the output distribution. Consistent with everything above, but it restates the
observation rather than evidencing it.

### Artifacts that could produce the pattern on their own

- **5 sampled layers.** "d ≈ 0.76" means "best of {0, 8, 16, 24, 31}". The true peak could sit
  anywhere in roughly 0.6–0.9.
- **t-SNE settings are not comparable across prompts.** `perplexity = min(30, max(5, n // 4))`,
  and point counts differ by an order of magnitude: `molecule_name_bare` and
  `molecule_formula_bare` have **1 template → 92 points** (perplexity 23), everything else has
  7–11 templates → 644–1012 points (perplexity 30). Any cross-prompt comparison spanning that
  boundary compares two different t-SNE configurations.
- **`random_state=42`** — a single draw, with no seed-sensitivity check. Apparent island
  structure can move between seeds.
- **The verdicts are visual judgments**, recorded per plot in `sampled_plots.csv` so they can be
  disputed individually.

A note on what *doesn't* explain it: low anisotropy alone is not sufficient.
`molecule_formula_bare` is the most isotropic prompt at 8B L24 (0.490) and still shows no
functional-group clustering — it has the weakest class signal of any prompt (probe best 0.688).
Both ingredients are needed. That observation is confounded by the 92-point difference above,
so treat it as suggestive.

## 6. Sample size and what would change these conclusions

Everything here comes from 11 plots. The most likely way any of it is wrong:

- **"Only pKa"** — 18 prompt types unchecked at the middle layer. If any of them clusters, the
  claim narrows further.
- **"Peak at d ≈ 0.76"** — 5 rendered layers out of 32/80. The true peak could be anywhere
  between L16 and L31 (8B).
- **Visual verdicts are judgments.** They are recorded in `sampled_plots.csv` so they can be
  disagreed with per-plot, but they are not measurements.

## 7. To settle this properly

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
| `sampled_plots.csv` | the 12 plots inspected: model, prompt, layer, colouring, silhouette, visual verdict |
| `analyze_tsne_context.py` + `data/anisotropy_by_prompt_layer.csv` | the anisotropy / effective-dimensionality join behind §5, for all 23 prompts x 5 layers x 2 models |
