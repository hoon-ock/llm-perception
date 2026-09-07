# Functional-group analogy — declarative `functional_group` prompt

Third review pass, after `Analysis/probe/` and `Analysis/tsne/`. Scope is the **declarative
`functional_group` prompt only**; the script takes `--entity-type` if the others are wanted.

> **Two things decide most of what follows.** (1) The activation space is strongly anisotropic
> — raw pairwise cosine runs 0.78–1.00 — so `full_within_mean` cannot be quoted on its own; the
> mean-centered numbers from `Results_HCC/anisotropy_diagnostic/` are the real ones. (2) The
> second-order analogy rests on **four hand-chosen quadruples**, and they disagree in sign, so
> their mean is not a summary of anything.

## 1. The anisotropy control

`Results_HCC/anisotropy_diagnostic/.../summary_all_layers.json` reports the same within/between
statistics twice — as computed, and after subtracting the global mean vector.

| model | L | depth | raw pairwise cos | orig within | orig between | **cent within** | **cent between** |
|---|---|---|---|---|---|---|---|
| 8B | 0 | 0.00 | 0.990 | 0.843 | 0.450 | 0.739 | −0.008 |
| 8B | 16 | 0.52 | 0.943 | 0.857 | 0.538 | 0.695 | −0.042 |
| 8B | 24 | 0.77 | 0.856 | 0.910 | 0.475 | 0.827 | −0.048 |
| 8B | 31 | 1.00 | 0.943 | 0.891 | 0.407 | 0.817 | −0.046 |
| 70B | 0 | 0.00 | 0.996 | 0.825 | 0.342 | 0.760 | −0.033 |
| 70B | 40 | 0.51 | 0.837 | 0.878 | 0.477 | 0.774 | −0.043 |
| 70B | 60 | 0.76 | 0.780 | 0.898 | 0.410 | 0.838 | −0.046 |
| 70B | 79 | 1.00 | 0.933 | 0.874 | 0.400 | 0.787 | −0.043 |

Uncentered, *unrelated* functional groups look 0.34–0.54 similar. Centered, they sit at ≈ −0.04
— zero, as they should. **The structure is real** (centered within-class stays at 0.59–0.84,
far above the ±0.05 / ±0.04 closed-form threshold), but roughly half of every raw cosine here
is the global mean direction, not chemistry. All verdicts below use the centered gap.

A related point about the reference lines: `empirical_null_cross_group`
(`fc_group/functional_group_analogy.py:239`) samples pairs of diff vectors **from different
groups** — the same population as `full_between_mean`. That is why the two agree to three
decimals in every layer of every run. It is a re-estimate of the between-class mean, not an
independent null. The usable reference is `closed_form_null_ci`, which is correctly per-model
(±0.0514 at 4096 dims, ±0.0364 at 8192).

## 2. The four claims

### "As layer depth increases, embeddings show increasingly valid clustering by functional group"

**Not monotone, in either model.** The centered gap:

| model | | | | | | shape | peak |
|---|---|---|---|---|---|---|---|
| 8B (L0/8/16/24/31) | 0.747 | 0.708 | 0.737 | **0.875** | 0.863 | dip then rise then fall | L24, d=0.77 |
| 70B (L0/20/40/60/79) | 0.794 | 0.621 | 0.817 | **0.884** | 0.830 | same | L60, d=0.76 |

Three things break the claim as stated:

- **Layer 0 is already high** (0.747 / 0.794) — higher than everything up to half depth. Depth
  initially makes the geometry *worse*.
- The **dip at quarter depth** is large and present in both models (70B falls to 0.621).
- The **final layer falls back** from the peak in both.

The L0 value should not be read as semantic structure. The molecule name contains the group
name, so diff vectors at layer 0 are largely orthographic — the same confound
`PROBE_NOTES.md` §4 identifies for the probe, where character n-grams alone score 0.975.

The honest version: *the geometry sharpens from quarter depth to about three-quarters depth,
then degrades slightly.* That peak at d ≈ 0.76 is the same depth the t-SNE clustering peaks at
(`Analysis/tsne/README.md` §4), and later than the probe's saturation at d ≈ 0.47–0.52.

### "The 8B is better than the 70B at capturing this"

**Not supported.** Centered gap at matched depth:

| depth | 8B | 70B | better |
|---|---|---|---|
| 0.00 | 0.747 | 0.794 | 70B |
| ~0.26 | 0.708 | 0.621 | 8B |
| ~0.52 | 0.737 | 0.817 | 70B |
| ~0.77 | 0.875 | 0.884 | 70B |
| 1.00 | 0.863 | 0.830 | 8B |

The 8B leads at **2 of 5** depths, and at the peak the two are effectively tied (0.875 vs
0.884). There is one place the 8B is genuinely ahead — the O→N analogy, §3 — but that is a
single quadruple.

### "Within-class similarity increases with depth"

**Same non-monotone shape** — this is the centered-within column, which drives the gap above:
8B 0.739 → 0.671 → 0.695 → **0.827** → 0.817; 70B 0.760 → 0.585 → 0.774 → **0.838** → 0.787.
Both dip at quarter depth before recovering. The L0-to-final direction is upward, but that is
not the same as increasing with depth.

### "Consistency across molecules within a group increases with depth"

**This is the same quantity as the previous claim, unless a different reading is intended.**
`within_class_similarity` *is* cosine between same-group diff vectors at different chain
lengths — i.e. consistency across molecules within a group. Measured that way it inherits the
non-monotone shape above.

Under the other available reading — `within_std`, the spread of within-scores *across* the 19
groups, i.e. how uniform the groups are in their internal coherence — **the claim holds, and it
is the only one of the four that does**:

| model | | | | | | shape |
|---|---|---|---|---|---|---|
| 8B | 0.108 | 0.068 | 0.049 | 0.038 | 0.052 | decreasing, final-layer uptick |
| 70B | 0.110 | 0.077 | 0.053 | 0.044 | 0.040 | **monotone decreasing** |

Worth separating these two in future write-ups: one is "are a group's members alike" and the
other is "are the groups alike in how alike their members are", and only the second is monotone.

## 3. The analogy result is 2-of-4, not a mean

The per-layer mean (8B: 0.118 → 0.273 → 0.150) averages four quadruples that **disagree in
sign**, two consistently positive and two consistently negative. Individually:

**8B**

| quadruple | L0 | L8 | L16 | L24 | L31 | |
|---|---|---|---|---|---|---|
| thioether−thiol ~ ether−alcohol (O↔S) | +0.488 | +0.643 | +0.704 | +0.659 | +0.643 | **works, stable** |
| imine−aldehyde ~ amine−alcohol (O→N) | +0.044 | +0.216 | +0.444 | **+0.683** | +0.609 | **works, grows with depth** |
| alkyl bromide−chloride ~ iodide−bromide | +0.078 | +0.107 | +0.094 | −0.250 | −0.349 | fails, worsens |
| sulfoxide−thioether ~ sulfone−sulfoxide | −0.140 | −0.023 | −0.151 | −0.141 | −0.301 | fails throughout |

**70B**

| quadruple | L0 | L20 | L40 | L60 | L79 | |
|---|---|---|---|---|---|---|
| O↔S | +0.749 | +0.541 | +0.719 | +0.732 | +0.700 | works, flat |
| O→N | −0.163 | +0.385 | +0.472 | +0.438 | +0.465 | works after L0 |
| halogens | −0.167 | −0.156 | −0.214 | −0.258 | −0.216 | fails at every layer |
| sulfur ladder | +0.267 | −0.037 | +0.009 | +0.081 | −0.085 | noise |

Only the **O→N substitution in the 8B** actually matches the "improves with depth" story
(+0.044 → +0.683). The O↔S analogy is the strongest but is already near its ceiling at layer 0
in the 70B (+0.749), so depth adds nothing to it.

**The halogen failure is probably a data defect, not chemistry.** `PROBE_NOTES.md` §7 records
that F, Br and I are written as SMILES in `functional_group_dataset.csv` while Cl is written as
a condensed formula. So `alkyl bromide − alkyl chloride` is partly a *notation* difference while
`alkyl iodide − alkyl bromide` is not — the two vectors encode different things, and
anti-alignment is what that predicts. This is the same confound that made `alkyl chloride` the
sole failing group in the probe's leave-one-group-out. Fix `extract_properties.py` before
concluding anything about halogen analogies.

## 4. Limits

- **5 layers per model** (8B 0/8/16/24/31; 70B 0/20/40/60/79). Every "as depth increases"
  statement, mine included, is read off 5 points. The dip at quarter depth and the peak at
  d ≈ 0.76 are located only to within one sampled layer.
- **n = 4 analogy quadruples**, hand-chosen at `functional_group_analogy.py:196`. With two
  positive and two negative, no aggregate over them supports a model-vs-model claim.
- **Layer 0 is orthographic, not semantic** — see §2.
- **The diff-vector space is nearly 1-D**: PC1 explains 0.48–0.62 of variance at every layer in
  both models, so "clustering" here lives largely along a single direction.

## 5. Regenerating

```bash
python fc_group/Analysis/analogy/analyze_analogy.py                    # functional_group
python fc_group/Analysis/analogy/analyze_analogy.py --entity-type pka  # any other prompt
```

Needs both `Results_HCC/functional_group_analogy/` and `Results_HCC/anisotropy_diagnostic/`;
the script asserts their layer sets match rather than silently emitting empty centered columns.

| file in `data/` | contents |
|---|---|
| `layer_trend.csv` | per model × layer: raw pairwise cosine, original and centered within/between/gap, within_std, PC1 ratio, analogy mean, signed significance counts |
| `analogy_quadruples.csv` | all 4 quadruples × layer × model, individual cosines, direction-aware significance flag |
| `within_by_group.csv` | per-group within-class score by layer — which groups drive the mean |
