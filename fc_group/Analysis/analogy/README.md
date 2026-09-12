# Functional-group analogy — declarative `functional_group` prompt

Third review pass, after `Analysis/probe/` and `Analysis/tsne/`. Scope is the **declarative
`functional_group` prompt only**; the script takes `--entity-type` if the others are wanted.

> **Two things decide most of what follows.** (1) The activation space is strongly anisotropic
> — raw pairwise cosine runs 0.78–1.00 — so `full_within_mean` cannot be quoted on its own; the
> mean-centered numbers from `Results/anisotropy_diagnostic/` are the real ones. (2) The
> second-order analogy rests on **four hand-chosen quadruples**, and they disagree in sign, so
> their mean is not a summary of anything.

## 1. The anisotropy control

`Results/anisotropy_diagnostic/.../summary_all_layers.json` reports the same within/between
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

> **Superseded in part.** This section analyses the original **four** quadruples. The set is
> now **seven** — both halogen quadruples were dropped as notation-confounded, and three
> controls were added (see `functional_group_analogy_carbon_matched.py:259` for the
> selection criteria). The HCC sweep behind the tables below has not been rerun, so these
> numbers still describe the old set. What the local 8B layers already show:
>
> | quadruple | L0 | L16 | L31 | |
> |---|---|---|---|---|
> | thioether−thiol ~ ether−alcohol | +0.382 | +0.740 | +0.599 | O↔S, notation-crossed |
> | thiol−alcohol ~ thioether−ether | −0.176 | +0.447 | **+0.588** | **its notation-clean control** |
> | imine−aldehyde ~ amine−alcohol | +0.056 | +0.107 | +0.556 | O→N, notation-crossed |
> | amide−carboxylic acid ~ amine−alcohol | +0.141 | +0.228 | **+0.616** | **its notation-clean control** |
> | carboxylic acid−alcohol ~ amide−amine | +0.713 | +0.861 | +0.629 | matched legs |
> | ester−carboxylic acid ~ ketone−aldehyde | +0.269 | +0.043 | **+0.129** | **mismatched legs, negative control** |
>
> Two things follow. **The O↔S and O→N results survive their notation controls** — the clean
> and crossed versions converge by the final layer (+0.588 vs +0.599, +0.616 vs +0.556),
> though they disagree sharply at layer 0 where the geometry is orthographic anyway. And
> **leg symmetry predicts success**: the two quadruples built from the same carbonyl groups
> differ only in whether their legs conserve hbd/hba, and score +0.629 against +0.129.


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
sole failing group in the probe's leave-one-group-out. `extract_properties.py` has since been
fixed — `smiles` and `condensed_formula` are now populated for every molecule from one set of
templates — but the fix cannot reach these results, because the notation is part of the prompt
text baked into the activations. Treat "the halogen analogy fails" as untested until the
prompts are re-extracted from the clean columns; the prediction is that it stops being
anti-aligned.

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

Needs both `Results/functional_group_analogy/` and `Results/anisotropy_diagnostic/`;
the script asserts their layer sets match rather than silently emitting empty centered columns.

| file in `data/` | contents |
|---|---|
| `layer_trend.csv` | per model × layer: raw pairwise cosine, original and centered within/between/gap, within_std, PC1 ratio, analogy mean, signed significance counts |
| `analogy_quadruples.csv` | all 4 quadruples × layer × model, individual cosines, direction-aware significance flag |
| `within_by_group.csv` | per-group within-class score by layer — which groups drive the mean |

## 6. The retrieval test

§3 scores each quadruple by the **cosine between two offset vectors**. That is not what the
word2vec analogy result measures. That result is *retrieval* — `king − man + woman` lands
nearest to `queen` — and offset cosines in those spaces are well short of 1. Scoring by
cosine alone measures the strictly harder quantity, and it changes the verdict on half the
quadruples.

`fc_group/analogy_retrieval.py` runs the retrieval version over the diff vectors the
analogy script already saved to `data/diff_vectors_layer_{L}.npz`, so it needs no rerun and no GPU. For
each quadruple it tests all four corner predictions (`b1 = b2 + (a1 − a2)`, and the three
rotations) at each chain length: 4 quadruples × 4 corners × 4 chain lengths = 64 trials per
layer, against 19 candidate groups.

**Retrieval sees 5 analogies where §3's cosine sees 7.** Retrieval is blind to how four
groups are paired — every corner target is two of them minus the third, and the exclusion
set is always the other three, so re-pairing `{w,x,y,z}` poses the same four questions with
the sources relabelled. `cos(a1−a2, b1−b2)` *does* depend on the pairing, which is why
`ANALOGY_QUADRUPLES` carries both pairings of two group sets as notation controls, and why
`analogy_retrieval.py` collapses them (`distinct_by_group_set`). Counting both would double
those group sets' weight in every pooled figure.

`functional_group`, carbon-matched, 80 trials per layer, random-baseline hit@1 = 6.2%:

| depth | 0.00 | 0.25 | 0.51 | 0.76 | 1.00 |
|---|---|---|---|---|---|
| 8B hit@1 | 42.5% | 45.0% | 71.2% | **75.0%** | 73.8% |
| 70B hit@1 | 46.2% | 48.8% | **82.5%** | 81.2% | 81.2% |

**The peak-then-fall shape reported for the four-quadruple set was the halogen quadruple.**
With the halogens gone, both models rise and then plateau across the whole second half
rather than falling back at the end. The earlier "falls back at
the final layer" reading, which appeared to corroborate §2's centered gap, was one
notation-confounded quadruple collapsing and dragging the pooled mean with it.

O↔S retrieves 16/16 from the midpoint on. **The sulfur ladder, which is negative by cosine
in §3, retrieves at rank 1–2 on every trial at 8B layer 31** — the clearest case of cosine
and retrieval disagreeing.

Two caveats that have to travel with these numbers:

- **Excluding the source terms is doing real work.** Standard word2vec practice excludes the
  three input terms from the candidate pool, and that convention is also why those results
  flatter themselves. Here the unexcluded nearest neighbour is simply the source group in
  **29–58%** of trials — the offset is real, but small next to the distance between groups.
  `rank_incl` and `degenerate_top1_rate` are emitted in every output for that reason.
- **The halogen failure is asymmetric by corner.** The two corners predicting `alkyl bromide`
  sit at rank 11–17 while predicting chloride and iodide is rank 1 at every chain length —
  which is what the SMILES-vs-condensed-formula confound in §3 predicts, and is invisible in
  the cosine view.

```bash
python fc_group/analogy_retrieval.py                    # functional_group
python fc_group/analogy_retrieval.py --entity-type pka  # any other prompt
python fc_group/analogy_retrieval.py --no-plots         # CSVs only
```

Outputs land under `Results/`, in the same shape every other experiment here uses —
`fc_group/Results/functional_group_analogy_retrieval/{model}/{entity_type}/`, figures at the
leaf and CSVs under `data/`. Model and entity are in the path, so they are not repeated in
the filenames, and each CSV holds exactly one model.

| file in `data/` | contents |
|---|---|
| `retrieval_trials.csv` | one row per trial: ranks with and without the source terms excluded, hit@1/@2/@3, cosine to the target, the unexcluded top-1 |
| `retrieval_layer_trend.csv` | per layer × mode: hit@1/@2/@3, mean and median rank, MRR, degenerate rate, and the matching `random_*` baselines |
| `retrieval_by_quadruple.csv` | the same aggregates split by quadruple |
| `retrieval_by_group.csv` | the same aggregates split by the functional group being predicted, with `n_trials` — groups are **unevenly sampled**, since one that answers two different corners gets twice the trials |

Seven figures per model:

| figure | shows |
|---|---|
| `retrieval_accuracy_pooled.png` | hit@1/@2/@3 pooled over the five distinct quadruples, on one axis so the three can be compared |
| `retrieval_accuracy_hit{1,2,3}_by_quadruple.png` | one file per k: four quadruple lines against that k's own baseline |
| `retrieval_rank_by_quadruple.png` | mean rank per quadruple. **No pooled mean line** — the four quadruples span six rank positions at some depths, so their average describes none of them |
| `retrieval_rank_by_group.png` | mean rank per functional group, plus the overall mean |
| `retrieval_rank_heatmap.png` | rank of the correct group for every corner prediction × chain length × layer |

**"Random baseline", not "chance".** Ordering the candidates at random puts the answer in
the top *k* with probability `k/n`, so each hit@k has its own baseline — 6.1% / 12.1% /
18.2%, and 8.75 for mean rank. They are not interchangeable: reading hit@3 against the
hit@1 line would overstate it threefold. The baselines are averaged over the *actual*
candidate pools rather than assumed constant, because pool size is 16 or 17 depending on
whether the quadruple reuses a group.

A quadruple keeps one colour across all four by-quadruple figures, so they can be read
side by side.

Every series in these plots is a solid line distinguished by colour and marker. Dash
patterns are deliberately not used to encode identity — at these font sizes `--` and `-.`
are indistinguishable from `-` in a legend swatch, so a dash-encoded series ends up
mislabelled in practice even when the code is right. The self-checks in §7 have no opinion
on this — it is a figure property, not a data one — so it is asserted separately by
building each figure and comparing every legend handle's linestyle, marker and colour
against the line it labels, with solid the only non-baseline style permitted.

## 7. Checking it

`analogy_retrieval.py` used to exit 0 whenever it did not crash, and both bugs found while
writing it were silent-wrong-answer bugs: a heatmap row key that collided for the two
quadruples reusing a group (14 rows drawn for 16 trials), and an exclusion set that deleted
the answer for a quadruple. Neither showed up in the printed table.

It now **self-checks before it writes**, so a run that fails leaves no plausible-looking but
wrong results tree behind — there is no separate verifier to remember to run:

```bash
python fc_group/analogy_retrieval.py                    # checks, then writes
python fc_group/analogy_retrieval.py --skip-checks      # escape hatch, not a normal flag
```

It asserts trial counts; that no two quadruples share a group set (which is what the first
seven-quadruple run needed and did not have); that no quadruple was silently skipped; that every trial is
uniquely addressable by (quadruple, source, predicted, chain length); that candidate-pool
sizes match the exclusion rule (`n_groups − 2` for the quadruple built from three distinct
groups, `n_groups − 3` otherwise); that ranks and hit flags agree row by row; that the
per-group rows partition the trials; that the random baselines equal `k/n` over the actual
pools; and that every reported aggregate matches the trials it summarizes.

That last one recomputes the aggregates with `recompute()` rather than `summarize()`. The
near-duplication is deliberate — folding them together would make the check vacuous — but
it is worth being clear about what it buys: it catches the wrong rows reaching `summarize`,
rows mutated between summarizing and writing, and a typo in one copy. It does not catch a
concept that is wrong in both.

It also holds a fixed regression on 8B/`functional_group` — hit@1 34/57/59 of 80 at layers
0/16/31, mean rank 3.79/2.17/1.35, and O↔S at 16/16. Those track the contents of
`Results/` **and** `ANALOGY_QUADRUPLES`, so after a legitimate re-extraction or a change
to the quadruple set the constants at the top of the file are what should change.
