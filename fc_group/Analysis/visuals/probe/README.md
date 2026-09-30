# Reading `probe_depth.png`: the metric, and the two reference lines

`probe_depth.png` plots **pooled out-of-fold balanced accuracy** of a linear probe on each
layer's activations, for five 8B checkpoints. The target is the **coarse** one: 5 heteroatom
families (`halide`, `hydrocarbon`, `nitrogen`, `oxygen`, `sulfur`). Cross-validation is
**leave-one-functional-group-out** — 19 folds, one per fine functional group — so every score
is measured on a group the probe never trained on.

## What "balanced accuracy" is, and how it differs from accuracy

**Balanced accuracy is the mean of the per-class recalls.** Plain accuracy is the fraction of
all test prompts called correctly. Plain accuracy weights each family by how many prompts it
has; balanced accuracy gives every family one vote regardless of size.
`balanced_accuracy_score` computes it at `fc_group/functional_group_probe.py:854`.

```
balanced = mean over families of (that family's prompts correct / that family's prompts)
plain    = all prompts correct / all prompts
```

The two coincide only on a balanced dataset, and this one is not: the 92 molecules split
oxygen 28 / nitrogen 24 / sulfur 20 / halide 16 / hydrocarbon 4, a 7× spread. The choice of
metric is a real choice here, not a formality.

**It does change the numbers, always in the same direction:** plain accuracy runs *below*
balanced accuracy early and converges to it by the final layer.

| model | L0 bal / plain | L8 | L16 | L31 |
|---|---|---|---|---|
| Base | 0.496 / 0.451 | 0.624 / 0.601 | 0.924 / 0.911 | 0.998 / 0.998 |
| Chem-R | 0.532 / 0.494 | 0.680 / 0.668 | 0.870 / 0.861 | 0.980 / 0.980 |
| Chem-F | 0.545 / 0.509 | 0.671 / 0.658 | 0.840 / 0.831 | 0.964 / 0.960 |
| Chem-DFM | 0.568 / 0.524 | 0.714 / 0.699 | 0.892 / 0.882 | 0.955 / 0.950 |
| Reason | 0.531 / 0.491 | 0.631 / 0.609 | 0.868 / 0.850 | 0.958 / 0.953 |

The largest gap anywhere is 0.045 (Base, layer 0). Nothing in the depth story depends on which
metric is plotted — the curves would sit a few points lower at the left and land in the same
place.

Why plain is the *lower* one is worth seeing, since the usual intuition runs the other way.
Base at layer 0:

| family | prompts | recall |
|---|---|---|
| halide | 160 | **1.000** |
| nitrogen | 240 | 0.492 |
| oxygen | 280 | 0.261 |
| sulfur | 200 | 0.230 |

At the embedding layer the probe is perfect on halides — fluorine, chlorine, bromine and
iodine are lexically unmistakable in the prompt string — and near chance on everything else.
Halide is one of the *smaller* classes, so giving it an equal vote pulls balanced accuracy
above the support-weighted number. By layer 31 all four recalls are within 0.006 of each other
and the two metrics agree to three decimals.

**The averaged recalls are four, not five.** `none (alkane)` is the sole fine group in the
`hydrocarbon` family, so it is skipped as a fold and never enters the pooled out-of-fold set
— see the `Random guess` section below. What is averaged is `halide` / `nitrogen` / `oxygen` /
`sulfur`.

**Do not read the per-fold columns as a comparison of the two metrics.** In
`Results/.../probe_scores_coarse_group.csv`, `balanced_acc` and `accuracy` are identical in
every row — all 608 of them for Base. That is not redundancy in the data but a consequence of
the design: leave-one-fine-group-out gives each fold a test set of a single coarse family, and
on a one-class test set macro-recall *is* plain accuracy. The two metrics separate only in the
pooled out-of-fold number, which is what the curve plots.

Two grey lines mark what the curve has to beat.

## `Random guess` — dotted, 0.20

`1 / n_classes` over the 5 declared coarse families, written by the probe as
`summary_coarse_group.json: chance` (`fc_group/functional_group_probe.py:994`) and drawn at
that value by `make_figures.f1_probe_depth`, which this figure follows.

**Labelled `Random guess`, not `chance`, on purpose.** "Chance" names no particular strategy,
and different uninformed strategies land on different numbers. 0.20 is specifically the
balanced accuracy of a guesser drawing **uniformly from the five labels the probe can emit**:
recall 0.20 on each true family. It is not the hardest uninformed bar — a constant
"always answer oxygen" classifier, which reads nothing at all, scores recall 1.0 on oxygen and
0.0 on the other three evaluated families, for a balanced accuracy of **0.25**. So a
degenerate classifier beats the line marked here. Naming the strategy is what keeps 0.20 from
looking like *the* floor when it is one of several.

The paper figure (`make_figures.py:102,104`) still draws these two lines with the old
lower-case labels, so the visuals and the manuscript float differ on the wording.

**The metric it sits under spans 4 true classes, not 5 — and 0.20 is still the right value
for the guesser just named.**
`none (alkane)` is the only fine group in the `hydrocarbon` family, so holding it out leaves no
hydrocarbon in training; it is skipped as a fold, and its 4 molecules never enter the pooled
out-of-fold set (`fc_group/functional_group_probe.py:257-266` prints a notice when it skips
one). The pooled evaluation is 880 rows (88 molecules × 10 templates) over `halide` / `nitrogen` /
`oxygen` / `sulfur` only. The note on `N_MOLECULES`
(`fc_group/Analysis/probe/analyze_sweep.py:60-64`) records the same thing.

That makes 0.25 a tempting correction — the chance level of a 4-class problem — but it is
the wrong one *for this classifier*. Hydrocarbon is absent from the out-of-fold **labels**, not
from the **output space**: every fold trains on the alkane molecules, so the probe keeps all
five output units and uses them. `probe/data/confusion_errors.csv` records **1,095 out-of-fold
prompts predicted `hydrocarbon`** across models and layers, against zero where it is the true
class. A guesser drawing uniformly from the five labels this probe can actually emit scores
recall 0.20 on each of the four evaluated families, and the mean of four 0.20s is **0.20**.

0.25 would be right only for a hypothetical guesser restricted to the four classes present in
the eval set, which is not what is being measured. Nothing in the paper turns on the
distinction — every curve clears both values at layer 0 and ends near 1.0 — but the dashed
line below is the reference that actually binds.

## `Character n-gram baseline` — dashed, 0.475

The same classification task, same folds, same label space — but with **no model in it at
all**. Character n-grams of the raw prompt string are fed to a logistic regression
(`run_surface_baseline`, `fc_group/functional_group_probe.py:634`):

```python
TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2)
LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
```

**Why it exists.** Every prompt template embeds `{iupac_name} ({formula})`, so
`Propan-1-ol (CH3CH2CH2OH)` hands a classifier the substrings `-ol` and `OH` directly. Some
part of "the probe can read functional-group identity off layer 24" is just orthography
spelled out in the prompt. This line is how much — **whatever sits below it was never
evidence about the model.**

The value is a mean over the 19 folds, and that mean hides a sharply bimodal result: **7
folds score 1.0 and 7 score 0.0**, with only 5 in between.

| | folds |
|---|---|
| 1.0 | amine, carboxylic acid, ester, imine, ketone, sulfone, sulfoxide |
| 0.0 | aldehyde, alkyl bromide, alkyl iodide, amide, nitrile, nitro, thiol |

So the surface string does not "half-solve" the task; it solves some families outright from
spelling and tells you nothing at all about others. A curve passing 0.475 is not uniformly
0.475-worth of orthography — it is carrying the seven groups the string cannot reach.

The baseline is identical (`0.475`, 19 folds) for all five models, as it must be: the n-gram
classifier never sees a checkpoint. That is also why averaging it across models in
`make_figures.py` is safe.

## One caveat on comparing the two

The curve is a **pooled** out-of-fold balanced accuracy over 4 coarse classes; the baseline is
a **mean over 19 single-group folds**, where each fold's test set holds one family and its
balanced accuracy collapses to that family's recall (the same collapse described at the end of
the metric section above). Both are recall-based and class-balanced, but they average over
different partitions, so the dashed line is a reference level rather than a
strictly paired comparison.

## Sources

Read from committed CSVs only, never `Results/`:

- curves — `fc_group/Analysis/probe/data/layer_curves.csv`
- baseline — `fc_group/Analysis/probe/data/surface_baseline.csv`
- hydrocarbon predictions — `fc_group/Analysis/probe/data/confusion_errors.csv`
- generator — `fc_group/Analysis/paper/make_visuals_probe.py`

**One exception, flagged because it breaks the rule above.** The plain-accuracy column of the
table in the metric section is *not* in any committed CSV — `layer_curves.csv` carries
`balanced_acc` only. It was pooled from the per-fold `accuracy` and `n_test` columns of
`Results/functional_group_probe/<model>/functional_group/data/probe_scores_coarse_group.csv`,
which is gitignored, so those numbers cannot be re-derived from the repo alone. The balanced
column beside it was pooled the same way and reproduces `layer_curves.csv` exactly at all 20
model × layer points checked, which is what establishes the reconstruction is right.
