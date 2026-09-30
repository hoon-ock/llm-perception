# Where the probe puts its probability on the ambiguous groups

```bash
python fc_group/Analysis/paper/make_visuals_ambiguity.py [--layer 31]
```

`f1_probe_depth.pdf`'s lower panel counts **argmax mistakes**. A model can be right on every
prompt and still differ enormously in how it splits probability between two defensible
answers, and a count cannot see that. These three figures draw the mass itself.

Four groups carry oxygen but are named for another heteroatom — `amide` and `nitro` are
labelled `nitrogen`, `sulfone` and `sulfoxide` `sulfur`. Under leave-one-group-out nothing in
the chemistry picks the home family over oxygen. Only IUPAC does. A model holding both
readings is not confused; it is reflecting real underdetermination.

`mass_split_L31.png` is the only figure here. **Two caveats that used to sit in grey type
along its bottom edge now live in this file instead, and they are load-bearing** -- see
*The confound* below before reading any cross-model ordering off the bars.

**`mass_split_L31.png`** — raw mean out-of-fold mass, split `p(home) / p(oxygen) / p(other three)`,
with each group's condensed formula on the left and its **accuracy** on the right.
Base's bars are nearly solid blue: on `amide` it puts **0.823** on nitrogen against **0.156**
on oxygen, where chem-faithful splits **0.488 / 0.464**.

The accuracy column is the point of the figure, not an ornament. It is the probe's fold
accuracy: one decision per prompt, over that group's molecules x 10 templates, scored as an
argmax over the five families. `amide` has 4 molecules, so its denominator is 40. On `amide`:

| | base | chem-r | chem-faithful | chemdfm | reason |
|---|---|---|---|---|---|
| accuracy | **40/40** | 36/40 | 22/40 | 9/40 | 15/40 |
| | **1.000** | 0.900 | 0.550 | 0.225 | 0.375 |

Base is perfect on **all nine** groups while holding only 0.156 on oxygen for amide, and the
tuned models give up accuracy almost exclusively on the convention-labelled groups — their
controls stay at 0.963–1.000. Accuracy reports a solved task precisely where the label is a
naming convention rather than a fact about the molecule, and scores a model *down* for
choosing the chemically defensible answer.

The five control groups are drawn as **two averaged rows**, one per family. That is a mean of
the masses; `delta` instead averages the ratio `p(O)/[p(O)+p(home)]`, and a mean of ratios is
not the ratio of means. At layer 31 the two differ by at most 0.021 (chem-r, sulfur) — too
small to change anything visible, but the control bar is a picture of the control level, not a
redraw of `delta`'s denominator.

## The correction the figure does not draw

Measuring each group against *its own oxygen-free family controls* (`amine|imine|nitrile` for
nitrogen, `thiol|thioether` for sulfur) gives a different answer, and the difference is the
finding. On raw mass base looks like the one model that refuses the ambiguity. On the
corrected measure it is mid-pack:

| amide, layer 31 | base | chem-r | chem-faithful | chemdfm | reason |
|---|---|---|---|---|---|
| raw `r = p(O)/[p(O)+p(home)]` | 0.159 | 0.345 | 0.487 | 0.569 | 0.532 |
| its controls | 0.012 | 0.226 | 0.219 | 0.290 | 0.134 |
| **delta** | **+0.147** | **+0.119** | **+0.269** | **+0.279** | **+0.398** |

Base's controls sit at **0.012** while chem-r's sit at **0.226**. Base is not specifically
denying the amide ambiguity — base is sharper *everywhere*, on groups that are not ambiguous
at all. Once that is divided out, base leans oxygen on amide slightly **more** than chem-r
does. That is why `mass_split` draws the control groups below the rule: so the global
sharpness is visible rather than inferred.

These numbers are all in `ambiguity/data/ambiguity_two_readings.csv` as `posterior_r`,
`posterior_r_controls` and `posterior_delta`. `posterior_delta` averages the RATIO
`p(O)/[p(O)+p(home)]` across the controls, which is not the same reduction as the averaged
control BAR on the figure -- see the note above.

## The confound

The probe selects its own L2 strength per model, and softmax sharpness follows it directly.
Modal `best_C` over the 19 folds **at layer 31**:

| base | chem-r | chem-faithful | chemdfm | reason |
|---|---|---|---|---|
| 1e-3 (18/19) | 1e-3 (15/19) | 1e-3 (16/19) | **1e-4** (17/19) | **1e-4** (19/19) |

chemdfm and reason got ten times stronger regularization, which flattens a softmax whether or
not the model is less certain. So:

- raw mass is rankable **only within a matched-C set** — `{base, chem-r, chem-faithful}` is
  one; chemdfm and reason are not comparable to it on raw mass;
- `delta` subtracts each group's own controls, so a merely diffuse model scores zero. That
  removes the first-order effect but is **not** a proof of invariance.

`analyze_ambiguity.py`'s docstring is stricter still and worth heeding: it holds that the
posterior column cannot carry a cross-model ranking at all, and that settling it needs
`ambiguity_metric.py --full --mode projection`, which has not been run. Treat the cross-model
ordering here as suggestive; the **within-model** result — ambiguous groups leaning oxygen
more than their own controls — is the one that is solid, and it is positive on `amide` for all
five models.

A parameter-free second reading exists (`geometric_delta` in `ambiguity_two_readings.csv`:
mean z-cosine to the six oxygen-named groups, no probe and no softmax). It confirms the
within-model effect and does **not** reproduce the posterior ordering across models.

## Sources

Committed CSVs only, never `Results/`:

- mass, accuracy, structures — `fc_group/Analysis/ambiguity/data/ambiguity_family_probability.csv`
  (1440 rows = 5 models × 32 layers × 9 groups, snapshotted by `analyze_ambiguity.py`).
  `accuracy` is copied from `Results/functional_group_probe/*/probe_scores_coarse_group.csv`;
  `structure` from the `functional_group_structure` column of
  `fc_group/functional_group_dataset.csv`, so the notation on the figure has one authority.
- delta, `best_C` — `fc_group/Analysis/ambiguity/data/ambiguity_two_readings.csv`
- generator — `fc_group/Analysis/paper/make_visuals_ambiguity.py`

A caution about the denominator, since it is not 40 independent trials: a molecule's 10
templates are near-duplicates of each other, so `amide` is 4 molecules measured 10 ways.
`probe_scores_coarse_group.csv` also carries `per_molecule_acc`, which pools a molecule's
templates before the argmax and gives one decision per molecule (chem-faithful 0.500,
chemdfm 0.000 on amide). It is the honester unit and far too coarse to plot at n=4, which is
why the per-prompt number is the one shown. The same concern is why `bootstrap_diff`
resamples molecules rather than rows.

Stored probabilities are float16 upstream, so a source row's five families sum to 1.0 only to
within 1e-4. `p_other` is summed from the real columns rather than derived as `1 - p_home -
p_oxygen`, so a bar shows its own true total instead of clamping the discrepancy away.
