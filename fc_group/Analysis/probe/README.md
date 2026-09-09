# Functional-group probe — HCC sweep findings

What the full sweep showed, and which of the obvious readings of it survive contact with
the numbers.

Companion to `fc_group/PROBE_NOTES.md`, which explains **how the probe works** and reports the
local smoke run. This file reports the **HCC sweep**: 2 models × 23 entity types × all layers.

> **Scope.** Everything below is the `coarse_group` tag — target `coarse` (5 heteroatom
> families), split `group` (leave-one-functional-group-out). That is the deliberate choice:
> per `PROBE_NOTES.md` §4 and §6, leave-one-group-out is the only split with dynamic range.
> The other splits are saturated by orthography and cannot separate anything.
>
> Models: `meta-llama/Llama-3.1-8B` (32 layers) and `meta-llama/Llama-3.1-70B` (80 layers).
> Source tree: `fc_group/Results_HCC/functional_group_probe/`, which is **gitignored** — so
> the CSVs in `data/` are the durable record, not the tree they were derived from.

---

## 1. Where the models gain depth-wise

The 23 entity types come in three families (`config_extract_activation.yaml`):

- **declarative** (10) — `"Propan-1-ol (CH3CH2CH2OH) contains the functional group "`
- **question** (10) — `"What functional group does Propan-1-ol (CH3CH2CH2OH) contain?"`
- **bare** (3) — `molecule`, `molecule_name_bare`, `molecule_formula_bare`; no question asked

Both models gain most of their accuracy near half depth — but only the **declarative** and
**bare** prompts do. Pooling all 23 entity types blends two distributions that do not describe
each other:

| model | family | n | largest single-layer jump | first reaching 90% of best | median depth |
|---|---|---|---|---|---|
| 8B | **declarative** | 10 | modal L16, 4/10 | **L16–L19**, modal L16 (7/10) | **0.52** |
| 8B | bare | 3 | scattered (L1, L2, L19) | L16–L21 | 0.65 |
| 8B | question | 10 | modal L7, 3/10 | L21–L30, modal L29 (5/10) | **0.94** |
| 70B | **declarative** | 10 | modal **L37, 6/10** | L21–L51, modal L37 (4/10) | **0.47** |
| 70B | bare | 3 | modal L1, 2/3 | L11–L40 | 0.25 |
| 70B | question | 10 | modal **L10, 6/10** | L43–L79, modal L78 (3/10) | **0.96** |

The declarative story is the clean one, and it is **the same in both models**: functional-group
identity is 90% readable by **half depth** — median d = 0.52 (8B) and 0.47 (70B). In the 8B
every one of the 10 declarative prompts saturates in the four-layer window **L16–L19**, seven of
them at L16 exactly. The pooled "8/23" in an earlier version of this table understated that
badly, because question prompts saturate at d ≈ 0.94–0.96 and dragged the pooled distribution
late.

The 70B also has a genuine **single-layer signature**: for 6 of 10 declarative prompts the
largest layer-to-layer gain in the whole 80-layer curve falls on exactly **L37**
(`functional_group_structure` gains 0.217 there; `functional_group` goes 0.790 → 0.900). The
8B's L16 is weaker as a jump (4/10) but is the dominant *saturation* point (7/10).

Question prompts have a different shape entirely: an early jump — modal L10 in the 70B, 6 of 10
— followed by a long slow climb that only reaches 90% of its peak at the top of the stack. The
mid-depth layer still matters to them (4 of 10 have their largest jump at L37 too), but it is
not where they finish.

> **Read `bare` with caution.** n = 3, and the statistic is relative to each curve's own
> floor-to-peak span, so a flat curve clears the 90% bar early without that meaning much. For
> 8B `molecule_formula_bare` that span is only **0.059** (0.629 → 0.688), against a declarative
> median of 0.527. Its saturation layer is essentially noise.

## 2. Prompt family: declarative vs question

Declarative prompts are markedly better probes of functional-group identity:

| model | declarative (n=10) | question (n=10) | bare (n=3) |
|---|---|---|---|
| 8B | 0.956 | 0.860 | 0.775 |
| 70B | 0.960 | 0.833 | 0.768 |

The same split also carries the model comparison in §4 — the 8B's advantage over the 70B is
entirely a question-prompt effect. It is a **partial** explanation of the late-layer behaviour
in §3, and it is not the whole story there.

## 3. Late-layer behaviour

Two distinct things happen near the top of the stack, and they are separate measurements that
do not always co-occur.

**Peak-to-last slide** (`drop_best_minus_last`) — how far below its own peak a curve ends.
Largest: 70B `boiling_point_c` 0.151, 8B `boiling_point_c` 0.144, 8B `tpsa` 0.119.

**Final-layer step** (`final_step`) — the last layer-to-layer change alone. **19 of 46 curves
end on a negative step.** Sharpest: 8B `molecule_formula_bare` −0.073, 70B `boiling_point_c`
−0.055, 8B `tpsa` −0.054, 70B `pkah` −0.045.

That these are different quantities is easiest to see in the cases that disagree: 8B `hbd`
slides 0.000 — it ends at its own peak — yet steps **+0.031** into the final layer, while 70B
`boiling_point_c` does both (slide 0.151, step −0.055). A curve can end at its maximum and
still have moved sharply on the last layer; a curve can slide a long way with barely any final
step. Reading only one of them off a plot will mislead.

### How much prompt family explains

Most of the spread, but not all of it:

| model | family | mean slide | max | mean final step | # peaking at final layer |
|---|---|---|---|---|---|
| 8B | question | **0.003** | 0.013 | +0.003 | 6/10 |
| 8B | declarative | 0.043 | 0.144 | −0.010 | 3/10 |
| 8B | bare | 0.057 | 0.100 | −0.018 | 1/3 |
| 70B | question | **0.001** | 0.014 | +0.013 | 9/10 |
| 70B | declarative | 0.059 | 0.151 | −0.015 | 2/10 |
| 70B | bare | 0.046 | 0.066 | −0.004 | 0/3 |

Question prompts slide roughly **an order of magnitude less** than the other two families. But
"less" is not "never", and the exceptions are worth naming rather than rounding away — five of
the 20 question prompts do not peak at the final layer: 8B `avg_carbon_oxidation_state
question` (−0.013), `functional_group question` (−0.007), `water_solubility question` (−0.006),
`pka question` (−0.003), and 70B `functional_group question` (−0.014). They are small, but they
are not zero.

Family also leaves real **within-family** variation unexplained. Within the 70B's declarative
prompts alone, `boiling_point_c` slides 0.151 while `functional_group` slides 0.000. And
`bare` prompts — which ask nothing at all — decline as much as declarative ones, which no
account based on *what the prompt asks* can explain.

### Mechanism — hypothesis, not result

The final layers specialise for **next-token prediction**. For a declarative prompt the next
token is the answer itself; for a bare name or formula it is the continuation of that string.
In both cases the residual gets committed to a specific token identity, and the broader
functional-group feature degrades as it does. After a `?`, the next token is near-trivial
(whitespace, "The"), so much less of that rotation happens and the feature survives to the
output layer.

This covers all three families, and it predicts that `molecule_formula_bare` — where next-token
prediction is most demanding, continuing a raw SMILES or condensed formula — should have the
sharpest final-layer drop. It does, at −0.073, the largest of all 46 curves.

Consistent with, not demonstrated by, this sweep. The test that would settle it: re-extract
with the aggregation position at a content token rather than the last token
(`aggregation: "last"` in `config_extract_activation.yaml`), and check whether the declarative
and bare declines disappear.

## 4. 8B vs 70B — not a model-level difference

Tested by cluster bootstrap (20,000 draws) resampling the **88 molecules**, each model at its
own best layer. The two models' prediction files are row-for-row alignable — identical
`y_true` sequences — which is what makes the pairing valid; `analyze_sweep.py` asserts it.

**10 of 23 entity types have a CI excluding zero — and they split 6 for the 8B, 4 for the 70B.**

| entity type | family | diff (8B − 70B) | 95% CI |
|---|---|---|---|
| `pkah question` | question | +0.058 | [+0.016, +0.100] |
| `water_solubility question` | question | +0.049 | [+0.006, +0.091] |
| `pka` | declarative | +0.046 | [+0.014, +0.083] |
| `hbd question` | question | +0.041 | [+0.010, +0.071] |
| `functional_group` | declarative | +0.035 | [+0.016, +0.059] |
| `functional_group_structure` | declarative | +0.028 | [+0.012, +0.045] |
| `hbd` | declarative | −0.026 | [−0.043, −0.010] |
| `hba` | declarative | −0.035 | [−0.063, −0.010] |
| `water_solubility` | declarative | −0.052 | [−0.082, −0.023] |
| `boiling_point_c` | declarative | −0.060 | [−0.104, −0.024] |

By family, the pattern is clean:

| family | n | mean (8B − 70B) | sd | t | 8B ahead |
|---|---|---|---|---|---|
| **question** | 10 | **+0.0270** | 0.0223 | **+3.82** | **9/10** |
| declarative | 10 | −0.0037 | 0.0381 | −0.30 | 5/10 |
| bare / molecule | 3 | +0.0070 | 0.0413 | +0.29 | 2/3 |

**The 8B is better on question-phrased prompts and indistinguishable on declarative ones.**
Averaging the two families together produces +0.011 overall, which is a misleading middle
between two opposite-signed effects.

Two caveats, both of which make the question-prompt effect a floor rather than a ceiling:

- Each model sits at **its own best layer**, so the 70B gets best-of-80 draws against the 8B's
  best-of-32. That favours the 70B.
- The resampling unit is the molecule (88), not the row (880–968). The 10–11 template rows per
  molecule are near-duplicates; resampling rows would shrink these intervals to a fraction of
  their honest width.

## 5. The surface baseline is flat — which is what makes §2–§4 usable

`PROBE_NOTES.md` §5 is blunt about this: the permutation null validates the machinery, only the
surface baseline validates the science. Character n-grams over the raw prompt text score
**0.482–0.520** across all 15 entity types where it was measured, with no ordering that tracks
the probe ranking (`functional_group` 0.512, `pka` 0.519, `tpsa` 0.496, `hbd` 0.482).

So the spread between prompts is **not** an orthography artifact — the n-gram baseline cannot
tell these prompts apart at all, because they differ only in the wrapper around an identical
molecule name and formula. The cross-prompt comparisons above are measuring representation.

**Gap, since closed for future runs.** The sweep that produced these results ran without
`--surface-baseline`, so it exists for 15 of 46 tasks and **none of the 70B ones**.
`fc_group/scripts/02_probe.sbatch` now passes `--surface-baseline` on every task, so a re-run
will have it throughout — but that cannot be applied retroactively to the numbers below. Recovering it costs ~7 s per entity type at one layer:

```bash
python fc_group/functional_group_probe.py --entity-type <name> --model-name <model> \
  --layers 31 --num-null-samples 0 --surface-baseline --output-dir <dir>
```

## 6. Readings that the data does not support

Recorded so they do not get repeated. The third row is a correction of a correction: the
original reading was directionally right, and this document's first version overshot by
calling the decline "confined to declarative prompts". §3 has the accurate version.

| claim | verdict | evidence |
|---|---|---|
| "Sharp increase at middle layers, both models" | **holds** — and is stronger than stated | modal jump L16 (d=0.52) / L37 (d=0.47); 10/23 on one layer in the 70B |
| "`functional_group` climbs steadily and holds to the end" | **holds** | 8B 0.997 @L27 → 0.992 @L31; 70B peaks at L79 |
| "Other prompts rise then fall late" | **holds, directionally** | true for declarative *and* bare prompts; question prompts slide ~10x less but 5/20 still decline (§3) |
| "`functional_group` / `pka` are good, others noticeably worse" | **not as stated** | `tpsa` 0.970 and `avg_carbon_oxidation_state` 0.969 sit with `pka` 0.980; the cut is declarative vs question |
| "The 8B shows more spread across prompt templates" | **inverted** | sd of best over 23 types: 8B 0.074 vs 70B 0.085; excluding bare prompts 0.058 vs 0.073 |
| "The 8B is better than the 70B overall" | **false as a model-level claim** | 6/4 split among significant entities; +0.027 on question prompts, −0.004 on declarative (§4) |

## 7. Defects in the outputs, still open

**`chance` is reported wrong.** `summary_coarse_group.json` carries `chance = 0.2` (1/5), but
`none (alkane)` is skipped as a group fold — it is the only hydrocarbon, so withholding it
leaves nothing to generalise from — and therefore hydrocarbon **never enters the test pool**.
The pooled predictions are 880 rows over 4 classes (oxygen 280 / nitrogen 240 / sulfur 200 /
halide 160). Effective chance is **0.25**. Every `layer_trend_*.png` draws its chance line low,
and every "×chance" statement shifts.

**Probabilities are discarded.** `run_cv` computes `predict_proba`
(`fc_group/functional_group_probe.py:330`) but persists only the argmax, so `predictions_*.csv`
holds `y_true,y_pred` alone. Log-loss and top-1 margin — `PROBE_NOTES.md` §9 item 4, "a metric
with resolution at ceiling" — are therefore unrecoverable without re-running the sweep. This is
why §4 is built on a resampling test over hard labels instead: the question was about
*significance*, which hard labels can answer, not about *resolution*.

**The `formula`-column notation confound is fixed for future runs, but not for the results in
this document.** `PROBE_NOTES.md` §7 — 9 of 20 groups are
always SMILES, 8 always condensed, and F/Br/I are SMILES while Cl is condensed. The halide
result stays confounded here. `extract_properties.py` now emits `smiles` and
`condensed_formula` for **every** molecule from one set of templates, so notation carries no
class information in either column, and `formula` is retained unchanged only for backward
compatibility. That cannot be applied retroactively: the notation confound is part of the
*prompt text*, which is baked into the activations these results were computed from. It clears
only on re-extraction.

**Reference lines are missing from the sweep.** It runs with `--num-null-samples 0` and without
`--surface-baseline` (both deliberate at the time, see the sbatch comments), so `null_mean`,
`p_value` and `surface_baseline_balanced_acc` are `null` in every summary. The surface baseline
has since been turned on in `fc_group/scripts/02_probe.sbatch`; the null remains off by design,
being a property of the fold structure rather than the activations. The null is a property of the fold
structure and is quotable once per model; the surface baseline is not (§5).

## 8. Regenerating

```bash
python fc_group/Analysis/probe/analyze_sweep.py            # writes data/*.csv
python fc_group/Analysis/probe/analyze_sweep.py --tag fine_to_coarse_group
```

Requires a populated `fc_group/Results_HCC/functional_group_probe/`. CPU-only, no model
loading, ~4 s. `--n-boot` and `--seed` control the bootstrap; CI bounds move by < 0.002 between
seeds at the default 20,000 draws.

| file in `data/` | contents |
|---|---|
| `layer_curves.csv` | model × entity × layer × balanced_acc — long-format source for everything else |
| `per_entity_summary.csv` | best/last, peak-to-last slide, final-layer step, whether the peak is the final layer, jump layer, 90%-of-best layer, all with normalized depth |
| `depth_by_family.csv` | per model × family: modal jump layer, 90%-of-best range and modal layer, median depth, floor-to-peak span |
| `model_diff_bootstrap.csv` | per entity: bAcc per model, difference, 95% CI, significance flag |
| `surface_baseline.csv` | the 15 measured character-n-gram baselines |
