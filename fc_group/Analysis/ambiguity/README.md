# The convention-labelled groups, read two ways

```bash
python fc_group/Analysis/ambiguity/analyze_ambiguity.py
```

Four groups carry oxygen but are named for another heteroatom — `amide` and `nitro` are
labelled `nitrogen`, `sulfoxide` and `sulfone` `sulfur`. Under leave-one-group-out they are
genuinely underdetermined: nothing in the chemistry picks nitrogen over oxygen for an amide,
only IUPAC does. A model that puts real mass on oxygen there is wrong in the chemically right
direction, and accuracy cannot see it. See `fc_group/PROBE_NOTES.md` §10.

This script emits the same question answered from two independent places, both contrasted
against the group's own **oxygen-free** family controls so a merely-diffuse model scores zero:

- **posterior** — the probe's out-of-fold `p(family)`, as `r = p(O) / [p(O) + p(home)]`,
  from `Results/ambiguity_metric/`.
- **geometric** — mean z-cosine from the group to the six oxygen-named groups, from
  `Results/functional_group_analogy/`. No probe, no softmax, no regularization constant.

## What holds: the within-model result, in all three models

Layer 31, `data/ambiguity_two_readings.csv`:

| group | post. Δ Llama | post. Δ Chem-R | post. Δ DSR1 | geom. Δ Llama | geom. Δ Chem-R | geom. Δ DSR1 |
|---|---|---|---|---|---|---|
| **amide** | +0.147 | +0.269 | +0.398 | +1.14 | +1.17 | +1.66 |
| **nitro** | +0.082 | +0.126 | +0.317 | +0.80 | −0.15 | +1.21 |
| sulfone | −0.003 | −0.095 | +0.033 | +0.69 | +0.08 | +0.73 |
| sulfoxide | −0.007 | +0.007 | −0.013 | +0.63 | +0.55 | +0.80 |

**Amide leans toward oxygen, well above its own nitrogen controls, on both readings in all
three models.** That is the durable claim, and it is an aim-1 claim: graded, chemically
correct ambiguity is present in the representation regardless of fine-tuning. It also shows
up with no metric at all — in Llama's layer-31 clustermap amide sits inside the nitrogen
block while carrying a visible warm band against ester/ketone/aldehyde/acid.

The sulfur half is the weak half by design, and comes out near zero or negative: `thiol`
(−SH) and `thioether` (−S−) are the direct analogues of `alcohol` (−OH) and `ether` (−O−),
so the sulfur *controls* sit close to the oxygen family themselves. That is a statement about
the controls. Quote amide and nitro; report the sulfur rows, do not lean on them.

## What does not hold: ranking models on this

Two independent reasons, both visible in the table.

**The two readings disagree.** The posterior puts Chem-R above Llama on both amide and nitro.
The geometry — which has no `C` in it — does not: amide is a tie (+1.17 vs +1.14) and nitro
is *reversed* (−0.15 vs +0.80). The two measure different objects (the probe reads whole
activations, the geometry reads chain-length-matched diff vectors), so this is not a
contradiction — but it does mean neither reading settles the ranking alone.

**`select_C` is not held fixed across models.** `best_C_mode` and `best_C_frac` are on every
row for this reason. At layer 31, the layer the probabilities are read at:

| | modal `best_C` | fraction of the 19 folds |
|---|---|---|
| Llama-3.1-8B | 1e-3 | 0.947 |
| Chem-R-Faithful | 1e-3 | 0.842 |
| DeepSeek-R1-Distill | **1e-4** | 1.000 |

Llama and Chem-R are **matched**, so that one posterior comparison is like-for-like. DeepSeek
sits two orders looser, emits flatter posteriors on everything, and tops the posterior
column — exactly the artifact §10 was written about.

> `PROBE_NOTES.md` §10 tabulates `best_C` at each model's *own best layer* and reports Chem-R
> at 1e-4 ×15. That is a different quantity from the one above and, on the current sweep, a
> different value; `select_C` varies with depth (Chem-R: 1e-3 ×18 at L19, 1e-2 ×15 at L24,
> 1e-3 ×16 at L31). Take it at the layer the probabilities are read at, which is what this
> script does.

Note this cuts against the tempting story either way: DeepSeek has the *worst* generation
accuracy of the three (0.804 vs Chem-R's 0.978), so even taken at face value the ambiguity
ranking does not explain the behavioural ranking.

**What would settle it:** `ambiguity_metric.py --full --mode projection`, which reads the
same question off the probe's own geometry with no softmax, plus `--c-grid 1e-3` to pin the
temperature identically across models. Neither has been run —
`Results/ambiguity_metric/*/functional_group/data/ambiguity_contrast.csv` does not exist for
any model. Until it does, report this section as within-model and say so.
