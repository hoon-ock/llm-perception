# What fine-tuning did to the representation

The aim-2 supporting table. Joins three result trees that each hold one piece of the same
story and are never otherwise read together.

```bash
python fc_group/Analysis/geometry/analyze_geometry.py
```

Sources (all **gitignored**, so `data/geometry_by_layer.csv` is the durable record):
`Results/anisotropy_diagnostic/`, `Results/functional_group_analogy/`,
`Results/functional_group_analogy_retrieval/`.

## 1. The honest-signal check comes first

Raw between-class cosines run 0.40–0.45 at layer 31. They are not evidence of anything on
their own: the residual stream is anisotropic — mean pairwise cosine **0.944** in Llama at
layer 31 — so two unrelated vectors are similar by default.

After mean-centering, the between-class mean lands **exactly on the empirical null** at every
layer in every model (−0.044 vs −0.042 for Llama at L31; the pattern is identical for the
other two). Within-class does not: 0.754 / 0.616 / 0.736 against a null of ±0.17–0.25.

So the durable geometric claim is **within-class**: the functional-group diff vector is
consistent across chain lengths. Any claim resting on between-class similarity — including
every dendrogram in the paper — has to be framed as *relative* structure, with the centered
null stated. `../taxonomy/` is built that way on purpose: it z-scores within each model's own
between-class distribution and never quotes a level.

## 2. Fine-tuning de-anisotropises

`raw_pairwise_cosine`:

| layer | Llama | Chem-R | DeepSeek |
|---|---|---|---|
| 0 | 0.989 | 0.986 | 0.988 |
| 16 | 0.950 | 0.896 | 0.924 |
| 24 | 0.859 | **0.589** | 0.649 |
| 31 | **0.944** | **0.676** | 0.806 |

and `diffvec_svd_top1` at L31 — the share of diff-vector variance on a single direction —
0.469 / **0.353** / 0.421.

Both fine-tunes end far less anisotropic than the base, and Chem-R's diff vectors spread over
more directions. This is worth stating explicitly because it *partially produces* the visual
impression that Chem-R's groups are "more standalone" in a clustermap: its raw between-class
max is 0.742 against Llama's 0.956. The impression is real; the cause is not extra chemical
separation alone.

## 3. Linear compositionality degrades

`degenerate_rate` — the share of retrieval trials where `b2 + (a1 − a2)` is nearest to `b2`
itself once the exclusion is lifted, i.e. the offset never left the source group's own
neighbourhood. Scored on the same trials as hit@1, so a model can retrieve well and still
have a weak offset.

| L31 | Llama | Chem-R | DeepSeek |
|---|---|---|---|
| pooled hit@1 | 0.80 | 0.80 | 0.60 |
| hit@1 excluding sulfoxide | 0.889 | 0.889 | 0.667 |
| **degenerate rate** | **0.25** | **0.65** | **0.70** |

Read as a **cost**: the fine-tuned models sharpen categorical identity and lose the linear
offset structure that makes the analogy an analogy. Chem-R ties Llama on hit@1 while failing
the degeneracy check nearly three times as often.

`hit1_ex_sulfoxide` is emitted beside `hit1` rather than in place of it. Sulfoxide is
hit@1 = 0.00 in **every** model at **every** layer, so dropping it shifts all three equally —
it is a property of the group, not a limitation of any one model. Chem-R's defensible
retrieval claim is that it **peaks earlier and higher**: 1.000 ex-sulfoxide at L24 against
Llama's 0.833, matching its probe peak at L19 against Llama's L29.

## Limits

- Five layers (0/8/16/24/31) — the anisotropy and analogy trees sample only these.
- Pooled retrieval is **20 trials over 13 groups**. A 0.80-vs-0.775 gap is one trial. Never
  quote a between-model retrieval difference without a bootstrap; there is not enough
  resolution here for one.
