# Per-group generation accuracy does not track per-group embedding quality

Interpretation note, not a generated artifact. See [`README.md`](README.md). Companion to
[`representation_behaviour_gap.md`](representation_behaviour_gap.md), which this closes one open
caveat of and opens another.

Generated evidence: `generation/data/group_representation_join.csv`,
`group_correlations.csv`, `group_reliability.csv`, written by
`Analysis/generation/analyze_generation_vs_representation.py`; figures in
`visuals/generation/`.

## The question

`Results/generation_eval/*/functional_group/data/summary.json` reports per-functional-group
free-generation recall, and the spread is large: in base, `alkyl iodide` 0.050 against
`carboxylic acid` 0.900. The natural hypothesis is that this is embedding quality — groups the
model represents badly are the groups it names badly — testable against the per-group retrieval
ranks and hit rates behind `visuals/retrieval/`.

It does not hold. What follows is the test, and then three reasons the test could not have come
out any other way given the measures available.

## The per-group spread is real, in three checkpoints of five

Split-half over templates (even vs odd), Spearman-Brown corrected, from
`group_reliability.csv`:

| model | strict | superclass credit |
|---|---|---|
| base | 0.871 | 0.763 |
| chem-r | **0.271** | **0.295** |
| chem | 0.793 | **0.321** |
| chemdfm | 0.502 | 0.700 |
| reason | 0.933 | 0.911 |

Base, reason and chemdfm carry genuine per-group signal. chem-r sits near ceiling on every group
and chem's superclass-credit rate is near-constant, so for those two a flat correlation means
*could not detect*, not *no relationship*. Every figure and table marks them, and none of the
claims below rest on them.

## The correlation is absent

Preregistered primary test — nearest-rival margin at layer 31 against superclass-credit recall,
Holm-corrected across the five checkpoints plus a pooled fit:

| | rho | 95% CI | p_holm |
|---|---|---|---|
| base | −0.313 | [−0.663, +0.160] | 0.857 |
| chemdfm | +0.035 | [−0.427, +0.465] | 1.000 |
| reason | +0.457 | [+0.035, +0.775] | 0.257 |
| pooled | +0.065 | [−0.210, +0.335] | 1.000 |

Nothing survives. The two axes the question actually named do no better: on
`retr_hit1` at L31, base rho=+0.16 [−0.30, +0.58] and reason rho=−0.31 [−0.71, +0.21].

The exploratory grid is 2539 cells. At uncorrected p<0.05 it returns **318** against **127**
expected by chance — so there is something in it — but after BH correction, the reliability
filter and the bootstrap CI, **24** survive, and they are not spread over the predictor space at
random. They concentrate on within-group cohesion at mid depth: `within_cos` at L8 in base
(rho=+0.81 against strict), `silhouette` at L24 in reason (+0.77), and `within_score` at L24
pooled (+0.47 [+0.23, +0.65], the same sign in all three models it covers).

That is the honest residue: **a modest mid-layer association with how tight a group's own cluster
is, and nothing at all with how well the analogy machinery retrieves it.** It is exploratory, it
is layer-specific — base's `within_cos` runs +0.64, +0.81, +0.06, −0.35, −0.07 across L0/8/16/24/31
— and the layer was not preregistered. It should be treated as a hypothesis for a run designed
around it, not a result.

## Why no correlation was reachable: the predictors

**The carbon-ladder axis has no variance on the quantity being predicted.**
`top1_same_label_rate` is 1.000 for all 19 groups in all 5 models, at every layer. When ladder
retrieval misses, it lands on the *right functional group at the wrong carbon count*. Its rank
and hit columns measure chain-length resolution. Generation is scored on group identity, and on
group identity that axis is constant.

**The group-identity axes are thin and saturated.** `inter_group` plus `halide` reach 17 of 20
groups at 4–16 trials each, and at L31 most groups sit at hit@1 = 1.000. A predictor that is
constant over two-thirds of its support cannot rank anything. The left-hand tail is also where
the contradictions live — base at L31:

| group | ladder hit@1 | retr hit@1 | strict | teacher-forced |
|---|---|---|---|---|
| sulfone | **0.000** | 1.000 | **0.875** | 1.000 |
| ester | 0.167 | 1.000 | 0.700 | 0.700 |
| imine | 0.667 | 0.750 | **0.275** | 0.800 |
| thioether | 0.250 | 0.625 | 0.425 | 0.800 |

`sulfone` is the worst group in the repository by ladder retrieval and among the best by
generation. No monotone relation fits that.

**The matched 20-way probe is orthography, not representation.** This is the correction to
`representation_behaviour_gap.md`, whose granularity caveat proposed exactly this run as the fix:
"The fix is a `functional_group_probe.py --target fine` run: it reads saved activations, needs no
GPU, and would give a matched 20-way number." The run was done —
`--target fine --split molecule`, all five checkpoints — and it does not give a usable number:

| model | best layer | best balanced acc | balanced acc at **layer 0** |
|---|---|---|---|
| base | 30 | 1.0000 | 0.9875 |
| chem-r | 17 | 1.0000 | 0.9525 |
| chem | 17 | 1.0000 | 0.9544 |
| chemdfm | **3** | 1.0000 | 0.9812 |
| reason | 18 | 1.0000 | 0.9988 |

A 20-way task at 0.95+ in the first layer is the tell, and the script's own control settles it:
the char n-gram baseline on the raw prompt text scores **1.0000** on the same folds, *above* every
probe. Every extraction template embeds `{iupac_name}`, and `1-bromobutane` spells its own answer.
Under a molecule split the fine probe reads the name. This is why the fine target is only ever run
with `--split group` elsewhere here — leave-one-group-out is what defeats the shortcut, and is
also why it cannot be run at this granularity, since holding a class out of a 20-way classifier
removes its output unit.

So: the fine-grained probe number that note asked for **does not exist and cannot be produced from
the cached activations.** Producing it needs a GPU extraction pass on the `functional_group_smiles`
entity type already defined in `config_extract_activation.yaml` but never extracted — SMILES-only
prompts, where `CCCCBr` still leaks less than `1-bromobutane` — and even then the leak is reduced,
not removed. The honest statement in a draft is that no matched-granularity representation measure
is available, not that the probe says the representation is perfect.

## What the spread is made of instead

`group_response_composition.png`, and the four halides are the case to read. Base at L31:

| group | strict | +superclass credit | underspecified | non-answer | teacher-forced |
|---|---|---|---|---|---|
| alkyl bromide | 0.325 | 0.775 | **0.450** | 0.050 | 0.875 |
| alkyl chloride | 0.150 | 0.625 | **0.475** | 0.100 | 0.950 |
| alkyl fluoride | 0.250 | 0.350 | 0.100 | **0.300** | 0.425 |
| alkyl iodide | 0.050 | 0.375 | 0.325 | 0.175 | 0.475 |

`alkyl chloride` scores 0.150 by strict free generation and **0.950** by teacher-forced recall.
The model is not failing to identify chlorine; it answers "alkyl halide" and
`free_generation_scoring.py` files that as `underspecified`. Credit the superclass and the group
moves to 0.625. Bromide behaves the same way. That gap is scoring resolution, and no embedding
metric could have predicted it because it is not a property of the embedding.

Fluoride and iodide fail differently — teacher-forced 0.425 and 0.475, the only two groups where
the forced-choice head is also weak — and those two *are* candidates for a genuine representation
deficit. `imine` (non-answer 0.400) and `thioether` (0.450) fail a third way, by not naming any
group, with teacher-forced recall at 0.800 in both.

Three failure modes, one number. Per-group strict recall pools underspecification, non-answer and
genuine error, and only the last of the three is a thing an embedding metric is even about. That
is the per-group form of the commitment story in `representation_behaviour_gap.md`: conditional on
committing, the chemistry is there.

## What to write, and what not to

> Per-functional-group free-generation accuracy ranges from 0.05 to 0.90 within a single
> checkpoint, and none of it is explained by per-group retrieval quality: the primary test returns
> rho=+0.07 [−0.21, +0.33] pooled, and the worst-retrieved group in the set (`sulfone`, ladder
> hit@1 0.000) is among the best generated (0.875). The spread decomposes instead into three
> distinct failure modes that strict scoring pools into one — the halides are answered at the
> superclass level (`alkyl chloride`: 0.150 strict, 0.950 teacher-forced), `imine` and `thioether`
> are not answered at all, and only `alkyl fluoride` and `alkyl iodide` show the joint
> free-generation and teacher-forced weakness that a representation deficit would predict.

Do not write: that the 20-way probe shows the representation is perfect at generation
granularity. It shows the prompt is readable.

## Caveats, in the order a reviewer hits them

**n = 20, and that is the ceiling.** Every correlation here has 17–20 functional groups as its
unit. The bootstrap CIs are roughly ±0.45 wide per model; only rho above ~0.55 could clear zero.
The pooled fit ranks within model before pooling and bootstraps over groups rather than rows, so
it does not buy power by double-counting — it is still 17–20 clusters.

**Per-group generation rests on 4–8 molecules.** Each group has 40–80 prompts but only 4–8
distinct molecules behind them; the split-half figures above are over templates, which is the
axis with enough levels to split, and so they do not capture molecule-sampling variance at all.
The true reliability is lower than the table says.

**The layer freedom in the exploratory grid.** Five layers × fourteen predictors × seven outcomes.
BH is applied across the whole grid, which is why 318 uncorrected cells become 24, but the
mid-layer cohesion pattern was found by looking, not by prediction.

**ChemDFM sits on Llama-3, not 3.1**, as everywhere else here. It is one of the three checkpoints
with usable per-group reliability, so it carries more weight in this note than in most.
