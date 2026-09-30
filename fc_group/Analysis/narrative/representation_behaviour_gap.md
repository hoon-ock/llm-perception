# The gap is commitment, not decoding

Interpretation note, not a generated artifact. See [`README.md`](README.md).

## The reading this started from

Table 1 (`paper/tables/t1_paradox.tex`) puts teacher-forced accuracy beside free-generation
accuracy on the same 920 prompts, and the two disagree sharply — base 0.837 against 0.585,
ChemDFM 0.892 against 0.635. Read alongside the clustermaps, the retrieval results and the
probe, all of which say these models hold good functional-group structure, the natural
conclusion is: *a good embedding space does not always yield good decoding.*

Directionally that survives. The evidence usually cited for it does not, and the corrected
version is a stronger claim than the original.

## Teacher-forced accuracy is not the representation measure

Both teacher-forced columns score candidate names as continuations and take the argmax. That
is the LM head — a decoding protocol under different conditions, not a probe of the residual
stream. A teacher-forced-vs-free-generation gap is protocol versus protocol, and putting the
representation-vs-behaviour claim on it puts the weight on the wrong pair of columns.

Teacher-forced accuracy is also not a ceiling on what the model knows, because it moves under
a choice that has nothing to do with chemistry. From `generation/data/generation_summary.csv`,
base scores **0.837** summing token log-probabilities and **0.401** dividing by token count —
identical prompts, identical model, a **0.436** swing from the scoring rule alone. That is
larger than its own teacher-forced → free-generation drop of 0.252. The caption already says
the two rules are not comparable across the pair; the same fact makes either one a poor stand-in
for "what the representation supports."

## The comparison that does carry the claim

Probe against free generation. Probe best from `probe/data/layer_curves.csv` (`family ==
declarative`, tag `coarse_group`), free-generation strict from
`generation/data/free_generation_summary.csv`:

| model | probe best | TF raw | free strict |
|---|---|---|---|
| **base** | **1.000** (L29) | 0.837 | **0.585** |
| chemdfm | 0.993 (L21) | 0.892 | 0.635 |
| chem | 0.983 (L19) | 0.978 | 0.843 |
| chem-r | 0.980 (L31) | 0.924 | **0.902** |
| reason | 0.963 (L29) | 0.804 | 0.789 |

Three measures, three orderings. Base is **first on the probe and last on free generation**;
reason is last on both teacher-forced columns and third on free generation. The paradox states
itself here without the teacher-forced column being involved at all.

## What the gap actually is

`free_generation_summary.csv` carries the decomposition, and it changes the mechanism.

| model | commit rate | accuracy **given** commit | non-answer | malformed |
|---|---|---|---|---|
| base | 0.637 | **0.918** | 0.149 | 0.132 |
| chem-r | 0.920 | 0.981 | 0.061 | 0.007 |
| chem | 0.858 | 0.984 | 0.042 | 0.007 |
| chemdfm | 0.665 | 0.954 | **0.286** | 0.002 |
| reason | 0.828 | 0.953 | 0.099 | 0.015 |

**Conditional accuracy is 0.918–0.984 in every checkpoint.** When any of these models emits a
functional-group name, it is right 92–98% of the time. The whole free-generation spread
(0.585 → 0.902) is the commit rate (0.637 → 0.920).

Base's 41.5 points of strict failure split as malformed 13.2, non-answer 14.9, underspecified
8.3, **wrong group 5.2** — one eighth of its failures are chemistry errors. The four rates sum
to `1 − strict_accuracy` exactly, for all five models, so nothing is unaccounted for.

ChemDFM fails by a different route to nearly the same score: 28.6 of its 36.5 failure points
are non-answers at a **0.2%** malformed rate. It is not garbling format — it describes the
molecule instead of naming the group.

Nor is the failure stable across phrasings. `generation/data/free_generation_by_template.csv`:
base runs **0.196** on template 8 and **0.880** on template 9, same chemistry, 0.68 spread
from wording. `free_generation_scoring.py` records template 8 as 75% malformed for base — most
of its format collapse sits in one phrasing that an overall rate averages away.

So the mechanism is not that decoding corrupts a good representation. **The model fails to
enter the answering behaviour at all**, and conditional on entering it, the chemistry is
there. Chemistry fine-tuning bought answer compliance (commit 0.637 → 0.920) far more than
chemical knowledge (conditional 0.918 → 0.981), against a probe already at ceiling in base.

## Caveats, in the order a reviewer hits them

**Granularity mismatch — the load-bearing one.** The probe is 5-way heteroatom family (chance
0.20); free generation is 20-way group naming (chance 0.05). Only `coarse_group` and
`fine_to_coarse_group` exist on disk, and `fine_to_coarse_group` trains on the 20 but still
*scores* the 5, so **no probe number exists at the generation task's granularity** — verified
across all five models under `Results/functional_group_probe/*/functional_group/data/`.
Comparing 1.000 to 0.585 therefore crosses a task-difficulty change as well as a
representation-behaviour one. State the comparison as coarse-probe versus fine-generation and
say so in the text.

> **The fix once proposed here has been tried and does not work.** A
> `functional_group_probe.py --target fine --split molecule` run was done on all five
> checkpoints. It returns 1.0000 balanced accuracy in every one — and 0.95–0.999 at *layer 0*,
> with the character n-gram surface baseline on the same folds at **1.0000**, above every probe.
> Every extraction template embeds `{iupac_name}`, so `1-bromobutane` spells its own answer and
> the molecule split leaves that shortcut intact; the group split, which defeats it, is ill-posed
> for a 20-way target. A matched-granularity representation number therefore still does not exist
> and cannot be made from the cached activations. See
> [`group_level_representation_behaviour.md`](group_level_representation_behaviour.md), which
> also tests the per-group form of this note's claim and finds no relation between per-group
> generation accuracy and per-group retrieval quality.

**ChemDFM sits on Llama-3, not 3.1.** Already daggered in the Table 1 caption. A ChemDFM−base
gap confounds chemistry training with the base-model change; chem-r carries the controlled
contrast. This matters here because ChemDFM is one of the two models driving the
commitment story.

**Sample size.** 92 molecules × 10 templates. `generation/data/generation_diff_bootstrap.csv`
and `free_generation_diff_bootstrap.csv` cover chem−base at the molecule level; the five-way
orderings in the tables above have no intervals attached and should not be quoted as rankings.

## The formulation to lift into a draft

> Linear decodability of functional-group family is at ceiling in every checkpoint, base
> included — yet the same models differ by 32 points in whether they will name the group when
> asked. The difference is almost entirely whether the model commits to an answer, not whether
> its answer is correct: conditional on committing, accuracy is 0.918–0.984 across all five
> checkpoints. Chemistry fine-tuning moves the commitment, not the chemistry.

What to drop: any claim that teacher-forced accuracy measures representation quality. The
raw/length-normalized swing in Table 1 refutes it from inside the table itself.
