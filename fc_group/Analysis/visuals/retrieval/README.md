# Analogy retrieval

`hit1/2/3.png` report how **often** `b2 + (a1 - a2)` retrieves `b1`; `meanrank.png` reports how far
off it lands when it misses; `rankdist.png` draws every trial behind those two summaries. The
`analogy3d_*` set shows what one success looks like. None of them reads `Results/`.

## hit@1 / hit@2 / hit@3

Three PNGs written by `fc_group/Analysis/paper/make_visuals_retrieval.py`:

```
python fc_group/Analysis/paper/make_visuals_retrieval.py [--layer 31] [--out-dir DIR]
```

| file | cutoff | chance line |
|---|---|---|
| `hit1.png` | hit@1 | `random_hit1` |
| `hit2.png` | hit@2 | `random_hit2` |
| `hit3.png` | hit@3 | `random_hit3` |

Each draws the three retrieval axes on x, the five registered 8B checkpoints as bars, at
layer 31 by default. They share one y-limit so the three can be read against each other.
Source: `fc_group/Analysis/geometry/data/retrieval_by_axis.csv`, which
`geometry/analyze_geometry.py` snapshots out of `Results/`. Nothing here reads `Results/`.

These are exploratory, not manuscript figures. `paper/figures/f4_retrieval.pdf` is the
manuscript's version and draws hit@1 only, for three models.

## What hit@k is

Each trial is an analogy quadruple `a1 : a2 :: b1 : b2` — two molecules from a source group
and two from a target group. The offset `b2 + (a1 - a2)` is computed in the residual stream
and its nearest neighbours are ranked by cosine, **with the source group excluded**. The
trial is a hit@k when the true completion is among the top k.

hit@k is monotone in k by construction: `hit@1 <= hit@2 <= hit@3`, always. Raising k does
not measure a different ability, it relaxes the same one. The three figures are read as a
single gradient, which is why they share a scale.

All numbers are pooled across each axis's groups **weighted by trial count**, so a group
contributing 12 trials counts twelve times a group contributing one.

## The three axes are not comparable to each other

Each axis is its own tree with its own candidate pool, so each carries its own chance line
and a bar on one axis cannot be read against another axis's baseline:

| axis | tree | source groups | trials | ranked over | chance @1/@2/@3 |
|---|---|---|---|---|---|
| inter-group | `functional_group_analogy_retrieval` | 13 | 20 | 16-17 groups | 0.062 / 0.124 / 0.185 |
| halide column | `..._retrieval_halide` | 4 | 12 | 16-17 groups | 0.060 / 0.120 / 0.180 |
| chain-length ladder | `..._retrieval_carbon` | 19 | 228 | 73-74 molecules | 0.014 / 0.027 / 0.041 |

The **figures label these axes `Inter-group` / `Halide ladder` / `Chain-length ladder`**
(`PRETTY`, `make_visuals_retrieval.py:53`, shared by all five). This page keeps calling the
second one the halide **column**, because that is what it chemically is — F, Cl, Br, I down one
column of the periodic table, not a ladder in the chain-length sense. The display name and the
description differ on purpose; the CSV key is `halide` either way.

The group count on each x tick label is the number of **source** groups contributing trials,
not the number of choices. The choices are the `n_candidates_excl` column of
`retrieval_trials.csv`, and `random_hit@k` is exactly `mean(k / n_candidates_excl)` over that
axis's trials — which is why the halide chance is 0.060 and not 1/4: a halide trial still
ranks its offset against every other functional group, not against the other three halides.
The carbon ladder ranks over individual molecules rather than groups, which is what makes its
chance line four times lower and its 228 trials worth more than the other two axes combined.

## Two caveats that do not fit on the figure

**1. The small axes saturate.** The inter-group axis is 20 trials and the halide column is
12. At k=2, base and chemdfm sit at exactly 1.00 on inter-group — that is 20/20, and one
trial is worth 0.05 on that bar. At k=3 nothing moves on inter-group at all, because every
model that was going to be right already was. Only the chain-length ladder (228 trials) has
the resolution to separate five models, and it is the axis where the gradient from k=1 to
k=3 is real. The trial counts are printed on the x tick labels for this reason.

**2. Retrieval is scored on the same trials as the degenerate rate.** A trial is *degenerate*
when `b2 + (a1 - a2)` is nearest to `b2` itself once the exclusion is lifted — the offset
never left the source group's neighbourhood, and the hit was scored on an exclusion that did
the work. `geometry/data/geometry_by_layer.csv` carries `degenerate_rate` at L31: base 0.25,
chem 0.65, reason 0.70. So a model can retrieve well and still have a weak offset, and the
fine-tunes are exactly where that gap is widest. hit@k should not be read as evidence about
the offset without that column beside it — see `analyze_geometry.py`'s module docstring.

## What the layer-31 numbers say

The chain-length ladder is the only axis that separates the five models, and it separates
them against the chemistry story: base (0.544 / 0.785 / 0.882) and reason
(0.636 / 0.829 / 0.895) beat both chemistry fine-tunes at every cutoff, with chem-r lowest
(0.395 / 0.557 / 0.614). Chemistry fine-tuning does not improve — and on this axis degrades —
the linear-offset structure that encodes chain length. On the two small axes all five models
are within a trial or two of each other at every k.

## mean rank

`meanrank.png`, written by `fc_group/Analysis/paper/make_visuals_retrieval_rank.py`:

```
python fc_group/Analysis/paper/make_visuals_retrieval_rank.py [--layer 31] [--out-dir DIR]
```

Same five checkpoints, same three axes, same layer, same source CSV. The bar is the mean of
`rank_excl` over the axis's trials, trial-weighted across groups exactly as the hit rates
are, pooled into `mean_rank` by `analyze_geometry.retrieval()`. The dark tick on each bar is
that model's **median** rank.

**Why it exists.** A hit rate cannot say how badly a miss misses, and on the halide column
that is not a hypothetical. At L31 chemdfm ties for the best hit@1 (0.667) and has the worst
mean rank of the five (6.33); base is the mirror image, hit@1 0.500 and mean rank 4.08. Read
through hit@k alone those two models look like the same result in opposite order. They are
not: chemdfm is right slightly more often and catastrophically wrong the rest of the time.

**Chance is the pool size, not a constant.** `random_mean_rank` is `mean((n + 1) / 2)` over
each trial's `n_candidates_excl`, the rank a uniformly random ordering would average. That
makes it 8.60 on the inter-group tree, 8.83 on the halide column and **37.33** on the carbon
ladder — the same asymmetry that puts the ladder's hit@1 chance four times lower than the
other two, seen from the other end.

**Read the tick against the bar top.** Mean rank is a tail-sensitive statistic and the
median is not, so the gap between them is the whole diagnostic. A tick pinned at the floor
under a tall bar means the model is usually right and occasionally hopeless — the bar is a
handful of trials. A tick that rises with the bar means the ordering itself got worse. Both
patterns are present at L31, in adjacent panels, and they support opposite conclusions; see
the caveats at the end of this section.

**Four consequences for how the figure is drawn**, none of them cosmetic:

* the panels are three real subplots and **cannot share a y-limit**, where `hit1.png` is one
  axes with three tick groups. A scale reaching 37 would flatten the first two panels to
  nothing. Bar heights are therefore comparable *within* a panel only — more strictly than on
  the hit figures, where at least the axis was shared;
* the y-floor is **rank 1, not 0**. Rank 0 does not exist, so each bar measures excess rank
  above perfect retrieval rather than rank itself;
* **up is worse.** The y-label carries the direction. This inverts `analogy_retrieval.py`'s
  own `retrieval_rank_by_group.png`, which flips the axis so better is up; these panels get
  read beside `hit1.png` instead, where chance sits near the floor and the bars rise away
  from it, and flipping would put chance at the bottom of the frame with the bars hanging
  from the ceiling;
* the median tick **overhangs its bar on both sides**, and carries a white halo. The median
  is exactly 1.00 on eleven of the fifteen bars, where a tick cut to the bar's width would
  lie flush along the black baseline spine and disappear — and a median pinned to the floor
  under a tall bar is the single most important thing this figure has to say.

The value label above every bar is load-bearing rather than decorative: with chance at 37.3
in frame the ladder panel's bars are short, and the labels are what keep a 1.81-to-6.10
spread readable. The tick is deliberately *not* labelled — a second number per bar would
crowd that panel past reading, and the tick's position against the bar top is the message.

**What the layer-31 numbers say.** The rank view agrees with hit@k on the chain-length ladder
and sharpens it: reason 1.81 and base 2.33 against chem-r 6.10 and chem-faithful 6.03, and
the medians move with the means (1 against 2), so the chemistry fine-tunes there are not
merely less often right — they are farther away when wrong, across the distribution. The
inter-group axis looks like the same story and **is not**: the bars split 1.15–1.20 against
2.55–2.60, but every median is 1 and the gap is two trials — see the caveat below. The
halide column is the one axis where hit@k and mean rank disagree outright, and the
disagreement is the chemdfm case above.

### Two more caveats, specific to rank

**1. The inter-group rank gap is one group, and the ladder gap is not.** This is the caveat
the median tick exists to carry, and the two axes answer it in opposite directions.

On **inter-group**, the entire chem-r / chem-faithful deficit is sulfoxide — 2 trials of 20.
base ranks sulfoxide 2; chem-r ranks it 15 and chem-faithful 16, out of a ~16-candidate pool,
which is last or next to last. Every other group is 1.00–1.50 for all three. Drop that one
group and the means are **1.11 / 1.17 / 1.11**: indistinguishable. The medians are already
1 / 1 / 1, which is what the floor-level ticks on that panel are telling you.

| inter-group, L31 | base | chem-r | chem-faithful |
|---|---|---|---|
| mean rank | 1.20 | 2.55 | 2.60 |
| median rank | 1.0 | 1.0 | 1.0 |
| mean, sulfoxide dropped | 1.11 | 1.17 | 1.11 |
| trials ranked > 10 | 0/20 | 2/20 | 2/20 |

On the **chain-length ladder** the same comparison is real and systematic. chem-r is worse
than base on **50 of the 76 (group, chain-length) cells**, equal on 25 and better on 1. The
median shifts 1 → 2, the 10%-trimmed mean still splits 1.56 against 3.09 and 2.96, and the
fine-tunes are worse at the *top* of the distribution too — rank 1 on 39–41% of trials
against base's 54%. Dropping sulfoxide changes the ladder numbers not at all. The damage is
chemically coherent rather than scattered: `ester@4`, `ester@5`, `ether@4`, `ether@5`,
`thioether@4` and `sulfone@4/5` collapse to rank 31–37 out of ~74 — essentially chance —
where base holds 5–9. The O- and S-containing groups at mid chain lengths are where
chemistry fine-tuning destroys the chain-length offset.

So the two panels cannot be summarised together. `n = 20` on inter-group is small enough
that one hostile group sets the ranking; `n = 228` on the ladder is not.

**2. The degenerate-rate caveat applies here unchanged.** A low mean rank scored on an
exclusion that did the work is still a low mean rank. `degenerate_rate` belongs beside this
figure for the same reason it belongs beside the hit figures.

## rank distribution — every trial as a dot

`rankdist.png`, written by `fc_group/Analysis/paper/make_visuals_retrieval_dist.py`:

```
python fc_group/Analysis/paper/make_visuals_retrieval_dist.py [--layer 31] [--out-dir DIR]
```

One dot per trial, at its rank, with the same median tick and a mean marker that
`meanrank.png` carries. This is the figure that settles the caveat above by showing it: on
the inter-group panel chem-r and chem-faithful are two lonely dots above the reference line
with everything else on 1 and 2, while on the ladder the same two models fill the whole
5-to-20 band. Same metric, same models, two different shapes of failure.

**This panel's legend is worded differently from `hit*.png` and `meanrank.png`.** Its dashed
line is labelled **`Random ranking`**, not `chance`: the value is `random_mean_rank`, the mean
of `(n + 1) / 2` over each trial's pool — the rank the true completion would average if the
candidates were ordered at random. `chance` named no mechanism, and the same objection was
already fixed on `probe_depth.png` (`Random guess`) and `label_agreement_depth.png`
(`Random neighbours`), each worded for the strategy *it* draws rather than to one shared word.
The other retrieval figures still say `chance`; they have not been swept yet.

Its y-axis reads **`Rank (log scale)`**. The old label said `(log)`, which parses as a unit —
the quantity is a plain integer rank, and the parenthetical describes the axis. The scaling
itself is load-bearing: ranks run 1–74 on the ladder while nearly every trial sits in the 1–3
band, which a linear axis would crush to nothing.

**Why dots and not a violin.** A violin is the obvious reach and it is the wrong tool on two
of these three panels. Ranks are small integers piled on 1, not a continuous variable, and
the small axes barely have a distribution to smooth:

| axis | n | distinct rank values per model | example |
|---|---|---|---|
| inter-group | 20 | 2–3 | base is `{1: 16, 2: 4}` |
| halide column | 12 | 2–5 | chemdfm is `{1: 8, 17: 4}` |
| chain-length ladder | 228 | 9–25 | base spans 14 values |

A KDE over `{1: 8, 17: 4}` draws a smooth body across ranks 2–16, where not one trial
landed, and spills density below rank 1, which cannot exist. It would invent precisely the
structure the figure exists to adjudicate. Dots cannot lie about a sample that small — at
n = 12 you are looking at all twelve.

**How a column is built.** Trials tied at one rank are spread horizontally at a fixed
spacing that clamps once the row fills the column, so a row's width reads as its count: two
tied trials are visibly two dots, a hundred are a saturated bar. Placement is deterministic
rather than jittered, so re-running cannot reshuffle the picture. The median tick is drawn
*beneath* the dots and reaches slightly wider than them — on a median of 1 under a full row
of ties, which is most columns, it reads as two ends poking out instead of a black bar laid
over the data it summarises.

**y is logarithmic**, because ranks run to 63 here while the structure worth seeing is the
1-2-3 band that holds most trials. **Panels keep their own limits**, as in `meanrank.png`:
the top tick on each is that axis's pool size — ~16 candidates, ~17, ~74 — because rank 8 is
chance on the first two panels and good on the third.

**Dot density is not comparable across panels.** The ladder draws 228 dots per column and
inter-group draws 20. A ladder column looks fuller because it holds more trials, not because
the model did worse.

Source: the committed `geometry/data/retrieval_rank_hist.csv`, a tidy
`(model, axis, layer, rank, n)` histogram snapshotted by `analyze_geometry.main()`. Lossless
for integer ranks — repeating each rank `n` times reconstructs the exact sample that produced
`mean_rank` and `median_rank`, which is checked rather than assumed.

---

# `analogy3d_*` — successful retrievals as 3D parallelograms

```
python fc_group/Analysis/paper/make_visuals_retrieval_3d.py [--layer 31] [--out-dir DIR]
```

One **PNG plus one `.txt` of the same stem** per (quadruple, chain length) cell that is 4/4
corners at hit@1 with zero degenerate corners — six of twenty at base/L31. Source:
`fc_group/Analysis/analogy/data/analogy_quiver3d.csv`, which `analogy/snapshot_quiver3d.py`
snapshots out of `Results/`. Nothing here reads `Results/`.

Stems are `analogy3d_{model}_L{layer}_{a1}-{a2}_{b1}-{b2}_C{N}`. The model comes first so a
directory of these sorts by checkpoint, and it is the full `Results/` directory slug rather
than `_common.SHORT`'s `base`, which means nothing outside this repo.

| stem (prefix `analogy3d_meta-llama-Llama-3.1-8B_L31_`) | C | rank in 4096-D | cos | margin |
|---|---|---|---|---|
| `thioether-thiol_ether-alcohol_C5` | 5 | 1 of 16 | 0.81 | X 3.9 from `ether`, 12.5 from `alkyl bromide` |
| `thioether-thiol_ether-alcohol_C6` | 6 | 1 of 16 | 0.79 | X 3.2 from `ether`, 14.8 from the next |
| `ester-carboxylic-acid_ketone-aldehyde_C4` | 4 | 1 of 16 | 0.83 | X 2.8 from `ketone`, 12.3 from `alcohol` |
| `imine-aldehyde_amine-alcohol_C4` | 4 | 1 of 16 | 0.78 | X 7.7 from `amine`, 12.6 from the next |
| `imine-aldehyde_amine-alcohol_C5` | 5 | 1 of 16 | 0.76 | X 7.7 from `amine`, 9.6 from `nitrile` |
| `imine-aldehyde_amine-alcohol_C6` | 6 | 1 of 16 | 0.75 | X 6.3 from `amine`, 11.6 from the next |

The six are chosen by `snapshot_quiver3d.rank_cells`, on the scored trials — not by hand, and
not here. See `analogy/README.md` §8 for the full twenty-cell ranking.

## The numbers are in the `.txt`, not on the image

Each PNG carries only what is needed to read the picture: four functional-group names, the
legend, the axis names. Its `.txt` carries everything else — the retrieval outcome, the
projection diagnostics, the compounds behind each group, and the caveats:

```
RETRIEVAL  (full 4096-D; copied from retrieval_trials.csv, never recomputed here)
  rank of ether                     1 of 16
  cosine to the constructed point   0.806
  corners at hit@1                  4 of 4
  degenerate corners                0
PROJECTION  (mean-centered top-3 PCA over all 19 groups; plot units)
  of the constructed point in view  47%
  3D rank of ether                  1   (agrees with the 4096-D rank)
  X lands                           3.9 from ether
  nearest rival                     12.5 from alkyl bromide
```

**There is no title on the image.** The quadruple and the chain length are in the filename
and at the head of the `.txt`, and the four labelled groups already say which quadruple a
panel is.

**The axes carry no tick labels.** The PC coordinates are in no unit a reader wants off an
axis, and every distance the panel claims is in the sidecar in those same units. The grid
stays, because it is what shows the three axes share one scale — and because mplot3d draws
the grid *at* the ticks, the ticks are kept and only their labels emptied.

**One exception travels on the image.** If the projection disagrees with the 4096-D rank, the
panel says so in red on its own face. An image travels without its sidecar, and a panel that
is quietly misleading is the failure this whole set exists to avoid.

## Every point is a compound

The pool is one chain length at a time, not an average over C3–C6, so every point is a
concrete compound — `ether` at C5 is methyl butyl ether. The image labels the functional
group only; the `.txt` names the compound behind each of the four, under `COMPOUNDS AT C{N}`.

**Three groups are still a mean.** Alcohol, thiol and amine each hold an `n-`/`sec-` isomer
pair at every chain length in `functional_group_dataset.csv`, so their vectors average two
molecules. The sidecar lists both names and flags them `(2-isomer mean)`. The other seven
groups in these quadruples are single compounds.

## How to read a panel

Each file draws one corner explicitly — `b1 = b2 + (a1 - a2)` — and the parallelogram implies
the other three:

- a **solid blue** arrow `a2 -> a1`, the offset the analogy asserts;
- the same arrow **dashed**, re-based at `b2`, ending in an **X** at the constructed point;
- a **solid orange** arrow `b2 -> b1`. If the analogy holds, it and the dashed arrow coincide
  — on these six they nearly do, and that coincidence is the result;
- a **dotted** residual from the X to the true `b1`, the error the retrieval tolerated;
- the other 15 groups as **grey dots**. They are the candidate pool, and they are why the
  panel is about retrieval rather than about a small residual: hit@1 is a claim about winning
  a race. The sidecar quotes the margin for the same reason.

All three axes share one scale. Anisotropic autoscaling would keep the two offset arrows
parallel — parallelism survives any affine scaling — but not distance, and distance is the
claim.

## The ester file is the negative control

`ester-carboxylic-acid_ketone-aldehyde_C4` is the quadruple set's
`DELIBERATE NEGATIVE CONTROL` (`functional_group_analogy_carbon_matched.py:306-311`): acid →
ester and aldehyde → ketone are formally the same substitution (H → CH3) but chemically are
not, and it is the quadruple that should fail if leg symmetry is what makes an analogy work.
It passes the selection rule cleanly at C4 and retrieves at rank 1. Its sidecar says so, so
the figure is not read as a plain success.

## Not the full space

These are 3 of 4096 dimensions. The sidecar prints `of the constructed point in view` — the
fraction the three drawn dimensions actually hold, 0.42 to 0.64 across the six. `rank` and
`cos` are the **full 4096-D** numbers copied from the scored trials; nothing is rescored for
the picture. And a projection can invert a result: at layer 16 one cell does exactly that,
and in the **uncentered** basis the older `Results/` quiver uses, the imine panel's
constructed point is nearest `alkyl fluoride` and the true `amine` falls to rank 3.

## Drawing choices worth knowing

**The projection is mean-centered**, unlike
`functional_group_analogy_carbon_matched.fit_uncentered_3d_basis`. Uncentered, the anisotropic
mean direction owns PC-1 and every group's arrow points the same way — the hairball in
`Results/.../diff_vector_quiver_3d_layer_*.png`. Offsets are translation-invariant so the
parallelogram loses nothing; what is given up is the origin reading as the alkane baseline.
`analogy/README.md` §8 has the numbers.

**No arrows from the origin.** Since the origin is the cloud centroid, a spoke to it means
nothing — and drawn anyway they were the longest lines on the panel and buried the offsets.

**The camera is chosen per panel**, by maximising the smallest on-screen separation among the
five points that carry the claim, skipping azimuths within 14° of a right angle (there two
panes go edge-on and mplot3d stacks two axis labels down one edge). A single fixed angle put
`alcohol` on top of `aldehyde`. The camera changes no coordinate, and the sidecar records the
angle it settled on.

**Labels are placed on screen, not in the data**, with a leader line back to their marker and
a relaxation pass that pushes overlapping names apart. Where the analogy is tight, `a2` and `b2`
land within a marker's width of each other — that closeness is the result, so the labels have
to separate even though the points cannot.

These are exploratory, not manuscript figures. There is no `paper/figures/` counterpart.
