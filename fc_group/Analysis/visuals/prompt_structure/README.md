# Prompt wording against chemistry, by depth

```bash
python fc_group/Analysis/paper/make_visuals_prompt_structure.py [--column raw]
```

**`label_agreement_depth.png`** — two panels. For each of the 920 prompts, the share of its 10
nearest neighbours **in activation space** carrying the same prompt template (left) or the same
functional group (right). One line per checkpoint, every layer, each panel against its own
**random-neighbour floor** — 0.099 for the 10 balanced templates, 0.054 for the 20 unbalanced
groups, so the same height means very different things in the two panels.

## How kNN label agreement is calculated

Per `(model, layer)`, over the 920 activation rows (92 molecules × 10 prompt templates), in
`knn_agreement` (`Analysis/prompt_structure/analyze_prompt_structure.py:62`):

1. **Mean-centre**, `acts - acts.mean(axis=0)`. The headline PNG is centred; `--column raw`
   skips this. Not housekeeping — the residual stream is strongly anisotropic (mean pairwise
   cosine 0.944 in base at layer 31), so on raw vectors the shared mean direction drives the
   neighbour ranking and every model is scored partly on how collapsed its space is.
2. **L2-normalise the rows and take `Z @ Z.T`.** Cosine on normalised rows ranks identically to
   Euclidean distance; the dot product is the cheap way to get there for 920 points.
3. **Set the diagonal to `-inf`**, so a point is never its own neighbour.
4. **Take each point's k = 10 nearest** by that similarity.
5. **Agreement = `mean(lab[idx] == lab[:, None])`** — the fraction of all 9,200
   (point, neighbour) pairs whose labels match. One flat average over pairs, *not* a mean of
   per-point rates.

**What it means.** The probe asks whether a linear reader *can* recover a label. This asks
something weaker and more intrinsic: what the representation's own neighbourhood structure is
already organised by, with no classifier fitted and nothing trained. A value of 0.99 on the
left panel says a prompt's nearest neighbours in activation space are overwhelmingly other
prompts sharing its wording — the model has filed it by phrasing. The crossover is where that
stops being true and chemistry takes over.

**"Same functional group" means the exact fine group, not the heteroatom family.** The right
panel uses the raw `functional_group` column — 20 classes (`alcohol`, `ether`, `aldehyde`,
`ketone`, `carboxylic acid`, `ester`, `amine`, `imine`, `nitrile`, `amide`, `nitro`, `thiol`,
`thioether`, `sulfoxide`, `sulfone`, the four alkyl halides, `none (alkane)`). An alcohol
neighbouring an ester counts as **disagreement**. The 5-family reading is a separate column,
`knn_coarse_family_*`, released but not drawn — see the dropped-panel note below. That is why
the two floors are so far apart: 0.054 across 20 groups against 0.239 across 5 families.

**One thing this panel cannot separate on its own.** Molecule identity is nested inside
functional group — a point's own molecule's other 9 templates are automatically same-group — so
with k = 10 a purely molecule-driven neighbourhood would score ≥ 0.9 on the right panel while
knowing no chemistry. There is no same-molecule column to check that against directly, but the
left panel bounds it: a same-*template* neighbour is necessarily a *different* molecule, since
each (molecule, template) pair is one point. Chem-F at layer 31 holds 0.489 template agreement,
so at most 51.1% of its neighbour slots can be same-molecule against 0.843 group agreement —
leaving at least **0.332** that must come from different-molecule, same-group neighbours,
against a 0.054 floor. The chemistry signal survives the confound, and the figure as drawn is
what lets you verify that rather than assume it.

## The `Random neighbours` floor

The dotted line is drawn as `Random neighbours 0.10` / `Random neighbours 0.05`, and the
wording is load-bearing. `chance_floor`
(`fc_group/Analysis/prompt_structure/analyze_prompt_structure.py:81`) is *"P(a uniformly random
OTHER point shares this point's label)"* — the agreement 10 neighbours drawn at random would
produce, which is just the label's base rate. Nothing guesses a label, so this is **not** the
same kind of quantity as `probe_depth.png`'s `Random guess` line and is deliberately not worded
like it. The function and the CSV column keep the name `chance_*`; only the drawn label
changed.

## Reading the curves

Left to right: the first half of every stack is organised by wording (template agreement
~0.99 through layer 15), chemical structure takes over around layer 19, and only then do the
checkpoints separate — every chemistry fine-tune ends below base on template and above it on
fine groups, while reason tracks base on both.

## The dropped heteroatom-family panel

A **heteroatom-family** panel used to sit between these two and was dropped. `COARSE_MAP` files
amide and nitro under nitrogen and the sulfoxides under sulfur — by the atom that names the
group, not the one it contains — so chemically sensible neighbours score as disagreement there
and the panel invites misreading. Its columns are still in the CSV, and they are what the
layer-19 crossover is measured against; see the analysis README. What that panel established
still holds and is worth knowing while reading these two: family agreement saturates at
0.90–0.94 for *every* checkpoint, so the separation here is not "chemistry models learn the
families" — base already has those — but "chemistry models suppress wording and resolve within
family".

`--column raw` redraws from the uncentred neighbour ranking. It is not the headline — see the
centring note in `Analysis/prompt_structure/README.md` — but the two agree on this data, and
the flag exists so that can be checked rather than believed.

The full argument, the k-sensitivity analysis, and what in it is **not** robust live in
[`fc_group/Analysis/prompt_structure/README.md`](../../prompt_structure/README.md). Read that
before quoting anything here; in particular, `reason`'s rank against base is not k-stable and
should not be stated as a result.

Companion to `visuals/tsne/` (the same question at five layers, measured on 2-D coordinates)
and to `f1_probe_depth` (whether a linear reader *can* recover the family, which happens
several layers earlier than the neighbourhood structure reorganising).
