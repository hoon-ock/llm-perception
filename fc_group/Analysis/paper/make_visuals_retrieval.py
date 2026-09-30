#!/usr/bin/env python3
"""Exploratory PNGs of analogy retrieval at one layer: hit@1, hit@2 and hit@3.

    python fc_group/Analysis/paper/make_visuals_retrieval.py [--layer 31] [--out-dir DIR]

`make_figures.py` builds `paper/figures/f4_retrieval.pdf` -- grouped bars over the three
independent retrieval axes, each against its own chance line -- for **hit@1 only**, and for
the three models in `_common.MODELS`. This widens that figure both ways and writes it to
`fc_group/Analysis/visuals/retrieval/`, a working directory beside the analysis outputs and
deliberately not `paper/figures/`, so nothing here can reach the manuscript by accident:

  hit1.png    3 axes x 5 models, chance = random_hit1
  hit2.png    ...                        random_hit2
  hit3.png    ...                        random_hit3

Four things differ from the paper figure, all of them presentation:

  * five models instead of three, and one file per cutoff instead of one file;
  * a SHARED y-limit across the three files. hit@k is monotone in k by construction, so the
    three PNGs are only worth anything read against each other, and a per-figure autoscale
    would draw hit@1 as tall as hit@3;
  * the trial count is on each x tick label. That is load-bearing rather than decorative: at
    k=2 two models sit at exactly 1.00 on the inter-group axis, off twenty trials, and a bar
    pinned to the ceiling has to say how few trials put it there. The paper figure can omit n
    because it draws k=1 only, where nothing saturates;
  * larger type, read on its own rather than as a column-width float.

Identity does not rest on hue alone even though bars carry no marker: the five bars appear in
the same left-to-right order in every group of every figure, and the legend is in that order.

Every value is read from the committed `geometry/data/retrieval_by_axis.csv` -- never
`Results/`, which is gitignored -- so this figure and `f4_retrieval` cannot disagree about a
number. What that CSV carries, `analyze_geometry.retrieval()` pooled by trial count across
each axis's groups, k-by-k with its own baseline.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import (ANALYSIS, BEHAVIOURAL, FIGURE_LAYER, REPO,  # noqa: E402
                     load, num)
from make_visuals_probe import COLOR5, DISPLAY5, SHORT5, recessive, style  # noqa: E402

DEFAULT_VISUALS = os.path.join(ANALYSIS, 'visuals', 'retrieval')

# The three trees, in the order f4_retrieval draws them: widest candidate pool first, the
# one-column halide probe second, the chain-length ladder last.
AXES = ['inter_group', 'halide', 'carbon_ladder']
# Display names, shared by every retrieval figure through `read_axis_rows` -- hit@k, mean
# rank and the rank distribution all label the same three axes, so they are spelled once.
# The AXES keys ('halide', ...) are the CSV values and must not drift.
PRETTY = {'inter_group': 'Inter-group', 'halide': 'Halide ladder',
          'carbon_ladder': 'Chain-length ladder'}
CUTOFFS = [1, 2, 3]

RETR_SIZE = (10.0, 4.8)
RETR_LABEL = 14
RETR_TICK = 12.5
RETR_LEGEND = 12
RETR_VALUE = 10
RETR_REF = 11
# Five bars where the paper has three. The 0.88 factor is f4_retrieval's, and is what leaves
# a surface gap between adjacent bars instead of fusing them into one block.
BAR_W = 0.16
# Shared across all three files; see the docstring. The headroom is for the value labels
# above the bars that reach 1.00.
YLIM = (0, 1.14)


def read_axis_rows(layer):
    """`(model, axis) -> row` at one layer, plus the tick label each axis earns.

    The group and trial counts in the labels are read off the rows rather than typed, so a
    re-run that changes the trial set cannot leave a stale n on the figure.
    """
    rows = [r for r in load('geometry/data/retrieval_by_axis.csv')
            if int(r['layer']) == layer]
    if not rows:
        raise SystemExit(
            f'no rows at layer {layer} in geometry/data/retrieval_by_axis.csv -- '
            'the snapshot carries layers 0, 8, 16, 24, 31')
    by = {(r['model'], r['axis']): r for r in rows}
    missing = [(SHORT5.get(m, m), a) for m in BEHAVIOURAL for a in AXES
               if (m, a) not in by]
    if missing:
        raise SystemExit(
            'missing ' + ', '.join(f'{m}/{a}' for m, a in missing) + ' -- re-run '
            'fc_group/Analysis/geometry/analyze_geometry.py')

    labels = {}
    for a in AXES:
        # Identical across models by construction -- same trials, different embeddings --
        # so asserting it is cheap and catches a half-finished results tree.
        ns = {(r['n_groups'], r['n_trials']) for (m, ax), r in by.items() if ax == a}
        if len(ns) != 1:
            raise SystemExit(f'{a}: models disagree on the trial set at layer {layer}: {ns}')
        g, n = ns.pop()
        labels[a] = f'{PRETTY[a]}\n({g} groups, {n} trials)'
    return by, labels


def hit_figure(by, labels, k, layer):
    """One axes: hit@k over the three axes, five models, each axis against its own chance."""
    fig, ax = plt.subplots(figsize=RETR_SIZE)
    x = np.arange(len(AXES))

    for i, m in enumerate(BEHAVIOURAL):
        vals = [num(by[(m, a)], f'hit{k}') for a in AXES]
        pos = x + (i - 2) * BAR_W
        ax.bar(pos, vals, BAR_W * 0.88, color=COLOR5[m], label=DISPLAY5[m], linewidth=0)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.016, f'{v:.2f}', ha='center', va='bottom',
                    fontsize=RETR_VALUE)

    # One chance segment per axis, spanning that axis's bars only. The pools differ per tree
    # -- 19 groups, the halide column, the whole ladder -- so a single line across the figure
    # would invite reading one axis's bar against another's baseline.
    for i, a in enumerate(AXES):
        rnd = num(by[(BEHAVIOURAL[0], a)], f'random_hit{k}')
        ax.plot([i - 2.9 * BAR_W, i + 2.9 * BAR_W], [rnd, rnd], color='0.35',
                linewidth=1.2, linestyle='--', zorder=3)
    chance_proxy = ax.plot([], [], color='0.35', linewidth=1.2, linestyle='--')[0]

    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in AXES])
    ax.set_ylabel(f'hit@{k}', fontsize=RETR_LABEL)
    ax.set_ylim(*YLIM)
    ax.tick_params(labelsize=RETR_TICK)
    # Outside the axes: at k=3 the bars fill the panel and leave no corner for a legend, and
    # the three figures must place it identically to be read side by side.
    # Handles passed explicitly: left to itself matplotlib returns Line2D handles before
    # BarContainers, which puts `chance` first and breaks the left-to-right correspondence
    # between the legend and the bars -- the figure's only secondary encoding of identity.
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles + [chance_proxy], [DISPLAY5[m] for m in BEHAVIOURAL] + ['chance'],
              loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=6,
              fontsize=RETR_LEGEND, handlelength=1.6, columnspacing=1.3)
    ax.text(0.995, 0.975, f'Layer {layer}', transform=ax.transAxes, ha='right', va='top',
            fontsize=RETR_REF, color='0.45')
    recessive(ax)
    ax.grid(axis='x', visible=False)
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layer', type=int, default=FIGURE_LAYER)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    args = p.parse_args()
    style()

    by, labels = read_axis_rows(args.layer)
    os.makedirs(args.out_dir, exist_ok=True)
    for k in CUTOFFS:
        fig = hit_figure(by, labels, k, args.layer)
        path = os.path.join(args.out_dir, f'hit{k}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {os.path.relpath(path, REPO)}')

    for a in AXES:
        cells = '  '.join(
            f'{SHORT5[m]} ' + '/'.join(f"{num(by[(m, a)], f'hit{k}'):.3f}" for k in CUTOFFS)
            for m in BEHAVIOURAL)
        chance = '/'.join(f"{num(by[(BEHAVIOURAL[0], a)], f'random_hit{k}'):.3f}"
                          for k in CUTOFFS)
        print(f'{a:14s} hit@1/2/3   {cells}   chance {chance}')


if __name__ == '__main__':
    main()
