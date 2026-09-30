#!/usr/bin/env python3
"""Dots where the sample is small, a violin where it is not.

    python fc_group/Analysis/paper/make_visuals_retrieval_mixed.py [--layer 31] [--out-dir DIR]
    -> fc_group/Analysis/visuals/retrieval/rankmixed.png

The third variant, beside `rankdist.png` (all dots) and `rankviolin.png` (all violins). Both
of those apply one mark to three panels whose sample sizes differ by a factor of nineteen;
this one picks the mark per panel:

    inter-group     n=20    2-3 distinct rank values per model   -> DOTS
    halide          n=12    2-5 distinct                         -> DOTS
    chain-length    n=228   9-25 distinct                        -> VIOLIN

The rule behind `MARKS` is the only thing this file decides. A density estimate needs a
sample that can support one. At n=12 over two distinct values -- chem-dfm's halide column is
`{1: 8, 17: 4}` -- a KDE draws a continuous body across ranks 2-16 where no trial landed,
which is invention rather than smoothing. At n=228 over 9-25 values the shape is real, and
the fat 4-to-20 bodies on chem-r and chem-faithful against reason's narrow spike is a genuine
difference that twenty overlapping dots communicate less well.

THE COST, STATED: one figure now carries two mark types, so a reader cannot compare the
width of a violin against the width of a dot row. Nothing here invites that -- the panels
already have separate y scales for the same reason, since a rank means nothing without the
pool it is out of -- but it is a real thing to know before quoting the figure.

Neither panel function is reimplemented. `dist_panel` and `violin_panel` are imported from
the two single-mark scripts and called unchanged, so a panel here is pixel-identical to the
same panel there, and a fix to either propagates without anyone remembering to copy it.

Reads the same committed CSVs -- `geometry/data/retrieval_rank_hist.csv` and
`geometry/data/retrieval_by_axis.csv`, never `Results/`.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from _common import BEHAVIOURAL, FIGURE_LAYER, REPO  # noqa: E402
from make_visuals_probe import COLOR5, DISPLAY5, style  # noqa: E402
from make_visuals_retrieval import (AXES, DEFAULT_VISUALS, RETR_LABEL,  # noqa: E402
                                    RETR_SIZE, RETR_TICK, read_axis_rows)
from make_visuals_retrieval_dist import (DIST_LEGEND, HIT_FS, MEAN,  # noqa: E402
                                         TICK, annotate_hit1, dist_panel, rank_hist)
from make_visuals_retrieval_violin import BODY, violin_panel  # noqa: E402

# The one decision this file makes. Keyed by the CSV's axis value.
MARKS = {'inter_group': 'dots', 'halide': 'dots', 'carbon_ladder': 'violin'}


def mixed_figure(by, hist, labels, layer):
    """Three panels, each drawn with the mark its sample size can support."""
    fig, axs = plt.subplots(1, 3, figsize=RETR_SIZE)
    for j, (ax, a) in enumerate(zip(axs, AXES)):
        panel = violin_panel if MARKS[a] == 'violin' else dist_panel
        panel(ax, by, hist, a, labels[a])
        annotate_hit1(ax, by, a, show_key=(j == 0))
        # Says which mark the panel uses, because the figure mixes them and a reader should
        # not have to infer it from the shapes. Sits on the caption line, under the axis
        # name, where it reads as a property of the panel rather than of the data.
        ax.set_xlabel(f'{labels[a]}\n{MARKS[a]}', fontsize=RETR_TICK)

    axs[0].set_ylabel('Rank (log scale)', fontsize=RETR_LABEL)

    # Model swatches are dots: two of the three panels draw dots, and a circle reads at
    # legend size where a 0.55-alpha rectangle goes muddy.
    handles = [plt.Line2D([], [], marker='o', markersize=5, linestyle='none',
                          color=COLOR5[m]) for m in BEHAVIOURAL]
    handles += [plt.Line2D([], [], **TICK), plt.Line2D([], [], **MEAN),
                plt.Line2D([], [], color='0.35', linewidth=1.2, linestyle='--')]
    names = [DISPLAY5[m] for m in BEHAVIOURAL] + ['Median', 'Mean', 'Random']
    fig.legend(handles, names, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(names), fontsize=DIST_LEGEND, handlelength=1.6,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.subplots_adjust(wspace=0.26)
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layer', type=int, default=FIGURE_LAYER)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    args = p.parse_args()
    style()

    by, labels = read_axis_rows(args.layer)
    hist = rank_hist(args.layer)
    os.makedirs(args.out_dir, exist_ok=True)
    fig = mixed_figure(by, hist, labels, args.layer)
    path = os.path.join(args.out_dir, f'rankmixed_L{args.layer}.png'
                        if args.layer != FIGURE_LAYER else 'rankmixed.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {os.path.relpath(path, REPO)}')

    for a in AXES:
        ns = {sum(hist[(m, a)].values()) for m in BEHAVIOURAL}
        dis = sorted({len(hist[(m, a)]) for m in BEHAVIOURAL})
        print(f'{a:14s} {MARKS[a]:6s}  n={ns.pop():3d}  '
              f'distinct per model {dis[0]}-{dis[-1]}')


if __name__ == '__main__':
    main()
