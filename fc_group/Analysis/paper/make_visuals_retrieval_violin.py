#!/usr/bin/env python3
"""`rankdist.png` as violins instead of dots -- the variant, built to be looked at.

    python fc_group/Analysis/paper/make_visuals_retrieval_violin.py [--layer 31] [--out-dir DIR]
    -> fc_group/Analysis/visuals/retrieval/rankviolin.png

A SEPARATE script and a SEPARATE output. `make_visuals_retrieval_dist.py` and its
`rankdist.png` are untouched: the dot figure is the one the README argues for, and this
exists beside it so the two can be compared rather than one replacing the other on a guess.

READ THIS BEFORE USING THE OUTPUT. `rankdist.png`'s docstring argues at length that a violin
is the wrong tool on two of these three panels, and building it does not make the argument
wrong. At layer 31 the samples are:

    inter-group     n=20    2-3 distinct rank values per model
    halide          n=12    2-5 distinct       e.g. chem-dfm is {1: 8, 17: 4}
    chain-length    n=228   9-25 distinct

A density over `{1: 8, 17: 4}` puts a continuous body across ranks 2-16, where not one trial
landed. That is invention, not smoothing, and it is exactly the structure the figure is meant
to adjudicate. Only the ladder panel has a sample a density estimate can honestly describe,
and even there 54-64% of trials sit on rank 1, which a violin renders as a bulge rather than
as the number it is.

Two things are done differently from a default violin, both because the default is worse:

  * THE KDE IS FITTED IN LOG10(RANK), not in rank. The y axis is logarithmic, so a density
    estimated on raw ranks and drawn on a log axis is stretched at the bottom and squashed at
    the top -- the picture would carry a distortion that is in the plotting, not the data.
  * THE BODY IS CLIPPED TO THE OBSERVED RANGE. Evaluating the KDE only on
    [min(rank), max(rank)] keeps it from spilling below rank 1, which cannot exist.
    `matplotlib.violinplot` already does this; `seaborn` does not without `cut=0`, which is
    why this draws the polygons itself rather than calling either.

Bandwidth is Scott's rule via `gaussian_kde`, stated rather than tuned: a bandwidth chosen to
make these panels look right would be a choice about the conclusion.

Reads the same committed CSVs as `rankdist.png` -- `geometry/data/retrieval_rank_hist.csv`
and `geometry/data/retrieval_by_axis.csv`, never `Results/`.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from _common import BEHAVIOURAL, FIGURE_LAYER, REPO, num  # noqa: E402
from make_visuals_probe import COLOR5, DISPLAY5, recessive, style  # noqa: E402
from make_visuals_retrieval import (AXES, DEFAULT_VISUALS, RETR_LABEL,  # noqa: E402
                                    RETR_SIZE, RETR_TICK, read_axis_rows)
from make_visuals_retrieval_dist import (DIST_LEGEND, MEAN, TICK,  # noqa: E402
                                         TICK_HALF, YTICKS, annotate_hit1, rank_hist)

# Half-width of the widest point of a violin, in column units. The columns sit one unit
# apart, so 0.42 leaves a visible gap between neighbours at full width.
HALF_W = 0.42
BODY = dict(alpha=0.55, linewidth=0.8, zorder=3)
# Points at which the density is evaluated across a column's observed range.
GRID_N = 200


def violin_xy(ranks, centre):
    """Polygon for one model's column: `(lo_x, hi_x, y)` in DATA coordinates.

    The KDE is fitted on log10(rank) because the axis is logarithmic -- see the module
    docstring -- and evaluated only between the smallest and largest rank observed, so the
    body cannot extend below rank 1. Returns `None` when the sample has no spread at all
    (`gaussian_kde` is singular there), which the caller draws as a flat bar instead.
    """
    x = np.log10(np.asarray(ranks, dtype=float))
    if np.allclose(x, x[0]):
        return None
    grid = np.linspace(x.min(), x.max(), GRID_N)
    dens = gaussian_kde(x)(grid)
    # Width-normalised: every violin peaks at HALF_W, so shapes are comparable across models
    # within a panel. n is equal across models on a given axis, so no information is lost.
    w = HALF_W * dens / dens.max()
    return centre - w, centre + w, 10 ** grid


def violin_panel(ax, by, hist, axis, label):
    """One axis: five violins, each with the median tick and mean marker `rankdist` uses."""
    chance = num(by[(BEHAVIOURAL[0], axis)], 'random_mean_rank')
    pool = 2 * chance - 1

    for i, m in enumerate(BEHAVIOURAL):
        counts = hist[(m, axis)]
        ranks = [r for r, c in sorted(counts.items()) for _ in range(c)]
        poly = violin_xy(ranks, i)
        if poly is None:
            # Every trial on one rank. A density is undefined; draw the bar that is true.
            r = ranks[0]
            ax.plot([i - HALF_W, i + HALF_W], [r, r], color=COLOR5[m], linewidth=3,
                    solid_capstyle='butt', zorder=3)
        else:
            lo, hi, y = poly
            ax.fill_betweenx(y, lo, hi, facecolor=COLOR5[m], edgecolor=COLOR5[m], **BODY)

        med = num(by[(m, axis)], 'median_rank')
        ax.plot([i - TICK_HALF, i + TICK_HALF], [med, med], **TICK)
        ax.plot([i], [num(by[(m, axis)], 'mean_rank')], **MEAN)

    ax.axhline(chance, color='0.35', linewidth=1.2, linestyle='--', zorder=4)

    ax.set_yscale('log')
    ax.set_ylim(0.93, pool * 1.06)
    ticks = [t for t in YTICKS if t <= pool * 0.8] + [round(pool)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(-0.75, len(BEHAVIOURAL) - 0.25)
    ax.set_xticks([])
    ax.set_xlabel(label, fontsize=RETR_TICK)
    ax.tick_params(labelsize=RETR_TICK)
    recessive(ax)
    ax.grid(axis='x', visible=False)


def violin_figure(by, hist, labels, layer):
    """Three panels, one per retrieval axis, each on its own pool's rank scale."""
    fig, axs = plt.subplots(1, 3, figsize=RETR_SIZE)
    for j, (ax, a) in enumerate(zip(axs, AXES)):
        violin_panel(ax, by, hist, a, labels[a])
        # Same helper `rankdist.png` uses, so the two variants cannot print different
        # hit@1 values for the same column.
        annotate_hit1(ax, by, a, show_key=(j == 0))

    axs[0].set_ylabel('Rank (log scale)', fontsize=RETR_LABEL)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR5[m], edgecolor=COLOR5[m],
                             alpha=BODY['alpha']) for m in BEHAVIOURAL]
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
    fig = violin_figure(by, hist, labels, args.layer)
    path = os.path.join(args.out_dir, f'rankviolin_L{args.layer}.png'
                        if args.layer != FIGURE_LAYER else 'rankviolin.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {os.path.relpath(path, REPO)}')

    # The sample each violin is drawn from, so the shape can be checked against the counts.
    for a in AXES:
        for m in BEHAVIOURAL:
            h = hist[(m, a)]
            n = sum(h.values())
            print(f'{a:14s} {DISPLAY5[m]:9s} n={n:3d}  distinct={len(h):2d}  '
                  f'rank1={h.get(1, 0) / n:.0%}')


if __name__ == '__main__':
    main()
