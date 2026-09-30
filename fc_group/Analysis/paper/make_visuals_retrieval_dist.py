#!/usr/bin/env python3
"""Every retrieval trial as one dot: the rank distribution behind `meanrank.png`.

    python fc_group/Analysis/paper/make_visuals_retrieval_dist.py [--layer 31] [--out-dir DIR]

`meanrank.png` reduces each model on each axis to a mean and a median. Those two numbers
disagree sharply on some panels, and the disagreement is the result -- so this figure draws
the sample itself. Writes `rankdist.png`.

**Why dots and not a violin.** A violin was the obvious thing to reach for and it is the
wrong tool here. These are small integers piled on rank 1, not a continuous variable:

  inter-group    n=20   2-3 distinct rank values per model   base is {1: 16, 2: 4}
  halide column  n=12   2-5 distinct values                  chemdfm is {1: 8, 17: 4}
  ladder         n=228  9-25 distinct values

A KDE over `{1: 8, 17: 4}` draws a smooth body across ranks 2-16, where not one trial
landed, and spills density below rank 1, which cannot exist. On two of the three panels the
violin would invent exactly the structure the figure is meant to adjudicate. Dots cannot
lie about a sample that small: at n=20 you are looking at all twenty.

**How a column is built.** Trials tied at the same rank are spread horizontally, evenly, at
a fixed spacing that clamps once the row fills the column. So a row's width reads as its
count -- two tied trials are visibly two dots, a hundred are a saturated bar -- and the
column as a whole is a discrete density built without smoothing anything. No random jitter:
the placement is deterministic, so re-running cannot reshuffle the picture.

Each column also carries the two summaries `meanrank.png` draws, in the same encoding it
uses: a dark tick at the median, and an open circle at the mean. Seeing them against the
dots is the point -- a mean floating far above a median that sits on 1, with a couple of
lonely dots up near the chance line, is a tail, not an ordering.

y is logarithmic. Ranks span 1 to 63 here and the interesting structure is all at the
bottom; a linear axis would crush the 1-2-3 band, where most trials sit, into nothing.

Panels keep their own y-limits for the reason `meanrank.png` does: a rank is only meaningful
against its pool, and the pool is ~16 candidates on the first two axes and ~74 on the
ladder. Rank 8 is chance on one and good on another.

Read from the committed `geometry/data/retrieval_rank_hist.csv` and
`geometry/data/retrieval_by_axis.csv` -- never `Results/`. The histogram is lossless for
integer ranks, so repeating each rank `n` times reconstructs the exact sample that produced
`mean_rank` and `median_rank`.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from _common import BEHAVIOURAL, FIGURE_LAYER, REPO, load, num  # noqa: E402
from make_visuals_probe import COLOR5, DISPLAY5, SHORT5, recessive, style  # noqa: E402
from make_visuals_retrieval import (AXES, DEFAULT_VISUALS, RETR_LABEL,  # noqa: E402
                                    RETR_SIZE, RETR_TICK, read_axis_rows)

# Column width, and the spacing between two tied dots before the row has to clamp to fit.
# W/12 keeps a handful of ties legible as separate dots, which is the whole point at n=12.
COL_W = 0.60
DOT_GAP = COL_W / 12
DOT = dict(s=7, alpha=0.72, linewidths=0, zorder=3)
# Same encoding as meanrank.png, deliberately. Two departures, both forced by the dots:
# the tick is drawn BENEATH them and reaches wider than the column, so on the rows where it
# matters most -- a median of 1 under a full row of tied trials -- it reads as two ends
# poking out rather than as a black bar laid over the data it is supposed to summarise; and
# it stops well short of the next column, which at half a unit apart would otherwise fuse
# five per-model medians into one rule across the panel.
TICK_HALF = 0.38
TICK = dict(color='0.12', linewidth=1.6, zorder=2, solid_capstyle='butt',
            path_effects=[pe.withStroke(linewidth=3.4, foreground='white')])
MEAN = dict(marker='o', markersize=4.6, markerfacecolor='white', markeredgecolor='0.12',
            markeredgewidth=1.2, linestyle='none', zorder=6)
# Ticks worth labelling on a log rank axis; filtered to the ones a panel actually reaches.
YTICKS = [1, 2, 3, 5, 10, 20, 40, 70]

# hit@1 printed above each column. Small and grey: it is a caption on the column, not a
# second data series competing with the dots.
HIT_FS = 11


def annotate_hit1(ax, by, axis, show_key=False):
    """Write each column's hit@1 just above the panel, in data-x / axes-y.

    Read from the RELEASED `hit1` column rather than recounted as the rank-1 share of the
    histogram. Both are available and they agree to 3.5e-07 (CSV rounding), but `hit1` is
    what `hit1.png` draws, and computing it a second way here would only create a second
    chance for the two figures to disagree.
    """
    tr = ax.get_xaxis_transform()
    for i, m in enumerate(BEHAVIOURAL):
        ax.text(i, 1.012, f"{num(by[(m, axis)], 'hit1'):.2f}", transform=tr,
                ha='center', va='bottom', fontsize=HIT_FS, color='0.25')
    if show_key:
        # In the left margin, not inside the panel: at x=-0.74 data it overlapped the first
        # column's own value, which sits centred on x=0. `savefig.bbox='tight'` grows the
        # canvas to include it, so nothing is clipped.
        ax.text(-0.03, 1.012, 'hit@1', transform=ax.transAxes, ha='right', va='bottom',
                fontsize=HIT_FS, color='0.45', style='italic')


# Larger than the shared RETR_LEGEND (12) that the other retrieval figures use. This legend
# carries eight entries against their six and is the only place the median tick, the mean
# marker and the reference line are explained, so it is doing more work and gets more type.
DIST_LEGEND = 14


def rank_hist(layer):
    """`(model, axis) -> {rank: count}` at one layer, from the committed histogram."""
    out = {}
    for r in load('geometry/data/retrieval_rank_hist.csv'):
        if int(r['layer']) == layer:
            out.setdefault((r['model'], r['axis']), {})[int(r['rank'])] = int(r['n'])
    missing = [(SHORT5.get(m, m), a) for m in BEHAVIOURAL for a in AXES
               if (m, a) not in out]
    if missing:
        raise SystemExit(
            'missing ' + ', '.join(f'{m}/{a}' for m, a in missing) + ' -- re-run '
            'fc_group/Analysis/geometry/analyze_geometry.py')
    return out


def dot_xs(centre, count):
    """`count` x-positions centred on `centre`, clamped to the column once it fills."""
    if count == 1:
        return [centre]
    gap = min(DOT_GAP, COL_W / (count - 1))
    return [centre + gap * (j - (count - 1) / 2) for j in range(count)]


def dist_panel(ax, by, hist, axis, label):
    """One axis: five columns of trial dots, each with its median tick and mean marker."""
    chance = num(by[(BEHAVIOURAL[0], axis)], 'random_mean_rank')
    # random_mean_rank is mean((n + 1) / 2) over the trials' pools, so this inverts back to
    # the mean pool size -- the largest rank the axis could in principle return.
    pool = 2 * chance - 1

    for i, m in enumerate(BEHAVIOURAL):
        counts = hist[(m, axis)]
        xs, ys = [], []
        for rank, c in sorted(counts.items()):
            xs.extend(dot_xs(i, c))
            ys.extend([rank] * c)
        ax.scatter(xs, ys, color=COLOR5[m], label=DISPLAY5[m], **DOT)

        med = num(by[(m, axis)], 'median_rank')
        ax.plot([i - TICK_HALF, i + TICK_HALF], [med, med], **TICK)
        ax.plot([i], [num(by[(m, axis)], 'mean_rank')], **MEAN)

    ax.axhline(chance, color='0.35', linewidth=1.2, linestyle='--', zorder=4)

    ax.set_yscale('log')
    ax.set_ylim(0.93, pool * 1.06)
    # The pool itself is the last tick: a rank means nothing without the count it is out
    # of, and this is where the panel says it. Dropped when a listed tick is already close
    # enough to double-label the same place.
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


def dist_figure(by, hist, labels, layer):
    """Three panels, one per retrieval axis, each on its own pool's rank scale."""
    fig, axs = plt.subplots(1, 3, figsize=RETR_SIZE)
    for j, (ax, a) in enumerate(zip(axs, AXES)):
        dist_panel(ax, by, hist, a, labels[a])
        annotate_hit1(ax, by, a, show_key=(j == 0))

    # '(log scale)', not '(log)': the quantity is a plain integer rank and the parenthetical
    # describes the AXIS, which the old wording made look like a unit.
    axs[0].set_ylabel('Rank (log scale)', fontsize=RETR_LABEL)

    # Built by hand in BEHAVIOURAL order: scatter handles come back tiny and half
    # transparent, which is right on the panel and unreadable in a legend.
    handles = [plt.Line2D([], [], marker='o', markersize=5, linestyle='none',
                          color=COLOR5[m]) for m in BEHAVIOURAL]
    handles += [plt.Line2D([], [], **TICK), plt.Line2D([], [], **MEAN),
                plt.Line2D([], [], color='0.35', linewidth=1.2, linestyle='--')]
    # 'Random', not 'chance': the dashed line is `random_mean_rank`, the rank the true
    # completion would average if the candidate pool were ordered at random. 'Chance' named
    # no mechanism; `probe_depth.png` and `label_agreement_depth.png` were reworded on the
    # same grounds, each to the strategy IT draws rather than to one shared word.
    names = [DISPLAY5[m] for m in BEHAVIOURAL] + ['Median', 'Mean', 'Random']
    # Below the panels, under the three axis captions. The `rect` reserves that strip before
    # tight_layout packs the axes, so the legend cannot overlap the captions; `wspace` is set
    # AFTER, because tight_layout would otherwise recompute it away.
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
    fig = dist_figure(by, hist, labels, args.layer)
    path = os.path.join(args.out_dir, 'rankdist.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {os.path.relpath(path, REPO)}')

    for a in AXES:
        for m in BEHAVIOURAL:
            counts = hist[(m, a)]
            n = sum(counts.values())
            tail = {r: c for r, c in counts.items() if r > 10}
            print(f'{a:14s} {SHORT5[m]:14s} n={n:3d}  distinct={len(counts):2d}  '
                  f'rank1={100 * counts.get(1, 0) / n:3.0f}%  '
                  f'tail>10: {tail if tail else "-"}')


if __name__ == '__main__':
    main()
