#!/usr/bin/env python3
"""Exploratory PNGs of analogy retrieval at one layer, scored by mean rank.

    python fc_group/Analysis/paper/make_visuals_retrieval_rank.py [--layer 31] [--out-dir DIR]

`make_visuals_retrieval.py` draws the same five checkpoints on the same three axes by
hit@k -- how *often* `b2 + (a1 - a2)` lands the answer at rank 1, 2 or 3. A hit rate cannot
say how badly a miss misses, and that blindness hides a result. At layer 31 on the halide
column `chemdfm` ties for the best hit@1 (0.67) and has the worst mean rank of the five
(6.33): when it misses, it misses by a mile. `base` is the mirror image, hit@1 0.50 and mean
rank 4.08. Nothing in the hit@k family distinguishes those two failure modes.

Writes `meanrank.png`. Four things differ from the hit figures, and each is forced by what
a rank is:

  * three real subplots, not three tick groups on one axes, because the panels CANNOT share
    a y-limit. Chance rank is the pool size, so it is 8.6 on the inter-group tree, 8.8 on the
    halide column and 37.3 on the carbon ladder. One scale would flatten the first two;
  * the y-floor is rank 1, not 0. Rank 0 does not exist, and a bar drawn from 0 would spend
    most of its length on the part of the scale no model can reach. Each bar measures excess
    rank above perfect retrieval;
  * up is worse. The y-label carries the direction rather than the geometry. This inverts
    `analogy_retrieval.plot_rank_by_group`, which flips the axis so better is up, and the
    choice is deliberate: these panels get read beside `hit1.png`, where the chance line sits
    near the floor and bars rise away from it. Flipping would hang the bars from the ceiling
    and put chance at the bottom, which reads as the opposite of what it means;
  * every bar carries a dark tick at its MEDIAN rank, because a mean rank alone cannot say
    whether a gap is the whole distribution or a couple of catastrophic trials. A tick
    pinned at the floor under a tall bar means the model is usually right and occasionally
    hopeless; a tick that rises with the bar means the ordering really did get worse.

That distinction is not hypothetical, which is why the tick is in the figure rather than in
the README. On the inter-group axis at L31 chem-r and chem-faithful mean 2.55 and 2.60
against base's 1.20 -- and all three have median 1. The entire gap is sulfoxide, 2 trials of
20, which base ranks 2 and the two fine-tunes rank 15 and 16 out of a ~16-candidate pool.
Drop that one group and the three are 1.11 / 1.17 / 1.11, indistinguishable. On the carbon
ladder the same two models mean 6.10 and 6.03 against 2.33 AND carry medians of 2 against
base's 1: there the ordering is genuinely worse, on 50 of 76 (group, chain-length) cells.
Two panels, two opposite stories, one metric -- the tick is what separates them.

The value label above every bar is load-bearing for a different reason: with chance at 37.3
in frame the ladder panel's bars are short, and the labels are what keep a 1.81-to-6.10
spread readable.

Identity works exactly as in the hit figures: no markers, but the five bars appear in the
same left-to-right order in every panel, and the legend is in that order.

Every value is read from the committed `geometry/data/retrieval_by_axis.csv` -- never
`Results/`. `mean_rank` and `random_mean_rank` are pooled there by
`analyze_geometry.retrieval()`, trial-weighted across each axis's groups, out of columns
`analogy_retrieval.summarize()` has emitted all along. `median_rank` is taken over the
per-trial ranks instead -- a trial-weighted mean of per-group medians is not a median.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from _common import BEHAVIOURAL, FIGURE_LAYER, REPO, num  # noqa: E402
from make_visuals_probe import COLOR5, DISPLAY5, SHORT5, recessive, style  # noqa: E402
from make_visuals_retrieval import (AXES, DEFAULT_VISUALS, RETR_LABEL,  # noqa: E402
                                    RETR_LEGEND, RETR_REF, RETR_SIZE, RETR_TICK,
                                    RETR_VALUE, read_axis_rows)

# Wider than the hit figure's bars: each panel holds five bars instead of fifteen, so the
# same 0.88 surface gap falls out of a much larger slot.
BAR_W = 0.72
# Headroom above the chance line, which is what sets every panel's ceiling: chance rank
# exceeds every model's mean rank on all three axes. Measured from rank 1, the floor, not 0.
PAD_CHANCE = 1.06
# The median tick overhangs its bar on both sides. Load-bearing rather than decorative: the
# median is exactly 1.00 on most bars, where a tick cut to the bar's width would lie flush
# along the baseline spine and disappear -- and a median pinned to the floor under a tall bar
# is the single most important thing this figure has to say.
TICK_W = 1.15
# The white halo is what makes the tick survive every background it has to cross: a
# saturated bar, the light panel, and -- where the median is 1.00, which is most bars -- the
# black baseline spine it would otherwise be indistinguishable from.
TICK = dict(color='0.12', linewidth=1.6, zorder=4, solid_capstyle='butt',
            path_effects=[pe.withStroke(linewidth=3.4, foreground='white')])


def rank_panel(ax, by, axis, label):
    """One axis's five models as mean rank, each bar ticked at its median."""
    vals = [num(by[(m, axis)], 'mean_rank') for m in BEHAVIOURAL]
    meds = [num(by[(m, axis)], 'median_rank') for m in BEHAVIOURAL]
    # Model-independent by construction -- same trials, different embeddings -- so the first
    # model's row is as good as any, the same shortcut hit_figure takes for random_hit@k.
    chance = num(by[(BEHAVIOURAL[0], axis)], 'random_mean_rank')

    top = 1 + (chance - 1) * PAD_CHANCE
    span = top - 1

    bars = []
    for i, (m, v, med) in enumerate(zip(BEHAVIOURAL, vals, meds)):
        # bottom=1: the bar is the excess over perfect retrieval, not the rank itself.
        bars.append(ax.bar(i, v - 1, BAR_W, bottom=1, color=COLOR5[m], label=DISPLAY5[m],
                           linewidth=0))
        ax.text(i, v + span * 0.02, f'{v:.2f}', ha='center', va='bottom',
                fontsize=RETR_VALUE)
        # Unlabelled on purpose: the bar already carries a number, and a second one per bar
        # would crowd the ladder panel past reading. The tick's position against the bar top
        # is the message, not its value.
        ax.plot([i - BAR_W * TICK_W / 2, i + BAR_W * TICK_W / 2], [med, med], **TICK)

    ax.axhline(chance, color='0.35', linewidth=1.2, linestyle='--', zorder=3)

    ax.set_xlim(-0.75, len(BEHAVIOURAL) - 0.25)
    ax.set_ylim(1, top)
    ax.set_xticks([])
    ax.set_xlabel(label, fontsize=RETR_TICK)
    ax.tick_params(labelsize=RETR_TICK)
    recessive(ax)
    ax.grid(axis='x', visible=False)
    return bars


def rank_figure(by, labels, layer):
    """Three panels, one per retrieval axis, each on its own rank scale."""
    fig, axs = plt.subplots(1, 3, figsize=RETR_SIZE)
    for ax, a in zip(axs, AXES):
        bars = rank_panel(ax, by, a, labels[a])

    axs[0].set_ylabel('mean rank (lower is better)', fontsize=RETR_LABEL)
    # Below the chance line, never on it: chance sits at ~94% of the panel's span by
    # construction, so the corner this annotation occupies in the hit figures is taken.
    axs[-1].text(0.97, 0.86, f'Layer {layer}', transform=axs[-1].transAxes, ha='right',
                 va='top', fontsize=RETR_REF, color='0.45')

    # Handles from the last panel's bars, in BEHAVIOURAL order, with the two reference keys
    # appended rather than left to matplotlib -- sorting Line2D before BarContainer would
    # put them first and break the legend's correspondence to the left-to-right bar order.
    handles = list(bars) + [
        axs[0].plot([], [], **TICK)[0],
        axs[0].plot([], [], color='0.35', linewidth=1.2, linestyle='--')[0]]
    names = [DISPLAY5[m] for m in BEHAVIOURAL] + ['median', 'chance']
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(0.5, 1.045),
               ncol=len(names), fontsize=RETR_LEGEND, handlelength=1.6,
               columnspacing=1.3)
    fig.tight_layout()
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
    fig = rank_figure(by, labels, args.layer)
    path = os.path.join(args.out_dir, 'meanrank.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {os.path.relpath(path, REPO)}')

    for a in AXES:
        cells = '  '.join(
            f'{SHORT5[m]} {num(by[(m, a)], "mean_rank"):.2f}'
            f'/{num(by[(m, a)], "median_rank"):.1f}' for m in BEHAVIOURAL)
        chance = num(by[(BEHAVIOURAL[0], a)], 'random_mean_rank')
        print(f'{a:14s} mean/median rank   {cells}   chance {chance:.2f}')


if __name__ == '__main__':
    main()
