#!/usr/bin/env python3
"""Exploratory PNGs: per-functional-group generation quality against per-group representation.

    python fc_group/Analysis/paper/make_visuals_generation_vs_representation.py [--out-dir DIR]

Three figures, because the question has three parts and one panel cannot hold them.

`gen_vs_retrieval.png` is the question as asked -- analogy-retrieval hit@1 and mean rank at
layer 31 against per-group free-generation accuracy, one facet per checkpoint, every point
labelled with its functional group. It is a null, and it is drawn rather than tabulated because
the scatter shows *why*: `sulfone` sits at hit@1 0.000 with strict recall 0.875 and `imine` at
hit@1 0.667 with 0.275, so the point cloud has no orientation to fit. Groups are labelled
individually for the same reason -- an unlabelled null scatter is unfalsifiable by the reader.

`rho_by_layer.png` is the answer that depends on depth. Correlation is plotted against layer for
six predictors, because the one signal that exists is layer-specific and a single-layer figure
would either hide it or overstate it. Bootstrap CIs are shaded for the two cohesion predictors
only; shading all six makes the panel unreadable, and those two are the ones carrying the claim.
Panels for checkpoints whose per-group outcome is unreliable are hatched, and the hatch is the
most important mark in the figure: chem-r's split-half reliability on superclass credit is 0.30,
so its flat line is a measurement failure, not evidence of no relationship.

`group_response_composition.png` carries what the per-group spread actually tracks. Each group is
one stacked bar of the five adjudicated response types, with teacher-forced raw recall overlaid as
a tick. The four halides are the case to read: bar after bar of `underspecified`, under a tick
near the top. The model names the family and not the member, so strict scoring files it as a
failure while the forced-choice head ranks the exact class at 0.875-0.950. That is a resolution
effect in the scoring, not a representation deficit, and no correlation with an embedding metric
could have revealed it.

Every value is read from the committed `generation/data/group_*.csv` -- never `Results/`, matching
`make_visuals_retrieval*.py`. `analyze_generation_vs_representation.py` is what snapshots them.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ANALYSIS, BEHAVIOURAL, load, num  # noqa: E402
from make_visuals_probe import COLOR5, DISPLAY5, recessive, style  # noqa: E402

DEFAULT_OUT = os.path.join(ANALYSIS, 'visuals', 'generation')
HEADLINE_LAYER = 31
OUTCOME = 'superclass_credit'
# Free-generation strict recall plus the underspecified rate. Strict alone would put the four
# halides at the floor for answering "alkyl halide", which is a scoring-resolution artefact rather
# than anything an embedding metric could predict -- see the composition figure.
OUTCOME_LABEL = 'free-generation accuracy\n(strict + superclass credit)'

# Ordered as they are discussed: the two the question named, then the two that carry what signal
# there is, then the two degenerate ones kept visible so the reader sees them flat.
RHO_PREDICTORS = [
    ('retr_hit1', 'retrieval hit@1', '#0072B2', (None, None)),
    ('retr_mean_rank', 'retrieval mean rank (inv.)', '#56B4E9', (4, 1.5)),
    ('silhouette', 'group silhouette', '#D55E00', (None, None)),
    ('within_cos', 'within-group cosine', '#E69F00', (4, 1.5)),
    ('ladder_hit1', 'carbon-ladder hit@1', '#999999', (1, 1.5)),
    ('rival_margin', 'nearest-rival margin', '#CC79A7', (2, 1.2)),
]
CI_SHADED = {'silhouette', 'within_cos'}

# The composition figure is read on its own -- 5 stacked panels over 20 functional groups --
# and its type was set at paper scale, too small for that. Everything below is scaled ~1.5x
# from the original (7.4in wide, 6.4pt labels) TOGETHER WITH the figure size, so the 20
# rotated x labels and the 6-entry legend stay as uncrowded as they were. Raising the type
# without raising COMP_SIZE collides both.
COMP_SIZE_W = 11.1        # was 7.4
COMP_PANEL_H = 2.28       # per model row; was 1.52
COMP_YLABEL = 11.0        # model name down the left; was 7.4
COMP_TICK = 9.6           # x group names and y 0/0.5/1.0; were 6.4 and rcParams' 7.5
COMP_LEGEND = 9.6         # was 6.4
COMP_TF_MARK = 214        # scatter `s` for the teacher-forced tick. Marker size goes as
                          # sqrt(s), so 95 * 1.5**2 keeps the dash spanning the same
                          # fraction of a bar as before.

# Sentence case, capitalised: these are legend labels a reader sees, not field names. The
# first element of each tuple is the CSV column and is what must match the data.
RESPONSE_STACK = [
    ('strict', 'Answered, correct', '#0072B2'),
    ('underspecified', 'Superclass only', '#56B4E9'),
    ('answer_wrong', 'Answered, wrong', '#D55E00'),
    ('non_answer', 'Named no group', '#E69F00'),
    ('malformed', 'Malformed', '#BBBBBB'),
]
REF = dict(color='0.45', linewidth=0.9)


def read_join():
    rows = load('generation/data/group_representation_join.csv')
    return {(r['model'], int(r['layer']), r['functional_group']): r for r in rows}


def read_corr():
    return load('generation/data/group_correlations.csv')


def read_reliability():
    return {(r['model'], r['outcome']): num(r, 'spearman_brown')
            for r in load('generation/data/group_reliability.csv')}


def corr_cell(corr, model, layer, predictor, outcome):
    for r in corr:
        if (r['model'] == model and int(r['layer']) == layer and r['predictor'] == predictor
                and r['outcome'] == outcome and r['scope'] == 'per_model'):
            return r
    return None


# ============================
# 1. the question as asked
# ============================

def label_subset(pts):
    """Which points to annotate: the ones that carry information, not the pile.

    At layer 31 most groups sit at hit@1 = 1.0, so labelling every point turns the informative
    left-hand tail into unreadable overstrike. Annotate anything off the modal predictor value --
    those are the groups the predictor actually separates -- plus the two extremes in y, so the
    range of the outcome stays legible.
    """
    if not pts:
        return set()
    xs = [p[0] for p in pts]
    modal = max(set(xs), key=xs.count)
    keep = {p[2] for p in pts if p[0] != modal}
    by_y = sorted(pts, key=lambda p: p[1])
    keep.update(p[2] for p in by_y[:2] + by_y[-2:])
    return keep


def fig_gen_vs_retrieval(join, corr, reliability, out_dir):
    # (column, axis label, plotted-sign). `correlate` scored ranks negated so that "higher is
    # better" holds for every predictor; the panel plots the raw rank, so the displayed rho is
    # flipped back to match the axis the reader is looking at.
    preds = [('retr_hit1', 'analogy-retrieval hit@1 (higher = better)', +1),
             ('retr_mean_rank', 'analogy-retrieval mean rank (lower = better)', -1)]
    fig, axes = plt.subplots(len(preds), len(BEHAVIOURAL),
                             figsize=(2.05 * len(BEHAVIOURAL), 2.15 * len(preds)),
                             sharey=True)
    for pi, (pred, plabel, plot_sign) in enumerate(preds):
        for mi, model in enumerate(BEHAVIOURAL):
            ax = axes[pi][mi]
            recessive(ax)
            pts = [(num(r, pred), num(r, OUTCOME), g)
                   for (m, L, g), r in join.items()
                   if m == model and L == HEADLINE_LAYER and r[pred] != '']
            pts = [p for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])]
            if pts:
                xs, ys, gs = zip(*pts)
                ax.scatter(xs, ys, s=17, color=COLOR5[model], alpha=0.85,
                           edgecolor='white', linewidth=0.4, zorder=3)
                keep = label_subset(pts)
                for x, y, g in pts:
                    if g in keep:
                        ax.annotate(g, (x, y), fontsize=4.2, color='0.3',
                                    xytext=(2.4, 1.8), textcoords='offset points')
            c = corr_cell(corr, model, HEADLINE_LAYER, pred, OUTCOME)
            if c and c['rho'] != '':
                lo, hi = (plot_sign * float(c['ci_hi']), plot_sign * float(c['ci_lo'])) \
                    if plot_sign < 0 and c['ci_lo'] != '' else \
                    ((float(c['ci_lo']), float(c['ci_hi'])) if c['ci_lo'] != '' else (None, None))
                ci = f"[{lo:+.2f}, {hi:+.2f}]" if lo is not None else ''
                ax.set_title(f"{DISPLAY5[model]}\n" + r'$\rho$='
                             + f"{plot_sign * float(c['rho']):+.2f} {ci}", fontsize=7.2)
            else:
                ax.set_title(DISPLAY5[model], fontsize=7.2)
            if reliability.get((model, OUTCOME), 0) < 0.5:
                ax.set_facecolor('#F7F2EC')
            if mi == 0:
                ax.set_ylabel(OUTCOME_LABEL, fontsize=7.2)
            ax.set_xlabel(plabel, fontsize=7.2)
            ax.set_ylim(-0.05, 1.05)
    fig.suptitle(f'Per-functional-group generation accuracy vs. analogy retrieval, layer '
                 f'{HEADLINE_LAYER}\n(shaded panel = per-group outcome unreliable, '
                 f'split-half $<$ 0.5)', fontsize=8.5, y=1.015)
    fig.tight_layout()
    path = os.path.join(out_dir, 'gen_vs_retrieval.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ANALYSIS)}")


# ============================
# 2. where the depth dependence lives
# ============================

def fig_rho_by_layer(corr, reliability, out_dir):
    scopes = list(BEHAVIOURAL) + ['']
    fig, axes = plt.subplots(1, len(scopes), figsize=(2.02 * len(scopes), 2.5), sharey=True)
    for si, model in enumerate(scopes):
        ax = axes[si]
        recessive(ax)
        pooled = model == ''
        rows = [r for r in corr
                if r['outcome'] == OUTCOME
                and ((r['scope'] == 'pooled') if pooled else
                     (r['scope'] == 'per_model' and r['model'] == model))]
        for pred, label, color, dash in RHO_PREDICTORS:
            sel = sorted((int(r['layer']), r) for r in rows if r['predictor'] == pred
                         and r['rho'] != '')
            if not sel:
                continue
            xs = [L for L, _ in sel]
            ys = [float(r['rho']) for _, r in sel]
            ax.plot(xs, ys, color=color, marker='o', markersize=2.7, linewidth=1.25,
                    dashes=dash if dash[0] else (), label=label, zorder=3)
            if pred in CI_SHADED:
                lo = [float(r['ci_lo']) if r['ci_lo'] != '' else np.nan for _, r in sel]
                hi = [float(r['ci_hi']) if r['ci_hi'] != '' else np.nan for _, r in sel]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.13, linewidth=0, zorder=1)
        ax.axhline(0, linestyle='-', **REF)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([0, 8, 16, 24, 31])
        ax.set_xlabel('Layer', fontsize=7.2)
        title = 'pooled' if pooled else DISPLAY5[model]
        rel = np.nan if pooled else reliability.get((model, OUTCOME), np.nan)
        if np.isfinite(rel) and rel < 0.5:
            # Hatch, not colour: this panel's flat line means "could not detect", not "no effect".
            ax.set_facecolor('#F7F2EC')
            for side in ('left', 'bottom'):
                ax.spines[side].set_color('0.55')
            title += f'\nreliability {rel:.2f} — underpowered'
        elif not pooled:
            title += f'\nreliability {rel:.2f}'
        ax.set_title(title, fontsize=7.2)
        if si == 0:
            ax.set_ylabel(r'Spearman $\rho$ with' + '\n' + OUTCOME_LABEL, fontsize=7.2)
    handles, labels = axes[0].get_legend_handles_labels()
    # Below the row rather than inside a panel: at these axis limits every in-panel corner
    # already has a line running through it.
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=len(RHO_PREDICTORS), fontsize=6.4)
    fig.suptitle('Correlation between per-group representation quality and per-group generation '
                 'accuracy, by depth\n(shaded band = cluster-bootstrap 95% CI over functional '
                 'groups; exploratory, BH-corrected in group_correlations.csv)',
                 fontsize=8.5, y=1.03)
    fig.tight_layout()
    path = os.path.join(out_dir, 'rho_by_layer.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ANALYSIS)}")


# ============================
# 3. what the spread actually is
# ============================

def fig_response_composition(join, out_dir):
    groups = sorted({g for (_, L, g) in join if L == HEADLINE_LAYER})
    fig, axes = plt.subplots(len(BEHAVIOURAL), 1,
                             figsize=(COMP_SIZE_W, COMP_PANEL_H * len(BEHAVIOURAL)),
                             sharex=True)
    for mi, model in enumerate(BEHAVIOURAL):
        ax = axes[mi]
        ax.grid(True, axis='y', color='0.9', linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        x = np.arange(len(groups))
        bottom = np.zeros(len(groups))
        for field, label, color in RESPONSE_STACK:
            vals = np.array([num(join[(model, HEADLINE_LAYER, g)], field) for g in groups])
            ax.bar(x, vals, 0.76, bottom=bottom, color=color, linewidth=0,
                   label=label if mi == 0 else None)
            bottom += vals
        tf = [num(join[(model, HEADLINE_LAYER, g)], 'tf_raw_recall') for g in groups]
        ax.scatter(x, tf, marker='_', s=COMP_TF_MARK, color='0.12', linewidth=2.0,
                   zorder=4,
                   label='Teacher-forced recall' if mi == 0 else None)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(DISPLAY5[model], fontsize=COMP_YLABEL)
        ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=COMP_TICK)
    axes[-1].set_xticks(np.arange(len(groups)))
    axes[-1].set_xticklabels(groups, rotation=55, ha='right', fontsize=COMP_TICK)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.02),
                   ncol=6, fontsize=COMP_LEGEND)
    fig.tight_layout()
    path = os.path.join(out_dir, 'group_response_composition.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, ANALYSIS)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    style()

    join, corr, reliability = read_join(), read_corr(), read_reliability()
    print("Writing figures:")
    fig_gen_vs_retrieval(join, corr, reliability, args.out_dir)
    fig_rho_by_layer(corr, reliability, args.out_dir)
    fig_response_composition(join, args.out_dir)


if __name__ == '__main__':
    main()
