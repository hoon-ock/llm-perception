#!/usr/bin/env python3
"""Exploratory PNGs of the F2 depth story, one file per panel.

    python fc_group/Analysis/paper/make_visuals.py [--only base] [--out-dir DIR]

`make_figures.py` builds `paper/figures/f2_block_depth.pdf`: ONE model (base) at THREE
depths, composed into a single float. This emits the same content for all FIVE sampled
layers and all FIVE registered 8B checkpoints as **separate** PNGs -- each clustermap and
each cohesion trajectory its own file, untitled, so they can be placed individually. Output
lands in `fc_group/Analysis/visuals/clustermap/`, a working directory beside the analysis
outputs they are drawn from and deliberately not `paper/figures/`, so nothing here can reach
the manuscript by accident.

  heat_<model>_L<layer>.png      25 files: 5 models x 5 layers
  cohesion_<model>.png            5 files: block cohesion vs depth
  combined_<model>.png            5 files: the same content composed -- five clustermaps
                                  across the top, the cohesion trajectory underneath
  heatrow_<model>.png             5 files: that clustermap row alone, no trajectory, set in
                                  larger type for reading on its own

The composed files are ADDITIONS, not replacements: all three forms are drawn from the same
arrays, in the same leaf order, on the same two scales as the individual files, so no two of
them can ever disagree. `combined_<model>.png` and `heatrow_<model>.png` share ONE function
for the clustermap row (`heat_row`) and differ only in page geometry and type size.

Only the presentation differs. Every number is read from the same committed Analysis CSVs
`make_figures.py` reads, and the two transforms -- the within-(model, layer) z-score and the
block-cohesion difference -- are lifted verbatim from `f2_block_depth` and
`analyze_block_cohesion.py`. Nothing is recomputed, rescaled or smoothed here.

Two choices keep 30 separate files comparable to each other rather than 30 unrelated
pictures:

  * ONE leaf order for all 25 clustermaps, taken from the BASE model's final-layer
    dendrogram -- the same order the paper figure uses. Every model is drawn in that layout,
    so a difference between two files is a difference in the numbers and never a
    re-clustering.
  * ONE symmetric colour limit and ONE cohesion y-range, both taken over all five models and
    all five layers in a first pass. Colourbars and axes are interchangeable between files.
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from _common import (ANALYSIS, BASE, CHEM, FIGURE_LAYER, LAYERS,  # noqa: E402
                     N_LAYERS, R1, REPO, load, num)

# The two chemistry checkpoints added after the paper CSVs were first snapshotted. Slugs are
# the Results/ directory names (`/` -> `-`), which is what the CSVs carry in `model`.
CHEM_R = 'weidawang-Chem-R-8B'
CHEMDFM = 'OpenDFM-ChemDFM-v1.5-8B'

# Presentation order: base, the two controlled Llama-3.1 chemistry tunes in training order
# (Chem-R-Faithful is GRPO-trained FROM Chem-R-8B), the off-base chemistry model, then the
# reasoning distill. See fc_group/model_registry.py for the tier argument -- ChemDFM sits on
# Llama-3, not 3.1, so a ChemDFM-vs-base gap is not a controlled contrast.
MODELS5 = [BASE, CHEM_R, CHEM, CHEMDFM, R1]
SHORT5 = {BASE: 'base', CHEM_R: 'chem-r', CHEM: 'chem-faithful',
          CHEMDFM: 'chemdfm', R1: 'reason'}

DEFAULT_VISUALS = os.path.join(ANALYSIS, 'visuals', 'clustermap')
REF = dict(color='0.45', linewidth=0.9)

# Each clustermap is now its own figure rather than one of three panels in a 5.5in row, so
# it carries both axes' group labels: a standalone matrix with only row labels cannot be
# read. Sized so the 19 labels stay legible without the file being larger than a column.
HEAT_SIZE = (4.0, 3.5)
HEAT_TICK = 8.0
HEAT_CB_LABEL = 10.5
HEAT_CB_TICK = 9.5

# The cohesion plot is read on its own, often scaled down in a slide or a grid of five, so
# its type is set larger than the paper figure's 8pt -- point 4 of the edit request.
COH_SIZE = (6.4, 3.8)
COH_LABEL = 13
COH_TICK = 12
COH_LEGEND = 12

# The composite packs five clustermaps into one row, so each lands at roughly the physical
# panel size of the paper figure's three across 5.5in. Its type therefore stays at paper
# scale rather than inheriting the enlarged standalone sizes above, which are set for a
# panel viewed on its own.
COMBO_SIZE = (9.2, 4.0)
COMBO_TICK = 6.0
COMBO_CB_LABEL = 7.5
COMBO_CB_TICK = 6.5

# The clustermap row read on its own, with no trajectory panel beneath it to set the scale.
# Type is enlarged over COMBO_* for the same reason COH_* is enlarged over the paper figure:
# a panel viewed alone, or scaled into a slide, is not a panel nested in a taller composite.
#
# The size is measured, not chosen. Two constraints bind it:
#   * the panels must be WIDTH-bound. `imshow` holds them square, so a figure taller than
#     (width per column) opens a band of dead space under the titles that `bbox='tight'`
#     cannot crop, being interior to the figure rather than around it. At 15.0 x 3.2 the
#     five panels come out 155pt square with zero slack in their axes boxes.
#   * 19 group labels must fit down the left edge. 155pt / 19 = 8.2pt of pitch per label,
#     which carries ROW_TICK 8.0 without collision. The composite runs 6.0pt labels in a
#     5.0pt pitch -- readable only because Times has a small x-height -- so this is the
#     first form of the row whose labels are not slightly overlapping.
# Changing ROW_SIZE without re-checking both of those will quietly break one of them.
ROW_SIZE = (15.0, 3.2)
ROW_TICK = 8.0        # the 19 group labels, column 0
ROW_TITLE = 10.0      # 'Layer N'
ROW_CB_LABEL = 9.5
ROW_CB_TICK = 8.5

# Lifted from f2_block_depth. The blocks are composition-based and deliberately NOT a
# partition: amide is in both nitrogen and carbonyl.
BLOCK_STYLES = {'halide': ('#0072B2', 'o', (None, None)),
                'nitrogen': ('#D55E00', 's', (4, 1.5)),
                'sulfur': ('#009E73', '^', (1, 1.5)),
                'carbonyl': ('#CC79A7', 'D', (5, 1.5, 1, 1.5))}


def style():
    """Identical to make_figures.style(), plus a PNG-appropriate save dpi."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.7, 'lines.linewidth': 1.4,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04, 'savefig.dpi': 300,
    })


def recessive(ax):
    """Grid and spines recede; the data is the only thing that should be dark."""
    ax.grid(True, color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------------------
def read_matrices():
    """`(model, layer) -> z-scored 19x19 matrix`, plus the group list.

    The z-score is `f2_block_depth.zmat` verbatim: standardise the off-diagonal cosines by
    their own mean and std within one model at one depth, then park the diagonal at 0. Raw
    cosines differ in level between models for anisotropy reasons that carry no chemistry,
    which is why no cross-model quantity is ever built from the raw values.
    """
    rows = load('taxonomy/data/between_class_matrix.csv')
    groups = sorted({r['group_a'] for r in rows} | {r['group_b'] for r in rows})
    idx = {g: i for i, g in enumerate(groups)}

    cells = defaultdict(list)
    for r in rows:
        cells[(r['model'], int(r['layer']))].append(r)

    z_by = {}
    for (model, layer), pairs in cells.items():
        sim = np.zeros((len(groups), len(groups)))
        for r in pairs:
            i, j = idx[r['group_a']], idx[r['group_b']]
            sim[i, j] = sim[j, i] = num(r, 'cosine_sim')
        np.fill_diagonal(sim, np.nan)
        off = sim[~np.isnan(sim)]
        z = (sim - off.mean()) / off.std(ddof=1)
        np.fill_diagonal(z, 0.0)
        z_by[(model, layer)] = z
    return groups, z_by


def leaf_order(z):
    """Average-linkage leaf order, as f2_block_depth derives it from the final layer."""
    dist = z.max() - z
    np.fill_diagonal(dist, 0.0)
    return dendrogram(linkage(squareform(dist, checks=False), method='average'),
                      no_plot=True)['leaves']


def read_cohesion():
    """`model -> block -> layer -> cohesion`, straight from the analysis CSV."""
    coh = defaultdict(lambda: defaultdict(dict))
    for r in load('geometry/data/block_cohesion_by_layer.csv'):
        coh[r['model']][r['block']][int(r['layer'])] = num(r, 'cohesion')
    return coh


def heat_figure(model, layer, groups, z_by, order, lim):
    """One clustermap: one model at one depth, standalone and untitled."""
    fig, ax = plt.subplots(figsize=HEAT_SIZE)
    zz = z_by[(model, layer)][np.ix_(order, order)]
    np.fill_diagonal(zz, np.nan)
    # `nearest` so no cell is interpolated into its neighbour, and a symmetric diverging map
    # because z has a meaningful zero -- the model's own mean similarity.
    im = ax.imshow(zz, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    ax.set_xticks(range(len(groups)))
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([groups[i] for i in order], fontsize=HEAT_TICK)
    ax.set_xticklabels([groups[i] for i in order], fontsize=HEAT_TICK, rotation=90)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    # Labelled just "z-score". The standardisation is still within (model, layer) -- see
    # read_matrices -- but each file shows one model at one depth, so the qualifier had no
    # contrast to draw inside the frame it sat in.
    cb.set_label('$z$-score', fontsize=HEAT_CB_LABEL)
    cb.ax.tick_params(labelsize=HEAT_CB_TICK)
    cb.outline.set_visible(False)
    return fig


def cohesion_figure(model, coh, ylim):
    """One trajectory: the number behind the clustermaps, standalone and untitled."""
    fig, ax = plt.subplots(figsize=COH_SIZE)
    for block, (color, marker, dash) in BLOCK_STYLES.items():
        layers = sorted(coh[model][block])
        ax.plot(layers, [coh[model][block][L] for L in layers], color=color, marker=marker,
                markersize=5.5, linewidth=1.9, dashes=dash if dash[0] else (), label=block)
    ax.axhline(0, **REF)
    ax.set_xlabel('Layer', fontsize=COH_LABEL)
    ax.set_ylabel('Block cohesion', fontsize=COH_LABEL)
    ax.set_xlim(0, N_LAYERS - 1)
    ax.set_xticks(LAYERS)
    ax.set_ylim(*ylim)
    ax.tick_params(labelsize=COH_TICK)
    # handlelength must span a FULL dash cycle: carbonyl's (5, 1.5, 1, 1.5) pattern is 9pt
    # long, and a shorter key samples it mid-cycle and renders as a different pattern than
    # the line it labels. Crowding is relieved with columnspacing instead.
    ax.legend(loc='upper left', ncol=4, fontsize=COH_LEGEND,
              handlelength=2.8, columnspacing=0.8)
    recessive(ax)
    return fig


def heat_row(fig, axes, model, groups, z_by, order, lim, tick_fs, cb_label_fs, cb_tick_fs,
             cb_fraction, cb_pad, title_fs=None):
    """Five clustermaps in one row on pre-made axes, sharing one colour limit.

    Used by BOTH `combined_figure` and `heatrow_figure`, so the two can never disagree about
    the arrays, the leaf order or the scale -- they differ only in page geometry and type
    size, which is everything the caller passes in. The axes arrive already allocated because
    the callers lay out differently: the composite takes its row from a 2-row gridspec, the
    standalone owns the whole figure.

    The colourbar is attached to THESE axes, never to `fig.axes`: in the composite that would
    stretch the bar over the trajectory panel below, which is a different quantity in
    different units.
    """
    im = None
    for col, layer in enumerate(LAYERS):
        ax = axes[col]
        zz = z_by[(model, layer)][np.ix_(order, order)]
        np.fill_diagonal(zz, np.nan)
        im = ax.imshow(zz, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        # The one place a title survives: without it the five columns are unidentifiable,
        # which is not true of the individual files, where the filename carries the depth.
        ax.set_title(f'Layer {layer}', pad=3,
                     **({} if title_fs is None else dict(fontsize=title_fs)))
        ax.set_xticks([])
        if col == 0:
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels([groups[i] for i in order], fontsize=tick_fs)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=cb_fraction, pad=cb_pad)
    cb.set_label('$z$-score', fontsize=cb_label_fs)
    cb.ax.tick_params(labelsize=cb_tick_fs)
    cb.outline.set_visible(False)


def heatrow_figure(model, groups, z_by, order, lim):
    """The composite's clustermap row, alone: five depths, one colour limit, no trajectory.

    Same `heat_row` the composite calls, so this is a re-layout of the top half of
    `combined_<model>.png` and never a second opinion about the numbers.
    """
    fig, axes = plt.subplots(1, len(LAYERS), figsize=ROW_SIZE,
                             gridspec_kw=dict(wspace=0.06))
    heat_row(fig, list(axes), model, groups, z_by, order, lim,
             ROW_TICK, ROW_CB_LABEL, ROW_CB_TICK, 0.018, 0.012, title_fs=ROW_TITLE)
    return fig


def combined_figure(model, groups, z_by, order, lim, coh, ylim):
    """All five depths in one row with the trajectory underneath -- the paper figure's
    composition, widened from three panels to five.

    Draws the identical arrays `heat_figure` and `cohesion_figure` draw, in the identical
    leaf order and on the identical colour limit and y-range, so this file is a re-layout of
    the individual ones and never a second opinion about the numbers.
    """
    fig = plt.figure(figsize=COMBO_SIZE)
    gs = fig.add_gridspec(2, len(LAYERS), height_ratios=[1.0, 1.0],
                          hspace=0.16, wspace=0.06)

    top_axes = [fig.add_subplot(gs[0, col]) for col in range(len(LAYERS))]
    # No `title_fs`: the composite keeps inheriting `axes.titlesize` from style().
    heat_row(fig, top_axes, model, groups, z_by, order, lim,
             COMBO_TICK, COMBO_CB_LABEL, COMBO_CB_TICK, 0.018, 0.012)

    ax = fig.add_subplot(gs[1, :])
    for block, (color, marker, dash) in BLOCK_STYLES.items():
        layers = sorted(coh[model][block])
        ax.plot(layers, [coh[model][block][L] for L in layers], color=color, marker=marker,
                markersize=3.4, dashes=dash if dash[0] else (), label=block)
    ax.axhline(0, **REF)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Block cohesion')
    ax.set_xlim(0, N_LAYERS - 1)
    ax.set_xticks(LAYERS)
    ax.set_ylim(*ylim)
    ax.legend(loc='upper left', ncol=4)
    recessive(ax)
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    p.add_argument('--only', choices=sorted(SHORT5.values()),
                   help='render a single model instead of all five')
    args = p.parse_args()
    style()

    groups, z_by = read_matrices()
    coh = read_cohesion()

    missing = [m for m in MODELS5 if (m, FIGURE_LAYER) not in z_by]
    if missing:
        raise SystemExit(
            'no rows for ' + ', '.join(missing) + ' in taxonomy/data/between_class_matrix.csv'
            ' -- re-run fc_group/Analysis/taxonomy/analyze_taxonomy.py, then'
            ' fc_group/Analysis/geometry/analyze_block_cohesion.py')

    # First pass, over every model and every layer, so the separate files share one layout
    # and one pair of scales. Computing these per file would make two PNGs look different
    # for reasons that are not in the data.
    order = leaf_order(z_by[(BASE, FIGURE_LAYER)])
    lim = max(np.nanmax(np.abs(z_by[(m, L)])) for m in MODELS5 for L in LAYERS)
    vals = [coh[m][b][L] for m in MODELS5 for b in BLOCK_STYLES for L in LAYERS]
    pad = 0.08 * (max(vals) - min(vals))
    ylim = (min(vals) - pad, max(vals) + pad)

    wanted = [m for m in MODELS5 if args.only is None or SHORT5[m] == args.only]
    os.makedirs(args.out_dir, exist_ok=True)
    for model in wanted:
        short = SHORT5[model]
        for layer in LAYERS:
            fig = heat_figure(model, layer, groups, z_by, order, lim)
            path = os.path.join(args.out_dir, f'heat_{short}_L{layer}.png')
            fig.savefig(path)
            plt.close(fig)
            print(f'wrote {os.path.relpath(path, REPO)}')
        fig = cohesion_figure(model, coh, ylim)
        path = os.path.join(args.out_dir, f'cohesion_{short}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {os.path.relpath(path, REPO)}')
        fig = combined_figure(model, groups, z_by, order, lim, coh, ylim)
        path = os.path.join(args.out_dir, f'combined_{short}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {os.path.relpath(path, REPO)}')
        fig = heatrow_figure(model, groups, z_by, order, lim)
        path = os.path.join(args.out_dir, f'heatrow_{short}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {os.path.relpath(path, REPO)}')
    print(f'shared z-cosine limit +/-{lim:.3f}, cohesion y-range '
          f'[{ylim[0]:.3f}, {ylim[1]:.3f}], leaf order from base L{FIGURE_LAYER}')


if __name__ == '__main__':
    main()
