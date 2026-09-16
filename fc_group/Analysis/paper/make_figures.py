#!/usr/bin/env python3
"""Generate the paper's figures as PDFs from the committed Analysis CSVs.

    python fc_group/Analysis/paper/make_figures.py [--out-dir paper]

Like `make_tables.py`, this reads only `fc_group/Analysis/*/data/*.csv` and never
`Results/`, which is gitignored -- running both from a checkout without `Results/` present
is the test that the paper is reproducible from committed artifacts.

Palette is Okabe-Ito, chosen by running the dataviz validator rather than by eye (see
`_common.COLOR`). Every series also carries a distinct marker and dash pattern, so identity
survives greyscale printing and colour-vision deficiency.
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

from _common import (BASE, CHEM, COLOR, DASH, DEFAULT_OUT, ERR_FLOOR,  # noqa: E402
                     FIGURE_LAYER, LAYERS,
                     MARKER, MODELS, N_LAYERS, REPO, SHORT, load, num)

WIDTH = 5.5           # NeurIPS \linewidth in inches
GRID = dict(color='0.9', linewidth=0.6)
REF = dict(color='0.45', linewidth=0.9)


def style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
        'axes.linewidth': 0.7, 'lines.linewidth': 1.4,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
    })


def recessive(ax):
    """Grid and spines recede; the data is the only thing that should be dark."""
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def series(ax, model, xs, ys, label=None, **kw):
    dash = DASH[model]
    ax.plot(xs, ys, color=COLOR[model], marker=MARKER[model], markersize=3.4,
            dashes=dash if dash[0] else (),
            label=label if label is not None else SHORT[model], **kw)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, 'figures', name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, REPO)}")


# --------------------------------------------------------------------------------------
def f1_probe_depth(out_dir):
    """Where functional-group identity becomes linearly decodable."""
    curves = defaultdict(dict)
    for r in load('probe/data/layer_curves.csv'):
        curves[r['model']][int(r['layer'])] = num(r, 'balanced_acc')
    surface = np.mean([num(r, 'mean_balanced_acc')
                       for r in load('probe/data/surface_baseline.csv')])

    # Nitrogen->oxygen confusions per layer, as a SHARE of each model's own errors. The
    # lower panel exists because probe accuracy alone does not say WHERE a model loses
    # accuracy, and the Aim-2 claim is about one specific boundary. The share, not the raw
    # count, is the right quantity: below the transition every model has 100+ N->O errors
    # simply because the probe is near chance, which says nothing about the N/O boundary.
    # Normalizing by each model's own error total removes that confound. The panel starts
    # at ERR_FLOOR, the first layer at which all three models clear 0.95 balanced accuracy,
    # so every point compares models that have essentially solved the coarse task.
    share = defaultdict(dict)
    for r in load('probe/data/error_composition_by_layer.csv'):
        if r['share_nitrogen_to_oxygen'] != '':
            share[r['model']][int(r['layer'])] = num(r, 'share_nitrogen_to_oxygen')

    fig, (ax, bot) = plt.subplots(
        2, 1, figsize=(WIDTH, 3.6), sharex=True,
        gridspec_kw=dict(height_ratios=[1.45, 1], hspace=0.12))

    for m in MODELS:
        layers = sorted(curves[m])
        series(ax, m, layers, [curves[m][L] for L in layers])
    ax.axhline(0.20, linestyle=':', **REF)
    # Both reference labels sit at layer 8, where every curve is above 0.6 and so clear of
    # them; at the right edge they collided with the legend.
    ax.text(8, 0.215, 'chance', ha='left', va='bottom', fontsize=7, color='0.45')
    ax.axhline(surface, linestyle='--', **REF)
    ax.text(8, surface + 0.015, 'character $n$-gram baseline', ha='left',
            va='bottom', fontsize=7, color='0.45')
    # The transition the paper reports is a single step, so mark the step, not a region.
    ax.axvline(15.5, color='0.75', linewidth=0.9, zorder=0)
    # Left of the rule, clear of the legend that sits bottom-right.
    ax.text(15.1, 0.185, 'L15$\\rightarrow$16', fontsize=7, color='0.5', ha='right',
            va='bottom')
    ax.set_ylabel('Balanced accuracy')
    ax.set_ylim(0.15, 1.04)
    # Outside the axes entirely: the plot area has reference lines across its full width
    # and three curves converging top-right, with no corner left to put a legend in.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3)
    recessive(ax)

    for m in MODELS:
        layers = [L for L in sorted(share[m]) if L >= ERR_FLOOR]
        series(bot, m, layers, [share[m][L] for L in layers])
    # Layer 22 is where chem's first non-zero share appears; base and chem are both clean
    # below it, which is the half of the claim a reader is most likely to miss.
    bot.axvline(21.5, color='0.75', linewidth=0.9, zorder=0)
    bot.text(21.9, 0.96, 'L22', fontsize=7, color='0.5', ha='left', va='top')
    # The axis stays shared with the panel above so depths line up, but the series starts
    # at ERR_FLOOR: below it the share is not a meaningful quantity, and drawing it would
    # invite exactly the misreading the normalization is there to prevent.
    bot.text(ERR_FLOOR - 0.6, 0.5, 'share not meaningful\nbelow L%d' % ERR_FLOOR,
             fontsize=6.5, color='0.55', ha='right', va='center')
    bot.set_xlabel('Layer')
    bot.set_ylabel('N$\\rightarrow$O share\nof errors')
    bot.set_xticks(LAYERS)
    bot.set_ylim(-0.03, 1.0)
    recessive(bot)
    save(fig, out_dir, 'f1_probe_depth.pdf')


def f2_block_depth(out_dir):
    """Aim 1: chemically coherent blocks are built by the stack, not present at input.

    Top row is the base model's between-class geometry at three depths, every panel in the
    SAME leaf order -- the order the final layer's dendrogram gives -- so the reader is
    watching one layout fill in rather than three different clusterings. Bottom is the
    number behind that picture, for all five sampled layers.
    """
    rows = [r for r in load('taxonomy/data/between_class_matrix.csv') if r['model'] == BASE]
    groups = sorted({r['group_a'] for r in rows} | {r['group_b'] for r in rows})
    idx = {g: i for i, g in enumerate(groups)}

    def zmat(layer):
        sim = np.zeros((len(groups), len(groups)))
        for r in rows:
            if int(r['layer']) != layer:
                continue
            i, j = idx[r['group_a']], idx[r['group_b']]
            sim[i, j] = sim[j, i] = num(r, 'cosine_sim')
        np.fill_diagonal(sim, np.nan)
        off = sim[~np.isnan(sim)]
        z = (sim - off.mean()) / off.std(ddof=1)
        np.fill_diagonal(z, 0.0)
        return z

    # One leaf order for every panel, taken from the final layer.
    zf = zmat(FIGURE_LAYER)
    dist = zf.max() - zf
    np.fill_diagonal(dist, 0.0)
    order = dendrogram(linkage(squareform(dist, checks=False), method='average'),
                       no_plot=True)['leaves']

    panels = [0, 16, FIGURE_LAYER]
    fig = plt.figure(figsize=(WIDTH, 3.6))
    gs = fig.add_gridspec(2, len(panels), height_ratios=[1.0, 1], hspace=0.18, wspace=0.08)

    lim = max(np.nanmax(np.abs(zmat(L))) for L in panels)
    top_axes = []
    for col, layer in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        top_axes.append(ax)
        zz = zmat(layer)[np.ix_(order, order)]
        np.fill_diagonal(zz, np.nan)
        im = ax.imshow(zz, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        ax.set_title(f'layer {layer}', pad=3)
        ax.set_xticks([])
        if col == 0:
            ax.set_yticks(range(len(groups)))
            ax.set_yticklabels([groups[i] for i in order], fontsize=4.6)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    # Attach to the clustermap row only; spanning fig.axes would stretch it over the
    # trajectory panel below, which is a different quantity in different units.
    cb = fig.colorbar(im, ax=top_axes, fraction=0.022, pad=0.015)
    cb.set_label('$z$-cosine (within model)', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)

    # Bottom: the same story as a number. Blocks are scored independently and are not a
    # partition -- amide is in both the nitrogen and the carbonyl block.
    coh = defaultdict(dict)
    for r in load('geometry/data/block_cohesion_by_layer.csv'):
        if r['model'] == BASE:
            coh[r['block']][int(r['layer'])] = num(r, 'cohesion')
    ax = fig.add_subplot(gs[1, :])
    styles = {'halide': ('#0072B2', 'o', (None, None)),
              'nitrogen': ('#D55E00', 's', (4, 1.5)),
              'sulfur': ('#009E73', '^', (1, 1.5)),
              'carbonyl': ('#CC79A7', 'D', (5, 1.5, 1, 1.5))}
    for block, (color, marker, dash) in styles.items():
        layers = sorted(coh[block])
        ax.plot(layers, [coh[block][L] for L in layers], color=color, marker=marker,
                markersize=3.4, dashes=dash if dash[0] else (), label=block)
    ax.axhline(0, **REF)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Block cohesion')
    ax.set_xlim(0, N_LAYERS - 1)
    ax.set_xticks(LAYERS)
    ax.legend(loc='upper left', ncol=4)
    recessive(ax)
    save(fig, out_dir, 'f2_block_depth.pdf')


def f3_dendrograms(out_dir):
    """Illustration for F2: the same 19 groups, clustered, base against chem."""
    rows = [r for r in load('taxonomy/data/between_class_matrix.csv')
            if int(r['layer']) == FIGURE_LAYER]
    groups = sorted({r['group_a'] for r in rows} | {r['group_b'] for r in rows})
    idx = {g: i for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0))
    for ax, model, title in zip(axes, (BASE, CHEM), ('base', 'chem')):
        sim = np.zeros((len(groups), len(groups)))
        for r in rows:
            if r['model'] != model:
                continue
            i, j = idx[r['group_a']], idx[r['group_b']]
            sim[i, j] = sim[j, i] = num(r, 'cosine_sim')
        np.fill_diagonal(sim, np.nan)
        # z-score inside the model, as everywhere else in the paper: raw cosines differ in
        # level between models for anisotropy reasons that carry no chemistry.
        off = sim[~np.isnan(sim)]
        z = (sim - off.mean()) / off.std(ddof=1)
        np.fill_diagonal(z, 0.0)

        dist = z.max() - z
        np.fill_diagonal(dist, 0.0)
        order = dendrogram(linkage(squareform(dist, checks=False), method='average'),
                           no_plot=True)['leaves']
        zz = z[np.ix_(order, order)]
        np.fill_diagonal(zz, np.nan)

        # Diverging, because z has a meaningful zero (the model's own mean similarity),
        # with a neutral midpoint -- never a rainbow.
        lim = np.nanmax(np.abs(zz))
        im = ax.imshow(zz, cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
        ax.set_xticks(range(len(groups)))
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([groups[i] for i in order], fontsize=5.2)
        ax.set_xticklabels([groups[i] for i in order], fontsize=5.2, rotation=90)
        # amide is the group whose block membership differs between the two panels, so
        # name it rather than leaving the reader to find it among 19 labels.
        for labels in (ax.get_yticklabels(), ax.get_xticklabels()):
            for lab in labels:
                if lab.get_text() == 'amide':
                    lab.set_color(COLOR[CHEM])
                    lab.set_fontweight('bold')
        ax.set_title(title, pad=4)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02)
    cb.set_label('$z$-cosine (within model)', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    save(fig, out_dir, 'f3_dendrograms.pdf')


def f4_retrieval(out_dir):
    """Three independent retrieval axes, each against its own chance line."""
    rows = [r for r in load('geometry/data/retrieval_by_axis.csv')
            if int(r['layer']) == FIGURE_LAYER]
    axes_order = ['inter_group', 'halide', 'carbon_ladder']
    pretty = {'inter_group': 'inter-group\n(19 groups)', 'halide': 'halide column\n(4)',
              'carbon_ladder': 'chain-length ladder\n(19)'}
    by = {(r['model'], r['axis']): r for r in rows}

    fig, ax = plt.subplots(figsize=(WIDTH, 2.2))
    x = np.arange(len(axes_order))
    w = 0.26
    for k, m in enumerate(MODELS):
        vals = [num(by[(m, a)], 'hit1') for a in axes_order]
        # A surface gap between adjacent bars, per the mark spec.
        ax.bar(x + (k - 1) * w, vals, w * 0.88, color=COLOR[m], label=SHORT[m],
               linewidth=0)
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.015, f'{v:.2f}', ha='center', va='bottom', fontsize=6)
    for i, a in enumerate(axes_order):
        rnd = num(by[(BASE, a)], 'random_hit1')
        ax.plot([i - 1.6 * w, i + 1.6 * w], [rnd, rnd], color='0.35', linewidth=1.0,
                linestyle='--', zorder=3)
    ax.plot([], [], color='0.35', linewidth=1.0, linestyle='--', label='chance')
    ax.set_xticks(x)
    ax.set_xticklabels([pretty[a] for a in axes_order])
    ax.set_ylabel('hit@1')
    ax.set_ylim(0, 0.95)
    ax.legend(loc='upper right', ncol=4)
    recessive(ax)
    ax.grid(axis='x', visible=False)
    save(fig, out_dir, 'f4_retrieval.pdf')


def f5_cost(out_dir):
    """The two side-effects: anisotropy falls, degenerate offsets rise."""
    geo = defaultdict(dict)
    for r in load('geometry/data/geometry_by_layer.csv'):
        geo[r['model']][int(r['layer'])] = r

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.0))
    for m in MODELS:
        layers = sorted(geo[m])
        series(a1, m, layers, [num(geo[m][L], 'raw_pairwise_cosine') for L in layers])
        series(a2, m, layers, [num(geo[m][L], 'degenerate_rate') for L in layers])
    a1.set_ylabel('mean pairwise cosine')
    a1.set_title('anisotropy', pad=4)
    a2.set_ylabel('degenerate top-1 rate')
    a2.set_title('offset failure', pad=4)
    for ax in (a1, a2):
        ax.set_xlabel('Layer')
        ax.set_xlim(0, N_LAYERS - 1)
        ax.set_xticks(LAYERS)
        recessive(ax)
    a1.legend(loc='lower left', ncol=1)
    fig.tight_layout(w_pad=1.6)
    save(fig, out_dir, 'f5_cost.pdf')


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()
    style()
    for fn in (f1_probe_depth, f2_block_depth, f3_dendrograms, f4_retrieval, f5_cost):
        fn(args.out_dir)


if __name__ == '__main__':
    main()
