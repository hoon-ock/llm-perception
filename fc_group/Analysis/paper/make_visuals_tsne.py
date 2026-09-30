#!/usr/bin/env python3
"""t-SNE of functional-group activations, as a depth grid.

    python fc_group/Analysis/paper/make_visuals_tsne.py [--layers ...] [--out-dir DIR]
    -> fc_group/Analysis/visuals/tsne/tsne_depth_coarse.png   (5 heteroatom families)
    -> fc_group/Analysis/visuals/tsne/tsne_depth_fine.png     (20 functional groups)
    -> fc_group/Analysis/visuals/tsne/tsne_depth_template.png (10 prompt templates)

Rows are models, columns are layers, so the figure shows structure appearing with depth and
the base-vs-fine-tune difference in one read -- the unsupervised counterpart to the probe
depth curve, which tells the same story numerically.

THREE THINGS THIS FIGURE DOES NOT SAY, all of them in that directory's README.md:

  * t-SNE coordinates are NOT comparable across panels. Each panel is an independent
    embedding with its own arbitrary orientation, scale and origin; only the within-panel
    grouping means anything. Hence no axis ticks, no shared limits, and no claim about a
    cluster "moving" between layers.
  * no silhouette score is printed. `Analysis/tsne/README.md` establishes that the score the
    original sweep annotated was computed on the 2-D coordinates rather than the activations
    and "does not rank plots by how well they cluster -- sometimes it ranks them backwards".
    Cluster quality is the probe's job (`probe/data/layer_curves.csv`), not this figure's.
  * the template panel is the control, not a result about chemistry. It answers whether a
    layout is driven by prompt wording, and through layer 16 the answer is yes -- which is
    what stops the coarse figure's early panels being read as weak chemical structure when
    they are in fact strong WORDING structure. Quantified in
    `tsne/data/tsne_neighbourhood.csv`.
  * cluster SEPARATION here is not evidence of linear decodability, and the two disagree on
    purpose: the base model's layer-31 panel is visually the tightest while its raw space is
    the most anisotropic (pairwise cosine 0.944, Table 3). t-SNE will happily separate points
    that no linear probe could.

Reads only `fc_group/Analysis/tsne/data/tsne_coords.csv`, never `Results/`, which is
gitignored. `Analysis/tsne/compute_tsne_coords.py` writes that snapshot for this purpose.
"""
import argparse
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import ANALYSIS, BEHAVIOURAL, REPO, load  # noqa: E402
from make_visuals_probe import DISPLAY5, SHORT5, style  # noqa: E402

DEFAULT_VISUALS = os.path.join(ANALYSIS, 'visuals', 'tsne')

# Okabe-Ito, the same colourblind-safe set COLOR5 draws the model colours from. Ordered so
# `hydrocarbon` -- the unsubstituted backbone every diff vector is measured against -- reads
# as the neutral grey it is, rather than competing with the five substituted families.
FAMILY_COLOR = {
    'hydrocarbon': '#999999', 'halide': '#0072B2', 'oxygen': '#D55E00',
    'nitrogen': '#009E73', 'sulfur': '#CC79A7',
}
FAMILY_ORDER = ['hydrocarbon', 'halide', 'oxygen', 'nitrogen', 'sulfur']

# 20 categories cannot be made reliably distinguishable by hue alone in print, so the fine
# figure pairs hue with marker: hue carries the FAMILY (same five colours as the coarse
# figure, so the two read together) and the marker distinguishes members within it. That is
# the appendix figure's whole reason to exist -- the coarse one is what the paper uses.
FINE_MARKERS = ['o', 's', '^', 'D', 'v', 'P']

# Prompt templates get a SEQUENTIAL ramp rather than a categorical palette, and deliberately
# not the Okabe-Ito set the other two figures use. Two reasons: a reader who has just looked
# at the family figure must not read the same hue as the same thing, and template identity is
# a bare index -- nothing distinguishes template 3 from template 7 except which one it is, so
# a ramp says "these are indices" where distinct hues would imply distinct kinds. Viridis is
# colourblind-safe and prints legibly in greyscale.
TEMPLATE_CMAP = 'viridis'

POINT_SIZE = 5.0
POINT_ALPHA = 0.75

# The solo panels are read on their own rather than as one cell of a 25-panel grid, so they
# are drawn larger. POINT_SIZE is deliberately NOT scaled with them: at 5pt on a 2.6in axes
# the clusters stay legible as clusters, and enlarging the dots to match the canvas is how a
# t-SNE panel starts looking denser than its data.
SOLO_SIZE = (2.6, 2.6)


def display(value):
    """Legend text for a data value: sentence case, first letter only.

    The raw value is the LOOKUP KEY -- `panel` filters rows on `r[key] == value` and the
    style dicts are keyed by it -- so it cannot be capitalised in place. Only what the
    legend prints changes. `str.capitalize()` is wrong here: it lower-cases the rest, which
    would turn any future label carrying an element symbol or an acronym into mush.
    """
    text = str(value)
    return text[:1].upper() + text[1:] if text else text


def read_coords():
    """`(model, layer) -> list of rows`, from the committed snapshot."""
    out = {}
    for r in load('tsne/data/tsne_coords.csv'):
        out.setdefault((r['model'], int(r['layer'])), []).append(r)
    return out


def fine_style(coords):
    """`group -> (colour, marker)`, hue by family and marker by member within it."""
    order, seen = [], set()
    for r in sorted(coords, key=lambda r: (FAMILY_ORDER.index(r['coarse_family']),
                                           r['functional_group'])):
        g = r['functional_group']
        if g not in seen:
            seen.add(g)
            order.append((g, r['coarse_family']))
    per_family = {}
    out = {}
    for g, fam in order:
        i = per_family.get(fam, 0)
        per_family[fam] = i + 1
        out[g] = (FAMILY_COLOR[fam], FINE_MARKERS[i % len(FINE_MARKERS)])
    return out, [g for g, _ in order]


def template_style(coords):
    """`template index -> colour`, sampled off a sequential ramp in index order."""
    labels = sorted({r['template_index'] for r in coords}, key=int)
    cmap = plt.get_cmap(TEMPLATE_CMAP)
    # Stop short of the ramp's ends: viridis's extremes are a very dark blue and a pale
    # yellow, and at 5pt both lose against a white ground.
    return ({t: cmap(0.08 + 0.84 * i / max(1, len(labels) - 1))
             for i, t in enumerate(labels)}, labels)


def as_pair(style):
    """Normalize a style entry to `(colour, marker)`.

    Cannot be written as `isinstance(style, tuple)`: a colormap returns RGBA, which IS a
    tuple, and unpacking four channels into two names is exactly how the template figure
    first broke. The test is for the two-element (colour, marker) shape specifically.
    """
    if isinstance(style, tuple) and len(style) == 2 and isinstance(style[1], str):
        return style
    return style, 'o'


def panel(ax, rows, key, styles):
    """One embedding. Ticks are removed on purpose -- see the module docstring."""
    for value in styles:
        sub = [r for r in rows if r[key] == value]
        if not sub:
            continue
        x = np.array([float(r['x']) for r in sub])
        y = np.array([float(r['y']) for r in sub])
        color, marker = as_pair(styles[value])
        # `color=`, never `c=`: a colormap hands back RGBA, and `c` would value-map it the
        # moment a group's point count happened to equal the channel count.
        ax.scatter(x, y, s=POINT_SIZE, color=color, marker=marker, alpha=POINT_ALPHA,
                   linewidths=0, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color('0.7')
    # Equal aspect: t-SNE has no meaningful axis scale, so letting the two axes stretch
    # independently would distort the only thing the panel does carry -- relative distance.
    ax.set_aspect('equal', adjustable='datalim')


def grid_figure(coords, layers, models, key, styles, labels, ncol_legend,
                legend_title=None):
    """Panels on a fixed pitch, with the legend given its own reserved strip.

    The strip is sized from the number of legend ROWS rather than fixed, because the fine
    figure needs four rows where the coarse one needs one. Getting this wrong is visible:
    a legend anchored below a full-height axes grid is drawn straight over the bottom row of
    panels, and `savefig.bbox='tight'` does not rescue it -- tight bbox expands the canvas
    around the overlap rather than undoing it.
    """
    n_r, n_c = len(models), len(layers)
    # A legend title is drawn ABOVE the entries, so it needs its own row in the reserved
    # strip. Left out, it lands on top of the bottom row of panels -- the same overlap the
    # strip exists to prevent, one row higher.
    n_leg = math.ceil(len(labels) / ncol_legend) + (1 if legend_title else 0)
    leg_h = 0.16 * n_leg + 0.12          # inches
    fig_h = 1.28 * n_r + 0.25 + leg_h
    fig, axes = plt.subplots(n_r, n_c, figsize=(1.28 * n_c, fig_h), squeeze=False)
    for i, m in enumerate(models):
        for j, L in enumerate(layers):
            ax = axes[i][j]
            rows = coords.get((m, L))
            if rows is None:
                ax.set_axis_off()
                continue
            panel(ax, rows, key, styles)
            if i == 0:
                ax.set_title(f'Layer {L}', pad=3)
            if j == 0:
                ax.set_ylabel(DISPLAY5[m], labelpad=4)

    handles = [plt.Line2D([], [], linestyle='none', marker=as_pair(styles[v])[1],
                          markersize=3.4, markeredgewidth=0,
                          color=as_pair(styles[v])[0], label=display(v)) for v in labels]
    fig.subplots_adjust(wspace=0.06, hspace=0.06,
                        bottom=leg_h / fig_h, top=1 - 0.25 / fig_h,
                        left=0.055, right=0.995)
    fig.legend(handles=handles, loc='lower center', ncol=ncol_legend,
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.35, columnspacing=1.0,
               borderaxespad=0.2, title=legend_title)
    return fig


def solo_figure(rows, key, styles):
    """One panel on its own: no legend, no title, no model label -- just the embedding.

    Everything identifying the panel is in its path (`tsne/<model>/fine_L<layer>.png`), so
    nothing is drawn twice. Calls the same `panel` the grid does, so a solo file and its cell
    in `tsne_depth_fine.png` are the same picture at a different size.
    """
    fig, ax = plt.subplots(figsize=SOLO_SIZE)
    panel(ax, rows, key, styles)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    p.add_argument('--layers', type=int, nargs='+', default=[0, 8, 16, 24, 31])
    args = p.parse_args()
    style()

    coords = read_coords()
    # BEHAVIOURAL, not MODELS: this figure covers every checkpoint with activations, in the
    # same row order as Tables 1 and 3. Rows are dropped, not blanked, when coordinates for a
    # model are absent, so the grid never carries an empty band.
    models = [m for m in BEHAVIOURAL if any((m, L) in coords for L in args.layers)]
    missing = [m for m in BEHAVIOURAL if m not in models]
    if not models:
        raise SystemExit('no coordinates -- run '
                         'fc_group/Analysis/tsne/compute_tsne_coords.py')
    if missing:
        print(f"note: no coordinates for {', '.join(SHORT5[m] for m in missing)}")

    os.makedirs(args.out_dir, exist_ok=True)
    flat = [r for rows in coords.values() for r in rows]

    for name, key, styles, labels, ncol, title in (
            ('coarse', 'coarse_family', FAMILY_COLOR, FAMILY_ORDER, 5, None),
            # ncol 7, not 5: 20 labels over 7 columns is 3 rows. `grid_figure` sizes the
            # reserved legend strip from the row count, so this also shortens the figure.
            ('fine', 'functional_group', *fine_style(flat), 7, None),
            ('template', 'template_index', *template_style(flat), 10,
             'Prompt template index')):
        fig = grid_figure(coords, args.layers, models, key, styles, labels, ncol, title)
        path = os.path.join(args.out_dir, f'tsne_depth_{name}.png')
        fig.savefig(path)
        plt.close(fig)
        print(f'wrote {os.path.relpath(path, REPO)}')

        # Only the fine colouring is split out per model. The grid stays the primary form --
        # the whole point of it is reading depth across a row and models down a column, which
        # 25 separate files cannot do -- and these exist for placing one panel on its own.
        if name != 'fine':
            continue
        for m in models:
            # The model's own identifier, not SHORT5's slug: this matches the directory
            # names under Results/ and the analogy3d_* filenames, so one model is one string
            # everywhere a path carries it. SHORT5 stays what the CLI and the clustermap
            # filenames use.
            sub = os.path.join(args.out_dir, m)
            os.makedirs(sub, exist_ok=True)
            for L in args.layers:
                rows = coords.get((m, L))
                if rows is None:
                    continue
                solo = solo_figure(rows, key, styles)
                spath = os.path.join(sub, f'{name}_L{L}.png')
                solo.savefig(spath)
                plt.close(solo)
            print(f'  wrote {len(args.layers)} solo panels to '
                  f'{os.path.relpath(sub, REPO)}/')


if __name__ == '__main__':
    main()
