#!/usr/bin/env python3
"""Every clean analogy retrieval drawn as a 3D parallelogram, one PNG + one .txt each.

    python fc_group/Analysis/paper/make_visuals_retrieval_3d.py [--layer 31] [--out-dir DIR]

`make_visuals_retrieval.py` reports how OFTEN `b2 + (a1 - a2)` retrieves `b1`. These show what
one success looks like. One file per (quadruple, chain length) cell that is 4/4 corners at
hit@1 with zero degenerate corners -- six of twenty at base/L31:

  analogy3d_<model>_L31_thioether-thiol_ether-alcohol_C5.png        and C6
  analogy3d_<model>_L31_ester-carboxylic-acid_ketone-aldehyde_C4.png
  analogy3d_<model>_L31_imine-aldehyde_amine-alcohol_C4.png         and C5, C6

Each PNG has a `.txt` of the same stem beside it, and **everything that is not needed to read
the picture lives in the .txt**. On the image: four functional-group names, the legend, the
axis names. In the sidecar: the retrieval outcome, the projection diagnostics, the compounds
behind each group, and the caveats. No title -- the quadruple and the chain length are in the
filename and at the head of the .txt, and the four labelled groups already say which
quadruple this is. No tick labels either: the PC coordinates are in no unit a reader wants
off an axis, and every distance the panel claims is in the sidecar in those units. The grid
stays, because it is what shows the three axes share one scale.

The selection is made in `analogy/snapshot_quiver3d.py:rank_cells`, on the scored trials --
not here and not by hand.

**Every point is a compound, not an average.** The pool is one chain length at a time, so
`ether` at C5 is methyl butyl ether rather than a mean over C3-C6, and each labelled point
carries its compound name under the group name. Three groups are the exception: alcohol, thiol
and amine each hold an `n-`/`sec-` isomer pair in the dataset, so those vectors are a
2-molecule mean, and any panel showing one says so.

What each file draws:

  * a solid arrow a2 -> a1, the offset the analogy asserts;
  * the SAME arrow dashed, re-based at b2, ending in an X at the constructed point;
  * a solid arrow b2 -> b1 -- if the analogy holds this and the dashed arrow coincide;
  * a dotted residual from the X to the true b1, the error the retrieval tolerated;
  * the other 15 groups as unlabelled grey dots. They are the candidate pool, and without
    them the panel would show a parallelogram but not a retrieval: hit@1 is a claim about
    winning a race, not about the residual being small. The panel quotes the margin for the
    same reason.

Two drawing choices worth knowing about:

  * **No arrows from the origin.** The projection is mean-centered, so the origin is the
    cloud's centroid and a spoke to it means nothing. Drawn anyway they were the longest
    lines on the panel and buried the offsets, which are what the figure is about.
  * **The camera is chosen per panel**, by maximising the smallest on-screen separation
    among the five points that carry the claim. One fixed angle put `alcohol` on top of
    `aldehyde`. The camera is a viewing choice and changes no coordinate; each file prints
    the angle it settled on.

Three things keep a picture from flattering itself. All three are in the sidecar, and the
second is ALSO on the image:

  * `rank` and `cos` are the FULL 4096-dimensional numbers, copied from the scored trials.
    Nothing is rescored here;
  * `3D rank` is that rank recomputed from the three plotted coordinates. It is 1 on all six
    files. Where it disagrees, the panel itself says so in red -- that one sentence stays on
    the pixels because an image travels without its sidecar, and a quietly misleading panel
    is the failure this whole file exists to avoid;
  * `of the constructed point in view` is the fraction these three dimensions hold -- 0.42 to
    0.64. Rather less than half of some of these panels is actually on the page.

The `ester : carboxylic acid :: ketone : aldehyde` file is the quadruple set's deliberate
negative control, and its sidecar says so. It passes the selection rule cleanly at C4, which
is a result rather than a reason to hide it, but it must not read as a plain success.

Reads only `analogy/data/analogy_quiver3d.csv`, never `Results/`, in line with every other
generator here. Written to `fc_group/Analysis/visuals/retrieval/`, not `paper/figures/`, so
nothing here can reach the manuscript by accident.
"""
import argparse
import itertools
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

from _common import ANALYSIS, FIGURE_LAYER, REPO, load, num  # noqa: E402
from make_visuals_probe import style  # noqa: E402

DEFAULT_VISUALS = os.path.join(ANALYSIS, 'visuals', 'retrieval')

# Okabe-Ito, the two hues _common.py already uses for base and chem. One per SIDE of the
# analogy rather than one per group: the panel's subject is that two pairs share an offset,
# so the pair is the unit that needs a colour.
SRC = '#0072B2'    # a1, a2 -- and the transported copy of their offset
TGT = '#D55E00'    # b1, b2
POOL = '0.70'      # the 15 groups that lost the race
ALERT = '#B22222'  # reserved for the one failure mode: the projection contradicting the rank
NOTE = '#8B5A00'   # the negative-control mark -- visible, but not the alert colour
# Shape separates the two sides as well as hue does, per the house rule that identity never
# rests on colour alone.
MARK = {'a1': 'o', 'a2': 'o', 'b1': 's', 'b2': 's'}
HUE = {'a1': SRC, 'a2': SRC, 'b1': TGT, 'b2': TGT}
ROLES = ('a1', 'a2', 'b1', 'b2')

# One square panel per file. These run larger than the two-panel version they replace: a
# standalone PNG is read on its own rather than as a column of a figure, so the type is sized
# for that instead of for a shared page.
Q3D_SIZE = (8.0, 7.8)
Q3D_LABEL = 19.0    # PC-1 / PC-2 / PC-3
Q3D_POINT = 20.0    # the four group names -- the only words inside the axes
Q3D_LEGEND = 15.0
Q3D_ALERT = 14.0    # the only text on a panel that is not an identifier

# One vertical budget for the figure, in axes fractions: the legend owns the strip above
# AXES_TOP, and the axes take everything below it. There is no title band -- the quadruple and
# the chain length are in the filename and at the head of the .txt, and on the image the four
# labelled groups already say which quadruple this is. The numbers went to the sidecar too,
# which between them is what frees this much height for the cube.
# ALERT_Y is the one exception -- see `alert()`.
AXES_TOP = 0.925
AXES_BOTTOM = 0.055
ALERT_Y = 0.012
# mplot3d fits its cube inside the axes box and then leaves a wide margin around it, so an
# axes spanning exactly the figure draws a small cube in a lot of white. Letting the box run
# past both edges is what pulls the cube up to a readable size; the cube itself stays inside.
AXES_BLEED = 0.085

# Candidate cameras for the per-panel search. Elevations stay shallow: from high above, a
# cloud this flat in PC-3 collapses onto the floor pane. Azimuths within AZIM_GUARD of a
# right angle are skipped -- there two of the three panes go exactly edge-on, and mplot3d
# responds by stacking two axis labels and both their tick sets down the same edge.
ELEVS = range(8, 33, 4)
AZIM_GUARD = 14
AZIMS = [a for a in range(-180, 180, 5) if min(abs(a - q) for q in (-180, -90, 0, 90, 180))
         >= AZIM_GUARD]


def read_cells(layer):
    """One entry per drawn cell: coordinates by role, the pool, names, and scalar fields."""
    rows = [r for r in load('analogy/data/analogy_quiver3d.csv')
            if int(r['layer']) == layer]
    if not rows:
        raise SystemExit(
            f'no rows at layer {layer} in analogy/data/analogy_quiver3d.csv -- re-run '
            f'fc_group/Analysis/analogy/snapshot_quiver3d.py --layer {layer}')

    def vec(r):
        return np.array([num(r, 'svd1'), num(r, 'svd2'), num(r, 'svd3')])

    cells = []
    for rank in sorted({int(r['cell_rank']) for r in rows}):
        mine = [r for r in rows if int(r['cell_rank']) == rank]
        head = mine[0]
        by_role = {r['role']: r for r in mine if r['role'] != 'other'}
        cells.append({
            'xyz': {role: vec(r) for role, r in by_role.items()},
            'others': np.stack([vec(r) for r in mine if r['role'] == 'other']),
            'label': {role: head[f'pair_{role}'] for role in ROLES},
            'mol': {role: by_role[role]['molecules'] for role in ROLES},
            'isomers': [by_role[role]['group'] for role in ROLES
                        if int(by_role[role]['n_molecules']) > 1],
            'meta': head,
        })
    return cells


def slug(cell, layer):
    """`<model>_L31_thioether-thiol_ether-alcohol_C5` -- the stem both outputs share.

    The model comes first because these figures are per-checkpoint and a directory of them
    should sort by checkpoint; it is the full directory slug the `Results/` tree uses rather
    than `_common.SHORT`'s `base`, which means nothing outside this repo.

    All four group names rather than just a1 and b1: two of the five quadruples share `amine`
    as b1, and a scheme that is unique only for the quadruples that happen to exist today
    will not stay unique. The pair-wise form follows the slug
    `functional_group_analogy_carbon_matched.py:797` already uses for its quiver PNGs.
    """
    m = cell['meta']
    part = lambda *ks: '-'.join(m[k].replace(' ', '-') for k in ks)
    carbon = m['carbon_count']
    return (f"{m['model']}_L{layer}_"
            f"{part('pair_a1', 'pair_a2')}_{part('pair_b1', 'pair_b2')}"
            + (f'_C{carbon}' if carbon else '_lumped'))


def screen(pts, elev, azim):
    """Where mplot3d's camera puts `pts` on the page, up to scale.

    The two basis vectors of `view_init(elev, azim)`. Only used to compare separations, so
    the missing scale factor and the sign conventions do not matter.
    """
    e, a = np.radians(elev), np.radians(azim)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.array([-np.cos(a) * np.sin(e), -np.sin(a) * np.sin(e), np.cos(e)])
    return np.stack([pts @ right, pts @ up], axis=1)


def pick_camera(cell):
    """The angle that separates the five load-bearing points best.

    A single fixed camera is what the existing quiver uses, and at `elev=18, azim=-50` it
    projects `alcohol` and `aldehyde` -- 10.6 apart in the plotted space -- onto nearly the
    same pixel, so the panel read as one arrow and two overprinted labels. Maximising the
    SMALLEST pairwise screen distance is the right objective rather than the mean: one
    collision ruins the panel however well separated the other pairs are.
    """
    key = np.stack([cell['xyz'][r] for r in ROLES] + [cell['xyz']['predicted']])
    best, chosen = -1.0, (18, -50)
    for elev in ELEVS:
        for azim in AZIMS:
            flat = screen(key, elev, azim)
            gaps = [np.linalg.norm(flat[i] - flat[j])
                    for i in range(len(flat)) for j in range(i + 1, len(flat))]
            if min(gaps) > best:
                best, chosen = min(gaps), (elev, azim)
    return chosen


def equalize(ax, pts):
    """One scale on all three axes, centred on the cloud.

    mplot3d autoscales each axis independently. That would keep the two offset arrows
    parallel -- parallelism survives any affine scaling -- but not distance, and the panel's
    claim is that the X is nearer `b1` than any grey dot is.
    """
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.10
    for setter, m in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setter(m - half, m + half)
    ax.set_box_aspect((1, 1, 1))
    return half


def recessive3d(ax):
    """`_common.recessive()` is 2D-only -- it hides spines a 3D axes has no attribute for.
    Same intent: panes and grid recede, the arrows are the only dark thing on the panel."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor('0.90')
        axis.pane.set_linewidth(0.5)
        axis._axinfo['grid'].update(color='0.91', linewidth=0.5)
        axis.set_major_locator(plt.MaxNLocator(4))
    # Tick LABELS off, ticks themselves kept. The PC coordinates are in no unit anyone reads
    # off an axis, and every distance the panel claims -- the residual, the gap to the
    # runner-up -- is in the sidecar in those same units. But mplot3d draws the grid AT the
    # ticks, so clearing the ticks outright takes the grid with it, and the grid is what
    # shows the three axes share one scale and gives the cube its depth. So: empty the
    # labels, zero the tick marks, leave the locator alone.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.tick_params(length=0)
    for setter, name in ((ax.set_xlabel, 'PC-1'), (ax.set_ylabel, 'PC-2'),
                         (ax.set_zlabel, 'PC-3')):
        setter(name, fontsize=Q3D_LABEL, labelpad=-8, color='0.35')


def arrow(ax, start, end, head=0.14, **kw):
    """`ax.quiver` wants a displacement, which reads badly at every call site here -- both
    endpoints are always already in hand."""
    ax.quiver(*start, *(np.asarray(end) - np.asarray(start)),
              arrow_length_ratio=head, **kw)


def to_axes(ax, point):
    """A 3D data point as a fraction of the axes box, once the camera and limits are set.

    Labels have to be placed in SCREEN space, not data space. Offsetting a label along the
    3D direction away from the cloud centre is the obvious thing and it does not work: that
    direction can project to almost nothing, which is how `alcohol` kept landing on its own
    marker. Everything about a collision is two-dimensional, so the offset has to be too.
    """
    flat = proj3d.proj_transform(*point, ax.get_proj())[:2]
    return ax.transAxes.inverted().transform(ax.transData.transform(flat))


# Axes fractions. LABEL_PAD clears the marker; GROUP_H is one line of Q3D_POINT in an axes
# this tall, and is what `spread_labels` keeps between two names.
# LABEL_PAD had to grow with the type: at 20pt the white backing behind a name is large
# enough to cover a short arrow, and in the cramped quadruples the arrows ARE short. Pushing
# the names clear of the action and letting the leader line carry the attribution costs
# nothing, because the leader is drawn under the label and over nothing else.
LABEL_PAD = 0.080
GROUP_H = 0.048
# Two labels closer than LABEL_MIN_DX horizontally are treated as one column and pushed apart
# vertically. Where the analogy is tight, a2 and b2 are genuinely close together on screen --
# that is the result, not a defect -- so the labels have to be separated even though the
# points cannot be. ester : carboxylic acid :: ketone : aldehyde at C4 is the case that needs
# it: carboxylic acid and aldehyde land within a marker's width of each other.
LABEL_MIN_DX = 0.30
LABEL_GAP = 0.014


def spread_labels(anchors, heights):
    """Push overlapping label anchors apart vertically, symmetrically, until they clear.

    Cheap relaxation rather than a real label-placement solver: there are only four labels,
    and the only collision that occurs in practice is a single close pair.
    """
    for _ in range(80):
        moved = False
        for r1, r2 in itertools.combinations(anchors, 2):
            (x1, y1), (x2, y2) = anchors[r1], anchors[r2]
            need = (heights[r1] + heights[r2]) / 2 + LABEL_GAP
            if abs(x2 - x1) >= LABEL_MIN_DX or abs(y2 - y1) >= need:
                continue
            push = (need - abs(y2 - y1)) / 2 + 0.002
            # Tie broken by role order rather than at random, so the same input always
            # produces the same figure.
            sign = 1.0 if (y2 - y1) > 0 or (y2 == y1 and r2 > r1) else -1.0
            anchors[r1] = np.array([x1, y1 - sign * push])
            anchors[r2] = np.array([x2, y2 + sign * push])
            moved = True
        if not moved:
            break
    return anchors


def panel(ax, cell):
    xyz, lab = cell['xyz'], cell['label']
    a1, a2, b1, b2, pred = (xyz['a1'], xyz['a2'], xyz['b1'], xyz['b2'], xyz['predicted'])
    elev, azim = pick_camera(cell)

    equalize(ax, np.vstack([cell['others'], a1, a2, b1, b2, pred]))
    ax.view_init(elev=elev, azim=azim)
    recessive3d(ax)

    # The candidate pool first, so every arrow draws over it.
    ax.scatter(*cell['others'].T, s=16, color=POOL, alpha=0.8, depthshade=False,
               linewidth=0, zorder=1)

    # b2 -> b1 goes down first and heavier. Where the analogy is tight the dashed transported
    # offset lands on top of it -- that coincidence IS the result, so the one underneath is
    # drawn wide enough to still show either side of the dashes.
    arrow(ax, b2, b1, color=TGT, linewidth=3.4, alpha=0.9, zorder=3)
    arrow(ax, a2, a1, color=SRC, linewidth=2.6, zorder=4)
    arrow(ax, b2, pred, color=SRC, linewidth=2.1, linestyle=(0, (5, 2.5)), zorder=5)
    ax.plot(*zip(pred, b1), color='0.2', linewidth=1.4, linestyle=':', zorder=6)
    ax.scatter(*pred, s=130, color='black', marker='x', linewidth=2.3,
               depthshade=False, zorder=8)

    for role in ROLES:
        ax.scatter(*xyz[role], s=64, color=HUE[role], marker=MARK[role], depthshade=False,
                   edgecolor='white', linewidth=0.8, zorder=7)

    # Placed on screen, after the camera is fixed. Each label is pushed away from the
    # on-screen centre of the five points that carry the claim, so two groups that sit close
    # together throw their names in opposite directions. The X is in that centre as well as
    # the four members: where the analogy is tight it lands right beside b1, and a b1 label
    # pushed away from the members alone would land on top of it.
    flat = {r: to_axes(ax, xyz[r]) for r in ROLES + ('predicted',)}
    centre = np.mean(list(flat.values()), axis=0)
    # A white stroke around the glyphs, not a white box behind them. The box was the honest
    # first try and it is wrong here: at 20pt it covers enough area to wash out whichever
    # arrow it lands on, and in the cramped quadruples the arrows are short enough that
    # losing a few millimetres of one matters. A stroke keeps the name legible over an arrow
    # while hiding almost nothing of it.
    halo = [pe.withStroke(linewidth=3.5, foreground='white')]

    heights = {r: GROUP_H for r in ROLES}
    anchors, side = {}, {}
    for role in ROLES:
        away = flat[role] - centre
        norm = np.linalg.norm(away)
        d = away / norm if norm > 1e-9 else np.array([0.0, 1.0])
        anchors[role] = flat[role] + d * LABEL_PAD
        side[role] = d
    anchors = spread_labels(anchors, heights)

    for role in ROLES:
        anchor, d = anchors[role], side[role]
        ha = 'left' if d[0] > 0.35 else 'right' if d[0] < -0.35 else 'center'
        # Near an edge the label has to grow inward whatever the outward direction says,
        # or it runs off the panel and over the tick labels on the way.
        if ha == 'left' and anchor[0] > 0.66:
            ha = 'right'
        elif ha == 'right' and anchor[0] < 0.34:
            ha = 'left'
        # A leader from the marker to its name. Once a label can be moved to clear another,
        # proximity no longer identifies it, and on a 3D scatter with four labelled points a
        # mis-attributed name is worse than a crowded one.
        ax.plot(*zip(flat[role], anchor), transform=ax.transAxes, color=HUE[role],
                linewidth=0.9, alpha=0.45, zorder=2)
        # The functional group only. The compounds behind it are in the .txt -- on the image
        # they doubled the height of every label for a name the reader does not need to place
        # the point, and they were the reason the group names had to stay small.
        ax.text2D(anchor[0], anchor[1], lab[role], transform=ax.transAxes,
                  fontsize=Q3D_POINT, color=HUE[role], ha=ha,
                  va='bottom' if d[1] >= 0 else 'top', zorder=9, path_effects=halo)
    return elev, azim


def alert(fig, cell):
    """The one piece of text left on a panel that is not an identifier.

    Everything else the panel used to print now lives in the .txt sidecar, but this cannot:
    it says the picture contradicts the number, and an image travels without its sidecar. A
    panel that is quietly misleading is the failure this whole file is built to avoid, so the
    warning stays on the pixels.
    """
    m = cell['meta']
    rank, rank3d = int(m['rank_excl']), int(m['rank3d_excl'])
    if rank3d == rank:
        return
    fig.text(0.5, ALERT_Y,
             f'this panel projects to rank {rank3d} but the 4096-D rank is {rank} -- '
             'read the number, not the picture',
             ha='center', va='bottom', fontsize=Q3D_ALERT, color=ALERT)


def report(cell, layer, elev, azim, png_name):
    """The panel's numbers, as text. Written beside the PNG with the same stem.

    Split out of the figure so the image carries only what identifies it -- the quadruple,
    the chain length, the legend and the axes. Keeping the two in one file made the caption
    the tallest thing on the page and still left it too small to read comfortably.
    """
    m, lab, mol = cell['meta'], cell['label'], cell['mol']
    rank, rank3d = int(m['rank_excl']), int(m['rank3d_excl'])
    carbon = m['carbon_count']
    agree = ('agrees with the 4096-D rank' if rank3d == rank
             else f'DISAGREES with the 4096-D rank of {rank} -- read the number, not the picture')
    pool = (f'carbon_matched -- chain length C{carbon} only, no averaging over C3-C6'
            if carbon else 'lumped -- each group averaged over C3-C6, so no point is a compound')

    out = [
        f'{lab["a1"]} : {lab["a2"]}  ::  {lab["b1"]} : {lab["b2"]}'
        + (f'      C{carbon}' if carbon else '      lumped'),
        '=' * 78,
        '',
        f'{"figure":<26}{png_name}',
        f'{"model":<26}{m["model"]}',
        f'{"layer":<26}{layer}',
        f'{"pool":<26}{pool}',
        '',
        'RETRIEVAL  (full 4096-D; copied from retrieval_trials.csv, never recomputed here)',
        f'  {"rank of " + m["pair_b1"]:<34}{rank} of {m["n_candidates_excl"]}',
        f'  {"cosine to the constructed point":<34}{num(m, "cos_to_target"):.3f}',
        f'  {"corners at hit@1":<34}{m["n_hit1"]} of {m["n_corners"]}',
        f'  {"degenerate corners":<34}{m["n_degenerate"]}',
        f'  {"mean cosine over corners":<34}{num(m, "mean_cos"):.3f}',
        '',
        'PROJECTION  (mean-centered top-3 PCA over all 19 groups; plot units)',
        f'  {"variance held by PC-1..3":<34}{num(m, "explained_var"):.0%}',
        f'  {"of the constructed point in view":<34}{num(m, "recon_frac"):.0%}',
        f'  {"3D rank of " + m["pair_b1"]:<34}{rank3d}   ({agree})',
        f'  {"X lands":<34}{num(m, "dist3d_true"):.1f} from {m["pair_b1"]}',
        f'  {"nearest rival":<34}{num(m, "dist3d_rival"):.1f} from {m["rival_group"]}',
        f'  {"camera (elev / azim)":<34}{elev} / {azim}',
        '',
    ]

    if carbon:
        out.append(f'COMPOUNDS AT C{carbon}')
        for role in ROLES:
            tag = '   (2-isomer mean)' if cell['label'][role] in cell['isomers'] else ''
            out.append(f'  {role}  {lab[role]:<18}{mol[role]}{tag}')
        out.append('')

    out.append('NOTES')
    out.append('  The axes carry no tick labels: the PC coordinates are in no unit a reader')
    out.append('  wants off an axis. All three share one scale, so the distances above are')
    out.append('  comparable with each other, and the grid on the panel shows that scale.')
    if cell['isomers']:
        names = ' and '.join(cell['isomers'])
        out.append(f'  {names} are the mean of their n- and sec- isomers at this chain')
        out.append('  length -- the one place this panel is still an average.')
    if int(m['is_negative_control']):
        out.append("  This quadruple is the set's DELIBERATE NEGATIVE CONTROL")
        out.append('  (functional_group_analogy_carbon_matched.py:306-311): the two legs are')
        out.append('  formally matched (H -> CH3) but chemically are not, and it is the one')
        out.append('  that should fail if leg symmetry is what makes an analogy work. It does')
        out.append('  not fail. That is a result, not a plain success.')
    if rank3d != rank:
        out.append(f'  The projection puts {m["pair_b1"]} at rank {rank3d} while it is rank')
        out.append(f'  {rank} in the full space. The picture is misleading here; the panel')
        out.append('  says so on its face.')
    return '\n'.join(out) + '\n'


def figure(cell, layer):
    m, lab = cell['meta'], cell['label']
    fig = plt.figure(figsize=Q3D_SIZE)
    ax = fig.add_subplot(111, projection='3d')
    elev, azim = panel(ax, cell)


    alert(fig, cell)

    # Sentence case, except `b2 + (a1 - a2)`: that one is a FORMULA, and `b2` is the same
    # variable the two pair labels above it and every analogy3d_*.txt readout spell in lower
    # case. Capitalising it would make the legend disagree with the rest of the figure.
    handles = [
        Line2D([], [], color=SRC, marker='o', linestyle='-', linewidth=2.6,
               markersize=7, label='Source pair  a2 $\\rightarrow$ a1'),
        Line2D([], [], color=TGT, marker='s', linestyle='-', linewidth=3.4,
               markersize=7, label='Target pair  b2 $\\rightarrow$ b1'),
        Line2D([], [], color=SRC, linestyle=(0, (5, 2.5)), linewidth=2.1,
               label='b2 + (a1 $-$ a2)'),
        Line2D([], [], color='black', marker='x', linestyle='none', markersize=9,
               markeredgewidth=2.3, label='Where it lands'),
        Line2D([], [], color='0.2', linestyle=':', linewidth=1.4, label='Residual'),
        # Counted, not typed. All six drawn cells carry 15 `other` rows today, but the
        # inter-group axis ranks over 16-17 groups depending on the cell, so a re-run of
        # snapshot_quiver3d.py with a different trial set would strand a wrong number here.
        Line2D([], [], color=POOL, marker='o', linestyle='none', markersize=5.5,
               label=f'Other {len(cell["others"])} groups'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=Q3D_LEGEND, handlelength=2.2, columnspacing=1.8, frameon=False)
    fig.subplots_adjust(left=-AXES_BLEED, right=1.0 + AXES_BLEED,
                        bottom=AXES_BOTTOM, top=AXES_TOP)
    return fig, elev, azim


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layer', type=int, default=FIGURE_LAYER)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    args = p.parse_args()
    style()

    cells = read_cells(args.layer)
    os.makedirs(args.out_dir, exist_ok=True)
    for cell in cells:
        m, lab = cell['meta'], cell['label']
        fig, elev, azim = figure(cell, args.layer)
        stem = f'analogy3d_{slug(cell, args.layer)}'
        path = os.path.join(args.out_dir, stem + '.png')
        fig.savefig(path)
        plt.close(fig)
        txt = os.path.join(args.out_dir, stem + '.txt')
        with open(txt, 'w') as fh:
            fh.write(report(cell, args.layer, elev, azim, stem + '.png'))
        print(f'wrote {os.path.relpath(path, REPO)}')
        print(f'      {os.path.relpath(txt, REPO)}')
        print(f'    {lab["a1"]} : {lab["a2"]} :: {lab["b1"]} : {lab["b2"]}  '
              f'C{m["carbon_count"] or "-"}  '
              f'rank {m["rank_excl"]}/{m["n_candidates_excl"]}  3D {m["rank3d_excl"]}  '
              f'cos {num(m, "cos_to_target"):.3f}  '
              f'{m["n_hit1"]}/{m["n_corners"]} hit@1  {m["n_degenerate"]} degen  '
              f'recon {num(m, "recon_frac"):.2f}  '
              f'margin {num(m, "dist3d_true"):.1f} vs {num(m, "dist3d_rival"):.1f}'
              + ('  [negative control]' if int(m['is_negative_control']) else ''))
        if int(m['rank3d_excl']) != int(m['rank_excl']):
            print(f'    WARNING: projects to rank {m["rank3d_excl"]} but is rank '
                  f'{m["rank_excl"]} in full 4096-D -- the panel says so, but consider '
                  'dropping it')
    print(f'\n{len(cells)} figure(s) + {len(cells)} report(s) in '
          f'{os.path.relpath(args.out_dir, REPO)}')


if __name__ == '__main__':
    main()
