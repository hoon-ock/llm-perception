#!/usr/bin/env python3
"""Where the probe puts its probability on the convention-labelled groups.

    python fc_group/Analysis/paper/make_visuals_ambiguity.py [--layer 31] [--out-dir DIR]
    -> fc_group/Analysis/visuals/ambiguity/mass_split_L<layer>.png

`f1_probe_depth.pdf`'s lower panel counts ARGMAX mistakes, so it cannot show how a model
splits probability between two defensible answers. Four groups carry oxygen but are named for
another heteroatom -- `amide` and `nitro` are labelled nitrogen, `sulfone` and `sulfoxide`
sulfur -- and under leave-one-group-out nothing in the chemistry picks the home family over
oxygen. Only IUPAC does. This draws the mass itself, with the accuracy that disagrees with it.

Output lands in `fc_group/Analysis/visuals/ambiguity/`, beside the analysis outputs and
deliberately not `paper/figures/`, so nothing here can reach the manuscript by accident.

TWO THINGS THIS FIGURE DOES NOT SAY, both of which live in that directory's README.md:

  * the raw mass is NOT rankable across all five models. The probe picks its own L2 strength
    per model and softmax sharpness follows it -- at layer 31 base, chem-r and chem-faithful
    sit at 1e-3 while chemdfm and reason sit at 1e-4, ten times stronger. Only a matched-C
    subset can be ranked on these bars.
  * base's low oxygen mass is NOT specific to the ambiguous groups. Its controls sit at
    r = 0.012 against chem-r's 0.226; base is sharper everywhere. `posterior_delta` in
    `ambiguity/data/ambiguity_two_readings.csv` is the control-corrected reading, and on
    `amide` it puts base ahead of chem-r rather than far behind.

Reads only `fc_group/Analysis/ambiguity/data/*.csv`, never `Results/`, which is gitignored.
`analyze_ambiguity.py` snapshots the family-probability table there for this purpose.
"""
import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import ANALYSIS, BEHAVIOURAL, REPO, load, num  # noqa: E402
from make_visuals_probe import DISPLAY5, style  # noqa: E402

DEFAULT_VISUALS = os.path.join(ANALYSIS, 'visuals', 'ambiguity')

AMBIGUOUS = ['amide', 'nitro', 'sulfone', 'sulfoxide']

# Sized for a printed page. This figure is dense and is meant to survive being scaled into a
# column, so the row pitch is derived from ACC_FS below rather than guessed -- at 45 rows the
# pitch fell under 9pt and the accuracy column overlapped itself.
LABEL_FS = 23        # x-axis title
DIVIDER_FS = 21      # the controls rule
GROUP_FS = 21        # group name, left of the bars
LEGEND_FS = 20
TICK_FS = 18         # x tick numbers
MODEL_FS = 17        # model names down the left of each block
ACC_FS = 17
STRUCT_FS = 17       # the condensed formula under the group name

# The three-way split of a bar. `home` is the IUPAC answer, `oxygen` the chemically
# defensible rival, `other` everything the probe put elsewhere.
STACK = [('p_home', '#0072B2', 'p(Home Family)'),
         ('p_oxygen', '#D55E00', 'p(Oxygen)'),
         ('p_other', '0.82', 'p(Other Three)')]

_DIGITS = re.compile(r'(?<=[A-Za-z])(\d+)')


def formula(text):
    """`-CONH2` -> `-CONH$_2$`. Inline mathtext for the digits, nothing else touched.

    Unicode subscripts (₂) were the obvious choice and do not work: Times New Roman has
    no glyph for them and matplotlib silently drops the character. It DOES carry ≡, so
    `-C≡N` is left exactly as the dataset wrote it and only the digits are rewritten.
    Every other character passes through untouched, so this cannot disagree with
    `functional_group_structure`.
    """
    return _DIGITS.sub(r'$_\1$', text)


def read_mass():
    """`(model, layer, group) -> row` from the committed snapshot."""
    return {(r['model'], int(r['layer']), r['group']): r
            for r in load('ambiguity/data/ambiguity_family_probability.csv')}


def control_groups(mass, layer):
    """The oxygen-free control groups, grouped by the family they control for."""
    fams = {}
    for (model, L, g), r in mass.items():
        if L == layer and r['role'] != 'ambiguous':
            fams.setdefault(r['home_family'], set()).add(g)
    return {k: sorted(v) for k, v in fams.items()}


def control_block(mass, layer, model, groups):
    """One averaged row standing for a family's oxygen-free control groups.

    The five controls are shown as two rows, not five, for the same reason `delta` uses their
    MEAN: individually they are a distraction from what they are there to establish, which is
    the level a model sits at when the label is NOT a convention. Collapsing them is also what
    lets the type be read at print size -- 45 rows put the row pitch below 9pt.

    A plain mean of the MASSES, unweighted by molecule count, so the three segments still sum
    to the stack's own total. Note this is NOT the reduction `posterior_delta` uses: that
    averages the RATIO r = p(O)/[p(O)+p(home)] over the control groups, and a mean of ratios
    is not the ratio of means. At layer 31 the two differ by at most 0.021 (chem-r, sulfur) --
    far too small to move anything visible here, but the control bar is a picture of the
    control level, not a redraw of `delta`'s denominator, and should not be read off as one.
    """
    rows = [mass[(model, layer, g)] for g in groups]
    out = {k: float(np.mean([num(r, k) for r in rows]))
           for k in ('p_home', 'p_oxygen', 'p_other')}
    out['accuracy'] = float(np.mean([num(r, 'accuracy') for r in rows]))
    return out


def mass_figure(mass, layer):
    """Stacked probability mass, with the accuracy that disagrees with it.

    Three things share one row: the group's condensed formula (left), how the probe spread its
    probability (centre), and the fold accuracy (right) -- still an argmax over the five
    families, scored one decision per prompt over that group's molecules x 10 templates. The
    right column is the reason the figure exists: base scores 1.000 on every group here while
    holding 0.156 on oxygen for amide, so accuracy reports a solved task on exactly the groups
    whose label is a naming convention rather than a fact about the molecule.

    The accuracy column is a real second axes sharing y, not text floated over the first. At
    this type size floated annotations collided with the bars, and a shared axis makes
    mis-registration between a number and its bar impossible rather than merely unlikely.
    """
    ctrl = control_groups(mass, layer)
    blocks = [(g.title(), mass[(BEHAVIOURAL[0], layer, g)]['structure'], None)
              for g in AMBIGUOUS]
    blocks.append((None, None, None))
    for fam, groups in sorted(ctrl.items()):
        blocks.append((f'{fam.title()} Controls',
                       ', '.join(g.title() for g in groups), groups))

    rows_per = len(BEHAVIOURAL)
    total = len(blocks) * (rows_per + 0.9) + 2.5
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(12.6, 0.232 * total + 1.7), sharey=True,
        gridspec_kw=dict(width_ratios=[1, 0.10], wspace=0.015))

    y, ticks, ticklab = 0.0, [], []
    divider_y = None
    for name, sub, members in blocks:
        if name is None:
            divider_y = y + 0.2
            # 3.0, not 2.2: the divider title sits at `divider_y + 1.2`, so the extra units
            # all land BELOW it, between the title and the first control bar. At 2.2 the
            # title crowded that bar and read as a label on it rather than on the block.
            y += 3.0
            continue
        for m in BEHAVIOURAL:
            r = (control_block(mass, layer, m, members) if members
                 else mass[(m, layer, name.lower())])
            left = 0.0
            for key, col, _ in STACK:
                w = float(r[key]) if not isinstance(r[key], str) else num(r, key)
                ax.barh(y, w, left=left, height=0.82, color=col,
                        edgecolor='white', linewidth=0.5)
                left += w
            if r['accuracy'] != '':
                axr.text(0.5, y, f'{float(r["accuracy"]):.3f}', ha='center', va='center',
                         fontsize=ACC_FS)
            # `DISPLAY5`, not `SHORT5`: the lower-case slugs are identifiers (filenames,
            # `--only`), and a reader should not be shown one where a name belongs.
            ticks.append(y); ticklab.append(DISPLAY5[m])
            y += 1.0
        mid = y - rows_per / 2.0 - 0.5
        # AXES fraction on x, data on y. Far enough left to clear the longest model label
        # ('Chem-DFM'); at a smaller offset the group name sat on top of it. -0.155 was tried
        # when `DISPLAY5` shortened the labels and is too tight: the condensed formula, which
        # sits one row lower, runs into 'Chem-DFM' on every block.
        ax.annotate(name, xy=(-0.185, mid - 0.40), xycoords=('axes fraction', 'data'),
                    ha='right', va='center', fontsize=GROUP_FS, fontweight='bold')
        ax.annotate(formula(sub), xy=(-0.185, mid + 0.70),
                    xycoords=('axes fraction', 'data'), ha='right', va='center',
                    fontsize=STRUCT_FS, color='0.3')
        y += 0.9

    ax.set_yticks(ticks); ax.set_yticklabels(ticklab, fontsize=MODEL_FS)
    ax.set_ylim(y - 0.4, -1.9)          # inverted, with headroom for the column header
    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean Out-of-fold Probability', fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis='x', color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)

    # The accuracy column: an axes with no scale of its own, only aligned text.
    axr.set_xlim(0, 1)
    axr.set_xticks([])
    axr.text(0.5, -1.15, 'Accuracy', ha='center', va='center',
             fontsize=ACC_FS, fontweight='bold')
    for side in axr.spines:
        axr.spines[side].set_visible(False)
    axr.tick_params(left=False, labelleft=False)

    if divider_y is not None:
        # Above the rule: the four underdetermined groups. Below: the oxygen-free members of
        # the same families, averaged, which are what those four are read against. Without
        # the rule the block reads as six equivalent groups.
        for a in (ax, axr):
            a.axhline(divider_y, color='0.45', linewidth=1.0, zorder=3)
        ax.annotate('Controls: Oxygen-free Members of the Same Families',
                    xy=(0.5, divider_y + 1.2), xycoords=('axes fraction', 'data'),
                    ha='center', va='center', fontsize=DIVIDER_FS, color='0.2')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in STACK]
    ax.legend(handles, [lbl for _, _, lbl in STACK], loc='lower center',
              bbox_to_anchor=(0.5, 1.012), ncol=3, fontsize=LEGEND_FS, frameon=False)
    return fig


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    p.add_argument('--layer', type=int, default=31,
                   help='any layer present in ambiguity_family_probability.csv (0-31)')
    args = p.parse_args()
    style()
    # Mathtext must be set in the same serif face as the surrounding label, or a subscript
    # arrives in DejaVu beside Times and reads as a typo.
    plt.rcParams.update({'mathtext.fontset': 'custom',
                         'mathtext.rm': plt.rcParams['font.serif'][0],
                         'mathtext.it': plt.rcParams['font.serif'][0],
                         'mathtext.bf': plt.rcParams['font.serif'][0]})

    mass = read_mass()
    missing = [m for m in BEHAVIOURAL if (m, args.layer, 'amide') not in mass]
    if missing:
        raise SystemExit('no layer-%d rows for %s -- re-run '
                         'fc_group/Analysis/ambiguity/analyze_ambiguity.py'
                         % (args.layer, ', '.join(missing)))

    os.makedirs(args.out_dir, exist_ok=True)
    fig = mass_figure(mass, args.layer)
    path = os.path.join(args.out_dir, f'mass_split_L{args.layer}.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {os.path.relpath(path, REPO)}')


if __name__ == '__main__':
    main()
