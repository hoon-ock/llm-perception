#!/usr/bin/env python3
"""Prompt wording against chemistry, as a depth curve.

    python fc_group/Analysis/paper/make_visuals_prompt_structure.py [--out-dir DIR]
    -> fc_group/Analysis/visuals/prompt_structure/label_agreement_depth.png

Two panels: for each point, the share of its 10 nearest neighbours in ACTIVATION space
carrying the same prompt template, and the share carrying the same functional group. One line
per checkpoint, every layer, each panel against its own random-neighbour floor -- the two
labels differ in class count and balance (10 balanced templates against 20 unbalanced groups),
so a single reference line would make them look comparable when they are not.

What it shows: through the first half of the stack every model's neighbourhoods are organised
by which template produced the prompt, not by the molecule. Chemical structure overtakes it
around layer 19 -- measured against the family label, which is in the CSV but not drawn here --
and from there the checkpoints separate: every chemistry fine-tune ends BELOW base on template
and ABOVE it on fine groups, while the reasoning distill tracks base.

Companion to `f1_probe_depth`, and deliberately a different question: the probe asks whether a
linear reader CAN recover the class, this asks what the representation's own neighbourhood
structure is organised by. They do not answer at the same depth, and the gap is the point --
see `Analysis/prompt_structure/README.md`.

Reads only `fc_group/Analysis/prompt_structure/data/label_agreement_by_layer.csv`, never
`Results/`, which is gitignored.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from _common import ANALYSIS, BEHAVIOURAL, REPO, load, num  # noqa: E402
from make_visuals_probe import (COLOR5, DASH5, DISPLAY5, MARKER5, REF,  # noqa: E402
                                recessive, style)

DEFAULT_VISUALS = os.path.join(ANALYSIS, 'visuals', 'prompt_structure')

# The heteroatom-family panel was dropped on purpose. `COARSE_MAP` files four groups by the
# atom that NAMES them rather than the one they contain -- amide and nitro under nitrogen,
# sulfoxide and sulfone under sulfur -- so a point whose neighbours are chemically sensible
# scores as disagreement there, and the panel's ceiling is not 1.0 for reasons that have
# nothing to do with the model. `functional_group` carries no such convention. The columns
# stay in the CSV: they are what the crossover layer is measured against, and dropping a
# panel is not a reason to stop releasing a number.
PANELS = [
    ('template_index', 'Same prompt template'),
    ('functional_group', 'Same functional group'),
]


def read_rows():
    """`(model, layer) -> row`, from the released per-layer CSV."""
    return {(r['model'], int(r['layer'])): r
            for r in load('prompt_structure/data/label_agreement_by_layer.csv')}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_VISUALS)
    p.add_argument('--column', default='centered', choices=('centered', 'raw'),
                   help="Which reading to draw. `centered` is the headline -- see the "
                        "analysis script on why raw neighbour ranking is contaminated by "
                        "anisotropy. `raw` exists so the choice can be inspected, not hidden.")
    args = p.parse_args()
    style()

    rows = read_rows()
    models = [m for m in BEHAVIOURAL if any((m, L) in rows for L in range(32))]
    if not models:
        raise SystemExit('no rows -- run '
                         'fc_group/Analysis/prompt_structure/analyze_prompt_structure.py')
    layers = sorted({L for (_, L) in rows})

    fig, axes = plt.subplots(1, len(PANELS), figsize=(5.0, 2.45), sharex=True)
    for ax, (label, title) in zip(axes, PANELS):
        for m in models:
            dash = DASH5[m]
            ax.plot(layers, [num(rows[(m, L)], f'knn_{label}_{args.column}') for L in layers],
                    color=COLOR5[m], marker=MARKER5[m], markersize=2.6, linewidth=1.2,
                    markevery=4, dashes=dash if dash[0] else (), label=DISPLAY5[m])
        # Each panel's own floor: 10 balanced templates sit near 0.099, the 5 families near
        # 0.239 and the 20 groups near 0.054. Drawing one shared line would imply the three
        # panels share a scale of evidence, which they do not.
        floor = num(rows[(models[0], layers[0])], f'chance_{label}')
        ax.axhline(floor, linestyle=':', **REF)
        # `chance_{label}` above is the CSV COLUMN NAME and must not drift; the drawn label is
        # display text and says what the floor actually is. `chance_floor` is documented as
        # "P(a uniformly random OTHER point shares this point's label)" -- the agreement 10
        # neighbours drawn at random would give, i.e. the label's base rate. Nothing guesses a
        # label here, so this is NOT `probe_depth`'s 'Random guess' and is not worded like it.
        ax.annotate(f'Random neighbours {floor:.2f}', (layers[-1], floor),
                    textcoords='offset points', xytext=(-1, 3), ha='right', fontsize=6,
                    color='0.45')
        recessive(ax)
        ax.set_title(title)
        ax.set_xlabel('Layer')
        ax.set_ylim(0, 1)
        ax.set_xlim(layers[0], layers[-1])
    axes[0].set_ylabel(f'kNN label agreement (k={rows[(models[0], layers[0])]["k"]})')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(models),
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.4)
    # bottom reserves everything under the axes: x tick labels, the 'Layer' xlabel, and the
    # one-row legend anchored at y=0. 0.20 is not enough -- the legend rides up into the
    # xlabel; 0.31 (the 3-panel value) leaves a visible gap on a figure this short.
    fig.subplots_adjust(bottom=0.27, top=0.90, left=0.098, right=0.994, wspace=0.17)

    os.makedirs(args.out_dir, exist_ok=True)
    name = ('label_agreement_depth.png' if args.column == 'centered'
            else f'label_agreement_depth_{args.column}.png')
    path = os.path.join(args.out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'wrote {os.path.relpath(path, REPO)}')


if __name__ == '__main__':
    main()
