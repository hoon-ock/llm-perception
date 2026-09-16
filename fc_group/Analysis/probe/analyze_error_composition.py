#!/usr/bin/env python3
"""What the probe's errors are ABOUT, per model and per layer.

    python fc_group/Analysis/probe/analyze_error_composition.py

`confusion_errors.csv` holds the raw cell counts. This reduces them to the quantity the
models section actually argues over: of the errors a model makes at one depth, what share
falls on the nitrogen->oxygen boundary.

The share, not the count, is the reportable quantity. Below the probe's transition every
model misclassifies heavily in every direction, so a large raw N->O count there reflects a
probe that does not work yet rather than anything about nitrogen and oxygen. Dividing by the
model's own error total at that layer removes that confound; `n_total_errors` is carried in
the output so a reader can see the denominator, which for the base model falls into single
digits in the deep layers and makes its share correspondingly noisy.

Reads only `probe/data/confusion_errors.csv` -- a committed Analysis CSV, not `Results/` --
so the reproducible-without-Results property holds.
"""
import argparse
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))
SRC = os.path.join(HERE, 'data', 'confusion_errors.csv')
OUT = os.path.join(HERE, 'data', 'error_composition_by_layer.csv')

BOUNDARY = ('nitrogen', 'oxygen')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=OUT)
    args = p.parse_args()

    if not os.path.exists(SRC):
        raise SystemExit(f'missing {SRC} -- run the probe sweep that writes it')
    with open(SRC) as fh:
        rows = list(csv.DictReader(fh))

    # Seed the full model x layer grid from the probe curves, not from the error file.
    # `confusion_errors.csv` holds error cells only, so a layer at which a model makes NO
    # errors contributes no rows and would silently vanish from the output. A missing row
    # and a perfect layer are very different facts, and the base model has two perfect
    # layers (29 and 30) that the section's argument depends on.
    grid = os.path.join(HERE, 'data', 'layer_curves.csv')
    if not os.path.exists(grid):
        raise SystemExit(f'missing {grid} -- run the probe sweep that writes it')
    with open(grid) as fh:
        total = {(r['model'], int(r['layer'])): 0 for r in csv.DictReader(fh)}
    boundary = dict.fromkeys(total, 0)
    for r in rows:
        if r['true_class'] == r['pred_class']:
            continue
        key = (r['model'], int(r['layer']))
        total[key] += int(r['n'])
        if (r['true_class'], r['pred_class']) == BOUNDARY:
            boundary[key] += int(r['n'])

    out = []
    for (model, layer) in sorted(total, key=lambda k: (k[0], k[1])):
        n_tot = total[(model, layer)]
        n_no = boundary[(model, layer)]
        out.append(dict(
            model=model, entity_type='functional_group', layer=layer,
            n_total_errors=n_tot, n_nitrogen_to_oxygen=n_no,
            share_nitrogen_to_oxygen=round(n_no / n_tot, 6) if n_tot else '',
        ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f'wrote {os.path.relpath(args.out, ANALYSIS)}  ({len(out)} rows)')

    short = {'meta-llama-Llama-3.1-8B': 'base',
             'phenixace-Chem-R-Faithful': 'chem',
             'deepseek-ai-DeepSeek-R1-Distill-Llama-8B': 'reason'}
    print("\nN->O share of each model's own errors (denominator in parentheses)")
    print(f"{'layer':>6}" + ''.join(f'{s:>18}' for s in short.values()))
    for L in range(19, 32):
        line = f'{L:>6}'
        for m in short:
            row = next((r for r in out if r['model'] == m and r['layer'] == L), None)
            if row is None or row['share_nitrogen_to_oxygen'] == '':
                line += f"{'-- (0)':>18}"
            else:
                line += f"{row['share_nitrogen_to_oxygen']:>12.2f} ({row['n_total_errors']})"
        print(line)


if __name__ == '__main__':
    main()
