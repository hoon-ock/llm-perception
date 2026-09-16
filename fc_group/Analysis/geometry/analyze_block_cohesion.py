#!/usr/bin/env python3
"""How tightly each chemical block holds together, per model and per layer.

    python fc_group/Analysis/geometry/analyze_block_cohesion.py

Backs the depth-trajectory claim in the structure section: the alkyl-halide block resolves
first and the nitrogen, sulfur and carbonyl blocks follow by the final layer. The figure
(`f2_block_depth.pdf`) shows the clustermaps; this writes the numbers behind them so the
prose can quote a value that `check_numbers.py` can trace.

Reads `taxonomy/data/between_class_matrix.csv` -- a committed Analysis CSV, not `Results/`
-- so the reproducible-without-Results property holds. Only the `cosine_sim` column is
used; the partition-membership columns in that file are not read.

The four blocks are composition-based and deliberately **not** a partition: `amide` belongs
to both the nitrogen block (it carries N) and the carbonyl block (it carries C=O), which is
the overlap the ambiguity analysis is about. Each block is scored on its own, so membership
overlap is not double-counting across a partition -- there is no partition.
"""
import argparse
import csv
import os
from collections import defaultdict
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))
SRC = os.path.join(ANALYSIS, 'taxonomy', 'data', 'between_class_matrix.csv')
OUT = os.path.join(HERE, 'data', 'block_cohesion_by_layer.csv')

BLOCKS = {
    'halide': ['alkyl fluoride', 'alkyl chloride', 'alkyl bromide', 'alkyl iodide'],
    'nitrogen': ['amine', 'imine', 'nitrile', 'amide', 'nitro'],
    'sulfur': ['thiol', 'thioether', 'sulfoxide', 'sulfone'],
    'carbonyl': ['aldehyde', 'ketone', 'ester', 'carboxylic acid', 'amide'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=OUT)
    args = p.parse_args()

    if not os.path.exists(SRC):
        raise SystemExit(f'missing {SRC} -- run analyze_taxonomy.py that writes it')
    with open(SRC) as fh:
        rows = list(csv.DictReader(fh))

    # Group the pair list by (model, layer) so the z-scoring happens inside one model at
    # one depth, as every between-class contrast in the paper does: raw cosines differ in
    # level between models for anisotropy reasons that carry no chemistry.
    cells = defaultdict(list)
    for r in rows:
        cells[(r['model'], int(r['layer']))].append(r)

    out = []
    for (model, layer), pairs in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        sims = [float(r['cosine_sim']) for r in pairs]
        mu = mean(sims)
        sd = (sum((s - mu) ** 2 for s in sims) / (len(sims) - 1)) ** 0.5
        z = {}
        for r in pairs:
            key = frozenset((r['group_a'], r['group_b']))
            z[key] = (float(r['cosine_sim']) - mu) / sd

        for block, members in BLOCKS.items():
            within = [z[frozenset((a, b))]
                      for i, a in enumerate(members) for b in members[i + 1:]
                      if frozenset((a, b)) in z]
            others = [v for k, v in z.items() if not k <= set(members)]
            out.append(dict(
                model=model, entity_type='functional_group', layer=layer, block=block,
                n_members=len(members), n_within_pairs=len(within),
                mean_within_z=round(mean(within), 6),
                mean_other_z=round(mean(others), 6),
                cohesion=round(mean(within) - mean(others), 6),
            ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f'wrote {os.path.relpath(args.out, ANALYSIS)}  ({len(out)} rows)')

    # Console view: the trajectory the figure draws, base model only.
    base = 'meta-llama-Llama-3.1-8B'
    layers = sorted({r['layer'] for r in out})
    print(f"\n{base} -- cohesion (mean within-block z minus mean other-pair z)")
    print(f"{'block':<10}" + ''.join(f'{f"L{L}":>9}' for L in layers))
    for block in BLOCKS:
        cells_ = {r['layer']: r['cohesion'] for r in out
                  if r['model'] == base and r['block'] == block}
        print(f'{block:<10}' + ''.join(f'{cells_[L]:>9.3f}' for L in layers))


if __name__ == '__main__':
    main()
