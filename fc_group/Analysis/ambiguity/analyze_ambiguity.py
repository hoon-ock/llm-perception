#!/usr/bin/env python3
"""The convention-labelled groups, read two ways -- and why only one ranks models.

Four groups carry oxygen but are named for another heteroatom: amide and nitro are
labelled `nitrogen`, sulfoxide and sulfone `sulfur`. A model that puts real mass on
oxygen there is wrong in the chemically right direction, and accuracy cannot see it.

This emits the same question answered from two independent places, side by side:

  posterior  from `ambiguity_metric/.../family_probability.csv` -- the probe's own
             out-of-fold p(family), as r = p(O) / [p(O) + p(home)], contrasted
             against the group's own oxygen-free family controls.
  geometric  from `functional_group_analogy/.../between_class_similarity_layer_*.csv`
             -- mean z-cosine from the group to the six oxygen-named groups, same
             control contrast. No probe, no softmax, no regularization constant.

They must be read together, because the posterior column CANNOT be compared ACROSS
models. `select_C` picks a different L2 strength per model (Chem-R and DeepSeek-8B
both land on 1e-4, two orders below Llama's 1e-3), and softmax sharpness follows it,
so a cross-model "who is more balanced" ranking would be reading the regularization
path. `best_C_mode` and `best_C_frac` are emitted on every row so that confound is
visible in the table rather than buried in a methods note. See PROBE_NOTES.md S10.

The geometric column has no such parameter, and on this data it does NOT reproduce
the posterior ordering -- which is the finding: within a model the effect is solid
and present in all three, but ranking models on it needs `ambiguity_metric.py --full
--mode projection`, which has not been run.

Reads only `Results/`. No activations, no model, CPU-only.
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DEFAULT_RESULTS = os.path.join(REPO, 'fc_group', 'Results')
DEFAULT_OUT = os.path.join(HERE, 'data')

MODELS = [
    'meta-llama-Llama-3.1-8B',
    'phenixace-Chem-R-Faithful',
    'deepseek-ai-DeepSeek-R1-Distill-Llama-8B',
]

# The rival family, and the groups that define it geometrically.
RIVAL = 'oxygen'
OXYGEN_GROUPS = ['alcohol', 'aldehyde', 'ketone', 'ester', 'carboxylic acid', 'ether']

# Controls are the group's own family members that contain NO oxygen, so a model
# that is merely diffuse scores zero -- its controls are diffuse too.
CONTROLS = {'nitrogen': ['amine', 'imine', 'nitrile'],
            'sulfur': ['thiol', 'thioether']}
AMBIGUOUS = {'amide': 'nitrogen', 'nitro': 'nitrogen',
             'sulfone': 'sulfur', 'sulfoxide': 'sulfur'}


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def posterior_ratio(results_dir, model, entity_type, layer):
    """r = p(oxygen) / [p(oxygen) + p(home family)] per group, at one layer."""
    path = os.path.join(results_dir, 'ambiguity_metric', model, entity_type, 'data',
                        'family_probability.csv')
    out = {}
    for row in read_csv(path):
        if int(row['layer']) != layer:
            continue
        p_ox = float(row[f'p_{RIVAL}'])
        p_home = float(row[f"p_{row['home_family']}"])
        denom = p_ox + p_home
        out[row['group']] = p_ox / denom if denom > 0 else float('nan')
    return out


def geometric_lean(results_dir, model, entity_type, layer):
    """Mean z-cosine from each group to the oxygen-named groups.

    z-scored within the model's own between-class distribution, so a model with
    globally lower cosines is not read as globally less oxygen-leaning.
    """
    path = os.path.join(results_dir, 'functional_group_analogy', model, entity_type,
                        'data', f'between_class_similarity_layer_{layer}.csv')
    rows = read_csv(path)
    vals = np.array([float(r['cosine_sim']) for r in rows])
    mu, sd = vals.mean(), vals.std(ddof=1)
    lean = {}
    for r, v in zip(rows, (vals - mu) / sd):
        for g, other in ((r['group_a'], r['group_b']), (r['group_b'], r['group_a'])):
            if other in OXYGEN_GROUPS and g not in OXYGEN_GROUPS:
                lean.setdefault(g, []).append(v)
    return {g: float(np.mean(v)) for g, v in lean.items()}


def best_C(results_dir, model, entity_type, layer, tag='coarse_group'):
    """Modal L2 strength over the 19 folds AT THE LAYER THE AMBIGUITY IS READ AT.

    This is the confound, not a diagnostic: it is reported so the posterior column
    is never read across models without it in view. It has to be taken at the same
    layer as the probabilities -- `select_C` varies with depth, and quoting it from
    each model's own best layer would compare regularizations that never coexisted.

    At layer 31 this matters: Llama (1e-3 x18) and Chem-R (1e-3 x16) are MATCHED,
    so that one comparison is like-for-like, while DeepSeek-8B (1e-4 x19) sits two
    orders looser and cannot be ranked against either on posterior mass.
    """
    path = os.path.join(results_dir, 'functional_group_probe', model, entity_type,
                        'data', f'probe_scores_{tag}.csv')
    folds = [r for r in read_csv(path) if int(r['layer']) == layer]
    counts = Counter(r['best_C'] for r in folds)
    mode, n = counts.most_common(1)[0]
    return {'best_C_mode': float(mode), 'best_C_frac': round(n / len(folds), 3),
            'n_folds': len(folds)}


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {os.path.relpath(path, REPO)}  ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default=DEFAULT_RESULTS)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--entity-type', default='functional_group')
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--layers', type=int, nargs='+', default=[24, 31],
                   help='layers with BOTH a probability table and a cosine matrix')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for model in args.models:
        for layer in args.layers:
            c = best_C(args.results_dir, model, args.entity_type, layer)
            post = posterior_ratio(args.results_dir, model, args.entity_type, layer)
            geo = geometric_lean(args.results_dir, model, args.entity_type, layer)
            for group, home in AMBIGUOUS.items():
                ctrl = CONTROLS[home]
                row = {'model': model, 'entity_type': args.entity_type,
                       'layer': layer, 'group': group, 'home_family': home,
                       'controls': '|'.join(ctrl)}
                row['posterior_r'] = post.get(group)
                row['posterior_r_controls'] = np.mean([post[c_] for c_ in ctrl])
                row['posterior_delta'] = row['posterior_r'] - row['posterior_r_controls']
                row['geometric_lean'] = geo.get(group)
                row['geometric_lean_controls'] = np.mean([geo[c_] for c_ in ctrl])
                row['geometric_delta'] = (row['geometric_lean']
                                          - row['geometric_lean_controls'])
                row.update(c)
                rows.append({k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in row.items()})
    write_csv(os.path.join(args.out_dir, 'ambiguity_two_readings.csv'), rows,
              ['model', 'entity_type', 'layer', 'group', 'home_family', 'controls',
               'posterior_r', 'posterior_r_controls', 'posterior_delta',
               'geometric_lean', 'geometric_lean_controls', 'geometric_delta',
               'best_C_mode', 'best_C_frac', 'n_folds'])


if __name__ == '__main__':
    main()
