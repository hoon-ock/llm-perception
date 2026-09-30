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

# APPENDED, never inserted: `main` loops over `args.models` in order and the writer
# preserves it, so adding to the end leaves the original three models' rows byte-identical.
# All five have both readings on disk -- a family_probability.csv and five
# between_class_similarity_layer_*.csv each.
MODELS = [
    'meta-llama-Llama-3.1-8B',
    'phenixace-Chem-R-Faithful',
    'deepseek-ai-DeepSeek-R1-Distill-Llama-8B',
    'weidawang-Chem-R-8B',
    'OpenDFM-ChemDFM-v1.5-8B',
]

# The five coarse families the probe scores, in the order `class_names` reports them.
FAMILIES = ['halide', 'hydrocarbon', 'nitrogen', 'oxygen', 'sulfur']
# The rival family, and the groups that define it geometrically.
RIVAL = 'oxygen'
OXYGEN_GROUPS = ['alcohol', 'aldehyde', 'ketone', 'ester', 'carboxylic acid', 'ether']

# Controls are the group's own family members that contain NO oxygen, so a model
# that is merely diffuse scores zero -- its controls are diffuse too.
CONTROLS = {'nitrogen': ['amine', 'imine', 'nitrile'],
            'sulfur': ['thiol', 'thioether']}
AMBIGUOUS = {'amide': 'nitrogen', 'nitro': 'nitrogen',
             'sulfone': 'sulfur', 'sulfoxide': 'sulfur'}

# Condensed structural formula per group, read from the dataset's own
# `functional_group_structure` column rather than written out here -- one authority for what
# a group IS, shared with everything else that quotes it. Carried into the snapshot so a
# figure can label `amide` with the notation without reloading the molecule table.
STRUCTURE_CSV = 'functional_group_dataset.csv'


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


def read_structures(repo):
    """`group -> condensed formula`, e.g. amide -> -CONH2, from the dataset."""
    path = os.path.join(repo, 'fc_group', STRUCTURE_CSV)
    out = {}
    for r in read_csv(path):
        out.setdefault(r['functional_group'], r['functional_group_structure'])
    return out


def fold_accuracy(results_dir, model, entity_type, tag='coarse_group'):
    """`(layer, group) -> argmax accuracy` on that group's own held-out fold.

    Leave-one-group-out makes each fold a single-class test set, so this is the recall on
    that group -- the fraction of its rows the probe assigned to the IUPAC home family.

    It belongs beside the probability mass because the two can disagree completely, and the
    disagreement is the whole argument: at layer 31 the base model scores 1.000 on `amide`
    while putting 0.156 on oxygen, and chemdfm scores 0.225 on the same group. Accuracy sees
    a model that got it right and a model that got it wrong; the mass shows one committed to
    a naming convention and the other to the chemistry.
    """
    path = os.path.join(results_dir, 'functional_group_probe', model, entity_type,
                        'data', f'probe_scores_{tag}.csv')
    return {(int(r['layer']), r['fold']): float(r['accuracy']) for r in read_csv(path)}


def snapshot_family_probability(results_dir, model, entity_type, structures):
    """Copy the probe's family-probability table into Analysis/ for every layer.

    `posterior_ratio` reduces this to one ratio at one layer, which is the right input for
    the two-readings table but throws away what a figure needs: the actual mass on each of
    the five families, and the CONTROL groups alongside the ambiguous ones. Without the
    controls in view a reader sees a model committing hard on `amide` and concludes it denies
    the ambiguity, when the same model may be committing just as hard everywhere -- which is
    what the base model in fact does (controls r = 0.012 against chem-r's 0.226).

    Snapshotted here rather than read from `Results/` by the figure script, so the
    reproducible-without-Results property holds for the visuals too.
    """
    path = os.path.join(results_dir, 'ambiguity_metric', model, entity_type, 'data',
                        'family_probability.csv')
    acc = fold_accuracy(results_dir, model, entity_type)
    keep = set(AMBIGUOUS) | {c for v in CONTROLS.values() for c in v}
    out = []
    for r in read_csv(path):
        if r['group'] not in keep:
            continue
        home = r['home_family']
        p_ox, p_home = float(r[f'p_{RIVAL}']), float(r[f'p_{home}'])
        denom = p_ox + p_home
        out.append({
            'model': model, 'entity_type': entity_type, 'layer': int(r['layer']),
            'group': r['group'], 'role': r['role'], 'home_family': home,
            'n_molecules': int(r['n_molecules']),
            'p_home': round(p_home, 6), 'p_oxygen': round(p_ox, 6),
            # Summed from the remaining three family columns, NOT derived as 1 - home - ox.
            # The stored probabilities are float16 and do not sum to exactly one: base/L0/
            # nitro already has home + oxygen = 1.000044, so subtraction would clamp to zero
            # and silently hide that the row is 4e-5 over. Summing what is actually there
            # keeps the bar honest about its own total.
            'p_other': round(sum(float(r[f'p_{f}']) for f in FAMILIES
                                 if f not in (RIVAL, home)), 6),
            'posterior_r': round(p_ox / denom, 6) if denom > 0 else '',
            'accuracy': acc.get((int(r['layer']), r['group']), ''),
            'structure': structures.get(r['group'], ''),
        })
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
    # Second output: the full per-layer mass table the figures draw from, ambiguous groups
    # and their controls together. Written after the headline table so a failure here cannot
    # leave the two-readings CSV half-written.
    structures = read_structures(REPO)
    fam = [r for model in args.models
           for r in snapshot_family_probability(args.results_dir, model, args.entity_type,
                                                structures)]
    write_csv(os.path.join(args.out_dir, 'ambiguity_family_probability.csv'), fam,
              ['model', 'entity_type', 'layer', 'group', 'role', 'home_family', 'structure',
               'n_molecules', 'p_home', 'p_oxygen', 'p_other', 'posterior_r', 'accuracy'])

    write_csv(os.path.join(args.out_dir, 'ambiguity_two_readings.csv'), rows,
              ['model', 'entity_type', 'layer', 'group', 'home_family', 'controls',
               'posterior_r', 'posterior_r_controls', 'posterior_delta',
               'geometric_lean', 'geometric_lean_controls', 'geometric_delta',
               'best_C_mode', 'best_C_frac', 'n_folds'])


if __name__ == '__main__':
    main()
