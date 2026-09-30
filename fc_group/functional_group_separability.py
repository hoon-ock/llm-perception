#!/usr/bin/env python3
"""Per-functional-group separability of the activation space, one number per group.

Every other per-group representation readout in this repo is either pooled over groups
(`Analysis/geometry/data/retrieval_by_axis.csv` collapses the group axis) or covers only some of
them (the analogy retrieval axes reach 17 of 20 groups, at 1-16 trials each). Neither can be put
on a scatter against per-group generation accuracy, which is what
`Analysis/generation/analyze_generation_vs_representation.py` needs.

So this asks the simplest per-group question the space can answer: **how far is group g's own
region from its nearest rival?** For every row of group g,

    rival_margin = cos(x, centroid_g) - max_{h != g} cos(x, centroid_h)

with `centroid_g` computed leave-one-out so a molecule is never compared against a centroid it
helped define -- without that, singleton-heavy groups score high for arithmetic reasons. Positive
margin means the row is closer to its own group than to any other; the magnitude is the headroom.
`nearest_rival` records which group came second, which is the per-group confusion story the
pooled retrieval numbers cannot give.

`silhouette` is reported beside it as the standard-normalization view of the same geometry
(cosine metric, per-sample, averaged within group). The two disagree when a group is tight but
close to a rival, which is exactly the case worth seeing.

Reads the saved activations only -- no GPU, no model load. Loaders are imported from
`functional_group_analogy_carbon_matched` for the same reason `functional_group_probe.py` imports
them: one copy of the filename convention, not three.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples

from model_registry import default_five_layers, get_model_config
from functional_group_analogy_carbon_matched import (
    find_layer_file, load_activations, save_json, save_long_format_csv,
)

MODEL_NAME = 'meta-llama/Llama-3.1-8B'
CSV_PATH = 'fc_group/functional_group_dataset.csv'
BASE_ACTIVATIONS_DIR = 'fc_group/activation_datasets_functional_groups'
DEFAULT_OUTPUT_DIR = 'fc_group/Results/functional_group_separability'
DEFAULT_ENTITY_TYPE = 'functional_group'
GROUP_COLUMN = 'functional_group'

COLUMNS = ['model', 'entity_type', 'layer', 'depth', 'group', 'n_rows', 'n_molecules',
           'rival_margin', 'rival_margin_sd', 'frac_rows_positive_margin',
           'centroid_cos_self', 'centroid_cos_rival', 'nearest_rival',
           'silhouette', 'within_cos', 'centroid_norm']


def build_rows(activations, df, template_mode):
    """Align saved activation rows with per-molecule group labels.

    Mirrors `functional_group_probe.build_design_matrix`: the extractor emits prompts
    molecule-major / template-minor, so row i belongs to molecule i // templates_per_molecule.
    The divisibility assert is the whole safety net against an activation file and a CSV that
    have drifted apart, so it stays.
    """
    n_rows = activations.shape[0]
    num_molecules = len(df)
    if n_rows % num_molecules != 0:
        raise SystemExit(
            f"{n_rows} activation rows are not divisible by {num_molecules} molecules "
            f"in {CSV_PATH} -- the activation file and the CSV are out of sync.")
    per_mol = n_rows // num_molecules
    X = activations.astype(np.float32)
    mol_index = np.repeat(np.arange(num_molecules), per_mol)
    if template_mode == 'average':
        X = X.reshape(num_molecules, per_mol, -1).mean(axis=1)
        mol_index = np.arange(num_molecules)
    labels = df[GROUP_COLUMN].values[mol_index]
    return X, labels, mol_index, per_mol


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def separability(X, labels, mol_index):
    """Leave-one-molecule-out cosine margin to own centroid vs the nearest rival centroid.

    Held out per *molecule*, not per row: with 10 templates of the same molecule in the matrix,
    dropping one row still leaves nine near-duplicates in the centroid and the margin would be
    measuring template noise instead of group membership.
    """
    Xn = unit(X)
    groups = sorted(set(labels))
    sums = {g: Xn[labels == g].sum(axis=0) for g in groups}
    counts = {g: int((labels == g).sum()) for g in groups}

    out = {}
    for g in groups:
        rows = np.where(labels == g)[0]
        mols = np.unique(mol_index[rows])
        self_cos, rival_cos, rival_name = [], [], []
        for m in mols:
            held = rows[mol_index[rows] == m]
            # Own centroid with this molecule's rows removed; rivals are untouched.
            own = (sums[g] - Xn[held].sum(axis=0)) / max(counts[g] - len(held), 1)
            own = own / max(np.linalg.norm(own), 1e-12)
            cs = Xn[held] @ own
            best_h, best_c = None, -np.inf
            for h in groups:
                if h == g:
                    continue
                c = sums[h] / counts[h]
                c = c / max(np.linalg.norm(c), 1e-12)
                v = float((Xn[held] @ c).mean())
                if v > best_c:
                    best_h, best_c = h, v
            self_cos.extend(cs.tolist())
            rival_cos.extend([best_c] * len(held))
            rival_name.extend([best_h] * len(held))
        self_cos = np.asarray(self_cos)
        rival_cos = np.asarray(rival_cos)
        margins = self_cos - rival_cos
        # The modal nearest rival across this group's molecules, not the last one seen.
        rival = pd.Series(rival_name).value_counts().idxmax()
        cent = sums[g] / counts[g]
        out[g] = dict(
            n_rows=counts[g],
            n_molecules=int(len(mols)),
            rival_margin=float(margins.mean()),
            rival_margin_sd=float(margins.std(ddof=0)),
            frac_rows_positive_margin=float((margins > 0).mean()),
            centroid_cos_self=float(self_cos.mean()),
            centroid_cos_rival=float(rival_cos.mean()),
            nearest_rival=rival,
            centroid_norm=float(np.linalg.norm(cent)),
        )
    return out


def within_cos(X, labels):
    """Mean pairwise cosine inside each group -- tightness, independent of any rival."""
    Xn = unit(X)
    out = {}
    for g in sorted(set(labels)):
        V = Xn[labels == g]
        if len(V) < 2:
            out[g] = float('nan')
            continue
        S = V @ V.T
        iu = np.triu_indices(len(V), k=1)
        out[g] = float(S[iu].mean())
    return out


def run_layer(layer, num_layers, df, entity_type, activations_dir, model_name, template_mode):
    path = os.path.join(activations_dir, find_layer_file(activations_dir, entity_type, layer))
    X, labels, mol_index, per_mol = build_rows(load_activations(path), df, template_mode)

    sep = separability(X, labels, mol_index)
    tight = within_cos(X, labels)
    # Cosine silhouette needs unit rows; on unit vectors euclidean and cosine order identically,
    # but ask for cosine explicitly so a future normalization change cannot silently alter it.
    sil = silhouette_samples(unit(X), labels, metric='cosine')

    rows = []
    for g in sorted(sep):
        s = sep[g]
        rows.append([model_name, entity_type, layer, layer / (num_layers - 1), g,
                     s['n_rows'], s['n_molecules'],
                     s['rival_margin'], s['rival_margin_sd'], s['frac_rows_positive_margin'],
                     s['centroid_cos_self'], s['centroid_cos_rival'], s['nearest_rival'],
                     float(sil[labels == g].mean()), tight[g], s['centroid_norm']])
    neg = [r[4] for r in rows if r[7] <= 0]
    print(f"  layer {layer:3d} | mean margin {np.mean([r[7] for r in rows]):+.4f} | "
          f"mean silhouette {np.mean([r[13] for r in rows]):+.4f} | "
          f"{len(neg)}/{len(rows)} group(s) with non-positive margin"
          + (f": {', '.join(neg)}" if neg else ""))
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--entity-type', default=DEFAULT_ENTITY_TYPE)
    p.add_argument('--model-name', default=MODEL_NAME)
    p.add_argument('--layers', default=None,
                   help="Comma-separated layer indices. Default: default_five_layers().")
    p.add_argument('--all-layers', action='store_true',
                   help="Every layer, overriding --layers.")
    p.add_argument('--template-mode', choices=('expand', 'average'), default='expand',
                   help="expand = one row per (molecule, template), matching the probe's default.")
    p.add_argument('--output-dir', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    entity_type = args.entity_type
    num_layers = get_model_config(args.model_name)['num_layers']
    model_slug = args.model_name.replace('/', '-')

    if args.all_layers:
        layers = list(range(num_layers))
    elif args.layers:
        layers = [int(l) for l in args.layers.split(',')]
    else:
        layers = default_five_layers(num_layers)

    activations_dir = os.path.join(BASE_ACTIVATIONS_DIR, model_slug, entity_type)
    if not os.path.isdir(activations_dir):
        print(f"No activation directory {activations_dir}")
        return

    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, model_slug, entity_type)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    print(f"{entity_type}: {len(df)} molecules, {df[GROUP_COLUMN].nunique()} groups "
          f"({args.template_mode} mode)")

    rows = []
    for layer in layers:
        rows.extend(run_layer(layer, num_layers, df, entity_type, activations_dir,
                              model_slug, args.template_mode))

    out_csv = os.path.join(output_dir, 'data', 'separability_by_group.csv')
    save_long_format_csv(rows, COLUMNS, out_csv)
    save_json({'model': model_slug, 'entity_type': entity_type,
               'template_mode': args.template_mode, 'layers': layers,
               'n_groups': int(df[GROUP_COLUMN].nunique()), 'n_molecules': len(df)},
              os.path.join(output_dir, 'data', 'separability_summary.json'))
    print(f"\nWrote {len(rows)} rows to {out_csv}")


if __name__ == '__main__':
    main()
