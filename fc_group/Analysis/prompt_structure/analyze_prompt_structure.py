#!/usr/bin/env python3
"""Is the representation organised by prompt wording or by chemistry, and where does it flip?

    python fc_group/Analysis/prompt_structure/analyze_prompt_structure.py
    -> fc_group/Analysis/prompt_structure/data/label_agreement_by_layer.csv

For each point, the share of its k nearest neighbours IN ACTIVATION SPACE carrying the same
label, for three labels: which prompt template produced it, its heteroatom family, and its
functional group. One row per (model, layer), every layer, so the wording-to-chemistry
crossover can be located instead of bracketed.

This exists because `visuals/tsne/README.md` found the effect at five layers on 2-D t-SNE
coordinates, and was right to scope that to describing the panels. The same quantity measured
on the activations is a claim about the model, which is what a depth curve needs to be. The
two are companions, not substitutes: the t-SNE numbers stay where they are.

CENTRING IS LOAD-BEARING, NOT HOUSEKEEPING. The residual stream is strongly anisotropic --
mean pairwise cosine 0.944 in the base model at layer 31 (`geometry/data/geometry_by_layer.csv`)
-- so on raw vectors the shared mean direction dominates the neighbour ranking and every model
is scored partly on how collapsed its space is rather than on what it groups by. Mean-centring
first is the same correction `analyze_geometry.py` applies before it will quote a between-class
cosine at all. Both readings are released, `_centered` and `_raw`: centring is a choice, and a
reader has to be able to see whether the result depends on it.

Each label gets its own CHANCE FLOOR, because the three are not comparable otherwise -- the
10 templates are balanced (floor ~0.098) while the 5 families and 20 groups are not. Computed
analytically: a uniformly random neighbour matches with probability
`sum_c n_c (n_c - 1) / n (n - 1)`.

CPU-only. Reads activations; writes one CSV.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'fc_group'))

from functional_group_probe import BASE_ACTIVATIONS_DIR, COARSE_MAP, CSV_PATH  # noqa: E402
from functional_group_analogy_carbon_matched import (  # noqa: E402
    find_layer_file, load_activations,
)

DEFAULT_OUT = os.path.join(HERE, 'data')

# `_common.BEHAVIOURAL` order, so rows land in the same sequence as Tables 1 and 3.
MODELS = [
    'meta-llama-Llama-3.1-8B',
    'weidawang-Chem-R-8B',
    'phenixace-Chem-R-Faithful',
    'OpenDFM-ChemDFM-v1.5-8B',
    'deepseek-ai-DeepSeek-R1-Distill-Llama-8B',
]
LABELS = ['template_index', 'coarse_family', 'functional_group']
K = 10


def knn_agreement(X, labels, k):
    """Share of each point's k nearest cosine neighbours sharing its label.

    Cosine on L2-normalised rows is a monotone function of Euclidean distance, so ranking by
    dot product and ranking by distance give the same neighbours -- the dot product is just
    the cheaper way to get there for 920 points.

    `argpartition` then `argsort` on the top k+1: a full sort of 920 columns per row would be
    ~7x the work for an answer that only needs the k largest. The +1 is the point itself,
    which always sits at similarity 1.0 and is dropped.
    """
    Z = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Z @ Z.T
    np.fill_diagonal(sim, -np.inf)          # exclude self by construction, not by slicing
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    return {name: float(np.mean(lab[idx] == lab[:, None]))
            for name, lab in labels.items()}


def chance_floor(lab):
    """P(a uniformly random OTHER point shares this point's label), averaged over points."""
    _, counts = np.unique(lab, return_counts=True)
    n = counts.sum()
    return float(np.sum(counts * (counts - 1)) / (n * (n - 1)))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--layers', type=int, nargs='+', default=list(range(32)))
    p.add_argument('--entity-type', default='functional_group')
    p.add_argument('--k', type=int, default=K)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--out-name', default='label_agreement_by_layer.csv')
    args = p.parse_args()

    df = pd.read_csv(os.path.join(REPO, CSV_PATH))
    n_mol = len(df)
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for model in args.models:
        act_dir = os.path.join(REPO, BASE_ACTIVATIONS_DIR, model, args.entity_type)
        if not os.path.isdir(act_dir):
            print(f"skip {model}: no activations at {os.path.relpath(act_dir, REPO)}")
            continue
        for layer in args.layers:
            acts = load_activations(os.path.join(
                act_dir, find_layer_file(act_dir, args.entity_type, layer))).astype(np.float32)
            assert acts.shape[0] % n_mol == 0, (
                f"{model} L{layer}: {acts.shape[0]} rows not divisible by {n_mol} molecules")
            per_mol = acts.shape[0] // n_mol
            # Molecule-major, template-minor -- the row layout `generate_prompts()` in
            # extract_activations_subset.py writes. Same assumption as
            # Analysis/tsne/compute_tsne_coords.py; if that layout changes, both break.
            groups = np.array([df.iloc[i // per_mol]['functional_group']
                               for i in range(acts.shape[0])])
            labels = {
                'template_index': np.arange(acts.shape[0]) % per_mol,
                'coarse_family': np.array([COARSE_MAP[g] for g in groups]),
                'functional_group': groups,
            }
            cent = knn_agreement(acts - acts.mean(axis=0, keepdims=True), labels, args.k)
            raw = knn_agreement(acts, labels, args.k)
            row = {'model': model, 'entity_type': args.entity_type, 'layer': layer,
                   'k': args.k, 'n_points': acts.shape[0]}
            for name in LABELS:
                row[f'knn_{name}_centered'] = round(cent[name], 4)
                row[f'knn_{name}_raw'] = round(raw[name], 4)
                row[f'chance_{name}'] = round(chance_floor(labels[name]), 4)
            rows.append(row)
        print(f"{model}: {len(args.layers)} layers")

    path = os.path.join(args.out_dir, args.out_name)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {os.path.relpath(path, REPO)}  ({len(rows)} rows)")


if __name__ == '__main__':
    main()
