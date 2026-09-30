#!/usr/bin/env python3
"""Snapshot t-SNE coordinates into `Analysis/tsne/data/`, so the figures need no activations.

    python fc_group/Analysis/tsne/compute_tsne_coords.py [--models ...] [--layers ...]
    -> fc_group/Analysis/tsne/data/tsne_coords.csv

`fc_group/Results/tsne_plots/` holds 50 PNGs and nothing else -- no coordinates were ever
written, so a publication figure could not be re-drawn from it at any quality. This computes
the embedding once and releases the coordinates, which is what every other `Analysis/*/data/`
tree does and what lets `make_visuals_tsne.py` promise "reads only Analysis/, never Results/".

The embedding itself is `tsne_functional_groups.py`'s, parameter for parameter -- PCA to 50
components then t-SNE at perplexity 30, both at `random_state=42` -- so these coordinates are
the same ones behind the existing PNGs, not a new method. What is NOT carried over is that
script's silhouette annotation: `Analysis/tsne/README.md` establishes that it is computed on
the 2-D coordinates rather than the activations and "does not rank plots by how well they
cluster -- sometimes it ranks them backwards". A number that cannot be quoted does not belong
on a figure, so it is neither computed nor stored here.

One row per prompt: 92 molecules x 10 templates = 920 per (model, layer). Row order is
molecule-major, template-minor -- the layout `generate_prompts()` in
`extract_activations_subset.py` writes and that `functional_group_probe.py` already relies on.

CPU-only. Reads activations; writes one CSV.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'fc_group'))

from functional_group_probe import BASE_ACTIVATIONS_DIR, COARSE_MAP, CSV_PATH  # noqa: E402
from functional_group_analogy_carbon_matched import (  # noqa: E402
    find_layer_file, load_activations,
)

DEFAULT_OUT = os.path.join(HERE, 'data')

# All five registered checkpoints, in `_common.BEHAVIOURAL` order -- the same order Tables 1
# and 3 present, so a reader moving between them never has to re-sort. A model whose
# activation tree is absent is skipped with a message rather than failing the run.
MODELS = [
    'meta-llama-Llama-3.1-8B',
    'weidawang-Chem-R-8B',
    'phenixace-Chem-R-Faithful',
    'OpenDFM-ChemDFM-v1.5-8B',
    'deepseek-ai-DeepSeek-R1-Distill-Llama-8B',
]
# The five depths the rest of the analysis reports at, and the ones Results/tsne_plots used.
LAYERS = [0, 8, 16, 24, 31]

PCA_COMPONENTS = 50
PERPLEXITY = 30
SEED = 42


def embed(activations):
    """PCA to 50 then t-SNE to 2, exactly as tsne_functional_groups.py does it.

    PCA first is not cosmetic: t-SNE on 4096 raw dimensions is both slow and dominated by
    the anisotropic mean direction that `anisotropy_diagnostic.py` measures at 0.944 pairwise
    cosine in the base model. Reducing first is what the original sweep did, so keeping it
    identical is what makes these coordinates the same embedding rather than a new one.
    """
    n_comp = min(PCA_COMPONENTS, *activations.shape)
    reduced = PCA(n_components=n_comp, random_state=SEED).fit_transform(activations)
    tsne = TSNE(n_components=2, random_state=SEED,
                perplexity=min(PERPLEXITY, max(5, activations.shape[0] // 4)),
                init='pca', learning_rate='auto')
    return reduced, tsne.fit_transform(reduced)


def neighbourhood_agreement(coords, k=10):
    """For each point, the share of its k nearest neighbours carrying the same label.

    This describes THE PICTURE, not the activations: it is computed on the 2-D coordinates,
    so it says what the panel is laid out by, and nothing about how separable the underlying
    representation is. That distinction is the one `Analysis/tsne/README.md` says the old
    silhouette annotation got wrong, so it is stated rather than left to be assumed. Cluster
    quality remains the probe's claim (`probe/data/layer_curves.csv`).

    It exists because the figure makes a visual claim -- early layers group by prompt
    template, late layers by chemistry -- that a reader should not have to take on trust.
    Released so the README can quote a number that traces back to a committed CSV.
    """
    out = []
    for (model, layer), sub in coords.groupby(['model', 'layer'], sort=True):
        xy = sub[['x', 'y']].to_numpy()
        idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(
            xy, return_distance=False)[:, 1:]          # drop self
        row = {'model': model, 'layer': int(layer), 'k': k, 'n_points': len(sub)}
        for key in ('template_index', 'coarse_family', 'functional_group'):
            lab = sub[key].to_numpy()
            row[f'knn_{key}'] = round(float(np.mean(lab[idx] == lab[:, None])), 4)
        out.append(row)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--layers', type=int, nargs='+', default=LAYERS)
    p.add_argument('--entity-type', default='functional_group')
    p.add_argument('--out-dir', default=DEFAULT_OUT)
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
            # find_layer_file returns the bare filename, as every other caller assumes.
            acts = load_activations(os.path.join(
                act_dir, find_layer_file(act_dir, args.entity_type, layer)))
            acts = acts.astype(np.float32)
            assert acts.shape[0] % n_mol == 0, (
                f"{model} L{layer}: {acts.shape[0]} rows not divisible by {n_mol} molecules")
            per_mol = acts.shape[0] // n_mol
            _, xy = embed(acts)
            # Molecule-major, template-minor: row i belongs to molecule i // per_mol.
            for i, (x, y) in enumerate(xy):
                mol = df.iloc[i // per_mol]
                group = mol['functional_group']
                rows.append({
                    'model': model, 'entity_type': args.entity_type, 'layer': layer,
                    'row': i, 'molecule': mol['iupac_name'],
                    'functional_group': group, 'coarse_family': COARSE_MAP[group],
                    'carbon_count': int(mol['carbon_count']),
                    'template_index': i % per_mol,
                    'x': round(float(x), 4), 'y': round(float(y), 4),
                })
            print(f"{model} L{layer}: {acts.shape[0]} rows, perplexity "
                  f"{min(PERPLEXITY, max(5, acts.shape[0] // 4))}")

    path = os.path.join(args.out_dir, 'tsne_coords.csv')
    coords = pd.DataFrame(rows)
    coords.to_csv(path, index=False)
    print(f"wrote {os.path.relpath(path, REPO)}  ({len(coords)} rows)")

    nb = neighbourhood_agreement(coords)
    path = os.path.join(args.out_dir, 'tsne_neighbourhood.csv')
    nb.to_csv(path, index=False)
    print(f"wrote {os.path.relpath(path, REPO)}  ({len(nb)} rows)")


if __name__ == '__main__':
    main()
