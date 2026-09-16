#!/usr/bin/env python3
"""Which taxonomy does each model's diff-vector geometry actually follow?

The probe's coarse labels come from `COARSE_MAP`, which assigns a group to a family
by the heteroatom that *names* it. That is one of at least two chemically defensible
partitions of the same 19 groups; the other sorts by reactivity class, which moves
amide to the carbonyls and splits the S=O pair off from the divalent sulfurs.

Asking which partition a model's geometry prefers turns "Chem-R's clustermap looks
different" into a number. For each model and layer this reads the between-class
cosine matrix the analogy script already wrote and reports, per partition,

    contrast = mean z-cosine(pairs inside a family) - mean z-cosine(pairs across)

z-scored within each model's own between-class distribution, so a model with
globally lower cosines (Chem-R: mean 0.403 at layer 31 against Llama's 0.449) is not
penalised for it. `delta = contrast(reactivity) - contrast(naming)` is the headline:
positive means the geometry sorts by reactivity, negative by nomenclature.

Three outputs, in increasing order of what they defend against:

  taxonomy_contrast.csv    the contrasts and delta, per model x layer x partition
  taxonomy_paired.csv      delta_A - delta_B between two models, bootstrap CI over
                           GROUPS -- the only comparison with enough power at n=19
  taxonomy_permutation.csv delta against random partitions matched in family count
                           and family sizes, so a named partition has to beat an
                           arbitrary one of the same shape
  between_class_matrix.csv the input matrices themselves, in long form. `Results/` is
                           gitignored, so without this snapshot neither these numbers nor
                           the paper's dendrogram figure could be rebuilt from a clean
                           checkout. ~120 KB for 3 models x 5 layers x 171 pairs.

Reads only `Results/functional_group_analogy/`. No activations, no model, CPU-only.
"""
import argparse
import csv
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DEFAULT_RESULTS = os.path.join(REPO, 'fc_group', 'Results', 'functional_group_analogy')
DEFAULT_OUT = os.path.join(HERE, 'data')

MODELS = [
    'meta-llama-Llama-3.1-8B',
    'phenixace-Chem-R-Faithful',
    'deepseek-ai-DeepSeek-R1-Distill-Llama-8B',
]
# The paired test needs a reference. Base vs chemistry-tuned is the contrast the
# paper is built on; --pair overrides it.
DEFAULT_PAIR = ('phenixace-Chem-R-Faithful', 'meta-llama-Llama-3.1-8B')

HALIDES = ['alkyl fluoride', 'alkyl chloride', 'alkyl bromide', 'alkyl iodide']

# Partition 1: the heteroatom that NAMES the group -- identical to the probe's
# COARSE_MAP, minus `none (alkane)`, which has no diff vector (it is the reference).
NAMING = {
    **{g: 'halide' for g in HALIDES},
    **{g: 'nitrogen' for g in ['amide', 'amine', 'imine', 'nitrile', 'nitro']},
    **{g: 'oxygen' for g in
       ['alcohol', 'aldehyde', 'ketone', 'ester', 'carboxylic acid', 'ether']},
    **{g: 'sulfur' for g in ['sulfone', 'sulfoxide', 'thioether', 'thiol']},
}

# Partition 2: reactivity class. Every reassignment against NAMING is standard
# organic-chemistry practice rather than a free parameter:
#   amide      -> carbonyl   (it is an acyl derivative; reacts as one)
#   nitro      -> C-N        (no carbonyl carbon; not an acyl derivative)
#   sulfone/sulfoxide -> S=O (oxidised sulfur, not divalent like thiol/thioether)
#   alcohol/ether/thiol/thioether -> single-heteroatom (the two isostere pairs)
REACTIVITY = {
    **{g: 'halide' for g in HALIDES},
    **{g: 'carbonyl' for g in
       ['aldehyde', 'ketone', 'ester', 'carboxylic acid', 'amide']},
    **{g: 'C-N' for g in ['amine', 'imine', 'nitrile', 'nitro']},
    **{g: 'single-heteroatom' for g in ['alcohol', 'ether', 'thiol', 'thioether']},
    **{g: 'S=O' for g in ['sulfone', 'sulfoxide']},
}

PARTITIONS = {'naming': NAMING, 'reactivity': REACTIVITY}


def load_matrix(results_dir, model, entity_type, layer):
    """The between-class cosines as (group_a, group_b) -> cosine, both directions."""
    path = os.path.join(results_dir, model, entity_type, 'data',
                        f'between_class_similarity_layer_{layer}.csv')
    sim = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            a, b, c = row['group_a'], row['group_b'], float(row['cosine_sim'])
            sim[(a, b)] = sim[(b, a)] = c
    return sim


def contrast(sim, partition, groups):
    """mean z-cosine within a family minus mean z-cosine across families.

    Both the z-scoring and the family split are computed over the *restricted*
    submatrix, so a bootstrap draw that happens to drop a whole family still yields
    a well-defined number from what remains. Returns nan when a draw leaves too
    little on either side of the split for the difference to mean anything.
    """
    groups = [g for g in groups if g in partition]
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]
             if (a, b) in sim]
    if len(pairs) < 10:
        return float('nan')
    vals = np.array([sim[p] for p in pairs])
    sd = vals.std(ddof=1)
    if sd == 0:
        return float('nan')
    z = (vals - vals.mean()) / sd
    same = np.array([partition[a] == partition[b] for a, b in pairs])
    if same.sum() < 2 or (~same).sum() < 2:
        return float('nan')
    return float(z[same].mean() - z[~same].mean())


def delta(sim, groups):
    """contrast(reactivity) - contrast(naming). Positive = sorts by reactivity."""
    return contrast(sim, REACTIVITY, groups) - contrast(sim, NAMING, groups)


def shuffled_partition(partition, groups, rng):
    """A random partition of `groups` with the same family-size profile.

    Matching the profile matters: a partition into many small families scores a
    higher within-minus-between contrast for purely combinatorial reasons, so an
    unmatched control would make REACTIVITY (5 families) look better than NAMING
    (4) before any chemistry is involved.
    """
    labels = [partition[g] for g in groups]
    rng.shuffle(labels)
    return dict(zip(groups, labels))


def layers_available(results_dir, model, entity_type):
    pattern = os.path.join(results_dir, model, entity_type, 'data',
                           'between_class_similarity_layer_*.csv')
    return sorted(int(os.path.basename(p).rsplit('_', 1)[1][:-4])
                  for p in glob.glob(pattern))


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
    p.add_argument('--pair', nargs=2, default=list(DEFAULT_PAIR),
                   metavar=('MODEL_A', 'MODEL_B'),
                   help='paired bootstrap reports delta_A - delta_B')
    p.add_argument('--n-boot', type=int, default=4000)
    p.add_argument('--n-perm', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    matrices, layers = {}, None
    for model in args.models:
        avail = layers_available(args.results_dir, model, args.entity_type)
        if not avail:
            raise SystemExit(f"no between-class matrices for {model}")
        layers = avail if layers is None else [L for L in layers if L in avail]
        matrices[model] = {L: load_matrix(args.results_dir, model,
                                          args.entity_type, L) for L in avail}
    groups = sorted(NAMING)

    # ---- 1. contrasts per model x layer x partition -----------------------------
    rows = []
    for model in args.models:
        for L in layers:
            sim = matrices[model][L]
            row = {'model': model, 'entity_type': args.entity_type, 'layer': L,
                   'n_groups': len(groups)}
            for name, part in PARTITIONS.items():
                row[f'contrast_{name}'] = round(contrast(sim, part, groups), 6)
            row['delta_reactivity_minus_naming'] = round(delta(sim, groups), 6)
            rows.append(row)
    write_csv(os.path.join(args.out_dir, 'taxonomy_contrast.csv'), rows,
              ['model', 'entity_type', 'layer', 'n_groups', 'contrast_naming',
               'contrast_reactivity', 'delta_reactivity_minus_naming'])

    # ---- 2. paired bootstrap, model A vs model B --------------------------------
    # Resampling unit is the GROUP. Per-model deltas do not clear zero at n=19; the
    # paired difference does, because both models are evaluated on the SAME drawn
    # groups and their sampling noise is largely shared.
    a, b = args.pair
    paired = []
    for L in layers:
        sim_a, sim_b = matrices[a][L], matrices[b][L]
        point = delta(sim_a, groups) - delta(sim_b, groups)
        draws = []
        for _ in range(args.n_boot):
            pick = sorted(set(rng.choice(groups, size=len(groups), replace=True)))
            d = delta(sim_a, pick) - delta(sim_b, pick)
            if not np.isnan(d):
                draws.append(d)
        draws = np.array(draws)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        paired.append({
            'model_a': a, 'model_b': b, 'entity_type': args.entity_type, 'layer': L,
            'delta_a': round(delta(sim_a, groups), 6),
            'delta_b': round(delta(sim_b, groups), 6),
            'paired_delta': round(point, 6),
            'ci_lo': round(float(lo), 6), 'ci_hi': round(float(hi), 6),
            'p_gt_zero': round(float((draws > 0).mean()), 4),
            'n_boot': len(draws), 'seed': args.seed,
        })
    write_csv(os.path.join(args.out_dir, 'taxonomy_paired.csv'), paired,
              ['model_a', 'model_b', 'entity_type', 'layer', 'delta_a', 'delta_b',
               'paired_delta', 'ci_lo', 'ci_hi', 'p_gt_zero', 'n_boot', 'seed'])

    # ---- 3. permutation control against shape-matched random partitions ---------
    # If a random partition of the same shape scores as well as a named one, the
    # contrast is not measuring taxonomy and outputs 1 and 2 mean nothing.
    perm = []
    for model in args.models:
        for L in layers:
            sim = matrices[model][L]
            for name, part in PARTITIONS.items():
                observed = contrast(sim, part, groups)
                null = np.array([
                    contrast(sim, shuffled_partition(part, groups, rng), groups)
                    for _ in range(args.n_perm)])
                null = null[~np.isnan(null)]
                perm.append({
                    'model': model, 'entity_type': args.entity_type, 'layer': L,
                    'partition': name, 'observed': round(observed, 6),
                    'null_mean': round(float(null.mean()), 6),
                    'null_sd': round(float(null.std(ddof=1)), 6),
                    'z_vs_null': round(float((observed - null.mean())
                                             / null.std(ddof=1)), 4),
                    # one-sided: the hypothesis is that a real taxonomy scores HIGHER
                    'p_perm': round(float((null >= observed).mean()), 4),
                    'n_perm': len(null), 'seed': args.seed,
                })
    write_csv(os.path.join(args.out_dir, 'taxonomy_permutation.csv'), perm,
              ['model', 'entity_type', 'layer', 'partition', 'observed', 'null_mean',
               'null_sd', 'z_vs_null', 'p_perm', 'n_perm', 'seed'])

    # ---- 4. snapshot the inputs ---------------------------------------------------
    # Emitted upper-triangle only (each unordered pair once), with both partition labels
    # attached, so a figure or a re-analysis needs nothing but this file.
    snap = []
    for model in args.models:
        for L in layers:
            sim = matrices[model][L]
            for i, a_ in enumerate(groups):
                for b_ in groups[i + 1:]:
                    if (a_, b_) not in sim:
                        continue
                    snap.append({
                        'model': model, 'entity_type': args.entity_type, 'layer': L,
                        'group_a': a_, 'group_b': b_,
                        'cosine_sim': round(sim[(a_, b_)], 6),
                        'same_naming': int(NAMING[a_] == NAMING[b_]),
                        'same_reactivity': int(REACTIVITY[a_] == REACTIVITY[b_]),
                    })
    write_csv(os.path.join(args.out_dir, 'between_class_matrix.csv'), snap,
              ['model', 'entity_type', 'layer', 'group_a', 'group_b', 'cosine_sim',
               'same_naming', 'same_reactivity'])


if __name__ == '__main__':
    main()
