"""
Tests whether Llama-3.1-8B activations encode a consistent geometric
"scaffold" for functional-group transformations, analogous to word2vec-style
analogies (king - man + woman ~= queen).

For each non-alkane molecule we define:
    diff_vec(group, C) = act(molecule with `group` at chain length C)
                        - act(alkane backbone at the same chain length C)

e.g. diff_vec(alcohol, 3) = act(propan-1-ol) - act(propane)

Within-class consistency: are the 4 diff-vectors of the same group (across
chain lengths 3-6) pointing in a similar direction?
Between-class distinctness: are different groups' diff-vectors distinguishable
from each other, and do chemically related groups (e.g. alcohol/thiol) cluster
together?
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from model_registry import default_five_layers, get_model_config

# ============================
# Config
# ============================

BASE_ACTIVATIONS_DIR = 'fc_group/activation_datasets_functional_groups'
MODEL_NAME = 'meta-llama/Llama-3.1-8B'
ENTITY_TYPE = 'functional_group'
CSV_PATH = 'fc_group/functional_group_dataset.csv'
LABEL_COLUMN = 'iupac_name'
ALKANE_LABEL = 'none (alkane)'
# Unused here -- main() now derives layers from default_five_layers() -- but
# anisotropy_diagnostic.py imports this constant from this module.
DEFAULT_LAYERS = [0, 16, 31]
DEFAULT_OUTPUT_DIR = 'fc_group/Results/functional_group_analogy'
DEFAULT_NUM_NULL_SAMPLES = 10000
HIDDEN_DIM = 4096
SEED = 42

# ============================
# Loaders (copied from fc_group/tsne_functional_groups.py)
# ============================

def load_labels_and_features(csv_path, label_column, features):
    df = pd.read_csv(csv_path)
    labels = df[label_column].tolist()
    features_dict = {feature: df[feature].tolist() for feature in features}
    return labels, features_dict


def find_layer_file(directory, entity_type, layer):
    """Locates the activation file for one (entity_type, layer) pair.

    Expected filename format: '{entity_type}.last.{prompt_name}.layer_{n}.pt'.
    prompt_name is matched as an opaque field so both the statement templates
    ('10_templates') and the question templates ('10_templates_questions')
    match -- re-encoding the writer's naming convention here is how this
    silently stopped matching the question variants once.
    """
    pattern = re.compile(rf'^{re.escape(entity_type)}\.last\.[^.]+\.layer_{layer}\.pt$')
    matches = [f for f in os.listdir(directory) if pattern.match(f)]
    assert len(matches) == 1, (
        f"Expected exactly 1 activation file for entity_type='{entity_type}' layer={layer} "
        f"in {directory}, found {len(matches)}: {matches}"
    )
    return matches[0]


def load_activations(file_path):
    tensor = torch.load(file_path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    if isinstance(tensor, dict):
        if 'activations' in tensor:
            return tensor['activations'].cpu().numpy()
        raise KeyError("Key 'activations' not found in the tensor dictionary.")
    raise ValueError(f"Unsupported tensor type in file {file_path}")


# ============================
# Core logic
# ============================

def load_and_average_templates(activations, num_molecules):
    activations = activations.astype(np.float32)
    assert activations.shape[0] % num_molecules == 0, (
        f"{activations.shape[0]} rows not divisible by {num_molecules} molecules"
    )
    templates_per_molecule = activations.shape[0] // num_molecules
    reshaped = activations.reshape(num_molecules, templates_per_molecule, -1)
    return reshaped.mean(axis=1)


def build_alkane_index(df):
    class_order = df[df['functional_group'] == df['functional_group']].drop_duplicates(
        subset='functional_group', keep='first'
    )['functional_group'].tolist()
    assert class_order[0] == ALKANE_LABEL, (
        f"Expected first functional_group class to be '{ALKANE_LABEL}', got '{class_order[0]}'"
    )
    alkane_rows = df[df['functional_group'] == ALKANE_LABEL]
    alkane_index = {int(row.carbon_count): row.Index for row in alkane_rows.itertuples()}
    assert set(alkane_index.keys()) == {3, 4, 5, 6}, (
        f"Expected alkane chain lengths {{3,4,5,6}}, got {set(alkane_index.keys())}"
    )
    return alkane_index


def build_diff_vectors(df, mol_vecs, alkane_index):
    # Some classes have multiple molecules at the same chain length (e.g.
    # alcohol merges propan-1-ol and propan-2-ol, both carbon_count=3) --
    # accumulate per (group, C) and average, rather than letting one silently
    # overwrite the other.
    diffs_raw = {}
    for row in df.itertuples():
        group = row.functional_group
        if group == ALKANE_LABEL:
            continue
        C = int(row.carbon_count)
        if C not in alkane_index:
            print(f"WARNING: no alkane backbone at carbon_count={C} for group '{group}' "
                  f"(row {row.Index}); skipping.")
            continue
        alkane_vec = mol_vecs[alkane_index[C]]
        group_vec = mol_vecs[row.Index]
        diffs_raw.setdefault(group, {}).setdefault(C, []).append(group_vec - alkane_vec)

    diffs = {
        group: {C: np.mean(vecs, axis=0) for C, vecs in by_c.items()}
        for group, by_c in diffs_raw.items()
    }

    for group, by_c in diffs.items():
        assert len(by_c) == 4, f"Group '{group}' has {len(by_c)} diff-vectors, expected 4"
    return diffs


def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One of the vectors is zero-length.")
    sim = np.dot(v1, v2) / (norm1 * norm2)
    return float(np.clip(sim, -1.0, 1.0))


def compute_within_class_matrix(diffs_for_group):
    chain_lengths = sorted(diffs_for_group.keys())
    n = len(chain_lengths)
    matrix = np.eye(n)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(diffs_for_group[chain_lengths[i]], diffs_for_group[chain_lengths[j]])
            matrix[i, j] = matrix[j, i] = sim
            pairs.append(sim)
    mean_offdiag = float(np.mean(pairs))
    return matrix, chain_lengths, mean_offdiag


def compute_mean_diff_vectors(diffs):
    return {group: np.mean(list(by_c.values()), axis=0) for group, by_c in diffs.items()}


def compute_between_class_matrix(mean_diff_by_group):
    groups = sorted(mean_diff_by_group.keys())
    n = len(groups)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(mean_diff_by_group[groups[i]], mean_diff_by_group[groups[j]])
            matrix[i, j] = matrix[j, i] = sim
    return matrix, groups


def compute_between_class_matrix_by_carbon(diffs):
    """Between-class similarity, carbon-matched: for each pair of groups the
    cosine is taken separately at each shared chain length and only then
    averaged, instead of averaging each group's diff-vectors over chain length
    first (compute_between_class_matrix). Same lumped-vs-carbon-matched split
    already made for the second-order analogy.

    Expanding cos(mean_A, mean_B) shows it mixes all 16 (C, C') pairings --
    matched and mismatched -- and weights each chain length by its diff-vector
    norm; this keeps only the 4 matched terms and weights them equally, which
    also yields a spread across chain length that the lumped version cannot.

    Returns (mean_matrix, std_matrix, sims_by_C, groups) where sims_by_C maps
    (group_a, group_b) -> {C: cosine_sim}.
    """
    groups = sorted(diffs.keys())
    n = len(groups)
    mean_matrix = np.eye(n)
    std_matrix = np.zeros((n, n))
    sims_by_C = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = groups[i], groups[j]
            # build_diff_vectors asserts 4 chain lengths per group, so this is
            # normally all of them -- intersect anyway rather than assume it.
            common_C = sorted(set(diffs[a]) & set(diffs[b]))
            if not common_C:
                print(f"WARNING: no shared carbon count between '{a}' and '{b}'; "
                      f"leaving carbon-matched similarity at 0.")
                continue
            sims = {C: cosine_similarity(diffs[a][C], diffs[b][C]) for C in common_C}
            sims_by_C[(a, b)] = sims
            values = list(sims.values())
            mean_matrix[i, j] = mean_matrix[j, i] = float(np.mean(values))
            std_matrix[i, j] = std_matrix[j, i] = float(np.std(values))
    return mean_matrix, std_matrix, sims_by_C, groups


def compute_second_order_analogy(mean_diff_by_group, pair_a, pair_b):
    """Tests a "diff of diffs" analogy, e.g. is (thioether - thiol) pointing
    the same way as (ether - alcohol)? This checks whether a substitution
    axis (here, S-for-O) is *consistent* across two different functional-group
    families, not just whether individual groups resemble each other.
    """
    a1, a2 = pair_a
    b1, b2 = pair_b
    vec_a = mean_diff_by_group[a1] - mean_diff_by_group[a2]
    vec_b = mean_diff_by_group[b1] - mean_diff_by_group[b2]
    return cosine_similarity(vec_a, vec_b)


def compute_second_order_analogy_by_carbon(diffs, pair_a, pair_b):
    """Same "diff of diffs" test as compute_second_order_analogy, but instead
    of first averaging each group's diff-vector over carbon count (which
    lumps chain lengths 3-6 together before comparing), this matches on
    carbon count: at each C present for all four groups, it compares
    (a1[C] - a2[C]) against (b1[C] - b2[C]) directly. Returns {C: cosine_sim}.
    """
    a1, a2 = pair_a
    b1, b2 = pair_b
    common_C = set(diffs[a1]) & set(diffs[a2]) & set(diffs[b1]) & set(diffs[b2])
    sims_by_C = {}
    for C in sorted(common_C):
        vec_a = diffs[a1][C] - diffs[a2][C]
        vec_b = diffs[b1][C] - diffs[b2][C]
        sims_by_C[C] = cosine_similarity(vec_a, vec_b)
    return sims_by_C


# Analogy quadruples to test: (pair_a, pair_b) where we check whether
# (pair_a[0] - pair_a[1]) tracks (pair_b[0] - pair_b[1]).
ANALOGY_QUADRUPLES = [
    # O<->S heteroatom substitution: alcohol/thiol and ether/thioether are the
    # only two clean matched pairs (sulfoxide/sulfone substitute at the
    # central atom, not a heteroatom hanging off it, so they don't fit here).
    (('thioether', 'thiol'), ('ether', 'alcohol')),
    # Halogen-column progression: does stepping F->Cl->Br->I add a consistent
    # "step" direction, like a periodic-table analogue of a one-year time-step?
    (('alkyl bromide', 'alkyl chloride'), ('alkyl iodide', 'alkyl bromide')),
    # Sulfur oxidation ladder: thioether -> sulfoxide -> sulfone is a real
    # stepwise S-oxidation series (S, S+O, S+2O); tests whether "add one more
    # S=O" is a consistent additive direction.
    (('sulfoxide', 'thioether'), ('sulfone', 'sulfoxide')),
    # A second O->N substitution pair, parallel to alcohol/amine: imine (C=NH)
    # is the same O->N swap on aldehyde's carbonyl carbon (C=O -> C=NH).
    (('imine', 'aldehyde'), ('amine', 'alcohol')),
]


def flatten_diff_vectors(diffs):
    flat = []
    for group, by_c in diffs.items():
        for C, vec in by_c.items():
            flat.append((group, C, vec))
    return flat


def compute_full_pairwise_matrix(diffs_flat):
    n = len(diffs_flat)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(diffs_flat[i][2], diffs_flat[j][2])
            matrix[i, j] = matrix[j, i] = sim
    within_sims, between_sims = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if diffs_flat[i][0] == diffs_flat[j][0]:
                within_sims.append(matrix[i, j])
            else:
                between_sims.append(matrix[i, j])
    return matrix, float(np.mean(within_sims)), float(np.mean(between_sims))


def empirical_null_cross_group(diffs_flat, num_samples, rng):
    n = len(diffs_flat)
    sims = []
    attempts = 0
    max_attempts = num_samples * 20
    while len(sims) < num_samples and attempts < max_attempts:
        i, j = rng.integers(0, n, size=2)
        attempts += 1
        if i == j or diffs_flat[i][0] == diffs_flat[j][0]:
            continue
        sims.append(cosine_similarity(diffs_flat[i][2], diffs_flat[j][2]))
    return np.array(sims)


def closed_form_null_ci(n=None, z=3.291):
    if n is None:
        n = HIDDEN_DIM
    sigma = 1.0 / np.sqrt(n)
    return -z * sigma, z * sigma


def generate_random_vectors(n, num_vectors, rng):
    vectors = rng.standard_normal((num_vectors, n))
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    return vectors


def random_vector_baseline(n, num_vectors, num_samples, rng):
    vectors = generate_random_vectors(n, num_vectors, rng)
    sims = []
    for _ in range(num_samples):
        i, j = rng.choice(num_vectors, size=2, replace=False)
        sims.append(cosine_similarity(vectors[i], vectors[j]))
    return np.array(sims)


# ============================
# Plotting
# ============================

def fit_uncentered_3d_basis(vectors_matrix):
    """SVD-based 3D projection basis, deliberately NOT mean-centered like
    standard PCA. The origin here represents "no change from the alkane
    baseline" -- a meaningful zero point, not an arbitrary centroid -- so
    centering would distort each arrow's direction from it (it wouldn't
    affect pairwise differences between arrows, but it would affect how each
    individual arrow reads relative to true zero).
    """
    _, s, vt = np.linalg.svd(vectors_matrix, full_matrices=False)
    components = vt[:3]
    explained = float((s[:3] ** 2).sum() / (s ** 2).sum())
    return components, explained


def build_diff_by_group_at_carbon(diffs, C):
    """Per-group diff-vector at a single carbon count C (no averaging over
    chain length), for groups that have data at that C. Mirrors
    compute_mean_diff_vectors but without the mean, so the quiver plot can
    show the same carbon-matched vectors used in the by-carbon analogy test.
    """
    return {group: by_c[C] for group, by_c in diffs.items() if C in by_c}


def plot_diff_vector_quiver_3d(mean_diff_by_group, layer, output_path, highlight_pair=None, carbon_count=None):
    """3D quiver plot of every group's diff-vector as an arrow from the
    origin, projected via an uncentered top-3 SVD basis. highlight_pair (one
    (pair_a, pair_b) analogy quadruple, matching an entry in
    ANALOGY_QUADRUPLES) has its 4 groups bolded and connected by black edges,
    so a second-order analogy shows up visually as two roughly parallel
    edges. One quadruple per plot -- with multiple quadruples' groups all
    bolded at once, the plot stops being readable.

    By default `mean_diff_by_group` holds each group's diff-vector averaged
    over carbon count 3-6. Pass a single-carbon dict (see
    build_diff_by_group_at_carbon) plus the matching `carbon_count` to plot
    the carbon-matched vectors instead -- the SVD basis is then fit on that
    one carbon count only, so it is not comparable across carbon_count values
    or against the mean-based plot.
    """
    groups = sorted(mean_diff_by_group.keys())
    matrix = np.stack([mean_diff_by_group[g] for g in groups])
    components, explained = fit_uncentered_3d_basis(matrix)
    projected = matrix @ components.T
    index_of = {g: i for i, g in enumerate(groups)}

    highlight_groups = set()
    if highlight_pair:
        pair_a, pair_b = highlight_pair
        highlight_groups.update(pair_a)
        highlight_groups.update(pair_b)

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(groups)))

    for (x, y, z), group, color in zip(projected, groups, colors):
        emphasized = group in highlight_groups
        ax.quiver(0, 0, 0, x, y, z, color=color, arrow_length_ratio=0.08,
                  alpha=1.0 if emphasized else 0.3, linewidth=2.2 if emphasized else 1.0)
        # A line's tip can get foreshortened into invisibility from some camera
        # angles; a marker at the tip stays identifiable regardless of angle.
        ax.scatter([x], [y], [z], color=color, s=50 if emphasized else 12,
                   alpha=1.0 if emphasized else 0.4, depthshade=False)
        ax.text(x, y, z, group, fontsize=9 if emphasized else 6,
                color=color, alpha=1.0 if emphasized else 0.45)

    if highlight_pair:
        pair_a, pair_b = highlight_pair
        for (g_to, g_from) in (pair_a, pair_b):  # pair = (g_to, g_from) -> arrow g_from to g_to
            p_from = projected[index_of[g_from]]
            p_to = projected[index_of[g_to]]
            delta = p_to - p_from
            ax.quiver(*p_from, *delta, color='black', linestyle='dashed',
                      linewidth=1.8, arrow_length_ratio=0.12)

    ax.view_init(elev=18, azim=-50)
    ax.set_xlabel('SVD-1')
    ax.set_ylabel('SVD-2')
    ax.set_zlabel('SVD-3')
    pair_label = ''
    if highlight_pair:
        pair_a, pair_b = highlight_pair
        pair_label = f'\n({pair_a[0]} - {pair_a[1]}) vs ({pair_b[0]} - {pair_b[1]})'
    title_main = f'Diff-vectors at C={carbon_count} in 3D (layer {layer})' if carbon_count is not None \
        else f'Mean diff-vectors in 3D (layer {layer})'
    ax.set_title(f'{title_main}\n'
                 f'uncentered top-3 SVD, {explained:.1%} variance explained{pair_label}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_second_order_analogy_by_carbon(by_carbon_results, layer, output_path):
    """Small multiples, one panel per analogy quadruple, showing the
    carbon-matched cosine similarity at each C as bars, with the lumped
    (carbon-averaged) similarity overlaid as a dashed reference line so the
    two views of the same analogy can be compared directly.
    by_carbon_results: list of (pair_a, pair_b, sims_by_C, lumped_sim).
    """
    n = len(by_carbon_results)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    for idx, (pair_a, pair_b, sims_by_C, lumped_sim) in enumerate(by_carbon_results):
        ax = axes[idx]
        carbons = sorted(sims_by_C.keys())
        values = [sims_by_C[C] for C in carbons]
        ax.bar([str(c) for c in carbons], values, color='#3498db')
        ax.axhline(lumped_sim, color='black', linestyle='--', linewidth=1.5,
                   label=f'lumped avg = {lumped_sim:+.2f}')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('Carbon count')
        ax.set_ylabel('Cosine sim')
        ax.set_title(f'({pair_a[0]} - {pair_a[1]}) vs ({pair_b[0]} - {pair_b[1]})', fontsize=9)
        ax.legend(fontsize=8)
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(f'Second-order analogy, carbon-matched (layer {layer})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_within_class_small_multiples(within_matrices, chain_labels_by_group, layer, output_path):
    groups = sorted(within_matrices.keys())
    ncols = 5
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for idx, group in enumerate(groups):
        ax = axes[idx]
        labels = chain_labels_by_group[group]
        sns.heatmap(within_matrices[group], ax=ax, vmin=-1, vmax=1, cmap='coolwarm',
                    annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels,
                    cbar=idx == 0, square=True)
        ax.set_title(group, fontsize=10)
    for idx in range(len(groups), len(axes)):
        axes[idx].axis('off')
    fig.suptitle(f'Within-class diff-vector cosine similarity (layer {layer})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_between_class_heatmap(matrix, groups, layer, output_path, title_suffix=''):
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(matrix, ax=ax, vmin=-1, vmax=1, cmap='coolwarm',
                xticklabels=groups, yticklabels=groups, square=True)
    suffix = f', {title_suffix}' if title_suffix else ''
    ax.set_title(f'Between-class diff-vector cosine similarity (layer {layer}{suffix})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_between_class_clustermap(matrix, groups, layer, output_path, title_suffix=''):
    distance = 1 - matrix
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method='average')
    cg = sns.clustermap(matrix, row_linkage=Z, col_linkage=Z, vmin=-1, vmax=1, cmap='coolwarm',
                         xticklabels=groups, yticklabels=groups, figsize=(9, 9))
    suffix = f', {title_suffix}' if title_suffix else ''
    cg.fig.suptitle(f'Between-class clustering (layer {layer}{suffix})', y=1.02)
    cg.savefig(output_path, dpi=150)
    plt.close(cg.fig)


def plot_between_class_by_carbon_spread(std_matrix, groups, layer, output_path):
    """Std of the carbon-matched cosine across chain lengths, per group pair --
    the diagnostic the lumped matrix cannot give: it separates group pairs
    whose relationship holds at every chain length from those carried by a
    single C.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(std_matrix, ax=ax, vmin=0, cmap='viridis',
                xticklabels=groups, yticklabels=groups, square=True)
    ax.set_title(f'Between-class similarity spread across chain length (layer {layer})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_layer_trend(summary_by_layer, output_path):
    layers = sorted(summary_by_layer.keys())
    within_means = [summary_by_layer[L]['within_mean'] for L in layers]
    within_stds = [summary_by_layer[L]['within_std'] for L in layers]
    between_means = [summary_by_layer[L]['between_mean'] for L in layers]
    lower_cf, upper_cf = closed_form_null_ci()

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(layers))
    ax.errorbar(x, within_means, yerr=within_stds, marker='o', label='Within-class (same group)',
                color='#2ecc71', capsize=4)
    ax.plot(x, between_means, marker='s', label='Between-class (different groups)', color='#e74c3c')
    # Optional: a summary.json written before the carbon-matched matrix existed
    # has no such key, so skip the series rather than crashing on an old run.
    between_by_carbon = [summary_by_layer[L].get('between_mean_by_carbon') for L in layers]
    if all(v is not None for v in between_by_carbon):
        ax.plot(x, between_by_carbon, marker='^', color='#9b59b6',
                label='Between-class (carbon-matched)')
    ax.axhspan(lower_cf, upper_cf, alpha=0.2, color='gray', label='Closed-form random-vector 99.9% CI')

    empirical_lowers = [summary_by_layer[L]['empirical_null_mean'] - summary_by_layer[L]['empirical_null_std']
                         for L in layers]
    empirical_uppers = [summary_by_layer[L]['empirical_null_mean'] + summary_by_layer[L]['empirical_null_std']
                         for L in layers]
    ax.fill_between(x, empirical_lowers, empirical_uppers, alpha=0.15, color='blue',
                     label='Empirical cross-group null (+-1 std)')

    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_xlabel(f"Layer (categorical axis - analyzed layers: {', '.join(str(L) for L in layers)})")
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Diff-vector similarity vs. layer depth')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_within_consistency_by_group_and_layer(within_by_group_layer, groups, layers, output_path):
    matrix = np.array([[within_by_group_layer[L][g] for L in layers] for g in groups])
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(matrix, ax=ax, cmap='viridis', annot=True, fmt='.2f',
                xticklabels=[str(L) for L in layers], yticklabels=groups)
    ax.set_xlabel('Layer')
    ax.set_title('Within-class consistency by group and layer')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ============================
# I/O helpers
# ============================

def save_long_format_csv(rows, columns, output_path):
    pd.DataFrame(rows, columns=columns).to_csv(output_path, index=False)


def save_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


# ============================
# Orchestration
# ============================

def run_layer(layer, df, alkane_index, entity_type, activations_dir, num_null_samples, output_dir, rng):
    print(f"\n=== Layer {layer} ===")
    file_path = os.path.join(activations_dir, find_layer_file(activations_dir, entity_type, layer))
    activations = load_activations(file_path)
    assert activations.shape[1] == HIDDEN_DIM, f"Unexpected shape {activations.shape} for layer {layer}"

    mol_vecs = load_and_average_templates(activations, num_molecules=len(df))
    assert mol_vecs.shape == (len(df), HIDDEN_DIM)

    diffs = build_diff_vectors(df, mol_vecs, alkane_index)
    groups = sorted(diffs.keys())

    within_matrices, chain_labels_by_group, within_scores = {}, {}, {}
    within_rows = []
    for group in groups:
        matrix, chain_lengths, mean_offdiag = compute_within_class_matrix(diffs[group])
        within_matrices[group] = matrix
        chain_labels_by_group[group] = chain_lengths
        within_scores[group] = mean_offdiag
        for i in range(len(chain_lengths)):
            for j in range(i + 1, len(chain_lengths)):
                within_rows.append([group, chain_lengths[i], chain_lengths[j], matrix[i, j]])
    save_long_format_csv(
        within_rows, ['group', 'chain_a', 'chain_b', 'cosine_sim'],
        os.path.join(output_dir, 'data', f'within_class_similarity_layer_{layer}.csv'),
    )

    mean_diff_by_group = compute_mean_diff_vectors(diffs)
    between_matrix, between_groups = compute_between_class_matrix(mean_diff_by_group)
    between_rows = [
        [between_groups[i], between_groups[j], between_matrix[i, j]]
        for i in range(len(between_groups)) for j in range(i + 1, len(between_groups))
    ]
    save_long_format_csv(
        between_rows, ['group_a', 'group_b', 'cosine_sim'],
        os.path.join(output_dir, 'data', f'between_class_similarity_layer_{layer}.csv'),
    )

    # Carbon-matched counterpart of the block above: cosine at each shared
    # chain length first, averaged only afterwards. Reported alongside the
    # lumped version rather than replacing it, so the two can be compared.
    bc_mean_matrix, bc_std_matrix, bc_sims_by_C, _ = compute_between_class_matrix_by_carbon(diffs)
    bc_by_carbon_rows = [
        [a, b, C, sim]
        for (a, b), sims in bc_sims_by_C.items() for C, sim in sims.items()
    ]
    save_long_format_csv(
        bc_by_carbon_rows, ['group_a', 'group_b', 'carbon_count', 'cosine_sim'],
        os.path.join(output_dir, 'data', f'between_class_similarity_by_carbon_layer_{layer}.csv'),
    )
    save_long_format_csv(
        [[between_groups[i], between_groups[j], bc_mean_matrix[i, j], bc_std_matrix[i, j]]
         for i in range(len(between_groups)) for j in range(i + 1, len(between_groups))],
        ['group_a', 'group_b', 'mean_cosine_sim', 'std_cosine_sim'],
        os.path.join(output_dir, 'data', f'between_class_similarity_by_carbon_mean_layer_{layer}.csv'),
    )

    # Primary: carbon-matched diff-of-diffs (compares a1[C]-a2[C] vs b1[C]-b2[C]
    # at each shared C, rather than lumping chain lengths 3-6 together first).
    # The lumped, carbon-averaged similarity is still computed alongside it as
    # a supplementary summary number, unchanged from before.
    analogy_rows = []
    by_carbon_rows = []
    by_carbon_plot_data = []
    for pair_a, pair_b in ANALOGY_QUADRUPLES:
        missing = [g for g in (*pair_a, *pair_b) if g not in mean_diff_by_group]
        if missing:
            print(f"Skipping analogy {pair_a} vs {pair_b}: missing group(s) {missing}")
            continue
        sim = compute_second_order_analogy(mean_diff_by_group, pair_a, pair_b)
        sims_by_C = compute_second_order_analogy_by_carbon(diffs, pair_a, pair_b)
        by_carbon_str = ', '.join(f"C{C}={s:+.4f}" for C, s in sims_by_C.items())
        print(f"Second-order analogy ({pair_a[0]} - {pair_a[1]}) vs ({pair_b[0]} - {pair_b[1]}): "
              f"carbon-matched [{by_carbon_str}]  (lumped avg = {sim:+.4f})")
        analogy_rows.append([pair_a[0], pair_a[1], pair_b[0], pair_b[1], sim])
        for C, s in sims_by_C.items():
            by_carbon_rows.append([pair_a[0], pair_a[1], pair_b[0], pair_b[1], C, s])
        by_carbon_plot_data.append((pair_a, pair_b, sims_by_C, sim))
    if analogy_rows:
        save_long_format_csv(
            analogy_rows, ['pair_a_group1', 'pair_a_group2', 'pair_b_group1', 'pair_b_group2', 'cosine_sim'],
            os.path.join(output_dir, 'data', f'second_order_analogy_lumped_layer_{layer}.csv'),
        )
    if by_carbon_rows:
        save_long_format_csv(
            by_carbon_rows,
            ['pair_a_group1', 'pair_a_group2', 'pair_b_group1', 'pair_b_group2', 'carbon_count', 'cosine_sim'],
            os.path.join(output_dir, 'data', f'second_order_analogy_by_carbon_layer_{layer}.csv'),
        )
    if by_carbon_plot_data:
        plot_second_order_analogy_by_carbon(
            by_carbon_plot_data, layer,
            os.path.join(output_dir, f'second_order_analogy_by_carbon_layer_{layer}.png'),
        )

    diffs_flat = flatten_diff_vectors(diffs)
    full_matrix, full_within_mean, full_between_mean = compute_full_pairwise_matrix(diffs_flat)
    full_rows = []
    for i in range(len(diffs_flat)):
        for j in range(i + 1, len(diffs_flat)):
            full_rows.append([
                diffs_flat[i][0], diffs_flat[i][1], diffs_flat[j][0], diffs_flat[j][1], full_matrix[i, j]
            ])
    save_long_format_csv(
        full_rows, ['group_a', 'chain_a', 'group_b', 'chain_b', 'cosine_sim'],
        os.path.join(output_dir, 'data', f'all_pairwise_diffvec_similarity_layer_{layer}.csv'),
    )

    empirical_null = empirical_null_cross_group(diffs_flat, num_null_samples, rng)
    lower_cf, upper_cf = closed_form_null_ci()

    within_mean_overall = float(np.mean(list(within_scores.values())))
    within_std_overall = float(np.std(list(within_scores.values())))

    print(f"Within-class consistency (mean over groups): {within_mean_overall:.4f} +- {within_std_overall:.4f}")
    print(f"Between-class similarity (mean diff-vecs):   {np.mean(between_matrix[np.triu_indices_from(between_matrix, k=1)]):.4f}")
    print(f"Between-class similarity (carbon-matched):    {np.mean(bc_mean_matrix[np.triu_indices_from(bc_mean_matrix, k=1)]):.4f}")
    print(f"Closed-form null 99.9% CI: [{lower_cf:.4f}, {upper_cf:.4f}]")
    print(f"Empirical cross-group null: {empirical_null.mean():.4f} +- {empirical_null.std():.4f}")

    np.savez(
        os.path.join(output_dir, 'data', f'diff_vectors_layer_{layer}.npz'),
        **{f'{group}_{C}': vec for group, C, vec in diffs_flat},
    )

    plot_within_class_small_multiples(
        within_matrices, chain_labels_by_group, layer,
        os.path.join(output_dir, f'within_class_heatmap_layer_{layer}.png'),
    )
    plot_between_class_heatmap(
        between_matrix, between_groups, layer,
        os.path.join(output_dir, f'between_class_heatmap_layer_{layer}.png'),
    )
    plot_between_class_clustermap(
        between_matrix, between_groups, layer,
        os.path.join(output_dir, f'between_class_clustermap_layer_{layer}.png'),
    )
    plot_between_class_heatmap(
        bc_mean_matrix, between_groups, layer,
        os.path.join(output_dir, f'between_class_heatmap_by_carbon_layer_{layer}.png'),
        title_suffix='carbon-matched',
    )
    plot_between_class_clustermap(
        bc_mean_matrix, between_groups, layer,
        os.path.join(output_dir, f'between_class_clustermap_by_carbon_layer_{layer}.png'),
        title_suffix='carbon-matched',
    )
    plot_between_class_by_carbon_spread(
        bc_std_matrix, between_groups, layer,
        os.path.join(output_dir, f'between_class_by_carbon_std_layer_{layer}.png'),
    )
    sims_by_C_by_quadruple = {(pair_a, pair_b): sims_by_C for pair_a, pair_b, sims_by_C, _ in by_carbon_plot_data}
    for pair_a, pair_b in ANALOGY_QUADRUPLES:
        missing = [g for g in (*pair_a, *pair_b) if g not in mean_diff_by_group]
        if missing:
            continue
        slug = '_'.join(g.replace(' ', '-') for g in (*pair_a, *pair_b))
        plot_diff_vector_quiver_3d(
            mean_diff_by_group, layer,
            os.path.join(output_dir, f'diff_vector_quiver_3d_layer_{layer}_{slug}.png'),
            highlight_pair=(pair_a, pair_b),
        )
        for C in sims_by_C_by_quadruple[(pair_a, pair_b)]:
            diff_at_C = build_diff_by_group_at_carbon(diffs, C)
            plot_diff_vector_quiver_3d(
                diff_at_C, layer,
                os.path.join(output_dir, f'diff_vector_quiver_3d_layer_{layer}_{slug}_C{C}.png'),
                highlight_pair=(pair_a, pair_b),
                carbon_count=C,
            )

    return {
        'within_scores': within_scores,
        'within_mean': within_mean_overall,
        'within_std': within_std_overall,
        'between_mean': float(np.mean(between_matrix[np.triu_indices_from(between_matrix, k=1)])),
        'between_mean_by_carbon': float(np.mean(bc_mean_matrix[np.triu_indices_from(bc_mean_matrix, k=1)])),
        'between_std_by_carbon': float(np.mean(bc_std_matrix[np.triu_indices_from(bc_std_matrix, k=1)])),
        'full_within_mean': full_within_mean,
        'full_between_mean': full_between_mean,
        'empirical_null_mean': float(empirical_null.mean()),
        'empirical_null_std': float(empirical_null.std()),
        'closed_form_ci': [lower_cf, upper_cf],
        'groups': groups,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze the geometric consistency of functional-group 'diff vectors' "
                    "(molecule with group - alkane backbone) across chain lengths and groups."
    )
    parser.add_argument('--entity-type', default=ENTITY_TYPE,
                         help="Which entity_type's activations to analyze (must match an entity_type "
                              "in fc_group/config_extract_activation.yaml). Default: %(default)s")
    parser.add_argument('--model-name', default=MODEL_NAME,
                         help="HF model_name whose saved activations to analyze (must be a key in "
                              "fc_group/model_registry.py and match the directory name under "
                              "BASE_ACTIVATIONS_DIR used at extraction time). Default: %(default)s")
    parser.add_argument('--layers', default=None,
                         help="Comma-separated layer indices, e.g. 0,16,31. "
                              "Default: 5 evenly-spaced layers (initial, mid-init, mid, "
                              "mid-final, final) computed from the chosen model's num_layers "
                              "in fc_group/model_registry.py")
    parser.add_argument('--output-dir', default=None,
                         help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}/{{model-name}}/{{entity-type}}")
    parser.add_argument('--num-null-samples', type=int, default=DEFAULT_NUM_NULL_SAMPLES,
                         help="Number of cross-group pairs to sample for the empirical null. "
                              "Default: %(default)s")
    return parser.parse_args()


def main():
    global MODEL_NAME, HIDDEN_DIM

    args = parse_args()
    entity_type = args.entity_type

    MODEL_NAME = args.model_name
    model_config = get_model_config(MODEL_NAME)
    HIDDEN_DIM = model_config['hidden_dim']

    layers = [int(l) for l in args.layers.split(',')] if args.layers \
        else default_five_layers(model_config['num_layers'])
    model_slug = MODEL_NAME.replace('/', '-')
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, model_slug, entity_type)

    activations_dir = os.path.join(BASE_ACTIVATIONS_DIR, model_slug, entity_type)
    # An entity_type that was never extracted for this model is a normal outcome
    # of sweeping the full config, not a crash -- report it the way
    # tsne_functional_groups.py does and leave no empty output tree behind.
    if not os.path.isdir(activations_dir):
        print(f"No layer files found in {activations_dir} (directory does not exist)")
        return

    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

    rng = np.random.default_rng(SEED)

    df = pd.read_csv(CSV_PATH)
    alkane_index = build_alkane_index(df)

    summary_by_layer = {}
    within_by_group_layer = {}
    for layer in layers:
        result = run_layer(layer, df, alkane_index, entity_type, activations_dir, args.num_null_samples, output_dir, rng)
        summary_by_layer[layer] = result
        within_by_group_layer[layer] = result['within_scores']

    groups = summary_by_layer[layers[0]]['groups']
    plot_layer_trend(summary_by_layer, os.path.join(output_dir, 'layer_trend.png'))
    plot_within_consistency_by_group_and_layer(
        within_by_group_layer, groups, layers,
        os.path.join(output_dir, 'within_consistency_by_group_and_layer.png'),
    )

    save_json(
        {str(L): {k: v for k, v in s.items() if k != 'groups'} for L, s in summary_by_layer.items()},
        os.path.join(output_dir, 'data', 'summary_layer_trend.json'),
    )

    # Smoke test at the deepest available layer
    deepest = max(layers)
    print(f"\n=== Smoke test (layer {deepest}) ===")
    deep_within = within_by_group_layer[deepest]
    lower_cf, upper_cf = closed_form_null_ci()
    for group, score in sorted(deep_within.items(), key=lambda kv: -kv[1]):
        flag = "OK" if abs(score) > upper_cf else "WITHIN NULL BAND"
        print(f"  {group:20s} within-class consistency = {score:+.4f}  [{flag}]")

    print(f"\nOutputs written to {output_dir}/")


if __name__ == '__main__':
    main()
