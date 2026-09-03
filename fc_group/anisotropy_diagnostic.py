"""
Diagnoses how much of the functional-group diff-vector geometry (see
functional_group_analogy_carbon_matched.py) is just transformer-hidden-state
anisotropy -- a dominant shared direction present in almost every activation,
regardless of content -- versus genuine chemistry-specific structure.

Reports:
  1. Raw-activation isotropy: pairwise cosine similarity among unrelated
     molecules' raw activations, and each activation's similarity to the
     global mean activation.
  2. Diff-vector isotropy: each diff vector's similarity to the global mean
     diff vector, and how much variance the top few uncentered singular
     vectors of the full diff-vector matrix explain.
  3. Within/between-class similarity (reusing compute_full_pairwise_matrix)
     both as originally computed and after subtracting the global mean diff
     vector from every diff vector, to see how much within>between
     separation survives once the dominant shared direction is removed.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functional_group_analogy_carbon_matched import (
    BASE_ACTIVATIONS_DIR, MODEL_NAME, CSV_PATH, LABEL_COLUMN, ALKANE_LABEL,
    DEFAULT_NUM_NULL_SAMPLES, HIDDEN_DIM, SEED,
    find_layer_file, load_activations, load_and_average_templates,
    build_alkane_index, build_diff_vectors, cosine_similarity,
    flatten_diff_vectors, compute_full_pairwise_matrix,
    empirical_null_cross_group, closed_form_null_ci,
    save_long_format_csv, save_json,
)
from model_registry import default_five_layers, get_model_config

# ============================
# Config
# ============================

# Matches the default in tsne_functional_groups.py and
# functional_group_analogy_carbon_matched.py, so all three analyses read the
# same entity_type (and the same CSV) unless told otherwise.
DEFAULT_ENTITY_TYPE = 'functional_group'
DEFAULT_OUTPUT_DIR = 'fc_group/Results/anisotropy_diagnostic'
SVD_K_VALUES = [1, 2, 3, 5, 10, 20]

# ============================
# Core logic
# ============================

def raw_activation_pairwise_sims(mol_vecs, labels):
    n = mol_vecs.shape[0]
    sims, rows = [], []
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(mol_vecs[i], mol_vecs[j])
            sims.append(sim)
            rows.append([labels[i], labels[j], sim])
    return np.array(sims), rows


def cosine_sims_to_reference(vectors, reference):
    return np.array([cosine_similarity(v, reference) for v in vectors])


def uncentered_svd_variance_ratios(matrix, k_values):
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    cumulative = np.cumsum(s ** 2) / np.sum(s ** 2)
    return {k: float(cumulative[k - 1]) for k in k_values if k <= len(s)}


def mean_center_diff_vectors(diffs_flat, global_mean):
    return [(group, C, vec - global_mean) for group, C, vec in diffs_flat]


# ============================
# Plotting
# ============================

def plot_isotropy_histograms(raw_sims, raw_to_mean_sims, diff_to_mean_sims, layer, output_path):
    lower_cf, upper_cf = closed_form_null_ci(n=HIDDEN_DIM)
    panels = [
        (raw_sims, 'Raw activation pairwise\n(unrelated molecules)'),
        (raw_to_mean_sims, 'Raw activation vs.\nglobal mean raw activation'),
        (diff_to_mean_sims, 'Diff vector vs.\nglobal mean diff vector'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (values, title) in zip(axes, panels):
        ax.hist(values, bins=30, color='#3498db', alpha=0.8)
        ax.axvline(values.mean(), color='black', linestyle='--', linewidth=1.5)
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Cosine similarity')
        ax.set_title(f'{title}\nmean={values.mean():+.3f}, std={values.std():.3f}', fontsize=9)
    axes[2].axvspan(lower_cf, upper_cf, alpha=0.2, color='gray', label='Closed-form random-vector 99.9% CI')
    axes[2].legend(fontsize=7)
    fig.suptitle(f'Anisotropy baseline (layer {layer})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_svd_variance_explained(svd_ratios_by_layer, k_values, output_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for layer in sorted(svd_ratios_by_layer.keys()):
        ratios = svd_ratios_by_layer[layer]
        ks = [k for k in k_values if k in ratios]
        ax.plot(ks, [ratios[k] for k in ks], marker='o', label=f'layer {layer}')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Top-K singular vectors (uncentered)')
    ax.set_ylabel('Cumulative variance ratio explained')
    ax.set_title('How much of the diff-vector geometry is one dominant direction?')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_original_vs_centered_layer_trend(summary_by_layer, output_path):
    layers = sorted(summary_by_layer.keys())
    x = np.arange(len(layers))
    lower_cf, upper_cf = closed_form_null_ci(n=HIDDEN_DIM)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, [summary_by_layer[L]['original']['full_within_mean'] for L in layers],
            marker='o', color='#2ecc71', label='Within-class (original)')
    ax.plot(x, [summary_by_layer[L]['original']['full_between_mean'] for L in layers],
            marker='s', color='#e74c3c', label='Between-class (original)')
    ax.plot(x, [summary_by_layer[L]['mean_centered']['full_within_mean'] for L in layers],
            marker='o', linestyle='--', color='#2ecc71', alpha=0.6, label='Within-class (mean-centered)')
    ax.plot(x, [summary_by_layer[L]['mean_centered']['full_between_mean'] for L in layers],
            marker='s', linestyle='--', color='#e74c3c', alpha=0.6, label='Between-class (mean-centered)')

    orig_null_mean = np.array([summary_by_layer[L]['original']['empirical_null_mean'] for L in layers])
    orig_null_std = np.array([summary_by_layer[L]['original']['empirical_null_std'] for L in layers])
    ax.fill_between(x, orig_null_mean - orig_null_std, orig_null_mean + orig_null_std,
                     alpha=0.15, color='blue', label='Empirical null (original, +-1 std)')

    centered_null_mean = np.array([summary_by_layer[L]['mean_centered']['empirical_null_mean'] for L in layers])
    centered_null_std = np.array([summary_by_layer[L]['mean_centered']['empirical_null_std'] for L in layers])
    ax.fill_between(x, centered_null_mean - centered_null_std, centered_null_mean + centered_null_std,
                     alpha=0.15, color='purple', label='Empirical null (mean-centered, +-1 std)')

    ax.axhspan(lower_cf, upper_cf, alpha=0.15, color='gray', label='Closed-form random-vector 99.9% CI')
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in layers])
    ax.set_xlabel('Layer (not evenly spaced)')
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Within/between-class similarity: original vs. mean-centered')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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

    labels = df[LABEL_COLUMN].tolist()

    # --- Raw-activation isotropy ---
    raw_sims, raw_pair_rows = raw_activation_pairwise_sims(mol_vecs, labels)
    global_mean_raw = mol_vecs.mean(axis=0)
    raw_to_mean_sims = cosine_sims_to_reference(mol_vecs, global_mean_raw)

    save_long_format_csv(
        raw_pair_rows, ['molecule_a', 'molecule_b', 'cosine_sim'],
        os.path.join(output_dir, 'data', f'raw_activation_pairwise_similarity_layer_{layer}.csv'),
    )
    save_long_format_csv(
        list(zip(labels, raw_to_mean_sims)), ['molecule', 'cosine_sim_to_global_mean'],
        os.path.join(output_dir, 'data', f'raw_activation_to_global_mean_layer_{layer}.csv'),
    )

    # --- Diff-vector isotropy ---
    diffs = build_diff_vectors(df, mol_vecs, alkane_index)
    diffs_flat = flatten_diff_vectors(diffs)
    diff_matrix = np.stack([vec for _, _, vec in diffs_flat])
    global_mean_diff = diff_matrix.mean(axis=0)
    diff_to_mean_sims = cosine_sims_to_reference(diff_matrix, global_mean_diff)
    svd_ratios = uncentered_svd_variance_ratios(diff_matrix, SVD_K_VALUES)

    save_long_format_csv(
        [[g, C, s] for (g, C, _), s in zip(diffs_flat, diff_to_mean_sims)],
        ['group', 'carbon_count', 'cosine_sim_to_global_mean'],
        os.path.join(output_dir, 'data', f'diff_vector_to_global_mean_layer_{layer}.csv'),
    )
    save_long_format_csv(
        [[k, v] for k, v in svd_ratios.items()], ['k', 'cumulative_variance_ratio'],
        os.path.join(output_dir, 'data', f'svd_variance_explained_layer_{layer}.csv'),
    )

    # --- Original within/between/null ---
    _, full_within_mean, full_between_mean = compute_full_pairwise_matrix(diffs_flat)
    empirical_null = empirical_null_cross_group(diffs_flat, num_null_samples, rng)
    lower_cf, upper_cf = closed_form_null_ci(n=HIDDEN_DIM)
    original = {
        'full_within_mean': full_within_mean,
        'full_between_mean': full_between_mean,
        'empirical_null_mean': float(empirical_null.mean()),
        'empirical_null_std': float(empirical_null.std()),
        'closed_form_ci': [lower_cf, upper_cf],
    }

    # --- Mean-centered within/between/null ---
    centered_flat = mean_center_diff_vectors(diffs_flat, global_mean_diff)
    _, centered_within_mean, centered_between_mean = compute_full_pairwise_matrix(centered_flat)
    centered_null = empirical_null_cross_group(centered_flat, num_null_samples, rng)
    mean_centered = {
        'full_within_mean': centered_within_mean,
        'full_between_mean': centered_between_mean,
        'empirical_null_mean': float(centered_null.mean()),
        'empirical_null_std': float(centered_null.std()),
        'closed_form_ci': [lower_cf, upper_cf],
    }

    summary = {
        'raw_isotropy': {
            'pairwise_mean': float(raw_sims.mean()),
            'pairwise_std': float(raw_sims.std()),
            'to_global_mean_mean': float(raw_to_mean_sims.mean()),
            'to_global_mean_std': float(raw_to_mean_sims.std()),
        },
        'diff_vector_isotropy': {
            'to_global_mean_mean': float(diff_to_mean_sims.mean()),
            'to_global_mean_std': float(diff_to_mean_sims.std()),
            'svd_variance_ratio': {str(k): v for k, v in svd_ratios.items()},
        },
        'original': original,
        'mean_centered': mean_centered,
    }
    save_json(summary, os.path.join(output_dir, 'data', f'summary_layer_{layer}.json'))

    print(f"Raw activation isotropy:      pairwise={raw_sims.mean():+.4f}  to-global-mean={raw_to_mean_sims.mean():+.4f}")
    print(f"Diff vector isotropy:         to-global-mean={diff_to_mean_sims.mean():+.4f}  "
          f"top-1={svd_ratios.get(1, float('nan')):.3f}  top-3={svd_ratios.get(3, float('nan')):.3f}")
    print(f"Original:      within={original['full_within_mean']:+.4f}  between={original['full_between_mean']:+.4f}  "
          f"null={original['empirical_null_mean']:+.4f}+-{original['empirical_null_std']:.4f}")
    print(f"Mean-centered: within={mean_centered['full_within_mean']:+.4f}  between={mean_centered['full_between_mean']:+.4f}  "
          f"null={mean_centered['empirical_null_mean']:+.4f}+-{mean_centered['empirical_null_std']:.4f}")

    plot_isotropy_histograms(
        raw_sims, raw_to_mean_sims, diff_to_mean_sims, layer,
        os.path.join(output_dir, f'isotropy_histograms_layer_{layer}.png'),
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose how much of the functional-group diff-vector geometry is generic "
                    "transformer-hidden-state anisotropy vs. genuine chemistry-specific structure."
    )
    parser.add_argument('--entity-type', default=DEFAULT_ENTITY_TYPE,
                         help="Which entity_type's activations to analyze. Default: %(default)s")
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
    # The model slug has to be part of the path: without it a second model's run
    # silently overwrites the first one's results for the same entity_type.
    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, model_slug, entity_type)

    activations_dir = os.path.join(BASE_ACTIVATIONS_DIR, model_slug, entity_type)
    # An entity_type that was never extracted for this model is a normal outcome
    # of sweeping the full config, not a crash -- report it the way the other
    # fc_group analyses do, before makedirs leaves an empty output tree behind.
    if not os.path.isdir(activations_dir):
        print(f"No layer files found in {activations_dir} (directory does not exist)")
        return

    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

    rng = np.random.default_rng(SEED)

    df = pd.read_csv(CSV_PATH)
    alkane_index = build_alkane_index(df)

    summary_by_layer = {}
    for layer in layers:
        summary_by_layer[layer] = run_layer(
            layer, df, alkane_index, entity_type, activations_dir, args.num_null_samples, output_dir, rng
        )

    svd_ratios_by_layer = {
        layer: {int(k): v for k, v in summary_by_layer[layer]['diff_vector_isotropy']['svd_variance_ratio'].items()}
        for layer in layers
    }
    plot_svd_variance_explained(svd_ratios_by_layer, SVD_K_VALUES, os.path.join(output_dir, 'svd_variance_explained.png'))
    plot_original_vs_centered_layer_trend(summary_by_layer, os.path.join(output_dir, 'original_vs_centered_layer_trend.png'))

    save_json(
        {str(L): s for L, s in summary_by_layer.items()},
        os.path.join(output_dir, 'data', 'summary_all_layers.json'),
    )

    print(f"\nOutputs written to {output_dir}/")


if __name__ == '__main__':
    main()
