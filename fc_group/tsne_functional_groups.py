import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from model_registry import get_model_config

# ============================
# User Configuration Variables
# ============================

BASE_ACTIVATIONS_DIR = 'fc_group/activation_datasets_functional_groups'
MODEL_NAME = 'meta-llama/Llama-3.1-8B'  # matches extraction.model_name in fc_group/config_extract_activation.yaml

# Default entity_type whose embeddings to visualize; override per-run with --entity-type.
# Must match one of the entity_type values in fc_group/config_extract_activation.yaml,
# e.g.: functional_group, functional_group_structure, pka, pkah, tpsa,
# avg_carbon_oxidation_state, hbd, hba, boiling_point_c, water_solubility, molecule
# (and their "* question" variants).
DEFAULT_ENTITY_TYPE = 'functional_group'

CSV_PATH = 'fc_group/functional_group_dataset.csv'
LABEL_COLUMN = 'iupac_name'

FEATURES_TO_USE = [
    'functional_group', 'functional_group_structure', 'carbon_count', 'mw',
    'pka', 'pkah', 'tpsa', 'avg_carbon_oxidation_state', 'hbd', 'hba',
    'boiling_point_c', 'water_solubility',
]  # Features to visualize

# Plot Settings
POINT_SIZE = 22
ANNOTATE_FONT_SIZE = 22
COLOR_MAP = plt.cm.rainbow
PLOT_SIZE = 8  # Square figure size (inches) for each individual feature plot

# ============================
# Function Definitions
# ============================

def load_labels_and_features(csv_path, label_column, features):
    """
    Load molecule labels and the specified features from the CSV file.
    """
    df = pd.read_csv(csv_path)
    labels = df[label_column].tolist()
    features_dict = {feature: df[feature].tolist() for feature in features}
    return labels, features_dict


def get_layer_files(directory, prefix):
    """
    Get a sorted list of layer files from the specified directory.
    Sorting is based on the layer number extracted from the filename.
    Expected filename format: '{prefix}.last.{n}_templates.{layer}.pt'
    """
    pattern = re.compile(rf'{re.escape(prefix)}\.last\.\d+_templates\.layer_(\d+)\.pt')
    files = []
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            files.append((int(match.group(1)), filename))
    return [filename for _, filename in sorted(files)]


def load_activations(file_path):
    """
    Load activations from a .pt file and convert to a numpy array.
    """
    tensor = torch.load(file_path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    if isinstance(tensor, dict):
        if 'activations' in tensor:
            return tensor['activations'].cpu().numpy()
        raise KeyError("Key 'activations' not found in the tensor dictionary.")
    raise ValueError(f"Unsupported tensor type in file {file_path}")


def default_five_layers(num_layers):
    """Initial, mid-init, mid, mid-final, final layer indices."""
    return [
        0,
        num_layers // 4,
        num_layers // 2,
        (3 * num_layers) // 4,
        num_layers - 1,
    ]


def perform_pca(data, n_components=50):
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(data), pca


def perform_tsne(data, n_components=2, perplexity=30):
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, n_iter=1000)
    return tsne.fit_transform(data)


def assign_colors_to_categories(categories):
    """
    Assign a distinct color for each unique category using the 'rainbow' colormap.
    """
    unique_categories = sorted(set(categories))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_categories)))
    category_to_color = {category: colors[i] for i, category in enumerate(unique_categories)}
    category_colors = [category_to_color.get(cat, (0.5, 0.5, 0.5, 1.0)) for cat in categories]
    return category_colors, unique_categories


def plot_single_feature_tsne(tsne_data, feature, values, title, output_path):
    """
    Plot a single feature's t-SNE visualization on a square (PLOT_SIZE x PLOT_SIZE) figure.
    """
    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    ax.set_box_aspect(1)  # keep the drawn plot box square regardless of data range or colorbar
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{title}\n{feature}")
    is_categorical = any(isinstance(v, (str, bool)) for v in values)
    metric_text = ""

    if is_categorical:
        # Some columns (e.g. pkah, water_solubility) mix real category strings
        # with missing (NaN) entries; sorted(set(...)) inside
        # assign_colors_to_categories can't compare float NaN with str, so
        # normalize missing values to a placeholder string first.
        values = ['N/A' if isinstance(v, float) and pd.isnull(v) else v for v in values]
        category_colors, unique_categories = assign_colors_to_categories(values)
        ax.scatter(tsne_data[:, 0], tsne_data[:, 1], s=POINT_SIZE, c=category_colors, marker='o', alpha=0.7)

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=cat, markerfacecolor=c, markersize=8)
            for cat, c in zip(unique_categories, plt.cm.rainbow(np.linspace(0, 1, len(unique_categories))))
        ]
        ncol = 2 if len(unique_categories) > 8 else 1
        ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.7, ncol=ncol)

        try:
            labels_numeric = pd.factorize(pd.Series(values))[0]
            if len(set(labels_numeric)) > 1:
                silhouette_avg = silhouette_score(tsne_data, labels_numeric)
                metric_text = f"Silhouette Score: {silhouette_avg:.2f}"
            else:
                metric_text = "Silhouette Score: N/A"
        except Exception:
            metric_text = "Silhouette Score: N/A"

    else:
        values_array = np.array(values, dtype=float)
        mask = ~pd.isnull(values_array)
        if mask.sum() < 2:
            ax.scatter(tsne_data[:, 0], tsne_data[:, 1], s=POINT_SIZE, c='gray', marker='o', alpha=0.7)
        else:
            scatter_present = ax.scatter(
                tsne_data[mask, 0], tsne_data[mask, 1],
                s=POINT_SIZE, c=values_array[mask], cmap=COLOR_MAP,
                norm=plt.Normalize(np.nanmin(values_array[mask]), np.nanmax(values_array[mask])),
                marker='o', alpha=0.7,
            )
            ax.scatter(tsne_data[~mask, 0], tsne_data[~mask, 1], s=POINT_SIZE, c='gray', marker='x', alpha=0.7)
            cbar = fig.colorbar(scatter_present, ax=ax, orientation='horizontal', pad=0.1, fraction=0.046)
            cbar.set_label(feature, fontsize=10)

    if metric_text:
        ax.text(0.95, 0.05, metric_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    # No bbox_inches='tight': keeps the saved canvas exactly PLOT_SIZE x PLOT_SIZE
    # regardless of legend/colorbar content, so every output image is square.
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-feature square t-SNE plots from extracted activations.")
    parser.add_argument("--entity-type", "-e", default=DEFAULT_ENTITY_TYPE,
                         help="Which entity_type's embeddings to visualize (must match an entity_type in "
                              "fc_group/config_extract_activation.yaml). Default: %(default)s")
    parser.add_argument("--model-name", default=MODEL_NAME,
                         help="HF model_name whose saved activations to visualize (must match the directory "
                              "name under BASE_ACTIVATIONS_DIR used at extraction time). Default: %(default)s")
    parser.add_argument("--output-dir", default=None,
                         help="Output directory. Default: fc_group/Results/tsne_plots/{model-name}")
    parser.add_argument("--layers", default=None,
                         help="Comma-separated layer indices, e.g. 0,16,31. "
                              "Default: 5 evenly-spaced layers (initial, mid-init, mid, "
                              "mid-final, final) computed from the chosen model's num_layers "
                              "in fc_group/model_registry.py")
    return parser.parse_args()


def main():
    args = parse_args()
    entity_type = args.entity_type
    model_name = args.model_name

    selected_layers = [int(l) for l in args.layers.split(',')] if args.layers \
        else default_five_layers(get_model_config(model_name)['num_layers'])

    activations_dir = os.path.join(BASE_ACTIVATIONS_DIR, model_name.replace('/', '-'), entity_type)
    output_dir = args.output_dir or os.path.join('fc_group/Results/tsne_plots', model_name.replace('/', '-'))
    filename_prefix = entity_type

    os.makedirs(output_dir, exist_ok=True)

    labels, features_dict = load_labels_and_features(CSV_PATH, LABEL_COLUMN, FEATURES_TO_USE)
    num_symbols = len(labels)

    layer_files = get_layer_files(activations_dir, filename_prefix)
    if not layer_files:
        print(f"No layer files found in {activations_dir}.")
        return

    pattern = re.compile(rf'{re.escape(filename_prefix)}\.last\.\d+_templates\.layer_(\d+)\.pt')
    layer_files = [f for f in layer_files if int(pattern.match(f).group(1)) in selected_layers]
    if not layer_files:
        print(f"No matching layer files found for layers {selected_layers}.")
        return

    for filename in layer_files:
        layer_num = int(re.match(rf'{re.escape(filename_prefix)}\.last\.\d+_templates\.layer_(\d+)\.pt', filename).group(1))
        file_path = os.path.join(activations_dir, filename)
        print(f"Processing {filename} (layer {layer_num})")

        try:
            activations = load_activations(file_path)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        if activations.shape[0] % num_symbols != 0:
            print(f"Skipping {filename}: {activations.shape[0]} rows not divisible by {num_symbols} molecules.")
            continue
        activations_per_symbol = activations.shape[0] // num_symbols

        repeated_features = {
            feature: [value for value in values for _ in range(activations_per_symbol)]
            for feature, values in features_dict.items()
        }

        try:
            pca_result, pca = perform_pca(activations, n_components=min(50, activations.shape[0], activations.shape[1]))
            print(f"PCA explained variance ratio for layer {layer_num}: {pca.explained_variance_ratio_.sum():.2f}")
        except Exception as e:
            print(f"Error performing PCA on layer {layer_num}: {e}")
            continue

        try:
            perplexity = min(30, max(5, activations.shape[0] // 4))
            tsne_result = perform_tsne(pca_result, perplexity=perplexity)
        except Exception as e:
            print(f"Error performing t-SNE on layer {layer_num}: {e}")
            continue

        for feature, values in repeated_features.items():
            feature_dir = os.path.join(output_dir, feature)
            os.makedirs(feature_dir, exist_ok=True)
            output_path = os.path.join(feature_dir, f"{filename_prefix}_layer_{layer_num}.png")
            try:
                plot_single_feature_tsne(
                    tsne_data=tsne_result,
                    feature=feature,
                    values=values,
                    title=f"t-SNE Visualization of Layer {layer_num} Activations",
                    output_path=output_path,
                )
                print(f"Saved t-SNE plot to {output_path}")
            except Exception as e:
                print(f"Error plotting t-SNE for layer {layer_num}, feature {feature}: {e}")


if __name__ == "__main__":
    main()
