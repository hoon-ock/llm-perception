"""Layer-wise linear probe for functional-group identity in LLM hidden states.

The other fc_group scripts test the geometry of the activations without labels
(diff-vector analogy consistency, within/between-class cosine, anisotropy,
t-SNE silhouette). This one asks the supervised question directly: *is the
functional group of a molecule linearly decodable from the last-token residual
stream, and at what depth does decodability peak?*

It is the functional-group counterpart of `Direct_recall/categorical_probe.py`
(the periodic-table probe), hardened for a much thinner statistical setting --
92 molecules, 20 classes, 4096 dimensions:

* L2 logistic regression with `C` chosen by a group-aware inner CV, instead of
  that script's `SVC(kernel='linear', C=2)`, which at p=4096 / n~70 is close to
  unregularized.
* Balanced accuracy as the headline metric (chance is exactly `1/n_classes`
  regardless of class imbalance; the majority class alone is 8.7%).
* Three splits of increasing strictness -- see `iter_folds`.
* A label-permutation null shuffled at the *molecule* level, so the null
  preserves the prompt-template block structure. Shuffling the expanded rows
  directly leaks templates of the same molecule across the split and puts the
  null well above chance.

Run from the repo root:

    python fc_group/functional_group_probe.py --entity-type functional_group
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model_registry import default_five_layers, get_model_config
# Shared loaders, reused rather than reimplemented -- anisotropy_diagnostic.py
# pulls from the same module for the same reason.
from functional_group_analogy_carbon_matched import (
    find_layer_file, load_activations, save_json, save_long_format_csv,
)

# ============================
# User Configuration Variables
# ============================

BASE_ACTIVATIONS_DIR = 'fc_group/activation_datasets_functional_groups'
MODEL_NAME = 'meta-llama/Llama-3.1-8B'
CSV_PATH = 'fc_group/functional_group_dataset.csv'
EXTRACTION_CONFIG_PATH = 'fc_group/config_extract_activation.yaml'

DEFAULT_ENTITY_TYPE = 'functional_group'
DEFAULT_OUTPUT_DIR = 'fc_group/Results/functional_group_probe'

FINE_COLUMN = 'functional_group'
CARBON_COLUMN = 'carbon_count'
ALKANE_LABEL = 'none (alkane)'

# Coarse target: the defining heteroatom family. Molecule counts per family in
# functional_group_dataset.csv are hydrocarbon 4 / halide 16 / oxygen 28 /
# nitrogen 24 / sulfur 20, summing to all 92 rows.
COARSE_MAP = {
    'none (alkane)': 'hydrocarbon',
    'alkyl fluoride': 'halide',
    'alkyl chloride': 'halide',
    'alkyl bromide': 'halide',
    'alkyl iodide': 'halide',
    'alcohol': 'oxygen',
    'ether': 'oxygen',
    'aldehyde': 'oxygen',
    'ketone': 'oxygen',
    'carboxylic acid': 'oxygen',
    'ester': 'oxygen',
    'amine': 'nitrogen',
    'imine': 'nitrogen',
    'nitrile': 'nitrogen',
    'amide': 'nitrogen',
    'nitro': 'nitrogen',
    'thiol': 'sulfur',
    'thioether': 'sulfur',
    'sulfoxide': 'sulfur',
    'sulfone': 'sulfur',
}

DEFAULT_C_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
DEFAULT_NUM_NULL_SAMPLES = 50
DEFAULT_N_SPLITS = 4
INNER_CV_SPLITS = 3
SEED = 42

SPLIT_CHOICES = ('molecule', 'carbon', 'group')
DEFAULT_SPLITS = ('group',)

# target -> (column the probe is trained on, column it is scored against).
# They differ only for `fine_to_coarse`, which trains the full 20-way probe and
# then asks whether a prediction landed in the right heteroatom family. That is
# how leave-one-group-out gets to keep the fine-grained training signal: the
# held-out group never needs an output unit of its own, so withholding it is
# well posed, and "iodide was called a bromide" is a stronger result than
# "iodide landed somewhere in the halide blob".
TARGET_COLUMNS = {
    'fine': (FINE_COLUMN, FINE_COLUMN),
    'coarse': ('coarse_group', 'coarse_group'),
    'fine_to_coarse': (FINE_COLUMN, 'coarse_group'),
}
TARGET_CHOICES = tuple(TARGET_COLUMNS) + ('all',)


# ============================
# Data assembly
# ============================

def build_design_matrix(activations, df, template_mode):
    """Align the saved activation rows with per-molecule labels.

    `extract_activations_subset.generate_prompts` emits prompts molecule-major /
    template-minor, so row `i` belongs to molecule `i // n_templates`.

    `n_templates` is inferred from the row count, never from the filename: the
    `prompt_name` field baked into the filename is a config label, so a file
    called `...10_templates...` can hold a single template's worth of rows if it
    came from a `--max-templates 1` run.
    """
    num_molecules = len(df)
    n_rows = activations.shape[0]
    if n_rows % num_molecules != 0:
        raise ValueError(
            f"{n_rows} activation rows are not divisible by {num_molecules} molecules "
            f"in {CSV_PATH} -- the activation file and the CSV are out of sync."
        )
    n_templates = n_rows // num_molecules

    X = activations.astype(np.float32)
    mol_idx = np.repeat(np.arange(num_molecules), n_templates)

    if template_mode == 'average':
        X = X.reshape(num_molecules, n_templates, -1).mean(axis=1)
        mol_idx = np.arange(num_molecules)
        rows_per_molecule = 1
    else:
        rows_per_molecule = n_templates

    return X, mol_idx, n_templates, rows_per_molecule


def expand(values, mol_idx):
    """Broadcast a per-molecule array onto the activation rows."""
    return np.asarray(values)[mol_idx]


def loadable_targets(df):
    df = df.copy()
    missing = sorted(set(df[FINE_COLUMN]) - set(COARSE_MAP))
    if missing:
        raise KeyError(
            f"functional_group values with no COARSE_MAP entry: {missing}. "
            f"Add them to COARSE_MAP in {__file__}."
        )
    df['coarse_group'] = df[FINE_COLUMN].map(COARSE_MAP)
    return df


# ============================
# Splits
# ============================

def iter_folds(split, y_mol, mol_idx, df, n_splits, seed):
    """Yield (fold_name, train_row_idx, test_row_idx) for one split scheme.

    molecule -- StratifiedGroupKFold over molecules. Defaults to 4 folds, not 5:
        17 of the 20 fine classes have exactly 4 molecules, so 5 folds would
        guarantee folds containing zero instances of some classes.
    carbon   -- leave-one-carbon-count-out (C3/C4/C5/C6). Every class appears on
        both sides of every fold, so this isolates invariance to backbone length
        -- the probe analogue of functional_group_analogy_carbon_matched.py.
    group    -- leave-one-fine-group-out, coarse target only. Holding a class out
        of an n-way classifier removes its output unit, so for the fine target
        held-out accuracy would be 0 by construction and measure nothing. On the
        coarse target it is well posed: withhold `alkyl iodide` and ask whether
        its molecules are still called `halide` having only seen F/Cl/Br.
    """
    y_rows = expand(y_mol, mol_idx)
    all_rows = np.arange(len(mol_idx))

    if split == 'molecule':
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (tr, te) in enumerate(cv.split(all_rows.reshape(-1, 1), y_rows, groups=mol_idx)):
            yield f'fold{i}', tr, te

    elif split == 'carbon':
        carbon_rows = expand(df[CARBON_COLUMN].values, mol_idx)
        for c in sorted(np.unique(carbon_rows)):
            te = all_rows[carbon_rows == c]
            yield f'C{c}', all_rows[carbon_rows != c], te

    elif split == 'group':
        fine_rows = expand(df[FINE_COLUMN].values, mol_idx)
        # A fine group whose coarse family has no other member *in this dataset*
        # cannot be tested: withholding it leaves its family absent from the
        # training labels entirely.
        present = pd.unique(fine_rows)
        family_sizes = pd.Series([COARSE_MAP[g] for g in present]).value_counts()
        for g in sorted(present):
            if family_sizes[COARSE_MAP[g]] < 2:
                print(f"  skipping held-out group '{g}': sole member of coarse family "
                      f"'{COARSE_MAP[g]}', nothing left to generalize from")
                continue
            te = all_rows[fine_rows == g]
            yield g, all_rows[fine_rows != g], te

    else:
        raise ValueError(f"Unknown split: {split}")


# ============================
# Probe
# ============================

def make_preprocessor(pca_components, seed):
    """StandardScaler -> (PCA). The half of the probe that never sees a label.

    Split out from `make_estimator` so it can be fit once per fold and reused
    across every loop that varies only `y` (the permutation null) or only `C`
    (the inner grid search). Because neither transform is supervised, hoisting
    it out of those loops is exact rather than an approximation -- fit it with
    `fit_transform` on the training rows to reproduce what `Pipeline.fit` does
    internally, then `transform` the test rows.

    `pca_components=0` (the default) rotates onto the training fold's own
    principal axes without dropping any: `n_components=None` keeps
    `min(n_samples, n_features)` components, which is full rank. That is an
    *exact* reparameterization here, not a lossy feature selection -- under an
    L2 penalty the optimum lies in the span of the training data, and PCA is an
    orthonormal rotation within that span, so the penalty is preserved. It
    reproduces the raw-4096-dim solution to the digit while running ~20x faster,
    which is what makes the permutation null affordable across every layer.

    Pass a positive value for a genuinely lossy truncation, or -1 to skip PCA
    and fit in the raw hidden-state basis.
    """
    steps = [('scaler', StandardScaler())]
    if pca_components >= 0:
        n_components = pca_components if pca_components > 0 else None
        steps.append(('pca', PCA(n_components=n_components, svd_solver='full',
                                 random_state=seed)))
    return Pipeline(steps)


def make_classifier(C):
    """The supervised half: multinomial L2 logistic regression.

    `penalty` is left at its default 'l2': passing it explicitly is deprecated
    in scikit-learn >=1.8, and `multi_class` was removed in 1.7 (lbfgs is
    multinomial by default), so neither is named here -- this keeps the script
    working on both the cluster's pinned 1.2.2 and a modern local install.
    """
    return LogisticRegression(C=C, solver='lbfgs', class_weight='balanced',
                              max_iter=2000)


def make_estimator(C, pca_components, seed):
    """The whole probe as one pipeline, for the fits that happen only once."""
    pre = make_preprocessor(pca_components, seed)
    return Pipeline(list(pre.steps) + [('clf', make_classifier(C))])


def select_C(X, y, groups, c_grid, pca_components, seed):
    """Pick C by a group-aware inner CV, so templates of one molecule stay together.

    The inner folds are the outer loop and the C grid the inner one, so each
    fold's preprocessor is fit once and shared across the whole grid rather than
    refit for every C. Ties are still broken toward the smallest C, since
    `argmax` returns the first maximum and the grid order is preserved.
    """
    if len(c_grid) == 1:
        return c_grid[0]
    n_groups = len(np.unique(groups))
    n_inner = min(INNER_CV_SPLITS, n_groups)
    inner = GroupKFold(n_splits=n_inner)
    scores_by_fold = []
    for tr, te in inner.split(X, y, groups=groups):
        pre = make_preprocessor(pca_components, seed)
        Z_tr = pre.fit_transform(X[tr])
        Z_te = pre.transform(X[te])
        scores_by_fold.append([
            balanced_accuracy_score(y[te], make_classifier(C).fit(Z_tr, y[tr]).predict(Z_te))
            for C in c_grid
        ])
    return c_grid[int(np.argmax(np.mean(scores_by_fold, axis=0)))]


def run_cv(X, y, mol_idx, folds, c_grid, pca_components, seed, label_map=None):
    """Fit the probe over one fold list; returns per-fold rows and OOF predictions.

    `label_map`, when given, is an integer array indexed by training-class id that
    returns the id of the class actually being *scored* -- it is what makes the
    `fine_to_coarse` target work: the probe is trained on all 20 functional
    groups, but a prediction is graded on whether it landed in the right
    heteroatom family. Predictions and truths pass through it before every metric
    and before entering `oof_pred`, so the pooled score, the confusion matrix and
    the predictions CSV all speak in the evaluation classes.

    Note that `select_C` is deliberately left scoring the *training* labels: C is
    chosen for the task the model is actually fitting, and the inner folds do not
    hold a whole group out, so the mapped score there would be both easier and
    less discriminative.
    """
    def to_eval(v):
        return v if label_map is None else label_map[v]

    oof_pred = np.full(len(y), -1, dtype=int)
    oof_mask = np.zeros(len(y), dtype=bool)
    fold_rows = []

    for fold_name, tr, te in folds:
        C = select_C(X[tr], y[tr], mol_idx[tr], c_grid, pca_components, seed)
        est = make_estimator(C, pca_components, seed).fit(X[tr], y[tr])
        proba = est.predict_proba(X[te])
        classes = est.named_steps['clf'].classes_
        pred = to_eval(classes[proba.argmax(axis=1)])
        y_te = to_eval(y[te])
        oof_pred[te] = pred
        oof_mask[te] = True

        # Per-molecule score: sum the class probabilities over a molecule's
        # prompt templates before the argmax, so the reported number is one
        # decision per molecule rather than one per prompt. The argmax picks a
        # training class and only then maps -- "what did it actually call this
        # molecule", rather than summing probability within each family.
        mols = np.unique(mol_idx[te])
        mol_correct = []
        for m in mols:
            sel = mol_idx[te] == m
            top = classes[proba[sel].sum(axis=0).argmax()]
            mol_correct.append(to_eval(top) == y_te[sel][0])

        fold_rows.append({
            'fold': fold_name,
            'best_C': C,
            'n_train': int(len(tr)),
            'n_test': int(len(te)),
            'balanced_acc': float(balanced_accuracy_score(y_te, pred)),
            'macro_f1': float(f1_score(y_te, pred, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y_te, pred)),
            'per_molecule_acc': float(np.mean(mol_correct)),
        })

    return fold_rows, oof_pred, oof_mask


def run_null(X, y_mol, mol_idx, folds, C, pca_components, num_samples, seed,
             label_map=None):
    """Balanced accuracy under molecule-level label permutation.

    The fold partition is held fixed and only the labels are permuted -- the
    standard permutation test, and what lets each fold be scaled once and reused
    across every permutation. Labels are shuffled over the 92 *molecules* and
    re-expanded, so all templates of a molecule keep the same (wrong) label.

    `label_map` must be passed whenever the real run used one, so the null is
    scored in the same space. Omitting it for `fine_to_coarse` would compare a
    5-class observation against a 20-class null and make the result look far
    more significant than it is.
    """
    rng = np.random.default_rng(seed)
    n_mol = len(y_mol)
    perms = np.stack([rng.permutation(n_mol) for _ in range(num_samples)])

    n_rows = len(mol_idx)
    null_pred = np.full((num_samples, n_rows), -1, dtype=int)
    null_mask = np.zeros(n_rows, dtype=bool)

    for _, tr, te in folds:
        null_mask[te] = True
        # Fit the scaler and PCA once for this fold: they are unsupervised, so
        # permuting y cannot change them. Only the classifier is refit per
        # permutation, which is where all the real work is.
        pre = make_preprocessor(pca_components, seed)
        Z_tr = pre.fit_transform(X[tr])
        Z_te = pre.transform(X[te])
        for p in range(num_samples):
            y_rows = expand(y_mol[perms[p]], mol_idx)
            null_pred[p, te] = make_classifier(C).fit(Z_tr, y_rows[tr]).predict(Z_te)

    scores = []
    for p in range(num_samples):
        y_rows = expand(y_mol[perms[p]], mol_idx)
        y_eval = y_rows if label_map is None else label_map[y_rows]
        pred = null_pred[p] if label_map is None else label_map[null_pred[p]]
        scores.append(float(balanced_accuracy_score(y_eval[null_mask], pred[null_mask])))
    return scores


# ============================
# Plotting
# ============================

def plot_layer_trend(layer_scores, null_scores, surface_acc, n_classes, num_layers,
                     target, split, output_path):
    layers = sorted(layer_scores)
    values = [layer_scores[L] for L in layers]
    x = [L / num_layers for L in layers]

    plt.figure(figsize=(6, 3.5))
    sns.set_style('whitegrid')
    plt.plot(x, values, marker='o', markersize=4, linewidth=1.5, color='tab:blue',
             label='Probe (out-of-fold)')

    if null_scores:
        m, s = float(np.mean(null_scores)), float(np.std(null_scores))
        plt.axhspan(m - s, m + s, alpha=0.18, color='grey',
                    label=f'Permutation null (+-1 std, n={len(null_scores)})')
        plt.axhline(m, color='grey', linewidth=1)

    if surface_acc is not None:
        # The comparison that decides what the curve means: anything the probe
        # achieves below this line is also achievable from the prompt's spelling.
        plt.axhline(surface_acc, color='tab:orange', linestyle='-.', linewidth=1.4,
                    label=f'Surface char n-gram baseline ({surface_acc:.2f})')

    plt.axhline(1.0 / n_classes, color='red', linestyle='--', linewidth=1,
                label=f'Chance (1/{n_classes})')

    best = layers[int(np.argmax(values))]
    plt.axvline(best / num_layers, color='tab:blue', linestyle=':', linewidth=1.2,
                label=f'Best layer {best}')

    plt.xlabel('Layer depth proportion', fontsize=11)
    plt.ylabel('Balanced accuracy', fontsize=11)
    plt.title(f'{target} target, {split} split', fontsize=12)
    plt.ylim(0, 1)
    plt.legend(fontsize=7, loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion(y_true, y_pred, class_names, layer, target, split, output_path):
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    size = max(5, 0.42 * len(class_names) + 3)
    plt.figure(figsize=(size, size - 0.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 7})
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{target} / {split} / layer {layer}', fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_logo_by_group(fold_rows, layer, n_classes, output_path):
    """Per-held-out-group transfer: did an unseen functional group land in the
    right coarse family? Bar height is that fold's balanced accuracy, which for a
    single-family test set is the recall on that family."""
    rows = sorted(fold_rows, key=lambda r: r['per_molecule_acc'], reverse=True)
    names = [r['fold'] for r in rows]
    values = [r['per_molecule_acc'] for r in rows]

    plt.figure(figsize=(max(6, 0.42 * len(names) + 2), 3.6))
    sns.set_style('whitegrid')
    plt.bar(names, values, color='tab:blue')
    plt.axhline(1.0 / n_classes, color='red', linestyle='--', linewidth=1,
                label=f'Chance (1/{n_classes})')
    plt.xticks(rotation=60, ha='right', fontsize=8)
    plt.ylabel('Per-molecule accuracy', fontsize=11)
    plt.ylim(0, 1)
    plt.title(f'Leave-one-functional-group-out, coarse target, layer {layer}', fontsize=11)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================
# Surface-string baseline
# ============================

def load_templates(entity_type, config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for e in cfg['extraction']['entities']:
        if e['entity_type'] == entity_type:
            return e['templates']
    raise KeyError(f"entity_type={entity_type!r} not found in {config_path}")


def run_surface_baseline(df, entity_type, n_templates, mol_idx, y, folds, seed,
                         label_map=None):
    """The same probe on char n-grams of the raw prompt text.

    Every template embeds `{iupac_name} ({formula})`, so "Propan-1-ol
    (CH3CH2CH2OH)" hands a probe the substrings `-ol` and `OH` directly. This is
    the number that separates recalled chemistry from orthography: whatever
    accuracy appears here is available without running the model at all.

    Takes the same `label_map` as `run_cv` so the baseline is trained and scored
    in exactly the same spaces as the probe it is being compared against --
    otherwise the line drawn on a `fine_to_coarse` plot would be a 20-class
    score sitting under a 5-class curve.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    templates = load_templates(entity_type, EXTRACTION_CONFIG_PATH)[:n_templates]
    prompts = [t.format(**row.to_dict()) for _, row in df.iterrows() for t in templates]
    if len(prompts) != len(mol_idx):
        # 'average' template mode, or a template count that cannot be reproduced.
        prompts = [templates[0].format(**row.to_dict()) for _, row in df.iterrows()]
    prompts = np.array(prompts, dtype=object)

    rows = []
    for fold_name, tr, te in folds:
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2)),
            ('clf', LogisticRegression(C=1.0, solver='lbfgs', class_weight='balanced',
                                       max_iter=2000)),
        ]).fit(prompts[tr], y[tr])
        pred = pipe.predict(prompts[te])
        if label_map is not None:
            pred = label_map[pred]
        y_te = y[te] if label_map is None else label_map[y[te]]
        rows.append({
            'fold': fold_name,
            'balanced_acc': float(balanced_accuracy_score(y_te, pred)),
            'macro_f1': float(f1_score(y_te, pred, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y_te, pred)),
        })
    return rows


# ============================
# Orchestration
# ============================

def parse_args():
    p = argparse.ArgumentParser(
        description="Layer-wise linear probe for functional-group identity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--entity-type', default=DEFAULT_ENTITY_TYPE,
                   help="Entity type whose activations to probe (see config_extract_activation.yaml).")
    p.add_argument('--model-name', default=MODEL_NAME,
                   help="Must match the model used at extraction time.")
    p.add_argument('--layers', default=None,
                   help="Comma-separated layer indices. Default: every layer of the model.")
    p.add_argument('--target', choices=TARGET_CHOICES, default='all',
                   help="fine = 20 functional_group classes; coarse = 5 heteroatom families; "
                        "fine_to_coarse = trained on the 20, scored on the 5. "
                        "'all' runs the three. (Replaces the former 'both'.)")
    p.add_argument('--split', action='append', choices=SPLIT_CHOICES, default=None,
                   help="Repeatable. Default: group only -- the other two are saturated by the "
                        "molecule name in the prompt, so they are opt-in until the prompts stop "
                        "leaking it.")
    p.add_argument('--template-mode', choices=('expand', 'average'), default='expand',
                   help="expand keeps one row per prompt (grouped by molecule); "
                        "average means over a molecule's templates first.")
    p.add_argument('--n-splits', type=int, default=DEFAULT_N_SPLITS,
                   help="Folds for the 'molecule' split.")
    p.add_argument('--c-grid', default=','.join(f'{c:g}' for c in DEFAULT_C_GRID),
                   help="Comma-separated L2 inverse-regularization values for the inner CV.")
    p.add_argument('--pca-components', type=int, default=0,
                   help="0 = full-rank PCA rotation (exact and ~20x faster, see "
                        "make_estimator); a positive K truncates to K components; "
                        "-1 fits in the raw hidden-state basis.")
    p.add_argument('--num-null-samples', type=int, default=DEFAULT_NUM_NULL_SAMPLES,
                   help="Molecule-level label permutations for the null. 0 disables it.")
    p.add_argument('--null-layers', default=None,
                   help="Comma-separated layers for the null, or 'all'. Default: five layers "
                        "spread over depth -- the permutation null is essentially "
                        "layer-independent, and running it everywhere dominates the runtime.")
    p.add_argument('--surface-baseline', action='store_true',
                   help="Also fit the probe on char n-grams of the raw prompt strings.")
    p.add_argument('--output-dir', default=None)
    p.add_argument('--seed', type=int, default=SEED)
    return p.parse_args()


def silence_expected_warnings():
    """Both of these are structural, not symptoms.

    The 'group' split tests one held-out functional group at a time, so a fold's
    y_true legitimately contains a single class while the probe may predict
    others -- sklearn warns on both. Suppressed by message so genuine
    UserWarnings still surface.
    """
    warnings.filterwarnings('ignore', message="y_pred contains classes not in y_true")
    warnings.filterwarnings('ignore', message="A single label was found in")
    warnings.filterwarnings('ignore', category=ConvergenceWarning)


def main():
    args = parse_args()
    silence_expected_warnings()
    entity_type = args.entity_type
    model_config = get_model_config(args.model_name)
    num_layers = model_config['num_layers']
    model_slug = args.model_name.replace('/', '-')

    layers = ([int(l) for l in args.layers.split(',')] if args.layers
              else list(range(num_layers)))
    targets = list(TARGET_COLUMNS) if args.target == 'all' else [args.target]
    splits = args.split or list(DEFAULT_SPLITS)
    c_grid = [float(c) for c in args.c_grid.split(',')]

    if args.null_layers == 'all':
        null_layers = set(layers)
    elif args.null_layers:
        null_layers = {int(l) for l in args.null_layers.split(',')}
    else:
        null_layers = set(default_five_layers(num_layers)) & set(layers)

    activations_dir = os.path.join(BASE_ACTIVATIONS_DIR, model_slug, entity_type)
    if not os.path.isdir(activations_dir):
        print(f"No layer files found in {activations_dir} (directory does not exist)")
        return

    output_dir = args.output_dir or os.path.join(DEFAULT_OUTPUT_DIR, model_slug, entity_type)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)

    df = loadable_targets(pd.read_csv(CSV_PATH))

    # Each (target, split) pair is one experiment; layers are the outer loop so
    # every activation file is read from disk exactly once.
    jobs = []
    for target in targets:
        for split in splits:
            if split == 'group' and target == 'fine':
                print("Skipping split='group' for the fine target: holding a class out of a "
                      "20-way classifier removes its output unit, so accuracy would be 0 by "
                      "construction. It runs on the coarse target instead.")
                continue
            jobs.append((target, split))
    if not jobs:
        print("Nothing to run for the requested --target/--split combination.")
        return

    # Two label spaces per target: the one the probe is trained on and the one it
    # is scored in. They coincide except for `fine_to_coarse`, where `label_maps`
    # carries fine-class id -> coarse-class id and every metric goes through it.
    encoders, y_mol, label_maps = {}, {}, {}
    for target in targets:
        train_col, eval_col = TARGET_COLUMNS[target]
        train_le = LabelEncoder().fit(df[train_col].values)
        eval_le = LabelEncoder().fit(df[eval_col].values)
        y_mol[target] = train_le.transform(df[train_col].values)
        encoders[target] = eval_le
        if train_col == eval_col:
            label_maps[target] = None
        else:
            # Derived from the data rather than from COARSE_MAP directly, so a
            # training class that mapped to two evaluation classes would be
            # caught here instead of silently mis-scoring.
            pairs = df[[train_col, eval_col]].drop_duplicates().set_index(train_col)[eval_col]
            if not pairs.index.is_unique:
                dupes = pairs.index[pairs.index.duplicated()].unique().tolist()
                raise ValueError(
                    f"target={target!r}: {dupes} map to more than one {eval_col} value")
            label_maps[target] = eval_le.transform(pairs.loc[train_le.classes_].values)

    results = {job: {'fold_rows': [], 'oof': {}, 'null': [], 'surface': None}
               for job in jobs}
    fold_cache = {}
    # Run-once flag rather than `layer == layers[0]`: the surface baseline does
    # not depend on the layer, and the first requested layer may have no file.
    surface_done = False

    for layer in layers:
        try:
            filename = find_layer_file(activations_dir, entity_type, layer)
        except AssertionError as e:
            print(f"layer {layer}: {e}")
            continue
        activations = load_activations(os.path.join(activations_dir, filename))
        X, mol_idx, n_templates, rows_per_molecule = build_design_matrix(
            activations, df, args.template_mode)
        if layer == layers[0]:
            print(f"{entity_type}: {len(df)} molecules x {n_templates} templates -> "
                  f"X{X.shape} ({args.template_mode} mode, {rows_per_molecule} row(s)/molecule)")

        for target, split in jobs:
            key = (target, split)
            if key not in fold_cache:
                fold_cache[key] = list(iter_folds(
                    split, y_mol[target], mol_idx, df, args.n_splits, args.seed))
            folds = fold_cache[key]
            y_rows = expand(y_mol[target], mol_idx)

            label_map = label_maps[target]
            fold_rows, oof_pred, oof_mask = run_cv(
                X, y_rows, mol_idx, folds, c_grid, args.pca_components, args.seed,
                label_map=label_map)

            # oof_pred is already in the evaluation space, so the truth must be too.
            y_eval = y_rows if label_map is None else label_map[y_rows]
            pooled = float(balanced_accuracy_score(y_eval[oof_mask], oof_pred[oof_mask]))
            for r in fold_rows:
                results[key]['fold_rows'].append({'layer': layer, **r})
            results[key]['oof'][layer] = {
                'pooled_balanced_acc': pooled,
                'y_true': y_eval[oof_mask],
                'y_pred': oof_pred[oof_mask],
                'fold_rows': fold_rows,
            }
            print(f"  layer {layer:3d} | {target:6s} | {split:8s} | "
                  f"balanced_acc={pooled:.4f} | "
                  f"per_molecule={np.mean([r['per_molecule_acc'] for r in fold_rows]):.4f}")

            if args.num_null_samples and layer in null_layers:
                # The null reuses the C most often chosen on the real labels at
                # this layer, skipping the inner CV -- rerunning model selection
                # inside every permutation multiplies the cost by the grid size
                # for no change in where the null sits.
                null_C = pd.Series([r['best_C'] for r in fold_rows]).mode().iloc[0]
                scores = run_null(X, y_mol[target], mol_idx, folds, null_C,
                                  args.pca_components, args.num_null_samples, args.seed,
                                  label_map=label_map)
                results[key]['null'].extend({'layer': layer, 'perm': i, 'balanced_acc': s}
                                            for i, s in enumerate(scores))
                print(f"             null (C={null_C:g}, n={len(scores)}): "
                      f"{np.mean(scores):.4f} +- {np.std(scores):.4f}")

        if args.surface_baseline and not surface_done:
            surface_done = True
            for target, split in jobs:
                rows = run_surface_baseline(
                    df, entity_type, n_templates, mol_idx,
                    expand(y_mol[target], mol_idx), fold_cache[(target, split)], args.seed,
                    label_map=label_maps[target])
                save_long_format_csv(
                    rows, ['fold', 'balanced_acc', 'macro_f1', 'accuracy'],
                    os.path.join(output_dir, 'data', f'surface_baseline_{target}_{split}.csv'))
                results[(target, split)]['surface'] = float(
                    np.mean([r['balanced_acc'] for r in rows]))
                print(f"  surface baseline | {target:6s} | {split:8s} | "
                      f"balanced_acc={results[(target, split)]['surface']:.4f}")

    # ---- Write out ----
    for (target, split), res in results.items():
        if not res['oof']:
            continue
        tag = f'{target}_{split}'
        class_names = [str(c) for c in encoders[target].classes_]
        n_classes = len(class_names)

        save_long_format_csv(
            res['fold_rows'],
            ['layer', 'fold', 'best_C', 'n_train', 'n_test',
             'balanced_acc', 'macro_f1', 'accuracy', 'per_molecule_acc'],
            os.path.join(output_dir, 'data', f'probe_scores_{tag}.csv'))

        null_scores = [r['balanced_acc'] for r in res['null']]
        if res['null']:
            save_long_format_csv(res['null'], ['layer', 'perm', 'balanced_acc'],
                                 os.path.join(output_dir, 'data', f'null_{tag}.csv'))

        layer_scores = {L: v['pooled_balanced_acc'] for L, v in res['oof'].items()}
        best_layer = max(layer_scores, key=layer_scores.get)
        best = res['oof'][best_layer]

        pd.DataFrame({
            'y_true': [class_names[i] for i in best['y_true']],
            'y_pred': [class_names[i] for i in best['y_pred']],
        }).to_csv(os.path.join(output_dir, 'data',
                               f'predictions_{tag}_layer_{best_layer}.csv'), index=False)

        p_value = (None if not null_scores else
                   float((np.sum(np.array(null_scores) >= layer_scores[best_layer]) + 1)
                         / (len(null_scores) + 1)))
        save_json({
            'model': args.model_name,
            'entity_type': entity_type,
            'target': target,
            'split': split,
            'template_mode': args.template_mode,
            'n_classes': n_classes,
            'class_names': class_names,
            'chance': 1.0 / n_classes,
            'layers': sorted(layer_scores),
            'balanced_acc_by_layer': {str(L): layer_scores[L] for L in sorted(layer_scores)},
            'best_layer': int(best_layer),
            'best_balanced_acc': layer_scores[best_layer],
            'null_layers': sorted({r['layer'] for r in res['null']}),
            'null_mean': float(np.mean(null_scores)) if null_scores else None,
            'null_std': float(np.std(null_scores)) if null_scores else None,
            'p_value_best_layer': p_value,
            'surface_baseline_balanced_acc': res['surface'],
        }, os.path.join(output_dir, 'data', f'summary_{tag}.json'))

        plot_layer_trend(layer_scores, null_scores, res['surface'], n_classes, num_layers,
                         target, split, os.path.join(output_dir, f'layer_trend_{tag}.png'))
        plot_confusion(best['y_true'], best['y_pred'], class_names, best_layer, target, split,
                       os.path.join(output_dir, f'confusion_matrix_{tag}_layer_{best_layer}.png'))
        if split == 'group':
            plot_logo_by_group(best['fold_rows'], best_layer, n_classes,
                               os.path.join(output_dir, f'logo_by_group_{target}.png'))

        print(f"{target}/{split}: best layer {best_layer} balanced_acc="
              f"{layer_scores[best_layer]:.4f} (chance {1.0 / n_classes:.4f}"
              + (f", null {np.mean(null_scores):.4f}, p={p_value:.3f})" if null_scores else ")"))

    print(f"\nWrote results to {output_dir}")


if __name__ == '__main__':
    main()
