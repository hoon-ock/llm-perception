#!/usr/bin/env python3
"""How a probe resolves the functional groups whose family is a naming convention.

`functional_group_probe.py` scores the coarse target against `COARSE_MAP`, which
assigns each of the 20 fine groups to exactly one of 5 heteroatom families by the
atom that *names* the group. Four groups contain oxygen and are labelled otherwise:

    amide      CCC(N)=O          -> nitrogen     (carbonyl O, named for N)
    nitro      CCC[N+](=O)[O-]   -> nitrogen
    sulfoxide  CS(=O)CC          -> sulfur
    sulfone    CS(=O)(=O)CC      -> sulfur

Under leave-one-group-out these are genuinely underdetermined. Withhold `amide` and
the probe has seen amine/imine/nitrile/nitro as nitrogen and alcohol/ether/aldehyde/
ketone/acid/ester as oxygen; an amide is a carbonyl bonded to a nitrogen, and nothing
in the chemistry forces one over the other. Only the naming convention does. The
existing predictions bear this out -- every amide and nitro error in every model goes
to `oxygen` and nowhere else.

Binary accuracy cannot see any of that. This script asks the graded question instead:
how much mass does the probe put on the chemically defensible rival, and is that more
than it puts there for the group's own unambiguous family-mates?

THE CONFOUND THIS IS BUILT AROUND. Softmax sharpness is set by C, and `select_C`
picks a different C per model -- measured over the 19 folds at each model's best
layer, Chem-R-Faithful sits at 1e-4 on 15 of them against DeepSeek-70B's 1e-2. Two
orders of magnitude more shrinkage flattens *every* posterior that model emits,
ambiguous and unambiguous alike. A raw "p(nitrogen) vs p(oxygen)" comparison across
models would therefore rank models by their regularization path and read it as
chemical nuance. Three things answer that, and none of them is optional:

  1. The headline is a CONTRAST, not a level: the ambiguous group minus its own
     unambiguous controls, within one model, one layer, one calibration. A probe that
     is flat everywhere scores zero, because its controls are flat too.
  2. Probabilities are temperature-calibrated on the unambiguous groups first, so all
     models are equalized on the cases where there is no ambiguity to capture.
  3. `--mode projection` reads the same question off the geometry with no softmax in
     it at all -- no C, no temperature, nothing to attribute the result to. If the
     two modes agree the finding is solid; if they disagree, that disagreement is
     itself a statement about calibration.

TWO THINGS TO KNOW BEFORE READING THE OUTPUT.

1. TEMPERATURE CALIBRATION FAILS AT SATURATED LAYERS, and does so quietly. T is
   fitted by NLL on the unambiguous groups; where the probe classifies those nearly
   perfectly, NLL is minimized by maximal confidence, so the fit drives T *below* 1
   and sharpens rather than flattens. Measured on Llama-3.1-8B/functional_group, the
   fitted T runs 5.47 at L0, 0.72 at L16, 0.157 at L31 -- a 6x sharpening at the last
   layer, which squeezes every ambiguous group's calibrated posterior to ~0.002 and
   leaves `delta_proba` with no resolution exactly where the probe is most accurate.
   This is a real property of near-ceiling data, not a bug, and it is why
   `delta_proba_raw` (uncalibrated) and `delta_proj` are emitted next to it rather
   than left out. At layers where T is far from 1, read those two.

2. THE SULFUR CONTROLS ARE THE WEAK PART OF THE DESIGN. `thiol` (-SH) and
   `thioether` (-S-) are the direct sulfur analogues of `alcohol` (-OH) and `ether`
   (-O-), both of which the probe has seen as oxygen. So the sulfur controls are
   themselves unusually close to the oxygen family, and `sulfoxide`/`sulfone` come
   out with a NEGATIVE delta against them at every layer -- they lean toward oxygen
   *less* than their own controls do. That is a statement about the controls as much
   as about the ambiguous groups. The nitrogen side has no such problem: amine, imine
   and nitrile have no oxygen counterpart in the dataset at all, which is what makes
   the amide and nitro contrasts the cleaner half of this analysis.

Run from the repo root, after a probe run that wrote `oof_proba_{tag}.npz`:

    python fc_group/ambiguity_metric.py --model-name meta-llama/Llama-3.1-8B
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)

from functional_group_probe import (  # noqa: E402
    BASE_ACTIVATIONS_DIR, COARSE_MAP, CSV_PATH, DEFAULT_ENTITY_TYPE,
    DEFAULT_OUTPUT_DIR, FINE_COLUMN, MODEL_NAME,
)
# The shared loaders, reused rather than reimplemented -- the probe and the
# anisotropy diagnostic both pull from this module for the same reason.
from functional_group_analogy_carbon_matched import (  # noqa: E402
    find_layer_file, load_activations, load_and_average_templates,
)
from model_registry import get_model_config  # noqa: E402

DEFAULT_OUTPUT_ROOT = os.path.join(REPO, 'fc_group', 'Results', 'ambiguity_metric')

# group -> (family it is labelled, family it could defensibly be called instead).
# The rival is oxygen in all four cases, but it is written per group rather than
# assumed: a future dataset with, say, a phosphine oxide would want the same shape
# and a different pairing, and a constant named RIVAL would quietly not survive it.
AMBIGUOUS = {
    'amide':     ('nitrogen', 'oxygen'),
    'nitro':     ('nitrogen', 'oxygen'),
    'sulfoxide': ('sulfur',   'oxygen'),
    'sulfone':   ('sulfur',   'oxygen'),
}

# The unambiguous members of each ambiguous group's family: same label, no oxygen
# anywhere in the molecule. These are what the ambiguous groups are measured
# against, and choosing them badly would break the whole design -- a control that
# itself contained oxygen would absorb the very effect being measured. Asserted
# against the dataset's SMILES in check_taxonomy() rather than trusted.
CONTROLS = {
    'nitrogen': ['amine', 'imine', 'nitrile'],
    'sulfur':   ['thiol', 'thioether'],
}

# float16 storage and classes absent from a fold's training set both put exact
# zeros in the probability array; log(0) would take the temperature fit to -inf.
PROBA_FLOOR = 1e-6
N_BOOTSTRAP = 2000
SEED = 42
TOL = 1e-6

INVARIANT_HEADER = 'FAILED SELF-CHECK'


def shown(path):
    """Repo-relative where that is shorter, absolute otherwise -- a --output-dir
    outside the repo otherwise prints as a wall of '../'."""
    rel = os.path.relpath(path, REPO)
    return path if rel.startswith('..') else rel


# ============================
# Loading
# ============================

def load_proba(output_dir, tag):
    """The probe's out-of-fold probabilities, or a message saying how to make them."""
    path = os.path.join(output_dir, 'data', f'oof_proba_{tag}.npz')
    if not os.path.isfile(path):
        raise SystemExit(
            f"No {shown(path)}.\n"
            f"This file is written by functional_group_probe.py, which saves it by "
            f"default; results produced before that existed do not carry it. Rerun the "
            f"probe for this model/entity_type (on the cluster: "
            f"`sbatch fc_group/scripts/02_probe.sbatch`, unchanged) and try again.")
    with np.load(path, allow_pickle=True) as z:
        return {
            'proba': z['proba'].astype(np.float64),
            'layers': z['layers'].astype(int),
            'class_names': [str(c) for c in z['class_names']],
            'y_true': z['y_true'].astype(int),
            'molecule_index': z['molecule_index'].astype(int),
            'fine_group': np.array([str(g) for g in z['fine_group']]),
        }


def scored_groups():
    """Every group this analysis touches, ambiguous and control alike.

    The controls are scored under exactly the same protocol as the ambiguous
    groups, which is the only way the contrast between them means anything.
    """
    out = dict(AMBIGUOUS)
    for home, groups in CONTROLS.items():
        for g in groups:
            out[g] = (home, 'oxygen')
    return out


# ============================
# Probability mode
# ============================

def fit_temperature(proba, y_true, mask):
    """One scalar T per layer, fitted by NLL on the UNAMBIGUOUS rows only.

    Why this is exact rather than an approximation on saved probabilities: the
    probe's logits are recoverable from p only up to a per-row additive constant,
    and softmax cancels exactly that, so softmax(log p / T) is the genuinely
    tempered distribution and not a stand-in for one.

    Fitted on the unambiguous groups because they are where the models are supposed
    to agree -- equalizing sharpness there is what makes a difference on the
    ambiguous groups attributable to something other than the regularization path.
    """
    logp = np.log(np.clip(proba[mask], PROBA_FLOOR, None))
    y = y_true[mask]
    rows = np.arange(len(y))

    def nll(log_t):
        z = logp / np.exp(log_t)
        z -= z.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[rows, y]))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method='bounded')
    return float(np.exp(res.x)), nll(np.log(1.0)), float(res.fun)


def apply_temperature(proba, temperature):
    z = np.log(np.clip(proba, PROBA_FLOOR, None)) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def restricted_posterior(proba, class_index, molecule_index, rows, home, rival):
    """p(rival) / (p(rival) + p(home)) per MOLECULE, templates pooled first.

    Templates are summed before the ratio, matching how `per_molecule_acc` already
    pools them in the probe -- one decision per molecule, not one per prompt.

    The quantity depends only on the difference of two logits, so anything that
    shifts a row's logits uniformly leaves it untouched. That is a useful
    robustness property but NOT a substitute for the calibration above, which
    corrects a scaling rather than a shift.
    """
    out = {}
    for m in np.unique(molecule_index[rows]):
        sel = rows[molecule_index[rows] == m]
        s = proba[sel].sum(axis=0)
        denom = s[class_index[rival]] + s[class_index[home]]
        out[int(m)] = float(s[class_index[rival]] / denom) if denom > 0 else np.nan
    return out


def contrast(values_group, values_control, rng, n_boot=N_BOOTSTRAP):
    """Delta = mean(ambiguous) - mean(controls), with a molecule-level CI and p.

    Resampling is over MOLECULES, never over prompt rows. The effective sample here
    is 4 molecules against 8-12, not 40 rows against 80-120: a molecule's templates
    are near-copies of each other, and bootstrapping them as if independent would
    report an interval several times too tight. The repo's existing permutation
    null shuffles at the molecule level for the same reason.
    """
    a, b = np.asarray(values_group, float), np.asarray(values_control, float)
    if len(a) == 0 or len(b) == 0:
        return dict(delta=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_value=np.nan)
    delta = float(a.mean() - b.mean())

    boot = [float(rng.choice(a, len(a), replace=True).mean()
                  - rng.choice(b, len(b), replace=True).mean())
            for _ in range(n_boot)]

    # One-sided, because the hypothesis is directional: an ambiguous group should
    # lean toward its rival MORE than its unambiguous family-mates do. A two-sided
    # test would spend half its power on an outcome that would falsify the premise
    # rather than support it.
    pooled = np.concatenate([a, b])
    perm = []
    for _ in range(n_boot):
        s = rng.permutation(pooled)
        perm.append(float(s[:len(a)].mean() - s[len(a):].mean()))
    p = (np.sum(np.asarray(perm) >= delta) + 1) / (n_boot + 1)

    return dict(delta=delta,
                ci_lo=float(np.percentile(boot, 2.5)),
                ci_hi=float(np.percentile(boot, 97.5)),
                p_value=float(p))


def family_probability_rows(data, df, model, entity_type, num_layers):
    """Mean probability on each family, per group, per layer. The default view.

    This is the whole analysis in one table, and it is deliberately the plainest
    thing that answers the question: of the mass the probe assigned, how much went
    to each family? No ratio, no renormalization to two classes, no temperature.
    Measured on Llama-3.1-8B at layer 31, amide puts 0.173 on oxygen against
    amine's 0.041 and imine's 0.017 -- a result that needs no explaining.

    The derived quantities (restricted posterior, control contrast, centroid
    projection) are what you reach for when a reviewer asks whether the effect
    survives the fact that `select_C` regularized each model differently. They
    answer that, and they are unreadable. They live behind `--full` for both
    reasons.

    Probabilities are pooled over a molecule's templates first and then averaged
    over molecules, so a group with 8 molecules and one with 4 each contribute one
    number per molecule rather than one per prompt.
    """
    groups = scored_groups()
    rows = []
    for li, layer in enumerate(data['layers']):
        P = data['proba'][li]
        for group, (home, rival) in groups.items():
            sel = np.where(data['fine_group'] == group)[0]
            if len(sel) == 0:
                continue
            per_mol = []
            for m in np.unique(data['molecule_index'][sel]):
                r = sel[data['molecule_index'][sel] == m]
                per_mol.append(P[r].sum(axis=0) / len(r))
            mean = np.mean(per_mol, axis=0)
            rows.append({
                'model': model, 'entity_type': entity_type, 'layer': int(layer),
                'depth': round(float(layer) / (num_layers - 1), 4),
                'group': group,
                'role': 'ambiguous' if group in AMBIGUOUS else 'control',
                'home_family': home, 'rival_family': rival,
                'n_molecules': len(per_mol),
                **{f'p_{n}': float(v) for n, v in zip(data['class_names'], mean)},
            })
    return rows


def display_order():
    """Ambiguous groups first, then the controls, grouped by family.

    Fixed rather than sorted: the point of the table is to read an ambiguous group
    against its own controls, and alphabetical order interleaves them.
    """
    order = [(g, 'ambiguous') for g in sorted(AMBIGUOUS)]
    for home in sorted(CONTROLS):
        order += [(g, f'control, {home}') for g in CONTROLS[home]]
    return order


# ============================
# Projection mode
# ============================

def projection_scores(mol_vecs, fine, families, group, home, rival):
    """Where `group`'s molecules sit on the home->rival centroid axis.

    t = 0 at the home centroid, 1 at the rival centroid, so it is read on the same
    scale as the restricted posterior -- and, unlike it, with no softmax, no C and
    no temperature anywhere in the derivation. This is the reading that cannot be
    attributed to how hard the probe was regularized.

    Unlike the posterior, t is NOT bounded to [0, 1]: a molecule can sit past either
    centroid or off the axis entirely, and at layer 0, where the family centroids
    barely mean anything, it routinely does (values around -3 are normal there).
    Large-magnitude t at shallow layers is a sign the axis is meaningless, not a
    strong result.

    `group` IS HELD OUT of both centroids and out of the standardization, which
    reproduces the leave-one-group-out fold the probe itself ran. The controls get
    the identical treatment when they are scored: computing a control's position
    against a home centroid that included the control would pull it toward 0 by
    construction and inflate every contrast it appears in. That bias is the whole
    reason this function takes `group` rather than scoring every molecule at once.
    """
    held = fine == group
    train = ~held

    # Standardized on the training molecules only, matching the StandardScaler the
    # probe fits per fold, so the axis is not dominated by whichever hidden
    # dimensions happen to have the largest scale.
    mu = mol_vecs[train].mean(axis=0)
    sd = mol_vecs[train].std(axis=0)
    sd[sd == 0] = 1.0
    Z = (mol_vecs - mu) / sd

    c_home = Z[train & (families == home)].mean(axis=0)
    c_rival = Z[train & (families == rival)].mean(axis=0)
    axis = c_rival - c_home
    denom = float(axis @ axis)
    if denom <= 0:
        return None, 0.0
    return (Z[held] - c_home) @ axis / denom, float(np.linalg.norm(axis))


# ============================
# Self-checks
# ============================

def check_taxonomy(df):
    """The control groups must contain no oxygen and the ambiguous ones must.

    Asserted from the dataset rather than trusted, so that editing
    functional_group_dataset.csv -- adding a group, changing a SMILES -- cannot
    silently turn a control into something that carries the effect being measured.
    """
    problems = []
    smiles = df.groupby(FINE_COLUMN)['smiles'].apply(list)
    for g, (home, rival) in AMBIGUOUS.items():
        if g not in smiles:
            problems.append(f"ambiguous group '{g}' is not in {CSV_PATH}")
            continue
        if not all('O' in s for s in smiles[g]):
            problems.append(f"ambiguous group '{g}' has molecules with no oxygen: "
                            f"{[s for s in smiles[g] if 'O' not in s]}")
        if COARSE_MAP.get(g) != home:
            problems.append(f"'{g}' is mapped to {COARSE_MAP.get(g)!r}, not {home!r} "
                            f"-- AMBIGUOUS and COARSE_MAP disagree")
    for home, groups in CONTROLS.items():
        for g in groups:
            if g not in smiles:
                problems.append(f"control group '{g}' is not in {CSV_PATH}")
                continue
            bad = [s for s in smiles[g] if 'O' in s]
            if bad:
                problems.append(f"control group '{g}' contains oxygen ({bad}) -- it "
                                f"cannot serve as an unambiguous control for {home}")
            if COARSE_MAP.get(g) != home:
                problems.append(f"control '{g}' is in family {COARSE_MAP.get(g)!r}, "
                                f"not {home!r}")
    return problems


def check_proba(data, df, output_dir, tag):
    """The saved array must be a probability table over the rows it claims."""
    problems = []
    P, names = data['proba'], data['class_names']

    sums = P.sum(axis=2)
    if not np.allclose(sums, 1.0, atol=1e-3):
        problems.append(f"probability rows sum to [{sums.min():.5f}, {sums.max():.5f}], "
                        f"not 1 -- the array is not a distribution over {len(names)} classes")

    fine_from_csv = df[FINE_COLUMN].values[data['molecule_index']]
    if not np.array_equal(fine_from_csv, data['fine_group']):
        n = int(np.sum(fine_from_csv != data['fine_group']))
        problems.append(f"{n} row(s) carry a fine_group that disagrees with "
                        f"{CSV_PATH} at their molecule_index -- the rows are misaligned")

    # The check that validates the new array against output the untouched code path
    # produced. Only for `coarse`: `fine_to_coarse` argmaxes in the 20-class space and
    # only then maps to a family, while these probabilities are family mass, so the
    # two legitimately disagree (~8% of rows, measured). Announced rather than
    # silently skipped -- "checked and passed" and "not applicable" must not look alike.
    if tag.startswith('fine_to_coarse'):
        print("  proba/prediction agreement check skipped: fine_to_coarse predicts by "
              "argmax in the fine space, then maps, so it does not equal the argmax of "
              "the summed family mass by construction")
        return problems

    for layer in data['layers']:
        csv = os.path.join(output_dir, 'data', f'predictions_{tag}_layer_{layer}.csv')
        if not os.path.isfile(csv):
            continue
        best = int(layer)
        pred = pd.read_csv(csv)['y_pred'].values
        li = int(np.where(data['layers'] == best)[0][0])  # noqa: E501
        got = np.array([names[i] for i in P[li].argmax(axis=1)])
        if len(pred) != len(got):
            problems.append(f"L{best}: predictions CSV has {len(pred)} rows, proba has "
                            f"{len(got)}")
        elif not np.array_equal(pred, got):
            n = int(np.sum(pred != got))
            problems.append(
                f"L{best}: argmax of the saved probabilities disagrees with "
                f"predictions_{tag}_layer_{best}.csv on {n}/{len(got)} rows -- the "
                f"probabilities do not belong to the predictions they were saved with")
    return problems


def raise_if(problems, what):
    if problems:
        raise SystemExit(f"{INVARIANT_HEADER} ({what}):\n"
                         + "\n".join(f"  {p}" for p in problems))


# ============================
# Collection
# ============================

def collect(data, df, model, entity_type, num_layers, mode, activations_dir, rng):
    groups = scored_groups()
    fine_all = df[FINE_COLUMN].values
    families_all = np.array([COARSE_MAP[g] for g in fine_all])
    class_index = {n: i for i, n in enumerate(data['class_names'])}
    fine_rows = data['fine_group']
    unambiguous = ~np.isin(fine_rows, list(AMBIGUOUS))

    mol_rows, group_rows, contrast_rows = [], [], []
    problems = []

    for li, layer in enumerate(data['layers']):
        common = {'model': model, 'entity_type': entity_type, 'layer': int(layer),
                  'depth': round(float(layer) / (num_layers - 1), 4)}
        per_group_r, per_group_raw, per_group_t = {}, {}, {}

        if mode in ('proba', 'both'):
            P = data['proba'][li]
            temperature, nll_before, nll_after = fit_temperature(
                P, data['y_true'], unambiguous)
            if not temperature > 0:
                problems.append(f"L{layer}: fitted temperature {temperature} is not positive")
            if nll_after > nll_before + TOL:
                problems.append(f"L{layer}: temperature fit raised NLL "
                                f"{nll_before:.5f} -> {nll_after:.5f}")
            # A warning, not a failure -- it is a true property of the data, and
            # the run should still produce numbers. T well below 1 means the
            # controls are classified so nearly perfectly that NLL is minimized by
            # maximal confidence, so calibration SHARPENS and squeezes the
            # ambiguous groups' posteriors toward 0. Read delta_proba_raw and the
            # projection mode at those layers; see the module docstring.
            if not 0.25 <= temperature <= 4.0:
                print(f"  note: L{layer} fitted temperature {temperature:.3f} -- "
                      f"{'sharpening' if temperature < 1 else 'flattening'} the "
                      f"posteriors {1 / temperature if temperature < 1 else temperature:.1f}x. "
                      f"The controls are near-saturated at this layer, so the "
                      f"calibrated posterior loses resolution; prefer delta_proj "
                      f"and delta_proba_raw here.")
            P_cal = apply_temperature(P, temperature)
        else:
            temperature = np.nan
            P, P_cal = None, None

        if mode in ('projection', 'both'):
            filename = find_layer_file(activations_dir, entity_type, int(layer))
            mol_vecs = load_and_average_templates(
                load_activations(os.path.join(activations_dir, filename)), len(df))
        else:
            mol_vecs = None

        for group, (home, rival) in sorted(groups.items()):
            rows = np.where(fine_rows == group)[0]
            if len(rows) == 0:
                problems.append(f"L{layer}: no out-of-fold rows for group '{group}'")
                continue
            mols = np.unique(data['molecule_index'][rows])

            r_raw = r_cal = {}
            if P is not None:
                r_raw = restricted_posterior(P, class_index, data['molecule_index'],
                                             rows, home, rival)
                r_cal = restricted_posterior(P_cal, class_index, data['molecule_index'],
                                             rows, home, rival)

            t_proj = {}
            if mol_vecs is not None:
                scores, axis_norm = projection_scores(
                    mol_vecs, fine_all, families_all, group, home, rival)
                if scores is None:
                    problems.append(f"L{layer}: the {home}->{rival} centroid axis is "
                                    f"degenerate for held-out '{group}'")
                elif not np.all(np.isfinite(scores)):
                    problems.append(f"L{layer}/{group}: non-finite projection score")
                else:
                    t_proj = dict(zip((int(m) for m in np.where(fine_all == group)[0]),
                                      (float(v) for v in scores)))

            for m in mols:
                mol_rows.append({
                    **common, 'group': group,
                    'role': 'ambiguous' if group in AMBIGUOUS else 'control',
                    'home_family': home, 'rival_family': rival,
                    'molecule_index': int(m),
                    'iupac_name': df['iupac_name'].values[int(m)],
                    'r_raw': r_raw.get(int(m), np.nan),
                    'r_cal': r_cal.get(int(m), np.nan),
                    't_proj': t_proj.get(int(m), np.nan),
                })

            per_group_r[group] = [r_cal.get(int(m), np.nan) for m in mols]
            per_group_raw[group] = [r_raw.get(int(m), np.nan) for m in mols]
            per_group_t[group] = [t_proj.get(int(m), np.nan) for m in mols]
            group_rows.append({
                **common, 'group': group,
                'role': 'ambiguous' if group in AMBIGUOUS else 'control',
                'home_family': home, 'rival_family': rival,
                'n_molecules': len(mols), 'temperature': temperature,
                'mean_r_raw': float(np.nanmean([r_raw.get(int(m), np.nan) for m in mols]))
                if r_raw else np.nan,
                'mean_r_cal': float(np.nanmean(per_group_r[group])) if r_cal else np.nan,
                'mean_t_proj': float(np.nanmean(per_group_t[group])) if t_proj else np.nan,
            })

        # The headline: each ambiguous group against the pooled controls of its own
        # family, at this layer, under this calibration.
        for group, (home, rival) in sorted(AMBIGUOUS.items()):
            ctrl = [g for g in CONTROLS[home]]
            row = {**common, 'group': group, 'home_family': home,
                   'rival_family': rival, 'controls': '|'.join(ctrl),
                   'temperature': temperature}
            for field, table in (('proba', per_group_r),
                                 ('proba_raw', per_group_raw),
                                 ('proj', per_group_t)):
                vals = table.get(group, [])
                cvals = [v for g in ctrl for v in table.get(g, [])]
                vals = [v for v in vals if np.isfinite(v)]
                cvals = [v for v in cvals if np.isfinite(v)]
                stats = contrast(vals, cvals, rng)
                row[f'delta_{field}'] = stats['delta']
                row[f'delta_{field}_ci_lo'] = stats['ci_lo']
                row[f'delta_{field}_ci_hi'] = stats['ci_hi']
                row[f'delta_{field}_p'] = stats['p_value']
                row[f'mean_{field}_group'] = float(np.mean(vals)) if vals else np.nan
                row[f'mean_{field}_control'] = float(np.mean(cvals)) if cvals else np.nan
                row[f'n_{field}_control'] = len(cvals)
            contrast_rows.append(row)

    raise_if(problems, 'metric computation')
    return mol_rows, group_rows, contrast_rows


# ============================
# Plots
# ============================

# Colour per ambiguous group, stable across both figures so they can be read side
# by side. Distinguished by colour and marker, never by dash pattern -- at these
# legend sizes a dashed swatch is indistinguishable from a solid one.
GROUP_STYLE = {
    'amide':     ('#c0392b', 'o'),
    'nitro':     ('#e67e22', 's'),
    'sulfoxide': ('#2980b9', '^'),
    'sulfone':   ('#16a085', 'D'),
}


def plot_family_probability(rows, layer, model, entity_type, class_names, path):
    """The default figure: groups x families, annotated with the probabilities.

    A heatmap rather than grouped bars because the interesting comparison runs
    down a column (amide's oxygen against amine's oxygen), and bars put those
    far apart. The ambiguous rows are separated from the controls by a rule so
    the two blocks are distinguishable without reading the labels.
    """
    df = pd.DataFrame([r for r in rows if r['layer'] == layer])
    if df.empty:
        return
    order = [(g, tag) for g, tag in display_order() if g in set(df.group)]
    cols = [f'p_{n}' for n in class_names]
    M = np.array([[df[df.group == g][c].mean() for c in cols] for g, _ in order])

    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 3.4, 0.46 * len(order) + 2.4))
    im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # White on the dark cells, dark on the light ones -- annotations that
            # vanish into the colormap are the usual way a heatmap becomes useless.
            ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if M[i, j] > 0.55 else '#2c3e50')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{g}  ({tag})' for g, tag in order], fontsize=8.5)
    n_amb = sum(1 for _, tag in order if tag == 'ambiguous')
    if 0 < n_amb < len(order):
        ax.axhline(n_amb - 0.5, color='#c0392b', lw=1.6)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='mean probability')
    fig.suptitle(f'{model} -- {entity_type} -- layer {layer}\n'
                 f'Where the probe put its mass, by held-out group')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {shown(path)}")


def plot_layer_trend(contrast_rows, model, entity_type, path):
    """Delta vs depth, one panel per mode.

    Two panels rather than two line styles in one: the modes are on different
    scales (a restricted posterior is bounded to [0,1], a projection coordinate is
    not) and overlaying them would invite reading a gap between them as a result.
    """
    df = pd.DataFrame(contrast_rows)
    if df.empty:
        return
    # The raw panel is drawn alongside the calibrated one, not instead of it,
    # because at saturated layers the temperature fit sharpens hard and drives the
    # calibrated delta to ~0 -- a calibrated-only figure would show that collapse
    # and invite reading it as "no effect at the last layer", which is the one
    # conclusion the other two panels contradict.
    panels = [(f, t) for f, t in (('proba', 'Calibrated posterior toward the rival'),
                                  ('proba_raw', 'Uncalibrated posterior toward the rival'),
                                  ('proj', 'Position on the home->rival axis'))
              if df[f'delta_{f}'].notna().any()]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.6),
                             squeeze=False)
    for ax, (field, title) in zip(axes[0], panels):
        for group, (color, marker) in GROUP_STYLE.items():
            sub = df[df.group == group].sort_values('layer')
            if sub.empty:
                continue
            ax.plot(sub.depth, sub[f'delta_{field}'], marker=marker, lw=2,
                    color=color, label=group)
            ax.fill_between(sub.depth, sub[f'delta_{field}_ci_lo'],
                            sub[f'delta_{field}_ci_hi'], color=color, alpha=0.12)
        # Zero is the whole reference: it is what a model that is simply flat
        # everywhere scores, because its controls are flat by the same amount.
        ax.axhline(0, color='gray', ls=':', lw=1.2,
                   label='no excess over unambiguous controls')
        ax.set_xlabel('relative depth')
        ax.set_ylabel('delta (group - its family controls)')
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f'{model} -- {entity_type}\n'
                 f'Excess mass on the chemically defensible rival family')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {shown(path)}")


def plot_by_group(group_rows, layer, model, entity_type, path):
    """Ambiguous groups and their controls side by side at one layer.

    The controls are drawn, not just subtracted, because the contrast is only
    interpretable if you can see what it was taken against -- a delta of 0.3 means
    something very different when the controls sit at 0.02 than at 0.45.

    Deliberately the UNCALIBRATED posterior rather than the calibrated one. At a
    saturated layer the temperature fit sharpens hard (0.157 at L31 on
    Llama-3.1-8B), which pins every bar to zero and produces a figure that shows
    nothing except the calibration. The calibrated numbers stay in the CSV and in
    the layer-trend figure, where the collapse is visible as a trend rather than
    mistakable for the measurement.
    """
    df = pd.DataFrame(group_rows)
    df = df[df.layer == layer]
    if df.empty:
        return

    order, colors, labels = [], [], []
    for group, (home, _) in sorted(AMBIGUOUS.items()):
        order.append(group)
        colors.append(GROUP_STYLE[group][0])
        labels.append(f'{group}\n({home})')
    for home, groups in sorted(CONTROLS.items()):
        for g in groups:
            order.append(g)
            colors.append('#95a5a6')
            labels.append(f'{g}\ncontrol, {home}')

    panels = [(f, t, ref) for f, t, ref in (
        ('mean_r_raw', 'p(oxygen) / [p(oxygen) + p(home)]', 0.5),
        ('mean_t_proj', 'position on the home->oxygen axis', None))
        if f in df and df[f].notna().any()]
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 4.4),
                             squeeze=False)
    for ax, (field, ylabel, ref) in zip(axes[0], panels):
        vals = [df[df.group == g][field].mean() for g in order]
        ax.bar(range(len(order)), vals, color=colors)
        ax.set_xticks(range(len(order)))
        # Rotated and right-anchored: nine two-line labels laid flat overlap into
        # an unreadable band at any figure width worth using.
        ax.set_xticklabels(labels, fontsize=7.5, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=9)
        if ref is not None:
            ax.axhline(ref, color='gray', ls=':', lw=1, label='evenly split')
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
        else:
            ax.axhline(0, color='gray', ls=':', lw=1)
    fig.suptitle(f'{model} -- {entity_type} -- layer {layer}\n'
                 f'Leaning toward oxygen: ambiguous groups against their controls')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {shown(path)}")


# ============================
# Orchestration
# ============================

def write_csv(path, rows):
    if not rows:
        print(f"no rows for {shown(path)} -- not written")
        return
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {shown(path)}  ({len(rows)} rows)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Graded resolution of the functional groups whose coarse family "
                    "is a naming convention rather than an elemental fact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model-name', default=MODEL_NAME)
    p.add_argument('--entity-type', default=DEFAULT_ENTITY_TYPE)
    p.add_argument('--tag', default='coarse_group',
                   help="Which probe experiment to read, as '{target}_{split}'. Only the "
                        "coarse target has family probabilities whose argmax is the "
                        "prediction; fine_to_coarse is accepted but skips that check.")
    p.add_argument('--full', action='store_true',
                   help="Also run the derived analysis: temperature-calibrated restricted "
                        "posteriors, the ambiguous-vs-control contrast with bootstrap CIs "
                        "and a permutation test, and the centroid-axis projection. That is "
                        "what answers 'could this just be how hard select_C regularized "
                        "each model', and it is far harder to read than the default table, "
                        "which is why it is opt-in.")
    p.add_argument('--mode', choices=('proba', 'projection', 'both'), default='both',
                   help="Only meaningful with --full. proba reads the saved probabilities; "
                        "projection reads the activations. Running both is the point -- "
                        "they are independent readings of the same claim.")
    p.add_argument('--probe-dir', default=None,
                   help="Where the probe wrote its results. Default: the probe's own "
                        "output location for this model and entity_type.")
    p.add_argument('--output-dir', default=None)
    p.add_argument('--seed', type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    model_slug = args.model_name.replace('/', '-')
    num_layers = get_model_config(args.model_name)['num_layers']

    probe_dir = args.probe_dir or os.path.join(
        REPO, DEFAULT_OUTPUT_DIR, model_slug, args.entity_type)
    output_dir = args.output_dir or os.path.join(
        DEFAULT_OUTPUT_ROOT, model_slug, args.entity_type)
    activations_dir = os.path.join(REPO, BASE_ACTIVATIONS_DIR, model_slug,
                                   args.entity_type)

    if args.full and args.mode in ('projection', 'both') and \
            not os.path.isdir(activations_dir):
        raise SystemExit(
            f"--full --mode {args.mode} needs activations, and {activations_dir} does "
            f"not exist. Use --mode proba to run on the saved probabilities alone.")

    df = pd.read_csv(os.path.join(REPO, CSV_PATH))
    raise_if(check_taxonomy(df), 'taxonomy')

    data = load_proba(probe_dir, args.tag)
    raise_if(check_proba(data, df, probe_dir, args.tag), 'saved probabilities')

    print(f"{args.model_name} / {args.entity_type} / {args.tag}: "
          f"{len(data['layers'])} layer(s), {len(data['y_true'])} out-of-fold rows, "
          f"{len(np.unique(data['molecule_index']))} molecules")
    print(f"  ambiguous: {', '.join(sorted(AMBIGUOUS))}")
    print(f"  controls:  " + '; '.join(f'{k}: {", ".join(v)}'
                                       for k, v in sorted(CONTROLS.items())))

    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    best_layer = int(data['layers'][-1])

    # ---- The default: the plain table, and nothing else ----
    fam_rows = family_probability_rows(data, df, model_slug, args.entity_type, num_layers)
    write_csv(os.path.join(output_dir, 'data', 'family_probability.csv'), fam_rows)
    plot_family_probability(fam_rows, best_layer, model_slug, args.entity_type,
                            data['class_names'],
                            os.path.join(output_dir, 'family_probability.png'))

    width = max(len(g) for g, _ in display_order()) + 2
    print(f"\nMean probability by family at layer {best_layer}:")
    print(f"  {'group':{width}s}{'':22s}" +
          ''.join(f'{n:>13s}' for n in data['class_names']))
    by_group = {r['group']: r for r in fam_rows if r['layer'] == best_layer}
    for group, tag in display_order():
        r = by_group.get(group)
        if r is None:
            continue
        mark = '*' if tag == 'ambiguous' else ' '
        print(f"{mark} {group:{width}s}{'(' + tag + ')':22s}" +
              ''.join(f"{r['p_' + n]:13.3f}" for n in data['class_names']))
    print(f"  (* = family is a naming convention; compare its rival column against "
          f"the controls below the rule)")

    if not args.full:
        print(f"\nWrote results to {shown(output_dir)}")
        print("Run again with --full for the calibrated contrast, bootstrap CIs and "
              "the centroid-axis projection.")
        return

    # ---- --full: the derived analysis ----
    rng = np.random.default_rng(args.seed)
    mol_rows, group_rows, contrast_rows = collect(
        data, df, model_slug, args.entity_type, num_layers, args.mode,
        activations_dir, rng)

    write_csv(os.path.join(output_dir, 'data', 'ambiguity_by_molecule.csv'), mol_rows)
    write_csv(os.path.join(output_dir, 'data', 'ambiguity_by_group.csv'), group_rows)
    write_csv(os.path.join(output_dir, 'data', 'ambiguity_contrast.csv'), contrast_rows)

    plot_layer_trend(contrast_rows, model_slug, args.entity_type,
                     os.path.join(output_dir, 'ambiguity_layer_trend.png'))
    plot_by_group(group_rows, best_layer, model_slug, args.entity_type,
                  os.path.join(output_dir, 'ambiguity_by_group.png'))

    print(f"\nDelta at layer {best_layer} (positive = leans to the rival more than its "
          f"own unambiguous family-mates do):")
    for row in contrast_rows:
        if row['layer'] != best_layer:
            continue
        bits = []
        for field, label in (('proba', 'posterior'), ('proba_raw', 'raw'),
                             ('proj', 'projection')):
            d = row[f'delta_{field}']
            if np.isfinite(d):
                bits.append(f"{label} {d:+.3f} "
                            f"[{row[f'delta_{field}_ci_lo']:+.3f}, "
                            f"{row[f'delta_{field}_ci_hi']:+.3f}] "
                            f"p={row[f'delta_{field}_p']:.3f}")
        print(f"  {row['group']:10s} vs {row['controls']:24s} " + ' | '.join(bits))
    print(f"\nWrote results to {shown(output_dir)}")


if __name__ == '__main__':
    main()
