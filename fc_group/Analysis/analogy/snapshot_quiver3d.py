#!/usr/bin/env python3
"""Project the analogy diff-vectors into 3D so the successful retrievals can be drawn.

    python fc_group/Analysis/analogy/snapshot_quiver3d.py [--layer 31] [--mode carbon_matched]

`analyze_analogy.py` snapshots the *cosine* view of the analogies. This writes the
*geometric* one: `data/analogy_quiver3d.csv`, the 3D coordinates
`Analysis/paper/make_visuals_retrieval_3d.py` draws as `visuals/retrieval/analogy3d_*.png`,
one PNG per row of the selection below. It is a separate file rather than another writer
inside `analyze_analogy.py` because it needs two Results trees and numpy, and because the
three CSVs that script emits feed the paper's tables -- they should not move when this one
is regenerated.

Retrieval asks whether `b2 + (a1 - a2)` lands nearest `b1`. That is a statement about a
parallelogram, and a parallelogram is the one thing the existing quiver
(`functional_group_analogy_carbon_matched.py:408`) cannot show: it projects onto an
**uncentered** top-3 SVD basis, where the anisotropic mean direction owns SVD-1 and every
group's arrow points the same way. Worse, that basis can invert the answer -- at base/L31 it
puts `alkyl fluoride` nearest the constructed point for the imine/amine quadruple and drops
the true `amine` to rank 3, in a case that is rank 1 in full 4096-D.

So this projection is **mean-centered**. Offsets are translation-invariant, so the
parallelogram loses nothing; what is given up is the uncentered origin's reading as "no
change from the alkane baseline", which is not what this figure is about. Measured over the
quadruples that retrieve perfectly, the 3D rank of the true answer is 1 everywhere centered,
against 1/3/1 uncentered. The same position `analyze_analogy.py`'s docstring already takes
about cosine: an uncentered number here is largely quoting the anisotropy.

The default pool is **carbon-matched**: one chain length at a time, so every point is a
concrete compound rather than an average over C3-C6. The `molecules` column names them. Seven
of the ten groups these quadruples use are a single compound at a fixed chain length; alcohol,
thiol and amine each carry an `n-`/`sec-` isomer pair in the dataset, so those three stay a
2-molecule mean and say so in `n_molecules`. `--mode lumped` restores the chain-length average.

Two guards against the figure flattering itself:

  * `rank_excl` and `cos_to_target` are **copied** from `retrieval_trials.csv`, never
    recomputed, so a panel cannot disagree with `f4_retrieval` about an outcome.
  * `rank3d_excl` recomputes that rank from the 3 plotted coordinates by Euclidean
    distance -- what a reader's eye actually does with a scatter. When it disagrees with
    `rank_excl` the projection is misleading and both the console and the figure have to
    say so. `dist3d_true` and `dist3d_rival` carry the same comparison as a margin, so the
    panel can show that the answer won rather than merely that the residual was short.

CPU-only. Reads `Results/functional_group_analogy/` (the diff-vector npz),
`Results/functional_group_analogy_retrieval/` (the scored trials) and the committed
`fc_group/functional_group_dataset.csv` (the compound names).
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FC = os.path.join(REPO, 'fc_group')
DEFAULT_ANALOGY = os.path.join(FC, 'Results', 'functional_group_analogy')
DEFAULT_RETRIEVAL = os.path.join(FC, 'Results', 'functional_group_analogy_retrieval')
DEFAULT_DATASET = os.path.join(FC, 'functional_group_dataset.csv')
DEFAULT_OUT = os.path.join(HERE, 'data')

sys.path.insert(0, FC)
# Import rather than restate, the rule analogy_retrieval.py:56 sets for the quadruple list:
# if the npz key format, the per-carbon pool or the set of scoreable corners ever changes,
# this file must change with it rather than quietly keep projecting the old thing.
from analogy_retrieval import load_diff_vectors, scoreable_corners  # noqa: E402
from functional_group_analogy_carbon_matched import (  # noqa: E402
    ANALOGY_QUADRUPLES, build_diff_by_group_at_carbon,
)
from model_registry import get_model_config  # noqa: E402

MODEL_8B = 'meta-llama-Llama-3.1-8B'
SLASH_FOR = {MODEL_8B: 'meta-llama/Llama-3.1-8B'}

# The quadruple `functional_group_analogy_carbon_matched.py:306-311` marks DELIBERATE
# NEGATIVE CONTROL: acid->ester and aldehyde->ketone are formally the same substitution but
# chemically are not, so it is the one that should fail if leg symmetry is what makes an
# analogy work. It does not fail, which is a result -- but a panel drawing it must not read
# as a plain success. Named here rather than parsed out of a source comment, and asserted
# below so that editing the quadruple list breaks this loudly instead of silently dropping
# the annotation.
NEGATIVE_CONTROLS = {('ester', 'carboxylic acid', 'ketone', 'aldehyde')}
assert all(tuple(a) + tuple(b) in {tuple(x) + tuple(y) for x, y in ANALOGY_QUADRUPLES}
           for a, b in [(('ester', 'carboxylic acid'), ('ketone', 'aldehyde'))]), \
    'NEGATIVE_CONTROLS no longer matches ANALOGY_QUADRUPLES'


def lumped_pool(diffs):
    """One vector per group, averaged over chain lengths 3-6 -- the same expression as
    `analogy_retrieval.collect`'s lumped branch. Kept selectable, but no point in a lumped
    panel is a molecule, which is what `carbon_matched` is the default for."""
    return {g: np.mean(list(by_c.values()), axis=0) for g, by_c in diffs.items()}


def fit_centered_3d_basis(matrix):
    """Top-3 PCA basis, mean-CENTERED -- the sibling of
    `functional_group_analogy_carbon_matched.fit_uncentered_3d_basis`, and deliberately not
    it. See the module docstring for why this figure inverts that choice.

    Returns (components (3, D), mean (D,), explained variance fraction).
    """
    mean = matrix.mean(axis=0)
    _, s, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    return vt[:3], mean, float((s[:3] ** 2).sum() / (s ** 2).sum())


def read_molecules(path):
    """`(group, chain length) -> [compound name]`, from the committed dataset.

    This is what makes the panels per-molecule rather than merely per-chain-length. Most
    cells hold one compound; alcohol, thiol and amine hold an `n-`/`sec-` isomer pair at
    every chain length, and the diff vector for those is the mean of the two.
    """
    if not os.path.exists(path):
        raise SystemExit(f'missing {os.path.relpath(path, REPO)}')
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r['functional_group'], int(r['carbon_count'])), []).append(
                r['common_name'])
    return {k: sorted(v) for k, v in out.items()}


def read_trials(retrieval_dir, model, entity_type, layer, mode):
    """The scored trials of one layer in one pool mode."""
    path = os.path.join(retrieval_dir, model, entity_type, 'data', 'retrieval_trials.csv')
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {os.path.relpath(path, REPO)} -- run "
            "fc_group/analogy_retrieval.py for this model first")
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh)
                if int(r['layer']) == layer and r['mode'] == mode]
    if not rows:
        raise SystemExit(
            f"no {mode} rows at layer {layer} in {os.path.relpath(path, REPO)} -- "
            "that tree carries layers 0, 8, 16, 24, 31")
    return rows


def rank_cells(trials):
    """The (quadruple, chain length) cells of one layer, most convincingly retrieved first.

    Three criteria, in order, and the second is the one that matters:

      1. every scoreable corner is hit@1;
      2. no corner is `degenerate` -- the constructed point's unexcluded nearest neighbour
         is the source group itself, meaning the offset never escaped the neighbourhood it
         started in and the exclusion did the work. A cell can be 4/4 on hit@1 and still be
         mostly degenerate, and ranking on hit@1 alone would put those on the figure;
      3. highest mean cosine to the constructed point, as a tiebreak only.

    Keyed per cell rather than per quadruple because a quadruple can be clean at one chain
    length and degenerate at another -- `ester` is 0 degenerate at C4 and 2 at C5 -- and
    averaging those together is exactly what the lumped pool did.
    """
    by_cell = {}
    for r in trials:
        carbon = None if r['carbon_count'] == '' else int(float(r['carbon_count']))
        key = (r['pair_a1'], r['pair_a2'], r['pair_b1'], r['pair_b2'], carbon)
        by_cell.setdefault(key, []).append(r)

    out = []
    for key, rows in by_cell.items():
        quad, carbon = key[:4], key[4]
        expected = len(scoreable_corners(quad[:2], quad[2:]))
        if len(rows) != expected:
            raise SystemExit(
                f"{key}: {len(rows)} trials but {expected} scoreable corners -- "
                "the trials CSV and scoreable_corners disagree")
        out.append({
            'quad': quad,
            'carbon': carbon,
            'rows': rows,
            'n_corners': len(rows),
            'n_hit1': sum(int(r['hit1']) for r in rows),
            'n_degenerate': sum(int(r['degenerate']) for r in rows),
            'mean_cos': sum(float(r['cos_to_target']) for r in rows) / len(rows),
        })
    out.sort(key=lambda c: (c['n_corners'] - c['n_hit1'], c['n_degenerate'], -c['mean_cos']))
    return out


def is_clean(cell):
    """Criteria 1 and 2 of `rank_cells` -- every corner a hit, none of them degenerate."""
    return cell['n_hit1'] == cell['n_corners'] and cell['n_degenerate'] == 0


def canonical_corner(cell):
    """The `b1 = b2 + (a1 - a2)` trial -- the corner the panel draws explicitly.

    One of four; the parallelogram the panel draws implies the other three, and the corner
    counts are annotated beside it.
    """
    a1, a2, b1, b2 = cell['quad']
    for r in cell['rows']:
        if r['predicted_group'] == b1 and r['source_group'] == b2:
            return r
    raise SystemExit(f"{cell['quad']}: no trial row for the b1 = b2 + (a1 - a2) corner")


def project(pool, cell, molecules, model, layer, entity_type, mode, cell_rank):
    """One CSV row per plotted point: the four members, the constructed point, the rest."""
    groups = sorted(pool)
    matrix = np.stack([pool[g] for g in groups])
    components, mean, explained = fit_centered_3d_basis(matrix)
    coords = (matrix - mean) @ components.T
    at = {g: coords[i] for i, g in enumerate(groups)}

    a1, a2, b1, b2 = cell['quad']
    corner = canonical_corner(cell)
    target = pool[b2] + pool[a1] - pool[a2]
    predicted = at[b2] + at[a1] - at[a2]

    # How much of the constructed point the three drawn dimensions actually hold. Well under
    # 1 for every cell here, which is why the panel prints it: the picture is a shadow of a
    # 4096-dimensional claim, not the claim.
    centered = target - mean
    recon = float(np.linalg.norm(components.T @ (components @ centered))
                  / np.linalg.norm(centered))

    # Euclidean, not cosine: this is the check on whether the PICTURE says what the numbers
    # say, and distance on a scatter is what a reader reads.
    exclude = {b2, a1, a2} - {b1}
    ranked = sorted((g for g in groups if g not in exclude),
                    key=lambda g: float(np.linalg.norm(at[g] - predicted)))
    rank3d = ranked.index(b1) + 1
    # The margin, in the plotted units. hit@1 is a claim about winning a race, so the panel
    # quotes the gap to the runner-up next to the residual; a small residual on its own would
    # be equally consistent with a pool that is tight everywhere.
    rival = next(g for g in ranked if g != b1)

    span = get_model_config(SLASH_FOR.get(model, model))['num_layers'] - 1
    shared = {
        'model': model, 'layer': layer, 'depth': round(layer / span, 4),
        'entity_type': entity_type, 'mode': mode, 'cell_rank': cell_rank,
        'carbon_count': '' if cell['carbon'] is None else cell['carbon'],
        'pair_a1': a1, 'pair_a2': a2, 'pair_b1': b1, 'pair_b2': b2,
        'is_negative_control': int(cell['quad'] in NEGATIVE_CONTROLS),
        'explained_var': round(explained, 6),
        'recon_frac': round(recon, 6),
        'n_groups': len(groups),
        'n_corners': cell['n_corners'],
        'n_hit1': cell['n_hit1'],
        'n_degenerate': cell['n_degenerate'],
        'mean_cos': round(cell['mean_cos'], 6),
        'n_candidates_excl': int(corner['n_candidates_excl']),
        'rank_excl': int(corner['rank_excl']),
        'rank3d_excl': rank3d,
        'cos_to_target': round(float(corner['cos_to_target']), 6),
        'dist3d_true': round(float(np.linalg.norm(at[b1] - predicted)), 6),
        'dist3d_rival': round(float(np.linalg.norm(at[rival] - predicted)), 6),
        'rival_group': rival,
    }

    role_of = {a1: 'a1', a2: 'a2', b1: 'b1', b2: 'b2'}
    rows = []
    for g in groups:
        names = molecules.get((g, cell['carbon']), []) if cell['carbon'] else []
        rows.append(dict(shared, role=role_of.get(g, 'other'), group=g,
                         molecules=' + '.join(names), n_molecules=len(names),
                         svd1=round(float(at[g][0]), 6),
                         svd2=round(float(at[g][1]), 6),
                         svd3=round(float(at[g][2]), 6)))
    rows.append(dict(shared, role='predicted', group='', molecules='', n_molecules=0,
                     svd1=round(float(predicted[0]), 6),
                     svd2=round(float(predicted[1]), 6),
                     svd3=round(float(predicted[2]), 6)))
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, REPO)}  ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layer', type=int, default=31)
    p.add_argument('--model', default=MODEL_8B)
    p.add_argument('--entity-type', default='functional_group')
    p.add_argument('--mode', default='carbon_matched', choices=['carbon_matched', 'lumped'],
                   help='carbon_matched draws one chain length at a time, so every point is '
                        'a compound; lumped averages each group over C3-C6 (default: '
                        'carbon_matched)')
    p.add_argument('--top-n', type=int, default=0,
                   help='cap the number of cells projected, best first (default 0: every '
                        'cell that is all-hit@1 with no degenerate corner)')
    p.add_argument('--all-cells', action='store_true',
                   help='project every cell, failures included, so they can be read against '
                        'the successes')
    p.add_argument('--analogy-dir', default=DEFAULT_ANALOGY)
    p.add_argument('--retrieval-dir', default=DEFAULT_RETRIEVAL)
    p.add_argument('--dataset', default=DEFAULT_DATASET)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()

    npz = os.path.join(args.analogy_dir, args.model, args.entity_type, 'data',
                       f'diff_vectors_layer_{args.layer}.npz')
    if not os.path.exists(npz):
        raise SystemExit(
            f"missing {os.path.relpath(npz, REPO)} -- run "
            "fc_group/functional_group_analogy_carbon_matched.py for this model first")
    diffs = load_diff_vectors(npz)
    molecules = read_molecules(args.dataset)
    ordered = rank_cells(read_trials(
        args.retrieval_dir, args.model, args.entity_type, args.layer, args.mode))

    chosen = ordered if args.all_cells else [c for c in ordered if is_clean(c)]
    if args.top_n:
        chosen = chosen[:args.top_n]
    if not chosen:
        raise SystemExit(
            f'no cell at layer {args.layer} is all-hit@1 with no degenerate corner -- '
            'pass --all-cells to project the near misses anyway')

    rows, disagreed = [], []
    for i, cell in enumerate(chosen, start=1):
        pool = (build_diff_by_group_at_carbon(diffs, cell['carbon'])
                if cell['carbon'] is not None else lumped_pool(diffs))
        made = project(pool, cell, molecules, args.model, args.layer, args.entity_type,
                       args.mode, i)
        rows.extend(made)
        if made[0]['rank3d_excl'] != made[0]['rank_excl']:
            disagreed.append((cell, made[0]['rank3d_excl']))

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, 'analogy_quiver3d.csv'), rows, list(rows[0]))

    drawn = {(c['quad'], c['carbon']) for c in chosen}
    print(f"\n=== {args.entity_type} @ L{args.layer}, {args.mode} pool, "
          f"{args.model} -- cells best first ===")
    print(f"{'':2s} {'quadruple':46s} {'C':>2s} {'hit@1':>6s} {'degen':>6s} "
          f"{'mean cos':>9s} {'rank':>5s} {'3D':>3s} {'recon':>6s} {'3D margin':>17s}")
    for cell in ordered:
        a1, a2, b1, b2 = cell['quad']
        mark = '*' if (cell['quad'], cell['carbon']) in drawn else ' '
        mine = [r for r in rows if r['pair_a1'] == a1 and r['pair_b1'] == b1
                and r['carbon_count'] == ('' if cell['carbon'] is None else cell['carbon'])]
        r3 = f"{mine[0]['rank3d_excl']:3d}" if mine else '  -'
        rc = f"{mine[0]['recon_frac']:6.2f}" if mine else '     -'
        mg = (f"{mine[0]['dist3d_true']:5.1f} vs {mine[0]['dist3d_rival']:5.1f}"
              if mine else '')
        print(f"{mark:2s} {f'{a1} : {a2} :: {b1} : {b2}':46s} "
              f"{(cell['carbon'] or 0):2d} {cell['n_hit1']:3d}/{cell['n_corners']:<2d} "
              f"{cell['n_degenerate']:6d} {cell['mean_cos']:9.3f} "
              f"{int(canonical_corner(cell)['rank_excl']):5d} {r3} {rc} {mg:>17s}")

    if disagreed:
        print('\nWARNING: the projection disagrees with the full-D result for '
              + ', '.join(f"{' : '.join(c['quad'])} "
                          f"{'C' + str(c['carbon']) if c['carbon'] else 'lumped'} "
                          f"(3D rank {r})" for c, r in disagreed)
              + ' -- the panel must say so, or the cell must come off the figure.')
    else:
        print(f'\nprojection agrees with full-D on all {len(chosen)} drawn cell(s) '
              '(rank3d_excl == rank_excl)')


if __name__ == '__main__':
    main()
