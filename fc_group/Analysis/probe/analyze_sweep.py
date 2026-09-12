#!/usr/bin/env python3
"""Derive the sweep-level findings from `fc_group/Results/functional_group_probe/`.

The probe writes one result tree per (model, entity_type); this script reads across
all of them and produces the four tables that `README.md` quotes, so no number in
that document has to be trusted from prose.

Nothing here loads a model or touches activations -- it reads only the probe's own
outputs (`summary_{tag}.json`, `predictions_{tag}_layer_{best}.csv`,
`surface_baseline_{tag}.csv`) and is CPU-only and fast.

The one statistically load-bearing choice is in `bootstrap_diff`: the resampling
unit is the *molecule*, not the row. See the note there.
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DEFAULT_RESULTS = os.path.join(REPO, 'fc_group', 'Results', 'functional_group_probe')
DEFAULT_OUT = os.path.join(HERE, 'data')

MODEL_8B = 'meta-llama-Llama-3.1-8B'
MODEL_70B = 'meta-llama-Llama-3.1-70B'
MODELS = [MODEL_8B, MODEL_70B]

# 92 molecules minus alkane's 4. `none (alkane)` is the only hydrocarbon, so it is
# skipped as a leave-one-group-out fold and its molecules never enter the pooled
# out-of-fold pool -- which is also why the pooled metric spans 4 classes, not the
# 5 that `summary_*.json` reports `chance` for.
N_MOLECULES = 88


def family(entity_type):
    """Which prompt family an entity type belongs to.

    This split is the main organising axis in the results: declarative templates end
    on the slot that would emit the answer, question templates end on '?'.
    """
    if entity_type.endswith(' question'):
        return 'question'
    if entity_type.startswith('molecule'):
        return 'bare'
    return 'declarative'


def load_curves(results_dir, tag):
    """{model: {entity_type: {layer: balanced_acc}}} from every summary_{tag}.json."""
    out = {}
    for model in MODELS:
        out[model] = {}
        pattern = os.path.join(results_dir, model, '*', 'data', f'summary_{tag}.json')
        for path in sorted(glob.glob(pattern)):
            with open(path) as fh:
                d = json.load(fh)
            out[model][d['entity_type']] = {
                int(k): v for k, v in d['balanced_acc_by_layer'].items()}
        if not out[model]:
            raise SystemExit(
                f"no summary_{tag}.json under {os.path.join(results_dir, model)} -- "
                "is Results populated for this tag?")
    return out


def curve_stats(curve):
    """Depth statistics for one layer curve."""
    layers = sorted(curve)
    span = layers[-1] - layers[0]
    best_layer = max(layers, key=lambda L: curve[L])
    deltas = [(curve[layers[i + 1]] - curve[layers[i]], layers[i + 1])
              for i in range(len(layers) - 1)]
    max_jump, jump_layer = max(deltas)

    # First layer reaching 90% of the way from the layer-0 floor to the peak. This
    # is the saturation point, and it is the more robust of the two depth measures:
    # the largest single-layer jump can land on noise, this cannot.
    floor, best = curve[layers[0]], curve[best_layer]
    threshold = floor + 0.9 * (best - floor)
    layer_90 = next(L for L in layers if curve[L] >= threshold)

    return {
        'n_layers': len(layers),
        'best': best,
        'best_layer': best_layer,
        'best_depth': best_layer / span,
        'last': curve[layers[-1]],
        # Two different things, and conflating them hides real behaviour. The slide
        # is how far below its own peak a curve ends; the step is the last
        # layer-to-layer change alone. A curve can end at its peak yet still have
        # taken a sharp final step (8B hbd: slide 0.000, step +0.031), or slide a
        # long way with only a small final step.
        'drop_best_minus_last': best - curve[layers[-1]],
        'final_step': curve[layers[-1]] - curve[layers[-2]],
        'peak_is_final': int(best_layer == layers[-1]),
        'max_jump': max_jump,
        'max_jump_layer': jump_layer,
        'max_jump_depth': jump_layer / span,
        'layer_90': layer_90,
        'layer_90_depth': layer_90 / span,
    }


def load_predictions(results_dir, model, entity_type, tag):
    """(layer, y_true, y_pred) from predictions_{tag}_layer_{best}.csv."""
    pattern = os.path.join(
        results_dir, model, entity_type, 'data', f'predictions_{tag}_layer_*.csv')
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one predictions file, got {matches}")
    path = matches[0]
    layer = int(path.rsplit('layer_', 1)[1].split('.')[0])
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return (layer,
            np.array([r['y_true'] for r in rows]),
            np.array([r['y_pred'] for r in rows]))


def bootstrap_diff(y_true, pred_a, pred_b, n_boot, rng):
    """Balanced-accuracy difference (a - b) with a molecule-level bootstrap CI.

    The resampling unit is the molecule, not the row. Rows are molecule-major /
    template-minor, so a molecule's N template rows are near-duplicates of each
    other; resampling rows would treat them as independent and shrink the interval
    to a fraction of its honest width. The effective sample size here is 88
    molecules regardless of how many rows the file holds.

    Balanced accuracy is recomputed from per-molecule correct-counts via bincount
    rather than by calling sklearn inside the loop -- same numbers, but fast enough
    to run 20k draws instead of settling for a few hundred.
    """
    n_rows = len(y_true)
    if n_rows % N_MOLECULES:
        raise SystemExit(f"{n_rows} rows is not a whole number of {N_MOLECULES} molecules")
    n_templates = n_rows // N_MOLECULES
    mol_idx = np.repeat(np.arange(N_MOLECULES), n_templates)

    classes = sorted(set(y_true))
    class_of_row = np.array([classes.index(c) for c in y_true])
    class_of_mol = class_of_row[::n_templates]
    # Molecule-major layout is an assumption everything downstream rests on; if the
    # writer ever changes, this catches it rather than silently mis-grouping.
    if not np.array_equal(np.repeat(class_of_mol, n_templates), class_of_row):
        raise SystemExit("labels are not constant within a molecule -- row layout changed")

    correct_a = np.bincount(mol_idx, (pred_a == y_true).astype(float), N_MOLECULES)
    correct_b = np.bincount(mol_idx, (pred_b == y_true).astype(float), N_MOLECULES)
    n_classes = len(classes)

    def balanced(pick, correct):
        rows_per_class = np.bincount(class_of_mol[pick], minlength=n_classes) * n_templates
        hits = np.bincount(class_of_mol[pick], correct[pick], n_classes)
        return float((hits / rows_per_class).mean())

    everything = np.arange(N_MOLECULES)
    acc_a, acc_b = balanced(everything, correct_a), balanced(everything, correct_b)

    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, N_MOLECULES, N_MOLECULES)
        draws[i] = balanced(pick, correct_a) - balanced(pick, correct_b)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return acc_a, acc_b, acc_a - acc_b, float(lo), float(hi)


def load_surface(results_dir, tag):
    """Mean surface baseline over the group folds, wherever it was measured.

    The sweep runs without --surface-baseline, so this is present for only a subset
    of the 8B tasks and none of the 70B ones -- an absence worth showing explicitly
    rather than silently omitting.
    """
    out = []
    for model in MODELS:
        pattern = os.path.join(
            results_dir, model, '*', 'data', f'surface_baseline_{tag}.csv')
        for path in sorted(glob.glob(pattern)):
            entity_type = path.split(os.sep)[-3]
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
            values = [float(r['balanced_acc']) for r in rows]
            out.append({
                'model': model,
                'entity_type': entity_type,
                'family': family(entity_type),
                'n_folds': len(values),
                'mean_balanced_acc': float(np.mean(values)),
            })
    return out


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
    p.add_argument('--tag', default='coarse_group',
                   help="{target}_{split}, e.g. coarse_group or fine_to_coarse_group")
    p.add_argument('--n-boot', type=int, default=20000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    curves = load_curves(args.results_dir, args.tag)

    # ---- 1. long-format layer curves -------------------------------------------
    long_rows = [
        {'model': model, 'entity_type': entity_type, 'family': family(entity_type),
         'layer': layer, 'balanced_acc': acc}
        for model, entities in curves.items()
        for entity_type, curve in sorted(entities.items())
        for layer, acc in sorted(curve.items())
    ]
    write_csv(os.path.join(args.out_dir, 'layer_curves.csv'), long_rows,
              ['model', 'entity_type', 'family', 'layer', 'balanced_acc'])

    # ---- 2. per-entity depth/shape summary --------------------------------------
    summary_rows = []
    for model, entities in curves.items():
        for entity_type, curve in sorted(entities.items()):
            row = {'model': model, 'entity_type': entity_type,
                   'family': family(entity_type)}
            row.update(curve_stats(curve))
            summary_rows.append(row)
    summary_fields = ['model', 'entity_type', 'family', 'n_layers', 'best', 'best_layer',
                      'best_depth', 'last', 'drop_best_minus_last', 'final_step',
                      'peak_is_final', 'max_jump', 'max_jump_layer', 'max_jump_depth',
                      'layer_90', 'layer_90_depth']
    write_csv(os.path.join(args.out_dir, 'per_entity_summary.csv'),
              summary_rows, summary_fields)

    # ---- 3. depth statistics broken out by prompt family -------------------------
    #
    # Pooling all 23 entity types blurs this badly: declarative prompts saturate near
    # half depth while question prompts saturate at the very top, so the pooled modal
    # layer is a blend of two distributions rather than a description of either.
    depth_rows = []
    for model in MODELS:
        for fam in ('declarative', 'question', 'bare'):
            rows = [r for r in summary_rows
                    if r['model'] == model and r['family'] == fam]
            jumps = Counter(r['max_jump_layer'] for r in rows)
            sats = Counter(r['layer_90'] for r in rows)
            jl, jn = jumps.most_common(1)[0]
            sl, sn = sats.most_common(1)[0]
            # The 90%-of-best bar is relative to each curve's own floor-to-peak span,
            # so a flat low-ceiling curve clears it early without that meaning much.
            # Carrying the span makes that visible instead of implicit.
            spans = [max(curves[model][r['entity_type']].values())
                     - curves[model][r['entity_type']][0] for r in rows]
            depth_rows.append({
                'model': model, 'family': fam, 'n': len(rows),
                'jump_modal_layer': jl, 'jump_modal_count': jn,
                'jump_median_depth': float(np.median([r['max_jump_depth'] for r in rows])),
                'sat_modal_layer': sl, 'sat_modal_count': sn,
                'sat_min_layer': min(r['layer_90'] for r in rows),
                'sat_max_layer': max(r['layer_90'] for r in rows),
                'sat_median_depth': float(np.median([r['layer_90_depth'] for r in rows])),
                'median_floor_to_peak_span': float(np.median(spans)),
                'min_floor_to_peak_span': float(min(spans)),
            })
    write_csv(os.path.join(args.out_dir, 'depth_by_family.csv'), depth_rows,
              ['model', 'family', 'n', 'jump_modal_layer', 'jump_modal_count',
               'jump_median_depth', 'sat_modal_layer', 'sat_modal_count', 'sat_min_layer',
               'sat_max_layer', 'sat_median_depth', 'median_floor_to_peak_span',
               'min_floor_to_peak_span'])

    # ---- 4. paired model comparison ---------------------------------------------
    shared = sorted(set(curves[MODEL_8B]) & set(curves[MODEL_70B]))
    diff_rows = []
    for entity_type in shared:
        layer_a, true_a, pred_a = load_predictions(
            args.results_dir, MODEL_8B, entity_type, args.tag)
        layer_b, true_b, pred_b = load_predictions(
            args.results_dir, MODEL_70B, entity_type, args.tag)
        # The pairing is only meaningful if both files describe the same molecules in
        # the same order. Assert it rather than assume it.
        if not np.array_equal(true_a, true_b):
            raise SystemExit(f"{entity_type}: y_true differs between models -- not pairable")
        acc_a, acc_b, diff, lo, hi = bootstrap_diff(
            true_a, pred_a, pred_b, args.n_boot, rng)
        diff_rows.append({
            'entity_type': entity_type, 'family': family(entity_type),
            'layer_8b': layer_a, 'layer_70b': layer_b,
            'bacc_8b': acc_a, 'bacc_70b': acc_b, 'diff_8b_minus_70b': diff,
            'ci_lo': lo, 'ci_hi': hi,
            'ci_excludes_zero': int(not (lo < 0 < hi)),
        })
    write_csv(os.path.join(args.out_dir, 'model_diff_bootstrap.csv'), diff_rows,
              ['entity_type', 'family', 'layer_8b', 'layer_70b', 'bacc_8b', 'bacc_70b',
               'diff_8b_minus_70b', 'ci_lo', 'ci_hi', 'ci_excludes_zero'])

    # ---- 5. surface baselines ----------------------------------------------------
    surface_rows = load_surface(args.results_dir, args.tag)
    write_csv(os.path.join(args.out_dir, 'surface_baseline.csv'), surface_rows,
              ['model', 'entity_type', 'family', 'n_folds', 'mean_balanced_acc'])

    # ---- console report ----------------------------------------------------------
    print(f"\n=== depth: where each model gains ({args.tag}) ===")
    for model in MODELS:
        rows = [r for r in summary_rows if r['model'] == model]
        jumps = Counter(r['max_jump_layer'] for r in rows)
        sat = Counter(r['layer_90'] for r in rows)
        n_layers = rows[0]['n_layers']
        jl, jn = jumps.most_common(1)[0]
        sl, sn = sat.most_common(1)[0]
        print(f"  {model}  (pooled over all {len(rows)} entity types)")
        print(f"    largest single-layer jump   modal L{jl} ({jn}/{len(rows)} entity types, "
              f"depth {jl / (n_layers - 1):.2f})")
        print(f"    first reaching 90% of best  modal L{sl} ({sn}/{len(rows)}, "
              f"depth {sl / (n_layers - 1):.2f})")
        # The pooled figures above average over families that behave differently
        # enough that neither is described by the blend. Break them out.
        for d in [d for d in depth_rows if d['model'] == model]:
            print(f"      {d['family']:12s} n={d['n']:2d}  "
                  f"jump modal L{d['jump_modal_layer']}"
                  f" ({d['jump_modal_count']}/{d['n']})  "
                  f"| 90%-of-best L{d['sat_min_layer']}-L{d['sat_max_layer']}, "
                  f"modal L{d['sat_modal_layer']} ({d['sat_modal_count']}/{d['n']}), "
                  f"median depth {d['sat_median_depth']:.2f}  "
                  f"| floor->peak span {d['median_floor_to_peak_span']:.3f} "
                  f"(min {d['min_floor_to_peak_span']:.3f})")

    # Every family, including `bare`. Reporting only two of the three is how the
    # first version of the write-up came to claim the late-layer decline was
    # "confined to declarative prompts" -- `bare` declines just as much, it was
    # simply not printed. Likewise `#peak at final` is reported as a count rather
    # than a >0.02 threshold, so an absolute claim ("all of them peak at the end")
    # can be checked here instead of inferred from a rounded-off table.
    print(f"\n=== prompt family: accuracy and late-layer behaviour ===")
    for model in MODELS:
        for fam in ('declarative', 'question', 'bare'):
            rows = [r for r in summary_rows
                    if r['model'] == model and r['family'] == fam]
            drops = [r['drop_best_minus_last'] for r in rows]
            steps = [r['final_step'] for r in rows]
            print(f"  {model[-3:]:>3s} {fam:12s} n={len(rows):2d}  "
                  f"mean best={np.mean([r['best'] for r in rows]):.3f}  "
                  f"mean slide={np.mean(drops):.3f} (max {max(drops):.3f})  "
                  f"mean final step={np.mean(steps):+.3f}  "
                  f"#peak at final: {sum(r['peak_is_final'] for r in rows)}/{len(rows)}")
    negative = [r for r in summary_rows if r['final_step'] < 0]
    print(f"  -> {len(negative)}/{len(summary_rows)} curves end on a negative final step; "
          f"sharpest: " + ", ".join(
              f"{r['model'][-3:]} {r['entity_type']} {r['final_step']:+.3f}"
              for r in sorted(negative, key=lambda r: r['final_step'])[:3]))

    print(f"\n=== 8B vs 70B, paired ({args.n_boot} molecule-level draws, seed {args.seed}) ===")
    sig = [r for r in diff_rows if r['ci_excludes_zero']]
    print(f"  CI excludes 0 for {len(sig)}/{len(diff_rows)} entity types: "
          f"8B ahead in {sum(1 for r in sig if r['diff_8b_minus_70b'] > 0)}, "
          f"70B ahead in {sum(1 for r in sig if r['diff_8b_minus_70b'] < 0)}")
    for fam in ('question', 'declarative', 'bare'):
        vals = [r['diff_8b_minus_70b'] for r in diff_rows if r['family'] == fam]
        if len(vals) < 2:
            continue
        mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        t = mean / (sd / len(vals) ** 0.5)
        print(f"  {fam:12s} n={len(vals):2d}  mean(8B-70B)={mean:+.4f}  sd={sd:.4f}  "
              f"t={t:+.2f}  8B ahead {sum(1 for v in vals if v > 0)}/{len(vals)}")

    measured = [r for r in surface_rows]
    if measured:
        vals = [r['mean_balanced_acc'] for r in measured]
        by_model = Counter(r['model'] for r in measured)
        print(f"\n=== surface baseline (character n-grams, no model) ===")
        print(f"  measured for {len(measured)} of {2 * len(shared)} tasks "
              f"({dict(by_model)}); range {min(vals):.3f}-{max(vals):.3f}")


if __name__ == '__main__':
    main()
