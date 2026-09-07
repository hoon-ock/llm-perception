#!/usr/bin/env python3
"""Derive the functional-group analogy findings from the two HCC result trees.

Reads `Results_HCC/functional_group_analogy/` (within/between cosine, second-order
analogies) and joins it against `Results_HCC/anisotropy_diagnostic/`, which carries
the **mean-centered** version of the same within/between numbers.

That join is the point of this script. Raw cosine among these activations runs
0.78-1.00 -- the space is strongly anisotropic, so `full_within_mean` on its own is
an inflated number that says as much about the global mean direction as about
functional groups. Centering drops between-class similarity to ~0, and what
survives is the real structure. Any claim quoting the uncentered value alone is
quoting the anisotropy.

CPU-only, no model loading, no activations -- just the probes' own outputs.
"""
import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DEFAULT_ANALOGY = os.path.join(REPO, 'fc_group', 'Results_HCC', 'functional_group_analogy')
DEFAULT_ANISO = os.path.join(REPO, 'fc_group', 'Results_HCC', 'anisotropy_diagnostic')
DEFAULT_OUT = os.path.join(HERE, 'data')

MODEL_8B = 'meta-llama-Llama-3.1-8B'
MODEL_70B = 'meta-llama-Llama-3.1-70B'
MODELS = [MODEL_8B, MODEL_70B]
# Layer index is reported as a fraction of the stack so a 32- and an 80-layer model
# can be compared at all. Both trees sample 5 layers only; see README "Limits".
N_LAYERS = {MODEL_8B: 32, MODEL_70B: 80}


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def analogy_scores(analogy_dir, model, entity_type, layer):
    """The individual second-order analogy cosines for one layer.

    Four hand-chosen quadruples (`functional_group_analogy.py:196`), so this is a
    list of length 4 -- small enough that the mean is reported alongside the raw
    values rather than instead of them.
    """
    path = os.path.join(analogy_dir, model, entity_type, 'data',
                        f'second_order_analogy_lumped_layer_{layer}.csv')
    return read_rows(path)


def collect(analogy_dir, aniso_dir, entity_type, models):
    trend, quadruples, by_group = [], [], []

    for model in models:
        a_path = os.path.join(analogy_dir, model, entity_type, 'data',
                              'summary_layer_trend.json')
        n_path = os.path.join(aniso_dir, model, entity_type, 'data',
                              'summary_all_layers.json')
        for path in (a_path, n_path):
            if not os.path.exists(path):
                raise SystemExit(f"missing {path} -- is Results_HCC populated for {entity_type}?")

        analogy = read_json(a_path)
        aniso = read_json(n_path)

        # Merging on layer only works if both sweeps sampled the same layers. If they
        # ever diverge, the centered columns would silently come back empty.
        if sorted(analogy, key=int) != sorted(aniso, key=int):
            raise SystemExit(
                f"{model}: layer sets differ between trees -- "
                f"analogy {sorted(analogy, key=int)} vs anisotropy {sorted(aniso, key=int)}")

        span = N_LAYERS[model] - 1
        for key in sorted(analogy, key=int):
            layer = int(key)
            a, n = analogy[key], aniso[key]
            orig, cent = n['original'], n['mean_centered']
            rows = analogy_scores(analogy_dir, model, entity_type, layer)
            sims = [float(r['cosine_sim']) for r in rows]
            # Per-model: sigma = 1/sqrt(hidden_dim), so the 4096- and 8192-dim models
            # get different thresholds. Taken from the outputs rather than recomputed.
            ci = a['closed_form_ci'][1]

            trend.append({
                'model': model, 'entity_type': entity_type,
                'layer': layer, 'depth': round(layer / span, 4),
                'raw_pairwise_cos': n['raw_isotropy']['pairwise_mean'],
                'orig_within': orig['full_within_mean'],
                'orig_between': orig['full_between_mean'],
                'orig_gap': orig['full_within_mean'] - orig['full_between_mean'],
                'cent_within': cent['full_within_mean'],
                'cent_between': cent['full_between_mean'],
                'cent_gap': cent['full_within_mean'] - cent['full_between_mean'],
                'within_std_across_groups': a['within_std'],
                'pc1_variance_ratio': n['diff_vector_isotropy']['svd_variance_ratio']['1'],
                'analogy_mean': sum(sims) / len(sims),
                'analogy_n': len(sims),
                # Split by direction on purpose. A cosine that clears the CI while
                # *negative* is evidence the two substitution axes point opposite ways
                # -- evidence against the analogy, not for it. An |x| > ci count would
                # score those as successes, and two of the four quadruples here are
                # persistently negative, so the distinction changes the conclusion.
                'analogy_n_positive_sig': sum(1 for s in sims if s > ci),
                'analogy_n_negative_sig': sum(1 for s in sims if s < -ci),
                'closed_form_ci': ci,
            })

            for r, sim in zip(rows, sims):
                quadruples.append({
                    'model': model, 'layer': layer, 'depth': round(layer / span, 4),
                    'pair_a': f"{r['pair_a_group1']} - {r['pair_a_group2']}",
                    'pair_b': f"{r['pair_b_group1']} - {r['pair_b_group2']}",
                    'cosine_sim': sim,
                    'clears_ci': int(abs(sim) > ci),
                })

            for group, score in sorted(a['within_scores'].items()):
                by_group.append({
                    'model': model, 'layer': layer, 'depth': round(layer / span, 4),
                    'group': group, 'within_score': score,
                })

    return trend, quadruples, by_group


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, REPO)}  ({len(rows)} rows)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--analogy-dir', default=DEFAULT_ANALOGY)
    p.add_argument('--aniso-dir', default=DEFAULT_ANISO)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--entity-type', default='functional_group',
                   help="prompt type; the review this was written for covers only the "
                        "declarative functional_group prompt")
    p.add_argument('--model', action='append', choices=MODELS, default=None)
    args = p.parse_args()

    models = args.model or MODELS
    os.makedirs(args.out_dir, exist_ok=True)
    trend, quadruples, by_group = collect(
        args.analogy_dir, args.aniso_dir, args.entity_type, models)

    write_csv(os.path.join(args.out_dir, 'layer_trend.csv'), trend, list(trend[0]))
    write_csv(os.path.join(args.out_dir, 'analogy_quadruples.csv'),
              quadruples, list(quadruples[0]))
    write_csv(os.path.join(args.out_dir, 'within_by_group.csv'),
              by_group, list(by_group[0]))

    print(f"\n=== {args.entity_type}: anisotropy control ===")
    print(f"{'model':>4s} {'L':>4s} {'depth':>6s} {'raw cos':>8s} | "
          f"{'orig w':>7s} {'orig b':>7s} {'orig gap':>9s} | "
          f"{'cent w':>7s} {'cent b':>7s} {'cent gap':>9s}")
    for r in trend:
        print(f"{r['model'][-3:]:>4s} {r['layer']:4d} {r['depth']:6.2f} "
              f"{r['raw_pairwise_cos']:8.3f} | {r['orig_within']:7.3f} {r['orig_between']:7.3f} "
              f"{r['orig_gap']:9.3f} | {r['cent_within']:7.3f} {r['cent_between']:7.3f} "
              f"{r['cent_gap']:9.3f}")

    print(f"\n=== depth trend (is it monotone?) ===")
    for model in models:
        rows = [r for r in trend if r['model'] == model]
        for field, label in [('cent_gap', 'centered gap'),
                             ('cent_within', 'centered within'),
                             ('within_std_across_groups', 'within_std across groups'),
                             ('analogy_mean', 'analogy mean')]:
            vals = [r[field] for r in rows]
            deltas = [b - a for a, b in zip(vals, vals[1:])]
            up = all(d > 0 for d in deltas)
            down = all(d < 0 for d in deltas)
            shape = 'monotone up' if up else 'monotone down' if down else 'NOT monotone'
            peak = rows[max(range(len(vals)), key=lambda i: vals[i])]
            print(f"  {model[-3:]:>3s} {label:26s} {' '.join(f'{v:6.3f}' for v in vals)}"
                  f"   {shape:14s} peak L{peak['layer']} (d={peak['depth']:.2f})")

    print(f"\n=== 8B vs 70B at matched depth (centered gap) ===")
    if len(models) == 2:
        a = [r for r in trend if r['model'] == MODEL_8B]
        b = [r for r in trend if r['model'] == MODEL_70B]
        wins = 0
        for ra, rb in zip(a, b):
            better = '8B' if ra['cent_gap'] > rb['cent_gap'] else '70B'
            wins += better == '8B'
            print(f"  d={ra['depth']:.2f}  8B {ra['cent_gap']:.3f}  vs  70B {rb['cent_gap']:.3f}"
                  f"   -> {better}")
        print(f"  8B ahead at {wins}/{len(a)} matched depths")

    print(f"\n=== analogy quadruples: n per layer and how many clear the CI ===")
    for r in trend:
        print(f"  {r['model'][-3:]:>3s} L{r['layer']:<3d} mean={r['analogy_mean']:6.3f}  "
              f"{r['analogy_n_positive_sig']}/{r['analogy_n']} sig-positive, "
              f"{r['analogy_n_negative_sig']}/{r['analogy_n']} sig-NEGATIVE "
              f"(+-{r['closed_form_ci']:.4f})")


if __name__ == '__main__':
    main()
