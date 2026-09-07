#!/usr/bin/env python3
"""Join the anisotropy diagnostic against the t-SNE verdicts.

The t-SNE sweep itself wrote no numbers (see README section 1), so the visual
verdicts in `sampled_plots.csv` are all there is on that side. This script pulls
the quantities that *are* measured -- from `Results_HCC/anisotropy_diagnostic/`,
which covers the same prompts and the same layers -- and lines them up against
those verdicts, so README section 5 argues from a table rather than from prose.

The headline it supports: the prompts whose activations are most spread out are
exactly the prompts that show visible functional-group clusters, and the layer
where each model is most spread out is exactly the layer where the clusters are
cleanest.

Read carefully though -- `perform_pca` in `tsne_functional_groups.py` centers
before t-SNE, so mean pairwise cosine is a *proxy* for effective dimensionality
here, not something t-SNE sees directly. See the README for that caveat.
"""
import argparse
import csv
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DEFAULT_ANISO = os.path.join(REPO, 'fc_group', 'Results_HCC', 'anisotropy_diagnostic')
DEFAULT_CONFIG = os.path.join(REPO, 'fc_group', 'config_extract_activation.yaml')
DEFAULT_VERDICTS = os.path.join(HERE, 'sampled_plots.csv')
DEFAULT_OUT = os.path.join(HERE, 'data')

MODEL_8B = 'meta-llama-Llama-3.1-8B'
MODEL_70B = 'meta-llama-Llama-3.1-70B'
MODELS = [MODEL_8B, MODEL_70B]
N_LAYERS = {MODEL_8B: 32, MODEL_70B: 80}
# The layer at which t-SNE clustering looks cleanest in each model (README s4).
PEAK_LAYER = {MODEL_8B: 24, MODEL_70B: 60}
N_MOLECULES = 92
SVD_K = ['1', '2', '3', '5', '10']


def read_json(path):
    with open(path) as fh:
        return json.load(fh)


def template_counts(config_path):
    """{entity_type: n_templates} -- sets how many points t-SNE was given.

    This is not a footnote. The two bare prompts have a single template and so
    92 points, against 644-1012 for everything else, and the perplexity rule
    `min(30, max(5, n // 4))` turns that into a different t-SNE configuration.
    Silhouettes and cluster appearance are not comparable across that boundary.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    return {e['entity_type']: len(e['templates']) for e in cfg['extraction']['entities']}


def load_verdicts(path):
    """{(model_suffix, prompt, layer): verdict} from the inspected-plot log."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[(r['model'][-3:], r['prompt_type'], int(r['layer']))] = r['visual_verdict']
    return out


def collect(aniso_dir, templates, verdicts):
    rows = []
    for model in MODELS:
        span = N_LAYERS[model] - 1
        model_dir = os.path.join(aniso_dir, model)
        if not os.path.isdir(model_dir):
            raise SystemExit(f"missing {model_dir} -- is Results_HCC populated?")
        for entity_type in sorted(os.listdir(model_dir)):
            path = os.path.join(model_dir, entity_type, 'data', 'summary_all_layers.json')
            if not os.path.exists(path):
                continue
            n_t = templates.get(entity_type)
            summary = read_json(path)
            for key in sorted(summary, key=int):
                layer, v = int(key), summary[key]
                cent = v['mean_centered']
                row = {
                    'model': model, 'entity_type': entity_type,
                    'layer': layer, 'depth': round(layer / span, 4),
                    'raw_pairwise_cos': v['raw_isotropy']['pairwise_mean'],
                    'cent_within': cent['full_within_mean'],
                    'cent_between': cent['full_between_mean'],
                    'cent_gap': cent['full_within_mean'] - cent['full_between_mean'],
                    'n_templates': n_t,
                    'n_points': None if n_t is None else n_t * N_MOLECULES,
                    'tsne_verdict': verdicts.get((model[-3:], entity_type, layer), ''),
                }
                for k in SVD_K:
                    row[f'diffvec_cumvar_k{k}'] = v['diff_vector_isotropy']['svd_variance_ratio'][k]
                rows.append(row)
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
    p.add_argument('--aniso-dir', default=DEFAULT_ANISO)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--verdicts', default=DEFAULT_VERDICTS)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    templates = template_counts(args.config)
    verdicts = load_verdicts(args.verdicts)
    rows = collect(args.aniso_dir, templates, verdicts)
    write_csv(os.path.join(args.out_dir, 'anisotropy_by_prompt_layer.csv'),
              rows, list(rows[0]))

    print("\n=== anisotropy across depth (lower = more spread out) ===")
    for model in MODELS:
        for entity_type in ('pka', 'functional_group'):
            sel = [r for r in rows if r['model'] == model and r['entity_type'] == entity_type]
            if not sel:
                continue
            best = min(sel, key=lambda r: r['raw_pairwise_cos'])
            traj = "  ".join(f"L{r['layer']}:{r['raw_pairwise_cos']:.3f}" for r in sel)
            print(f"  {model[-3:]:>3s} {entity_type:18s} {traj}")
            print(f"  {'':>3s} {'':18s} -> most isotropic at L{best['layer']} "
                  f"(d={best['depth']:.2f}); t-SNE peak is L{PEAK_LAYER[model]}")

    # Compare against the verdicts at the layers the verdicts were actually made
    # at -- most were read off L16/L40, not the peak layer -- so the ranking is
    # scored on like-for-like rather than against a different layer's numbers.
    judged = sorted({(r['model'], r['layer']) for r in rows if r['tsne_verdict']})
    print("\n=== prompt ranking where verdicts exist (sorted by anisotropy) ===")
    for model, layer in judged:
        sel = sorted((r for r in rows if r['model'] == model and r['layer'] == layer
                      and r['tsne_verdict']),
                     key=lambda r: r['raw_pairwise_cos'])
        if len(sel) < 2:
            continue
        print(f"\n  --- {model[-3:]} L{layer} ---")
        print(f"     {'prompt':36s} {'aniso':>7s} {'PC1':>6s} {'cent gap':>9s} "
              f"{'pts':>5s}  verdict")
        for r in sel:
            print(f"     {r['entity_type']:36s} {r['raw_pairwise_cos']:7.3f} "
                  f"{r['diffvec_cumvar_k1']:6.3f} {r['cent_gap']:9.3f} "
                  f"{r['n_points'] if r['n_points'] else '?':>5}  {r['tsne_verdict']}")

    print("\n=== full prompt ranking at each model's t-SNE peak layer ===")
    for model in MODELS:
        peak = PEAK_LAYER[model]
        sel = sorted((r for r in rows if r['model'] == model and r['layer'] == peak),
                     key=lambda r: r['raw_pairwise_cos'])
        print(f"\n  --- {model[-3:]} L{peak} ---")
        print(f"     {'prompt':36s} {'aniso':>7s} {'PC1':>6s} {'cent gap':>9s} {'pts':>5s}")
        for r in sel:
            print(f"     {r['entity_type']:36s} {r['raw_pairwise_cos']:7.3f} "
                  f"{r['diffvec_cumvar_k1']:6.3f} {r['cent_gap']:9.3f} "
                  f"{r['n_points'] if r['n_points'] else '?':>5}")

    print("\n=== point count / perplexity groups ===")
    by_n = {}
    for et, n_t in sorted(templates.items()):
        n_pts = n_t * N_MOLECULES
        perplexity = min(30, max(5, n_pts // 4))
        by_n.setdefault((n_pts, perplexity), []).append(et)
    for (n_pts, perp), ets in sorted(by_n.items()):
        print(f"  {n_pts:5d} points, perplexity {perp:2d}: {len(ets):2d} prompts"
              + (f"  ({', '.join(ets)})" if n_pts < 200 else ""))


if __name__ == '__main__':
    main()
