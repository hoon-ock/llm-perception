#!/usr/bin/env python3
"""The layer-by-layer geometry table: what fine-tuning did to the representation.

Joins three result trees that each hold one piece of the same story and are never
otherwise read together:

  anisotropy_diagnostic/   how much of any cosine is the shared dominant direction,
                           and what survives mean-centering
  functional_group_analogy/    within-class (same group, different chain lengths)
                           and between-class diff-vector cosines
  functional_group_analogy_retrieval/  degenerate_top1_rate -- how often the analogy
                           offset fails to leave the source group's neighbourhood

Two columns carry most of the weight.

`centered_between` versus `centered_null` is the honest-signal check: raw between-class
cosines run 0.40-0.45 at layer 31 purely because the residual stream is anisotropic
(raw pairwise cosine 0.94 in Llama), and after centering the between-class MEAN lands
exactly on the empirical null. Within-class does not. Any claim resting on
between-class similarity has to be framed as relative structure, not as magnitude.

`degenerate_rate` is the quantitative form of "the groups look more standalone": a
trial is degenerate when b2 + (a1 - a2) is nearest to b2 itself once the exclusion is
lifted. It is scored on the same trials as hit@1, so a model can retrieve well and
still have a weak offset -- which is what the fine-tuned models do.

Reads only `Results/`. No activations, no model, CPU-only.
"""
import argparse
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DEFAULT_RESULTS = os.path.join(REPO, 'fc_group', 'Results')
DEFAULT_OUT = os.path.join(HERE, 'data')

MODELS = [
    'meta-llama-Llama-3.1-8B',
    'phenixace-Chem-R-Faithful',
    'deepseek-ai-DeepSeek-R1-Distill-Llama-8B',
]
# Reported alongside hit@1 so the two are never read apart; see the docstring.
RETRIEVAL_MODE = 'lumped'

# Three retrieval trees, three independent axes of the same structure. The paper reports
# all three against their own random baselines, which differ because the candidate pools
# differ (19 groups vs 4 halides vs the full chain-length ladder) -- so an axis cannot be
# read against another axis's chance line.
RETRIEVAL_AXES = {
    'inter_group': ('functional_group_analogy_retrieval', 'lumped'),
    'halide': ('functional_group_analogy_retrieval_halide', 'lumped'),
    'carbon_ladder': ('functional_group_analogy_retrieval_carbon', 'ladder'),
}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def anisotropy(results_dir, model, entity_type):
    path = os.path.join(results_dir, 'anisotropy_diagnostic', model, entity_type,
                        'data', 'summary_all_layers.json')
    with open(path) as fh:
        raw = json.load(fh)
    out = {}
    for layer, v in raw.items():
        svd = v['diff_vector_isotropy']['svd_variance_ratio']
        out[int(layer)] = {
            'raw_pairwise_cosine': v['raw_isotropy']['pairwise_mean'],
            'diffvec_svd_top1': svd['1'],
            'diffvec_svd_top5': svd['5'],
            'centered_within': v['mean_centered']['full_within_mean'],
            'centered_between': v['mean_centered']['full_between_mean'],
            'centered_null': v['mean_centered']['empirical_null_mean'],
            'centered_null_sd': v['mean_centered']['empirical_null_std'],
        }
    return out


def similarity(results_dir, model, entity_type, layer):
    base = os.path.join(results_dir, 'functional_group_analogy', model, entity_type,
                        'data')
    within = [float(r['cosine_sim'])
              for r in read_csv(os.path.join(
                  base, f'within_class_similarity_layer_{layer}.csv'))]
    between = [float(r['cosine_sim'])
               for r in read_csv(os.path.join(
                   base, f'between_class_similarity_layer_{layer}.csv'))]
    w, b = mean(within), mean(between)
    return {'raw_within': w, 'raw_between': b,
            'raw_gap': None if w is None or b is None else w - b,
            'raw_between_max': max(between) if between else None}


def retrieval(results_dir, model, entity_type,
              tree='functional_group_analogy_retrieval', mode=RETRIEVAL_MODE):
    """Pooled hit@1 and degenerate rate by layer, weighted by trial count.

    Also pooled with sulfoxide dropped: it is hit@1 = 0.00 in every model at every
    layer, so it shifts the level for all three equally and only obscures the
    between-model comparison. Both columns are emitted rather than choosing.

    `random_hit1` is carried through because it differs per tree -- the candidate pool is
    19 groups for the inter-group set, 4 for the halide column, and the whole ladder for
    the carbon set -- so hit@1 is meaningless without the baseline it belongs to.
    """
    path = os.path.join(results_dir, tree, model, entity_type, 'data',
                        'retrieval_by_group.csv')
    rows = [r for r in read_csv(path) if r['mode'] == mode]
    out = {}
    for layer in sorted({int(r['layer']) for r in rows}):
        at = [r for r in rows if int(r['layer']) == layer]

        def pooled(subset, field):
            n = sum(int(r['n_trials']) for r in subset)
            if not n:
                return None
            return sum(float(r[field]) * int(r['n_trials']) for r in subset) / n

        ex = [r for r in at if r['group'] != 'sulfoxide']
        out[layer] = {
            'hit1': pooled(at, 'hit1_rate'),
            'hit1_ex_sulfoxide': pooled(ex, 'hit1_rate'),
            'hit3': pooled(at, 'hit3_rate'),
            'random_hit1': pooled(at, 'random_hit1'),
            'degenerate_rate': pooled(at, 'degenerate_top1_rate'),
            'n_trials': sum(int(r['n_trials']) for r in at),
            'n_groups': len(at),
        }
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
    p.add_argument('--entity-type', default='functional_group')
    p.add_argument('--models', nargs='+', default=MODELS)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fields = ['model', 'entity_type', 'layer', 'raw_pairwise_cosine',
              'diffvec_svd_top1', 'diffvec_svd_top5', 'raw_within', 'raw_between',
              'raw_gap', 'raw_between_max', 'centered_within', 'centered_between',
              'centered_null', 'centered_null_sd', 'hit1', 'hit1_ex_sulfoxide',
              'degenerate_rate', 'n_trials']
    rows = []
    for model in args.models:
        aniso = anisotropy(args.results_dir, model, args.entity_type)
        retr = retrieval(args.results_dir, model, args.entity_type)
        for layer in sorted(aniso):
            row = {'model': model, 'entity_type': args.entity_type, 'layer': layer}
            row.update(aniso[layer])
            row.update(similarity(args.results_dir, model, args.entity_type, layer))
            # The inter-group tree is the one this table carries; `hit3`/`random_hit1`/
            # `n_groups` belong to the per-axis table below, where the baseline that makes
            # them readable lives alongside them.
            row.update({k: v for k, v in retr.get(layer, {}).items() if k in fields})
            rows.append({k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in row.items()})
    write_csv(os.path.join(args.out_dir, 'geometry_by_layer.csv'), rows, fields)

    # ---- retrieval across all three axes -----------------------------------------
    axis_rows = []
    for model in args.models:
        for axis, (tree, mode) in RETRIEVAL_AXES.items():
            for layer, v in sorted(retrieval(args.results_dir, model,
                                             args.entity_type, tree, mode).items()):
                axis_rows.append({
                    'model': model, 'entity_type': args.entity_type, 'axis': axis,
                    'layer': layer,
                    **{k: (round(x, 6) if isinstance(x, float) else x)
                       for k, x in v.items()}})
    write_csv(os.path.join(args.out_dir, 'retrieval_by_axis.csv'), axis_rows,
              ['model', 'entity_type', 'axis', 'layer', 'hit1', 'hit1_ex_sulfoxide',
               'hit3', 'random_hit1', 'degenerate_rate', 'n_trials', 'n_groups'])


if __name__ == '__main__':
    main()
