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
    # Appended, never inserted: the written rows are model-major, so adding slugs at the end
    # leaves every pre-existing row of both CSVs byte-identical and appends the new ones.
    'weidawang-Chem-R-8B',
    'OpenDFM-ChemDFM-v1.5-8B',
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


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


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
            # The spread travels with the mean rather than being left in the JSON:
            # check_numbers.py traces prose numerals only to committed Analysis/*/data/*.csv,
            # so an SD quoted in the text is untraceable until it lives in this file. It is a
            # DESCRIPTIVE spread over all C(92,2)=4186 molecule pairs, not a standard error --
            # each molecule sits in 91 of them, so the pairs are not independent.
            'raw_pairwise_sd': v['raw_isotropy']['pairwise_std'],
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


def trial_ranks(results_dir, model, entity_type, tree, mode):
    """`{layer: [rank_excl, ...]}` straight off the per-trial rows.

    The only file that can give a real median. `retrieval_by_group.csv` carries per-group
    medians, and a trial-weighted mean of medians is not the median of anything -- so the
    median has to be taken over the trials themselves.
    """
    path = os.path.join(results_dir, tree, model, entity_type, 'data',
                        'retrieval_trials.csv')
    rows = [r for r in read_csv(path) if r['mode'] == mode]
    if not rows:
        # read_csv swallows a missing file into [], which would sail on as a None median
        # rather than stopping. A results tree without trials is a broken results tree.
        raise SystemExit(f'no {mode} trials in {os.path.relpath(path, REPO)} -- re-run '
                         'fc_group/analogy_retrieval.py for this model')
    out = {}
    for r in rows:
        out.setdefault(int(r['layer']), []).append(int(r['rank_excl']))
    return out


def retrieval(results_dir, model, entity_type,
              tree='functional_group_analogy_retrieval', mode=RETRIEVAL_MODE):
    """Pooled hit@1 and degenerate rate by layer, weighted by trial count.

    Also pooled with sulfoxide dropped: it is hit@1 = 0.00 in every model at every
    layer, so it shifts the level for all three equally and only obscures the
    between-model comparison. Both columns are emitted rather than choosing.

    That "equally" is a fact about hit@1 alone, and does NOT carry to the rank columns
    below. Sulfoxide is a miss for everyone, but at L31 on the inter-group tree base misses
    it by one place (rank 2) while chem-r and chem-faithful miss it by the whole pool
    (15 and 16 of ~16). It is the only group where the three disagree at all, and on
    `mean_rank` those 2 trials of 20 are the entire between-model spread. There is
    deliberately no `mean_rank_ex_sulfoxide`: `median_rank` exposes the same tail without
    privileging one group by name.

    `random_hit1` is carried through because it differs per tree -- the candidate pool is
    19 groups for the inter-group set, 4 for the halide column, and the whole ladder for
    the carbon set -- so hit@1 is meaningless without the baseline it belongs to. The same
    holds cutoff by cutoff, which is why `hit2`/`hit3` each travel with their own
    `random_hit2`/`random_hit3` rather than sharing the k=1 line.

    `hit2` reaches the per-axis table only; `geometry_by_layer.csv` filters it out along
    with `hit3` in main(), so that table's columns do not move.

    `mean_rank` answers what no hit rate can: how badly a miss misses. It travels with
    `random_mean_rank` for the same reason every hit column travels with its own baseline,
    and more urgently -- chance rank is the pool size, so it is 8.6 on the inter-group tree
    and 37.3 on the carbon ladder, and a rank read without it says nothing. `mrr` comes
    along as the bounded companion.

    `median_rank` is read from `retrieval_trials.csv` via trial_ranks() rather than pooled
    out of the per-group column here, because a trial-weighted mean of per-group medians is
    not a median. It is worth the second file read: a mean far above its median is a tail,
    and on the inter-group tree at L31 that distinction is the whole result. chem-r and
    chem-faithful mean 2.55 and 2.60 against base's 1.20 while all three have median 1 --
    the gap is two sulfoxide trials, not a worse ordering.
    """
    path = os.path.join(results_dir, tree, model, entity_type, 'data',
                        'retrieval_by_group.csv')
    rows = [r for r in read_csv(path) if r['mode'] == mode]
    ranks = trial_ranks(results_dir, model, entity_type, tree, mode)
    out = {}
    for layer in sorted({int(r['layer']) for r in rows}):
        at = [r for r in rows if int(r['layer']) == layer]
        # The two files are written by the same run over the same trials, so a disagreement
        # here means a half-finished results tree rather than a rounding question.
        n_trials = sum(int(r['n_trials']) for r in at)
        if len(ranks.get(layer, [])) != n_trials:
            raise SystemExit(
                f'{tree}/{model} L{layer} {mode}: retrieval_by_group.csv says {n_trials} '
                f'trials, retrieval_trials.csv has {len(ranks.get(layer, []))}')

        def pooled(subset, field):
            n = sum(int(r['n_trials']) for r in subset)
            if not n:
                return None
            return sum(float(r[field]) * int(r['n_trials']) for r in subset) / n

        ex = [r for r in at if r['group'] != 'sulfoxide']
        out[layer] = {
            'hit1': pooled(at, 'hit1_rate'),
            'hit1_ex_sulfoxide': pooled(ex, 'hit1_rate'),
            'hit2': pooled(at, 'hit2_rate'),
            'hit3': pooled(at, 'hit3_rate'),
            'random_hit1': pooled(at, 'random_hit1'),
            # Each cutoff gets its own chance line, for the same reason hit1 does: the pools
            # differ per tree, and a hit@3 bar read against the hit@1 baseline would look
            # three times better than it is.
            'random_hit2': pooled(at, 'random_hit2'),
            'random_hit3': pooled(at, 'random_hit3'),
            'degenerate_rate': pooled(at, 'degenerate_top1_rate'),
            # Each is a per-group mean over that group's trials, so weighting by trial
            # count recovers the true pooled mean rather than approximating it.
            'mean_rank': pooled(at, 'mean_rank'),
            'random_mean_rank': pooled(at, 'random_mean_rank'),
            'mrr': pooled(at, 'mrr'),
            'median_rank': median(ranks[layer]),
            'n_trials': n_trials,
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
    fields = ['model', 'entity_type', 'layer', 'raw_pairwise_cosine', 'raw_pairwise_sd',
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
               'hit2', 'hit3', 'random_hit1', 'random_hit2', 'random_hit3',
               'degenerate_rate', 'n_trials', 'n_groups',
               # Appended, never inserted: every consumer indexes by name, but the
               # column-position diff is what proves a re-run was additive.
               'mean_rank', 'random_mean_rank', 'mrr', 'median_rank'])

    # ---- the rank distribution behind those summaries -----------------------------
    # A mean and a median are two numbers off a sample that is far from normal: ranks are
    # small integers piled on 1, with a thin tail that on some axes IS the whole result.
    # Snapshotting the histogram lets a figure show the sample itself without reaching into
    # `Results/`, which nothing under Analysis/visuals is allowed to read. Lossless, because
    # ranks are integers -- repeating each rank n times reconstructs the sample exactly.
    hist_rows = []
    for model in args.models:
        for axis, (tree, mode) in RETRIEVAL_AXES.items():
            ranks = trial_ranks(args.results_dir, model, args.entity_type, tree, mode)
            for layer in sorted(ranks):
                counts = {}
                for r in ranks[layer]:
                    counts[r] = counts.get(r, 0) + 1
                for rank in sorted(counts):
                    hist_rows.append({
                        'model': model, 'entity_type': args.entity_type, 'axis': axis,
                        'layer': layer, 'rank': rank, 'n': counts[rank]})
    write_csv(os.path.join(args.out_dir, 'retrieval_rank_hist.csv'), hist_rows,
              ['model', 'entity_type', 'axis', 'layer', 'rank', 'n'])


if __name__ == '__main__':
    main()
