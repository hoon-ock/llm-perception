#!/usr/bin/env python3
"""Does per-functional-group generation quality track per-group representation quality?

`Results/generation_eval/*/functional_group/data/summary.json` has free-generation strict recall
ranging 0.05-0.90 across the 20 functional groups. The obvious hypothesis is that the spread is
embedding quality: groups the model represents badly are the groups it names badly. This script
tests it against every per-group representation readout that exists, and writes the answer plus
the evidence needed to interpret it.

Three things have to be established before a correlation means anything, and each gets its own
output file:

`group_reliability.csv`   Is there per-group signal at all? Split-half over templates (even vs
                          odd), Spearman-Brown corrected. A model below ~0.5 cannot support any
                          per-group claim and is reported as underpowered, not as a null.

`group_representation_join.csv`
                          One row per (model, layer, group): every generation outcome beside every
                          representation predictor. Carries the two degenerate predictors
                          explicitly -- `ladder_top1_same_label` (constant 1.0) and
                          `probe_fine_recall` (surface-readable, see below) -- because "this
                          predictor has no variance" is the finding, and deleting the column would
                          hide it.

`group_correlations.csv`  Spearman per (predictor, outcome, model) plus a pooled fit that ranks
                          within model before pooling, each with a cluster bootstrap resampling
                          FUNCTIONAL GROUPS, not rows: the five models share the same 20 groups,
                          so a row-level interval would treat five looks at `alkyl iodide` as five
                          independent observations.

On the two degenerate predictors, both established before any correlation with them was inspected:

  * `ladder_top1_same_label` is 1.000 for all 19 groups in all 5 models. When carbon-ladder
    retrieval misses it lands on the right functional group at the wrong carbon count, so that
    axis measures chain-length resolution and has zero variance on group identity.

  * The 20-way `--target fine --split molecule` probe reaches 0.9875 balanced accuracy at layer 0
    and ~0.999 by L30. Its own surface control settles why: a char n-gram model of the raw prompt
    text scores 1.0000 on the same folds, because every template embeds `{iupac_name}` and
    "1-bromobutane" spells its own answer. The fine probe under the molecule split is an
    orthography measure, not a representation measure, which is also why the fine target is only
    ever run here with `--split group` elsewhere in the repo. It is kept in the join as a
    documented dead end.

That leaves `rival_margin` / `silhouette` (from `functional_group_separability.py`, the only
predictor with all 20 groups and real variance) as the primary, and the analogy-retrieval axes as
secondary. Preregistration note: the primary predictor named in the plan was
`probe_fine_recall`; it was disqualified by the surface baseline above before its correlation with
any outcome was computed, and `rival_margin` was substituted. Every other predictor x outcome cell
is marked exploratory and BH-corrected across the whole grid -- an uncorrected scan of this grid
returns ~18 cells at p<0.05 by chance alone.

Reads `Results/` directly, as the other `Analysis/*/analyze_*.py` do, and snapshots into
`Analysis/generation/data/` so the figure script never has to.
"""
import argparse
import collections
import csv
import json
import os

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))
FC = os.path.abspath(os.path.join(ANALYSIS, '..'))
RESULTS = os.path.join(FC, 'Results')
DEFAULT_OUT = os.path.join(HERE, 'data')

# Presentation order: base, the Tier-1 chemistry pair in training order, the off-base chemistry
# model, then the reasoning distill. Same order as Analysis/paper/_common.py BEHAVIOURAL.
MODELS = ['meta-llama-Llama-3.1-8B', 'weidawang-Chem-R-8B', 'phenixace-Chem-R-Faithful',
          'OpenDFM-ChemDFM-v1.5-8B', 'deepseek-ai-DeepSeek-R1-Distill-Llama-8B']
SHORT = {MODELS[0]: 'base', MODELS[1]: 'chem-r', MODELS[2]: 'chem',
         MODELS[3]: 'chemdfm', MODELS[4]: 'reason'}
ENTITY = 'functional_group'
# The layers every readout shares: analogy retrieval is only saved at these five.
LAYERS = [0, 8, 16, 24, 31]
HEADLINE_LAYER = 31
N_BOOT = 5000
SEED = 0
# Reliability below this is treated as too noisy to interpret a per-group correlation.
RELIABILITY_FLOOR = 0.5
# From `functional_group_probe.py --target fine --split molecule --surface-baseline`: char n-gram
# TF-IDF on the raw prompt text, same folds, same seed. Model-independent by construction (the
# baseline never touches activations), so one run covers all five checkpoints.
SURFACE_BASELINE_FINE_MOLECULE = 1.0

RESPONSE_TYPES = ('answer_correct', 'answer_wrong', 'underspecified', 'non_answer', 'malformed')


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {os.path.relpath(path, FC)} -- see the module docstring for "
                         "which script writes it")
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


# ============================
# Generation outcomes
# ============================

def generation_outcomes():
    """{(model, group): {outcome: value}} from the adjudicated per-prompt table.

    Rebuilt from response types rather than read out of `free_generation_by_class.csv`, which
    carries strict recall only. The five types partition every prompt, so the shares sum to 1 and
    `superclass_credit` is exactly the strict rate plus the underspecified rate -- the outcome
    that removes the halide confound, where the model answers "alkyl halide" and is filed
    `underspecified` despite naming the right family.
    """
    rows = read_csv(os.path.join(ANALYSIS, 'generation', 'data',
                                 'free_generation_adjudicated.csv'))
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        counts[(r['model'], r['true_label'])][r['response_type']] += 1

    out = {}
    for key, c in counts.items():
        n = sum(c.values())
        committed = c['answer_correct'] + c['answer_wrong']
        out[key] = {
            'n_prompts': n,
            'strict': c['answer_correct'] / n,
            'superclass_credit': (c['answer_correct'] + c['underspecified']) / n,
            'commit': committed / n,
            'conditional_acc': c['answer_correct'] / committed if committed else float('nan'),
            'answer_wrong': c['answer_wrong'] / n,
            'underspecified': c['underspecified'] / n,
            'non_answer': c['non_answer'] / n,
            'malformed': c['malformed'] / n,
        }
    return out


def teacher_forced():
    rows = read_csv(os.path.join(ANALYSIS, 'generation', 'data', 'generation_by_class.csv'))
    return {(r['model'], r['functional_group']): float(r['recall'])
            for r in rows if r['scoring'] == 'raw'}


def split_half_reliability(outcomes_of_interest):
    """Spearman between per-group rates computed on even vs odd template indices.

    Templates, not molecules: the per-group n is 40-80 prompts but only 4-8 molecules, so a
    molecule-level split would leave 2-4 molecules per half and measure nothing. Spearman-Brown
    steps the half-length correlation up to the full-length reliability, which is the number the
    attenuation correction needs.
    """
    rows = read_csv(os.path.join(ANALYSIS, 'generation', 'data',
                                 'free_generation_adjudicated.csv'))
    halves = collections.defaultdict(collections.Counter)
    for r in rows:
        half = int(r['template_index']) % 2
        halves[(r['model'], r['true_label'], half)][r['response_type']] += 1

    def rate(counter, outcome):
        n = sum(counter.values())
        if outcome == 'strict':
            return counter['answer_correct'] / n
        if outcome == 'superclass_credit':
            return (counter['answer_correct'] + counter['underspecified']) / n
        if outcome == 'commit':
            return (counter['answer_correct'] + counter['answer_wrong']) / n
        if outcome == 'conditional_acc':
            d = counter['answer_correct'] + counter['answer_wrong']
            return counter['answer_correct'] / d if d else float('nan')
        return counter[{'answer_wrong': 'answer_wrong', 'underspecified': 'underspecified',
                        'non_answer': 'non_answer', 'malformed': 'malformed'}[outcome]] / n

    out, rel_rows = {}, []
    for model in MODELS:
        groups = sorted({k[1] for k in halves if k[0] == model})
        if not groups:
            continue
        for outcome in outcomes_of_interest:
            a = [rate(halves[(model, g, 0)], outcome) for g in groups]
            b = [rate(halves[(model, g, 1)], outcome) for g in groups]
            ok = [i for i in range(len(groups)) if np.isfinite(a[i]) and np.isfinite(b[i])]
            if len(ok) < 5 or len(set(np.asarray(a)[ok])) < 3:
                r_half = float('nan')
            else:
                r_half = float(spearmanr(np.asarray(a)[ok], np.asarray(b)[ok]).statistic)
            sb = 2 * r_half / (1 + r_half) if np.isfinite(r_half) and r_half > -1 else float('nan')
            out[(model, outcome)] = sb
            rel_rows.append({
                'model': model, 'short': SHORT[model], 'entity_type': ENTITY,
                'outcome': outcome, 'n_groups': len(ok),
                'split_half_rho': round(r_half, 6) if np.isfinite(r_half) else '',
                'spearman_brown': round(sb, 6) if np.isfinite(sb) else '',
                'interpretable': int(np.isfinite(sb) and sb >= RELIABILITY_FLOOR),
            })
    return out, rel_rows


# ============================
# Representation predictors
# ============================

def separability():
    """{(model, layer, group): row} from functional_group_separability.py."""
    out = {}
    for model in MODELS:
        path = os.path.join(RESULTS, 'functional_group_separability', model, ENTITY,
                            'data', 'separability_by_group.csv')
        for r in read_csv(path):
            out[(model, int(r['layer']), r['group'])] = r
    return out


def retrieval(tree, mode):
    out = {}
    for model in MODELS:
        path = os.path.join(RESULTS, tree, model, ENTITY, 'data', 'retrieval_by_group.csv')
        for r in read_csv(path):
            if r['mode'] == mode:
                out[(model, int(r['layer']), r['group'])] = r
    return out


def probe_fine_recall():
    """Per-class recall of the 20-way probe, from the long-format confusion table.

    Diagonal cell rate is recall by construction (`n / n_true_total` with pred == true), so no
    matrix reconstruction is needed. Returns {} rather than failing if the fine run is absent --
    it is a documented dead end, not a dependency.
    """
    out = {}
    for model in MODELS:
        path = os.path.join(RESULTS, 'functional_group_probe', model, ENTITY,
                            'data', 'confusion_fine_molecule.csv')
        if not os.path.exists(path):
            continue
        for r in read_csv(path):
            if r['true_class'] == r['pred_class']:
                out[(model, int(r['layer']), r['true_class'])] = float(r['rate'])
    return out


def probe_fine_p_true():
    """Mean out-of-fold p(true class) per group, from the saved probability cube."""
    out = {}
    for model in MODELS:
        path = os.path.join(RESULTS, 'functional_group_probe', model, ENTITY,
                            'data', 'oof_proba_fine_molecule.npz')
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        proba, layers = z['proba'].astype(np.float32), z['layers']
        y_true, groups = z['y_true'], z['fine_group'].astype(str)
        for li, layer in enumerate(layers):
            p = proba[li][np.arange(len(y_true)), y_true]
            for g in sorted(set(groups)):
                out[(model, int(layer), g)] = float(p[groups == g].mean())
    return out


def within_by_group():
    """Per-group cluster cohesion already snapshotted by the analogy analysis (3 models only)."""
    out = {}
    for r in read_csv(os.path.join(ANALYSIS, 'analogy', 'data', 'within_by_group.csv')):
        out[(r['model'], int(r['layer']), r['group'])] = float(r['within_score'])
    return out


# ============================
# Statistics
# ============================

def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved, monotone."""
    idx = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals):
    """BH adjusted p-values (step-up), order preserved, monotone."""
    order = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m * pvals[i] / (rank + 1))
        adj[i] = min(1.0, running)
    return adj


def _spearman_rows(A, B):
    """Spearman rho for every row of A against the matching row of B, vectorized.

    Equivalent to calling `scipy.stats.spearmanr` once per row -- same average-rank tie handling,
    since `rankdata(axis=1)` is the same ranking scipy applies internally -- but two C-level
    ranking calls instead of 5000 Python ones. The bootstrap grid here is ~3000 cells x 5000
    draws; per-row scipy calls take about 25 minutes, this takes seconds.
    """
    ra = rankdata(A, axis=1).astype(np.float64)
    rb = rankdata(B, axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, (ra * rb).sum(axis=1) / denom, np.nan)


def cluster_bootstrap(x, y, clusters, n_boot=N_BOOT, seed=SEED):
    """Percentile CI for Spearman rho, resampling whole clusters (functional groups).

    Clusters, not rows: the five models share the same 20 groups, so resampling rows would treat
    five looks at `alkyl iodide` as five independent observations and shrink the interval by
    roughly sqrt(5).

    Draws that collapse either variable to fewer than 3 distinct values carry no rank information
    and are dropped rather than counted as rho=0; `n_effective` reports how many survived.
    """
    x, y, clusters = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    uniq = sorted(set(clusters))
    index = [np.where(clusters == c)[0] for c in uniq]
    sizes = {len(ix) for ix in index}
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), (n_boot, len(uniq)))

    if len(sizes) == 1:
        # Every cluster the same size, so each draw has the same length and the whole bootstrap
        # is one (n_boot, n) gather. This is the path that actually runs.
        flat = np.stack(index)                      # (n_clusters, cluster_size)
        idx = flat[pick].reshape(n_boot, -1)        # (n_boot, n_clusters * cluster_size)
        A, B = x[idx], y[idx]
    else:
        # Unequal cluster sizes (a predictor missing for some model x group): draw lengths vary,
        # so pad to the longest with NaN. rankdata puts NaN last consistently in both arrays, and
        # a constant trailing block adds the same tied rank to each, leaving rho unchanged.
        rows_a, rows_b = [], []
        width = max(len(np.concatenate([index[i] for i in row])) for row in pick)
        for row in pick:
            ix = np.concatenate([index[i] for i in row])
            pad = width - len(ix)
            rows_a.append(np.concatenate([x[ix], np.full(pad, np.nan)]))
            rows_b.append(np.concatenate([y[ix], np.full(pad, np.nan)]))
        A, B = np.stack(rows_a), np.stack(rows_b)

    # A draw that lost the predictor's variation cannot inform the interval.
    keep = ((np.diff(np.sort(A, axis=1), axis=1) != 0).sum(axis=1) >= 2) & \
           ((np.diff(np.sort(B, axis=1), axis=1) != 0).sum(axis=1) >= 2)
    boots = _spearman_rows(A[keep], B[keep]) if keep.any() else np.empty(0)
    boots = boots[np.isfinite(boots)]
    if len(boots) < 100:
        return float('nan'), float('nan'), int(len(boots))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), int(len(boots))


def correlate(x, y, clusters, reliability):
    """Spearman with a cluster-bootstrap CI and an attenuation-corrected rho.

    Dividing by sqrt(reliability) undoes the shrinkage that measurement noise in the outcome puts
    on rho, so a small rho at high reliability ("really is no relationship") is distinguishable
    from a small rho at low reliability ("could not have detected one").
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or len(set(x)) < 3 or len(set(y)) < 3:
        return None
    s = spearmanr(x, y)
    lo, hi, n_eff = cluster_bootstrap(x, y, clusters)
    corrected = (s.statistic / np.sqrt(reliability)
                 if reliability and np.isfinite(reliability) and reliability > 0 else float('nan'))
    return {
        'rho': float(s.statistic), 'p': float(s.pvalue),
        'ci_lo': lo, 'ci_hi': hi, 'n_boot_effective': n_eff,
        'ci_excludes_zero': int(np.isfinite(lo) and lo * hi > 0),
        'rho_attenuation_corrected': float(np.clip(corrected, -1.5, 1.5))
        if np.isfinite(corrected) else float('nan'),
        'reliability_used': float(reliability) if np.isfinite(reliability) else float('nan'),
    }


# ============================
# Assembly
# ============================

JOIN_COLUMNS = [
    'model', 'short', 'entity_type', 'layer', 'functional_group', 'n_prompts',
    # generation outcomes
    'strict', 'superclass_credit', 'commit', 'conditional_acc',
    'answer_wrong', 'underspecified', 'non_answer', 'malformed', 'tf_raw_recall',
    # representation: separability (all 20 groups, real variance)
    'rival_margin', 'rival_margin_sd', 'silhouette', 'within_cos', 'nearest_rival',
    'frac_rows_positive_margin',
    # representation: analogy retrieval, group-identity axes (17 of 20 groups)
    'retr_hit1', 'retr_hit2', 'retr_mrr', 'retr_mean_rank', 'retr_median_rank',
    'retr_n_trials', 'retr_axis',
    # representation: carbon ladder (19 of 20 groups; same-label rate is the degenerate column)
    'ladder_hit1', 'ladder_mrr', 'ladder_mean_rank', 'ladder_median_rank',
    'ladder_n_trials', 'ladder_top1_same_label',
    # representation: 20-way probe (surface-confounded dead end, kept on the record)
    'probe_fine_recall', 'probe_fine_mean_p_true', 'probe_fine_surface_baseline',
    # representation: cohesion already snapshotted by the analogy analysis (3 models)
    'within_score',
]

PREDICTORS = [
    # (column, higher_is_better, family)
    ('rival_margin', True, 'separability'),
    ('silhouette', True, 'separability'),
    ('within_cos', True, 'separability'),
    ('retr_hit1', True, 'retrieval_group_identity'),
    ('retr_mrr', True, 'retrieval_group_identity'),
    ('retr_mean_rank', False, 'retrieval_group_identity'),
    ('retr_median_rank', False, 'retrieval_group_identity'),
    ('ladder_hit1', True, 'retrieval_carbon_ladder'),
    ('ladder_mrr', True, 'retrieval_carbon_ladder'),
    ('ladder_mean_rank', False, 'retrieval_carbon_ladder'),
    ('ladder_top1_same_label', True, 'retrieval_carbon_ladder'),
    ('probe_fine_recall', True, 'probe_fine_surface_confounded'),
    ('probe_fine_mean_p_true', True, 'probe_fine_surface_confounded'),
    ('within_score', True, 'analogy_cohesion'),
]
OUTCOMES = ['strict', 'superclass_credit', 'commit', 'conditional_acc',
            'underspecified', 'non_answer', 'malformed']
PRIMARY_PREDICTOR = 'rival_margin'
PRIMARY_OUTCOME = 'superclass_credit'


def num(row, field):
    if row is None:
        return ''
    v = row.get(field, '')
    return round(float(v), 6) if v not in ('', None) else ''


def build_join():
    gen, tf = generation_outcomes(), teacher_forced()
    sep, pf_recall, pf_p = separability(), probe_fine_recall(), probe_fine_p_true()
    ladder = retrieval('functional_group_analogy_retrieval_carbon', 'ladder')
    inter = retrieval('functional_group_analogy_retrieval', 'carbon_matched')
    halide = retrieval('functional_group_analogy_retrieval_halide', 'carbon_matched')
    within = within_by_group()

    rows = []
    for model in MODELS:
        groups = sorted(g for (m, g) in gen if m == model)
        for layer in LAYERS:
            for g in groups:
                o = gen[(model, g)]
                s = sep.get((model, layer, g))
                # inter_group and halide are disjoint by construction: the halide tree exists
                # precisely because the four halides are pooled out of the inter-group one.
                r, axis = inter.get((model, layer, g)), 'inter_group'
                if r is None:
                    r, axis = halide.get((model, layer, g)), 'halide'
                if r is None:
                    axis = ''
                L = ladder.get((model, layer, g))
                rows.append({
                    'model': model, 'short': SHORT[model], 'entity_type': ENTITY,
                    'layer': layer, 'functional_group': g, 'n_prompts': o['n_prompts'],
                    'strict': round(o['strict'], 6),
                    'superclass_credit': round(o['superclass_credit'], 6),
                    'commit': round(o['commit'], 6),
                    'conditional_acc': round(o['conditional_acc'], 6),
                    'answer_wrong': round(o['answer_wrong'], 6),
                    'underspecified': round(o['underspecified'], 6),
                    'non_answer': round(o['non_answer'], 6),
                    'malformed': round(o['malformed'], 6),
                    'tf_raw_recall': round(tf[(model, g)], 6) if (model, g) in tf else '',
                    'rival_margin': num(s, 'rival_margin'),
                    'rival_margin_sd': num(s, 'rival_margin_sd'),
                    'silhouette': num(s, 'silhouette'),
                    'within_cos': num(s, 'within_cos'),
                    'nearest_rival': s['nearest_rival'] if s else '',
                    'frac_rows_positive_margin': num(s, 'frac_rows_positive_margin'),
                    'retr_hit1': num(r, 'hit1_rate'), 'retr_hit2': num(r, 'hit2_rate'),
                    'retr_mrr': num(r, 'mrr'), 'retr_mean_rank': num(r, 'mean_rank'),
                    'retr_median_rank': num(r, 'median_rank'),
                    'retr_n_trials': num(r, 'n_trials'), 'retr_axis': axis,
                    'ladder_hit1': num(L, 'hit1_rate'), 'ladder_mrr': num(L, 'mrr'),
                    'ladder_mean_rank': num(L, 'mean_rank'),
                    'ladder_median_rank': num(L, 'median_rank'),
                    'ladder_n_trials': num(L, 'n_trials'),
                    'ladder_top1_same_label': num(L, 'top1_same_label_rate'),
                    'probe_fine_recall': round(pf_recall[(model, layer, g)], 6)
                    if (model, layer, g) in pf_recall else '',
                    'probe_fine_mean_p_true': round(pf_p[(model, layer, g)], 6)
                    if (model, layer, g) in pf_p else '',
                    'probe_fine_surface_baseline': SURFACE_BASELINE_FINE_MOLECULE
                    if (model, layer, g) in pf_recall else '',
                    'within_score': round(within[(model, layer, g)], 6)
                    if (model, layer, g) in within else '',
                })
    return rows


def build_correlations(join, reliability):
    at_layer = collections.defaultdict(list)
    for r in join:
        at_layer[(r['model'], r['layer'])].append(r)

    out = []
    for pred, higher_better, family in PREDICTORS:
        sign = 1.0 if higher_better else -1.0
        for outcome in OUTCOMES:
            for layer in LAYERS:
                # --- per model ---
                pooled_x, pooled_y, pooled_c = [], [], []
                for model in MODELS:
                    rows = [r for r in at_layer[(model, layer)]
                            if r[pred] != '' and r[outcome] != '']
                    if len(rows) < 5:
                        continue
                    x = [sign * float(r[pred]) for r in rows]
                    y = [float(r[outcome]) for r in rows]
                    groups = [r['functional_group'] for r in rows]
                    rel = reliability.get((model, outcome), float('nan'))
                    res = correlate(x, y, groups, rel)
                    if res is None:
                        continue
                    out.append(dict(
                        scope='per_model', model=model, short=SHORT[model], layer=layer,
                        predictor=pred, predictor_family=family,
                        predictor_higher_is_better=int(higher_better), outcome=outcome,
                        n_groups=len(rows), **res,
                        is_primary=int(pred == PRIMARY_PREDICTOR
                                       and outcome == PRIMARY_OUTCOME
                                       and layer == HEADLINE_LAYER),
                        interpretable=int(np.isfinite(rel) and rel >= RELIABILITY_FLOOR),
                    ))
                    # Ranks are taken within model before pooling so a model-level offset in
                    # either variable cannot masquerade as a group-level relationship.
                    pooled_x.extend(rankdata(x))
                    pooled_y.extend(rankdata(y))
                    pooled_c.extend(groups)
                # --- pooled ---
                if len(pooled_x) >= 10:
                    rel = float(np.nanmean([reliability.get((m, outcome), np.nan)
                                            for m in MODELS]))
                    res = correlate(pooled_x, pooled_y, pooled_c, rel)
                    if res is not None:
                        out.append(dict(
                            scope='pooled', model='', short='pooled', layer=layer,
                            predictor=pred, predictor_family=family,
                            predictor_higher_is_better=int(higher_better), outcome=outcome,
                            n_groups=len(set(pooled_c)), **res,
                            is_primary=int(pred == PRIMARY_PREDICTOR
                                           and outcome == PRIMARY_OUTCOME
                                           and layer == HEADLINE_LAYER),
                            interpretable=int(np.isfinite(rel) and rel >= RELIABILITY_FLOOR),
                        ))

    # Holm over the preregistered primary family only; BH over everything else. Splitting them is
    # the point: a primary test corrected against 1300 exploratory cells has no power left.
    prim = [i for i, r in enumerate(out) if r['is_primary']]
    expl = [i for i, r in enumerate(out) if not r['is_primary']]
    for r in out:
        r['p_holm'], r['p_bh'] = '', ''
    if prim:
        for i, adj in zip(prim, holm([out[i]['p'] for i in prim])):
            out[i]['p_holm'] = round(float(adj), 6)
    if expl:
        for i, adj in zip(expl, benjamini_hochberg([out[i]['p'] for i in expl])):
            out[i]['p_bh'] = round(float(adj), 6)
    for r in out:
        for k in ('rho', 'p', 'ci_lo', 'ci_hi', 'rho_attenuation_corrected', 'reliability_used'):
            r[k] = round(r[k], 6) if np.isfinite(r[k]) else ''
    return out


def write_csv(path, rows, columns):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {os.path.relpath(path, FC)} ({len(rows)} rows)")


def report(join, corr, rel_rows, reliability):
    print("\n=== per-group reliability (Spearman-Brown, even vs odd templates) ===")
    for r in rel_rows:
        if r['outcome'] in ('strict', 'superclass_credit'):
            flag = '' if r['interpretable'] else '   <- UNDERPOWERED, correlations uninterpretable'
            print(f"  {r['short']:8s} {r['outcome']:18s} {r['spearman_brown']}{flag}")

    print(f"\n=== degenerate predictors (layer {HEADLINE_LAYER}) ===")
    for pred in ('ladder_top1_same_label', 'probe_fine_recall'):
        vals = sorted({r[pred] for r in join
                       if r['layer'] == HEADLINE_LAYER and r[pred] != ''})
        if vals:
            print(f"  {pred:24s} {len(vals)} distinct value(s) across "
                  f"{sum(1 for r in join if r['layer'] == HEADLINE_LAYER and r[pred] != '')} "
                  f"(model, group) cells: {vals[:6]}{' ...' if len(vals) > 6 else ''}")
    print(f"  probe_fine surface char n-gram baseline on the same folds: "
          f"{SURFACE_BASELINE_FINE_MOLECULE:.4f} -- the 20-way probe is readable from the "
          f"prompt's spelling alone.")

    print(f"\n=== PRIMARY test: {PRIMARY_PREDICTOR} vs {PRIMARY_OUTCOME}, "
          f"layer {HEADLINE_LAYER} ===")
    for r in corr:
        if r['is_primary']:
            flag = '' if r['interpretable'] else '  (underpowered)'
            print(f"  {r['short']:8s} n={r['n_groups']:2d} rho={r['rho']:+.3f} "
                  f"CI[{r['ci_lo']},{r['ci_hi']}] p_holm={r['p_holm']} "
                  f"rho_corrected={r['rho_attenuation_corrected']}{flag}")

    surv = [r for r in corr if not r['is_primary'] and r['p_bh'] != ''
            and float(r['p_bh']) < 0.05 and r['interpretable'] and r['ci_excludes_zero']]
    print(f"\n=== exploratory grid: {sum(1 for r in corr if not r['is_primary'])} cells, "
          f"{len(surv)} surviving BH<0.05 with a bootstrap CI excluding zero ===")
    for r in sorted(surv, key=lambda r: -abs(float(r['rho'])))[:20]:
        print(f"  {r['short']:8s} L{r['layer']:<2d} {r['predictor']:24s} vs "
              f"{r['outcome']:18s} rho={r['rho']:+.3f} CI[{r['ci_lo']},{r['ci_hi']}] "
              f"p_bh={r['p_bh']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    reliability, rel_rows = split_half_reliability(OUTCOMES)
    join = build_join()
    corr = build_correlations(join, reliability)

    print("Writing snapshots:")
    write_csv(os.path.join(args.out_dir, 'group_representation_join.csv'), join, JOIN_COLUMNS)
    write_csv(os.path.join(args.out_dir, 'group_reliability.csv'), rel_rows,
              ['model', 'short', 'entity_type', 'outcome', 'n_groups', 'split_half_rho',
               'spearman_brown', 'interpretable'])
    write_csv(os.path.join(args.out_dir, 'group_correlations.csv'), corr,
              ['scope', 'model', 'short', 'layer', 'predictor', 'predictor_family',
               'predictor_higher_is_better', 'outcome', 'n_groups', 'rho', 'p',
               'ci_lo', 'ci_hi', 'ci_excludes_zero', 'n_boot_effective',
               'rho_attenuation_corrected', 'reliability_used', 'is_primary',
               'interpretable', 'p_holm', 'p_bh'])
    with open(os.path.join(args.out_dir, 'group_representation_meta.json'), 'w') as fh:
        json.dump({'models': MODELS, 'layers': LAYERS, 'headline_layer': HEADLINE_LAYER,
                   'n_boot': N_BOOT, 'seed': SEED,
                   'primary_predictor': PRIMARY_PREDICTOR, 'primary_outcome': PRIMARY_OUTCOME,
                   'reliability_floor': RELIABILITY_FLOOR,
                   'probe_fine_surface_baseline': SURFACE_BASELINE_FINE_MOLECULE,
                   'outcomes': OUTCOMES,
                   'predictors': [p[0] for p in PREDICTORS]}, fh, indent=2)

    report(join, corr, rel_rows, reliability)


if __name__ == '__main__':
    main()
