#!/usr/bin/env python3
"""Retrieval-based second-order analogy test, as a counterpart to the cosine one.

`functional_group_analogy_carbon_matched.py` scores each quadruple by the cosine
between two offset vectors -- cos(thioether-thiol, ether-alcohol). Those top out
around +0.66, which reads as "the king-man+woman analogy isn't really here".

But offset cosine is not what the word2vec analogy result measures. That result is
*retrieval*: king - man + woman lands nearest to queen. Offset cosines in those
spaces are well short of 1; the analogy holds because the target wins the
nearest-neighbour race, not because the offsets are parallel. Scoring by cosine
alone measures the strictly harder quantity and understates the result -- the
sulfur ladder scores -0.288 by cosine and still retrieves at rank 1-2 everywhere.

Reads the diff vectors the analogy script already saved to
`data/diff_vectors_layer_{L}.npz` (19 non-alkane groups x chain lengths 3-6), so
this needs no rerun, no activations and no GPU.

Two details decide whether the numbers mean anything; both are handled below and
both are reported:

  1. The exclusion set is {src, plus, minus} - {predicted}, not {src, plus, minus}.
     The halogen quadruple reuses 'alkyl bromide' as both a1 and b2, so excluding
     all three source terms deletes the correct answer.

  2. Excluding the source terms is standard word2vec practice and is also the known
     reason those results flatter themselves. Without exclusion the top hit is
     simply the source group in roughly half of all trials here, so `rank_incl` and
     `degenerate_top1_rate` are emitted next to `rank_excl` rather than left out.
"""
import argparse
import csv
import json
import os
import statistics
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
# What 05_analogy writes, on the cluster and locally alike.
DEFAULT_ANALOGY = os.path.join(REPO, 'fc_group', 'Results', 'functional_group_analogy')
# Outputs follow the same shape as every other experiment here --
# Results/{experiment}/{model-slug}/{entity_type}/ with figures at the leaf and CSVs
# under data/ (cf. functional_group_analogy_carbon_matched.py:47, functional_group_probe.py:65).
DEFAULT_OUTPUT_ROOT = os.path.join(REPO, 'fc_group', 'Results',
                                   'functional_group_analogy_retrieval')

sys.path.insert(0, HERE)
# The quadruple list and its chemistry rationale live in the analogy script; import
# rather than restate so the two analyses can never drift apart.
from functional_group_analogy_carbon_matched import (  # noqa: E402
    ANALOGY_QUADRUPLES, cosine_similarity,
)
from model_registry import MODEL_CONFIGS, get_model_config  # noqa: E402

# The model list comes from the registry, not a local copy. models.sh exists because
# the probe, t-SNE, anisotropy and analogy jobs each used to carry their own two-model
# list, and forgetting to update one produced a sweep that was quietly narrower than
# intended rather than an error. Directory names use '-' while registry keys use '/',
# so both forms are kept: the output tree and --model speak the first, the registry
# lookup speaks the second.
SLASH_FOR = {m.replace('/', '-'): m for m in MODEL_CONFIGS}
MODELS = sorted(SLASH_FOR)
MODEL_8B = 'meta-llama-Llama-3.1-8B'   # only still named for the reference case


def load_diff_vectors(npz_path):
    """The saved diff vectors, as {group: {carbon_count: vec}}.

    Keys are written as f'{group}_{C}' by the analogy script, and group names
    themselves contain spaces but never underscores ('alkyl bromide'), so an
    rsplit on the last underscore recovers both fields.
    """
    with np.load(npz_path) as handle:
        diffs = {}
        for key in handle.files:
            group, carbon = key.rsplit('_', 1)
            diffs.setdefault(group, {})[int(carbon)] = handle[key]
    return diffs


def corner_predictions(pair_a, pair_b):
    """The four corner predictions of the analogy (a1 - a2) ~= (b1 - b2).

    Yields (predicted, src, plus, minus), to be read as
        target = vec[src] + vec[plus] - vec[minus]
    and the retrieval is a success when `predicted` tops the ranking.

    Testing all four corners rather than one is free here and quadruples the
    sample; it also exposes direction asymmetries, which is how the halogen
    quadruple's notation defect shows itself (two corners perfect, two at rank 15).
    """
    a1, a2 = pair_a
    b1, b2 = pair_b
    return [
        (b1, b2, a1, a2),   # b1 = b2 + (a1 - a2)
        (a1, a2, b1, b2),   # a1 = a2 + (b1 - b2)
        (b2, b1, a2, a1),   # b2 = b1 - (a1 - a2)
        (a2, a1, b2, b1),   # a2 = a1 - (b1 - b2)
    ]


def rank_of(vectors, target, predicted, exclude):
    """1-based rank of `predicted` when candidates are sorted by cosine to `target`."""
    ranked = sorted(
        (g for g in vectors if g not in exclude),
        key=lambda g: -cosine_similarity(vectors[g], target),
    )
    return ranked.index(predicted) + 1, ranked[0]


def run_trials(vectors, carbon_count):
    """Every corner prediction of every quadruple against one candidate pool.

    `vectors` is {group: vec} -- either the diff vectors at a single chain length
    (carbon-matched) or averaged over chain lengths (lumped). `carbon_count` is
    carried through to the output rows and is None for the lumped pool.
    """
    trials = []
    for pair_a, pair_b in RETRIEVAL_QUADRUPLES:
        missing = [g for g in (*pair_a, *pair_b) if g not in vectors]
        if missing:
            print(f"  skipping analogy {pair_a} vs {pair_b}: missing group(s) {missing}")
            continue
        for predicted, src, plus, minus in corner_predictions(pair_a, pair_b):
            target = vectors[src] + vectors[plus] - vectors[minus]
            # Subtracting the answer out of its own candidate pool would make the
            # halogen quadruple unscoreable, since 'alkyl bromide' is both a1 and b2.
            exclude = {src, plus, minus} - {predicted}
            rank_excl, _ = rank_of(vectors, target, predicted, exclude)
            rank_incl, top1_incl = rank_of(vectors, target, predicted, set())
            trials.append({
                'pair_a1': pair_a[0], 'pair_a2': pair_a[1],
                'pair_b1': pair_b[0], 'pair_b2': pair_b[1],
                'predicted_group': predicted,
                'source_group': src,
                'carbon_count': '' if carbon_count is None else carbon_count,
                'n_candidates_excl': len(vectors) - len(exclude),
                'rank_excl': rank_excl,
                'rank_incl': rank_incl,
                'hit1': int(rank_excl == 1),
                'hit2': int(rank_excl <= 2),
                'hit3': int(rank_excl <= 3),
                'cos_to_target': cosine_similarity(vectors[predicted], target),
                'top1_group_incl': top1_incl,
                # The classic word2vec failure mode: the offset is too small to
                # escape the source group's own neighbourhood, so the unexcluded
                # winner is just the term we started from.
                'degenerate': int(top1_incl == src and src != predicted),
            })
    return trials


def summarize(trials):
    """Aggregate metrics plus the random baselines they have to be read against."""
    ranks = [t['rank_excl'] for t in trials]
    # Candidate-pool size varies by one when a quadruple reuses a group, so the
    # baseline is averaged over the actual pools rather than assumed constant.
    pools = [t['n_candidates_excl'] for t in trials]
    return {
        'n_trials': len(trials),
        'hit1_rate': sum(t['hit1'] for t in trials) / len(trials),
        'hit2_rate': sum(t['hit2'] for t in trials) / len(trials),
        'hit3_rate': sum(t['hit3'] for t in trials) / len(trials),
        'mean_rank': statistics.mean(ranks),
        'median_rank': statistics.median(ranks),
        'mrr': statistics.mean(1.0 / r for r in ranks),
        'degenerate_top1_rate': sum(t['degenerate'] for t in trials) / len(trials),
        # One baseline per k: ordering the candidates at random puts the answer in the
        # top k with probability k/n, so hit@2 and hit@3 cannot be read against hit@1's.
        'random_hit1': statistics.mean(1.0 / n for n in pools),
        'random_hit2': statistics.mean(min(2.0, n) / n for n in pools),
        'random_hit3': statistics.mean(min(3.0, n) / n for n in pools),
        'random_mean_rank': statistics.mean((n + 1) / 2 for n in pools),
    }


def layers_present(model_dir):
    data_dir = os.path.join(model_dir, 'data')
    if not os.path.isdir(data_dir):
        return []
    layers = []
    for name in os.listdir(data_dir):
        if name.startswith('diff_vectors_layer_') and name.endswith('.npz'):
            layers.append(int(name[len('diff_vectors_layer_'):-len('.npz')]))
    return sorted(layers)


def quadruple_key(pair_a, pair_b):
    """How a quadruple is named in quad_rows, and keyed for styling."""
    return (f'{pair_a[0]} - {pair_a[1]}', f'{pair_b[0]} - {pair_b[1]}')


def distinct_by_group_set(quadruples):
    """One quadruple per set of four groups, plus what was collapsed into what.

    Retrieval is blind to how the four groups are paired. Every corner target is two
    of them minus the third, and the exclusion set is always the other three, so
    re-pairing {w,x,y,z} poses the SAME four questions with the sources relabelled --
    identical targets, answers, ranks and candidate pools, differing only in which
    group gets called the source.

    cos(a1-a2, b1-b2) *does* depend on the pairing, which is why
    ANALOGY_QUADRUPLES deliberately carries both pairings of two group sets as
    notation controls, and why this filter belongs here and not there. Without it the
    duplicated group sets would count twice in every pooled retrieval figure.
    """
    first, kept, collapsed = {}, [], []
    for pair_a, pair_b in quadruples:
        key = frozenset(pair_a + pair_b)
        if key in first:
            collapsed.append((quadruple_key(pair_a, pair_b), first[key]))
        else:
            first[key] = quadruple_key(pair_a, pair_b)
            kept.append((pair_a, pair_b))
    return kept, collapsed


# The cosine analysis keeps all of ANALOGY_QUADRUPLES; retrieval sees only the
# distinct ones. Reported at startup by main() rather than left to be inferred.
RETRIEVAL_QUADRUPLES, COLLAPSED_QUADRUPLES = distinct_by_group_set(ANALOGY_QUADRUPLES)


def collect(analogy_dir, entity_type, models):
    trial_rows, trend_rows, quad_rows, group_rows = [], [], [], []

    for model in models:
        model_dir = os.path.join(analogy_dir, model, entity_type)
        layers = layers_present(model_dir)
        if not layers:
            print(f"no diff_vectors_layer_*.npz under {model_dir} -- skipping {model}")
            continue
        # Depth as a fraction of the stack, so models of different depth share an axis.
        span = get_model_config(SLASH_FOR[model])['num_layers'] - 1

        for layer in layers:
            diffs = load_diff_vectors(
                os.path.join(model_dir, 'data', f'diff_vectors_layer_{layer}.npz'))
            carbons = sorted(set.intersection(*(set(by_c) for by_c in diffs.values())))

            by_mode = {
                # Primary: one candidate pool per chain length, so the retrieval never
                # compares a C3 diff vector against a C6 one.
                'carbon_matched': [
                    t for C in carbons
                    for t in run_trials({g: by_c[C] for g, by_c in diffs.items()}, C)
                ],
                # Secondary: chain lengths averaged first, matching the lumped/
                # carbon-matched split the analogy script already reports.
                'lumped': run_trials(
                    {g: np.mean(list(by_c.values()), axis=0) for g, by_c in diffs.items()},
                    None),
            }

            for mode, trials in by_mode.items():
                if not trials:
                    continue
                common = {'model': model, 'entity_type': entity_type,
                          'layer': layer, 'depth': round(layer / span, 4), 'mode': mode}
                for t in trials:
                    trial_rows.append({**common, **t})
                trend_rows.append({**common, **summarize(trials)})

                for pair_a, pair_b in RETRIEVAL_QUADRUPLES:
                    sub = [t for t in trials
                           if (t['pair_a1'], t['pair_a2']) == pair_a
                           and (t['pair_b1'], t['pair_b2']) == pair_b]
                    if sub:
                        label_a, label_b = quadruple_key(pair_a, pair_b)
                        quad_rows.append({
                            **common, 'pair_a': label_a, 'pair_b': label_b,
                            **summarize(sub),
                        })

                # Per functional group, pooled over whichever corners predict it.
                # Sampling is uneven -- a group that answers two different corners
                # gets twice the trials -- so n_trials rides along with every row
                # and a mean over 8 is never silently compared against one over 4.
                for group in sorted({t['predicted_group'] for t in trials}):
                    sub = [t for t in trials if t['predicted_group'] == group]
                    group_rows.append({**common, 'group': group, **summarize(sub)})

    return trial_rows, trend_rows, quad_rows, group_rows


def write_csv(path, rows, fieldnames=None):
    if not rows:
        print(f"no rows for {os.path.relpath(path, REPO)} -- not written")
        return
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {os.path.relpath(path, REPO)}  ({len(rows)} rows)")


# ============================
# Self-checks
# ============================

N_CORNERS = 4
TOL = 1e-9

# The reference case. These track the diff vectors under --analogy-dir *and*
# ANALOGY_QUADRUPLES:
# adding or removing a quadruple changes the trial count and every pooled number with it.
# Regenerate them deliberately when either changes; do not nudge them to make a run pass.
REFERENCE_MODEL = 'meta-llama-Llama-3.1-8B'
REFERENCE_ENTITY = 'functional_group'
REFERENCE_HIT1 = {0: 34, 16: 57, 31: 59}             # out of 80 carbon-matched trials
REFERENCE_MEAN_RANK = {0: 3.79, 16: 2.17, 31: 1.35}  # rounded to 2dp
REFERENCE_PERFECT_OS = [16, 31]  # layers where thioether-thiol ~ ether-alcohol is 16/16

INVARIANT_HEADER = 'FAILED SELF-CHECK'


def quadruple_of(row):
    return (row['pair_a1'], row['pair_a2'], row['pair_b1'], row['pair_b2'])


def recompute(rows):
    """The aggregates a set of trials must produce.

    Deliberately NOT shared with summarize(). One produces, the other checks, and
    folding them together would make the aggregates-match-their-rows invariant
    vacuous -- so resist the DRY refactor this near-duplicate invites.

    Be clear about what the duplication buys, though: it catches the wrong rows
    reaching summarize(), rows mutated between summarizing and writing, and a typo in
    one copy. It does NOT catch a concept that is wrong in both.
    """
    ranks = [r['rank_excl'] for r in rows]
    return {
        'n_trials': len(rows),
        'hit1_rate': statistics.mean(r['hit1'] for r in rows),
        'hit2_rate': statistics.mean(r['hit2'] for r in rows),
        'hit3_rate': statistics.mean(r['hit3'] for r in rows),
        'mean_rank': statistics.mean(ranks),
        'median_rank': statistics.median(ranks),
        'mrr': statistics.mean(1.0 / r for r in ranks),
        'degenerate_top1_rate': statistics.mean(r['degenerate'] for r in rows),
    }


def check_structure(trials, trend, n_groups):
    """Invariants that hold for any model and any entity_type.

    Both bugs this analysis has had were silent-wrong-answer bugs rather than crashes
    -- a corner key that collided so one trial overwrote another, and an exclusion set
    that deleted the correct answer from its own candidate pool. Neither changed the
    printed table. These are the assertions that would have caught them.
    """
    problems = []
    by_key = {}
    for row in trials:
        by_key.setdefault((row['layer'], row['mode']), []).append(row)

    for (layer, mode), rows in sorted(by_key.items()):
        where = f"L{layer}/{mode}"
        carbons = {r['carbon_count'] for r in rows}
        n_carbons = len(carbons) if mode == 'carbon_matched' else 1
        if mode == 'lumped' and carbons != {''}:
            problems.append(f"{where}: lumped trials carry a carbon_count: {sorted(carbons)}")

        # 1. every quadruple contributes every corner at every chain length
        expected = len(RETRIEVAL_QUADRUPLES) * N_CORNERS * n_carbons
        if len(rows) != expected:
            problems.append(
                f"{where}: {len(rows)} trials, expected "
                f"{len(RETRIEVAL_QUADRUPLES)}x{N_CORNERS}x{n_carbons} = {expected}")

        # 2. every trial must be uniquely addressable. (quadruple, predicted) alone
        #    collides for quadruples that reuse a group -- the 14-rows-for-16-trials
        #    heatmap bug -- so the source term is part of the key. chain length is in
        #    the key too: without it a collision confined to one chain length still
        #    leaves the key SET the right size and slips through.
        keys = {(*quadruple_of(r), r['source_group'], r['predicted_group'],
                 r['carbon_count']) for r in rows}
        if len(keys) != len(rows):
            problems.append(
                f"{where}: {len(rows)} trials but only {len(keys)} unique "
                f"(quadruple, source, predicted, chain length) keys -- corner keys "
                f"collide, so trials are overwriting each other")

        # 3. no quadruple silently dropped by run_trials' `missing` branch
        quads = {quadruple_of(r) for r in rows}
        if len(quads) != len(RETRIEVAL_QUADRUPLES):
            problems.append(
                f"{where}: {len(quads)} quadruples present, expected "
                f"{len(RETRIEVAL_QUADRUPLES)} -- a quadruple was skipped as unscoreable")

        # 10. no two quadruples may pose the same retrieval problem. Two pairings of
        #     one group set produce identical targets, answers and ranks, differing
        #     only in the source label -- so counting both silently doubles that
        #     group set's weight in every pooled figure. This is the check that the
        #     first seven-quadruple run needed and did not have.
        by_groups = {}
        for quad in quads:
            by_groups.setdefault(frozenset(quad), []).append(quad)
        for gs, dupes in sorted(by_groups.items(), key=lambda kv: sorted(kv[0])):
            if len(dupes) > 1:
                shown = '; '.join(f'{q[0]}-{q[1]} ~ {q[2]}-{q[3]}' for q in sorted(dupes))
                problems.append(
                    f"{where}: {len(dupes)} quadruples share the group set "
                    f"{sorted(gs)} -- {shown}. Retrieval cannot tell them apart, so "
                    f"they would count twice; see distinct_by_group_set()")

        for quad in sorted(quads):
            qrows = [r for r in rows if quadruple_of(r) == quad]
            # 4. the exclusion set is {src, plus, minus} - {predicted}, so a quadruple
            #    built from 3 distinct groups excludes 2 and one from 4 excludes 3
            n_excluded = 2 if len(set(quad)) < 4 else 3
            want = n_groups - n_excluded
            bad = sorted({r['n_candidates_excl'] for r in qrows} - {want})
            if bad:
                problems.append(
                    f"{where}: {quad[0]}-{quad[1]} ~ {quad[2]}-{quad[3]} "
                    f"({len(set(quad))} distinct groups) has candidate pool(s) {bad}, "
                    f"expected {want} = {n_groups} - {n_excluded}")

        # 5. each row internally consistent
        for r in rows:
            pool, rank_excl, rank_incl = (r['n_candidates_excl'],
                                          r['rank_excl'], r['rank_incl'])
            tag = f"{where} {r['source_group']}->{r['predicted_group']}"
            if not 1 <= rank_excl <= pool:
                problems.append(f"{tag}: rank_excl {rank_excl} outside [1, {pool}]")
            if not 1 <= rank_incl <= n_groups:
                problems.append(f"{tag}: rank_incl {rank_incl} outside [1, {n_groups}]")
            for k in (1, 2, 3):
                if r[f'hit{k}'] != int(rank_excl <= k):
                    problems.append(
                        f"{tag}: hit{k}={r[f'hit{k}']} but rank_excl={rank_excl}")
            cos = r['cos_to_target']
            if not np.isfinite(cos) or not -1.0 <= cos <= 1.0:
                problems.append(f"{tag}: cos_to_target={cos} is not a finite cosine")

    for srow in trend:
        key = (srow['layer'], srow['mode'])
        rows = by_key.get(key)
        if rows is None:
            problems.append(f"L{key[0]}/{key[1]}: summarized but has no trials")
            continue

        # 6. the reported aggregates must summarize the rows they came from
        for field, want in recompute(rows).items():
            if abs(srow[field] - want) > TOL:
                problems.append(
                    f"L{key[0]}/{key[1]}: {field} reported {srow[field]!r} but the "
                    f"trials give {want!r} -- the summary does not match its own rows")

        # 8. the random baselines are k/n averaged over the ACTUAL candidate pools;
        #    pool size is not constant, so they cannot be checked against a literal
        pools = [r['n_candidates_excl'] for r in rows]
        for field, want in {
            'random_hit1': statistics.mean(1.0 / n for n in pools),
            'random_hit2': statistics.mean(min(2.0, n) / n for n in pools),
            'random_hit3': statistics.mean(min(3.0, n) / n for n in pools),
            'random_mean_rank': statistics.mean((n + 1) / 2 for n in pools),
        }.items():
            if field not in srow:
                problems.append(f"L{key[0]}/{key[1]}: missing '{field}'")
            elif abs(srow[field] - want) > TOL:
                problems.append(
                    f"L{key[0]}/{key[1]}: {field} reported {srow[field]!r} but the "
                    f"candidate pools give {want!r}")
    return problems


def check_groups(trials, group_rows):
    """7. The per-group rows must partition the trials exactly.

    Groups are unevenly sampled -- one that answers two different corners gets twice
    the trials -- so this checks the counts add up rather than assuming they are equal.
    """
    problems = []
    by_key = {}
    for row in trials:
        by_key.setdefault((row['layer'], row['mode']), []).append(row)

    seen = {}
    for grow in group_rows:
        key = (grow['layer'], grow['mode'])
        rows = by_key.get(key, [])
        sub = [r for r in rows if r['predicted_group'] == grow['group']]
        if not sub:
            problems.append(
                f"L{key[0]}/{key[1]}: group '{grow['group']}' is summarized but is "
                f"never a predicted_group in the trials")
            continue
        seen[key] = seen.get(key, 0) + len(sub)
        for field, want in recompute(sub).items():
            if abs(grow[field] - want) > TOL:
                problems.append(
                    f"L{key[0]}/{key[1]} group '{grow['group']}': {field} reported "
                    f"{grow[field]!r} but its trials give {want!r}")

    for key, rows in by_key.items():
        if key in seen and seen[key] != len(rows):
            problems.append(
                f"L{key[0]}/{key[1]}: the by-group rows cover {seen[key]} trials but "
                f"{len(rows)} exist -- the groups do not partition the trials")
    return problems


def check_reference(trials, trend):
    """9. Fixed regression on the one case whose numbers are published."""
    problems = []
    note = ("(these depend on the diff vectors under --analogy-dir and on "
            "ANALOGY_QUADRUPLES. Pointing at a different extraction -- the single-template "
            "smoke activations rather than the full sweep, say -- moves them legitimately, "
            "as does changing the quadruple set. Update the constants deliberately in that "
            "case rather than hunting a regression.)")

    for srow in trend:
        if srow['mode'] != 'carbon_matched' or srow['layer'] not in REFERENCE_HIT1:
            continue
        layer, n = srow['layer'], srow['n_trials']
        got = round(srow['hit1_rate'] * n)
        if got != REFERENCE_HIT1[layer]:
            problems.append(f"L{layer}: hit@1 {got}/{n}, expected "
                            f"{REFERENCE_HIT1[layer]}/{n}  {note}")
        got_rank = round(srow['mean_rank'], 2)
        if abs(got_rank - REFERENCE_MEAN_RANK[layer]) > TOL:
            problems.append(f"L{layer}: mean rank {got_rank}, expected "
                            f"{REFERENCE_MEAN_RANK[layer]}  {note}")

    for layer in REFERENCE_PERFECT_OS:
        rows = [r for r in trials
                if r['mode'] == 'carbon_matched' and r['layer'] == layer
                and quadruple_of(r) == ('thioether', 'thiol', 'ether', 'alcohol')]
        if not rows:
            problems.append(f"L{layer}: no thioether-thiol ~ ether-alcohol trials found")
        elif sum(r['hit1'] for r in rows) != len(rows):
            problems.append(
                f"L{layer}: O<->S retrieves {sum(r['hit1'] for r in rows)}/{len(rows)}, "
                f"expected {len(rows)}/{len(rows)}  {note}")
    return problems


def check_invariants(rows, model, entity_type):
    """Raise unless the trials, the summaries and each other all agree.

    Runs before anything is written, so a run that fails this writes nothing rather
    than leaving a plausible-looking but wrong results tree on disk.
    """
    trial_rows, trend_rows, quad_rows, group_rows = rows
    # Derived, not hardcoded: quadruples of 4 distinct groups exclude 3, and at least
    # one such quadruple always exists, so the smallest pool is n_groups - 3.
    n_groups = min(r['n_candidates_excl'] for r in trial_rows) + 3

    problems = check_structure(trial_rows, trend_rows, n_groups)
    problems += check_groups(trial_rows, group_rows)
    if model == REFERENCE_MODEL and entity_type == REFERENCE_ENTITY:
        problems += check_reference(trial_rows, trend_rows)

    if problems:
        raise SystemExit(
            f"{INVARIANT_HEADER} {model} / {entity_type}:\n"
            + "\n".join(f"  {p}" for p in problems))
    return n_groups


# ============================
# Plots
# ============================

# Solid lines distinguished by colour and marker -- never by dash pattern. At the font
# sizes these legends use, '--' and '-.' are indistinguishable from '-' in the legend
# swatch, so a dash-encoded series is mislabelled in practice even when the code is right.
# Dotted is reserved for the random baselines, which are references rather than series.
HIT_SERIES = [
    ('hit1_rate', 'random_hit1', 'hit@1', '#2c3e50', 'o'),
    ('hit2_rate', 'random_hit2', 'hit@2', '#2980b9', 's'),
    ('hit3_rate', 'random_hit3', 'hit@3', '#16a085', '^'),
]
GROUP_MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>', 'h', 'p']
# One entry per quadruple, not a cycling palette -- indexing with % would silently
# give quadruples 5-7 the colours of 1-3 and break the guarantee that a quadruple looks
# the same in every figure. Asserted below rather than left to trust.
QUADRUPLE_PALETTE = ['#c0392b', '#2980b9', '#16a085', '#8e44ad',
                     '#e67e22', '#7f8c8d', '#d81b60']
QUADRUPLE_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
assert len(QUADRUPLE_PALETTE) >= len(RETRIEVAL_QUADRUPLES), (
    f"{len(RETRIEVAL_QUADRUPLES)} quadruples but only {len(QUADRUPLE_PALETTE)} colours -- "
    f"add entries to QUADRUPLE_PALETTE and QUADRUPLE_MARKERS")
assert len(QUADRUPLE_MARKERS) >= len(RETRIEVAL_QUADRUPLES), "add entries to QUADRUPLE_MARKERS"
assert len(set(QUADRUPLE_PALETTE[:len(RETRIEVAL_QUADRUPLES)])) == len(RETRIEVAL_QUADRUPLES), (
    "QUADRUPLE_PALETTE has duplicate colours in the range actually used")
# Keyed off RETRIEVAL_QUADRUPLES order, so a quadruple keeps one colour across every
# figure that draws it -- the accuracy-by-quadruple and rank-by-quadruple plots are
# meant to be read side by side.
QUADRUPLE_STYLE = {
    quadruple_key(pa, pb): (QUADRUPLE_PALETTE[i], QUADRUPLE_MARKERS[i])
    for i, (pa, pb) in enumerate(RETRIEVAL_QUADRUPLES)
}


def ordered_pairs(rows):
    """Quadruples in RETRIEVAL_QUADRUPLES order, so colour and legend order are stable
    across figures rather than following whatever the row set happens to sort to."""
    return sorted({(r['pair_a'], r['pair_b']) for r in rows},
                  key=lambda k: list(QUADRUPLE_STYLE).index(k)
                  if k in QUADRUPLE_STYLE else len(QUADRUPLE_STYLE))


def carbon_matched(rows, model):
    """The carbon-matched rows for one model, in layer order."""
    return sorted((r for r in rows if r['model'] == model and r['mode'] == 'carbon_matched'),
                  key=lambda r: r['layer'])


def save(fig, path, suptitle):
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, REPO)}")


def plot_accuracy_pooled(trend_rows, model, entity_type, path):
    """hit@1/@2/@3 pooled over all four quadruples, each against its own baseline."""
    rows = carbon_matched(trend_rows, model)
    if not rows:
        return
    depths = [r['depth'] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for field, base_field, label, color, marker in HIT_SERIES:
        base = statistics.mean(r[base_field] for r in rows)
        ax.plot(depths, [r[field] for r in rows], marker=marker, lw=2.2, color=color,
                label=f'{label}   (random baseline {base:.1%})')
        ax.axhline(base, color=color, ls=':', lw=1, alpha=0.55)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('relative depth')
    ax.set_ylabel('retrieval accuracy (source terms excluded)')
    ax.legend(fontsize=8, handlelength=2.6)
    save(fig, path, f'{model} -- {entity_type}\n'
                    f'Correct group in the top k, all {len(RETRIEVAL_QUADRUPLES)} quadruples pooled')


def plot_accuracy_by_quadruple(quad_rows, model, entity_type, series, path):
    """One hit@k, one file: four quadruple lines against that k's own baseline."""
    field, base_field, label, _, _ = series
    rows = carbon_matched(quad_rows, model)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for pair in ordered_pairs(rows):
        prows = [r for r in rows if (r['pair_a'], r['pair_b']) == pair]
        color, marker = QUADRUPLE_STYLE.get(pair, ('#7f8c8d', 'o'))
        ax.plot([r['depth'] for r in prows], [r[field] for r in prows],
                marker=marker, lw=2, color=color, label=f'{pair[0]}  ~  {pair[1]}')
    base = statistics.mean(r[base_field] for r in rows)
    ax.axhline(base, color='gray', ls=':', lw=1,
               label=f'random baseline ({base:.1%})')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('relative depth')
    ax.set_ylabel(label)
    # These curves rise with depth, so the top-left is where the strongest quadruple
    # lives and a legend there would cover it.
    ax.legend(fontsize=6, handlelength=2.6, loc='lower right')
    save(fig, path, f'{model} -- {entity_type}\n{label} per quadruple')


def plot_rank_by_quadruple(trend_rows, quad_rows, model, entity_type, path):
    """Mean rank of the correct group, one line per quadruple plus the overall mean."""
    rows = carbon_matched(quad_rows, model)
    pooled = carbon_matched(trend_rows, model)
    if not rows or not pooled:
        return
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for pair in ordered_pairs(rows):
        prows = [r for r in rows if (r['pair_a'], r['pair_b']) == pair]
        color, marker = QUADRUPLE_STYLE.get(pair, ('#7f8c8d', 'o'))
        ax.plot([r['depth'] for r in prows], [r['mean_rank'] for r in prows],
                marker=marker, lw=2, color=color, label=f'{pair[0]}  ~  {pair[1]}')
    # No pooled mean line here: averaging four quadruples that disagree by 6 rank
    # positions describes none of them. `pooled` is still read, for the baseline.
    base = statistics.mean(r['random_mean_rank'] for r in pooled)
    ax.axhline(base, color='gray', ls=':', lw=1, label=f'random baseline ({base:.2f})')
    # One axis per figure, so this inverts exactly once: rank 1 on top, better is up.
    ax.invert_yaxis()
    ax.set_xlabel('relative depth')
    ax.set_ylabel('mean rank of correct group (rank 1 = best)')
    ax.legend(fontsize=6, handlelength=2.6)
    save(fig, path, f'{model} -- {entity_type}\nRank of the correct group, per quadruple')


def plot_rank_by_group(trend_rows, group_rows, model, entity_type, path):
    """Mean rank broken out by the functional group being predicted."""
    rows = carbon_matched(group_rows, model)
    pooled = carbon_matched(trend_rows, model)
    if not rows or not pooled:
        return
    groups = sorted({r['group'] for r in rows})
    cmap = plt.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for i, group in enumerate(groups):
        grows = [r for r in rows if r['group'] == group]
        ax.plot([r['depth'] for r in grows], [r['mean_rank'] for r in grows],
                marker=GROUP_MARKERS[i % len(GROUP_MARKERS)], ms=4, lw=1.2, alpha=0.85,
                color=cmap(i % 20), label=f"{group} (n={grows[0]['n_trials']})")
    ax.plot([r['depth'] for r in pooled], [r['mean_rank'] for r in pooled],
            marker='o', lw=3.2, color='#c0392b', label='overall mean', zorder=5)
    base = statistics.mean(r['random_mean_rank'] for r in pooled)
    ax.axhline(base, color='gray', ls=':', lw=1, label=f'random baseline ({base:.2f})')
    ax.invert_yaxis()
    ax.set_xlabel('relative depth')
    ax.set_ylabel('mean rank of correct group (rank 1 = best)')
    ax.legend(fontsize=6, ncol=2, handlelength=2.6,
              loc='center left', bbox_to_anchor=(1.01, 0.5))
    save(fig, path, f'{model} -- {entity_type}\n'
                    f'Rank of the correct group, per functional group')


def plot_rank_heatmap(trial_rows, model, entity_type, path):
    """Rank grid, quadruple x corner x chain length, one panel per layer."""
    rows = [t for t in trial_rows if t['model'] == model and t['mode'] == 'carbon_matched']
    layers = sorted({t['layer'] for t in rows})
    if not layers:
        return
    # The corner must be keyed by its source term as well as its answer. Two of the
    # four corners of a quadruple that reuses a group (halogen: 'alkyl bromide' is
    # both a1 and b2; sulfur: 'sulfoxide' is both a1 and b2) predict the *same*
    # group, so keying on (quadruple, predicted) alone collapses them and one trial
    # silently overwrites the other -- 14 rows drawn for 16 trials.
    labels = sorted({(t['pair_a1'], t['pair_a2'], t['pair_b1'], t['pair_b2'],
                      t['source_group'], t['predicted_group']) for t in rows})
    carbons = sorted({t['carbon_count'] for t in rows})

    fig, axes = plt.subplots(1, len(layers),
                             figsize=(2.6 * len(layers) + 4, 0.32 * len(labels) + 2),
                             squeeze=False, sharey=True)
    for col, layer in enumerate(layers):
        grid = np.full((len(labels), len(carbons)), np.nan)
        for t in rows:
            if t['layer'] != layer:
                continue
            key = (t['pair_a1'], t['pair_a2'], t['pair_b1'], t['pair_b2'],
                   t['source_group'], t['predicted_group'])
            grid[labels.index(key), carbons.index(t['carbon_count'])] = t['rank_excl']
        ax = axes[0][col]
        im = ax.imshow(grid, cmap='RdYlGn_r', vmin=1, vmax=max(3, np.nanmax(grid)),
                       aspect='auto')
        for i in range(len(labels)):
            for j in range(len(carbons)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, int(grid[i, j]), ha='center', va='center', fontsize=6)
        ax.set_xticks(range(len(carbons)))
        ax.set_xticklabels([f'C{c}' for c in carbons], fontsize=7)
        ax.set_title(f'layer {layer}', fontsize=9)
        if col == 0:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(
                [f'{a1[:11]}-{a2[:11]} ~ {b1[:11]}-{b2[:11]}  {src[:13]}->{p[:13]}'
                 for a1, a2, b1, b2, src, p in labels], fontsize=6)
    fig.colorbar(im, ax=axes[0].tolist(), label='rank of correct group', shrink=0.7)
    fig.suptitle(f'{model} -- {entity_type}: retrieval rank per corner prediction',
                 fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, REPO)}")


def write_model_outputs(out_dir, entity_type, model, rows, make_plots):
    """Everything for one model x entity_type, laid out the way Results/ expects."""
    trial_rows, trend_rows, quad_rows, group_rows = rows
    os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)
    for name, table in [('retrieval_trials', trial_rows),
                        ('retrieval_layer_trend', trend_rows),
                        ('retrieval_by_quadruple', quad_rows),
                        ('retrieval_by_group', group_rows)]:
        write_csv(os.path.join(out_dir, 'data', f'{name}.csv'), table)
    with open(os.path.join(out_dir, 'data', 'retrieval_summary.json'), 'w') as fh:
        json.dump({'model': model, 'entity_type': entity_type,
                   'layer_trend': trend_rows, 'by_quadruple': quad_rows,
                   'by_group': group_rows}, fh, indent=2)
    if not make_plots:
        return
    plot_accuracy_pooled(trend_rows, model, entity_type,
                         os.path.join(out_dir, 'retrieval_accuracy_pooled.png'))
    for series in HIT_SERIES:
        k = series[0].split('_')[0]          # 'hit1_rate' -> 'hit1'
        plot_accuracy_by_quadruple(
            quad_rows, model, entity_type, series,
            os.path.join(out_dir, f'retrieval_accuracy_{k}_by_quadruple.png'))
    plot_rank_by_quadruple(trend_rows, quad_rows, model, entity_type,
                           os.path.join(out_dir, 'retrieval_rank_by_quadruple.png'))
    plot_rank_by_group(trend_rows, group_rows, model, entity_type,
                       os.path.join(out_dir, 'retrieval_rank_by_group.png'))
    plot_rank_heatmap(trial_rows, model, entity_type,
                      os.path.join(out_dir, 'retrieval_rank_heatmap.png'))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--analogy-dir', default=DEFAULT_ANALOGY)
    p.add_argument('--output-dir', default=DEFAULT_OUTPUT_ROOT,
                   help="Root of the output tree. A run sweeps several models, so this is "
                        "the root and '{model}/{entity_type}/' is appended, rather than "
                        "being the leaf directory itself. Default: %(default)s")
    p.add_argument('--entity-type', default='functional_group',
                   help="prompt type, matching a directory under --analogy-dir")
    # --model-name, matching the other scripts at fc_group/ (the README documents it as
    # a common flag). Either naming form is accepted: models.sh and the registry speak
    # 'meta-llama/Llama-3.1-8B', the output tree speaks 'meta-llama-Llama-3.1-8B'.
    p.add_argument('--model-name', action='append', default=None,
                   help="Repeatable. Accepts either naming form. "
                        "Default: every model in fc_group/model_registry.py")
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--skip-checks', action='store_true',
                   help="Skip the self-checks. They are cheap and they are what catches "
                        "the silent-wrong-answer bugs this analysis has had, so this is "
                        "an escape hatch, not a normal flag.")
    args = p.parse_args()

    # Say this once, up front: a reader comparing a 5-quadruple retrieval run against a
    # 7-quadruple cosine run should not have to infer the difference from trial counts.
    if COLLAPSED_QUADRUPLES:
        print(f"collapsed {len(COLLAPSED_QUADRUPLES)} quadruple(s) posing duplicate "
              f"retrieval problems (same four groups, different pairing):")
        for dup, kept in COLLAPSED_QUADRUPLES:
            print(f"  {dup[0] + '  ~  ' + dup[1]:46s} -> {kept[0]}  ~  {kept[1]}")

    models = [m.replace('/', '-') for m in (args.model_name or MODELS)]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit(
            f"unknown model(s) {unknown}; known: {MODELS}\n"
            f"  (add it to fc_group/model_registry.py first)")
    trial_rows, trend_rows, quad_rows, group_rows = collect(
        args.analogy_dir, args.entity_type, models)
    if not trend_rows:
        raise SystemExit(f"nothing to report for entity_type={args.entity_type!r}")

    for model in sorted({r['model'] for r in trend_rows}):
        rows = tuple([r for r in table if r['model'] == model]
                     for table in (trial_rows, trend_rows, quad_rows, group_rows))
        # Checked before anything is written, so a run that fails leaves no
        # plausible-looking but wrong results tree behind.
        if not args.skip_checks:
            n_groups = check_invariants(rows, model, args.entity_type)
            n_cm = sum(1 for r in rows[0] if r['mode'] == 'carbon_matched'
                       and r['layer'] == min(x['layer'] for x in rows[0]))
            print(f"self-check ok  {model} / {args.entity_type}: "
                  f"{len(RETRIEVAL_QUADRUPLES)} quadruples, {n_cm} carbon-matched trials "
                  f"per layer over {n_groups} candidate groups")
        # entity_type is used raw as a directory name, spaces and all, matching the
        # Results tree this reads from ('functional_group question' et al).
        out_dir = os.path.join(args.output_dir, model, args.entity_type)
        write_model_outputs(out_dir, args.entity_type, model, rows, not args.no_plots)

    print(f"\n=== {args.entity_type}: retrieval vs random baseline (carbon-matched) ===")
    print(f"{'model':>4s} {'L':>4s} {'depth':>6s} {'n':>4s} {'hit@1':>7s} {'hit@2':>7s} "
          f"{'hit@3':>7s} {'rand@1':>8s} {'mean rk':>8s} {'rand rk':>8s} {'MRR':>6s} "
          f"{'degen':>7s}")
    for r in sorted((r for r in trend_rows if r['mode'] == 'carbon_matched'),
                    key=lambda r: (r['model'], r['layer'])):
        print(f"{r['model'][-3:]:>4s} {r['layer']:4d} {r['depth']:6.2f} {r['n_trials']:4d} "
              f"{r['hit1_rate']:6.1%} {r['hit2_rate']:6.1%} {r['hit3_rate']:6.1%} "
              f"{r['random_hit1']:7.1%} {r['mean_rank']:8.2f} {r['random_mean_rank']:8.2f} "
              f"{r['mrr']:6.3f} {r['degenerate_top1_rate']:6.0%}")

    print(f"\n=== per quadruple: hit@1 by layer (carbon-matched) ===")
    for model in sorted({r['model'] for r in quad_rows}):
        layers = sorted({q['layer'] for q in quad_rows if q['model'] == model})
        print(f"  {model}")
        print(f"    {'quadruple':46s} " + ' '.join(f'L{L:<5d}' for L in layers))
        for pair in sorted({(q['pair_a'], q['pair_b']) for q in quad_rows}):
            cells = []
            for L in layers:
                match = [q for q in quad_rows if q['model'] == model and q['layer'] == L
                         and q['mode'] == 'carbon_matched'
                         and (q['pair_a'], q['pair_b']) == pair]
                cells.append(f"{match[0]['hit1_rate']:5.0%} " if match else '   -- ')
            print(f"    {pair[0] + '  ~  ' + pair[1]:46s} " + ' '.join(cells))

    print("\nNote: hit@k excludes the three source terms, as word2vec evaluation does. "
          "`degen` is how often the\nunexcluded nearest neighbour is just the source "
          "group -- the offset failing to escape its own neighbourhood.")


if __name__ == '__main__':
    main()
