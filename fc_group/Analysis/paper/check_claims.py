#!/usr/bin/env python3
"""Assert the numbers the paper's argument rests on, against the released CSVs.

    python fc_group/Analysis/paper/check_claims.py

A companion to `check_numbers.py`, which asks whether every numeral in the prose has *some*
source. This asks the sharper question: do the specific values the argument depends on still
hold after a pipeline change? Re-run it after touching any `analyze_*.py` -- the geometry and
probe scripts both feed several of these, and a refactor that silently shifts one would
otherwise be caught only by a reader.

Exit status is 1 on any failure, so this can gate a submission.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))

BASE = 'meta-llama-Llama-3.1-8B'
CHEM = 'phenixace-Chem-R-Faithful'
R1 = 'deepseek-ai-DeepSeek-R1-Distill-Llama-8B'
NAME = {BASE: 'base', CHEM: 'chem', R1: 'reason'}
LAYER = 31


def rows(rel):
    path = os.path.join(ANALYSIS, rel)
    if not os.path.exists(path):
        raise SystemExit(f'missing {rel} -- run the analyze_*.py that writes it')
    with open(path) as fh:
        return list(csv.DictReader(fh))


class Checks:
    def __init__(self):
        self.results = []

    def eq(self, label, got, want, tol=5e-3):
        ok = abs(float(got) - float(want)) <= tol
        self.results.append((label, float(got), float(want), ok))

    def true(self, label, ok, detail=''):
        self.results.append((label, detail or ('yes' if ok else 'no'), 'yes', bool(ok)))

    def report(self):
        width = max(len(r[0]) for r in self.results)
        for label, got, want, ok in self.results:
            g = f'{got:.6g}' if isinstance(got, float) else got
            w = f'{want:.6g}' if isinstance(want, float) else want
            print(f"{'PASS' if ok else 'FAIL':<5} {label:<{width}}  got {g:>10}  want {w}")
        bad = [r for r in self.results if not r[3]]
        print(f'\n{len(self.results) - len(bad)}/{len(self.results)} checks passed')
        return 1 if bad else 0


def main():
    c = Checks()

    # --- the paradox: both halves, both intervals, opposite signs ---------------------
    gen = {r['model']: r for r in rows('generation/data/generation_summary.csv')
           if r['scoring'] == 'raw'}
    for m, want in ((BASE, 0.837), (CHEM, 0.978), (R1, 0.804)):
        c.eq(f'generation accuracy {NAME[m]}', gen[m]['accuracy'], want)
    gd = [r for r in rows('generation/data/generation_diff_bootstrap.csv')
          if r['scoring'] == 'raw'][0]
    pd_ = rows('probe/data/model_diff_bootstrap.csv')[0]
    c.eq('generation chem-base diff', gd['diff_a_minus_b'], 0.141)
    c.eq('generation CI low', gd['ci_lo'], 0.097)
    c.eq('probe chem-base diff', pd_['diff_a_minus_b'], -0.017, 1e-3)
    c.eq('probe CI high', pd_['ci_hi'], -0.006, 1e-3)
    c.true('the two differences have opposite signs',
           float(gd['diff_a_minus_b']) * float(pd_['diff_a_minus_b']) < 0)
    c.true('both CIs exclude zero',
           int(gd['ci_excludes_zero']) and int(pd_['ci_excludes_zero']))

    # --- free generation: a third ordering, and the format effect it isolates ---------
    # Table 1's forced choice ranks chem > base > reason. Adjudicated free generation ranks
    # chem > reason > base, because the base model's knowledge does not survive contact
    # with having to produce the answer. Guarded here so the two orderings cannot silently
    # converge if the parser is ever loosened.
    fg = {r['model']: r for r in rows('generation/data/free_generation_summary.csv')}
    for m, want in ((BASE, 0.585), (CHEM, 0.825), (R1, 0.789)):
        c.eq(f'free-generation strict accuracy {NAME[m]}', fg[m]['strict_accuracy'], want)
    c.true('free generation ranks reason above base, forced choice does not',
           float(fg[R1]['strict_accuracy']) > float(fg[BASE]['strict_accuracy'])
           and float(gen[R1]['accuracy']) < float(gen[BASE]['accuracy']))
    # The reason the metric adjudicates rather than string-matches: containment inflates
    # the base model by ~6.5 points and the other two by under 1.5.
    c.eq('base containment minus strict',
         float(fg[BASE]['containment_accuracy']) - float(fg[BASE]['strict_accuracy']),
         0.065, 1e-3)
    c.true('containment inflates base far more than either fine-tune',
           all(float(fg[BASE]['containment_accuracy']) - float(fg[BASE]['strict_accuracy'])
               > 3 * (float(fg[m]['containment_accuracy']) - float(fg[m]['strict_accuracy']))
               for m in (CHEM, R1)))
    # Format collapse is the base model's alone: it enumerates options on 13% of prompts,
    # the fine-tunes on under 2%.
    c.eq(f'malformed rate {NAME[BASE]}', fg[BASE]['malformed_rate'], 0.132, 1e-3)
    c.true('both fine-tunes are malformed on under 2% of prompts',
           all(float(fg[m]['malformed_rate']) < 0.02 for m in (CHEM, R1)))
    # The five response types partition the prompts -- no row is counted twice or dropped.
    # The exact equality is asserted on integer counts in score_generations.summarize; this
    # re-checks it on the published rates, so the tolerance only has to absorb the 6-decimal
    # rounding of five summed columns.
    for m in (BASE, CHEM, R1):
        total = sum(float(fg[m][k]) for k in (
            'strict_accuracy', 'answer_wrong_rate', 'underspecified_rate',
            'non_answer_rate', 'malformed_rate'))
        c.eq(f'response types partition {NAME[m]}', total, 1.0, 1e-5)
    fgd = rows('generation/data/free_generation_diff_bootstrap.csv')[0]
    c.eq('free-generation chem-base diff', fgd['diff_a_minus_b'], 0.240, 1e-3)
    c.true('free-generation chem-base CI excludes zero', int(fgd['ci_excludes_zero']))

    # --- Aim 1: the chemical blocks are built by the stack, not present at input -------
    # The halide block is the control: it is lexically marked (every member is named
    # "alkyl <halogen>ide"), so it should already be cohesive at the embedding layer. The
    # other three are not lexically marked, so if they are cohesive at depth the stack put
    # them there. Nitrogen starting BELOW zero is the sharpest form of that claim.
    coh = {(r['model'], r['block'], int(r['layer'])): float(r['cohesion'])
           for r in rows('geometry/data/block_cohesion_by_layer.csv')}
    c.eq('halide cohesion base L0 (lexically marked, cohesive at input)',
         coh[(BASE, 'halide', 0)], 1.761)
    c.eq('nitrogen cohesion base L0 (anti-cohesive at input)',
         coh[(BASE, 'nitrogen', 0)], -0.417)
    c.eq('nitrogen cohesion base L31', coh[(BASE, 'nitrogen', LAYER)], 1.567)
    c.eq('sulfur cohesion base L31', coh[(BASE, 'sulfur', LAYER)], 2.229)
    c.eq('carbonyl cohesion base L31', coh[(BASE, 'carbonyl', LAYER)], 1.714)
    c.true('nitrogen block rises monotonically with depth in base',
           all(coh[(BASE, 'nitrogen', a)] < coh[(BASE, 'nitrogen', b)]
               for a, b in zip((0, 8, 16, 24), (8, 16, 24, 31))))
    c.true('all three unmarked blocks end above the halide control in base',
           all(coh[(BASE, blk, LAYER)] > coh[(BASE, 'halide', LAYER)]
               for blk in ('nitrogen', 'sulfur', 'carbonyl')))

    # --- Aim 2: the N->O claim is about DEPTH, not about one layer --------------------
    # The section's lead was originally written at layer 31 and was wrong: at chem's own
    # best probe layer it makes no N->O errors at all. These assertions pin the depth
    # structure, so a rewrite back to a single-layer claim fails here.
    comp = {(r['model'], int(r['layer'])): r
            for r in rows('probe/data/error_composition_by_layer.csv')}
    curves = {(r['model'], int(r['layer'])): float(r['balanced_acc'])
              for r in rows('probe/data/layer_curves.csv')}
    FLOOR = 19
    c.true(f'all three models exceed 0.94 at the error floor (L{FLOOR})',
           all(curves[(m, FLOOR)] > 0.94 for m in (BASE, CHEM, R1)),
           ', '.join(f'{NAME[m]} {curves[(m, FLOOR)]:.3f}' for m in (BASE, CHEM, R1)))
    for m in (BASE, CHEM):
        c.true(f'{NAME[m]} has ZERO N->O errors at layers 19-21',
               all(int(comp[(m, L)]['n_nitrogen_to_oxygen']) == 0 for L in (19, 20, 21)))
    c.true('chem acquires the confusion from layer 22 and holds it',
           all(int(comp[(CHEM, L)]['n_nitrogen_to_oxygen']) > 0 for L in range(22, 32)))
    c.true('base never acquires it: no N->O error at any layer >= 24',
           all(int(comp[(BASE, L)]['n_nitrogen_to_oxygen']) == 0 for L in range(24, 32)))
    c.true(f'L{FLOOR} is the FIRST such layer (L18 does not qualify)',
           not all(curves[(m, FLOOR - 1)] > 0.94 for m in (BASE, CHEM, R1)),
           f'reason L18 {curves[(R1, FLOOR - 1)]:.3f}')
    c.true('the error floor is also chem\'s best probe layer',
           max(range(32), key=lambda L: curves[(CHEM, L)]) == FLOOR)
    for L, want in ((24, 0.375), (26, 0.627), (29, 0.756)):
        c.eq(f'chem N->O share L{L}', comp[(CHEM, L)]['share_nitrogen_to_oxygen'], want,
             1e-2)
    c.eq('reason N->O share L19', comp[(R1, 19)]['share_nitrogen_to_oxygen'], 0.51, 1e-2)
    c.eq('reason N->O share L31', comp[(R1, 31)]['share_nitrogen_to_oxygen'], 0.90, 1e-2)
    # The denominator caveat the figure caption states, held as a fact.
    c.true('base error totals are <= 5 from layer 21 on',
           all(int(comp[(BASE, L)]['n_total_errors']) <= 5 for L in range(21, 32)))

    # --- Aim 2: the graded ambiguity claim, held to exactly its stated width -----------
    # The section claims base-vs-chem on `amide` (both readings) and nothing wider. These
    # checks exist to make an over-broad rewrite fail loudly: if someone later writes "base
    # leans least on all four groups", the last two assertions here contradict it.
    amb = {(r['model'], r['group']): r
           for r in rows('ambiguity/data/ambiguity_two_readings.csv')
           if int(r['layer']) == LAYER}
    c.true('base and chem select the SAME probe C (posteriors commensurable)',
           amb[(BASE, 'amide')]['best_C_mode'] == amb[(CHEM, 'amide')]['best_C_mode'],
           amb[(BASE, 'amide')]['best_C_mode'])
    c.true('reason selects a DIFFERENT C (excluded from the posterior contrast)',
           amb[(R1, 'amide')]['best_C_mode'] != amb[(BASE, 'amide')]['best_C_mode'],
           amb[(R1, 'amide')]['best_C_mode'])
    for reading in ('posterior_delta', 'geometric_delta'):
        c.true(f'amide: base leans less than chem on {reading.split("_")[0]}',
               float(amb[(BASE, 'amide')][reading]) < float(amb[(CHEM, 'amide')][reading]))
    c.true('nitro REVERSES between readings (why it cannot carry the claim)',
           (float(amb[(BASE, 'nitro')]['posterior_delta'])
            < float(amb[(CHEM, 'nitro')]['posterior_delta']))
           != (float(amb[(BASE, 'nitro')]['geometric_delta'])
               < float(amb[(CHEM, 'nitro')]['geometric_delta'])))
    c.true('sulfone runs AGAINST the story: chem below base on both readings',
           all(float(amb[(CHEM, 'sulfone')][k]) < float(amb[(BASE, 'sulfone')][k])
               for k in ('posterior_delta', 'geometric_delta')))

    # --- the anisotropy precondition --------------------------------------------------
    geo = {(r['model'], int(r['layer'])): r
           for r in rows('geometry/data/geometry_by_layer.csv')}
    for m, want in ((BASE, 0.944), (CHEM, 0.676), (R1, 0.806)):
        c.eq(f'anisotropy {NAME[m]} L31', geo[(m, LAYER)]['raw_pairwise_cosine'], want)
        d = abs(float(geo[(m, LAYER)]['centered_between'])
                - float(geo[(m, LAYER)]['centered_null']))
        c.true(f'centered between-class == null, {NAME[m]}', d < 5e-3, f'{d:.5f}')
    for m, want in ((BASE, 0.25), (CHEM, 0.65), (R1, 0.70)):
        c.eq(f'degenerate rate {NAME[m]} L31', geo[(m, LAYER)]['degenerate_rate'], want)

    # --- retrieval, all three axes, each against its own chance -----------------------
    ax = {(r['model'], r['axis']): r for r in rows('geometry/data/retrieval_by_axis.csv')
          if int(r['layer']) == LAYER}
    for axis in ('inter_group', 'halide', 'carbon_ladder'):
        c.true(f'{axis} beats chance in every model',
               all(float(ax[(m, axis)]['hit1']) > 4 * float(ax[(m, axis)]['random_hit1'])
                   for m in (BASE, CHEM, R1)))
    c.eq('halide hit@1 chem', ax[(CHEM, 'halide')]['hit1'], 0.667)
    c.true('sulfoxide exclusion moves base and chem equally',
           abs(float(geo[(BASE, LAYER)]['hit1_ex_sulfoxide'])
               - float(geo[(CHEM, LAYER)]['hit1_ex_sulfoxide'])) < 1e-6)

    # --- ambiguity: within-model only -------------------------------------------------
    amb = {(r['model'], r['group']): r
           for r in rows('ambiguity/data/ambiguity_two_readings.csv')
           if int(r['layer']) == LAYER}
    c.true('amide leans to oxygen on BOTH readings in all three models',
           all(float(amb[(m, 'amide')]['posterior_delta']) > 0
               and float(amb[(m, 'amide')]['geometric_delta']) > 0
               for m in (BASE, CHEM, R1)))
    c.true('the two readings disagree on chem vs base (why no ranking)',
           (float(amb[(CHEM, 'nitro')]['posterior_delta'])
            > float(amb[(BASE, 'nitro')]['posterior_delta']))
           != (float(amb[(CHEM, 'nitro')]['geometric_delta'])
               > float(amb[(BASE, 'nitro')]['geometric_delta'])))
    c.true('best_C matched for base and chem, looser for reason',
           float(amb[(BASE, 'amide')]['best_C_mode'])
           == float(amb[(CHEM, 'amide')]['best_C_mode'])
           != float(amb[(R1, 'amide')]['best_C_mode']))

    # --- the probe's errors point the same way ----------------------------------------
    conf = [r for r in rows('probe/data/confusion_errors.csv') if int(r['layer']) == LAYER]
    for m, total, n_to_o in ((BASE, 2, 0), (CHEM, 35, 20), (R1, 41, 37)):
        c.eq(f'total probe errors {NAME[m]}',
             sum(int(r['n']) for r in conf if r['model'] == m), total, 0)
        c.eq(f'nitrogen->oxygen errors {NAME[m]}',
             sum(int(r['n']) for r in conf if r['model'] == m
                 and r['true_class'] == 'nitrogen' and r['pred_class'] == 'oxygen'),
             n_to_o, 0)

    return c.report()


if __name__ == '__main__':
    sys.exit(main())
