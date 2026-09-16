#!/usr/bin/env python3
"""Score the free generations that `generation_eval.py` writes and never reads.

`generation_eval.py` emits two readouts per prompt. The forced-choice one (`scores.csv`)
is scored and drives Table 1. The free one (`generations.csv`) is logged verbatim and
dropped -- its docstring says "Never scored by string match", to avoid a metric that
measures the parser instead of the model.

That caution was aimed at the wrong failure. The parser risk is real but bounded; the
cost of not scoring at all is that the behavioural half of the paper cannot see a format
collapse. 12.6% of the base model's continuations are multiple-choice enumerations --

    prompt  'The functional-group identity of Pentanal (CCCCC=O) is '
    output  '\nA. aldehyde\nB. ketone'

-- which contain the gold string and commit to nothing. Chem-R and R1-Distill never do
this (0.0%). A containment metric would hand ~10 points to the one model failing the
task. So this scores what the model *committed to*, and files format failures as their
own outcome rather than as free credit.

The result is a third ranking, distinct from both the forced choice and the probe:

    model        forced choice   free generation (strict)
    chem              0.978              0.824
    base              0.837              0.541
    reason            0.804              0.777

The base model can rank the right class better than R1 and cannot produce it. That gap
is the point of this file.

Why `--max-new-tokens 10` is not the confound. 75-95% of stored generations end
mid-token, so a "no answer" could in principle be a clipped one. Measured here: among
generations that do name a class, the first mention lands at median word 1, p90 word 2,
max word 8, with 96-99% inside the first five words, for all three models. When these
models answer they answer immediately, so the non-answers are evasion -- description,
superclass, prompt restatement -- and not truncation. No rerun buys anything.

Reads only `Results/generation_eval/*/functional_group/data/generations.csv`. CPU-only.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(HERE, '..', '..'))))

# Reused rather than reimplemented: the bootstrap's resampling unit is the molecule (92),
# not the row (920), and it must stay identical to the probe's for Table 1's intervals and
# these to be comparable.
from analyze_generation import (  # noqa: E402
    DEFAULT_PAIR, DEFAULT_RESULTS, MODELS, bootstrap_diff, read_csv, write_csv,
)
# The adjudicator itself lives beside generation_eval.py, because the HPC job imports it too
# and must score exactly the same way this does. One lexicon, two callers.
from free_generation_scoring import (  # noqa: E402
    adjudicate_rows, build_block, summarize, summarize_by_template,
)

DEFAULT_OUT = os.path.join(HERE, 'data')

def write_results_summary(results_dir, model, entity_type, block):
    """Merge the free-generation block into the model's own summary.json.

    `generation_eval.py` writes that file with the forced-choice readout only, so a model's
    result directory shows half the behavioural picture. This joins the other half to it.

    Assigning one key rather than updating in place makes the merge idempotent: a second run
    replaces the block instead of nesting it. Everything `generation_eval.py` wrote is left
    untouched -- this block sits beside the forced-choice record, it does not edit it.

    Note that `Results*` is gitignored and `scripts/pull_results.sh` rsyncs summary.json down
    from Swan without --delete, so the next pull silently reverts this. That is why the block
    carries its own provenance: a missing or stale one is visible on sight, and re-running
    this script restores it.
    """
    path = os.path.join(results_dir, model, entity_type, 'data', 'summary.json')
    if not os.path.exists(path):
        print(f"  no summary.json for {model} -- skipped (the CSVs are the artifact)")
        return
    with open(path) as fh:
        summary = json.load(fh)
    summary['free_generation'] = block
    with open(path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"  merged free_generation into {os.path.relpath(path)}")


def score_model(results_dir, model, entity_type, strict_names_only):
    """Re-derive the adjudication from a stored generations.csv.

    `generation_eval.py` now adjudicates as it generates, so this route exists for the models
    run before that (the Sep 13 results, which have no block) and for propagating a lexicon
    change without occupying a GPU.
    """
    path = os.path.join(results_dir, model, entity_type, 'data', 'generations.csv')
    return adjudicate_rows(read_csv(path), model, strict_names_only)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default=DEFAULT_RESULTS)
    p.add_argument('--out-dir', default=DEFAULT_OUT)
    p.add_argument('--entity-type', default='functional_group')
    p.add_argument('--models', nargs='+', default=MODELS)
    p.add_argument('--pair', nargs=2, default=list(DEFAULT_PAIR),
                   metavar=('MODEL_A', 'MODEL_B'))
    p.add_argument('--n-boot', type=int, default=20000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-results-summary', action='store_true',
                   help='do not merge the free_generation block into each model\'s '
                        'summary.json under --results-dir, leaving Results/ a pure mirror '
                        'of what the HPC run produced.')
    p.add_argument('--strict-names-only', action='store_true',
                   help='drop the element and formula forms ("the bromine atom", "-SH", '
                        '"hydrocarbon"), so only a class name or a named synonym counts. '
                        'The conservative tier, reproducible on demand.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    adjudicated = {m: score_model(args.results_dir, m, args.entity_type,
                                  args.strict_names_only)
                   for m in args.models}

    summary_rows, class_rows, tmpl_rows, recalls = [], [], [], {}
    for model in args.models:
        summary, recall = summarize(model, adjudicated[model], args.entity_type)
        summary_rows.append(summary)
        recalls[model] = recall
        for ti, block in sorted(summarize_by_template(adjudicated[model]).items()):
            tmpl_rows.append({'model': model, 'entity_type': args.entity_type,
                              'template_index': ti, **block})
        for cls, value in sorted(recall.items()):
            class_rows.append({'model': model, 'entity_type': args.entity_type,
                               'functional_group': cls, 'strict_recall': round(value, 6)})

    write_csv(os.path.join(args.out_dir, 'free_generation_summary.csv'), summary_rows,
              list(summary_rows[0]))
    write_csv(os.path.join(args.out_dir, 'free_generation_by_class.csv'), class_rows,
              ['model', 'entity_type', 'functional_group', 'strict_recall'])
    write_csv(os.path.join(args.out_dir, 'free_generation_by_template.csv'), tmpl_rows,
              list(tmpl_rows[0]))

    flat = [r for m in args.models for r in adjudicated[m]]
    write_csv(os.path.join(args.out_dir, 'free_generation_adjudicated.csv'), flat,
              ['model', 'iupac_name', 'template_index', 'true_label', 'response_type',
               'predicted_label', 'matched_form', 'correct_strict', 'gold_contained',
               'generation'])

    # ---- molecule-level bootstrap, same resampling unit as the probe ----------------
    a, b = args.pair
    for name in (a, b):
        if name not in adjudicated:
            raise SystemExit(f"--pair names {name}, which --models did not load")
    rows_a, rows_b = adjudicated[a], adjudicated[b]
    names = [r['iupac_name'] for r in rows_a]
    n_mol = len(dict.fromkeys(names))
    if len(rows_a) % n_mol:
        raise SystemExit(f"{a}: {len(rows_a)} rows is not a whole number of {n_mol} molecules")
    n_tpl = len(rows_a) // n_mol
    if names != [n for n in dict.fromkeys(names) for _ in range(n_tpl)]:
        raise SystemExit(f"{a}: generations.csv is not molecule-major -- row layout changed")

    by = {r['model']: r for r in summary_rows}
    diff, lo, hi = bootstrap_diff(rows_a, rows_b, n_mol, n_tpl, 'correct_strict',
                                  args.n_boot, rng)
    write_csv(os.path.join(args.out_dir, 'free_generation_diff_bootstrap.csv'), [{
        'model_a': a, 'model_b': b, 'entity_type': args.entity_type,
        'scoring': 'strict_names_only' if args.strict_names_only else 'strict',
        'acc_a': by[a]['strict_accuracy'], 'acc_b': by[b]['strict_accuracy'],
        'diff_a_minus_b': round(diff, 6), 'ci_lo': round(lo, 6), 'ci_hi': round(hi, 6),
        'ci_excludes_zero': int(not (lo < 0 < hi)),
        'n_molecules': n_mol, 'n_boot': args.n_boot, 'seed': args.seed,
    }], ['model_a', 'model_b', 'entity_type', 'scoring', 'acc_a', 'acc_b',
         'diff_a_minus_b', 'ci_lo', 'ci_hi', 'ci_excludes_zero', 'n_molecules',
         'n_boot', 'seed'])

    # ---- merge into each model's own summary.json ----------------------------------
    # Last, so the committed CSVs in Analysis/ are already on disk before anything under
    # the gitignored Results/ tree is touched.
    if args.strict_names_only:
        # This flag reproduces the conservative tier and is normally paired with a scratch
        # --out-dir. Letting it overwrite the published lenient block in Results/ would
        # swap one tier for another with nothing in the output saying so.
        print("--strict-names-only: leaving Results/ summary.json alone")
    elif not args.no_results_summary:
        for model in args.models:
            block = build_block(model, adjudicated[model], args.entity_type)
            block['derived_from'] = 'generations.csv'
            write_results_summary(args.results_dir, model, args.entity_type, block)

    for s in summary_rows:
        print(f"{s['model']:44s} strict={s['strict_accuracy']:.3f} "
              f"commit={s['commit_rate']:.3f} given-commit={s['conditional_accuracy']:.3f} "
              f"| malformed={s['malformed_rate']:.3f} underspec={s['underspecified_rate']:.3f} "
              f"non-answer={s['non_answer_rate']:.3f} "
              f"| containment={s['containment_accuracy']:.3f}")


if __name__ == '__main__':
    main()
