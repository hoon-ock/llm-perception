#!/usr/bin/env python3
"""The behavioural readout, and the only half of Table 1 that is not a probe.

`generation_eval.py` scores every functional-group name as a teacher-forced continuation of
the same declarative prompts the probe reads its activations from, and takes the argmax over
20 classes. That makes it directly comparable to the probe rather than merely correlated --
and it is the measurement the paper's central paradox rests on, because it ranks the three
models in the opposite order to the probe.

Its results live in `Results/generation_eval/`, which is gitignored, and no other Analysis
script reads them. This snapshots them so the paper builds from committed files.

It also adds what `summary.json` lacks: a **molecule-level bootstrap CI** on the accuracy
difference between two models, matching `../probe/analyze_sweep.py:bootstrap_diff`. Without
it Table 1 would pair a probe difference that has an interval against a generation difference
that does not.

Two scorings are carried side by side and neither is dropped:

  raw               argmax of summed log-prob over each class's surface forms
  length_normalized the same, divided by token count

They disagree sharply -- the base model scores 0.837 raw and 0.401 normalized -- because
normalization trades a bias toward short class names for a bias toward long ones. `raw` is
the headline (it is what the model would actually emit); `length_normalized` is reported
beside it so the choice is visible rather than silent.

Reads only `Results/generation_eval/`. No activations, no model, CPU-only.
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DEFAULT_RESULTS = os.path.join(REPO, 'fc_group', 'Results', 'generation_eval')
DEFAULT_OUT = os.path.join(HERE, 'data')

MODEL_8B = 'meta-llama-Llama-3.1-8B'
MODEL_CHEM = 'phenixace-Chem-R-Faithful'
MODEL_R1 = 'deepseek-ai-DeepSeek-R1-Distill-Llama-8B'
MODELS = [MODEL_8B, MODEL_CHEM, MODEL_R1]
# Chemistry-tuned vs base is the contrast the paradox is stated over.
DEFAULT_PAIR = (MODEL_CHEM, MODEL_8B)
SCORINGS = ('raw', 'length_normalized')


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def load_summary(results_dir, model, entity_type):
    path = os.path.join(results_dir, model, entity_type, 'data', 'summary.json')
    with open(path) as fh:
        return json.load(fh)


def load_scores(results_dir, model, entity_type):
    """Per-prompt rows, asserted molecule-major so the bootstrap can group by block.

    `generation_eval.py` writes molecule-major / template-minor. The bootstrap below relies
    on that to reshape into (n_molecules, n_templates) without carrying an index, so it is
    checked rather than assumed -- if the writer ever changes, this fails loudly instead of
    silently resampling across molecules.
    """
    rows = read_csv(os.path.join(results_dir, model, entity_type, 'data', 'scores.csv'))
    names = [r['iupac_name'] for r in rows]
    n_mol = len(dict.fromkeys(names))
    if len(rows) % n_mol:
        raise SystemExit(f"{model}: {len(rows)} rows is not a whole number of {n_mol} molecules")
    n_tpl = len(rows) // n_mol
    if names != [n for n in dict.fromkeys(names) for _ in range(n_tpl)]:
        raise SystemExit(f"{model}: scores.csv is not molecule-major -- row layout changed")
    return rows, n_mol, n_tpl


def bootstrap_diff(rows_a, rows_b, n_mol, n_tpl, field, n_boot, rng):
    """Accuracy difference (a - b) with a molecule-level bootstrap CI.

    The resampling unit is the molecule, not the row: a molecule's 10 template rows are
    near-copies of each other, so resampling rows would treat them as independent and shrink
    the interval to a fraction of its honest width. Effective n is 92, not 920.

    The two models must have been scored on the same prompts in the same order for the
    pairing to mean anything; asserted, not assumed.
    """
    if [r['iupac_name'] for r in rows_a] != [r['iupac_name'] for r in rows_b]:
        raise SystemExit("the two models were not scored on the same prompts -- not pairable")
    a = np.array([int(r[field]) for r in rows_a], float).reshape(n_mol, n_tpl)
    b = np.array([int(r[field]) for r in rows_b], float).reshape(n_mol, n_tpl)
    per_mol = a.mean(axis=1) - b.mean(axis=1)
    point = float(per_mol.mean())
    draws = per_mol[rng.integers(0, n_mol, (n_boot, n_mol))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


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
    p.add_argument('--pair', nargs=2, default=list(DEFAULT_PAIR),
                   metavar=('MODEL_A', 'MODEL_B'))
    p.add_argument('--n-boot', type=int, default=20000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summaries = {m: load_summary(args.results_dir, m, args.entity_type)
                 for m in args.models}
    scores = {m: load_scores(args.results_dir, m, args.entity_type)
              for m in args.models}

    # ---- 1. headline per model x scoring ----------------------------------------
    rows = []
    for model in args.models:
        s = summaries[model]
        for scoring in SCORINGS:
            rows.append({
                'model': model, 'entity_type': args.entity_type, 'scoring': scoring,
                'n_prompts': s['n_prompts'], 'n_molecules': s['n_molecules'],
                'n_templates': s['n_templates'], 'n_classes': s['n_classes_scored'],
                'chance': s['chance'],
                'accuracy': round(s[scoring]['accuracy'], 6),
                'balanced_accuracy': round(s[scoring]['balanced_accuracy'], 6),
                # log_loss and margin describe the full distribution, not an argmax, so they
                # are properties of the scoring run rather than of either tie-break rule.
                'log_loss': round(s['log_loss'], 6),
                'mean_top1_margin': round(s['mean_top1_margin'], 6),
            })
    write_csv(os.path.join(args.out_dir, 'generation_summary.csv'), rows,
              ['model', 'entity_type', 'scoring', 'n_prompts', 'n_molecules',
               'n_templates', 'n_classes', 'chance', 'accuracy', 'balanced_accuracy',
               'log_loss', 'mean_top1_margin'])

    # ---- 2. per-class recall ------------------------------------------------------
    class_rows = []
    for model in args.models:
        for scoring in SCORINGS:
            for cls, recall in sorted(summaries[model][scoring]['per_class_recall'].items()):
                class_rows.append({
                    'model': model, 'entity_type': args.entity_type, 'scoring': scoring,
                    'functional_group': cls, 'recall': round(recall, 6)})
    write_csv(os.path.join(args.out_dir, 'generation_by_class.csv'), class_rows,
              ['model', 'entity_type', 'scoring', 'functional_group', 'recall'])

    # ---- 3. paired molecule-level bootstrap --------------------------------------
    a, b = args.pair
    for name in (a, b):
        if name not in scores:
            raise SystemExit(f"--pair names {name}, which --models did not load")
    rows_a, n_mol, n_tpl = scores[a]
    rows_b, _, _ = scores[b]
    diff_rows = []
    for scoring, field in (('raw', 'correct_raw'), ('length_normalized', 'correct_norm')):
        diff, lo, hi = bootstrap_diff(rows_a, rows_b, n_mol, n_tpl, field,
                                      args.n_boot, rng)
        diff_rows.append({
            'model_a': a, 'model_b': b, 'entity_type': args.entity_type,
            'scoring': scoring,
            'acc_a': round(summaries[a][scoring]['accuracy'], 6),
            'acc_b': round(summaries[b][scoring]['accuracy'], 6),
            'diff_a_minus_b': round(diff, 6),
            'ci_lo': round(lo, 6), 'ci_hi': round(hi, 6),
            'ci_excludes_zero': int(not (lo < 0 < hi)),
            'n_molecules': n_mol, 'n_boot': args.n_boot, 'seed': args.seed,
        })
    write_csv(os.path.join(args.out_dir, 'generation_diff_bootstrap.csv'), diff_rows,
              ['model_a', 'model_b', 'entity_type', 'scoring', 'acc_a', 'acc_b',
               'diff_a_minus_b', 'ci_lo', 'ci_hi', 'ci_excludes_zero', 'n_molecules',
               'n_boot', 'seed'])


if __name__ == '__main__':
    main()
