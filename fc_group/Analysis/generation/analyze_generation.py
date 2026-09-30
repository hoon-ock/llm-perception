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
import math
import os

from statistics import stdev

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DEFAULT_RESULTS = os.path.join(REPO, 'fc_group', 'Results', 'generation_eval')
DEFAULT_OUT = os.path.join(HERE, 'data')

MODEL_8B = 'meta-llama-Llama-3.1-8B'
MODEL_CHEM = 'phenixace-Chem-R-Faithful'
MODEL_R1 = 'deepseek-ai-DeepSeek-R1-Distill-Llama-8B'
# The second chemistry tier. Chem-R-8B is what Chem-R-Faithful was GRPO-trained FROM, so the
# pair separates domain tuning from faithfulness tuning; ChemDFM-v1.5-8B sits on Llama-3
# rather than 3.1 and is external-validity only (see fc_group/model_registry.py).
MODEL_CHEM_R = 'weidawang-Chem-R-8B'
MODEL_CHEMDFM = 'OpenDFM-ChemDFM-v1.5-8B'
# Appended, never inserted: every section below loops this list in order and the one `rng`
# is consumed after them, so a model added at the END leaves existing rows byte-identical.
MODELS = [MODEL_8B, MODEL_CHEM, MODEL_R1, MODEL_CHEM_R, MODEL_CHEMDFM]
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


def log_loss_from(rows, labels, prefix):
    """Mean cross-entropy of the true class under a softmax over one score column set.

    `generation_eval.py` records log-loss for the RAW scores only: `p_true` there is a
    softmax over the summed log-probabilities, so `summary.json` carries one `log_loss` and
    it belongs to raw scoring alone. This re-derives it from the stored per-candidate scores,
    which lets the same quantity be computed for the length-normalized rule as well.

    A caution that belongs with the number, not in a reader's head: the length-normalized
    scores are per-token MEANS, not log-probabilities, so a softmax over them is not a
    calibrated posterior -- it is a softmax at an arbitrary temperature. Dividing by token
    count compresses the spread of the 20 scores (1.74x narrower for the base model), and a
    flatter score vector raises cross-entropy mechanically, whether or not the model is
    really less certain. The normalized column is therefore comparable ACROSS MODELS, which
    all get identical treatment, and NOT against the raw column beside it.
    """
    total = 0.0
    for r in rows:
        scores = [float(r[prefix + lab]) for lab in labels]
        m = max(scores)
        denom = m + math.log(sum(math.exp(x - m) for x in scores))
        total += -(float(r[prefix + r['true_label']]) - denom)
    return total / len(rows)


def write_results_log_loss(results_dir, model, entity_type, per_scoring):
    """Attach each scoring rule's log-loss to its own block in the model's summary.json.

    The top-level `log_loss` that `generation_eval.py` wrote is left exactly as it is -- it
    is the raw one, and other readers may depend on it. This adds `raw.log_loss` and
    `length_normalized.log_loss` beside the accuracies they belong to, so the file says which
    rule each number came from instead of leaving it implicit.

    Assigning into the existing blocks keeps the merge idempotent. `Results*` is gitignored
    and `scripts/pull_results.sh` rsyncs summary.json down without --delete, so a later pull
    reverts this; re-running the script restores it.
    """
    path = os.path.join(results_dir, model, entity_type, 'data', 'summary.json')
    if not os.path.exists(path):
        print(f"  no summary.json for {model} -- skipped (the CSVs are the artifact)")
        return
    with open(path) as fh:
        summary = json.load(fh)
    for scoring, value in per_scoring.items():
        if scoring in summary:
            summary[scoring]['log_loss'] = round(value, 6)
    with open(path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"  merged per-scoring log_loss into {os.path.relpath(path)}")


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
    p.add_argument('--no-results-summary', action='store_true',
                   help='do not merge per-scoring log_loss into each model\'s '
                        'summary.json under --results-dir, leaving Results/ a pure '
                        'mirror of what the HPC run produced.')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summaries = {m: load_summary(args.results_dir, m, args.entity_type)
                 for m in args.models}
    scores = {m: load_scores(args.results_dir, m, args.entity_type)
              for m in args.models}

    # The 20 class names, read off the score columns rather than hardcoded.
    any_rows = scores[args.models[0]][0]
    labels = [c[len('logp_'):] for c in any_rows[0] if c.startswith('logp_')]
    # Log-loss per scoring rule. `raw` re-derives what generation_eval.py already published
    # (asserted below); `length_normalized` is the one that did not exist before.
    ll = {m: {'raw': log_loss_from(scores[m][0], labels, 'logp_'),
              'length_normalized': log_loss_from(scores[m][0], labels, 'logpnorm_')}
          for m in args.models}
    for m in args.models:
        published = summaries[m]['log_loss']
        if abs(ll[m]['raw'] - published) > 1e-6:
            raise SystemExit(
                f"{m}: re-derived raw log-loss {ll[m]['raw']:.6f} != published "
                f"{published:.6f} -- the scoring columns and summary.json disagree")

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
                # log_loss describes the full distribution rather than an argmax, but it
                # IS a property of the scoring rule -- the softmax is taken over that rule's
                # scores -- so each row carries its own. The margin stays raw-only: it is
                # read from the raw score gap in generation_eval.py and has no normalized
                # counterpart to report.
                'log_loss': round(ll[model][scoring], 6),
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

    # ---- 3b. prompt sensitivity: spread of accuracy over the 10 templates ----------
    #
    # Greedy decoding and teacher-forced scoring are both deterministic, so this -- not
    # repeated sampling -- is where each readout's variance lives. The caption used to quote
    # it for three models from `free_generation_by_template.csv`; the teacher-forced half had
    # no committed source at all, which is what this adds.
    #
    # THE SD IS NOT COMPARABLE ACROSS ROWS ON ITS OWN, and that is why `dispersion_ratio` is
    # here beside it. Accuracy near 1.0 has nowhere to vary: across these five models, mean
    # accuracy and raw SD correlate at r = -0.84, so a low SD mostly reports a high mean.
    # The ratio divides the observed SD by the binomial null sqrt(p(1-p)/n_per_template) --
    # the spread 92 Bernoulli draws per template would produce at that accuracy even if every
    # template were equally good -- so 1.0 means "no more template effect than sampling
    # noise". Every model here lands between 2 and 5.
    sens_rows = []
    free_tpl = {}
    free_path = os.path.join(args.out_dir, 'free_generation_by_template.csv')
    if os.path.exists(free_path):
        for r in read_csv(free_path):
            free_tpl.setdefault(r['model'], {})[int(r['template_index'])] = \
                float(r['strict_accuracy'])
    for model in args.models:
        rows_m, n_mol, n_tpl = scores[model]
        readouts = {}
        # Teacher-forced, recomputed per template from the per-prompt rows. `load_scores`
        # has already asserted the molecule-major layout, so `i % n_tpl` is the template.
        for scoring, col in (('raw', 'correct_raw'), ('length_normalized', 'correct_norm')):
            per = [[] for _ in range(n_tpl)]
            for i, r in enumerate(rows_m):
                per[i % n_tpl].append(int(r[col]))
            readouts[scoring] = [sum(v) / len(v) for v in per]
        if model in free_tpl:
            # Read back from the released per-template CSV, whose values are stored at 6dp.
            # Averaging them can therefore differ from the pooled accuracy in
            # free_generation_summary.csv by ~1e-6 (reason: 0.789131 here against 0.789130,
            # exactly 726/920 = 0.7891304). That is the rounding of already-rounded inputs,
            # not a disagreement, and it is far below anything sd or dispersion_ratio resolve.
            readouts['free_strict'] = [free_tpl[model][i] for i in sorted(free_tpl[model])]
        for scoring, acc in readouts.items():
            mu = sum(acc) / len(acc)
            sd = stdev(acc)
            null = math.sqrt(mu * (1 - mu) / n_mol) if 0 < mu < 1 else float('nan')
            sens_rows.append({
                'model': model, 'entity_type': args.entity_type, 'scoring': scoring,
                'n_templates': len(acc), 'n_per_template': n_mol,
                'mean_accuracy': round(mu, 6), 'sd_accuracy': round(sd, 6),
                'min_accuracy': round(min(acc), 6), 'max_accuracy': round(max(acc), 6),
                'binomial_sd': round(null, 6),
                'dispersion_ratio': round(sd / null, 6) if null == null else '',
            })
    write_csv(os.path.join(args.out_dir, 'template_sensitivity.csv'), sens_rows,
              ['model', 'entity_type', 'scoring', 'n_templates', 'n_per_template',
               'mean_accuracy', 'sd_accuracy', 'min_accuracy', 'max_accuracy',
               'binomial_sd', 'dispersion_ratio'])

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
    if not args.no_results_summary:
        for model in args.models:
            write_results_log_loss(args.results_dir, model, args.entity_type, ll[model])

    write_csv(os.path.join(args.out_dir, 'generation_diff_bootstrap.csv'), diff_rows,
              ['model_a', 'model_b', 'entity_type', 'scoring', 'acc_a', 'acc_b',
               'diff_a_minus_b', 'ci_lo', 'ci_hi', 'ci_excludes_zero', 'n_molecules',
               'n_boot', 'seed'])


if __name__ == '__main__':
    main()
