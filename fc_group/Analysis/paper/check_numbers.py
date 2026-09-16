#!/usr/bin/env python3
"""Assert that every numeral in the paper's prose traces to a released CSV.

    python fc_group/Analysis/paper/check_numbers.py [--sections paper/sections]

Tables and figures are generated, so they cannot drift. The prose can: a number typed into a
sentence has no link back to the script that produced it, and a later re-run of the pipeline
will not update it. This is the check that closes that hole, and it earned its place -- on
its first run it caught the confusion-matrix counts being quoted from a gitignored tree with
no committed source.

Exempt by design:
  * facts asserted in Setup that come from the dataset or the prompt config rather than from
    an analysis (molecule counts, layer indices, template counts) -- listed in SETUP;
  * years inside `\\cit{...}` placeholder marks, which are unverified by construction and
    flagged in red in the draft until they become real `\\cite{}` commands.

Exit status is 1 if anything is untraced, so this can gate a submission.
"""
import argparse
import csv
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(ANALYSIS, '..', '..'))

# Dataset and prompt-configuration facts stated in Setup. Each is checkable against
# functional_group_dataset.csv or config_extract_activation.yaml, not against an analysis.
SETUP = {
    '92', '20', '23', '19', '17', '10', '920',      # molecules, classes, templates, prompts
    '3', '4', '5', '6', '8',                        # carbon counts, class sizes
    '32', '0', '1', '2', '15', '16', '18', '24', '31',   # layers
    '171', '95', '3.1', '0.975', '1.00',            # pairs, CI level, model name, quoted span
}


def released_values():
    """Every numeric value in the committed Analysis CSVs, at any rounding the prose may use."""
    vals = set()
    for path in glob.glob(os.path.join(ANALYSIS, '*', 'data', '*.csv')):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                for v in row.values():
                    try:
                        x = abs(float(v))
                    except (TypeError, ValueError):
                        continue
                    vals.update(f'{x:.{p}f}' for p in range(5))
                    vals.add(f'{x:g}')
    if not vals:
        raise SystemExit('no Analysis CSVs found -- run the analyze_*.py scripts first')
    return vals


def prose(path):
    """Section text with comments and \\cit{} marks removed, as one string per line number."""
    out = []
    src = open(path).read()
    # Strip \cit{...} across line breaks before splitting, so a multi-line mark is fully gone.
    src = re.sub(r'\\cit\{[^}]*\}', '', src, flags=re.S)
    for n, line in enumerate(src.splitlines(), 1):
        out.append((n, '' if line.lstrip().startswith('%') else line))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sections', default=os.path.join(REPO, 'paper', 'sections'))
    args = p.parse_args()

    vals = released_values()
    bad = []
    for path in sorted(glob.glob(os.path.join(args.sections, '*.tex'))):
        for n, line in prose(path):
            for m in re.finditer(r'(?<![\w.])(\d+\.\d+|\d+)(?![\w.])', line):
                tok = m.group(1)
                if tok not in SETUP and tok not in vals:
                    bad.append((os.path.relpath(path, REPO), n, tok, line.strip()[:90]))

    if bad:
        print(f'FAIL -- {len(bad)} numeral(s) with no released source:\n')
        for f, n, tok, line in bad:
            print(f'  {f}:{n}  "{tok}"\n      {line}\n')
        return 1
    print('PASS -- every numeral in the prose traces to a released CSV or a Setup fact')
    return 0


if __name__ == '__main__':
    sys.exit(main())
