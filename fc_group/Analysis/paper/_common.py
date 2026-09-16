#!/usr/bin/env python3
"""Shared loading and naming for the paper's table and figure generators.

Both generators read only `fc_group/Analysis/*/data/*.csv` -- never `Results/`, which is
gitignored. That is deliberate and is the reproducibility test in the plan: if either
generator ever needs `Results/`, a snapshot step was missed upstream.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.abspath(os.path.join(HERE, '..'))
REPO = os.path.abspath(os.path.join(ANALYSIS, '..', '..'))
DEFAULT_OUT = os.path.join(REPO, 'paper')

BASE = 'meta-llama-Llama-3.1-8B'
CHEM = 'phenixace-Chem-R-Faithful'
R1 = 'deepseek-ai-DeepSeek-R1-Distill-Llama-8B'
# Presentation order everywhere: base first, then the two fine-tunes.
MODELS = [BASE, CHEM, R1]

# Display names. The analysis scripts carry their own terse SHORT maps for console tables
# (`base`/`chem`/`r1`); the paper needs the real checkpoint names, so this is a separate
# mapping rather than a third copy of the same one.
LABEL = {
    BASE: r'Llama-3.1-8B \textsc{(base)}',
    CHEM: r'Chem-R-Faithful \textsc{(chem)}',
    R1: r'R1-Distill-Llama-8B \textsc{(reason)}',
}
SHORT = {BASE: 'base', CHEM: 'chem', R1: 'reason'}
# Okabe-Ito blue / vermillion / bluish-green. Chosen by running the dataviz palette
# validator, not by eye: the seaborn default (#4C72B0/#C44E52/#55A868) puts red beside
# green at deuteranopic dE 7.3, inside the band that is only legal with secondary
# encoding. This triple clears every check (worst adjacent dE 11.0 deutan, 25.8 normal).
COLOR = {BASE: '#0072B2', CHEM: '#D55E00', R1: '#009E73'}
# Secondary encoding so identity never rests on hue alone.
MARKER = {BASE: 'o', CHEM: 's', R1: '^'}
DASH = {BASE: (None, None), CHEM: (4, 1.5), R1: (1, 1.5)}

FIGURE_LAYER = 31
# First layer at which ALL THREE models exceed 0.94 balanced accuracy (base 0.977, chem
# 0.983, reason 0.950 -- which is 0.9498 unrounded, hence 0.94 and not 0.95 as the
# threshold). It is also chem's own best probe layer, which is why the section can say
# chem is clean on the N/O boundary exactly where its headline probe number comes from. Error *composition* is only interpretable once a model has
# essentially solved the coarse task: below this depth every model misclassifies heavily
# in every direction, so the share of errors falling on one boundary says nothing about
# that boundary. Verified against probe/data/layer_curves.csv by check_claims.py.
ERR_FLOOR = 19
LAYERS = [0, 8, 16, 24, 31]
N_LAYERS = 32


def load(rel):
    """Read one Analysis CSV by path relative to `fc_group/Analysis/`."""
    path = os.path.join(ANALYSIS, rel)
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {os.path.relpath(path, REPO)} -- run the analyze_*.py that writes it")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(row, field):
    v = row[field]
    return float(v) if v not in ('', 'None') else float('nan')


def fmt(x, places=3):
    """Fixed-width number for a LaTeX cell; en-dash for missing."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '--'
    if x != x:
        return '--'
    return f'{x:.{places}f}'


def signed(x, places=3):
    """Signed number in math mode.

    Math mode is not cosmetic here: outside it LaTeX typesets a leading `-` as a hyphen,
    which is visibly shorter than a minus and misaligns a column of mixed signs.
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '--'
    return f'${x:+.{places}f}$'
