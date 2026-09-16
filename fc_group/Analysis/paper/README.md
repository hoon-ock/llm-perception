# The paper build

Everything that turns `fc_group/Analysis/*/data/*.csv` into the submission. The LaTeX itself
lives in `paper/`, which is an **Overleaf clone** and is gitignored here — Overleaf's git is
the version control for the `.tex` sources. This directory is what is tracked in this repo.

```bash
python fc_group/Analysis/paper/make_tables.py    # tables/*.tex
python fc_group/Analysis/paper/make_figures.py   # figures/*.pdf
python fc_group/Analysis/paper/check_numbers.py  # every numeral in the prose has a source
python fc_group/Analysis/paper/check_claims.py   # the specific values the argument rests on
```

All four default to `--out-dir <repo>/paper`.

## The rule everything here enforces

**No number is hand-typed into the LaTeX.** Tables are generated; figures are generated; the
two checkers close the one remaining hole, which is numbers typed into sentences. A re-run of
the analysis pipeline therefore updates the paper, and every value traces to the script that
produced it.

## Reproducible without `Results/`

Both generators read only `fc_group/Analysis/*/data/*.csv` and never
`fc_group/Results*`, which is gitignored. That is a test, not a convention — run them against
a checkout with no `Results/` present and they must still succeed:

```bash
mkdir -p /tmp/repro/fc_group && cp -R fc_group/Analysis /tmp/repro/fc_group/Analysis
cd /tmp/repro/fc_group/Analysis/paper && python3 make_tables.py --out-dir /tmp/repro/paper \
  && python3 make_figures.py --out-dir /tmp/repro/paper
```

If either needs `Results/`, a snapshot step is missing upstream. Three such gaps were closed
this way: `generation/` (the behavioural half of Table 1), `taxonomy/between_class_matrix.csv`
(the dendrogram figure), and `probe/confusion_errors.csv` (the nitrogen→oxygen counts, caught
by `check_numbers.py` after the prose quoted them from an uncommitted tree).

## Files

| File | Role |
|---|---|
| `_common.py` | paths, model order, display labels, the validated palette, number formatting |
| `make_tables.py` | T1 paradox, T2 taxonomy, T3 geometry, T4 ambiguity (appendix) |
| `make_figures.py` | F1 probe depth, F2 taxonomy trajectory, F3 dendrograms, F4 retrieval, F5 costs |
| `check_numbers.py` | fails if any numeral in `paper/sections/*.tex` has no released source |
| `check_claims.py` | fails if any value the argument depends on has shifted; run after touching any `analyze_*.py` |

## Colour

The palette in `_common.COLOR` is Okabe-Ito (`#0072B2` / `#D55E00` / `#009E73`), chosen by
running a colour-vision validator rather than by eye. The seaborn default this replaced put
red beside green at deuteranopic ΔE 7.3 — inside the band that is only legal with secondary
encoding. Every series additionally carries its own marker and dash pattern, so identity
survives greyscale printing. If you change a colour, re-validate rather than eyeballing it.

## Overleaf, via GitHub

The Overleaf project is built from the stock **ICLR 2026** template and linked to
**`github.com/hoon-ock/llm_perception`** — note the underscore; that is a different repo from
this code repo (`llm-perception`, hyphen). `paper/` here is a clone of it.

```bash
git clone https://github.com/hoon-ock/llm_perception.git paper   # from the repo root
```

Use the **HTTPS** remote. SSH to GitHub is not configured on this machine
(`Permission denied (publickey)`), while HTTPS works and `gh` is authenticated.

Sync is **manual in both directions**. After pushing here, pull it into Overleaf with
*Menu → GitHub → Pull GitHub changes into Overleaf*; after editing in Overleaf, push from
there before pulling here, or the two will diverge.

### Things the template decides for you

| | |
|---|---|
| The main document is **`iclr2026_conference.tex`** | Not `main.tex`. Renaming it would mean changing Overleaf's main-document setting by hand. |
| `\textwidth` is `5.5in` | Exactly `make_figures.WIDTH`, so figures need no resizing. |
| `iclr2026_conference.sty` does `\RequirePackage{natbib}`, and `math_commands.tex` does `\usepackage{amsmath,amsfonts,bm}` | Never load those four again — an option clash is the usual symptom. |
| **Strict 9-page main-text limit** | Three figures live in the appendix for this reason; see below. |
| `\iclrfinalcopy` is commented out | ICLR is double-blind and the style renders an anonymous author block. Non-anonymous submissions are rejected without review, so leave it commented and keep the repository URL in the appendix anonymized until camera-ready. |

### Float placement is a page-budget decision

Main text carries F1, F2 and all three tables. F3 (dendrograms), F4 (retrieval) and F5 (costs)
sit in `appendix.tex` under `\label{app:floats}`, and F4's argument is carried by a prose
sentence quoting the same hit@1 and chance numbers. None of the three carries a claim the
argument depends on: F3 is captioned as illustration rather than evidence, and F5 plots across
depth what T3 tabulates at the final layer. If the Intro and Discussion come in shorter than
budgeted, F4 is the first to move back.

### No local compile

There is no LaTeX toolchain on this machine, so **the first real compile happens in Overleaf**.
What can be checked here is structural — every `\input` resolves, every `\includegraphics`
exists, no dangling `\ref`, balanced braces and `$` — plus the two checkers above. Verify the
page count in Overleaf before writing out the Intro, Related Work and Discussion scaffolds, so
trimming happens in bullets rather than in finished prose.
