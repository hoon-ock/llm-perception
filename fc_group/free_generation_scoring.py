#!/usr/bin/env python3
"""Adjudicating what a model's free generation actually commits to.

`generation_eval.py` scores every functional-group name as a teacher-forced continuation and
takes the argmax. That asks which of 20 names the model ranks highest -- a question it can
always answer. This asks a different one: left to continue the sentence on its own, does the
model produce the name?

The two come apart, and the gap is the point. The base model ranks the right class better
than R1-Distill (0.837 vs 0.804 forced choice) and produces it far worse (0.585 vs 0.789),
because 13% of its continuations enumerate the option set instead of answering:

    prompt  'The functional-group identity of Pentanal (CCCCC=O) is '
    output  '\nA. aldehyde\nB. ketone'

The gold string is present and nothing was answered. Scoring by containment would pay the
model for that, and would pay it most where it is failing worst -- so every generation is
sorted into exactly one of five outcomes, with format failures kept as outcomes of their own
rather than folded into "wrong" or, worse, credited as right:

    answer_correct   one unambiguous class name, and it is the gold one
    answer_wrong     one unambiguous class name, and it is not
    underspecified   only a superclass ("carbonyl", "alkyl halide") -- true of 4-5 classes
    non_answer       no group named: describes the molecule, restates the prompt
    malformed        an option list, a blank-fill, a preamble, a hedge, a self-retraction

Format failures are settled *before* any class match, which is what stops an enumeration
containing the gold word from scoring as correct.

`underspecified` is reported twice over: as its own rate, and folded into
`superclass_credit_accuracy` alongside `answer_correct`. Naming the family but not the
member is a different failure from naming nothing, and the gap between that and
`strict_accuracy` is how much of a model's shortfall is resolution rather than knowledge.

Imported by both `generation_eval.py`, which adjudicates on the HPC as it generates, and
`Analysis/generation/score_generations.py`, which re-derives from a stored `generations.csv`.
One copy, because two would drift. Standard library only, so importing it on a GPU node costs
nothing.
"""
import re
import statistics

# The class lexicon is deliberately NOT `generation_eval.SURFACE_FORMS`. Those strings are
# teacher-forced as token sequences and decide the primary forced-choice metric; widening
# them to parse free text would silently move Table 1. This is a separate, read-only
# superset used for parsing only.
CLASS_FORMS = {
    'alcohol':          ['alcohol', 'hydroxyl', '-OH'],
    'aldehyde':         ['aldehyde'],
    'alkyl bromide':    ['alkyl bromide', 'bromide', 'bromo', 'bromine'],
    'alkyl chloride':   ['alkyl chloride', 'chloride', 'chloro', 'chlorine'],
    'alkyl fluoride':   ['alkyl fluoride', 'fluoride', 'fluoro', 'fluorine'],
    'alkyl iodide':     ['alkyl iodide', 'iodide', 'iodo', 'iodine'],
    'amide':            ['amide'],
    'amine':            ['amine', 'amino'],
    'carboxylic acid':  ['carboxylic acid', 'carboxyl'],
    'ester':            ['ester'],
    'ether':            ['ether'],
    'imine':            ['imine'],
    'ketone':           ['ketone'],
    'nitrile':          ['nitrile', 'cyano'],
    'nitro':            ['nitro'],
    'none (alkane)':    ['alkane', 'none', 'hydrocarbon', 'aliphatic', 'paraffin'],
    'sulfone':          ['sulfone'],
    'sulfoxide':        ['sulfoxide'],
    'thioether':        ['thioether', 'sulfide'],
    'thiol':            ['thiol', 'mercaptan', 'mercapto', 'sulfhydryl', '-SH'],
}

# Forms beyond `generation_eval.SURFACE_FORMS`: the element or the group formula, which in
# this dataset picks out exactly one class ("the bromine atom" can only be the alkyl
# bromide; "-SH" can only be the thiol). --strict-names-only drops these back out.
LENIENT_FORMS = {
    'alcohol': {'-OH'}, 'alkyl bromide': {'bromine'}, 'alkyl chloride': {'chlorine'},
    'alkyl fluoride': {'fluorine'}, 'alkyl iodide': {'iodine'},
    'none (alkane)': {'hydrocarbon', 'aliphatic', 'paraffin'},
    'thiol': {'mercapto', '-SH'},
}

# True superclasses: chemically correct of four or five classes at once, so they cannot be
# resolved to one. Kept out of CLASS_FORMS on purpose -- crediting "carbonyl" to any single
# class would inflate it at the others' expense, the same reasoning
# `generation_eval.SURFACE_FORMS` gives for excluding it there.
SUPERCLASS = re.compile(
    r'\b(?:carbonyl|halides?|halogens?|halogeno-?alkanes?|halo-?alkanes?'
    r'|alkyl\s+(?:group|chain))\b', re.I)

# Shapes that are not an answer at all, checked before any class match so that an option
# list containing the gold word is never credited.
OPTION_LINE = re.compile(r'(?m)(?:^|\n)\s*[A-E][.):]\s')
# The same enumeration run inline rather than down lines (": a) aldehyde b) amine c").
# Two markers are required: one lone "a)" is too easy to hit by accident.
OPTION_INLINE = re.compile(r'[a-e][.):]\s.*?\b[a-e][.):]\s', re.I)
# What separates two class names decides whether the model hedged or built a compound
# noun. A coordinator or a retraction means it named two candidates and picked neither;
# bare adjacency means a head-final compound -- "carboxylic acid ester" is an ester,
# "aliphatic nitrile" is a nitrile -- and the head is the commitment.
COORDINATOR = re.compile(
    r'(?:\bor\b|\band\b|\bversus\b|\bvs\b|\bbut\b|\bwait\b|\bno\b|[/,;])', re.I)
BLANK_FILL = re.compile(r'_{3,}')
PREAMBLE = re.compile(r'\b(?:Okay|Alright|Hmm)\b\s*,?\s*so\b', re.I)
# A self-retraction inside the answer span: the model named a class and took it back
# without settling on another ("as a thiol, but wait, no,"). It committed to nothing, so
# crediting the name it happened to utter first would be reading a coin-flip as an answer.
RETRACTION = re.compile(r'\bbut\s+wait\b|\bwait\s*,?\s*no\b|\bactually\s*,?\s*no\b', re.I)
# Chem-R emits CamelCase class names -- `CarboxylicAcid`, `PrimaryAmine` -- an SFT output
# convention rather than a different answer. `\bcarboxylic acid\b` cannot match the first,
# and `\bamine\b` cannot match inside the second because `yA` carries no word boundary.
# Opening the boundary is a *fallback*, not a rewrite: `SulfOxide` already matches
# `\bsulfoxide\b` under re.I, and splitting unconditionally breaks 8 rows that currently
# score correct. Tried only when the as-written reading named nothing, so it can add a
# commitment and never retract one.
CAMEL_BOUNDARY = re.compile(r'(?<=[a-z])(?=[A-Z])')

RESPONSE_TYPES = ('answer_correct', 'answer_wrong', 'underspecified',
                  'non_answer', 'malformed')


def build_matcher(strict_names_only=False):
    """One alternation over every surface form, longest first.

    Longest-first so `alkyl bromide` wins over `bromide` and `carboxylic acid` over
    `carboxyl`. Word boundaries on both sides rather than a lookbehind alone: a bare
    `(?<![A-Za-z])bromo` fires inside `Bromopentane` and `(?<![A-Za-z])none` fires inside
    `nonenal`, both of which appear in this dataset. The optional plural still admits
    "the amines are". Formula forms start with a hyphen, where `\\b` would demand a word
    character before it, so they get their own guard.
    """
    forms = {}
    for label, names in CLASS_FORMS.items():
        drop = LENIENT_FORMS.get(label, set()) if strict_names_only else set()
        for name in names:
            if name not in drop:
                forms[name.lower()] = label
    ordered = sorted(forms, key=len, reverse=True)
    parts = []
    for name in ordered:
        if name[0].isalnum():
            parts.append(r'\b' + re.escape(name) + r'(?:e?s)?\b')
        else:
            parts.append(r'(?<![\w-])' + re.escape(name) + r'\b')
    return re.compile('|'.join(parts), re.I), forms


def mentions(text, pattern, forms):
    """(label, form, start, end) for every surface form in `text`, in order of appearance."""
    out = []
    for m in pattern.finditer(text):
        raw = m.group(0).lower()
        # The plural is optional in the pattern, so strip it back off to look the form up.
        # Singular first, then -s, then -es -- stripping 'es' from 'amines' would leave
        # 'amin' and lose the match.
        label = next((forms[k] for k in (raw, raw[:-1], raw[:-2]) if k in forms), None)
        if label:
            out.append((label, m.group(0), m.start(), m.end()))
    return out


def adjudicate(generation, iupac_name, true_label, pattern, forms):
    """Classify one generation into exactly one of RESPONSE_TYPES.

    Order matters and is the whole design: format failures are settled before any class
    match, so an enumeration that happens to list the gold class scores as `malformed`
    rather than correct.
    """
    text = generation or ''
    # IUPAC names encode the group -- `Pentanal`, `1-Bromopentane` -- so an echo of the
    # prompt would hand over the gold answer for free. Affects 5.5% of base-model rows.
    if iupac_name:
        text = re.sub(re.escape(str(iupac_name)), ' ', text, flags=re.I)

    if (OPTION_LINE.search(text) or OPTION_INLINE.search(text)
            or BLANK_FILL.search(text) or PREAMBLE.search(text)
            or RETRACTION.search(text)):
        return 'malformed', '', '', int(bool(gold_present(text, true_label, pattern, forms)))

    clause = first_clause(text)
    found = mentions(clause, pattern, forms)
    if not found:
        # Re-read with CamelCase boundaries opened. All three names are rebound together so
        # that hedged()'s gap offsets and gold_present's text stay consistent with the
        # reading that produced `found`.
        alt = CAMEL_BOUNDARY.sub(' ', text)
        alt_clause = first_clause(alt)
        alt_found = mentions(alt_clause, pattern, forms)
        if alt_found:
            text, clause, found = alt, alt_clause, alt_found
    contained = int(bool(gold_present(text, true_label, pattern, forms)))

    if len({label for label, _, _, _ in found}) > 1:
        if hedged(clause, found):
            return 'malformed', '', '', contained
        # A head-final compound, so the commitment is the last name, not the first.
        label, form = found[-1][0], found[-1][1]
        kind = 'answer_correct' if label == true_label else 'answer_wrong'
        return kind, label, form, contained
    if found:
        label, form = found[0][0], found[0][1]
        kind = 'answer_correct' if label == true_label else 'answer_wrong'
        return kind, label, form, contained
    if SUPERCLASS.search(clause):
        return 'underspecified', '', '', contained
    return 'non_answer', '', '', contained


def first_clause(text):
    """The span the model's commitment lives in: the first clause that says anything.

    Splitting on [.\n] and taking element zero is wrong whenever the continuation opens
    with a newline or a colon ("\nHexane is an alkane with six carbon"), which leaves an
    empty element zero and hides the answer in element one.
    """
    for part in re.split(r'[.\n]', text):
        if re.search(r'\w', part):
            return part
    return text


def hedged(clause, found):
    """True when two class names are listed as alternatives rather than composed.

    Reads the text between consecutive mentions of different classes. A coordinator or a
    retraction there -- "aldehyde/ketone", "alcohol and ether", "a thiol, but wait, no" --
    means the model offered a menu and picked nothing. Bare adjacency inside a couple of
    words is a head-final compound noun instead, and gets resolved to its head.
    """
    for (label_a, _, _, end), (label_b, _, start, _) in zip(found, found[1:]):
        if label_a == label_b:
            continue
        gap = clause[end:start]
        if COORDINATOR.search(gap) or len(gap.split()) > 2:
            return True
    return False


def gold_present(text, true_label, pattern, forms):
    """The permissive reading: does the gold class appear anywhere at all?"""
    return any(label == true_label for label, _, _, _ in mentions(text, pattern, forms))


def summarize(model, rows, entity_type):
    n = len(rows)
    counts = {k: sum(r['response_type'] == k for r in rows) for k in RESPONSE_TYPES}
    if sum(counts.values()) != n:
        raise SystemExit(f"{model}: response types sum to {sum(counts.values())}, not {n}")
    committed = counts['answer_correct'] + counts['answer_wrong']
    labels = sorted({r['true_label'] for r in rows})
    recall = {}
    for label in labels:
        sub = [r for r in rows if r['true_label'] == label]
        recall[label] = sum(r['correct_strict'] for r in sub) / len(sub)
    return {
        'model': model, 'entity_type': entity_type, 'n_prompts': n,
        'n_classes_present': len(labels),
        'strict_accuracy': round(counts['answer_correct'] / n, 6),
        'balanced_strict_accuracy': round(sum(recall.values()) / len(recall), 6),
        'commit_rate': round(committed / n, 6),
        'conditional_accuracy': (round(counts['answer_correct'] / committed, 6)
                                 if committed else None),
        # Partial credit for naming the family without resolving it: "carbonyl" for an
        # ester, "alkyl halide" for a chloride. The gap to strict_accuracy is how much of
        # a model's shortfall is failing to resolve a class it has already located, rather
        # than not locating one at all -- and the two are not the same failure. Note this
        # is NOT bounded by containment_accuracy: a superclass name usually does not
        # contain the gold string, so the two permissive readings are not nested.
        #
        # Deliberately unconditional on whether the superclass actually contains the gold
        # class, matching how `adjudicate` assigns `underspecified` in the first place. It
        # costs a little: SUPERCLASS also matches "alkyl group"/"alkyl chain", which names
        # a carbon skeleton rather than a family, and 7 base and 11 R1-Distill rows are
        # credited for saying it about a halide. No model commits the strong error of
        # naming a family that excludes the gold class, so the looseness stays bounded.
        'superclass_credit_accuracy': round(
            (counts['answer_correct'] + counts['underspecified']) / n, 6),
        # The naive "gold appears anywhere" score, carried as a declared upper bound the
        # way `raw` and `length_normalized` are carried side by side in
        # analyze_generation.py. The gap to strict_accuracy is how much a containment
        # metric would have paid the model for enumerating instead of answering.
        'containment_accuracy': round(sum(r['gold_contained'] for r in rows) / n, 6),
        'malformed_rate': round(counts['malformed'] / n, 6),
        'underspecified_rate': round(counts['underspecified'] / n, 6),
        'non_answer_rate': round(counts['non_answer'] / n, 6),
        'answer_wrong_rate': round(counts['answer_wrong'] / n, 6),
    }, recall


def summarize_by_template(rows):
    """The same rates, computed within each prompt template.

    The 10 templates ask the same chemical question in 10 phrasings, so the spread across
    them is the model's sensitivity to wording rather than to chemistry. It is worth having
    because it is large and lopsided: on the committed data the base model runs 0.196 to
    0.880 across templates while the chemistry-tuned model runs 0.685 to 1.000. The single
    worst case, template 8, is 75% malformed for the base model -- almost all of its format
    collapse sits in one phrasing, which an overall rate averages away.

    Costs nothing: these are the same rows, grouped.
    """
    out = {}
    for ti in sorted({int(r['template_index']) for r in rows}):
        sub = [r for r in rows if int(r['template_index']) == ti]
        block, _ = summarize(f'template {ti}', sub, None)
        out[ti] = {k: v for k, v in block.items()
                   if k not in ('model', 'entity_type', 'n_classes_present')}
    return out


def template_spread(by_template, field='strict_accuracy'):
    """Spread of one rate across the templates, with the extremes named.

    Sample standard deviation (ddof=1, `statistics.stdev`): the 10 templates are a sample of
    the phrasings the question could take, not the population of interest. Naming argmin and
    argmax matters more than the std here -- the interesting fact is *which* phrasing breaks
    a model, not only that some does.
    """
    if len(by_template) < 2:
        return None
    values = {ti: b[field] for ti, b in by_template.items()}
    lo = min(values, key=values.get)
    hi = max(values, key=values.get)
    return {
        f'{field}_std': round(statistics.stdev(values.values()), 6),
        f'{field}_min': values[lo], 'argmin_template': lo,
        f'{field}_max': values[hi], 'argmax_template': hi,
    }


def adjudicate_rows(generations, model=None, strict_names_only=False):
    """Adjudicate a list of generation records into scored rows.

    Takes whatever carries `generation`, `iupac_name`, `true_label` and `template_index` --
    dicts straight out of `generation_eval.py`'s loop, or `csv.DictReader` rows read back
    from a stored generations.csv. Both routes land on the same rows, which is what lets the
    HPC job and the offline rescore produce identical blocks.
    """
    pattern, forms = build_matcher(strict_names_only)
    out = []
    for r in generations:
        kind, pred, form, contained = adjudicate(
            r['generation'], r['iupac_name'], r['true_label'], pattern, forms)
        out.append({
            'model': model, 'iupac_name': r['iupac_name'],
            'template_index': r['template_index'], 'true_label': r['true_label'],
            'response_type': kind, 'predicted_label': pred, 'matched_form': form,
            'correct_strict': int(kind == 'answer_correct'),
            'gold_contained': contained,
            'generation': r['generation'],
        })
    return out


def build_block(model, rows, entity_type, lexicon='lenient'):
    """The `free_generation` block written into summary.json.

    Built here rather than at either call site so the block `generation_eval.py` writes on
    the HPC and the one `score_generations.py` re-derives from a stored generations.csv are
    the same object by construction, not by two implementations agreeing.
    """
    overall, recall = summarize(model, rows, entity_type)
    by_template = summarize_by_template(rows)
    block = {
        'lexicon': lexicon,
        'source': 'fc_group/free_generation_scoring.py',
        **{k: v for k, v in overall.items() if k not in ('model', 'entity_type')},
        'per_class_strict_recall': {c: round(v, 6) for c, v in sorted(recall.items())},
        'by_template': {str(ti): b for ti, b in sorted(by_template.items())},
    }
    spread = template_spread(by_template)
    if spread:
        block['template_spread'] = spread
    return block
