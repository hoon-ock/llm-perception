#!/usr/bin/env python3
"""Behavioural readout: does the model actually say the right functional group?

Every other fc_group analysis probes frozen activations. None of them records
whether the model *answers correctly*. This does, by scoring what the model
would emit at the end of the same declarative prompts the probe reads its
activations from.

Two readouts:

  forced choice (primary) -- score every functional-group name as a continuation
    by teacher forcing and take the argmax. This is the same 20-way decision the
    probe makes, so the two are directly comparable rather than merely
    correlated. It also sidesteps parsing free text, where "hydroxyl", "an
    alcohol group" and "-OH" would all be scored wrong by exact match and the
    number would measure the parser rather than the model. And it yields a full
    distribution, so log-loss and top-1 margin come for free -- the
    resolution-at-ceiling metrics the probe lacks, since run_cv keeps only the
    argmax of predict_proba.

  free generation (secondary) -- greedy continuation, logged verbatim, for
    qualitative error analysis only. Never scored by string match.

No activation extraction is needed, so this runs on any prompt condition
immediately.

    python fc_group/generation_eval.py --model-name phenixace/Chem-R-Faithful \
        --entity-types functional_group --max-templates 1
"""
import argparse
import json
import math
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM  # noqa: E402

from extract_activations_subset import (  # noqa: E402
    generate_prompts, load_config, load_model, load_tokenizer,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, 'config_extract_activation.yaml')
DEFAULT_OUTPUT_DIR = os.path.join(HERE, 'Results', 'generation_eval')
LABEL_COLUMN = 'functional_group'

# Surface forms per class, scored by log-sum-exp over the forms.
#
# This map is the main judgement call in the file, so the reasoning is explicit:
#
#  * `none (alkane)` is a dataset label, not something a model would ever emit --
#    without alternatives that class would score zero for reasons unrelated to
#    chemistry.
#  * Several classes have synonyms a model may legitimately prefer (sulfide for
#    thioether, mercaptan for thiol). Excluding them would count a right answer
#    as wrong.
#  * Forms that are ambiguous *between* classes are deliberately excluded.
#    "carbonyl" is the clearest case: it is equally true of aldehyde, ketone,
#    ester and carboxylic acid, so crediting it to any one of them would inflate
#    that class at the others' expense.
SURFACE_FORMS = {
    'alcohol':          ['alcohol', 'hydroxyl'],
    'aldehyde':         ['aldehyde'],
    'alkyl bromide':    ['alkyl bromide', 'bromide', 'bromo'],
    'alkyl chloride':   ['alkyl chloride', 'chloride', 'chloro'],
    'alkyl fluoride':   ['alkyl fluoride', 'fluoride', 'fluoro'],
    'alkyl iodide':     ['alkyl iodide', 'iodide', 'iodo'],
    'amide':            ['amide'],
    'amine':            ['amine', 'amino'],
    'carboxylic acid':  ['carboxylic acid', 'carboxyl'],
    'ester':            ['ester'],
    'ether':            ['ether'],
    'imine':            ['imine'],
    'ketone':           ['ketone'],
    'nitrile':          ['nitrile', 'cyano'],
    'nitro':            ['nitro'],
    'none (alkane)':    ['alkane', 'none'],
    'sulfone':          ['sulfone'],
    'sulfoxide':        ['sulfoxide'],
    'thioether':        ['thioether', 'sulfide'],
    'thiol':            ['thiol', 'mercaptan', 'sulfhydryl'],
}


def build_candidates(tokenizer):
    """Tokenize each surface form once, as a continuation of a prompt.

    The declarative templates end with a trailing space ("... functional group ").
    Tokenizing the candidate after that space would split it differently from how
    the model would actually emit it, so the prompt is right-stripped and the
    space moved onto the candidate instead -- " alcohol" is usually one token,
    "alcohol" after a dangling space is not.
    """
    out = {}
    for label, forms in SURFACE_FORMS.items():
        out[label] = [
            (form, tokenizer(' ' + form, add_special_tokens=False).input_ids)
            for form in forms
        ]
        for form, ids in out[label]:
            if not ids:
                raise ValueError(f"surface form {form!r} tokenized to nothing")
    return out


@torch.no_grad()
def score_prompt(model, tokenizer, prompt, candidates, device):
    """Log-probability of every candidate continuation, raw and per-token.

    One forward pass per candidate batch: the prompt is shared, so the batch is
    prompt+candidate for each surface form, left-padded to a common length.
    """
    prompt_ids = tokenizer(prompt.rstrip(), return_tensors=None).input_ids

    flat = [(label, form, ids) for label, forms in candidates.items()
            for form, ids in forms]
    seqs = [prompt_ids + ids for _, _, ids in flat]
    width = max(len(s) for s in seqs)
    pad_id = tokenizer.pad_token_id or 0

    # Left padding keeps every sequence's real tokens flush against the right
    # edge, so the continuation positions line up without per-row bookkeeping.
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, seq in enumerate(seqs):
        input_ids[i, width - len(seq):] = torch.tensor(seq, dtype=torch.long)
        attention[i, width - len(seq):] = 1
    input_ids, attention = input_ids.to(device), attention.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)

    per_label = {}
    for i, (label, form, ids) in enumerate(flat):
        n = len(ids)
        # Token at position j is predicted by the logits at j-1.
        positions = range(width - n, width)
        total = sum(logprobs[i, j - 1, input_ids[i, j]].item() for j in positions)
        per_label.setdefault(label, []).append((form, total, total / n))
    return per_label


@torch.no_grad()
def greedy_continuation(model, tokenizer, prompt, device, max_new_tokens):
    ids = tokenizer(prompt.rstrip(), return_tensors='pt').input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def summarize(rows, labels):
    """Accuracy, balanced accuracy, log-loss and margin, for both normalizations."""
    def block(pred_key):
        correct = [r['true_label'] == r[pred_key] for r in rows]
        recall = {}
        for label in labels:
            sub = [r['true_label'] == r[pred_key] for r in rows if r['true_label'] == label]
            if sub:
                recall[label] = sum(sub) / len(sub)
        return {
            'accuracy': sum(correct) / len(correct) if correct else None,
            'balanced_accuracy': (sum(recall.values()) / len(recall)) if recall else None,
            'per_class_recall': recall,
        }

    # Chance is 1/(classes actually present), not 1/(classes scored). Balanced
    # accuracy averages recall over the classes that appear in the data, so on a
    # truncated run those two differ and quoting 1/20 would understate chance.
    present = sorted({r['true_label'] for r in rows})
    out = {'n_prompts': len(rows),
           'n_classes_scored': len(labels),
           'n_classes_present': len(present),
           'chance': 1.0 / len(present) if present else None,
           'raw': block('pred_raw'), 'length_normalized': block('pred_norm')}
    losses = [-math.log(max(r['p_true'], 1e-12)) for r in rows]
    out['log_loss'] = sum(losses) / len(losses) if losses else None
    out['mean_top1_margin'] = (sum(r['margin'] for r in rows) / len(rows)) if rows else None
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--model-name', default=None)
    p.add_argument('--entity-types', nargs='+', default=['functional_group'])
    p.add_argument('--max-templates', type=int, default=None)
    p.add_argument('--max-molecules', type=int, default=None,
                   help='truncate the dataset, for a quick smoke run. Takes the first '
                        'N rows, which are ordered by class -- see --sample-per-class '
                        'for a spread instead.')
    p.add_argument('--sample-per-class', type=int, default=None,
                   help='take N molecules from each functional group. A short run that '
                        'covers all 20 classes rather than the first N rows, which would '
                        'be one or two classes and make balanced accuracy meaningless.')
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--device-map', default=None,
                   help="accelerate device_map, e.g. 'auto'. When set, the model is "
                        "loaded directly with offloading instead of through the shared "
                        "load_model, which does a plain load followed by .to(device) and "
                        "so needs the whole model resident. Use this wherever the weights "
                        "do not fit in memory -- an 8B in fp16 is ~16GB.")
    p.add_argument('--max-memory', default=None,
                   help="per-device cap passed to accelerate, e.g. 'cpu=10GiB'. Repeat "
                        "with commas for several devices.")
    p.add_argument('--offload-folder', default=None,
                   help='scratch directory for layers accelerate spills to disk')
    p.add_argument('--max-new-tokens', type=int, default=10)
    p.add_argument('--no-generation', action='store_true',
                   help='skip the free-generation pass (scoring only)')
    p.add_argument('--ablate-molecule', action='store_true',
                   help='replace the molecule name and formula with a placeholder. '
                        'The sanity floor: with no molecule identity in the prompt, '
                        'balanced accuracy must fall to roughly chance. If it does '
                        'not, the scoring is leaking.')
    p.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    args = p.parse_args()

    config_data = load_config(args.config)
    hf_token = config_data.get('HF_TOKEN')
    extraction = config_data.get('extraction', {})
    model_name = args.model_name or extraction.get('model_name')
    quantization = extraction.get('quantization', {})
    quantization = {**quantization, 'load_in_4bit': bool(args.load_in_4bit)}

    entities = [e for e in extraction.get('entities', [])
                if e['entity_type'] in args.entity_types]
    if not entities:
        raise SystemExit(f"no entity types matching {args.entity_types} in {args.config}")

    print(f"Model: {model_name}")
    tokenizer = load_tokenizer(model_name, hf_token)
    if args.device_map:
        # Constrained-memory path: let accelerate place layers across devices and
        # spill the remainder to disk. Every forward pass then re-reads the
        # offloaded weights, so this is slow -- but it is bounded, whereas a plain
        # load of a model larger than RAM just swaps until the machine is unusable.
        max_memory = None
        if args.max_memory:
            max_memory = {}
            for item in args.max_memory.split(','):
                device, _, limit = item.partition('=')
                key = device.strip()
                max_memory[int(key) if key.isdigit() else key] = limit.strip()
        dtype_map = {'float16': torch.float16, 'float32': torch.float32,
                     'bfloat16': torch.bfloat16}
        dtype = dtype_map.get(quantization.get('bnb_4bit_compute_dtype', 'float16'),
                              torch.float16)
        print(f"  device_map={args.device_map} max_memory={max_memory} "
              f"offload_folder={args.offload_folder} dtype={dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, token=hf_token, torch_dtype=dtype,
            device_map=args.device_map, max_memory=max_memory,
            offload_folder=args.offload_folder)
    else:
        model = load_model(model_name, hf_token, quantization)
    model.eval()
    device = next(model.parameters()).device
    candidates = build_candidates(tokenizer)
    labels = sorted(SURFACE_FORMS)

    for entity in entities:
        entity_type = entity['entity_type']
        df = pd.read_csv(entity['data_file'])
        if args.sample_per_class:
            df = (df.groupby(LABEL_COLUMN, sort=False, group_keys=False)
                    .head(args.sample_per_class))
        if args.max_molecules:
            df = df.head(args.max_molecules)
        templates = entity['templates']
        if args.max_templates:
            templates = templates[:args.max_templates]

        frame = df.copy()
        if args.ablate_molecule:
            for column in ('iupac_name', 'formula', 'smiles', 'condensed_formula'):
                if column in frame.columns:
                    frame[column] = 'X'
        prompts = generate_prompts(frame, templates)
        expected = len(frame) * len(templates)
        if len(prompts) != expected:
            raise SystemExit(
                f"{entity_type}: {len(prompts)} prompts but expected {expected} -- "
                f"a template placeholder did not resolve (generate_prompts skips those)")

        truth = [row[LABEL_COLUMN] for _, row in df.iterrows() for _ in templates]
        names = [row['iupac_name'] for _, row in df.iterrows() for _ in templates]
        tmpl_ix = [i for _ in range(len(df)) for i in range(len(templates))]

        rows, generations = [], []
        for k, (prompt, true_label, name, ti) in enumerate(
                zip(prompts, truth, names, tmpl_ix)):
            scored = score_prompt(model, tokenizer, prompt, candidates, device)
            # log-sum-exp over a class's surface forms: the class is credited for
            # whichever wording the model prefers, not just the canonical one.
            raw = {lab: torch.logsumexp(torch.tensor([t for _, t, _ in v]), 0).item()
                   for lab, v in scored.items()}
            norm = {lab: max(n for _, _, n in v) for lab, v in scored.items()}
            best_form = {lab: max(v, key=lambda x: x[1])[0] for lab, v in scored.items()}

            pred_raw = max(raw, key=raw.get)
            pred_norm = max(norm, key=norm.get)
            ordered = sorted(raw.values(), reverse=True)
            denom = torch.logsumexp(torch.tensor(list(raw.values())), 0).item()
            p_true = math.exp(raw[true_label] - denom)

            rows.append({
                'iupac_name': name, 'template_index': ti, 'true_label': true_label,
                'pred_raw': pred_raw, 'pred_norm': pred_norm,
                'correct_raw': int(pred_raw == true_label),
                'correct_norm': int(pred_norm == true_label),
                'best_surface_form': best_form[pred_raw],
                'p_true': p_true, 'margin': ordered[0] - ordered[1],
                **{f'logp_{lab}': raw[lab] for lab in labels},
                **{f'logpnorm_{lab}': norm[lab] for lab in labels},
            })
            if not args.no_generation:
                generations.append({
                    'iupac_name': name, 'template_index': ti, 'true_label': true_label,
                    'prompt': prompt,
                    'generation': greedy_continuation(
                        model, tokenizer, prompt, device, args.max_new_tokens),
                })
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(prompts)} prompts")

        slug = model_name.replace('/', '-')
        out = os.path.join(args.output_dir, slug, entity_type, 'data')
        os.makedirs(out, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, 'scores.csv'), index=False)
        if generations:
            pd.DataFrame(generations).to_csv(
                os.path.join(out, 'generations.csv'), index=False)
        summary = {'model': model_name, 'entity_type': entity_type,
                   'n_templates': len(templates), 'n_molecules': len(df),
                   'ablate_molecule': bool(args.ablate_molecule),
                   **summarize(rows, labels)}
        with open(os.path.join(out, 'summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=2)

        print(f"{entity_type}: balanced_acc raw={summary['raw']['balanced_accuracy']:.4f} "
              f"norm={summary['length_normalized']['balanced_accuracy']:.4f} "
              f"| log_loss={summary['log_loss']:.4f} (chance {summary['chance']:.4f})")
        print(f"  wrote {out}")


if __name__ == '__main__':
    main()
