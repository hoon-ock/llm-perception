#!/usr/bin/env python3
"""Check that a smoke extraction produced activations of the shape the registry claims.

A smoke run that exits 0 has proved the model loads and a forward pass completes.
It has not proved the *result* is usable. The failure this guards against is
quiet: `extract_activations_subset.py` takes its layer count from the loaded model
(`len(model.model.layers)`), while every analysis downstream -- probe, analogy,
anisotropy -- resolves depth and width through `model_registry.get_model_config`.
If those disagree, extraction succeeds and the analyses silently read the wrong
layers.

So this asserts, per model: the expected layer files exist, each holds one row per
molecule, and the width matches `hidden_dim`.

    python fc_group/verify_smoke_output.py --model-name meta-llama/Llama-3.1-8B

Exits non-zero with a specific reason on any mismatch.
"""
import argparse
import os
import re
import sys

import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import get_model_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, 'config_extract_activation.yaml')
DEFAULT_CSV = os.path.join(HERE, 'functional_group_dataset.csv')


def expected_layers(num_layers):
    """The three layers the smoke jobs request: `0 middle top`.

    Mirrors `resolve_layers` in extract_activations_subset.py rather than
    hard-coding indices, so this stays correct across models of different depth.
    """
    return sorted({0, num_layers // 2, num_layers - 1})


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-name', required=True)
    p.add_argument('--entity-type', default='functional_group')
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--csv', default=DEFAULT_CSV)
    p.add_argument('--expect-rows', type=int, default=None,
                   help='rows expected per layer file; defaults to the molecule count '
                        '(one template, as the smoke jobs request)')
    args = p.parse_args()

    registry = get_model_config(args.model_name)
    with open(args.config) as fh:
        extraction = yaml.safe_load(fh)['extraction']
    save_dir = extraction.get('save_dir', 'activation_datasets')
    aggregation = extraction.get('aggregation', 'last')

    n_expected = args.expect_rows
    if n_expected is None:
        n_expected = len(pd.read_csv(args.csv))

    directory = os.path.join(save_dir, args.model_name.replace('/', '-'), args.entity_type)
    if not os.path.isdir(directory):
        print(f"FAIL {args.model_name}: no output directory {directory}")
        return 1

    problems = []
    for layer in expected_layers(registry['num_layers']):
        pattern = re.compile(
            rf'^{re.escape(args.entity_type)}\.{re.escape(aggregation)}\.[^.]+\.layer_{layer}\.pt$')
        matches = [f for f in os.listdir(directory) if pattern.match(f)]
        if len(matches) != 1:
            problems.append(f"layer {layer}: expected 1 file, found {len(matches)}: {matches}")
            continue
        tensor = torch.load(os.path.join(directory, matches[0]), map_location='cpu')
        if tensor.ndim != 2:
            problems.append(f"layer {layer}: expected a 2-D tensor, got shape {tuple(tensor.shape)}")
            continue
        rows, width = tensor.shape
        if width != registry['hidden_dim']:
            problems.append(
                f"layer {layer}: width {width} != registry hidden_dim {registry['hidden_dim']}")
        if rows != n_expected:
            problems.append(f"layer {layer}: {rows} rows, expected {n_expected}")
        if not torch.isfinite(tensor.float()).all():
            problems.append(f"layer {layer}: contains NaN or Inf")

    if problems:
        print(f"FAIL {args.model_name}:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    layers = expected_layers(registry['num_layers'])
    print(f"OK {args.model_name}: layers {layers} present, "
          f"{n_expected} x {registry['hidden_dim']} each, finite.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
