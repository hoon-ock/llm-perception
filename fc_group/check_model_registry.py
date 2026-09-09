#!/usr/bin/env python3
"""Pre-flight for a multi-model sweep: is every registry model reachable and correct?

`fc_group/model_registry.py` carries `hidden_dim` and `num_layers` for each model,
hand-entered -- its own docstring says they are not fetched live. Nothing at
extraction time checks them: `extract_activations_subset.py` reads the layer count
off the *loaded model* (`len(model.model.layers)`), so a wrong registry entry does
not fail the extraction. It fails much later and much quieter, by mis-indexing
layers in the probe, analogy and anisotropy analyses, which all resolve depth
through `get_model_config`.

This compares the registry against each model's real `config.json` on HuggingFace
and reports how much weight data a sweep would have to download. CPU-only, no
model loading, no torch, and deliberately no numpy -- the HCC login node caps
per-user threads and numpy's OpenBLAS backend trips it, so this stays importable
there without the usual thread-limit exports.

    python fc_group/check_model_registry.py

Exits non-zero if any registry entry disagrees with the published config, so it
can gate a sweep.
"""
import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_registry import MODEL_CONFIGS  # noqa: E402

CONFIG_URL = 'https://huggingface.co/{model}/raw/main/config.json'
TREE_URL = 'https://huggingface.co/api/models/{model}/tree/main?recursive=1'
WEIGHT_SUFFIXES = ('.safetensors', '.bin')
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'config_extract_activation.yaml')


def read_token(config_path):
    """HF token from the extraction config, falling back to the environment.

    Same source `extract_activations_subset.py` uses (`config_data["HF_TOKEN"]`),
    so this resolves the gated meta-llama repos on any machine where extraction
    itself would work. Without it those two return HTTP 401 and are reported as
    gated rather than as failures -- that is an artifact of running unauthenticated,
    not a problem with the model.
    """
    for env in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        if os.environ.get(env):
            return os.environ[env]
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(config_path) as fh:
            return (yaml.safe_load(fh) or {}).get('HF_TOKEN')
    except FileNotFoundError:
        return None


def fetch_json(url, token, timeout):
    req = urllib.request.Request(url)
    if token:
        req.add_header('Authorization', f'Bearer {token}')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def weight_bytes(model, token, timeout):
    """Total size of the weight shards, for a download estimate."""
    try:
        files = fetch_json(TREE_URL.format(model=model), token, timeout)
    except Exception:
        return None
    return sum(f.get('size', 0) for f in files
               if f.get('path', '').endswith(WEIGHT_SUFFIXES))


def architecture_supported(model_type):
    """Whether the installed transformers recognises this architecture.

    This is the exact lookup that raises `KeyError: 'qwen3'` deep inside
    AutoConfig.from_pretrained -- a failure that costs a GPU allocation and a
    model download to discover, but is answerable here in milliseconds. Returns
    (supported, installed_version).
    """
    try:
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except ImportError:
        return None, None
    return model_type in CONFIG_MAPPING_NAMES, transformers.__version__


def cached_locally(model):
    """Whether HF already has this model unpacked, so it need not be re-fetched."""
    home = os.environ.get('HF_HOME')
    if not home:
        return None
    slug = 'models--' + model.replace('/', '--')
    return os.path.isdir(os.path.join(home, 'hub', slug)) or \
        os.path.isdir(os.path.join(home, slug))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=DEFAULT_CONFIG,
                   help='extraction YAML to read HF_TOKEN from')
    p.add_argument('--timeout', type=float, default=25.0)
    p.add_argument('--skip-sizes', action='store_true',
                   help='skip the tree API calls used for the download estimate')
    args = p.parse_args()

    socket.setdefaulttimeout(args.timeout)
    token = read_token(args.config)
    tf = architecture_supported("llama")[1]
    print(f"transformers: {tf or 'not importable'}")
    print(f"HF token: {'found' if token else 'NOT found -- gated repos will report as gated'}\n")

    header = (f"{'model':44s} {'arch':>10s} {'hidden_dim':>18s} {'num_layers':>16s} "
              f"{'weights':>10s}  cached")
    print(header)
    print('-' * len(header))

    mismatches, gated, unreachable, unsupported = [], [], [], []
    total_bytes, to_download, tf_version = 0, 0, None
    for model, registry in MODEL_CONFIGS.items():
        cached = cached_locally(model)
        cached_str = '-' if cached is None else ('yes' if cached else 'no')
        try:
            config = fetch_json(CONFIG_URL.format(model=model), token, args.timeout)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                gated.append(model)
                print(f"{model:44s} {'gated (HTTP %d)' % exc.code:>40s} {'':>10s}  {cached_str}")
            else:
                unreachable.append((model, f'HTTP {exc.code}'))
                print(f"{model:44s} {'HTTP %d' % exc.code:>40s} {'':>10s}  {cached_str}")
            continue
        except Exception as exc:
            unreachable.append((model, type(exc).__name__))
            print(f"{model:44s} {type(exc).__name__:>40s} {'':>10s}  {cached_str}")
            continue

        model_type = config.get('model_type', '?')
        supported, tf_version = architecture_supported(model_type)
        if supported is False:
            unsupported.append((model, model_type))
        arch = model_type if supported is not False else f'{model_type}!'

        actual_dim = config.get('hidden_size')
        actual_layers = config.get('num_hidden_layers')
        dim_ok = actual_dim == registry['hidden_dim']
        layers_ok = actual_layers == registry['num_layers']
        if not dim_ok:
            mismatches.append((model, 'hidden_dim', registry['hidden_dim'], actual_dim))
        if not layers_ok:
            mismatches.append((model, 'num_layers', registry['num_layers'], actual_layers))

        size = None if args.skip_sizes else weight_bytes(model, token, args.timeout)
        if size:
            total_bytes += size
            if cached is not True:
                to_download += size
        size_str = '-' if not size else f'{size / 1e9:.1f} GB'

        print(f"{model:44s} {arch:>10s} "
              f"{('%s %s' % (actual_dim, 'ok' if dim_ok else '!= reg %s' % registry['hidden_dim'])):>18s} "
              f"{('%s %s' % (actual_layers, 'ok' if layers_ok else '!= reg %s' % registry['num_layers'])):>16s} "
              f"{size_str:>10s}  {cached_str}")

    print()
    if total_bytes:
        print(f"weights across all reachable models: {total_bytes / 1e9:.1f} GB")
        if os.environ.get('HF_HOME'):
            print(f"not yet cached under HF_HOME:        {to_download / 1e9:.1f} GB")
        else:
            print("set HF_HOME to also report how much is already cached")
    if gated:
        print(f"\ngated, reported without a token ({len(gated)}): {', '.join(gated)}")
        print("  Not a failure -- their registry entries can be confirmed against any "
              "activations already extracted for them.")
    if unreachable:
        print(f"\nUNREACHABLE ({len(unreachable)}):")
        for model, why in unreachable:
            print(f"  {model}: {why}")
    if unsupported:
        print(f"\nUNLOADABLE ON THIS TRANSFORMERS ({len(unsupported)}) "
              f"-- installed: {tf_version or 'unknown'}:")
        for model, model_type in unsupported:
            print(f"  {model}: model_type={model_type!r} not in CONFIG_MAPPING_NAMES")
        print("  These fail inside AutoConfig.from_pretrained after the weights download.")
        print("  Either upgrade transformers or drop them from the registry.")
        return 1
    if mismatches:
        print(f"\nREGISTRY MISMATCHES ({len(mismatches)}) -- fix fc_group/model_registry.py:")
        for model, field, reg, actual in mismatches:
            print(f"  {model}: {field} registry={reg} actual={actual}")
        return 1
    checked = len(MODEL_CONFIGS) - len(gated) - len(unreachable)
    print(f"\nOK: {checked}/{len(MODEL_CONFIGS)} registry entries verified against live configs.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
