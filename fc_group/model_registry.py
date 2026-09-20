"""Per-model architecture constants used by the fc_group activation-analysis scripts.

`default_layers` mirrors the existing convention of [first, middle, last] layer
indices (0-indexed, inclusive of the final decoder layer).

Verify hidden_dim/num_layers against each model's actual config.json on
HuggingFace before relying on these for a new model -- they're not fetched
live.
"""

MODEL_CONFIGS = {
    "meta-llama/Llama-3.1-8B": {
        "hidden_dim": 4096,
        "num_layers": 32,
        "default_layers": [0, 16, 31],
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "hidden_dim": 4096,
        "num_layers": 32,
        "default_layers": [0, 16, 31],
    },
    # The chemistry arm is three checkpoints in two tiers.
    #
    # Tier 1 -- controlled. Chem-R-8B and Chem-R-Faithful are both built on
    # Llama-3.1-8B (via -Instruct), so the config fingerprint matches the base
    # entry above exactly: vocab 128256, rope_theta 500000, 4096/32, 8 kv heads,
    # max_position_embeddings 131072. Sharing a base with a model already here is
    # the point -- chemistry-tuned vs base varies one thing, with the same
    # architecture and the same tokenizer. Chem-R-Faithful is in turn GRPO-trained
    # *from* Chem-R-8B with fabrication-gated rewards, so holding both separates
    # domain tuning from faithfulness tuning; with only the Faithful checkpoint
    # the two are confounded in the single arm that carries the result.
    "weidawang/Chem-R-8B": {
        "hidden_dim": 4096,
        "num_layers": 32,
        "default_layers": [0, 16, 31],
    },
    "phenixace/Chem-R-Faithful": {
        "hidden_dim": 4096,
        "num_layers": 32,
        "default_layers": [0, 16, 31],
    },
    # Tier 2 -- off-base, for external validity only. ChemDFM-v1.5-8B is an
    # independently trained chemistry model of the same size and the same shape
    # (4096/32, so every depth- and width-indexed analysis here still applies),
    # but it sits on Llama-3-8B, not 3.1: max_position_embeddings 8192,
    # rope_scaling null, vocab_size 128264. It answers "does the effect survive a
    # different lab's checkpoint" and nothing narrower -- a ChemDFM-vs-base gap
    # confounds domain training with the 3 -> 3.1 difference and must never be
    # reported as a controlled contrast. That is what Tier 1 is for.
    "OpenDFM/ChemDFM-v1.5-8B": {
        "hidden_dim": 4096,
        "num_layers": 32,
        "default_layers": [0, 16, 31],
    },
    "meta-llama/Llama-3.1-70B": {
        "hidden_dim": 8192,
        "num_layers": 80,
        "default_layers": [0, 40, 79],
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
        "hidden_dim": 8192,
        "num_layers": 80,
        "default_layers": [0, 40, 79],
    },
}


def default_five_layers(num_layers):
    """Initial, mid-init, mid, mid-final, final layer indices.

    A denser alternative to `default_layers` for analyses that plot a trend
    across depth, where three points are too few to see the shape.
    """
    return [
        0,
        num_layers // 4,
        num_layers // 2,
        (3 * num_layers) // 4,
        num_layers - 1,
    ]


def get_model_config(model_name):
    try:
        return MODEL_CONFIGS[model_name]
    except KeyError:
        raise KeyError(
            f"No entry for model_name={model_name!r} in MODEL_CONFIGS "
            f"(fc_group/model_registry.py). Add its hidden_dim/num_layers/"
            f"default_layers there before using it."
        )
