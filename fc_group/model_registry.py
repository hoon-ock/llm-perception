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
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": {
        "hidden_dim": 3584,
        "num_layers": 28,
        "default_layers": [0, 14, 27],
    },
    "Qwen/Qwen3-8B": {
        "hidden_dim": 4096,
        "num_layers": 36,
        "default_layers": [0, 18, 35],
    },
    "Qwen/Qwen2.5-Math-7B": {
        "hidden_dim": 3584,
        "num_layers": 28,
        "default_layers": [0, 14, 27],
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
    "Qwen/Qwen3-32B": {
        "hidden_dim": 5120,
        "num_layers": 64,
        "default_layers": [0, 32, 63],
    },
    "Qwen/Qwen2.5-Math-72B": {
        "hidden_dim": 8192,
        "num_layers": 80,
        "default_layers": [0, 40, 79],
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "hidden_dim": 5120,
        "num_layers": 64,
        "default_layers": [0, 32, 63],
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
