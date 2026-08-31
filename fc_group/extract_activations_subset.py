import argparse
import json
import os
import torch
import einops
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd

# Load configuration and model parameters
def load_config(config_file="config_extract_activation.yaml"):
    """
    Load the configuration file.

    Args:
        config_file (str): Path to the configuration YAML file.

    Returns:
        dict: Configuration data.
    """
    with open(config_file, 'r') as f:
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def load_tokenizer(model_name, hf_token):
    """
    Load the tokenizer for the specified model.

    Args:
        model_name (str): Name of the pre-trained model.
        hf_token (str): Hugging Face token.

    Returns:
        AutoTokenizer: Loaded tokenizer with pad_token set to eos_token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_name, hf_token, quantization_config):
    """
    Load the pre-trained model, using 4-bit quantization when requested and
    falling back to a plain fp16/bf16 load on MPS/CPU otherwise (bitsandbytes
    4-bit quantization only works on Linux+NVIDIA).

    Args:
        model_name (str): Name of the pre-trained model.
        hf_token (str): Hugging Face token.
        quantization_config (dict): Quantization configuration parameters.

    Returns:
        AutoModelForCausalLM: Loaded model.
    """
    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    compute_dtype = dtype_map.get(quantization_config.get("bnb_4bit_compute_dtype", "float16"), torch.float16)

    if quantization_config.get("load_in_4bit", True):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=quantization_config.get("bnb_4bit_use_double_quant", False),
            bnb_4bit_quant_type=quantization_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            use_auth_token=hf_token,
        )
    else:
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
            use_auth_token=hf_token,
        )
        model = model.to(device)
    return model

# Prepare input data
def get_batch_mask(prompts, tokenizer):
    """
    Tokenize the input prompts.

    Args:
        prompts (list of str): Input prompts.
        tokenizer (AutoTokenizer): Tokenizer.

    Returns:
        tuple: input_ids and attention_mask tensors.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    
    # Print number of valid tokens per input
    # print("Number of valid tokens per input:")
    # print(inputs["attention_mask"].sum(dim=1))
    return inputs["input_ids"], inputs["attention_mask"]

# Register activation hooks
def get_activation_hook(name, activations):
    """
    Create a forward hook to save activations.

    Args:
        name (str): Name of the layer.
        activations (dict): Dictionary to store activations.

    Returns:
        function: Hook function.
    """
    def hook(model, input, output):
        activations[name] = detach_tensor(output)
    return hook

def detach_tensor(tensor):
    """
    Detach tensor from computation graph.

    Args:
        tensor (torch.Tensor or tuple/list): Tensor to detach.

    Returns:
        torch.Tensor or tuple/list: Detached tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)(detach_tensor(x) for x in tensor)
    else:
        return tensor

def register_hooks(model, activations, layer_indices=None):
    """
    Register forward hooks for each layer in the model.

    Args:
        model (AutoModelForCausalLM): The model.
        activations (dict): Dictionary to store activations.
        layer_indices (set of int, optional): Only hook these layer indices.
            None hooks every layer (default behavior).

    Returns:
        list: List of hook handles.
    """
    hooks = []
    for i, layer in enumerate(model.model.layers):
        if layer_indices is not None and i not in layer_indices:
            continue
        hook = layer.register_forward_hook(get_activation_hook(f'layer_{i}', activations))
        hooks.append(hook)
    return hooks

# Retrieve activations
def get_activations(model, input_ids, batch_mask, layer_indices=None):
    """
    Get activations from the model for given inputs.

    Args:
        model (AutoModelForCausalLM): The model.
        input_ids (torch.Tensor): Input IDs tensor.
        batch_mask (torch.Tensor): Attention mask tensor.
        layer_indices (set of int, optional): Only capture these layer indices.

    Returns:
        dict: Activations per layer.
    """
    activations = {}
    hooks = register_hooks(model, activations, layer_indices)

    input_ids = input_ids.to(model.device)
    batch_mask = batch_mask.to(model.device)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=batch_mask, output_hidden_states=True)
    
    for hook in hooks:
        hook.remove()
    return activations

# Process activations
def process_activation_batch(activations, batch_mask, aggregation='last'):
    """
    Process activations based on the aggregation method.

    Args:
        activations (torch.Tensor): Activations tensor.
        batch_mask (torch.Tensor): Attention mask tensor.
        aggregation (str): Aggregation method ('last', 'mean', 'max', 'none').

    Returns:
        torch.Tensor: Processed activations.
    """
    if isinstance(activations, tuple):
        activations = activations[0]
    
    batch_mask = batch_mask.to(activations.device)

    if aggregation == 'last':
        # Get the activation of the last valid token
        last_ix = batch_mask.flip(dims=[1]).argmax(dim=1)
        processed_activations = activations[torch.arange(activations.size(0)), activations.size(1) - 1 - last_ix]
    
    elif aggregation == 'mean':
        # Mean activation of all valid tokens
        masked_activations = activations * batch_mask.unsqueeze(-1)
        valid_token_count = batch_mask.sum(dim=1, keepdim=True)
        processed_activations = masked_activations.sum(dim=1) / valid_token_count
    
    elif aggregation == 'max':
        # Max activation among all tokens
        masked_activations = activations * batch_mask.unsqueeze(-1)
        masked_activations[batch_mask == 0] = float('-inf')
        processed_activations = masked_activations.max(dim=1)[0]
    
    elif aggregation == 'none':
        # No aggregation, return activations for all valid tokens
        processed_activations = einops.rearrange(activations, 'b s d -> (b s) d')
        processed_activations = processed_activations[batch_mask.view(-1) == 1]
    
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    return processed_activations

# Get and process activations
def get_and_process_activations(model, tokenizer, prompts, aggregation='last', layer_indices=None):
    """
    Get and process activations for a set of prompts.

    Args:
        model (AutoModelForCausalLM): The model.
        tokenizer (AutoTokenizer): The tokenizer.
        prompts (list of str): Input prompts.
        aggregation (str): Aggregation method.
        layer_indices (set of int, optional): Only capture these layer indices.

    Returns:
        dict: Processed activations per layer.
    """
    input_ids, batch_mask = get_batch_mask(prompts, tokenizer)
    activations = get_activations(model, input_ids, batch_mask, layer_indices)

    processed_activations = {}
    for layer_name, layer_activations in activations.items():
        processed_activations[layer_name] = process_activation_batch(layer_activations, batch_mask, aggregation)
    
    return processed_activations

# Save activations

def save_activations(model_name, activations, entity_type, prompt_name, layer_ix, aggregation='last', save_dir='activation_datasets'):
    """
    Save activations to a specified directory as a .pt file.

    Args:
        model_name (str): Name of the model.
        activations (torch.Tensor): The activations to save.
        entity_type (str): Type of the entity (e.g., 'person', 'element').
        prompt_name (str): Name of the prompt.
        layer_ix (int): Index of the layer.
        aggregation (str): Type of aggregation used for activations.
        save_dir (str): Base directory where activations will be saved.
    """
    # Define the model-specific directory
    model_dir = os.path.join(save_dir, model_name.replace('/', '-'))
    
    # Define the save path
    activation_save_path = os.path.join(model_dir, entity_type)
    os.makedirs(activation_save_path, exist_ok=True)
    
    # Define the save file name
    save_name = f'{entity_type}.{aggregation}.{prompt_name}.layer_{layer_ix}.pt'
    save_path = os.path.join(activation_save_path, save_name)
    
    # Save the activations as a .pt file
    torch.save(activations, save_path)
    print(f"Activations saved at: {save_path}")


# Generate prompts
def generate_prompts(df, templates):
    """
    Generate prompts based on templates and DataFrame rows.

    Args:
        df (pandas.DataFrame): DataFrame containing entity data.
        templates (list of str): List of prompt templates.

    Returns:
        list of str: Generated prompts.
    """
    prompts = []
    for _, row in df.iterrows():
        for template in templates:
            try:
                prompt = template.format(**row.to_dict())
                prompts.append(prompt)
            except KeyError as e:
                print(f"Missing key in data for template: {e}")
    return prompts

def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-layer activations for configured entity/prompt sets.")
    parser.add_argument("--config", "-c", default="config_extract_activation.yaml",
                         help="Path to the extraction config YAML (default: config_extract_activation.yaml).")
    parser.add_argument("--entity-types", nargs="+", default=None,
                         help="Only process entities whose entity_type matches one of these (default: all entities in the config).")
    parser.add_argument("--max-templates", type=int, default=None,
                         help="If set, only use the first this-many templates per entity (for fast smoke-test runs).")
    parser.add_argument("--layers", nargs="+", default=None,
                         help="If set, only extract these layers instead of every layer. Accepts integer indices "
                              "and/or the keywords bottom/middle/top (e.g. --layers bottom middle top, or --layers 0 15 31).")
    parser.add_argument("--model-name", default=None,
                         help="Override extraction.model_name from the config file (e.g. "
                              "deepseek-ai/DeepSeek-R1-Distill-Llama-8B, Qwen/Qwen3-8B). "
                              "Default: use whatever the config file specifies.")
    return parser.parse_args()

def resolve_layers(layer_args, num_layers):
    """
    Resolve --layers arguments (integers and/or bottom/middle/top keywords)
    into a sorted list of layer indices. None means every layer.
    """
    if layer_args is None:
        return list(range(num_layers))

    keyword_map = {
        "bottom": 0,
        "middle": num_layers // 2,
        "top": num_layers - 1,
    }
    resolved = set()
    for token in layer_args:
        key = token.lower()
        if key in keyword_map:
            resolved.add(keyword_map[key])
        else:
            resolved.add(int(token))
    return sorted(resolved)

# Main processing function
def main():
    """
    Main function to extract activations from language models.

    Configuration is loaded from the file passed via --config (default:
    config_extract_activation.yaml). Key settings include:
    - extraction.model_name: Which model to use
    - extraction.batch_size: Batch size for processing
    - extraction.aggregation: How to aggregate token activations
    - extraction.save_dir: Directory to save activation files
    - extraction.quantization: Model quantization parameters
    - extraction.entities: List of entity types and their templates

    To change any of these settings, edit the config file, or pass --entity-types
    to run only a subset of the configured entities.
    """
    args = parse_args()

    # Load configuration
    config_data = load_config(args.config)
    HF_TOKEN = config_data.get("HF_TOKEN")

    # Get extraction configuration
    extraction_config = config_data.get("extraction", {})
    model_name = args.model_name or extraction_config.get("model_name", "meta-llama/Llama-2-7b-hf")
    batch_size = extraction_config.get("batch_size", 550)
    aggregation = extraction_config.get("aggregation", "last")
    base_save_dir = extraction_config.get("save_dir", "activation_datasets")
    quantization_config = extraction_config.get("quantization", {})
    entities = extraction_config.get("entities", [])

    if args.entity_types:
        entities = [e for e in entities if e["entity_type"] in args.entity_types]

    print(f"Using model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Aggregation method: {aggregation}")
    print(f"Save directory: {base_save_dir}")

    # Load tokenizer and model
    tokenizer = load_tokenizer(model_name, HF_TOKEN)
    model = load_model(model_name, HF_TOKEN, quantization_config)

    # Validate entities configuration
    if not entities:
        print("No matching entities found in configuration. Please check your config file and --entity-types filter.")
        return

    print(f"Found {len(entities)} entity types to process.")

    num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers.")

    selected_layers = resolve_layers(args.layers, num_layers)
    layer_indices = set(selected_layers)
    print(f"Extracting layers: {selected_layers}")

    # Process each entity type from configuration
    for entity in entities:
        entity_type = entity["entity_type"]
        data_file = entity["data_file"]
        templates = entity["templates"]
        prompt_name = entity["prompt_name"]

        if args.max_templates is not None:
            templates = templates[:args.max_templates]

        print(f"Processing entity type: {entity_type}")

        # Load data
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Skipping entity {entity_type}.")
            continue
        df = pd.read_csv(data_file)

        # Generate prompts
        prompts = generate_prompts(df, templates)
        print(f"Generated {len(prompts)} prompts for entity type '{entity_type}'.")

        # Each batch needs only one forward pass: get_and_process_activations already
        # returns every layer's activations in one call, so we accumulate per layer
        # across batches instead of rerunning the model once per (layer, batch) pair.
        layer_activations = {layer_ix: [] for layer_ix in selected_layers}

        for start_ix in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start_ix:start_ix + batch_size]
            batch_num = start_ix // batch_size + 1

            try:
                processed_activations = get_and_process_activations(model, tokenizer, batch_prompts, aggregation, layer_indices)
                for layer_ix in selected_layers:
                    layer_key = f'layer_{layer_ix}'
                    if layer_key in processed_activations:
                        layer_activations[layer_ix].append(processed_activations[layer_key])
                    else:
                        print(f"Layer {layer_ix} not found in activations for batch {batch_num}.")
                print(f"Processed batch {batch_num} ({len(batch_prompts)} prompts) for entity '{entity_type}'.")

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                print(f"Out of memory: {e}. Skipping batch {batch_num}.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                continue

        # Concatenate and save the accumulated activations, one file per layer
        for layer_ix in selected_layers:
            if layer_activations[layer_ix]:
                concatenated_activations = torch.cat(layer_activations[layer_ix], dim=0)
                save_activations(
                    model_name=model_name,
                    activations=concatenated_activations,
                    entity_type=entity_type,
                    prompt_name=prompt_name,
                    layer_ix=layer_ix,
                    aggregation=aggregation,
                    save_dir=base_save_dir
                )
            else:
                print(f"No activations found for Layer {layer_ix}.")

        print(f"Entity '{entity_type}' activations processed and saved.")

    print("All activations processed and saved.")

if __name__ == "__main__":
    main()