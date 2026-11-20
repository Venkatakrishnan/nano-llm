import json
from pathlib import Path
import shutil

import torch

from tokenizer import HuggingFaceBpeTokenizer


def map_to_llama(checkpoint):
    config = checkpoint.get("config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("Checkpoint must contain config and model_state_dict entries.")
    if config.get("tokenizer_format") != "gpt2":
        raise ValueError("Checkpoint must be retrained with --tokenizer-format gpt2.")
    if "bos_token_id" not in config or "eos_token_id" not in config:
        raise ValueError("Checkpoint has no tokenizer special-token IDs; retrain with the updated train.py.")

    required_settings = {
        "use_rope": True,
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
    }
    incompatible = [
        f"{name}={config.get(name)!r}"
        for name, expected in required_settings.items()
        if config.get(name) != expected
    ]
    if incompatible:
        raise ValueError(
            "Llama export requires the Llama-compatible architecture "
            f"({', '.join(incompatible)}). Retrain with RoPE, RMSNorm, and SwiGLU."
        )

    vocab_size = config["vocab_size"]
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    if hidden_dim % num_heads or (hidden_dim // num_heads) % 2:
        raise ValueError("Llama RoPE export requires an even head dimension.")
    if "block_size" not in config:
        raise ValueError("Checkpoint has no block_size; retrain with the updated train.py.")

    if not torch.equal(state["token_embedding.weight"], state["lm_head.weight"]):
        raise ValueError("Llama export currently requires tied token embedding and LM-head weights.")

    mapped = {
        "model.embed_tokens.weight": state["token_embedding.weight"],
        "lm_head.weight": state["lm_head.weight"],
        "model.norm.weight": state["norm_final.weight"],
    }
    for layer_index in range(num_layers):
        source = f"layers.{layer_index}."
        target = f"model.layers.{layer_index}."
        mapped[target + "input_layernorm.weight"] = state[source + "norm1.weight"]
        mapped[target + "post_attention_layernorm.weight"] = state[source + "norm2.weight"]
        for source_name, target_name in (
            ("attn.q_proj.weight", "self_attn.q_proj.weight"),
            ("attn.k_proj.weight", "self_attn.k_proj.weight"),
            ("attn.v_proj.weight", "self_attn.v_proj.weight"),
            ("attn.out_proj.weight", "self_attn.o_proj.weight"),
            ("mlp.gate_proj.weight", "mlp.gate_proj.weight"),
            ("mlp.up_proj.weight", "mlp.up_proj.weight"),
            ("mlp.down_proj.weight", "mlp.down_proj.weight"),
        ):
            mapped[target + target_name] = state[source + source_name]

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_size": hidden_dim,
        "intermediate_size": 4 * hidden_dim,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "head_dim": hidden_dim // num_heads,
        "max_position_embeddings": config["block_size"],
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "use_cache": True,
        "torch_dtype": "float32",
    }
    return mapped, hf_config


def export_hf(checkpoint_path, tokenizer_path, output_dir):
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_path}")

    tokenizer = HuggingFaceBpeTokenizer()
    tokenizer.load(tokenizer_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mapped_state, hf_config = map_to_llama(checkpoint)
    if tokenizer.vocab_size != hf_config["vocab_size"]:
        raise ValueError(
            f"Tokenizer has {tokenizer.vocab_size} tokens but the checkpoint expects "
            f"{hf_config['vocab_size']}."
        )
    if tokenizer.bos_id != hf_config["bos_token_id"] or tokenizer.eos_id != hf_config["eos_token_id"]:
        raise ValueError("Tokenizer BOS/EOS IDs do not match the exported Llama config.")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mapped_state, output_dir / "pytorch_model.bin")
    shutil.copyfile(tokenizer_path, output_dir / "tokenizer.json")

    tokenizer_config = {
        "tokenizer_class": "GPT2TokenizerFast",
        "model_max_length": hf_config["max_position_embeddings"],
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
        "unk_token": tokenizer.bos_token,
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "legacy": False,
    }
    special_tokens = {
        "bos_token": tokenizer.bos_token,
        "eos_token": tokenizer.eos_token,
    }
    generation_config = {
        "bos_token_id": hf_config["bos_token_id"],
        "eos_token_id": hf_config["eos_token_id"],
    }
    for filename, data in (
        ("config.json", hf_config),
        ("tokenizer_config.json", tokenizer_config),
        ("special_tokens_map.json", special_tokens),
        ("generation_config.json", generation_config),
    ):
        (output_dir / filename).write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )

    return output_dir
