"""Load and parse HuggingFace model config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .schema import ModelConfig


# Map of architecture type keywords to model family
_FAMILY_MAP = {
    "LlamaForCausalLM": "llama",
    "Qwen2ForCausalLM": "llama",  # same TP pattern
    "MistralForCausalLM": "llama",
    "DeepseekV2ForCausalLM": "deepseek_v2",
    "DeepseekV3ForCausalLM": "deepseek_v2",
}


def detect_family(hf_config: dict[str, Any]) -> str:
    """Detect model family from HF config architectures field."""
    archs = hf_config.get("architectures", [])
    for arch in archs:
        for key, family in _FAMILY_MAP.items():
            if key in arch:
                return family
    raise ValueError(f"Unsupported architecture: {archs}")


def load_hf_config(model_id: str, cache_dir: str | None = None) -> dict[str, Any]:
    """Download and parse config.json from HuggingFace Hub."""
    config_path = hf_hub_download(
        repo_id=model_id, filename="config.json", cache_dir=cache_dir
    )
    with open(config_path) as f:
        return json.load(f)


def load_local_config(path: str | Path) -> dict[str, Any]:
    """Load config.json from a local file."""
    with open(path) as f:
        return json.load(f)


def parse_model_config(hf_config: dict[str, Any], family: str) -> ModelConfig:
    """Parse HF config dict into our ModelConfig schema."""
    hidden_size = hf_config["hidden_size"]
    num_attention_heads = hf_config["num_attention_heads"]
    num_kv_heads = hf_config.get(
        "num_key_value_heads", num_attention_heads
    )
    head_dim = hf_config.get("head_dim", hidden_size // num_attention_heads)
    intermediate_size = hf_config.get("intermediate_size", 0)
    num_hidden_layers = hf_config["num_hidden_layers"]
    vocab_size = hf_config["vocab_size"]

    kwargs: dict[str, Any] = {}

    if family == "deepseek_v2":
        kwargs["kv_lora_rank"] = hf_config.get("kv_lora_rank")
        kwargs["q_lora_rank"] = hf_config.get("q_lora_rank")
        kwargs["qk_nope_head_dim"] = hf_config.get("qk_nope_head_dim")
        kwargs["qk_rope_head_dim"] = hf_config.get("qk_rope_head_dim")
        kwargs["v_head_dim"] = hf_config.get("v_head_dim")
        kwargs["n_routed_experts"] = hf_config.get("n_routed_experts")
        kwargs["n_shared_experts"] = hf_config.get("n_shared_experts")
        kwargs["num_experts_per_tok"] = hf_config.get("num_experts_per_tok")
        kwargs["moe_intermediate_size"] = hf_config.get("moe_intermediate_size")
        kwargs["first_k_dense_replace"] = hf_config.get("first_k_dense_replace")
        # DeepSeek uses different intermediate size field
        if intermediate_size == 0:
            intermediate_size = hf_config.get("intermediate_size", 0)

    return ModelConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        **kwargs,
    )
