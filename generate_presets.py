"""Script to generate preset JSON files for the frontend.

Uses hardcoded HF configs to avoid requiring network access.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from sglang_tp_viz.tp_analyzer import analyze

# ── Llama 3.1 8B ──
LLAMA_3_8B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 128256,
}

# ── Llama 3.1 70B ──
LLAMA_3_70B_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 8192,
    "intermediate_size": 28672,
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 128256,
}

# ── DeepSeek V2 Lite ──
DEEPSEEK_V2_LITE_CONFIG = {
    "architectures": ["DeepseekV2ForCausalLM"],
    "hidden_size": 2048,
    "intermediate_size": 10944,
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "vocab_size": 102400,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 1408,
    "first_k_dense_replace": 1,
}

# ── DeepSeek V3 ──
DEEPSEEK_V3_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "hidden_size": 7168,
    "intermediate_size": 18432,
    "num_attention_heads": 128,
    "num_hidden_layers": 61,
    "num_key_value_heads": 128,
    "head_dim": 128,
    "vocab_size": 129280,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "first_k_dense_replace": 3,
}

PRESETS = {
    "llama3_8b": ("meta-llama/Llama-3.1-8B", LLAMA_3_8B_CONFIG),
    "llama3_70b": ("meta-llama/Llama-3.1-70B", LLAMA_3_70B_CONFIG),
    "deepseek_v2_lite": ("deepseek-ai/DeepSeek-V2-Lite", DEEPSEEK_V2_LITE_CONFIG),
    "deepseek_v3": ("deepseek-ai/DeepSeek-V3", DEEPSEEK_V3_CONFIG),
}


def main():
    out_dir = Path(__file__).parent / "frontend" / "public" / "presets"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    for preset_name, (model_id, hf_config) in PRESETS.items():
        print(f"Generating {preset_name}...")
        arch = analyze(model_id=model_id, hf_config=hf_config)
        out_file = out_dir / f"{preset_name}.json"
        out_file.write_text(arch.model_dump_json(indent=2))
        manifest_entries.append({
            "id": preset_name,
            "model_id": model_id,
            "family": arch.model_family,
            "num_layers": arch.config.num_hidden_layers,
            "hidden_size": arch.config.hidden_size,
            "file": f"{preset_name}.json",
        })
        print(f"  -> {out_file}")

    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest_entries, indent=2))
    print(f"\nManifest -> {manifest_file}")


if __name__ == "__main__":
    main()
