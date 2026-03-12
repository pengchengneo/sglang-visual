"""Tests for TP shape calculations.

Verifies that our shape formulas match the actual SGLang behavior:
- Llama: QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
- DeepSeek V2: MLA (ReplicatedLinear, ColumnParallelLinear) + MoE
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sglang_tp_viz.schema import ModelConfig
from sglang_tp_viz.families.llama import LlamaTemplate
from sglang_tp_viz.families.deepseek_v2 import DeepSeekV2Template


# ── Llama 3.1 8B ──
LLAMA_8B_CONFIG = ModelConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=14336,
    num_hidden_layers=32,
    vocab_size=128256,
)


class TestLlamaShapes:
    def setup_method(self):
        self.template = LlamaTemplate(LLAMA_8B_CONFIG)

    def test_qkv_proj_full_shape(self):
        layer = self.template.get_layer(0)
        qkv = next(op for op in layer.operators if op.name == "qkv_proj")
        # Q: 32 heads * 128 = 4096, K: 8 * 128 = 1024, V: 8 * 128 = 1024
        # Total = 4096 + 1024 + 1024 = 6144
        assert qkv.full_weight_shape == [6144, 4096]

    def test_o_proj_full_shape(self):
        layer = self.template.get_layer(0)
        o = next(op for op in layer.operators if op.name == "o_proj")
        # [hidden_size, num_heads * head_dim] = [4096, 4096]
        assert o.full_weight_shape == [4096, 4096]

    def test_gate_up_proj_full_shape(self):
        layer = self.template.get_layer(0)
        gu = next(op for op in layer.operators if op.name == "gate_up_proj")
        # [2 * intermediate_size, hidden_size] = [28672, 4096]
        assert gu.full_weight_shape == [28672, 4096]

    def test_down_proj_full_shape(self):
        layer = self.template.get_layer(0)
        d = next(op for op in layer.operators if op.name == "down_proj")
        # [hidden_size, intermediate_size] = [4096, 14336]
        assert d.full_weight_shape == [4096, 14336]

    def test_comm_pattern(self):
        layer = self.template.get_layer(0)
        comms = [(op.name, op.comm_after) for op in layer.operators if op.comm_after]
        assert comms == [
            ("o_proj", "all_reduce"),
            ("down_proj", "all_reduce"),
        ]

    def test_layer_count(self):
        layers = self.template.get_all_layers()
        assert len(layers) == 32

    def test_all_layers_dense(self):
        layers = self.template.get_all_layers()
        assert all(l.layer_type == "dense" for l in layers)


# ── Llama 3.1 70B ──
LLAMA_70B_CONFIG = ModelConfig(
    hidden_size=8192,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=28672,
    num_hidden_layers=80,
    vocab_size=128256,
)


class TestLlama70BShapes:
    def setup_method(self):
        self.template = LlamaTemplate(LLAMA_70B_CONFIG)

    def test_qkv_proj_full_shape(self):
        layer = self.template.get_layer(0)
        qkv = next(op for op in layer.operators if op.name == "qkv_proj")
        # Q: 64*128=8192, K: 8*128=1024, V: 8*128=1024 -> total=10240
        assert qkv.full_weight_shape == [10240, 8192]

    def test_o_proj_full_shape(self):
        layer = self.template.get_layer(0)
        o = next(op for op in layer.operators if op.name == "o_proj")
        assert o.full_weight_shape == [8192, 8192]

    def test_gate_up_proj_full_shape(self):
        layer = self.template.get_layer(0)
        gu = next(op for op in layer.operators if op.name == "gate_up_proj")
        assert gu.full_weight_shape == [57344, 8192]

    def test_layer_count(self):
        layers = self.template.get_all_layers()
        assert len(layers) == 80


# ── DeepSeek V2 Lite ──
DEEPSEEK_V2_LITE_CONFIG = ModelConfig(
    hidden_size=2048,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=128,
    intermediate_size=10944,
    num_hidden_layers=27,
    vocab_size=102400,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    n_routed_experts=64,
    n_shared_experts=2,
    num_experts_per_tok=6,
    moe_intermediate_size=1408,
    first_k_dense_replace=1,
)


class TestDeepSeekV2LiteShapes:
    def setup_method(self):
        self.template = DeepSeekV2Template(DEEPSEEK_V2_LITE_CONFIG)

    def test_fused_qkv_a_proj_shape(self):
        layer = self.template.get_layer(1)  # MoE layer
        fused_a = next(op for op in layer.operators if op.name == "fused_qkv_a_proj_with_mqa")
        # q_lora_rank + kv_lora_rank + qk_rope_head_dim = 1536 + 512 + 64 = 2112
        assert fused_a.full_weight_shape == [2112, 2048]
        assert fused_a.linear_type == "ReplicatedLinear"

    def test_q_b_proj_shape(self):
        layer = self.template.get_layer(1)
        q_b = next(op for op in layer.operators if op.name == "q_b_proj")
        # num_heads * (qk_nope + qk_rope) = 16 * (128 + 64) = 3072
        assert q_b.full_weight_shape == [3072, 1536]
        assert q_b.linear_type == "ColumnParallelLinear"

    def test_kv_b_proj_shape(self):
        layer = self.template.get_layer(1)
        kv_b = next(op for op in layer.operators if op.name == "kv_b_proj")
        # num_heads * (qk_nope + v_head_dim) = 16 * (128 + 128) = 4096
        assert kv_b.full_weight_shape == [4096, 512]
        assert kv_b.linear_type == "ColumnParallelLinear"

    def test_o_proj_shape(self):
        layer = self.template.get_layer(1)
        o = next(op for op in layer.operators if op.name == "o_proj")
        # [hidden_size, num_heads * v_head_dim] = [2048, 16 * 128] = [2048, 2048]
        assert o.full_weight_shape == [2048, 2048]
        assert o.comm_after == "all_reduce"

    def test_first_layer_is_dense(self):
        layer = self.template.get_layer(0)
        assert layer.layer_type == "dense"

    def test_second_layer_is_moe(self):
        layer = self.template.get_layer(1)
        assert layer.layer_type == "moe"

    def test_moe_gate_shape(self):
        layer = self.template.get_layer(1)
        gate = next(op for op in layer.operators if op.name == "gate")
        assert gate.full_weight_shape == [64, 2048]
        assert gate.linear_type == "ReplicatedLinear"

    def test_moe_experts_shape(self):
        layer = self.template.get_layer(1)
        experts_gu = next(op for op in layer.operators if op.name == "experts_gate_up")
        # [n_experts, 2*moe_inter, hidden] = [64, 2816, 2048]
        assert experts_gu.full_weight_shape == [64, 2816, 2048]

        experts_d = next(op for op in layer.operators if op.name == "experts_down")
        # [n_experts, hidden, moe_inter] = [64, 2048, 1408]
        assert experts_d.full_weight_shape == [64, 2048, 1408]

    def test_shared_experts_shape(self):
        layer = self.template.get_layer(1)
        shared_gu = next(op for op in layer.operators if op.name == "shared_experts_gate_up")
        # n_shared=2, moe_inter=1408, shared_inter=2816
        # [2*2816, 2048] = [5632, 2048]
        assert shared_gu.full_weight_shape == [5632, 2048]

    def test_moe_comm_pattern(self):
        layer = self.template.get_layer(1)
        comms = [(op.name, op.comm_after) for op in layer.operators if op.comm_after]
        # o_proj all-reduce + moe_output_reduce all-reduce
        assert ("o_proj", "all_reduce") in comms
        assert ("moe_output_reduce", "all_reduce") in comms


# ── DeepSeek V3 ──
DEEPSEEK_V3_CONFIG = ModelConfig(
    hidden_size=7168,
    num_attention_heads=128,
    num_key_value_heads=128,
    head_dim=128,
    intermediate_size=18432,
    num_hidden_layers=61,
    vocab_size=129280,
    kv_lora_rank=512,
    q_lora_rank=1536,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    n_routed_experts=256,
    n_shared_experts=1,
    num_experts_per_tok=8,
    moe_intermediate_size=2048,
    first_k_dense_replace=3,
)


class TestDeepSeekV3Shapes:
    def setup_method(self):
        self.template = DeepSeekV2Template(DEEPSEEK_V3_CONFIG)

    def test_fused_qkv_a_proj_shape(self):
        layer = self.template.get_layer(3)  # First MoE layer
        fused_a = next(op for op in layer.operators if op.name == "fused_qkv_a_proj_with_mqa")
        # 1536 + 512 + 64 = 2112
        assert fused_a.full_weight_shape == [2112, 7168]

    def test_q_b_proj_shape(self):
        layer = self.template.get_layer(3)
        q_b = next(op for op in layer.operators if op.name == "q_b_proj")
        # 128 heads * (128+64) = 24576
        assert q_b.full_weight_shape == [24576, 1536]

    def test_kv_b_proj_shape(self):
        layer = self.template.get_layer(3)
        kv_b = next(op for op in layer.operators if op.name == "kv_b_proj")
        # 128 * (128 + 128) = 32768
        assert kv_b.full_weight_shape == [32768, 512]

    def test_first_3_layers_dense(self):
        for i in range(3):
            layer = self.template.get_layer(i)
            assert layer.layer_type == "dense", f"Layer {i} should be dense"

    def test_layer_3_is_moe(self):
        layer = self.template.get_layer(3)
        assert layer.layer_type == "moe"

    def test_moe_experts_shape(self):
        layer = self.template.get_layer(3)
        experts_gu = next(op for op in layer.operators if op.name == "experts_gate_up")
        # [256, 2*2048, 7168] = [256, 4096, 7168]
        assert experts_gu.full_weight_shape == [256, 4096, 7168]

    def test_shared_experts_shape(self):
        layer = self.template.get_layer(3)
        shared_gu = next(op for op in layer.operators if op.name == "shared_experts_gate_up")
        # n_shared=1, moe_inter=2048, shared_inter=2048
        # [2*2048, 7168] = [4096, 7168]
        assert shared_gu.full_weight_shape == [4096, 7168]

    def test_total_layer_count(self):
        layers = self.template.get_all_layers()
        assert len(layers) == 61


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
