"""Llama / Qwen2 / Mistral dense model TP template.

TP pattern per layer (Megatron-style):
  Attention: QKVParallelLinear -> local attn -> RowParallel(o_proj) -> all-reduce
  MLP:       MergedColumnParallel(gate_up) -> SiLU -> RowParallel(down) -> all-reduce
"""

from __future__ import annotations

from ..schema import (
    CommOp,
    Layer,
    LinearType,
    ModelConfig,
    Operator,
    PartitionInfo,
    PartitionStrategy,
)
from .base import ModelFamilyTemplate


class LlamaTemplate(ModelFamilyTemplate):
    family_name = "llama"

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def get_layer(self, layer_id: int) -> Layer:
        c = self.config
        H = c.num_attention_heads
        K = c.num_key_value_heads
        d = c.head_dim
        hidden = c.hidden_size
        inter = c.intermediate_size

        # QKV fused: output = (H*d + 2*K*d)
        qkv_out = H * d + 2 * K * d
        qkv_proj = Operator(
            name="qkv_proj",
            op_type="linear",
            linear_type=LinearType.QKV,
            full_weight_shape=[qkv_out, hidden],
            tp_weight_shape=[qkv_out, hidden],
            partition=PartitionInfo(
                dim=1,
                strategy=PartitionStrategy.QKV_PARALLEL,
                details=f"Q: {H} heads, KV: {K} heads, each head_dim={d}. "
                f"Per rank: (H/tp + 2*K/tp) * d output features.",
            ),
            comm_after=None,
            sub_components=["q_proj", "k_proj", "v_proj"],
        )

        o_proj = Operator(
            name="o_proj",
            op_type="linear",
            linear_type=LinearType.ROW,
            full_weight_shape=[hidden, H * d],
            tp_weight_shape=[hidden, H * d],
            partition=PartitionInfo(
                dim=0,
                strategy=PartitionStrategy.ROW,
                details="Input dim split across ranks. "
                "Each rank holds hidden × (H*d / tp).",
            ),
            comm_after=CommOp.ALL_REDUCE,
        )

        gate_up_proj = Operator(
            name="gate_up_proj",
            op_type="linear",
            linear_type=LinearType.MERGED_COLUMN,
            full_weight_shape=[2 * inter, hidden],
            tp_weight_shape=[2 * inter, hidden],
            partition=PartitionInfo(
                dim=1,
                strategy=PartitionStrategy.MERGED_COLUMN,
                details=f"gate_proj [{inter}, {hidden}] + up_proj [{inter}, {hidden}] "
                f"fused. Each sub-matrix column-split by tp.",
            ),
            comm_after=None,
            sub_components=["gate_proj", "up_proj"],
        )

        down_proj = Operator(
            name="down_proj",
            op_type="linear",
            linear_type=LinearType.ROW,
            full_weight_shape=[hidden, inter],
            tp_weight_shape=[hidden, inter],
            partition=PartitionInfo(
                dim=0,
                strategy=PartitionStrategy.ROW,
                details="Input dim split across ranks. "
                f"Each rank holds {hidden} × (inter / tp).",
            ),
            comm_after=CommOp.ALL_REDUCE,
        )

        return Layer(
            layer_id=layer_id,
            layer_type="dense",
            operators=[qkv_proj, o_proj, gate_up_proj, down_proj],
        )
