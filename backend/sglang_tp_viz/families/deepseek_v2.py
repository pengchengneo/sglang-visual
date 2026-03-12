"""DeepSeek V2/V3 MLA + MoE TP template.

MLA attention:
  fused_qkv_a_proj: ReplicatedLinear (small latent dim, not worth sharding)
  q_b_proj:         ColumnParallelLinear (expand to per-head Q)
  kv_b_proj:        ColumnParallelLinear (expand to per-head KV)
  o_proj:           RowParallelLinear -> all-reduce

MoE layer:
  gate (router):    ReplicatedLinear (not sharded)
  experts w13/w2:   TP-split moe_intermediate_size
  shared_experts:   MergedColumnParallel + RowParallel (same as Llama MLP)
  output:           all-reduce

Dense layers (first_k_dense_replace) use standard MLP instead of MoE.
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


class DeepSeekV2Template(ModelFamilyTemplate):
    family_name = "deepseek_v2"

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _is_moe_layer(self, layer_id: int) -> bool:
        first_k = self.config.first_k_dense_replace or 0
        return layer_id >= first_k and self.config.n_routed_experts is not None

    def _attention_operators(self) -> list[Operator]:
        c = self.config
        hidden = c.hidden_size
        H = c.num_attention_heads
        kv_lora_rank = c.kv_lora_rank or 512
        q_lora_rank = c.q_lora_rank
        qk_nope_head_dim = c.qk_nope_head_dim or 128
        qk_rope_head_dim = c.qk_rope_head_dim or 64
        v_head_dim = c.v_head_dim or 128
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        ops: list[Operator] = []

        if q_lora_rank is not None and q_lora_rank > 0:
            # Fused qkv_a projection: hidden -> q_lora_rank + kv_lora_rank + qk_rope_head_dim
            fused_a_out = q_lora_rank + kv_lora_rank + qk_rope_head_dim
            ops.append(Operator(
                name="fused_qkv_a_proj_with_mqa",
                op_type="linear",
                linear_type=LinearType.REPLICATED,
                full_weight_shape=[fused_a_out, hidden],
                tp_weight_shape=[fused_a_out, hidden],
                partition=PartitionInfo(
                    dim=-1,
                    strategy=PartitionStrategy.REPLICATED,
                    details=f"Replicated: projects to small latent space "
                    f"(q_lora={q_lora_rank} + kv_lora={kv_lora_rank} + rope={qk_rope_head_dim}). "
                    f"Not worth sharding.",
                ),
                comm_after=None,
                sub_components=["q_a_proj", "kv_a_proj_with_mqa"],
            ))

            # q_b_proj: q_lora_rank -> H * qk_head_dim
            q_b_out = H * qk_head_dim
            ops.append(Operator(
                name="q_b_proj",
                op_type="linear",
                linear_type=LinearType.COLUMN,
                full_weight_shape=[q_b_out, q_lora_rank],
                tp_weight_shape=[q_b_out, q_lora_rank],
                partition=PartitionInfo(
                    dim=1,
                    strategy=PartitionStrategy.COLUMN,
                    details=f"Column-parallel: expands latent q ({q_lora_rank}) "
                    f"to {H} heads × {qk_head_dim} dim. Split by heads.",
                ),
                comm_after=None,
            ))
        else:
            # No q_lora: direct q_proj
            qk_head_dim_val = qk_nope_head_dim + qk_rope_head_dim
            q_out = H * qk_head_dim_val
            ops.append(Operator(
                name="q_proj",
                op_type="linear",
                linear_type=LinearType.COLUMN,
                full_weight_shape=[q_out, hidden],
                tp_weight_shape=[q_out, hidden],
                partition=PartitionInfo(
                    dim=1,
                    strategy=PartitionStrategy.COLUMN,
                    details=f"Column-parallel: {H} heads × {qk_head_dim_val} dim.",
                ),
                comm_after=None,
            ))

            # kv_a_proj_with_mqa: hidden -> kv_lora_rank + qk_rope_head_dim
            kv_a_out = kv_lora_rank + qk_rope_head_dim
            ops.append(Operator(
                name="kv_a_proj_with_mqa",
                op_type="linear",
                linear_type=LinearType.REPLICATED,
                full_weight_shape=[kv_a_out, hidden],
                tp_weight_shape=[kv_a_out, hidden],
                partition=PartitionInfo(
                    dim=-1,
                    strategy=PartitionStrategy.REPLICATED,
                    details=f"Replicated: small KV latent projection "
                    f"(kv_lora={kv_lora_rank} + rope={qk_rope_head_dim}).",
                ),
                comm_after=None,
            ))

        # kv_b_proj: kv_lora_rank -> H * (qk_nope_head_dim + v_head_dim)
        kv_b_out = H * (qk_nope_head_dim + v_head_dim)
        ops.append(Operator(
            name="kv_b_proj",
            op_type="linear",
            linear_type=LinearType.COLUMN,
            full_weight_shape=[kv_b_out, kv_lora_rank],
            tp_weight_shape=[kv_b_out, kv_lora_rank],
            partition=PartitionInfo(
                dim=1,
                strategy=PartitionStrategy.COLUMN,
                details=f"Column-parallel: expands KV latent ({kv_lora_rank}) "
                f"to {H} heads × ({qk_nope_head_dim} + {v_head_dim}) dim.",
            ),
            comm_after=None,
        ))

        # o_proj: H * v_head_dim -> hidden
        o_in = H * v_head_dim
        ops.append(Operator(
            name="o_proj",
            op_type="linear",
            linear_type=LinearType.ROW,
            full_weight_shape=[hidden, o_in],
            tp_weight_shape=[hidden, o_in],
            partition=PartitionInfo(
                dim=0,
                strategy=PartitionStrategy.ROW,
                details=f"Row-parallel: input {o_in} split by heads across ranks.",
            ),
            comm_after=CommOp.ALL_REDUCE,
        ))

        return ops

    def _dense_mlp_operators(self) -> list[Operator]:
        """Standard dense MLP (used for first_k_dense_replace layers)."""
        c = self.config
        hidden = c.hidden_size
        inter = c.intermediate_size

        gate_up = Operator(
            name="gate_up_proj",
            op_type="linear",
            linear_type=LinearType.MERGED_COLUMN,
            full_weight_shape=[2 * inter, hidden],
            tp_weight_shape=[2 * inter, hidden],
            partition=PartitionInfo(
                dim=1,
                strategy=PartitionStrategy.MERGED_COLUMN,
                details=f"gate [{inter}, {hidden}] + up [{inter}, {hidden}] fused, column-split.",
            ),
            comm_after=None,
            sub_components=["gate_proj", "up_proj"],
        )

        down = Operator(
            name="down_proj",
            op_type="linear",
            linear_type=LinearType.ROW,
            full_weight_shape=[hidden, inter],
            tp_weight_shape=[hidden, inter],
            partition=PartitionInfo(
                dim=0,
                strategy=PartitionStrategy.ROW,
                details=f"Row-parallel: input {inter} split across ranks.",
            ),
            comm_after=CommOp.ALL_REDUCE,
        )

        return [gate_up, down]

    def _moe_operators(self) -> list[Operator]:
        """MoE layer: gate + routed experts + shared experts."""
        c = self.config
        hidden = c.hidden_size
        n_experts = c.n_routed_experts or 256
        n_shared = c.n_shared_experts or 2
        moe_inter = c.moe_intermediate_size or 1408
        shared_inter = moe_inter * n_shared

        ops: list[Operator] = []

        # Gate (router): replicated
        ops.append(Operator(
            name="gate",
            op_type="moe_gate",
            linear_type=LinearType.REPLICATED,
            full_weight_shape=[n_experts, hidden],
            tp_weight_shape=[n_experts, hidden],
            partition=PartitionInfo(
                dim=-1,
                strategy=PartitionStrategy.REPLICATED,
                details=f"Router: {n_experts} experts. Replicated across all ranks.",
            ),
            comm_after=None,
        ))

        # Routed experts: each expert has gate+up [2*moe_inter, hidden] and down [hidden, moe_inter]
        # TP splits moe_inter dimension
        ops.append(Operator(
            name="experts_gate_up",
            op_type="linear",
            linear_type=LinearType.MERGED_COLUMN,
            full_weight_shape=[n_experts, 2 * moe_inter, hidden],
            tp_weight_shape=[n_experts, 2 * moe_inter, hidden],
            partition=PartitionInfo(
                dim=1,
                strategy=PartitionStrategy.COLUMN,
                details=f"{n_experts} experts, each gate+up [{2 * moe_inter}, {hidden}]. "
                f"moe_inter dimension split by tp.",
            ),
            comm_after=None,
            sub_components=["w1 (gate)", "w3 (up)"],
        ))

        ops.append(Operator(
            name="experts_down",
            op_type="linear",
            linear_type=LinearType.ROW,
            full_weight_shape=[n_experts, hidden, moe_inter],
            tp_weight_shape=[n_experts, hidden, moe_inter],
            partition=PartitionInfo(
                dim=0,
                strategy=PartitionStrategy.ROW,
                details=f"{n_experts} experts, each down [{hidden}, {moe_inter}]. "
                f"Input dim (moe_inter) split by tp.",
            ),
            comm_after=None,  # all-reduce is on combined output
            sub_components=["w2 (down)"],
        ))

        # Shared experts: standard MLP pattern
        ops.append(Operator(
            name="shared_experts_gate_up",
            op_type="linear",
            linear_type=LinearType.MERGED_COLUMN,
            full_weight_shape=[2 * shared_inter, hidden],
            tp_weight_shape=[2 * shared_inter, hidden],
            partition=PartitionInfo(
                dim=1,
                strategy=PartitionStrategy.MERGED_COLUMN,
                details=f"Shared experts ({n_shared}): gate+up fused, column-split.",
            ),
            comm_after=None,
            sub_components=["shared_gate_proj", "shared_up_proj"],
        ))

        ops.append(Operator(
            name="shared_experts_down",
            op_type="linear",
            linear_type=LinearType.ROW,
            full_weight_shape=[hidden, shared_inter],
            tp_weight_shape=[hidden, shared_inter],
            partition=PartitionInfo(
                dim=0,
                strategy=PartitionStrategy.ROW,
                details=f"Shared experts down: row-parallel.",
            ),
            comm_after=None,  # combined all-reduce below
        ))

        # MoE output: combined all-reduce for routed + shared
        ops.append(Operator(
            name="moe_output_reduce",
            op_type="comm",
            linear_type=None,
            full_weight_shape=[0, 0],
            tp_weight_shape=[0, 0],
            partition=None,
            comm_after=CommOp.ALL_REDUCE,
        ))

        return ops

    def get_layer(self, layer_id: int) -> Layer:
        attn_ops = self._attention_operators()
        is_moe = self._is_moe_layer(layer_id)

        if is_moe:
            mlp_ops = self._moe_operators()
            layer_type = "moe"
        else:
            mlp_ops = self._dense_mlp_operators()
            layer_type = "dense"

        return Layer(
            layer_id=layer_id,
            layer_type=layer_type,
            operators=attn_ops + mlp_ops,
        )
