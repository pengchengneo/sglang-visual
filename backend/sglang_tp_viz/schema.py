"""Pydantic schema for the model architecture JSON output."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class PartitionStrategy(str, Enum):
    COLUMN = "column_parallel"
    ROW = "row_parallel"
    REPLICATED = "replicated"
    QKV_PARALLEL = "qkv_parallel"
    MERGED_COLUMN = "merged_column_parallel"


class CommOp(str, Enum):
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"


class LinearType(str, Enum):
    REPLICATED = "ReplicatedLinear"
    COLUMN = "ColumnParallelLinear"
    MERGED_COLUMN = "MergedColumnParallelLinear"
    QKV = "QKVParallelLinear"
    ROW = "RowParallelLinear"


class PartitionInfo(BaseModel):
    dim: int  # 0 = row (input), 1 = column (output)
    strategy: PartitionStrategy
    details: str  # human-readable explanation


class Operator(BaseModel):
    name: str
    op_type: str  # "linear", "norm", "activation", "moe_gate", etc.
    linear_type: Optional[LinearType] = None
    full_weight_shape: list[int]  # [out_features, in_features]
    tp_weight_shape: list[int]  # shape on each rank at tp_size=1 (same as full)
    partition: Optional[PartitionInfo] = None
    comm_after: Optional[CommOp] = None
    sub_components: Optional[list[str]] = None  # e.g. ["q_proj", "k_proj", "v_proj"]


class Layer(BaseModel):
    layer_id: int
    layer_type: str  # "dense", "moe", "attention", "mlp"
    operators: list[Operator]


class EmbeddingInfo(BaseModel):
    full_shape: list[int]  # [vocab_size, hidden_size]
    tp_shape: list[int]
    partition_strategy: PartitionStrategy
    comm_after: Optional[CommOp] = None


class CommSummary(BaseModel):
    ops_per_layer: int
    total_ops: int


class ModelConfig(BaseModel):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    # MLA-specific (DeepSeek)
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    # MoE-specific
    n_routed_experts: Optional[int] = None
    n_shared_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    # Layer type mapping
    first_k_dense_replace: Optional[int] = None  # first N layers are dense


class ModelArchitecture(BaseModel):
    model_id: str
    model_family: str
    tp_size: int  # always 1 in presets; frontend recalculates
    config: ModelConfig
    embedding: EmbeddingInfo
    lm_head: EmbeddingInfo
    layers: list[Layer]
    communication_summary: CommSummary
