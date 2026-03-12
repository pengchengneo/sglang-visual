/** TypeScript interfaces matching the backend Pydantic schema. */

export type PartitionStrategy =
  | "column_parallel"
  | "row_parallel"
  | "replicated"
  | "qkv_parallel"
  | "merged_column_parallel";

export type CommOp = "all_reduce" | "all_gather";

export type LinearType =
  | "ReplicatedLinear"
  | "ColumnParallelLinear"
  | "MergedColumnParallelLinear"
  | "QKVParallelLinear"
  | "RowParallelLinear";

export interface PartitionInfo {
  dim: number;
  strategy: PartitionStrategy;
  details: string;
}

export interface Operator {
  name: string;
  op_type: string;
  linear_type: LinearType | null;
  full_weight_shape: number[];
  tp_weight_shape: number[];
  partition: PartitionInfo | null;
  comm_after: CommOp | null;
  sub_components: string[] | null;
}

export interface Layer {
  layer_id: number;
  layer_type: string;
  operators: Operator[];
}

export interface EmbeddingInfo {
  full_shape: number[];
  tp_shape: number[];
  partition_strategy: PartitionStrategy;
  comm_after: CommOp | null;
}

export interface CommSummary {
  ops_per_layer: number;
  total_ops: number;
}

export interface ModelConfig {
  hidden_size: number;
  num_attention_heads: number;
  num_key_value_heads: number;
  head_dim: number;
  intermediate_size: number;
  num_hidden_layers: number;
  vocab_size: number;
  kv_lora_rank: number | null;
  q_lora_rank: number | null;
  qk_nope_head_dim: number | null;
  qk_rope_head_dim: number | null;
  v_head_dim: number | null;
  n_routed_experts: number | null;
  n_shared_experts: number | null;
  num_experts_per_tok: number | null;
  moe_intermediate_size: number | null;
  first_k_dense_replace: number | null;
}

export interface ModelArchitecture {
  model_id: string;
  model_family: string;
  tp_size: number;
  config: ModelConfig;
  embedding: EmbeddingInfo;
  lm_head: EmbeddingInfo;
  layers: Layer[];
  communication_summary: CommSummary;
}

export interface PresetManifestEntry {
  id: string;
  model_id: string;
  family: string;
  num_layers: number;
  hidden_size: number;
  file: string;
}
