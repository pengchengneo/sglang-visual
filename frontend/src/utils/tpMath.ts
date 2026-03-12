/**
 * Client-side TP shape recalculation.
 *
 * All formulas are simple integer division, so we can instantly
 * recalculate shapes when the user changes TP size — no backend needed.
 */

import type { Operator, ModelConfig, Layer, EmbeddingInfo, ModelArchitecture } from "../types/model";

/** Recalculate a single operator's TP weight shape. */
export function recomputeOperatorTpShape(
  op: Operator,
  config: ModelConfig,
  tpSize: number
): number[] {
  const full = op.full_weight_shape;

  if (!op.partition || op.partition.strategy === "replicated") {
    return [...full]; // replicated: no change
  }

  switch (op.partition.strategy) {
    case "qkv_parallel":
      return computeQkvTpShape(config, tpSize);

    case "column_parallel":
    case "merged_column_parallel":
      return computeColumnTpShape(op, config, tpSize);

    case "row_parallel":
      return computeRowTpShape(op, config, tpSize);

    default:
      return [...full];
  }
}

function computeQkvTpShape(config: ModelConfig, tp: number): number[] {
  const H = config.num_attention_heads;
  const K = config.num_key_value_heads;
  const d = config.head_dim;
  const hidden = config.hidden_size;

  const qPerRank = Math.floor(H / tp) * d;
  const kPerRank = Math.floor(K / tp) * d;
  const vPerRank = kPerRank;
  return [qPerRank + kPerRank + vPerRank, hidden];
}

function computeColumnTpShape(
  op: Operator,
  _config: ModelConfig,
  tp: number
): number[] {
  const full = op.full_weight_shape;
  if (full.length === 3) {
    // MoE expert tensor: [n_experts, out, in] — split out dim
    return [full[0], Math.floor(full[1] / tp), full[2]];
  }
  // [out, in] — split out dim
  return [Math.floor(full[0] / tp), full[1]];
}

function computeRowTpShape(
  op: Operator,
  _config: ModelConfig,
  tp: number
): number[] {
  const full = op.full_weight_shape;
  if (full.length === 3) {
    // MoE expert tensor: [n_experts, out, in] — split in dim
    return [full[0], full[1], Math.floor(full[2] / tp)];
  }
  // [out, in] — split in dim
  return [full[0], Math.floor(full[1] / tp)];
}

/** Recalculate embedding/lm_head TP shape. */
export function recomputeEmbeddingTpShape(
  emb: EmbeddingInfo,
  tp: number
): number[] {
  // VocabParallelEmbedding: split vocab dim
  return [Math.floor(emb.full_shape[0] / tp), emb.full_shape[1]];
}

/** Recalculate all operator shapes in a layer. */
export function recomputeLayerTpShapes(
  layer: Layer,
  config: ModelConfig,
  tp: number
): Operator[] {
  return layer.operators.map((op) => ({
    ...op,
    tp_weight_shape: recomputeOperatorTpShape(op, config, tp),
  }));
}

/** Count total parameters from a shape array. */
export function shapeToParams(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Format large numbers with K/M/B suffixes. */
export function formatParams(n: number): string {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return n.toString();
}

/** Format bytes with KB/MB/GB suffixes (assuming fp16 = 2 bytes per param). */
export function formatMemory(params: number, bytesPerParam = 2): string {
  const bytes = params * bytesPerParam;
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + " GB";
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + " MB";
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + " KB";
  return bytes + " B";
}

/** Check if a TP size is compatible (all head counts must be divisible). */
export function isTpCompatible(config: ModelConfig, tp: number): boolean {
  if (tp === 1) return true;
  return (
    config.num_attention_heads % tp === 0 &&
    config.num_key_value_heads % tp === 0
  );
}

/** Compute the MLA-specific QKV shapes for DeepSeek. */
export function computeMlaTpShapes(
  op: Operator,
  config: ModelConfig,
  tp: number
): number[] {
  if (op.name === "q_b_proj") {
    const H = config.num_attention_heads;
    const qkHead = (config.qk_nope_head_dim || 128) + (config.qk_rope_head_dim || 64);
    return [Math.floor(H / tp) * qkHead, config.q_lora_rank || 1536];
  }
  if (op.name === "kv_b_proj") {
    const H = config.num_attention_heads;
    const nope = config.qk_nope_head_dim || 128;
    const v = config.v_head_dim || 128;
    return [Math.floor(H / tp) * (nope + v), config.kv_lora_rank || 512];
  }
  return recomputeOperatorTpShape(op, config, tp);
}

/** Get rank colors for visualization. */
export const RANK_COLORS = [
  "#818cf8", // indigo-400
  "#f87171", // red-400
  "#fbbf24", // amber-400
  "#34d399", // emerald-400
  "#fb923c", // orange-400
  "#22d3ee", // cyan-400
  "#a5b4fc", // indigo-300
  "#fca5a5", // red-300
];

export function getRankColor(rank: number): string {
  return RANK_COLORS[rank % RANK_COLORS.length];
}

/** Compute total parameters per rank after TP partitioning. */
export function computePerRankParams(
  model: ModelArchitecture,
  tp: number,
): number {
  let total = 0;
  total += shapeToParams(recomputeEmbeddingTpShape(model.embedding, tp));
  total += shapeToParams(recomputeEmbeddingTpShape(model.lm_head, tp));
  for (const layer of model.layers) {
    for (const op of layer.operators) {
      if (op.op_type !== "comm") {
        const tpShape = recomputeOperatorTpShape(op, model.config, tp);
        total += shapeToParams(tpShape);
      }
    }
  }
  return total;
}
