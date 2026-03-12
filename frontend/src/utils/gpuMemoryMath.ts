/**
 * GPU memory calculation utilities.
 *
 * Computes memory breakdown (weights, KV cache, reserved) and KV cache slot
 * capacity for a given GPU configuration.
 */

import type { ModelConfig } from "../types/model";

export interface GpuMemoryOption {
  label: string;
  bytes: number;
}

/** Common GPU memory sizes. */
export const GPU_MEMORY_OPTIONS: GpuMemoryOption[] = [
  { label: "24 GB", bytes: 24 * 1e9 },
  { label: "40 GB", bytes: 40 * 1e9 },
  { label: "48 GB", bytes: 48 * 1e9 },
  { label: "80 GB", bytes: 80 * 1e9 },
  { label: "141 GB", bytes: 141 * 1e9 },
];

export interface GpuBreakdown {
  totalBytes: number;
  totalUsable: number;
  weights: number;
  kvCache: number;
  reserved: number;
  oom: boolean;
}

/** Weight memory in bytes (FP16 default). */
export function computeWeightMemoryBytes(
  perRankParams: number,
  bytesPerParam = 2,
): number {
  return perRankParams * bytesPerParam;
}

/**
 * Bytes per token for KV cache on one rank.
 *
 * Standard (Llama/Mistral):
 *   numLayers * 2 * floor(numKvHeads / tp) * headDim * 2
 *
 * DeepSeek MLA (detected via kv_lora_rank !== null):
 *   numLayers * (kvLoraRank + qkRopeHeadDim) * 2   (replicated, no TP split)
 */
export function computeKvPerTokenBytes(
  config: ModelConfig,
  tpSize: number,
  kvBytesPerElement = 2,
): number {
  if (config.kv_lora_rank != null) {
    // MLA: compressed KV is replicated across ranks
    const kvLoraRank = config.kv_lora_rank;
    const qkRopeHeadDim = config.qk_rope_head_dim ?? 64;
    return config.num_hidden_layers * (kvLoraRank + qkRopeHeadDim) * kvBytesPerElement;
  }
  // Standard multi-head / grouped-query attention
  const kvHeadsPerRank = Math.floor(config.num_key_value_heads / tpSize);
  return config.num_hidden_layers * 2 * kvHeadsPerRank * config.head_dim * kvBytesPerElement;
}

/** Compute memory breakdown for a single GPU. */
export function computeGpuBreakdown(
  gpuBytes: number,
  memFraction: number,
  weightBytes: number,
): GpuBreakdown {
  const totalUsable = gpuBytes * memFraction;
  const reserved = gpuBytes - totalUsable;
  const oom = weightBytes > totalUsable;
  const kvCache = Math.max(0, totalUsable - weightBytes);

  return {
    totalBytes: gpuBytes,
    totalUsable,
    weights: weightBytes,
    kvCache,
    reserved,
    oom,
  };
}

/** Number of tokens that fit in the KV cache. */
export function computeKvSlots(
  kvCacheBytes: number,
  kvPerTokenBytes: number,
): number {
  if (kvPerTokenBytes <= 0) return 0;
  return Math.floor(kvCacheBytes / kvPerTokenBytes);
}

/** Format bytes as human-readable string. */
export function formatBytes(bytes: number): string {
  if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + " GB";
  if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + " MB";
  if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + " KB";
  return bytes + " B";
}
