import { useMemo } from "react";
import type { ModelConfig } from "../../types/model";
import { getRankColor } from "../../utils/tpMath";
import {
  computeWeightMemoryBytes,
  computeKvPerTokenBytes,
  computeGpuBreakdown,
  computeKvSlots,
  formatBytes,
} from "../../utils/gpuMemoryMath";
import { GpuCard } from "./GpuCard";

interface Props {
  config: ModelConfig;
  tpSize: number;
  dpSize: number;
  ppSize: number;
  enableDpAttention: boolean;
  perRankParams: number;
  gpuMemoryBytes: number;
  memFractionStatic: number;
  bytesPerParam: number;
  kvBytesPerElement: number;
  contextLength: number;
}

export function GpuMemoryPanel({
  config,
  tpSize,
  dpSize,
  ppSize,
  enableDpAttention,
  perRankParams,
  gpuMemoryBytes,
  memFractionStatic,
  bytesPerParam,
  kvBytesPerElement,
  contextLength,
}: Props) {
  const weightBytes = useMemo(
    () => computeWeightMemoryBytes(perRankParams, bytesPerParam),
    [perRankParams, bytesPerParam],
  );

  const kvPerToken = useMemo(
    () => computeKvPerTokenBytes(config, tpSize, kvBytesPerElement),
    [config, tpSize, kvBytesPerElement],
  );

  const breakdown = useMemo(
    () => computeGpuBreakdown(gpuMemoryBytes, memFractionStatic, weightBytes),
    [gpuMemoryBytes, memFractionStatic, weightBytes],
  );

  const kvSlots = useMemo(
    () => computeKvSlots(breakdown.kvCache, kvPerToken),
    [breakdown.kvCache, kvPerToken],
  );

  const isMla = config.kv_lora_rank != null;
  const useDpAttn = enableDpAttention && dpSize > 1;
  const attnTpSize = useDpAttn ? tpSize / dpSize : tpSize;
  const totalGpus = tpSize * ppSize;

  return (
    <div className="gpu-memory-panel">
      {/* Panel header */}
      <div className="gpu-panel-header">
        <h3 className="gpu-panel-title">GPU Memory</h3>
        <div className="gpu-summary-stats">
          <div className="gpu-summary-item">
            <span className="gpu-summary-val">
              {kvSlots.toLocaleString()}
            </span>
            <span className="gpu-summary-lbl">KV slots / GPU</span>
          </div>
          <div className="gpu-summary-item">
            <span className="gpu-summary-val">{formatBytes(kvPerToken)}</span>
            <span className="gpu-summary-lbl">
              / token{isMla ? " (MLA)" : ""}
            </span>
          </div>
        </div>
      </div>

      {/* Context Length Comparison */}
      <div className="context-comparison">
        <span className="context-comparison-label">Context vs KV capacity:</span>
        {kvSlots >= contextLength ? (
          <span className="context-ok">
            {kvSlots.toLocaleString()} slots ≥ {contextLength.toLocaleString()} ctx
          </span>
        ) : (
          <span className="context-warning">
            {kvSlots.toLocaleString()} slots &lt; {contextLength.toLocaleString()} ctx — KV cache insufficient
          </span>
        )}
      </div>

      {/* GPU Topology Grid — DP Attention mode */}
      {useDpAttn && (
        <div className="gpu-topology-grid">
          <div className="gpu-topology-title">
            GPU Topology (DP Attention: {dpSize} groups × {attnTpSize} TP)
          </div>
          <table className="gpu-topology-table">
            <thead>
              <tr>
                <th />
                {Array.from({ length: attnTpSize }, (_, t) => (
                  <th key={t}>attn_tp{t}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: dpSize }, (_, d) => (
                <tr key={d}>
                  <td className="gpu-topology-row-label">dp_group{d}</td>
                  {Array.from({ length: attnTpSize }, (_, t) => {
                    const gpuId = d * attnTpSize + t;
                    return (
                      <td key={t}>
                        <span
                          className="gpu-topology-cell"
                          style={{ background: getRankColor(gpuId) }}
                        >
                          {gpuId}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* GPU Topology Grid — PP mode (no DP attention) */}
      {ppSize > 1 && !useDpAttn && (
        <div className="gpu-topology-grid">
          <div className="gpu-topology-title">
            GPU Topology (PP={ppSize}, TP={tpSize}) — {totalGpus} GPUs
          </div>
          <table className="gpu-topology-table">
            <thead>
              <tr>
                <th />
                {Array.from({ length: tpSize }, (_, t) => (
                  <th key={t}>TP{t}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: ppSize }, (_, p) => (
                <tr key={p}>
                  <td className="gpu-topology-row-label">PP{p}</td>
                  {Array.from({ length: tpSize }, (_, t) => {
                    const gpuId = p * tpSize + t;
                    return (
                      <td key={t}>
                        <span
                          className="gpu-topology-cell"
                          style={{ background: getRankColor(t) }}
                        >
                          {gpuId}
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Shared Per-GPU Breakdown (rendered once) */}
      <GpuCard
        breakdown={breakdown}
        kvSlots={kvSlots}
        kvPerToken={kvPerToken}
      />

      {/* GPU Chip Grid */}
      {useDpAttn ? (
        <div className="gpu-chip-groups">
          {Array.from({ length: dpSize }, (_, d) => (
            <div key={d} className="gpu-chip-group">
              <div className="gpu-chip-group-title">DP Group {d}</div>
              <div className="gpu-chip-grid">
                {Array.from({ length: attnTpSize }, (_, t) => {
                  const gpuId = d * attnTpSize + t;
                  return (
                    <div
                      key={gpuId}
                      className="gpu-chip"
                      style={{ background: getRankColor(gpuId) }}
                    >
                      GPU {gpuId} · attn_dp{d} attn_tp{t}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      ) : ppSize > 1 ? (
        <div className="gpu-chip-groups">
          {Array.from({ length: ppSize }, (_, p) => (
            <div key={p} className="gpu-chip-group">
              <div className="gpu-chip-group-title">PP Stage {p}</div>
              <div className="gpu-chip-grid">
                {Array.from({ length: tpSize }, (_, t) => {
                  const gpuId = p * tpSize + t;
                  return (
                    <div
                      key={gpuId}
                      className="gpu-chip"
                      style={{ background: getRankColor(t) }}
                    >
                      GPU {gpuId}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="gpu-chip-grid">
          {Array.from({ length: tpSize }, (_, rank) => (
            <div
              key={rank}
              className="gpu-chip"
              style={{ background: getRankColor(rank) }}
            >
              GPU {rank}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
