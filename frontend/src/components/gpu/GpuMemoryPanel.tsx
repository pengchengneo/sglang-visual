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
  perRankParams: number;
  gpuMemoryBytes: number;
  memFractionStatic: number;
}

export function GpuMemoryPanel({
  config,
  tpSize,
  perRankParams,
  gpuMemoryBytes,
  memFractionStatic,
}: Props) {
  const weightBytes = useMemo(
    () => computeWeightMemoryBytes(perRankParams),
    [perRankParams],
  );

  const kvPerToken = useMemo(
    () => computeKvPerTokenBytes(config, tpSize),
    [config, tpSize],
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

  return (
    <div className="gpu-memory-panel">
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

      <div className="gpu-cards">
        {Array.from({ length: tpSize }, (_, rank) => (
          <GpuCard
            key={rank}
            rank={rank}
            color={getRankColor(rank)}
            breakdown={breakdown}
            kvSlots={kvSlots}
            kvPerToken={kvPerToken}
          />
        ))}
      </div>
    </div>
  );
}
