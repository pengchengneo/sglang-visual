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
  perRankParams: number;
  gpuMemoryBytes: number;
  memFractionStatic: number;
}

export function GpuMemoryPanel({
  config,
  tpSize,
  dpSize,
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

      {/* GPU Topology Grid — shown when DP > 1 */}
      {dpSize > 1 && (
        <div className="gpu-topology-grid">
          <div className="gpu-topology-title">
            GPU Topology (DP={dpSize}, TP={tpSize})
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
              {Array.from({ length: dpSize }, (_, d) => (
                <tr key={d}>
                  <td className="gpu-topology-row-label">DP{d}</td>
                  {Array.from({ length: tpSize }, (_, t) => {
                    const gpuId = d * tpSize + t;
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

      {/* GPU Cards — grouped by DP when dpSize > 1 */}
      {dpSize > 1 ? (
        <div className="dp-groups-container">
          {Array.from({ length: dpSize }, (_, dpRank) => (
            <div key={dpRank} className="dp-group">
              <div className="dp-group-header">DP Group {dpRank}</div>
              <div className="gpu-cards">
                {Array.from({ length: tpSize }, (_, tpRank) => {
                  const globalGpuId = dpRank * tpSize + tpRank;
                  return (
                    <GpuCard
                      key={globalGpuId}
                      rank={tpRank}
                      color={getRankColor(tpRank)}
                      breakdown={breakdown}
                      kvSlots={kvSlots}
                      kvPerToken={kvPerToken}
                      dpRank={dpRank}
                      tpRank={tpRank}
                      globalGpuId={globalGpuId}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      ) : (
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
      )}
    </div>
  );
}
