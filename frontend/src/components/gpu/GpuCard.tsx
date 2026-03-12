import type { GpuBreakdown } from "../../utils/gpuMemoryMath";
import { formatBytes } from "../../utils/gpuMemoryMath";

interface Props {
  rank: number;
  color: string;
  breakdown: GpuBreakdown;
  kvSlots: number;
  kvPerToken: number;
  dpRank?: number;
  tpRank?: number;
  globalGpuId?: number;
}

export function GpuCard({
  rank,
  color,
  breakdown,
  kvSlots,
  kvPerToken,
  dpRank,
  tpRank,
  globalGpuId,
}: Props) {
  const { totalBytes, weights, kvCache, reserved, oom } = breakdown;

  const wPct = (weights / totalBytes) * 100;
  const kvPct = (kvCache / totalBytes) * 100;
  const rPct = (reserved / totalBytes) * 100;

  const hasDP = dpRank !== undefined;
  const headerLabel = hasDP
    ? `GPU ${globalGpuId} · DP${dpRank} TP${tpRank}`
    : `GPU ${rank} (Rank ${rank})`;

  return (
    <div className={`gpu-card${oom ? " gpu-card-oom" : ""}`}>
      <div className="gpu-card-header" style={{ background: color }}>
        <span className="gpu-card-rank">{headerLabel}</span>
        <span className="gpu-card-total">{formatBytes(totalBytes)}</span>
      </div>

      {/* Stacked memory bar */}
      <div className="memory-bar">
        <div
          className="memory-bar-seg memory-bar-weights"
          style={{ width: `${wPct}%` }}
          title={`Weights: ${formatBytes(weights)}`}
        />
        <div
          className="memory-bar-seg memory-bar-kv"
          style={{ width: `${kvPct}%` }}
          title={`KV Cache: ${formatBytes(kvCache)}`}
        />
        <div
          className="memory-bar-seg memory-bar-reserved"
          style={{ width: `${rPct}%` }}
          title={`Reserved: ${formatBytes(reserved)}`}
        />
      </div>

      {/* Stat rows */}
      <div className="gpu-stat-rows">
        <div className="gpu-stat-row">
          <span className="gpu-stat-dot" style={{ background: "#818cf8" }} />
          <span className="gpu-stat-label">Model Weights</span>
          <span className="gpu-stat-value">{formatBytes(weights)}</span>
          <span className="gpu-stat-pct">{wPct.toFixed(0)}%</span>
        </div>
        <div className="gpu-stat-row">
          <span className="gpu-stat-dot" style={{ background: "#34d399" }} />
          <span className="gpu-stat-label">KV Cache</span>
          <span className="gpu-stat-value">{formatBytes(kvCache)}</span>
          <span className="gpu-stat-pct">{kvPct.toFixed(0)}%</span>
        </div>
        <div className="gpu-stat-row">
          <span className="gpu-stat-dot" style={{ background: "#9ca3af" }} />
          <span className="gpu-stat-label">Reserved</span>
          <span className="gpu-stat-value">{formatBytes(reserved)}</span>
          <span className="gpu-stat-pct">{rPct.toFixed(0)}%</span>
        </div>

        <div className="gpu-stat-divider" />

        <div className="gpu-stat-row">
          <span className="gpu-stat-dot" style={{ background: "transparent" }} />
          <span className="gpu-stat-label">KV Slots</span>
          <span className="gpu-stat-value gpu-stat-highlight">
            {kvSlots.toLocaleString()}
          </span>
        </div>
        <div className="gpu-stat-row">
          <span className="gpu-stat-dot" style={{ background: "transparent" }} />
          <span className="gpu-stat-label">Per-Token KV</span>
          <span className="gpu-stat-value">{formatBytes(kvPerToken)}</span>
        </div>
      </div>

      {oom && (
        <div className="gpu-oom-warning">
          Weights exceed usable memory — OOM
        </div>
      )}
    </div>
  );
}
