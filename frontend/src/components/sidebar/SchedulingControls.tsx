import type { SchedulePolicy } from "../../contexts/AppContext";

interface Props {
  schedulePolicy: SchedulePolicy;
  onSchedulePolicyChange: (policy: SchedulePolicy) => void;
  chunkedPrefillSize: number;
  onChunkedPrefillSizeChange: (size: number) => void;
  disableRadixCache: boolean;
  onDisableRadixCacheChange: (disabled: boolean) => void;
}

const POLICY_OPTIONS: { value: SchedulePolicy; label: string }[] = [
  { value: "fcfs", label: "FCFS" },
  { value: "lpm", label: "LPM" },
  { value: "random", label: "Random" },
  { value: "dfs-weight", label: "DFS-W" },
];

const CHUNK_OPTIONS = [2048, 4096, 8192, 16384];

export function SchedulingControls({
  schedulePolicy,
  onSchedulePolicyChange,
  chunkedPrefillSize,
  onChunkedPrefillSizeChange,
  disableRadixCache,
  onDisableRadixCacheChange,
}: Props) {
  return (
    <div className="parallelism-controls">
      <div className="parallelism-row">
        <span className="parallelism-label">policy</span>
        <div className="parallelism-buttons">
          {POLICY_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              className={`selector-btn param-btn${schedulePolicy === opt.value ? " active" : ""}`}
              onClick={() => onSchedulePolicyChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="parallelism-row">
        <span className="parallelism-label">chunk</span>
        <div className="parallelism-buttons">
          {CHUNK_OPTIONS.map((size) => (
            <button
              key={size}
              className={`selector-btn param-btn${chunkedPrefillSize === size ? " active" : ""}`}
              onClick={() => onChunkedPrefillSizeChange(size)}
            >
              {size >= 1024 ? `${size / 1024}K` : size}
            </button>
          ))}
        </div>
      </div>

      <div className="parallelism-row">
        <span className="parallelism-label">radix</span>
        <div className="parallelism-buttons">
          <label className="param-toggle">
            <input
              type="checkbox"
              checked={!disableRadixCache}
              onChange={(e) => onDisableRadixCacheChange(!e.target.checked)}
            />
            <span className="param-toggle-text">
              {disableRadixCache ? "disabled" : "enabled"}
            </span>
          </label>
        </div>
      </div>
    </div>
  );
}
