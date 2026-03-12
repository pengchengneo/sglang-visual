import { GPU_MEMORY_OPTIONS } from "../../utils/gpuMemoryMath";

interface Props {
  gpuMemoryBytes: number;
  onGpuMemoryChange: (bytes: number) => void;
  memFractionStatic: number;
  onMemFractionChange: (fraction: number) => void;
}

export function GpuControls({
  gpuMemoryBytes,
  onGpuMemoryChange,
  memFractionStatic,
  onMemFractionChange,
}: Props) {
  return (
    <>
      <div className="tp-selector">
        <label>GPU Memory</label>
        <div className="selector-buttons">
          {GPU_MEMORY_OPTIONS.map((opt) => (
            <button
              key={opt.bytes}
              className={`selector-btn tp-btn${gpuMemoryBytes === opt.bytes ? " active" : ""}`}
              onClick={() => onGpuMemoryChange(opt.bytes)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="tp-selector">
        <label>
          mem-fraction-static{" "}
          <span className="mem-fraction-value">{memFractionStatic.toFixed(2)}</span>
        </label>
        <div className="mem-fraction-slider">
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={memFractionStatic}
            onChange={(e) => onMemFractionChange(parseFloat(e.target.value))}
          />
        </div>
      </div>
    </>
  );
}
