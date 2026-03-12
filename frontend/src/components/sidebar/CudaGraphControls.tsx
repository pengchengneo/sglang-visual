interface Props {
  cudaGraphMaxBs: number;
  onCudaGraphMaxBsChange: (bs: number) => void;
  disableCudaGraph: boolean;
  onDisableCudaGraphChange: (disabled: boolean) => void;
}

const BS_OPTIONS = [16, 32, 64, 128, 256, 512];

export function CudaGraphControls({
  cudaGraphMaxBs,
  onCudaGraphMaxBsChange,
  disableCudaGraph,
  onDisableCudaGraphChange,
}: Props) {
  return (
    <div className="parallelism-controls">
      <div className="parallelism-row">
        <span className="parallelism-label">enable</span>
        <div className="parallelism-buttons">
          <label className="param-toggle">
            <input
              type="checkbox"
              checked={!disableCudaGraph}
              onChange={(e) => onDisableCudaGraphChange(!e.target.checked)}
            />
            <span className="param-toggle-text">
              {disableCudaGraph ? "disabled" : "enabled"}
            </span>
          </label>
        </div>
      </div>

      {!disableCudaGraph && (
        <div className="parallelism-row">
          <span className="parallelism-label">maxBS</span>
          <div className="parallelism-buttons">
            {BS_OPTIONS.map((bs) => (
              <button
                key={bs}
                className={`selector-btn param-btn${cudaGraphMaxBs === bs ? " active" : ""}`}
                onClick={() => onCudaGraphMaxBsChange(bs)}
              >
                {bs}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
