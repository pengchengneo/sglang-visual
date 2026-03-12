import type { ModelConfig } from "../../types/model";
import { isTpCompatible } from "../../utils/tpMath";

const TP_OPTIONS = [1, 2, 4, 8];

interface Props {
  config: ModelConfig | null;
  tpSize: number;
  onTpSizeChange: (tp: number) => void;
}

const FUTURE_DIMS = ["DP", "EP", "PP", "CP", "MTP"] as const;

export function ParallelismControls({ config, tpSize, onTpSizeChange }: Props) {
  return (
    <div className="parallelism-controls">
      <div className="parallelism-row">
        <span className="parallelism-label">TP</span>
        <div className="parallelism-buttons">
          {TP_OPTIONS.map((tp) => {
            const compatible = config ? isTpCompatible(config, tp) : true;
            return (
              <button
                key={tp}
                className={`selector-btn tp-btn${tpSize === tp ? " active" : ""}`}
                disabled={!compatible}
                title={
                  compatible
                    ? `TP=${tp}`
                    : `TP=${tp} incompatible (heads not divisible)`
                }
                onClick={() => onTpSizeChange(tp)}
              >
                {tp}
              </button>
            );
          })}
        </div>
      </div>

      {FUTURE_DIMS.map((dim) => (
        <div className="parallelism-row" key={dim}>
          <span className="parallelism-label">{dim}</span>
          <div className="parallelism-buttons">
            <button className="selector-btn tp-btn active" disabled>
              1
            </button>
            <span className="coming-soon">coming soon</span>
          </div>
        </div>
      ))}
    </div>
  );
}
