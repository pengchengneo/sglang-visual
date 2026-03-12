import type { ModelConfig } from "../../types/model";
import { isTpCompatible } from "../../utils/tpMath";

const TP_OPTIONS = [1, 2, 4, 8];
const DP_OPTIONS = [1, 2, 4, 8];

interface Props {
  config: ModelConfig | null;
  tpSize: number;
  onTpSizeChange: (tp: number) => void;
  dpSize: number;
  onDpSizeChange: (dp: number) => void;
}

const FUTURE_DIMS = ["EP", "PP", "CP", "MTP"] as const;

export function ParallelismControls({
  config,
  tpSize,
  onTpSizeChange,
  dpSize,
  onDpSizeChange,
}: Props) {
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

      <div className="parallelism-row">
        <span className="parallelism-label">DP</span>
        <div className="parallelism-buttons">
          {DP_OPTIONS.map((dp) => (
            <button
              key={dp}
              className={`selector-btn tp-btn${dpSize === dp ? " active" : ""}`}
              title={`DP=${dp}`}
              onClick={() => onDpSizeChange(dp)}
            >
              {dp}
            </button>
          ))}
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

      {(dpSize > 1 || tpSize > 1) && (
        <div className="total-gpus-row">
          Total GPUs: {dpSize} × {tpSize} = {dpSize * tpSize}
        </div>
      )}
    </div>
  );
}
