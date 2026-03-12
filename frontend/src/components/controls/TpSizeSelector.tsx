import type { ModelConfig } from "../../types/model";
import { isTpCompatible } from "../../utils/tpMath";

const TP_OPTIONS = [1, 2, 4, 8];

interface Props {
  config: ModelConfig | null;
  selected: number;
  onSelect: (tp: number) => void;
}

export function TpSizeSelector({ config, selected, onSelect }: Props) {
  return (
    <div className="tp-selector">
      <label>TP Size</label>
      <div className="selector-buttons">
        {TP_OPTIONS.map((tp) => {
          const compatible = config ? isTpCompatible(config, tp) : true;
          return (
            <button
              key={tp}
              className={`selector-btn tp-btn ${selected === tp ? "active" : ""}`}
              disabled={!compatible}
              title={compatible ? `TP=${tp}` : `TP=${tp} incompatible (heads not divisible)`}
              onClick={() => onSelect(tp)}
            >
              {tp}
            </button>
          );
        })}
      </div>
    </div>
  );
}
