import type { ModelConfig } from "../../types/model";
import { isTpCompatible } from "../../utils/tpMath";

const TP_OPTIONS = [1, 2, 4, 8];
const DP_OPTIONS = [1, 2, 4, 8];
const PP_OPTIONS = [1, 2, 4];
const EP_OPTIONS = [1, 2, 4, 8];

interface Props {
  config: ModelConfig | null;
  tpSize: number;
  onTpSizeChange: (tp: number) => void;
  dpSize: number;
  onDpSizeChange: (dp: number) => void;
  ppSize: number;
  onPpSizeChange: (pp: number) => void;
  epSize: number;
  onEpSizeChange: (ep: number) => void;
  enableDpAttention: boolean;
  onEnableDpAttentionChange: (enabled: boolean) => void;
}

const FUTURE_DIMS = ["CP", "MTP"] as const;

export function ParallelismControls({
  config,
  tpSize,
  onTpSizeChange,
  dpSize,
  onDpSizeChange,
  ppSize,
  onPpSizeChange,
  epSize,
  onEpSizeChange,
  enableDpAttention,
  onEnableDpAttentionChange,
}: Props) {
  const totalGpus = tpSize * ppSize;
  const attnTpSize = enableDpAttention && dpSize > 1 ? tpSize / dpSize : tpSize;
  const hasMoe = config?.n_routed_experts != null;

  const handleDpAttentionToggle = (checked: boolean) => {
    onEnableDpAttentionChange(checked);
    if (!checked) {
      onDpSizeChange(1);
    }
  };

  return (
    <div className="parallelism-controls">
      {/* TP */}
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

      {/* DP Attention */}
      <div className="parallelism-row">
        <span className="parallelism-label">DP</span>
        <div className="parallelism-buttons">
          {DP_OPTIONS.map((dp) => {
            const disabled = !enableDpAttention || (tpSize % dp !== 0);
            return (
              <button
                key={dp}
                className={`selector-btn tp-btn${dpSize === dp ? " active" : ""}`}
                disabled={disabled}
                title={
                  !enableDpAttention
                    ? "Enable DP Attention first"
                    : tpSize % dp !== 0
                      ? `TP=${tpSize} not divisible by DP=${dp}`
                      : `DP Attention=${dp}`
                }
                onClick={() => onDpSizeChange(dp)}
              >
                {dp}
              </button>
            );
          })}
          <label className="param-toggle" title="Enable DP Attention (dp_attention)">
            <input
              type="checkbox"
              checked={enableDpAttention}
              onChange={(e) => handleDpAttentionToggle(e.target.checked)}
            />
            <span className="param-toggle-text">dp_attn</span>
          </label>
        </div>
      </div>

      {/* PP */}
      <div className="parallelism-row">
        <span className="parallelism-label">PP</span>
        <div className="parallelism-buttons">
          {PP_OPTIONS.map((pp) => (
            <button
              key={pp}
              className={`selector-btn tp-btn${ppSize === pp ? " active" : ""}`}
              title={`PP=${pp}`}
              onClick={() => onPpSizeChange(pp)}
            >
              {pp}
            </button>
          ))}
        </div>
      </div>

      {/* EP — only shown for MoE models */}
      {hasMoe && (
        <div className="parallelism-row">
          <span className="parallelism-label">EP</span>
          <div className="parallelism-buttons">
            {EP_OPTIONS.map((ep) => {
              const disabled = ep > (config?.n_routed_experts ?? 0);
              return (
                <button
                  key={ep}
                  className={`selector-btn tp-btn${epSize === ep ? " active" : ""}`}
                  disabled={disabled}
                  title={
                    disabled
                      ? `EP=${ep} exceeds num experts (${config?.n_routed_experts})`
                      : `EP=${ep}`
                  }
                  onClick={() => onEpSizeChange(ep)}
                >
                  {ep}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Future dims */}
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

      {/* Summary */}
      {(tpSize > 1 || ppSize > 1) && (
        <div className="total-gpus-row">
          Total GPUs: {tpSize}
          {ppSize > 1 && ` × ${ppSize}`}
          {(tpSize > 1 || ppSize > 1) && ` = ${totalGpus}`}
        </div>
      )}

      {enableDpAttention && dpSize > 1 && (
        <div className="parallelism-sub-info">
          Attn: {dpSize} groups × {attnTpSize} TP
        </div>
      )}
    </div>
  );
}
