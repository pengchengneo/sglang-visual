import type { Dtype, Quantization, KvCacheDtype } from "../../contexts/AppContext";

interface Props {
  dtype: Dtype;
  onDtypeChange: (dtype: Dtype) => void;
  quantization: Quantization;
  onQuantizationChange: (q: Quantization) => void;
  kvCacheDtype: KvCacheDtype;
  onKvCacheDtypeChange: (kv: KvCacheDtype) => void;
}

const DTYPE_OPTIONS: { value: Dtype; label: string }[] = [
  { value: "fp16", label: "FP16" },
  { value: "bf16", label: "BF16" },
  { value: "fp32", label: "FP32" },
];

const QUANT_OPTIONS: { value: Quantization; label: string }[] = [
  { value: "none", label: "None" },
  { value: "int8", label: "INT8" },
  { value: "int4", label: "INT4" },
  { value: "fp8", label: "FP8" },
];

const KV_OPTIONS: { value: KvCacheDtype; label: string }[] = [
  { value: "fp16", label: "FP16" },
  { value: "fp8", label: "FP8" },
];

export function QuantizationControls({
  dtype,
  onDtypeChange,
  quantization,
  onQuantizationChange,
  kvCacheDtype,
  onKvCacheDtypeChange,
}: Props) {
  return (
    <div className="parallelism-controls">
      <div className="parallelism-row">
        <span className="parallelism-label">dtype</span>
        <div className="parallelism-buttons">
          {DTYPE_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              className={`selector-btn param-btn${dtype === opt.value ? " active" : ""}`}
              onClick={() => onDtypeChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="parallelism-row">
        <span className="parallelism-label">quant</span>
        <div className="parallelism-buttons">
          {QUANT_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              className={`selector-btn param-btn${quantization === opt.value ? " active" : ""}`}
              onClick={() => onQuantizationChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="parallelism-row">
        <span className="parallelism-label">KV$</span>
        <div className="parallelism-buttons">
          {KV_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              className={`selector-btn param-btn${kvCacheDtype === opt.value ? " active" : ""}`}
              onClick={() => onKvCacheDtypeChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
