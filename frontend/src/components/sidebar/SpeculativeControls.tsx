import type { SpecAlgorithm } from "../../App";

interface Props {
  specAlgorithm: SpecAlgorithm;
  onSpecAlgorithmChange: (alg: SpecAlgorithm) => void;
  specNumDraftTokens: number;
  onSpecNumDraftTokensChange: (n: number) => void;
}

const SPEC_OPTIONS: { value: SpecAlgorithm; label: string }[] = [
  { value: "none", label: "None" },
  { value: "eagle", label: "EAGLE" },
  { value: "eagle3", label: "EAGLE3" },
  { value: "nextn", label: "NextN" },
  { value: "ngram", label: "NGram" },
];

const DRAFT_TOKEN_OPTIONS = [3, 5, 8, 12];

export function SpeculativeControls({
  specAlgorithm,
  onSpecAlgorithmChange,
  specNumDraftTokens,
  onSpecNumDraftTokensChange,
}: Props) {
  return (
    <div className="parallelism-controls">
      <div className="parallelism-row">
        <span className="parallelism-label">algo</span>
        <div className="parallelism-buttons">
          {SPEC_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              className={`selector-btn param-btn${specAlgorithm === opt.value ? " active" : ""}`}
              onClick={() => onSpecAlgorithmChange(opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {specAlgorithm !== "none" && (
        <div className="parallelism-row">
          <span className="parallelism-label">draft</span>
          <div className="parallelism-buttons">
            {DRAFT_TOKEN_OPTIONS.map((n) => (
              <button
                key={n}
                className={`selector-btn tp-btn${specNumDraftTokens === n ? " active" : ""}`}
                onClick={() => onSpecNumDraftTokensChange(n)}
              >
                {n}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
