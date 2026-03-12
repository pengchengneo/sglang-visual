interface Props {
  contextLength: number;
  onContextLengthChange: (ctx: number) => void;
}

const CONTEXT_OPTIONS = [2048, 4096, 8192, 32768, 131072];

export function ContextControls({
  contextLength,
  onContextLengthChange,
}: Props) {
  return (
    <div className="parallelism-controls">
      <div className="parallelism-row">
        <span className="parallelism-label">ctx</span>
        <div className="parallelism-buttons">
          {CONTEXT_OPTIONS.map((ctx) => (
            <button
              key={ctx}
              className={`selector-btn param-btn${contextLength === ctx ? " active" : ""}`}
              onClick={() => onContextLengthChange(ctx)}
            >
              {ctx >= 1024 ? `${ctx / 1024}K` : ctx}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
