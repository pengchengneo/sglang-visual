import "./shared.css";

export interface Metric {
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
}

interface Props { metrics: Metric[]; }

export default function MetricsPanel({ metrics }: Props) {
  return (
    <div className="metrics-panel">
      {metrics.map((m) => (
        <div key={m.label} className="metric-item">
          <span className="metric-value" style={m.color ? { color: m.color } : undefined}>
            {m.value}{m.unit && <span className="metric-unit">{m.unit}</span>}
          </span>
          <span className="metric-label">{m.label}</span>
        </div>
      ))}
    </div>
  );
}
