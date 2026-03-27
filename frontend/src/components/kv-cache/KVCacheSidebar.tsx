import { SCENARIOS } from "./scenarioPresets";

interface Props {
  selectedScenario: string;
  onSelectScenario: (id: string) => void;
  maxBlocks: number;
  onMaxBlocksChange: (n: number) => void;
}

export default function KVCacheSidebar({
  selectedScenario,
  onSelectScenario,
  maxBlocks,
  onMaxBlocksChange,
}: Props) {
  return (
    <div className="kv-cache-sidebar">
      <span className="control-label">Scenario</span>
      <div className="scenario-list">
        {SCENARIOS.map((s) => (
          <button
            key={s.id}
            className={`scenario-btn${selectedScenario === s.id ? " active" : ""}`}
            onClick={() => onSelectScenario(s.id)}
            title={s.description}
          >
            {s.name}
          </button>
        ))}
      </div>

      <span className="control-label" style={{ marginLeft: "auto" }}>
        Capacity: {maxBlocks}
      </span>
      <input
        type="range"
        min={10}
        max={40}
        value={maxBlocks}
        onChange={(e) => onMaxBlocksChange(Number(e.target.value))}
        className="sidebar-range"
      />
    </div>
  );
}
