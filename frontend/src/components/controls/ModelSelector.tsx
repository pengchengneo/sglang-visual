import type { PresetManifestEntry } from "../../types/model";

interface Props {
  manifest: PresetManifestEntry[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export function ModelSelector({ manifest, selected, onSelect }: Props) {
  return (
    <div className="model-selector">
      <label>Model</label>
      <div className="selector-buttons">
        {manifest.map((entry) => (
          <button
            key={entry.id}
            className={`selector-btn ${selected === entry.id ? "active" : ""}`}
            onClick={() => onSelect(entry.id)}
          >
            <span className="btn-label">{entry.model_id.split("/").pop()}</span>
            <span className="btn-meta">
              {entry.family} · {entry.num_layers}L · h={entry.hidden_size}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
