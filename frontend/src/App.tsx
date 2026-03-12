import { useState } from "react";
import { useManifest, useModelData } from "./hooks/useModelData";
import { ModelSelector } from "./components/controls/ModelSelector";
import { TpSizeSelector } from "./components/controls/TpSizeSelector";
import { PipelineView } from "./components/pipeline/PipelineView";
import "./App.css";

function App() {
  const { manifest, loading: manifestLoading } = useManifest();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const { model, loading: modelLoading } = useModelData(selectedPreset);
  const [tpSize, setTpSize] = useState(1);

  const handleModelSelect = (id: string) => {
    setSelectedPreset(id);
    setTpSize(1);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-left">
          <h1>SGLang TP Visualizer</h1>
          <span className="subtitle">
            Tensor parallelism weight partitioning across GPUs
          </span>
        </div>
      </header>

      <div className="controls">
        {manifestLoading ? (
          <div className="loading">Loading models...</div>
        ) : (
          <ModelSelector
            manifest={manifest}
            selected={selectedPreset}
            onSelect={handleModelSelect}
          />
        )}
        <TpSizeSelector
          config={model?.config ?? null}
          selected={tpSize}
          onSelect={setTpSize}
        />
      </div>

      <main className="main-content">
        {modelLoading && <div className="loading">Loading model data...</div>}

        {model && (
          <PipelineView
            key={model.model_id}
            model={model}
            tpSize={tpSize}
          />
        )}

        {!model && !modelLoading && !manifestLoading && (
          <div className="empty-state">
            <p>Select a model above to explore its tensor parallelism layout.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
