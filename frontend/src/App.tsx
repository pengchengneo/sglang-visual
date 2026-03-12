import { useState, useMemo } from "react";
import { useManifest, useModelData } from "./hooks/useModelData";
import { ModelSelector } from "./components/controls/ModelSelector";
import { TpSizeSelector } from "./components/controls/TpSizeSelector";
import { GpuControls } from "./components/controls/GpuControls";
import { PipelineView } from "./components/pipeline/PipelineView";
import { GpuMemoryPanel } from "./components/gpu/GpuMemoryPanel";
import { computePerRankParams } from "./utils/tpMath";
import "./App.css";

const DEFAULT_GPU_BYTES = 80e9; // 80 GB
const DEFAULT_MEM_FRACTION = 0.88;

function App() {
  const { manifest, loading: manifestLoading } = useManifest();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const { model, loading: modelLoading } = useModelData(selectedPreset);
  const [tpSize, setTpSize] = useState(1);
  const [gpuMemoryBytes, setGpuMemoryBytes] = useState(DEFAULT_GPU_BYTES);
  const [memFractionStatic, setMemFractionStatic] = useState(DEFAULT_MEM_FRACTION);

  const handleModelSelect = (id: string) => {
    setSelectedPreset(id);
    setTpSize(1);
  };

  const perRankParams = useMemo(
    () => (model ? computePerRankParams(model, tpSize) : 0),
    [model, tpSize],
  );

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
        <GpuControls
          gpuMemoryBytes={gpuMemoryBytes}
          onGpuMemoryChange={setGpuMemoryBytes}
          memFractionStatic={memFractionStatic}
          onMemFractionChange={setMemFractionStatic}
        />
      </div>

      <main className="main-content">
        {modelLoading && <div className="loading">Loading model data...</div>}

        {model && (
          <div className="split-layout">
            <div className="panel-left">
              <PipelineView
                key={model.model_id}
                model={model}
                tpSize={tpSize}
              />
            </div>
            <div className="panel-right">
              <GpuMemoryPanel
                config={model.config}
                tpSize={tpSize}
                perRankParams={perRankParams}
                gpuMemoryBytes={gpuMemoryBytes}
                memFractionStatic={memFractionStatic}
              />
            </div>
          </div>
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
