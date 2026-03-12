import { useState, useMemo } from "react";
import { useManifest, useModelData } from "./hooks/useModelData";
import { PlaneTabBar, type Plane } from "./components/controls/PlaneTabBar";
import { Sidebar } from "./components/sidebar/Sidebar";
import { PipelineView } from "./components/pipeline/PipelineView";
import { GpuMemoryPanel } from "./components/gpu/GpuMemoryPanel";
import { ControlPlaneView } from "./components/controlplane/ControlPlaneView";
import { computePerRankParams } from "./utils/tpMath";
import "./App.css";

const DEFAULT_GPU_BYTES = 80e9; // 80 GB
const DEFAULT_MEM_FRACTION = 0.88;

function App() {
  const { manifest, loading: manifestLoading } = useManifest();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const { model, loading: modelLoading } = useModelData(selectedPreset);
  const [tpSize, setTpSize] = useState(1);
  const [dpSize, setDpSize] = useState(1);
  const [gpuMemoryBytes, setGpuMemoryBytes] = useState(DEFAULT_GPU_BYTES);
  const [memFractionStatic, setMemFractionStatic] = useState(DEFAULT_MEM_FRACTION);
  const [activePlane, setActivePlane] = useState<Plane>("compute");

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
      <PlaneTabBar active={activePlane} onChange={setActivePlane} />

      <div className="app-body">
        <Sidebar
          manifest={manifest}
          selectedPreset={selectedPreset}
          onSelectModel={handleModelSelect}
          manifestLoading={manifestLoading}
          tpSize={tpSize}
          onTpSizeChange={setTpSize}
          dpSize={dpSize}
          onDpSizeChange={setDpSize}
          modelConfig={model?.config ?? null}
          activePlane={activePlane}
          gpuMemoryBytes={gpuMemoryBytes}
          onGpuMemoryChange={setGpuMemoryBytes}
          memFractionStatic={memFractionStatic}
          onMemFractionChange={setMemFractionStatic}
        />

        <main className="main-content">
          {activePlane === "compute" && (
            <>
              {modelLoading && (
                <div className="loading">Loading model data...</div>
              )}

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
                      dpSize={dpSize}
                      perRankParams={perRankParams}
                      gpuMemoryBytes={gpuMemoryBytes}
                      memFractionStatic={memFractionStatic}
                    />
                  </div>
                </div>
              )}

              {!model && !modelLoading && !manifestLoading && (
                <div className="empty-state">
                  <p>
                    Select a model to explore its tensor parallelism layout.
                  </p>
                </div>
              )}
            </>
          )}

          {activePlane === "control" && (
            <ControlPlaneView tpSize={tpSize} dpSize={dpSize} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
