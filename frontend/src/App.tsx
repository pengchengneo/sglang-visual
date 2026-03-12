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

export type Dtype = "fp16" | "bf16" | "fp32";
export type Quantization = "none" | "int8" | "int4" | "fp8";
export type KvCacheDtype = "fp16" | "fp8";
export type SchedulePolicy = "fcfs" | "lpm" | "random" | "dfs-weight";
export type SpecAlgorithm = "none" | "eagle" | "eagle3" | "nextn" | "ngram";

const BYTES_PER_PARAM: Record<Quantization, number> & Record<Dtype, number> = {
  none: 2, int8: 1, int4: 0.5, fp8: 1,
  fp16: 2, bf16: 2, fp32: 4,
};

function App() {
  const { manifest, loading: manifestLoading } = useManifest();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const { model, loading: modelLoading } = useModelData(selectedPreset);

  // Parallelism
  const [tpSize, setTpSize] = useState(1);
  const [dpSize, setDpSize] = useState(1);
  const [ppSize, setPpSize] = useState(1);
  const [epSize, setEpSize] = useState(1);
  const [enableDpAttention, setEnableDpAttention] = useState(false);

  // GPU
  const [gpuMemoryBytes, setGpuMemoryBytes] = useState(DEFAULT_GPU_BYTES);
  const [memFractionStatic, setMemFractionStatic] = useState(DEFAULT_MEM_FRACTION);

  // Quantization
  const [dtype, setDtype] = useState<Dtype>("fp16");
  const [quantization, setQuantization] = useState<Quantization>("none");
  const [kvCacheDtype, setKvCacheDtype] = useState<KvCacheDtype>("fp16");

  // Scheduling
  const [schedulePolicy, setSchedulePolicy] = useState<SchedulePolicy>("fcfs");
  const [chunkedPrefillSize, setChunkedPrefillSize] = useState(8192);
  const [disableRadixCache, setDisableRadixCache] = useState(false);

  // Context
  const [contextLength, setContextLength] = useState(4096);

  // Speculative Decoding
  const [specAlgorithm, setSpecAlgorithm] = useState<SpecAlgorithm>("none");
  const [specNumDraftTokens, setSpecNumDraftTokens] = useState(5);

  // CUDA Graph
  const [cudaGraphMaxBs, setCudaGraphMaxBs] = useState(128);
  const [disableCudaGraph, setDisableCudaGraph] = useState(false);

  const [activePlane, setActivePlane] = useState<Plane>("compute");

  const handleModelSelect = (id: string) => {
    setSelectedPreset(id);
    setTpSize(1);
  };

  // Derived quantization values
  const bytesPerParam = quantization !== "none"
    ? BYTES_PER_PARAM[quantization]
    : BYTES_PER_PARAM[dtype];
  const kvBytesPerElement = kvCacheDtype === "fp8" ? 1 : 2;

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
          ppSize={ppSize}
          onPpSizeChange={setPpSize}
          epSize={epSize}
          onEpSizeChange={setEpSize}
          enableDpAttention={enableDpAttention}
          onEnableDpAttentionChange={setEnableDpAttention}
          modelConfig={model?.config ?? null}
          activePlane={activePlane}
          gpuMemoryBytes={gpuMemoryBytes}
          onGpuMemoryChange={setGpuMemoryBytes}
          memFractionStatic={memFractionStatic}
          onMemFractionChange={setMemFractionStatic}
          dtype={dtype}
          onDtypeChange={setDtype}
          quantization={quantization}
          onQuantizationChange={setQuantization}
          kvCacheDtype={kvCacheDtype}
          onKvCacheDtypeChange={setKvCacheDtype}
          schedulePolicy={schedulePolicy}
          onSchedulePolicyChange={setSchedulePolicy}
          chunkedPrefillSize={chunkedPrefillSize}
          onChunkedPrefillSizeChange={setChunkedPrefillSize}
          disableRadixCache={disableRadixCache}
          onDisableRadixCacheChange={setDisableRadixCache}
          contextLength={contextLength}
          onContextLengthChange={setContextLength}
          specAlgorithm={specAlgorithm}
          onSpecAlgorithmChange={setSpecAlgorithm}
          specNumDraftTokens={specNumDraftTokens}
          onSpecNumDraftTokensChange={setSpecNumDraftTokens}
          cudaGraphMaxBs={cudaGraphMaxBs}
          onCudaGraphMaxBsChange={setCudaGraphMaxBs}
          disableCudaGraph={disableCudaGraph}
          onDisableCudaGraphChange={setDisableCudaGraph}
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
                      bytesPerParam={bytesPerParam}
                      dtype={dtype}
                      quantization={quantization}
                    />
                  </div>
                  <div className="panel-right">
                    <GpuMemoryPanel
                      config={model.config}
                      tpSize={tpSize}
                      dpSize={dpSize}
                      ppSize={ppSize}
                      enableDpAttention={enableDpAttention}
                      perRankParams={perRankParams}
                      gpuMemoryBytes={gpuMemoryBytes}
                      memFractionStatic={memFractionStatic}
                      bytesPerParam={bytesPerParam}
                      kvBytesPerElement={kvBytesPerElement}
                      contextLength={contextLength}
                    />
                  </div>
                </div>
              )}

              {!model && !modelLoading && !manifestLoading && (
                <div className="empty-state">
                  <p>
                    Select a model to explore inference optimizations.
                  </p>
                </div>
              )}
            </>
          )}

          {activePlane === "control" && (
            <ControlPlaneView
              tpSize={tpSize}
              dpSize={dpSize}
              ppSize={ppSize}
              enableDpAttention={enableDpAttention}
              schedulePolicy={schedulePolicy}
              chunkedPrefillSize={chunkedPrefillSize}
              disableRadixCache={disableRadixCache}
              specAlgorithm={specAlgorithm}
              specNumDraftTokens={specNumDraftTokens}
              cudaGraphMaxBs={cudaGraphMaxBs}
              disableCudaGraph={disableCudaGraph}
            />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
