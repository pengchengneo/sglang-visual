import { useState, useMemo, useEffect } from "react";
import { useManifest, useModelData } from "./hooks/useModelData";
import { PlaneTabBar, type Plane } from "./components/controls/PlaneTabBar";
import { Sidebar } from "./components/sidebar/Sidebar";
import { PipelineView } from "./components/pipeline/PipelineView";
import { GpuMemoryPanel } from "./components/gpu/GpuMemoryPanel";
import { ControlPlaneView } from "./components/controlplane/ControlPlaneView";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { computePerRankParams, computePerRankParamsForPpStage } from "./utils/tpMath";
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

type Theme = "light" | "dark" | "system";

function getInitialTheme(): Theme {
  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") return stored;
  return "system";
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", theme);
  }
}

function useTheme() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);

  useEffect(() => {
    applyTheme(theme);
    if (theme === "system") {
      localStorage.removeItem("theme");
    } else {
      localStorage.setItem("theme", theme);
    }
  }, [theme]);

  const toggle = () => {
    setTheme((prev) => {
      if (prev === "system") {
        const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        return isDark ? "light" : "dark";
      }
      return prev === "light" ? "dark" : "light";
    });
  };

  const isDark =
    theme === "dark" ||
    (theme === "system" &&
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);

  return { theme, isDark, toggle };
}

function App() {
  const { manifest, loading: manifestLoading, error: manifestError } = useManifest();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const { model, loading: modelLoading, error: modelError } = useModelData(selectedPreset);
  const { isDark, toggle: toggleTheme } = useTheme();

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
    () => {
      if (!model) return 0;
      if (ppSize <= 1) return computePerRankParams(model, tpSize, epSize);
      // Use the max across all PP stages for worst-case GPU memory
      let maxParams = 0;
      for (let p = 0; p < ppSize; p++) {
        maxParams = Math.max(maxParams, computePerRankParamsForPpStage(model, tpSize, p, ppSize, epSize));
      }
      return maxParams;
    },
    [model, tpSize, ppSize, epSize],
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
          {manifestError && <div className="load-error">{manifestError}</div>}
          {modelError && <div className="load-error">{modelError}</div>}

          {activePlane === "compute" && (
            <>
              {modelLoading && (
                <div className="loading">Loading model data...</div>
              )}

              {model && (
                <div className="split-layout">
                  <div className="panel-left">
                    <ErrorBoundary fallbackTitle="Pipeline view error">
                      <PipelineView
                        key={model.model_id}
                        model={model}
                        tpSize={tpSize}
                        ppSize={ppSize}
                        epSize={epSize}
                        bytesPerParam={bytesPerParam}
                        dtype={dtype}
                        quantization={quantization}
                      />
                    </ErrorBoundary>
                  </div>
                  <div className="panel-right">
                    <ErrorBoundary fallbackTitle="GPU memory panel error">
                      <GpuMemoryPanel
                        config={model.config}
                        tpSize={tpSize}
                        dpSize={dpSize}
                        ppSize={ppSize}
                        epSize={epSize}
                        enableDpAttention={enableDpAttention}
                        perRankParams={perRankParams}
                        gpuMemoryBytes={gpuMemoryBytes}
                        memFractionStatic={memFractionStatic}
                        bytesPerParam={bytesPerParam}
                        kvBytesPerElement={kvBytesPerElement}
                        contextLength={contextLength}
                      />
                    </ErrorBoundary>
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
            <ErrorBoundary fallbackTitle="Control plane error">
              <ControlPlaneView
                tpSize={tpSize}
                dpSize={dpSize}
                ppSize={ppSize}
                epSize={epSize}
                enableDpAttention={enableDpAttention}
                modelConfig={model?.config ?? null}
                schedulePolicy={schedulePolicy}
                chunkedPrefillSize={chunkedPrefillSize}
                disableRadixCache={disableRadixCache}
                specAlgorithm={specAlgorithm}
                specNumDraftTokens={specNumDraftTokens}
                cudaGraphMaxBs={cudaGraphMaxBs}
                disableCudaGraph={disableCudaGraph}
              />
            </ErrorBoundary>
          )}
        </main>
      </div>

      <button
        className="theme-toggle"
        onClick={toggleTheme}
        title="Toggle theme"
        aria-label="Toggle theme"
      >
        {isDark ? "\u2600\uFE0F" : "\uD83C\uDF19"}
      </button>
    </div>
  );
}

export default App;
