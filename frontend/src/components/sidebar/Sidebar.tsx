import { useState } from "react";
import type { PresetManifestEntry, ModelConfig } from "../../types/model";
import type { Plane } from "../controls/PlaneTabBar";
import type {
  Dtype,
  Quantization,
  KvCacheDtype,
  SchedulePolicy,
  SpecAlgorithm,
} from "../../App";
import { ModelSelector } from "../controls/ModelSelector";
import { GpuControls } from "../controls/GpuControls";
import { SidebarSection } from "./SidebarSection";
import { ParallelismControls } from "./ParallelismControls";
import { QuantizationControls } from "./QuantizationControls";
import { SchedulingControls } from "./SchedulingControls";
import { ContextControls } from "./ContextControls";
import { SpeculativeControls } from "./SpeculativeControls";
import { CudaGraphControls } from "./CudaGraphControls";

interface Props {
  manifest: PresetManifestEntry[];
  selectedPreset: string | null;
  onSelectModel: (id: string) => void;
  manifestLoading: boolean;
  tpSize: number;
  onTpSizeChange: (tp: number) => void;
  dpSize: number;
  onDpSizeChange: (dp: number) => void;
  ppSize: number;
  onPpSizeChange: (pp: number) => void;
  epSize: number;
  onEpSizeChange: (ep: number) => void;
  enableDpAttention: boolean;
  onEnableDpAttentionChange: (enabled: boolean) => void;
  modelConfig: ModelConfig | null;
  activePlane: Plane;
  gpuMemoryBytes: number;
  onGpuMemoryChange: (bytes: number) => void;
  memFractionStatic: number;
  onMemFractionChange: (fraction: number) => void;
  dtype: Dtype;
  onDtypeChange: (dtype: Dtype) => void;
  quantization: Quantization;
  onQuantizationChange: (q: Quantization) => void;
  kvCacheDtype: KvCacheDtype;
  onKvCacheDtypeChange: (kv: KvCacheDtype) => void;
  schedulePolicy: SchedulePolicy;
  onSchedulePolicyChange: (policy: SchedulePolicy) => void;
  chunkedPrefillSize: number;
  onChunkedPrefillSizeChange: (size: number) => void;
  disableRadixCache: boolean;
  onDisableRadixCacheChange: (disabled: boolean) => void;
  contextLength: number;
  onContextLengthChange: (ctx: number) => void;
  specAlgorithm: SpecAlgorithm;
  onSpecAlgorithmChange: (alg: SpecAlgorithm) => void;
  specNumDraftTokens: number;
  onSpecNumDraftTokensChange: (n: number) => void;
  cudaGraphMaxBs: number;
  onCudaGraphMaxBsChange: (bs: number) => void;
  disableCudaGraph: boolean;
  onDisableCudaGraphChange: (disabled: boolean) => void;
}

const DEFAULT_SECTIONS = new Set(["model", "parallelism"]);

export function Sidebar({
  manifest,
  selectedPreset,
  onSelectModel,
  manifestLoading,
  tpSize,
  onTpSizeChange,
  dpSize,
  onDpSizeChange,
  ppSize,
  onPpSizeChange,
  epSize,
  onEpSizeChange,
  enableDpAttention,
  onEnableDpAttentionChange,
  modelConfig,
  activePlane,
  gpuMemoryBytes,
  onGpuMemoryChange,
  memFractionStatic,
  onMemFractionChange,
  dtype,
  onDtypeChange,
  quantization,
  onQuantizationChange,
  kvCacheDtype,
  onKvCacheDtypeChange,
  schedulePolicy,
  onSchedulePolicyChange,
  chunkedPrefillSize,
  onChunkedPrefillSizeChange,
  disableRadixCache,
  onDisableRadixCacheChange,
  contextLength,
  onContextLengthChange,
  specAlgorithm,
  onSpecAlgorithmChange,
  specNumDraftTokens,
  onSpecNumDraftTokensChange,
  cudaGraphMaxBs,
  onCudaGraphMaxBsChange,
  disableCudaGraph,
  onDisableCudaGraphChange,
}: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [openSections, setOpenSections] = useState<Set<string>>(DEFAULT_SECTIONS);

  const toggle = (key: string) => {
    setOpenSections((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  return (
    <>
      <aside className={`sidebar ${collapsed ? "sidebar-collapsed" : ""}`}>
        <div className="sidebar-header">
          <div className="sidebar-title-group">
            <h1 className="sidebar-title">SGLang Visualizer</h1>
            <span className="sidebar-subtitle">
              Inference optimization visualizer
            </span>
          </div>
          <button
            className="sidebar-collapse-btn"
            onClick={() => setCollapsed(true)}
            title="Collapse sidebar"
          >
            &laquo;
          </button>
        </div>

        <SidebarSection
          title="Model"
          open={openSections.has("model")}
          onToggle={() => toggle("model")}
        >
          {manifestLoading ? (
            <div className="loading">Loading models...</div>
          ) : (
            <ModelSelector
              manifest={manifest}
              selected={selectedPreset}
              onSelect={onSelectModel}
              vertical
            />
          )}
        </SidebarSection>

        <SidebarSection
          title="Parallelism"
          open={openSections.has("parallelism")}
          onToggle={() => toggle("parallelism")}
        >
          <ParallelismControls
            config={modelConfig}
            tpSize={tpSize}
            onTpSizeChange={onTpSizeChange}
            dpSize={dpSize}
            onDpSizeChange={onDpSizeChange}
            ppSize={ppSize}
            onPpSizeChange={onPpSizeChange}
            epSize={epSize}
            onEpSizeChange={onEpSizeChange}
            enableDpAttention={enableDpAttention}
            onEnableDpAttentionChange={onEnableDpAttentionChange}
          />
        </SidebarSection>

        <SidebarSection
          title="Quantization"
          open={openSections.has("quantization")}
          onToggle={() => toggle("quantization")}
        >
          <QuantizationControls
            dtype={dtype}
            onDtypeChange={onDtypeChange}
            quantization={quantization}
            onQuantizationChange={onQuantizationChange}
            kvCacheDtype={kvCacheDtype}
            onKvCacheDtypeChange={onKvCacheDtypeChange}
          />
        </SidebarSection>

        {activePlane === "control" && (
          <>
            <SidebarSection
              title="Scheduling"
              open={openSections.has("scheduling")}
              onToggle={() => toggle("scheduling")}
            >
              <SchedulingControls
                schedulePolicy={schedulePolicy}
                onSchedulePolicyChange={onSchedulePolicyChange}
                chunkedPrefillSize={chunkedPrefillSize}
                onChunkedPrefillSizeChange={onChunkedPrefillSizeChange}
                disableRadixCache={disableRadixCache}
                onDisableRadixCacheChange={onDisableRadixCacheChange}
              />
            </SidebarSection>

            <SidebarSection
              title="Speculative Decoding"
              open={openSections.has("speculative")}
              onToggle={() => toggle("speculative")}
            >
              <SpeculativeControls
                specAlgorithm={specAlgorithm}
                onSpecAlgorithmChange={onSpecAlgorithmChange}
                specNumDraftTokens={specNumDraftTokens}
                onSpecNumDraftTokensChange={onSpecNumDraftTokensChange}
              />
            </SidebarSection>

            <SidebarSection
              title="CUDA Graph"
              open={openSections.has("cudagraph")}
              onToggle={() => toggle("cudagraph")}
            >
              <CudaGraphControls
                cudaGraphMaxBs={cudaGraphMaxBs}
                onCudaGraphMaxBsChange={onCudaGraphMaxBsChange}
                disableCudaGraph={disableCudaGraph}
                onDisableCudaGraphChange={onDisableCudaGraphChange}
              />
            </SidebarSection>
          </>
        )}

        {activePlane === "compute" && (
          <>
            <SidebarSection
              title="Context"
              open={openSections.has("context")}
              onToggle={() => toggle("context")}
            >
              <ContextControls
                contextLength={contextLength}
                onContextLengthChange={onContextLengthChange}
              />
            </SidebarSection>

            <SidebarSection
              title="GPU"
              open={openSections.has("gpu")}
              onToggle={() => toggle("gpu")}
            >
              <GpuControls
                gpuMemoryBytes={gpuMemoryBytes}
                onGpuMemoryChange={onGpuMemoryChange}
                memFractionStatic={memFractionStatic}
                onMemFractionChange={onMemFractionChange}
              />
            </SidebarSection>
          </>
        )}
      </aside>

      {collapsed && (
        <button
          className="sidebar-expand-btn"
          onClick={() => setCollapsed(false)}
          title="Expand sidebar"
        >
          &raquo;
        </button>
      )}
    </>
  );
}
