import { useState } from "react";
import { useLocation } from "react-router-dom";
import { useAppContext } from "../../contexts/AppContext";
import { ModelSelector } from "../controls/ModelSelector";
import { GpuControls } from "../controls/GpuControls";
import { SidebarSection } from "./SidebarSection";
import "./Sidebar.css";
import { ParallelismControls } from "./ParallelismControls";
import { QuantizationControls } from "./QuantizationControls";
import { SchedulingControls } from "./SchedulingControls";
import { ContextControls } from "./ContextControls";
import { SpeculativeControls } from "./SpeculativeControls";
import { CudaGraphControls } from "./CudaGraphControls";

const DEFAULT_SECTIONS = new Set(["model", "parallelism"]);

export function Sidebar() {
  const ctx = useAppContext();
  const { pathname } = useLocation();
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

  const isCompute = pathname === "/compute";
  const isControl = pathname === "/control";
  const isScheduling = pathname === "/scheduling";
  const isKVCache = pathname === "/kv-cache";

  const showScheduling = isControl || isScheduling;
  const showSpeculative = isControl;
  const showCudaGraph = isControl;
  const showContext = isCompute || isKVCache;
  const showGpu = isCompute || isKVCache;

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
          {ctx.manifestLoading ? (
            <div className="loading">Loading models...</div>
          ) : (
            <ModelSelector
              manifest={ctx.manifest}
              selected={ctx.selectedPreset}
              onSelect={(id: string) => {
                ctx.setSelectedPreset(id);
                ctx.setTpSize(1);
              }}
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
            config={ctx.modelConfig}
            tpSize={ctx.tpSize}
            onTpSizeChange={ctx.setTpSize}
            dpSize={ctx.dpSize}
            onDpSizeChange={ctx.setDpSize}
            ppSize={ctx.ppSize}
            onPpSizeChange={ctx.setPpSize}
            epSize={ctx.epSize}
            onEpSizeChange={ctx.setEpSize}
            enableDpAttention={ctx.enableDpAttention}
            onEnableDpAttentionChange={ctx.setEnableDpAttention}
          />
        </SidebarSection>

        <SidebarSection
          title="Quantization"
          open={openSections.has("quantization")}
          onToggle={() => toggle("quantization")}
        >
          <QuantizationControls
            dtype={ctx.dtype}
            onDtypeChange={ctx.setDtype}
            quantization={ctx.quantization}
            onQuantizationChange={ctx.setQuantization}
            kvCacheDtype={ctx.kvCacheDtype}
            onKvCacheDtypeChange={ctx.setKvCacheDtype}
          />
        </SidebarSection>

        {showScheduling && (
          <SidebarSection
            title="Scheduling"
            open={openSections.has("scheduling")}
            onToggle={() => toggle("scheduling")}
          >
            <SchedulingControls
              schedulePolicy={ctx.schedulePolicy}
              onSchedulePolicyChange={ctx.setSchedulePolicy}
              chunkedPrefillSize={ctx.chunkedPrefillSize}
              onChunkedPrefillSizeChange={ctx.setChunkedPrefillSize}
              disableRadixCache={ctx.disableRadixCache}
              onDisableRadixCacheChange={ctx.setDisableRadixCache}
            />
          </SidebarSection>
        )}

        {showSpeculative && (
          <SidebarSection
            title="Speculative Decoding"
            open={openSections.has("speculative")}
            onToggle={() => toggle("speculative")}
          >
            <SpeculativeControls
              specAlgorithm={ctx.specAlgorithm}
              onSpecAlgorithmChange={ctx.setSpecAlgorithm}
              specNumDraftTokens={ctx.specNumDraftTokens}
              onSpecNumDraftTokensChange={ctx.setSpecNumDraftTokens}
            />
          </SidebarSection>
        )}

        {showCudaGraph && (
          <SidebarSection
            title="CUDA Graph"
            open={openSections.has("cudagraph")}
            onToggle={() => toggle("cudagraph")}
          >
            <CudaGraphControls
              cudaGraphMaxBs={ctx.cudaGraphMaxBs}
              onCudaGraphMaxBsChange={ctx.setCudaGraphMaxBs}
              disableCudaGraph={ctx.disableCudaGraph}
              onDisableCudaGraphChange={ctx.setDisableCudaGraph}
            />
          </SidebarSection>
        )}

        {showContext && (
          <SidebarSection
            title="Context"
            open={openSections.has("context")}
            onToggle={() => toggle("context")}
          >
            <ContextControls
              contextLength={ctx.contextLength}
              onContextLengthChange={ctx.setContextLength}
            />
          </SidebarSection>
        )}

        {showGpu && (
          <SidebarSection
            title="GPU"
            open={openSections.has("gpu")}
            onToggle={() => toggle("gpu")}
          >
            <GpuControls
              gpuMemoryBytes={ctx.gpuMemoryBytes}
              onGpuMemoryChange={ctx.setGpuMemoryBytes}
              memFractionStatic={ctx.memFractionStatic}
              onMemFractionChange={ctx.setMemFractionStatic}
            />
          </SidebarSection>
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
