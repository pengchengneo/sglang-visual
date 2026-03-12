import { useState } from "react";
import type { PresetManifestEntry, ModelConfig } from "../../types/model";
import type { Plane } from "../controls/PlaneTabBar";
import { ModelSelector } from "../controls/ModelSelector";
import { GpuControls } from "../controls/GpuControls";
import { SidebarSection } from "./SidebarSection";
import { ParallelismControls } from "./ParallelismControls";

interface Props {
  manifest: PresetManifestEntry[];
  selectedPreset: string | null;
  onSelectModel: (id: string) => void;
  manifestLoading: boolean;
  tpSize: number;
  onTpSizeChange: (tp: number) => void;
  modelConfig: ModelConfig | null;
  activePlane: Plane;
  gpuMemoryBytes: number;
  onGpuMemoryChange: (bytes: number) => void;
  memFractionStatic: number;
  onMemFractionChange: (fraction: number) => void;
}

const DEFAULT_SECTIONS = new Set(["model", "parallelism", "gpu"]);

export function Sidebar({
  manifest,
  selectedPreset,
  onSelectModel,
  manifestLoading,
  tpSize,
  onTpSizeChange,
  modelConfig,
  activePlane,
  gpuMemoryBytes,
  onGpuMemoryChange,
  memFractionStatic,
  onMemFractionChange,
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
          />
        </SidebarSection>

        {activePlane === "compute" && (
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
