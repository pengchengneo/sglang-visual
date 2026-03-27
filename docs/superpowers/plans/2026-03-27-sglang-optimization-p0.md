# SGLang Optimization Visual Learning Platform — Implementation Plan (P0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add KV Cache (RadixAttention) and Request Scheduling (Continuous Batching) interactive visualization pages to the SGLang visual learning platform, with a new multi-page navigation structure.

**Architecture:** Introduce react-router-dom for URL-based routing. Extract App.tsx's 37-prop state into a React Context provider so all pages can access shared parameters without prop drilling. Each new page is a self-contained route component with its own animation state. Shared animation utilities (useAnimation hook, playback controls) are extracted into reusable components.

**Tech Stack:** React 19, TypeScript 5.9, react-router-dom 7, Vite 7, pure SVG + CSS animations (no new animation libraries)

---

## File Structure

```
src/
  main.tsx                          # MODIFY: wrap App in BrowserRouter
  App.tsx                           # MODIFY: replace state with context, replace conditional render with <Outlet/>
  contexts/
    AppContext.tsx                   # CREATE: all shared state + derived values in one context
  pages/
    ComputePlanePage.tsx            # CREATE: wrapper for existing Compute Plane content
    ControlPlanePage.tsx            # CREATE: wrapper for existing Control Plane content
    KVCachePage.tsx                 # CREATE: P0 - KV Cache page shell
    SchedulingPage.tsx             # CREATE: P0 - Request Scheduling page shell
  components/
    navigation/
      TopNav.tsx                   # CREATE: replaces PlaneTabBar, router-based nav
    shared/
      useAnimation.ts             # CREATE: animation frame hook (play/pause/step/speed)
      AnimationControls.tsx        # CREATE: playback UI (play/pause/step/speed buttons)
      ComparisonToggle.tsx         # CREATE: A/B mode toggle for before/after comparisons
      MetricsPanel.tsx             # CREATE: real-time metrics display bar
    kv-cache/
      RadixTreeEngine.ts          # CREATE: trie data structure + operations (pure logic)
      RadixTreeViz.tsx            # CREATE: SVG tree visualization
      RadixTreeAnimation.ts       # CREATE: animation sequence generator for tree ops
      MemoryPoolViz.tsx           # CREATE: GPU memory block grid
      KVCacheSidebar.tsx          # CREATE: sidebar controls for KV Cache page
      scenarioPresets.ts          # CREATE: preset scenario data
    scheduling/
      SchedulingEngine.ts         # CREATE: batch simulation logic (pure functions)
      ContinuousBatchingViz.tsx   # CREATE: dual-queue animation
      ChunkedPrefillViz.tsx       # CREATE: timeline visualization
      PolicyComparisonViz.tsx     # CREATE: 3-column comparison
      SchedulingSidebar.tsx       # CREATE: sidebar controls for Scheduling page
  components/sidebar/
    Sidebar.tsx                    # MODIFY: consume context instead of props, add new page sections
  components/controls/
    PlaneTabBar.tsx                # DELETE (replaced by TopNav)
```

---

## Task 1: Install react-router-dom and Set Up Routing

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/pages/ComputePlanePage.tsx`
- Create: `frontend/src/pages/ControlPlanePage.tsx`

- [ ] **Step 1: Install react-router-dom**

Run:
```bash
cd frontend && npm install react-router-dom
```

Expected: `react-router-dom` added to `package.json` dependencies.

- [ ] **Step 2: Create ComputePlanePage wrapper**

Create `frontend/src/pages/ComputePlanePage.tsx`:

```tsx
import { useAppContext } from "../contexts/AppContext";
import PipelineView from "../components/pipeline/PipelineView";
import GpuMemoryPanel from "../components/gpu/GpuMemoryPanel";

export default function ComputePlanePage() {
  const {
    model, tpSize, ppSize, epSize, dpSize, enableDpAttention,
    bytesPerParam, kvBytesPerElement, dtype, quantization, kvCacheDtype,
    gpuMemoryBytes, memFractionStatic, contextLength,
  } = useAppContext();

  if (!model) {
    return <div className="empty-state">请先选择一个模型</div>;
  }

  return (
    <div className="split-layout">
      <div className="panel-left">
        <PipelineView
          model={model}
          tpSize={tpSize}
          ppSize={ppSize}
          epSize={epSize}
          bytesPerParam={bytesPerParam}
          dtype={dtype}
          quantization={quantization}
        />
      </div>
      <div className="panel-right">
        <GpuMemoryPanel
          model={model}
          tpSize={tpSize}
          dpSize={dpSize}
          ppSize={ppSize}
          epSize={epSize}
          enableDpAttention={enableDpAttention}
          bytesPerParam={bytesPerParam}
          kvBytesPerElement={kvBytesPerElement}
          dtype={dtype}
          kvCacheDtype={kvCacheDtype}
          gpuMemoryBytes={gpuMemoryBytes}
          memFractionStatic={memFractionStatic}
          contextLength={contextLength}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create ControlPlanePage wrapper**

Create `frontend/src/pages/ControlPlanePage.tsx`:

```tsx
import { useAppContext } from "../contexts/AppContext";
import ControlPlaneView from "../components/controlplane/ControlPlaneView";

export default function ControlPlanePage() {
  const {
    tpSize, dpSize, ppSize, epSize, enableDpAttention, modelConfig,
    schedulePolicy, chunkedPrefillSize, disableRadixCache,
    specAlgorithm, specNumDraftTokens, cudaGraphMaxBs, disableCudaGraph,
  } = useAppContext();

  return (
    <ControlPlaneView
      tpSize={tpSize}
      dpSize={dpSize}
      ppSize={ppSize}
      epSize={epSize}
      enableDpAttention={enableDpAttention}
      modelConfig={modelConfig}
      schedulePolicy={schedulePolicy}
      chunkedPrefillSize={chunkedPrefillSize}
      disableRadixCache={disableRadixCache}
      specAlgorithm={specAlgorithm}
      specNumDraftTokens={specNumDraftTokens}
      cudaGraphMaxBs={cudaGraphMaxBs}
      disableCudaGraph={disableCudaGraph}
    />
  );
}
```

- [ ] **Step 4: Update main.tsx with BrowserRouter**

Modify `frontend/src/main.tsx`:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.tsx";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter basename={import.meta.env.BASE_URL}>
      <App />
    </BrowserRouter>
  </StrictMode>
);
```

- [ ] **Step 5: Verify the app still builds**

Run:
```bash
cd frontend && npm run build
```

Expected: Build succeeds with no errors. (App.tsx will be updated in Task 2 to use routes.)

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/main.tsx frontend/src/pages/ComputePlanePage.tsx frontend/src/pages/ControlPlanePage.tsx
git commit -m "feat: install react-router-dom, create page wrappers for existing views"
```

---

## Task 2: Extract State into AppContext

**Files:**
- Create: `frontend/src/contexts/AppContext.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create AppContext**

Create `frontend/src/contexts/AppContext.tsx`:

```tsx
import { createContext, useContext, useState, useMemo, type ReactNode } from "react";
import { useManifest, useModelData } from "../hooks/useModelData";
import type { ModelArchitecture, ModelConfig } from "../types/model";

export type Dtype = "fp16" | "bf16" | "fp32";
export type Quantization = "none" | "int8" | "int4" | "fp8";
export type KvCacheDtype = "fp16" | "fp8";
export type SchedulePolicy = "fcfs" | "lpm" | "random" | "dfs-weight";
export type SpecAlgorithm = "none" | "eagle" | "eagle3" | "nextn" | "ngram";

interface AppContextValue {
  // Data
  model: ModelArchitecture | null;
  modelLoading: boolean;
  modelError: string | null;
  manifest: ReturnType<typeof useManifest>["manifest"];
  selectedPreset: string | null;
  setSelectedPreset: (id: string | null) => void;

  // Parallelism
  tpSize: number;
  setTpSize: (n: number) => void;
  dpSize: number;
  setDpSize: (n: number) => void;
  ppSize: number;
  setPpSize: (n: number) => void;
  epSize: number;
  setEpSize: (n: number) => void;
  enableDpAttention: boolean;
  setEnableDpAttention: (b: boolean) => void;

  // GPU
  gpuMemoryBytes: number;
  setGpuMemoryBytes: (n: number) => void;
  memFractionStatic: number;
  setMemFractionStatic: (n: number) => void;

  // Quantization
  dtype: Dtype;
  setDtype: (d: Dtype) => void;
  quantization: Quantization;
  setQuantization: (q: Quantization) => void;
  kvCacheDtype: KvCacheDtype;
  setKvCacheDtype: (d: KvCacheDtype) => void;

  // Scheduling
  schedulePolicy: SchedulePolicy;
  setSchedulePolicy: (p: SchedulePolicy) => void;
  chunkedPrefillSize: number;
  setChunkedPrefillSize: (n: number) => void;
  disableRadixCache: boolean;
  setDisableRadixCache: (b: boolean) => void;

  // Context
  contextLength: number;
  setContextLength: (n: number) => void;

  // Speculative
  specAlgorithm: SpecAlgorithm;
  setSpecAlgorithm: (a: SpecAlgorithm) => void;
  specNumDraftTokens: number;
  setSpecNumDraftTokens: (n: number) => void;

  // CUDA Graph
  cudaGraphMaxBs: number;
  setCudaGraphMaxBs: (n: number) => void;
  disableCudaGraph: boolean;
  setDisableCudaGraph: (b: boolean) => void;

  // Derived
  bytesPerParam: number;
  kvBytesPerElement: number;
  modelConfig: ModelConfig | null;
}

const AppContext = createContext<AppContextValue | null>(null);

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
}

export function AppProvider({ children }: { children: ReactNode }) {
  // Data
  const { manifest } = useManifest();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const { model, loading: modelLoading, error: modelError } = useModelData(selectedPreset);

  // Parallelism
  const [tpSize, setTpSize] = useState(1);
  const [dpSize, setDpSize] = useState(1);
  const [ppSize, setPpSize] = useState(1);
  const [epSize, setEpSize] = useState(1);
  const [enableDpAttention, setEnableDpAttention] = useState(false);

  // GPU
  const [gpuMemoryBytes, setGpuMemoryBytes] = useState(80 * 1024 ** 3);
  const [memFractionStatic, setMemFractionStatic] = useState(0.88);

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

  // Speculative
  const [specAlgorithm, setSpecAlgorithm] = useState<SpecAlgorithm>("none");
  const [specNumDraftTokens, setSpecNumDraftTokens] = useState(5);

  // CUDA Graph
  const [cudaGraphMaxBs, setCudaGraphMaxBs] = useState(128);
  const [disableCudaGraph, setDisableCudaGraph] = useState(false);

  // Derived
  const bytesPerParam = useMemo(() => {
    if (quantization === "int8" || quantization === "fp8") return 1;
    if (quantization === "int4") return 0.5;
    if (dtype === "fp32") return 4;
    return 2; // fp16 / bf16
  }, [dtype, quantization]);

  const kvBytesPerElement = useMemo(() => {
    return kvCacheDtype === "fp8" ? 1 : 2;
  }, [kvCacheDtype]);

  const modelConfig = model?.config ?? null;

  const value = useMemo<AppContextValue>(
    () => ({
      model, modelLoading, modelError, manifest, selectedPreset, setSelectedPreset,
      tpSize, setTpSize, dpSize, setDpSize, ppSize, setPpSize, epSize, setEpSize,
      enableDpAttention, setEnableDpAttention,
      gpuMemoryBytes, setGpuMemoryBytes, memFractionStatic, setMemFractionStatic,
      dtype, setDtype, quantization, setQuantization, kvCacheDtype, setKvCacheDtype,
      schedulePolicy, setSchedulePolicy, chunkedPrefillSize, setChunkedPrefillSize,
      disableRadixCache, setDisableRadixCache,
      contextLength, setContextLength,
      specAlgorithm, setSpecAlgorithm, specNumDraftTokens, setSpecNumDraftTokens,
      cudaGraphMaxBs, setCudaGraphMaxBs, disableCudaGraph, setDisableCudaGraph,
      bytesPerParam, kvBytesPerElement, modelConfig,
    }),
    [
      model, modelLoading, modelError, manifest, selectedPreset,
      tpSize, dpSize, ppSize, epSize, enableDpAttention,
      gpuMemoryBytes, memFractionStatic,
      dtype, quantization, kvCacheDtype,
      schedulePolicy, chunkedPrefillSize, disableRadixCache,
      contextLength,
      specAlgorithm, specNumDraftTokens,
      cudaGraphMaxBs, disableCudaGraph,
      bytesPerParam, kvBytesPerElement, modelConfig,
    ]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
```

- [ ] **Step 2: Rewrite App.tsx to use context + routes**

Replace `frontend/src/App.tsx` with:

```tsx
import { Routes, Route, Navigate } from "react-router-dom";
import { AppProvider } from "./contexts/AppContext";
import TopNav from "./components/navigation/TopNav";
import Sidebar from "./components/sidebar/Sidebar";
import ComputePlanePage from "./pages/ComputePlanePage";
import ControlPlanePage from "./pages/ControlPlanePage";
import KVCachePage from "./pages/KVCachePage";
import SchedulingPage from "./pages/SchedulingPage";
import ErrorBoundary from "./components/ErrorBoundary";
import "./App.css";

function useTheme() {
  // (keep existing useTheme hook logic exactly as-is)
}

function AppContent() {
  const { theme, setTheme, isDark } = useTheme();

  return (
    <div className="app">
      <TopNav />
      <div className="app-body">
        <Sidebar />
        <main className="main-content">
          <ErrorBoundary>
            <Routes>
              <Route path="/compute" element={<ComputePlanePage />} />
              <Route path="/control" element={<ControlPlanePage />} />
              <Route path="/kv-cache" element={<KVCachePage />} />
              <Route path="/scheduling" element={<SchedulingPage />} />
              <Route path="/" element={<Navigate to="/compute" replace />} />
            </Routes>
          </ErrorBoundary>
        </main>
      </div>
      <button
        className="theme-toggle"
        onClick={() => setTheme(isDark ? "light" : "dark")}
        title={`Switch to ${isDark ? "light" : "dark"} mode`}
      >
        {isDark ? "☀️" : "🌙"}
      </button>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
```

Note: Keep the exact `useTheme` hook implementation from the current `App.tsx` — it uses `useState` + `useEffect` + `localStorage` with `data-theme` attribute on `documentElement`. Just move it to a local function inside `App.tsx`.

- [ ] **Step 3: Create placeholder pages for KVCache and Scheduling**

Create `frontend/src/pages/KVCachePage.tsx`:

```tsx
export default function KVCachePage() {
  return (
    <div style={{ padding: "24px" }}>
      <h2 style={{ marginBottom: 8 }}>KV Cache Management</h2>
      <p style={{ color: "var(--text-secondary)" }}>
        RadixAttention prefix tree visualization — coming soon
      </p>
    </div>
  );
}
```

Create `frontend/src/pages/SchedulingPage.tsx`:

```tsx
export default function SchedulingPage() {
  return (
    <div style={{ padding: "24px" }}>
      <h2 style={{ marginBottom: 8 }}>Request Scheduling</h2>
      <p style={{ color: "var(--text-secondary)" }}>
        Continuous batching animation — coming soon
      </p>
    </div>
  );
}
```

- [ ] **Step 4: Verify build succeeds**

Run:
```bash
cd frontend && npm run build
```

Expected: No errors. (TopNav and updated Sidebar will be created in next tasks.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/contexts/AppContext.tsx frontend/src/App.tsx frontend/src/pages/KVCachePage.tsx frontend/src/pages/SchedulingPage.tsx
git commit -m "feat: extract state into AppContext, set up route structure with placeholder pages"
```

---

## Task 3: Create TopNav and Update Sidebar

**Files:**
- Create: `frontend/src/components/navigation/TopNav.tsx`
- Create: `frontend/src/components/navigation/TopNav.css`
- Modify: `frontend/src/components/sidebar/Sidebar.tsx`
- Delete: `frontend/src/components/controls/PlaneTabBar.tsx` (after TopNav is working)

- [ ] **Step 1: Create TopNav component**

Create `frontend/src/components/navigation/TopNav.tsx`:

```tsx
import { NavLink } from "react-router-dom";
import "./TopNav.css";

const TABS = [
  { to: "/compute", label: "Compute" },
  { to: "/control", label: "Control" },
  { to: "/kv-cache", label: "KV Cache" },
  { to: "/scheduling", label: "Scheduling" },
] as const;

export default function TopNav() {
  return (
    <nav className="top-nav">
      <div className="top-nav-brand">SGLang Visual</div>
      <div className="top-nav-tabs">
        {TABS.map((tab) => (
          <NavLink
            key={tab.to}
            to={tab.to}
            className={({ isActive }) =>
              `top-nav-tab${isActive ? " active" : ""}`
            }
          >
            {tab.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
```

- [ ] **Step 2: Create TopNav CSS**

Create `frontend/src/components/navigation/TopNav.css`:

```css
.top-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 52px;
  display: flex;
  align-items: center;
  gap: 24px;
  padding: 0 24px;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  z-index: 100;
  backdrop-filter: blur(12px);
}

.top-nav-brand {
  font-weight: 700;
  font-size: 15px;
  color: var(--accent);
  white-space: nowrap;
}

.top-nav-tabs {
  display: flex;
  gap: 4px;
  background: var(--bg-hover);
  border-radius: 8px;
  padding: 3px;
}

.top-nav-tab {
  padding: 6px 14px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-secondary);
  text-decoration: none;
  transition: all 0.15s ease;
  white-space: nowrap;
}

.top-nav-tab:hover {
  color: var(--text);
  background: var(--bg-card);
}

.top-nav-tab.active {
  color: #fff;
  background: var(--accent);
  box-shadow: 0 1px 4px rgba(99, 102, 241, 0.3);
}
```

- [ ] **Step 3: Update Sidebar to use context and support new pages**

Modify `frontend/src/components/sidebar/Sidebar.tsx`:

Replace the props-based approach with context consumption. The sidebar should use `useLocation()` from react-router to determine which sections to show:

```tsx
import { useState } from "react";
import { useLocation } from "react-router-dom";
import { useAppContext } from "../../contexts/AppContext";
import SidebarSection from "./SidebarSection";
import ModelSelector from "../controls/ModelSelector";
import ParallelismControls from "./ParallelismControls";
import QuantizationControls from "./QuantizationControls";
import SchedulingControls from "./SchedulingControls";
import ContextControls from "./ContextControls";
import SpeculativeControls from "./SpeculativeControls";
import CudaGraphControls from "./CudaGraphControls";
// GPU controls
import GpuControls from "../controls/GpuControls";
import "./Sidebar.css";

export default function Sidebar() {
  const ctx = useAppContext();
  const { pathname } = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const [openSections, setOpenSections] = useState<Set<string>>(
    new Set(["model", "parallelism"])
  );

  const toggle = (key: string) =>
    setOpenSections((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });

  // Determine which sections to show based on current route
  const isCompute = pathname === "/compute";
  const isControl = pathname === "/control";
  const isKVCache = pathname === "/kv-cache";
  const isScheduling = pathname === "/scheduling";

  // Model selector is always visible
  // Parallelism and Quantization are visible on compute, control, kv-cache
  const showParallelism = isCompute || isControl || isKVCache;
  const showQuantization = isCompute || isControl;
  const showScheduling = isControl || isScheduling;
  const showSpeculative = isControl;
  const showCudaGraph = isControl;
  const showContext = isCompute || isKVCache;
  const showGpu = isCompute || isKVCache;

  if (collapsed) {
    return (
      <aside className="sidebar collapsed">
        <button className="sidebar-toggle" onClick={() => setCollapsed(false)}>
          ›
        </button>
      </aside>
    );
  }

  return (
    <aside className="sidebar">
      <button className="sidebar-toggle" onClick={() => setCollapsed(true)}>
        ‹
      </button>

      <SidebarSection title="Model" isOpen={openSections.has("model")} onToggle={() => toggle("model")}>
        <ModelSelector
          manifest={ctx.manifest}
          selectedPreset={ctx.selectedPreset}
          onSelect={ctx.setSelectedPreset}
        />
      </SidebarSection>

      {showParallelism && (
        <SidebarSection title="Parallelism" isOpen={openSections.has("parallelism")} onToggle={() => toggle("parallelism")}>
          <ParallelismControls
            tpSize={ctx.tpSize} setTpSize={ctx.setTpSize}
            dpSize={ctx.dpSize} setDpSize={ctx.setDpSize}
            ppSize={ctx.ppSize} setPpSize={ctx.setPpSize}
            epSize={ctx.epSize} setEpSize={ctx.setEpSize}
            enableDpAttention={ctx.enableDpAttention}
            setEnableDpAttention={ctx.setEnableDpAttention}
            modelConfig={ctx.modelConfig}
          />
        </SidebarSection>
      )}

      {showQuantization && (
        <SidebarSection title="Quantization" isOpen={openSections.has("quantization")} onToggle={() => toggle("quantization")}>
          <QuantizationControls
            dtype={ctx.dtype} setDtype={ctx.setDtype}
            quantization={ctx.quantization} setQuantization={ctx.setQuantization}
            kvCacheDtype={ctx.kvCacheDtype} setKvCacheDtype={ctx.setKvCacheDtype}
          />
        </SidebarSection>
      )}

      {showScheduling && (
        <SidebarSection title="Scheduling" isOpen={openSections.has("scheduling")} onToggle={() => toggle("scheduling")}>
          <SchedulingControls
            schedulePolicy={ctx.schedulePolicy} setSchedulePolicy={ctx.setSchedulePolicy}
            chunkedPrefillSize={ctx.chunkedPrefillSize} setChunkedPrefillSize={ctx.setChunkedPrefillSize}
            disableRadixCache={ctx.disableRadixCache} setDisableRadixCache={ctx.setDisableRadixCache}
          />
        </SidebarSection>
      )}

      {showSpeculative && (
        <SidebarSection title="Speculative Decoding" isOpen={openSections.has("speculative")} onToggle={() => toggle("speculative")}>
          <SpeculativeControls
            specAlgorithm={ctx.specAlgorithm} setSpecAlgorithm={ctx.setSpecAlgorithm}
            specNumDraftTokens={ctx.specNumDraftTokens} setSpecNumDraftTokens={ctx.setSpecNumDraftTokens}
          />
        </SidebarSection>
      )}

      {showCudaGraph && (
        <SidebarSection title="CUDA Graph" isOpen={openSections.has("cudagraph")} onToggle={() => toggle("cudagraph")}>
          <CudaGraphControls
            cudaGraphMaxBs={ctx.cudaGraphMaxBs} setCudaGraphMaxBs={ctx.setCudaGraphMaxBs}
            disableCudaGraph={ctx.disableCudaGraph} setDisableCudaGraph={ctx.setDisableCudaGraph}
          />
        </SidebarSection>
      )}

      {showContext && (
        <SidebarSection title="Context" isOpen={openSections.has("context")} onToggle={() => toggle("context")}>
          <ContextControls
            contextLength={ctx.contextLength} setContextLength={ctx.setContextLength}
          />
        </SidebarSection>
      )}

      {showGpu && (
        <SidebarSection title="GPU" isOpen={openSections.has("gpu")} onToggle={() => toggle("gpu")}>
          <GpuControls
            gpuMemoryBytes={ctx.gpuMemoryBytes} setGpuMemoryBytes={ctx.setGpuMemoryBytes}
            memFractionStatic={ctx.memFractionStatic} setMemFractionStatic={ctx.setMemFractionStatic}
          />
        </SidebarSection>
      )}
    </aside>
  );
}
```

Note: The sub-components (`ParallelismControls`, `QuantizationControls`, etc.) still accept props — they don't need to change. Only the `Sidebar` parent switches from receiving props to using context.

- [ ] **Step 4: Update App.css for new nav height**

In `frontend/src/App.css`, change `.app` padding-top from `60px` to `52px` to match the new nav height, and remove any `.tab-bar` floating styles if present (those move to TopNav.css).

- [ ] **Step 5: Remove PlaneTabBar import from App.tsx**

Ensure `App.tsx` no longer imports `PlaneTabBar`. The `PlaneTabBar.tsx` file can be deleted. Also remove the `Plane` type export from it — it's no longer needed since routing replaces plane state.

- [ ] **Step 6: Verify build + dev server**

Run:
```bash
cd frontend && npm run build && npm run dev
```

Expected: Build succeeds. Dev server shows the new top nav with 4 tabs. Clicking tabs navigates between routes. Compute and Control pages render identically to before. KV Cache and Scheduling show placeholder content. Sidebar dynamically shows/hides sections based on current route.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: replace PlaneTabBar with TopNav router navigation, refactor Sidebar to use context"
```

---

## Task 4: Create Shared Animation Utilities

**Files:**
- Create: `frontend/src/components/shared/useAnimation.ts`
- Create: `frontend/src/components/shared/AnimationControls.tsx`
- Create: `frontend/src/components/shared/AnimationControls.css`
- Create: `frontend/src/components/shared/ComparisonToggle.tsx`
- Create: `frontend/src/components/shared/MetricsPanel.tsx`
- Create: `frontend/src/components/shared/shared.css`

- [ ] **Step 1: Create useAnimation hook**

Create `frontend/src/components/shared/useAnimation.ts`:

```ts
import { useState, useRef, useCallback, useEffect } from "react";

export interface AnimationState {
  /** Current frame index (0-based) */
  frame: number;
  /** Whether animation is playing */
  playing: boolean;
  /** Playback speed multiplier (0.5, 1, 2, 4) */
  speed: number;
  /** Total number of frames */
  totalFrames: number;
}

export interface AnimationControls {
  play: () => void;
  pause: () => void;
  step: () => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
  setFrame: (frame: number) => void;
}

/**
 * Hook for managing frame-based animations.
 * @param totalFrames - Total number of animation frames
 * @param msPerFrame - Milliseconds per frame at 1x speed (default 500ms)
 * @param onFrame - Optional callback invoked with each new frame index
 */
export function useAnimation(
  totalFrames: number,
  msPerFrame = 500,
  onFrame?: (frame: number) => void
): [AnimationState, AnimationControls] {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const rafRef = useRef<number>(0);
  const lastTimeRef = useRef<number>(0);

  const frameRef = useRef(frame);
  frameRef.current = frame;

  useEffect(() => {
    if (!playing) return;

    const interval = msPerFrame / speed;
    let lastTime = performance.now();

    const tick = (now: number) => {
      const delta = now - lastTime;
      if (delta >= interval) {
        lastTime = now - (delta % interval);
        const nextFrame = frameRef.current + 1;
        if (nextFrame >= totalFrames) {
          setPlaying(false);
          return;
        }
        setFrame(nextFrame);
        onFrame?.(nextFrame);
      }
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, speed, totalFrames, msPerFrame, onFrame]);

  const controls: AnimationControls = {
    play: useCallback(() => setPlaying(true), []),
    pause: useCallback(() => setPlaying(false), []),
    step: useCallback(() => {
      setPlaying(false);
      setFrame((f) => {
        const next = Math.min(f + 1, totalFrames - 1);
        onFrame?.(next);
        return next;
      });
    }, [totalFrames, onFrame]),
    reset: useCallback(() => {
      setPlaying(false);
      setFrame(0);
      onFrame?.(0);
    }, [onFrame]),
    setSpeed: useCallback((s: number) => setSpeed(s), []),
    setFrame: useCallback(
      (f: number) => {
        setFrame(Math.max(0, Math.min(f, totalFrames - 1)));
        onFrame?.(f);
      },
      [totalFrames, onFrame]
    ),
  };

  return [{ frame, playing, speed, totalFrames }, controls];
}
```

- [ ] **Step 2: Create AnimationControls UI component**

Create `frontend/src/components/shared/AnimationControls.tsx`:

```tsx
import type { AnimationState, AnimationControls as Controls } from "./useAnimation";
import "./AnimationControls.css";

interface Props {
  state: AnimationState;
  controls: Controls;
  label?: string;
}

const SPEEDS = [0.5, 1, 2, 4];

export default function AnimationControls({ state, controls, label }: Props) {
  return (
    <div className="anim-controls">
      {label && <span className="anim-label">{label}</span>}
      <div className="anim-buttons">
        <button className="anim-btn" onClick={controls.reset} title="Reset">
          ⏮
        </button>
        {state.playing ? (
          <button className="anim-btn anim-btn-primary" onClick={controls.pause} title="Pause">
            ⏸
          </button>
        ) : (
          <button className="anim-btn anim-btn-primary" onClick={controls.play} title="Play">
            ▶
          </button>
        )}
        <button className="anim-btn" onClick={controls.step} title="Step">
          ⏭
        </button>
      </div>
      <div className="anim-speed">
        {SPEEDS.map((s) => (
          <button
            key={s}
            className={`anim-speed-btn${state.speed === s ? " active" : ""}`}
            onClick={() => controls.setSpeed(s)}
          >
            {s}x
          </button>
        ))}
      </div>
      <div className="anim-progress">
        <input
          type="range"
          min={0}
          max={state.totalFrames - 1}
          value={state.frame}
          onChange={(e) => controls.setFrame(Number(e.target.value))}
          className="anim-slider"
        />
        <span className="anim-frame-label">
          {state.frame + 1} / {state.totalFrames}
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create AnimationControls CSS**

Create `frontend/src/components/shared/AnimationControls.css`:

```css
.anim-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.anim-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  white-space: nowrap;
}

.anim-buttons {
  display: flex;
  gap: 4px;
}

.anim-btn {
  width: 32px;
  height: 32px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
}

.anim-btn:hover {
  background: var(--bg-hover);
}

.anim-btn-primary {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}

.anim-btn-primary:hover {
  opacity: 0.9;
}

.anim-speed {
  display: flex;
  gap: 2px;
  background: var(--bg-hover);
  border-radius: 6px;
  padding: 2px;
}

.anim-speed-btn {
  padding: 4px 8px;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--text-secondary);
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}

.anim-speed-btn.active {
  background: var(--accent);
  color: #fff;
}

.anim-progress {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  min-width: 0;
}

.anim-slider {
  flex: 1;
  height: 4px;
  -webkit-appearance: none;
  appearance: none;
  background: var(--border);
  border-radius: 2px;
  outline: none;
}

.anim-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
}

.anim-frame-label {
  font-size: 11px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
```

- [ ] **Step 4: Create ComparisonToggle component**

Create `frontend/src/components/shared/ComparisonToggle.tsx`:

```tsx
import "./shared.css";

interface Props {
  labelA: string;
  labelB: string;
  active: "a" | "b";
  onChange: (mode: "a" | "b") => void;
}

export default function ComparisonToggle({ labelA, labelB, active, onChange }: Props) {
  return (
    <div className="comparison-toggle">
      <button
        className={`comparison-btn${active === "a" ? " active" : ""}`}
        onClick={() => onChange("a")}
      >
        {labelA}
      </button>
      <button
        className={`comparison-btn${active === "b" ? " active" : ""}`}
        onClick={() => onChange("b")}
      >
        {labelB}
      </button>
    </div>
  );
}
```

- [ ] **Step 5: Create MetricsPanel component**

Create `frontend/src/components/shared/MetricsPanel.tsx`:

```tsx
import "./shared.css";

export interface Metric {
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
}

interface Props {
  metrics: Metric[];
}

export default function MetricsPanel({ metrics }: Props) {
  return (
    <div className="metrics-panel">
      {metrics.map((m) => (
        <div key={m.label} className="metric-item">
          <span className="metric-value" style={m.color ? { color: m.color } : undefined}>
            {m.value}
            {m.unit && <span className="metric-unit">{m.unit}</span>}
          </span>
          <span className="metric-label">{m.label}</span>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 6: Create shared.css**

Create `frontend/src/components/shared/shared.css`:

```css
/* ComparisonToggle */
.comparison-toggle {
  display: inline-flex;
  background: var(--bg-hover);
  border-radius: 8px;
  padding: 3px;
}

.comparison-btn {
  padding: 6px 16px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}

.comparison-btn.active {
  background: var(--accent);
  color: #fff;
  box-shadow: 0 1px 4px rgba(99, 102, 241, 0.3);
}

/* MetricsPanel */
.metrics-panel {
  display: flex;
  gap: 24px;
  padding: 12px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.metric-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.metric-value {
  font-size: 18px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--text);
}

.metric-unit {
  font-size: 12px;
  font-weight: 400;
  color: var(--text-secondary);
  margin-left: 2px;
}

.metric-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
```

- [ ] **Step 7: Verify build**

Run:
```bash
cd frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/shared/
git commit -m "feat: add shared animation utilities (useAnimation hook, AnimationControls, ComparisonToggle, MetricsPanel)"
```

---

## Task 5: KV Cache — RadixTree Engine (Pure Logic)

**Files:**
- Create: `frontend/src/components/kv-cache/RadixTreeEngine.ts`

- [ ] **Step 1: Create RadixTreeEngine**

Create `frontend/src/components/kv-cache/RadixTreeEngine.ts`:

```ts
/**
 * Pure data structure and operations for a Radix Tree (trie)
 * used to visualize SGLang's RadixAttention prefix caching.
 *
 * This is a simulation — no actual KV cache data, just the tree structure
 * and reference counting to illustrate the algorithm.
 */

export interface RadixNode {
  id: string;
  /** Token sequence stored at this node */
  tokens: string[];
  /** Number of active requests referencing this node and its ancestors */
  refCount: number;
  /** Child nodes keyed by first token of child's sequence */
  children: Map<string, RadixNode>;
  /** Parent node (null for root) */
  parent: RadixNode | null;
  /** Visual state for animation */
  state: "active" | "cached" | "evictable" | "inserting" | "evicting" | "matching";
}

export interface RadixTree {
  root: RadixNode;
  /** Total number of blocks (nodes) in the tree */
  blockCount: number;
  /** Maximum blocks allowed */
  maxBlocks: number;
}

let nodeIdCounter = 0;

export function createNode(tokens: string[], parent: RadixNode | null): RadixNode {
  return {
    id: `node-${nodeIdCounter++}`,
    tokens,
    refCount: 0,
    children: new Map(),
    parent,
    state: "cached",
  };
}

export function createTree(maxBlocks: number): RadixTree {
  nodeIdCounter = 0;
  return {
    root: createNode(["<root>"], null),
    blockCount: 1,
    maxBlocks,
  };
}

/**
 * Find the longest matching prefix in the tree for a given token sequence.
 * Returns the path of matched nodes and the index where matching stopped.
 */
export function findPrefix(
  tree: RadixTree,
  tokens: string[]
): { matchedNodes: RadixNode[]; matchedLength: number } {
  const matchedNodes: RadixNode[] = [tree.root];
  let pos = 0;

  let current = tree.root;
  while (pos < tokens.length) {
    const nextToken = tokens[pos];
    const child = current.children.get(nextToken);
    if (!child) break;

    // Check how many tokens in this child's sequence match
    let i = 0;
    while (i < child.tokens.length && pos + i < tokens.length && child.tokens[i] === tokens[pos + i]) {
      i++;
    }

    if (i === child.tokens.length) {
      // Full match of this node
      matchedNodes.push(child);
      pos += child.tokens.length;
      current = child;
    } else {
      // Partial match — for simplicity, we treat this as no match on this child
      break;
    }
  }

  return { matchedNodes, matchedLength: pos };
}

/**
 * Insert a token sequence into the tree.
 * Returns the leaf node of the inserted path.
 */
export function insertSequence(tree: RadixTree, tokens: string[]): RadixNode {
  const { matchedNodes, matchedLength } = findPrefix(tree, tokens);
  let current = matchedNodes[matchedNodes.length - 1];

  // Insert remaining tokens as new nodes (one node per "chunk" for visual clarity)
  let pos = matchedLength;
  while (pos < tokens.length) {
    // Create a new node with a chunk of tokens (up to 4 per node for readability)
    const chunkSize = Math.min(4, tokens.length - pos);
    const chunk = tokens.slice(pos, pos + chunkSize);
    const newNode = createNode(chunk, current);
    newNode.state = "inserting";
    current.children.set(chunk[0], newNode);
    tree.blockCount++;
    current = newNode;
    pos += chunkSize;
  }

  // Increment refCount along the full path
  for (const node of matchedNodes) {
    node.refCount++;
    node.state = "active";
  }
  // Also increment for newly created nodes
  let n: RadixNode | null = current;
  while (n && !matchedNodes.includes(n)) {
    n.refCount++;
    n.state = "active";
    n = n.parent;
  }

  return current;
}

/**
 * Release a reference to a sequence (request completed).
 * Decrements refCount along the path. Nodes with refCount 0 become evictable.
 */
export function releaseSequence(tree: RadixTree, leafNode: RadixNode): void {
  let node: RadixNode | null = leafNode;
  while (node) {
    node.refCount = Math.max(0, node.refCount - 1);
    node.state = node.refCount > 0 ? "active" : "evictable";
    node = node.parent;
  }
}

/**
 * Evict nodes to free space. Uses LRU-like strategy:
 * evict leaf nodes with refCount 0, deepest first.
 * Returns the evicted nodes (for animation).
 */
export function evictNodes(tree: RadixTree, count: number): RadixNode[] {
  const evicted: RadixNode[] = [];

  function findEvictableLeaves(node: RadixNode): RadixNode[] {
    if (node.children.size === 0 && node.refCount === 0 && node !== tree.root) {
      return [node];
    }
    const leaves: RadixNode[] = [];
    for (const child of node.children.values()) {
      leaves.push(...findEvictableLeaves(child));
    }
    return leaves;
  }

  for (let i = 0; i < count; i++) {
    const leaves = findEvictableLeaves(tree.root);
    if (leaves.length === 0) break;
    const victim = leaves[0];
    victim.state = "evicting";
    evicted.push(victim);

    // Remove from parent
    if (victim.parent) {
      for (const [key, child] of victim.parent.children) {
        if (child === victim) {
          victim.parent.children.delete(key);
          break;
        }
      }
    }
    tree.blockCount--;
  }

  return evicted;
}

/**
 * Flatten tree into a list for rendering.
 * Returns nodes with their depth and position info.
 */
export interface FlatNode {
  node: RadixNode;
  depth: number;
  parentId: string | null;
  childIndex: number;
  totalSiblings: number;
}

export function flattenTree(tree: RadixTree): FlatNode[] {
  const result: FlatNode[] = [];

  function walk(node: RadixNode, depth: number, childIndex: number, totalSiblings: number) {
    result.push({
      node,
      depth,
      parentId: node.parent?.id ?? null,
      childIndex,
      totalSiblings,
    });
    const children = Array.from(node.children.values());
    children.forEach((child, i) => walk(child, depth + 1, i, children.length));
  }

  walk(tree.root, 0, 0, 1);
  return result;
}

/**
 * Count how many blocks would be saved (cache hits) vs allocated (misses)
 * for a given request against the current tree.
 */
export function analyzeRequest(
  tree: RadixTree,
  tokens: string[]
): { hits: number; misses: number; hitRatio: number } {
  const { matchedLength } = findPrefix(tree, tokens);
  const hits = matchedLength;
  const misses = tokens.length - matchedLength;
  const hitRatio = tokens.length > 0 ? hits / tokens.length : 0;
  return { hits, misses, hitRatio };
}
```

- [ ] **Step 2: Verify build**

Run:
```bash
cd frontend && npm run build
```

Expected: Build succeeds (file is importable, types are correct).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/kv-cache/RadixTreeEngine.ts
git commit -m "feat: add RadixTree engine - trie data structure for RadixAttention visualization"
```

---

## Task 6: KV Cache — Scenario Presets

**Files:**
- Create: `frontend/src/components/kv-cache/scenarioPresets.ts`

- [ ] **Step 1: Create scenario presets**

Create `frontend/src/components/kv-cache/scenarioPresets.ts`:

```ts
export interface Scenario {
  id: string;
  name: string;
  description: string;
  /** Requests as token sequences. Each string[] is one request's tokens. */
  requests: {
    tokens: string[];
    /** Delay in frames before this request arrives */
    arrivalFrame: number;
    /** Duration in frames before this request completes */
    durationFrames: number;
  }[];
  maxBlocks: number;
}

const SYSTEM_PROMPT = ["[SYS]", "You", "are", "a", "helpful", "assistant", "."];
const FEWSHOT_PREFIX = ["[SYS]", "Translate", "EN", "to", "CN", ":"];
const EXAMPLE_1 = ["Ex1:", "Hello", "→", "你好"];
const EXAMPLE_2 = ["Ex2:", "Thanks", "→", "谢谢"];

export const SCENARIOS: Scenario[] = [
  {
    id: "multi-turn",
    name: "Multi-Turn Dialogue",
    description: "Multiple conversations sharing the same system prompt — demonstrates system prompt prefix reuse",
    requests: [
      {
        tokens: [...SYSTEM_PROMPT, "User:", "What", "is", "AI", "?"],
        arrivalFrame: 0,
        durationFrames: 20,
      },
      {
        tokens: [...SYSTEM_PROMPT, "User:", "Tell", "me", "a", "joke"],
        arrivalFrame: 5,
        durationFrames: 15,
      },
      {
        tokens: [...SYSTEM_PROMPT, "User:", "What", "is", "AI", "?", "Bot:", "AI", "is", "...", "User:", "More", "details"],
        arrivalFrame: 10,
        durationFrames: 25,
      },
    ],
    maxBlocks: 20,
  },
  {
    id: "few-shot",
    name: "Few-Shot Sharing",
    description: "Multiple requests sharing the same few-shot examples — demonstrates example prefix reuse",
    requests: [
      {
        tokens: [...FEWSHOT_PREFIX, ...EXAMPLE_1, ...EXAMPLE_2, "Input:", "Good", "morning"],
        arrivalFrame: 0,
        durationFrames: 18,
      },
      {
        tokens: [...FEWSHOT_PREFIX, ...EXAMPLE_1, ...EXAMPLE_2, "Input:", "Goodbye"],
        arrivalFrame: 3,
        durationFrames: 15,
      },
      {
        tokens: [...FEWSHOT_PREFIX, ...EXAMPLE_1, "Input:", "Nice", "weather"],
        arrivalFrame: 8,
        durationFrames: 20,
      },
    ],
    maxBlocks: 24,
  },
  {
    id: "no-sharing",
    name: "No Sharing",
    description: "Completely independent requests — no prefix sharing, no cache hits",
    requests: [
      {
        tokens: ["Summarize", "the", "article", "about", "climate"],
        arrivalFrame: 0,
        durationFrames: 15,
      },
      {
        tokens: ["Write", "Python", "code", "for", "sorting"],
        arrivalFrame: 4,
        durationFrames: 12,
      },
      {
        tokens: ["Explain", "quantum", "computing", "basics"],
        arrivalFrame: 8,
        durationFrames: 18,
      },
    ],
    maxBlocks: 20,
  },
  {
    id: "mixed",
    name: "Mixed Traffic",
    description: "Some requests share prefix, others don't — realistic production traffic pattern",
    requests: [
      {
        tokens: [...SYSTEM_PROMPT, "User:", "Hello"],
        arrivalFrame: 0,
        durationFrames: 10,
      },
      {
        tokens: ["Code:", "def", "fib", "(", "n", ")", ":"],
        arrivalFrame: 3,
        durationFrames: 14,
      },
      {
        tokens: [...SYSTEM_PROMPT, "User:", "Explain", "SGLang"],
        arrivalFrame: 6,
        durationFrames: 20,
      },
      {
        tokens: ["Translate:", "Bonjour", "le", "monde"],
        arrivalFrame: 12,
        durationFrames: 12,
      },
      {
        tokens: [...SYSTEM_PROMPT, "User:", "Hello", "again"],
        arrivalFrame: 15,
        durationFrames: 16,
      },
    ],
    maxBlocks: 24,
  },
];
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/kv-cache/scenarioPresets.ts
git commit -m "feat: add KV Cache scenario presets (multi-turn, few-shot, no-sharing, mixed)"
```

---

## Task 7: KV Cache — RadixTree SVG Visualization

**Files:**
- Create: `frontend/src/components/kv-cache/RadixTreeViz.tsx`
- Create: `frontend/src/components/kv-cache/RadixTreeViz.css`

- [ ] **Step 1: Create RadixTreeViz**

Create `frontend/src/components/kv-cache/RadixTreeViz.tsx`:

```tsx
import { useMemo } from "react";
import type { RadixTree, FlatNode } from "./RadixTreeEngine";
import { flattenTree } from "./RadixTreeEngine";
import "./RadixTreeViz.css";

interface Props {
  tree: RadixTree;
  /** Optional: node IDs to highlight (for animation step) */
  highlightIds?: Set<string>;
}

const NODE_W = 120;
const NODE_H = 48;
const H_GAP = 20;
const V_GAP = 60;

interface LayoutNode extends FlatNode {
  x: number;
  y: number;
}

function layoutTree(flatNodes: FlatNode[]): LayoutNode[] {
  // Assign y based on depth
  // Assign x using a simple left-to-right pass per depth level
  const byDepth = new Map<number, FlatNode[]>();
  for (const fn of flatNodes) {
    const list = byDepth.get(fn.depth) ?? [];
    list.push(fn);
    byDepth.set(fn.depth, list);
  }

  const positions = new Map<string, { x: number; y: number }>();

  for (const [depth, nodes] of byDepth) {
    const totalWidth = nodes.length * NODE_W + (nodes.length - 1) * H_GAP;
    const startX = -totalWidth / 2;
    nodes.forEach((fn, i) => {
      positions.set(fn.node.id, {
        x: startX + i * (NODE_W + H_GAP) + NODE_W / 2,
        y: depth * (NODE_H + V_GAP),
      });
    });
  }

  return flatNodes.map((fn) => ({
    ...fn,
    ...positions.get(fn.node.id)!,
  }));
}

const STATE_COLORS: Record<string, string> = {
  active: "var(--green)",
  cached: "var(--accent)",
  evictable: "var(--text-secondary)",
  inserting: "var(--teal)",
  evicting: "var(--red)",
  matching: "var(--yellow)",
};

export default function RadixTreeViz({ tree, highlightIds }: Props) {
  const flatNodes = useMemo(() => flattenTree(tree), [tree]);
  const layout = useMemo(() => layoutTree(flatNodes), [flatNodes]);

  // Compute SVG viewBox
  const xs = layout.map((n) => n.x);
  const ys = layout.map((n) => n.y);
  const minX = Math.min(...xs) - NODE_W / 2 - 20;
  const maxX = Math.max(...xs) + NODE_W / 2 + 20;
  const maxY = Math.max(...ys) + NODE_H + 20;
  const viewBox = `${minX} -20 ${maxX - minX} ${maxY + 40}`;

  // Build position lookup for edges
  const posMap = new Map(layout.map((n) => [n.node.id, n]));

  return (
    <svg className="radix-tree-svg" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
      {/* Edges */}
      {layout
        .filter((n) => n.parentId)
        .map((n) => {
          const parent = posMap.get(n.parentId!);
          if (!parent) return null;
          return (
            <line
              key={`edge-${n.node.id}`}
              x1={parent.x}
              y1={parent.y + NODE_H / 2}
              x2={n.x}
              y2={n.y - NODE_H / 2}
              className="radix-edge"
              stroke={STATE_COLORS[n.node.state]}
              strokeOpacity={0.4}
            />
          );
        })}

      {/* Nodes */}
      {layout.map((n) => {
        const isHighlighted = highlightIds?.has(n.node.id);
        const color = STATE_COLORS[n.node.state];
        const tokenLabel =
          n.node.tokens.length <= 3
            ? n.node.tokens.join(" ")
            : n.node.tokens.slice(0, 3).join(" ") + "…";

        return (
          <g
            key={n.node.id}
            transform={`translate(${n.x - NODE_W / 2}, ${n.y - NODE_H / 2})`}
            className={`radix-node${isHighlighted ? " highlighted" : ""}`}
          >
            <rect
              width={NODE_W}
              height={NODE_H}
              rx={8}
              fill={color}
              fillOpacity={0.15}
              stroke={color}
              strokeWidth={isHighlighted ? 2.5 : 1.5}
            />
            <text
              x={NODE_W / 2}
              y={18}
              textAnchor="middle"
              className="radix-node-label"
              fill={color}
            >
              {tokenLabel}
            </text>
            <text
              x={NODE_W / 2}
              y={36}
              textAnchor="middle"
              className="radix-node-ref"
              fill="var(--text-secondary)"
            >
              ref: {n.node.refCount}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
```

- [ ] **Step 2: Create RadixTreeViz CSS**

Create `frontend/src/components/kv-cache/RadixTreeViz.css`:

```css
.radix-tree-svg {
  width: 100%;
  height: 100%;
  min-height: 300px;
}

.radix-edge {
  stroke-width: 2;
  transition: stroke 0.3s, stroke-opacity 0.3s;
}

.radix-node rect {
  transition: fill-opacity 0.3s, stroke 0.3s, stroke-width 0.2s;
}

.radix-node.highlighted rect {
  fill-opacity: 0.3;
  filter: drop-shadow(0 0 6px currentColor);
}

.radix-node-label {
  font-size: 11px;
  font-weight: 600;
  font-family: "SF Mono", "Fira Code", monospace;
}

.radix-node-ref {
  font-size: 10px;
}
```

- [ ] **Step 3: Verify build**

Run:
```bash
cd frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/kv-cache/RadixTreeViz.tsx frontend/src/components/kv-cache/RadixTreeViz.css
git commit -m "feat: add RadixTree SVG visualization component with color-coded node states"
```

---

## Task 8: KV Cache — Memory Pool Visualization

**Files:**
- Create: `frontend/src/components/kv-cache/MemoryPoolViz.tsx`
- Create: `frontend/src/components/kv-cache/MemoryPoolViz.css`

- [ ] **Step 1: Create MemoryPoolViz**

Create `frontend/src/components/kv-cache/MemoryPoolViz.tsx`:

```tsx
import type { RadixTree, RadixNode } from "./RadixTreeEngine";
import { flattenTree } from "./RadixTreeEngine";
import { useMemo } from "react";
import "./MemoryPoolViz.css";

interface Props {
  tree: RadixTree;
  hoveredNodeId?: string | null;
  onHoverNode?: (id: string | null) => void;
}

const STATE_COLORS: Record<string, string> = {
  active: "#22c55e",
  cached: "#6366f1",
  evictable: "#9ca3af",
  inserting: "#14b8a6",
  evicting: "#ef4444",
  matching: "#eab308",
};

export default function MemoryPoolViz({ tree, hoveredNodeId, onHoverNode }: Props) {
  const blocks = useMemo(() => {
    const flat = flattenTree(tree);
    // Skip root node for display
    return flat.filter((fn) => fn.depth > 0);
  }, [tree]);

  const freeBlocks = tree.maxBlocks - tree.blockCount;
  const usagePercent = ((tree.blockCount / tree.maxBlocks) * 100).toFixed(0);

  return (
    <div className="memory-pool">
      <div className="memory-pool-header">
        <span className="memory-pool-title">KV Cache Memory Pool</span>
        <span className="memory-pool-usage">
          {tree.blockCount} / {tree.maxBlocks} blocks ({usagePercent}%)
        </span>
      </div>
      <div className="memory-pool-grid">
        {blocks.map((b) => (
          <div
            key={b.node.id}
            className={`memory-block${hoveredNodeId === b.node.id ? " hovered" : ""}`}
            style={{
              backgroundColor: STATE_COLORS[b.node.state],
              opacity: hoveredNodeId && hoveredNodeId !== b.node.id ? 0.3 : 1,
            }}
            onMouseEnter={() => onHoverNode?.(b.node.id)}
            onMouseLeave={() => onHoverNode?.(null)}
            title={`${b.node.tokens.join(" ")} | ref: ${b.node.refCount} | ${b.node.state}`}
          />
        ))}
        {/* Free blocks */}
        {Array.from({ length: freeBlocks }).map((_, i) => (
          <div key={`free-${i}`} className="memory-block free" />
        ))}
      </div>
      <div className="memory-pool-legend">
        <span className="legend-item"><span className="legend-dot" style={{ background: "#22c55e" }} /> Active</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: "#6366f1" }} /> Cached</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: "#9ca3af" }} /> Evictable</span>
        <span className="legend-item"><span className="legend-dot free-dot" /> Free</span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create MemoryPoolViz CSS**

Create `frontend/src/components/kv-cache/MemoryPoolViz.css`:

```css
.memory-pool {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}

.memory-pool-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.memory-pool-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}

.memory-pool-usage {
  font-size: 12px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.memory-pool-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, 28px);
  gap: 4px;
  margin-bottom: 12px;
}

.memory-block {
  width: 28px;
  height: 28px;
  border-radius: 4px;
  transition: opacity 0.2s, transform 0.2s;
  cursor: pointer;
}

.memory-block.hovered {
  transform: scale(1.2);
  box-shadow: 0 0 8px rgba(255, 255, 255, 0.3);
  z-index: 1;
}

.memory-block.free {
  background: var(--bg-hover);
  border: 1px dashed var(--border);
}

.memory-pool-legend {
  display: flex;
  gap: 16px;
  font-size: 11px;
  color: var(--text-secondary);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 3px;
}

.legend-dot.free-dot {
  background: var(--bg-hover);
  border: 1px dashed var(--border);
}
```

- [ ] **Step 3: Verify build and commit**

Run:
```bash
cd frontend && npm run build
git add frontend/src/components/kv-cache/MemoryPoolViz.tsx frontend/src/components/kv-cache/MemoryPoolViz.css
git commit -m "feat: add KV Cache memory pool grid visualization"
```

---

## Task 9: KV Cache — Page Assembly

**Files:**
- Create: `frontend/src/components/kv-cache/KVCacheSidebar.tsx`
- Modify: `frontend/src/pages/KVCachePage.tsx`
- Create: `frontend/src/pages/KVCachePage.css`
- Modify: `frontend/src/components/sidebar/Sidebar.tsx` (add KV Cache sidebar section)

- [ ] **Step 1: Create KVCacheSidebar controls**

Create `frontend/src/components/kv-cache/KVCacheSidebar.tsx`:

```tsx
import type { Scenario } from "./scenarioPresets";
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
      <label className="control-label">Scenario</label>
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

      <label className="control-label" style={{ marginTop: 12 }}>
        Cache Capacity: {maxBlocks} blocks
      </label>
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
```

- [ ] **Step 2: Implement full KVCachePage**

Replace `frontend/src/pages/KVCachePage.tsx`:

```tsx
import { useState, useMemo, useCallback, useRef } from "react";
import RadixTreeViz from "../components/kv-cache/RadixTreeViz";
import MemoryPoolViz from "../components/kv-cache/MemoryPoolViz";
import KVCacheSidebar from "../components/kv-cache/KVCacheSidebar";
import AnimationControls from "../components/shared/AnimationControls";
import MetricsPanel from "../components/shared/MetricsPanel";
import { useAnimation } from "../components/shared/useAnimation";
import {
  createTree,
  insertSequence,
  releaseSequence,
  evictNodes,
  findPrefix,
  type RadixTree,
  type RadixNode,
} from "../components/kv-cache/RadixTreeEngine";
import { SCENARIOS } from "../components/kv-cache/scenarioPresets";
import "./KVCachePage.css";

interface FrameState {
  tree: RadixTree;
  highlightIds: Set<string>;
  hits: number;
  misses: number;
  evictions: number;
  message: string;
}

function generateFrames(scenarioId: string, maxBlocks: number): FrameState[] {
  const scenario = SCENARIOS.find((s) => s.id === scenarioId);
  if (!scenario) return [];

  const frames: FrameState[] = [];
  let tree = createTree(maxBlocks);
  let totalHits = 0;
  let totalMisses = 0;
  let totalEvictions = 0;
  const activeLeaves: Map<number, RadixNode> = new Map(); // requestIndex -> leafNode

  // Initial frame
  frames.push({
    tree: JSON.parse(JSON.stringify(tree)),
    highlightIds: new Set(),
    hits: 0,
    misses: 0,
    evictions: 0,
    message: `Scenario: ${scenario.name} — Press Play to start`,
  });

  // Determine max frame
  const maxFrame = Math.max(
    ...scenario.requests.map((r) => r.arrivalFrame + r.durationFrames)
  );

  // Deep clone helper that preserves Map structures
  function cloneTree(t: RadixTree): RadixTree {
    // For frame capture, we re-derive from operations
    // Instead, we rebuild per frame since trees are small
    return JSON.parse(JSON.stringify(t, (_key, value) => {
      if (value instanceof Map) {
        return { __type: "Map", entries: Array.from(value.entries()) };
      }
      return value;
    }), (_key, value) => {
      if (value && value.__type === "Map") {
        return new Map(value.entries);
      }
      return value;
    });
  }

  // Simulate frame by frame
  // Simplified: rebuild tree state from scratch each frame for correctness
  for (let frame = 0; frame <= maxFrame; frame++) {
    tree = createTree(maxBlocks);
    const highlights = new Set<string>();
    let frameHits = totalHits;
    let frameMisses = totalMisses;
    let frameEvictions = totalEvictions;
    let msg = "";
    const activeLeavesThisFrame: Map<number, RadixNode> = new Map();

    // Process all requests that have arrived by this frame
    for (let ri = 0; ri < scenario.requests.length; ri++) {
      const req = scenario.requests[ri];
      if (frame >= req.arrivalFrame) {
        const isActive = frame < req.arrivalFrame + req.durationFrames;
        const isArriving = frame === req.arrivalFrame;

        // Evict if needed before insert
        while (tree.blockCount >= tree.maxBlocks - 2) {
          const evicted = evictNodes(tree, 1);
          if (evicted.length === 0) break;
          frameEvictions += evicted.length;
        }

        const leaf = insertSequence(tree, req.tokens);

        if (isArriving) {
          const { matchedLength } = findPrefix(createTree(maxBlocks), req.tokens);
          msg = `Request ${ri + 1} arrives: "${req.tokens.slice(0, 4).join(" ")}..."`;
          // Highlight newly created nodes
          let n: RadixNode | null = leaf;
          while (n) {
            highlights.add(n.id);
            n = n.parent;
          }
        }

        if (!isActive) {
          // Request has completed — release
          releaseSequence(tree, leaf);
          if (frame === req.arrivalFrame + req.durationFrames) {
            msg = `Request ${ri + 1} completed — releasing cache references`;
          }
        } else {
          activeLeavesThisFrame.set(ri, leaf);
        }
      }
    }

    frames.push({
      tree: cloneTree(tree),
      highlightIds: highlights,
      hits: frameHits,
      misses: frameMisses,
      evictions: frameEvictions,
      message: msg || `Frame ${frame}: ${activeLeavesThisFrame.size} active requests`,
    });
  }

  return frames;
}

export default function KVCachePage() {
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0].id);
  const [maxBlocks, setMaxBlocks] = useState(SCENARIOS[0].maxBlocks);

  const frames = useMemo(
    () => generateFrames(scenarioId, maxBlocks),
    [scenarioId, maxBlocks]
  );

  const [animState, animControls] = useAnimation(frames.length, 400);
  const currentFrame = frames[animState.frame] ?? frames[0];

  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  const handleScenarioChange = useCallback((id: string) => {
    setScenarioId(id);
    const scenario = SCENARIOS.find((s) => s.id === id);
    if (scenario) setMaxBlocks(scenario.maxBlocks);
    animControls.reset();
  }, [animControls]);

  return (
    <div className="kv-cache-page">
      <div className="kv-cache-header">
        <h2>KV Cache — RadixAttention</h2>
        <p className="page-subtitle">
          Visualize how SGLang uses a radix tree to share KV cache across requests with common prefixes
        </p>
      </div>

      <div className="kv-cache-controls">
        <KVCacheSidebar
          selectedScenario={scenarioId}
          onSelectScenario={handleScenarioChange}
          maxBlocks={maxBlocks}
          onMaxBlocksChange={setMaxBlocks}
        />
      </div>

      <AnimationControls state={animState} controls={animControls} label="RadixTree" />

      <div className="kv-cache-message">{currentFrame.message}</div>

      <MetricsPanel
        metrics={[
          { label: "Cache Blocks", value: currentFrame.tree.blockCount, unit: `/ ${currentFrame.tree.maxBlocks}` },
          { label: "Cache Hits", value: currentFrame.hits, color: "var(--green)" },
          { label: "Misses", value: currentFrame.misses, color: "var(--orange)" },
          { label: "Evictions", value: currentFrame.evictions, color: "var(--red)" },
        ]}
      />

      <div className="kv-cache-content">
        <div className="kv-cache-tree-panel">
          <RadixTreeViz tree={currentFrame.tree} highlightIds={currentFrame.highlightIds} />
        </div>
        <div className="kv-cache-memory-panel">
          <MemoryPoolViz
            tree={currentFrame.tree}
            hoveredNodeId={hoveredNodeId}
            onHoverNode={setHoveredNodeId}
          />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create KVCachePage CSS**

Create `frontend/src/pages/KVCachePage.css`:

```css
.kv-cache-page {
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.kv-cache-header h2 {
  font-size: 20px;
  font-weight: 700;
  margin: 0 0 4px;
}

.page-subtitle {
  font-size: 13px;
  color: var(--text-secondary);
  margin: 0;
}

.kv-cache-controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.kv-cache-controls .scenario-list {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.kv-cache-controls .scenario-btn {
  padding: 5px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}

.kv-cache-controls .scenario-btn.active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}

.kv-cache-controls .control-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  display: block;
  margin-bottom: 6px;
}

.kv-cache-controls .sidebar-range {
  width: 100%;
}

.kv-cache-message {
  padding: 8px 12px;
  background: var(--bg-hover);
  border-radius: 8px;
  font-size: 13px;
  color: var(--text);
  font-family: "SF Mono", "Fira Code", monospace;
  min-height: 20px;
}

.kv-cache-content {
  display: flex;
  gap: 20px;
  flex: 1;
  min-height: 0;
}

.kv-cache-tree-panel {
  flex: 1;
  min-width: 0;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  overflow: auto;
}

.kv-cache-memory-panel {
  width: 300px;
  flex-shrink: 0;
}

@media (max-width: 900px) {
  .kv-cache-content {
    flex-direction: column;
  }
  .kv-cache-memory-panel {
    width: 100%;
  }
}
```

- [ ] **Step 4: Add KV Cache section to Sidebar**

In `frontend/src/components/sidebar/Sidebar.tsx`, add a conditional section for the KV Cache page. After the GPU section, add:

```tsx
{isKVCache && (
  <SidebarSection title="KV Cache" isOpen={openSections.has("kvcache")} onToggle={() => toggle("kvcache")}>
    <p style={{ fontSize: 12, color: "var(--text-secondary)", margin: 0 }}>
      Use the controls above the visualization to select scenarios and adjust cache capacity.
    </p>
  </SidebarSection>
)}
```

Note: The KV Cache page has its own inline controls (`KVCacheSidebar`) since its parameters are page-specific (scenario selection, cache capacity) rather than global model parameters.

- [ ] **Step 5: Verify build + dev**

Run:
```bash
cd frontend && npm run build && npm run dev
```

Expected: Navigate to `/kv-cache` tab. Page shows scenario selector, animation controls, placeholder tree (press play to animate), and memory pool grid.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/KVCachePage.tsx frontend/src/pages/KVCachePage.css frontend/src/components/kv-cache/KVCacheSidebar.tsx frontend/src/components/sidebar/Sidebar.tsx
git commit -m "feat: assemble KV Cache page with RadixTree animation, memory pool, and scenario presets"
```

---

## Task 10: Scheduling — Engine (Pure Logic)

**Files:**
- Create: `frontend/src/components/scheduling/SchedulingEngine.ts`

- [ ] **Step 1: Create SchedulingEngine**

Create `frontend/src/components/scheduling/SchedulingEngine.ts`:

```ts
/**
 * Pure simulation engine for SGLang's request scheduling.
 * Generates frame-by-frame state for visualization.
 */

export interface Request {
  id: number;
  /** Number of prefill tokens */
  prefillLength: number;
  /** Number of decode tokens to generate */
  decodeLength: number;
  /** Frame at which this request arrives */
  arrivalFrame: number;
  /** Color for visualization */
  color: string;
}

export interface RunningRequest {
  request: Request;
  /** How many prefill tokens have been processed */
  prefillProgress: number;
  /** How many decode tokens have been generated */
  decodeProgress: number;
  /** Frame when this request started running */
  startFrame: number;
}

export interface SchedulerFrame {
  frame: number;
  waitingQueue: Request[];
  runningBatch: RunningRequest[];
  completedRequests: Request[];
  gpuUtilization: number; // 0-1
  message: string;
}

export type BatchingMode = "continuous" | "static";

export interface SchedulingConfig {
  requests: Request[];
  maxBatchSize: number;
  mode: BatchingMode;
  /** For chunked prefill: max tokens per chunk (0 = no chunking) */
  chunkedPrefillSize: number;
}

const REQUEST_COLORS = [
  "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6",
  "#14b8a6", "#f97316", "#ec4899", "#06b6d4", "#84cc16",
];

export function generateRequests(
  count: number,
  prefillRange: [number, number],
  decodeRange: [number, number],
  arrivalSpread: number
): Request[] {
  const requests: Request[] = [];
  for (let i = 0; i < count; i++) {
    const prefillLength =
      prefillRange[0] + Math.floor(Math.random() * (prefillRange[1] - prefillRange[0]));
    const decodeLength =
      decodeRange[0] + Math.floor(Math.random() * (decodeRange[1] - decodeRange[0]));
    requests.push({
      id: i,
      prefillLength,
      decodeLength,
      arrivalFrame: Math.floor(i * arrivalSpread),
      color: REQUEST_COLORS[i % REQUEST_COLORS.length],
    });
  }
  return requests;
}

function isRequestDone(rr: RunningRequest): boolean {
  return (
    rr.prefillProgress >= rr.request.prefillLength &&
    rr.decodeProgress >= rr.request.decodeLength
  );
}

export function simulateScheduling(config: SchedulingConfig): SchedulerFrame[] {
  const frames: SchedulerFrame[] = [];
  const waiting: Request[] = [];
  let running: RunningRequest[] = [];
  const completed: Request[] = [];

  const maxFrame =
    Math.max(...config.requests.map((r) => r.arrivalFrame)) +
    Math.max(...config.requests.map((r) => r.prefillLength + r.decodeLength)) +
    20;

  for (let frame = 0; frame <= maxFrame; frame++) {
    let msg = "";

    // 1. New arrivals
    for (const req of config.requests) {
      if (req.arrivalFrame === frame) {
        waiting.push(req);
        msg += `Request ${req.id} arrives (prefill: ${req.prefillLength}, decode: ${req.decodeLength}). `;
      }
    }

    // 2. Process running batch — advance each request by 1 step
    for (const rr of running) {
      if (rr.prefillProgress < rr.request.prefillLength) {
        // Prefill phase: process tokens (chunked or all at once)
        const chunkSize =
          config.chunkedPrefillSize > 0
            ? config.chunkedPrefillSize
            : rr.request.prefillLength;
        rr.prefillProgress = Math.min(
          rr.prefillProgress + chunkSize,
          rr.request.prefillLength
        );
      } else {
        // Decode phase: one token per step
        rr.decodeProgress++;
      }
    }

    // 3. Remove completed requests
    if (config.mode === "continuous") {
      // Continuous batching: remove completed individually
      const justCompleted = running.filter(isRequestDone);
      for (const rr of justCompleted) {
        completed.push(rr.request);
        msg += `Request ${rr.request.id} completed! `;
      }
      running = running.filter((rr) => !isRequestDone(rr));
    } else {
      // Static batching: only remove when ALL are done
      if (running.length > 0 && running.every(isRequestDone)) {
        for (const rr of running) {
          completed.push(rr.request);
        }
        msg += `Batch completed (${running.length} requests). `;
        running = [];
      }
    }

    // 4. Schedule new requests from waiting queue
    if (config.mode === "continuous") {
      // Fill up to maxBatchSize
      while (running.length < config.maxBatchSize && waiting.length > 0) {
        const req = waiting.shift()!;
        running.push({
          request: req,
          prefillProgress: 0,
          decodeProgress: 0,
          startFrame: frame,
        });
        msg += `Scheduled request ${req.id}. `;
      }
    } else {
      // Static: only schedule a new batch when current batch is empty
      if (running.length === 0 && waiting.length > 0) {
        const batchSize = Math.min(config.maxBatchSize, waiting.length);
        for (let i = 0; i < batchSize; i++) {
          const req = waiting.shift()!;
          running.push({
            request: req,
            prefillProgress: 0,
            decodeProgress: 0,
            startFrame: frame,
          });
        }
        msg += `New batch started (${batchSize} requests). `;
      }
    }

    // GPU utilization = running / maxBatchSize
    const gpuUtil = running.length / config.maxBatchSize;

    frames.push({
      frame,
      waitingQueue: [...waiting],
      runningBatch: running.map((rr) => ({ ...rr })),
      completedRequests: [...completed],
      gpuUtilization: gpuUtil,
      message: msg.trim() || `Frame ${frame}`,
    });

    // Early exit if everything is done
    if (completed.length === config.requests.length && running.length === 0 && waiting.length === 0) {
      break;
    }
  }

  return frames;
}

/**
 * Generate frames for chunked prefill comparison.
 * Shows a timeline of what executes each step.
 */
export interface PrefillFrame {
  frame: number;
  /** What's executing this step: prefill chunks and decode tokens */
  executing: {
    requestId: number;
    type: "prefill" | "decode";
    tokenRange?: [number, number]; // for prefill chunks
    color: string;
  }[];
  decodeLatencies: number[]; // per-decode-request latency this step
}

export function simulateChunkedPrefill(
  longPrefillLength: number,
  decodeRequestCount: number,
  chunkSize: number // 0 means no chunking
): PrefillFrame[] {
  const frames: PrefillFrame[] = [];
  let prefillRemaining = longPrefillLength;
  const decodeTokensGenerated = new Array(decodeRequestCount).fill(0);
  const targetDecodeTokens = 10; // each decode request generates 10 tokens

  if (chunkSize === 0) {
    // No chunking: prefill blocks everything
    frames.push({
      frame: 0,
      executing: [{ requestId: 0, type: "prefill", tokenRange: [0, longPrefillLength], color: REQUEST_COLORS[0] }],
      decodeLatencies: [],
    });

    // Then decode requests run
    for (let step = 0; step < targetDecodeTokens; step++) {
      const executing = [];
      for (let i = 0; i < decodeRequestCount; i++) {
        executing.push({ requestId: i + 1, type: "decode" as const, color: REQUEST_COLORS[(i + 1) % REQUEST_COLORS.length] });
      }
      frames.push({ frame: frames.length, executing, decodeLatencies: [longPrefillLength + step] });
    }
  } else {
    // Chunked: interleave prefill chunks with decode
    let step = 0;
    while (prefillRemaining > 0 || decodeTokensGenerated.some((d) => d < targetDecodeTokens)) {
      const executing = [];

      if (prefillRemaining > 0) {
        const chunk = Math.min(chunkSize, prefillRemaining);
        const start = longPrefillLength - prefillRemaining;
        executing.push({
          requestId: 0,
          type: "prefill" as const,
          tokenRange: [start, start + chunk] as [number, number],
          color: REQUEST_COLORS[0],
        });
        prefillRemaining -= chunk;
      }

      // Decode requests also get a step
      for (let i = 0; i < decodeRequestCount; i++) {
        if (decodeTokensGenerated[i] < targetDecodeTokens) {
          executing.push({
            requestId: i + 1,
            type: "decode" as const,
            color: REQUEST_COLORS[(i + 1) % REQUEST_COLORS.length],
          });
          decodeTokensGenerated[i]++;
        }
      }

      frames.push({ frame: step, executing, decodeLatencies: [step] });
      step++;
    }
  }

  return frames;
}
```

- [ ] **Step 2: Verify build and commit**

Run:
```bash
cd frontend && npm run build
git add frontend/src/components/scheduling/SchedulingEngine.ts
git commit -m "feat: add scheduling simulation engine (continuous/static batching, chunked prefill)"
```

---

## Task 11: Scheduling — Continuous Batching Visualization

**Files:**
- Create: `frontend/src/components/scheduling/ContinuousBatchingViz.tsx`
- Create: `frontend/src/components/scheduling/ContinuousBatchingViz.css`

- [ ] **Step 1: Create ContinuousBatchingViz**

Create `frontend/src/components/scheduling/ContinuousBatchingViz.tsx`:

```tsx
import type { SchedulerFrame, RunningRequest, Request } from "./SchedulingEngine";
import "./ContinuousBatchingViz.css";

interface Props {
  frame: SchedulerFrame;
  maxBatchSize: number;
}

function RequestBar({
  request,
  progress,
  total,
  phase,
}: {
  request: Request;
  progress: number;
  total: number;
  phase: "waiting" | "prefill" | "decode";
}) {
  const pct = total > 0 ? (progress / total) * 100 : 0;
  return (
    <div className="request-bar" title={`Request ${request.id} — ${phase}`}>
      <div className="request-bar-label">R{request.id}</div>
      <div className="request-bar-track">
        <div
          className={`request-bar-fill ${phase}`}
          style={{ width: `${pct}%`, backgroundColor: request.color }}
        />
      </div>
      <div className="request-bar-info">
        {progress}/{total}
      </div>
    </div>
  );
}

export default function ContinuousBatchingViz({ frame, maxBatchSize }: Props) {
  return (
    <div className="cb-viz">
      {/* Waiting Queue */}
      <div className="cb-queue">
        <div className="cb-queue-header">
          <span className="cb-queue-title">Waiting Queue</span>
          <span className="cb-queue-count">{frame.waitingQueue.length}</span>
        </div>
        <div className="cb-queue-list">
          {frame.waitingQueue.length === 0 && (
            <div className="cb-empty">Empty</div>
          )}
          {frame.waitingQueue.map((req) => (
            <RequestBar
              key={req.id}
              request={req}
              progress={0}
              total={req.prefillLength + req.decodeLength}
              phase="waiting"
            />
          ))}
        </div>
      </div>

      {/* Arrow */}
      <div className="cb-arrow">
        <svg width="40" height="24" viewBox="0 0 40 24">
          <path d="M4 12 L30 12 M24 6 L30 12 L24 18" fill="none" stroke="var(--accent)" strokeWidth="2" />
        </svg>
        <span className="cb-arrow-label">Schedule</span>
      </div>

      {/* Running Batch */}
      <div className="cb-batch">
        <div className="cb-queue-header">
          <span className="cb-queue-title">Running Batch</span>
          <span className="cb-queue-count">
            {frame.runningBatch.length} / {maxBatchSize}
          </span>
        </div>
        <div className="cb-queue-list">
          {frame.runningBatch.length === 0 && (
            <div className="cb-empty">Empty</div>
          )}
          {frame.runningBatch.map((rr) => {
            const isPrefill = rr.prefillProgress < rr.request.prefillLength;
            const progress = isPrefill
              ? rr.prefillProgress
              : rr.request.prefillLength + rr.decodeProgress;
            const total = rr.request.prefillLength + rr.request.decodeLength;
            return (
              <RequestBar
                key={rr.request.id}
                request={rr.request}
                progress={progress}
                total={total}
                phase={isPrefill ? "prefill" : "decode"}
              />
            );
          })}
          {/* Empty slots */}
          {Array.from({ length: maxBatchSize - frame.runningBatch.length }).map((_, i) => (
            <div key={`empty-${i}`} className="request-bar empty-slot">
              <div className="request-bar-track" />
            </div>
          ))}
        </div>
      </div>

      {/* Completed */}
      <div className="cb-completed">
        <div className="cb-queue-header">
          <span className="cb-queue-title">Completed</span>
          <span className="cb-queue-count">{frame.completedRequests.length}</span>
        </div>
        <div className="cb-completed-chips">
          {frame.completedRequests.map((req) => (
            <span
              key={req.id}
              className="cb-completed-chip"
              style={{ backgroundColor: req.color }}
            >
              R{req.id}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create ContinuousBatchingViz CSS**

Create `frontend/src/components/scheduling/ContinuousBatchingViz.css`:

```css
.cb-viz {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.cb-queue,
.cb-batch {
  flex: 1;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  min-width: 0;
}

.cb-queue-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.cb-queue-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
}

.cb-queue-count {
  font-size: 12px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
  background: var(--bg-hover);
  padding: 2px 8px;
  border-radius: 4px;
}

.cb-queue-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.cb-empty {
  font-size: 12px;
  color: var(--text-secondary);
  text-align: center;
  padding: 12px;
}

.request-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 28px;
}

.request-bar.empty-slot {
  opacity: 0.3;
}

.request-bar-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text);
  width: 28px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.request-bar-track {
  flex: 1;
  height: 20px;
  background: var(--bg-hover);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}

.request-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.request-bar-fill.prefill {
  opacity: 0.7;
  background-image: repeating-linear-gradient(
    45deg,
    transparent,
    transparent 4px,
    rgba(255, 255, 255, 0.15) 4px,
    rgba(255, 255, 255, 0.15) 8px
  );
}

.request-bar-info {
  font-size: 10px;
  color: var(--text-secondary);
  width: 48px;
  text-align: left;
  font-variant-numeric: tabular-nums;
}

.cb-arrow {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding-top: 40px;
}

.cb-arrow-label {
  font-size: 10px;
  color: var(--text-secondary);
}

.cb-completed {
  width: 120px;
  flex-shrink: 0;
}

.cb-completed-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.cb-completed-chip {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 20px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  color: #fff;
}

@media (max-width: 900px) {
  .cb-viz {
    flex-direction: column;
  }
  .cb-arrow {
    transform: rotate(90deg);
    padding-top: 0;
  }
}
```

- [ ] **Step 3: Verify build and commit**

Run:
```bash
cd frontend && npm run build
git add frontend/src/components/scheduling/ContinuousBatchingViz.tsx frontend/src/components/scheduling/ContinuousBatchingViz.css
git commit -m "feat: add Continuous Batching dual-queue visualization component"
```

---

## Task 12: Scheduling — Chunked Prefill Visualization

**Files:**
- Create: `frontend/src/components/scheduling/ChunkedPrefillViz.tsx`
- Create: `frontend/src/components/scheduling/ChunkedPrefillViz.css`

- [ ] **Step 1: Create ChunkedPrefillViz**

Create `frontend/src/components/scheduling/ChunkedPrefillViz.tsx`:

```tsx
import type { PrefillFrame } from "./SchedulingEngine";
import "./ChunkedPrefillViz.css";

interface Props {
  frames: PrefillFrame[];
  currentFrame: number;
  totalPrefillLength: number;
}

export default function ChunkedPrefillViz({ frames, currentFrame, totalPrefillLength }: Props) {
  const visibleFrames = frames.slice(0, currentFrame + 1);

  return (
    <div className="cp-viz">
      <div className="cp-timeline-header">
        <span className="cp-timeline-label">Step</span>
        <span className="cp-timeline-label" style={{ flex: 1 }}>Execution Timeline</span>
      </div>
      <div className="cp-timeline">
        {visibleFrames.map((f, i) => (
          <div key={i} className={`cp-step${i === currentFrame ? " current" : ""}`}>
            <span className="cp-step-num">{i}</span>
            <div className="cp-step-bars">
              {f.executing.map((ex, j) => (
                <div
                  key={j}
                  className={`cp-exec-block ${ex.type}`}
                  style={{ backgroundColor: ex.color }}
                  title={
                    ex.type === "prefill"
                      ? `Prefill [${ex.tokenRange?.[0]}:${ex.tokenRange?.[1]}]`
                      : `Decode R${ex.requestId}`
                  }
                >
                  {ex.type === "prefill" ? (
                    <span>P [{ex.tokenRange?.[0]}:{ex.tokenRange?.[1]}]</span>
                  ) : (
                    <span>D{ex.requestId}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create ChunkedPrefillViz CSS**

Create `frontend/src/components/scheduling/ChunkedPrefillViz.css`:

```css
.cp-viz {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}

.cp-timeline-header {
  display: flex;
  gap: 12px;
  margin-bottom: 8px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}

.cp-timeline-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.cp-timeline {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 400px;
  overflow-y: auto;
}

.cp-step {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 8px;
  border-radius: 6px;
  transition: background 0.15s;
}

.cp-step.current {
  background: var(--bg-hover);
}

.cp-step-num {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  width: 24px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.cp-step-bars {
  display: flex;
  gap: 4px;
  flex: 1;
}

.cp-exec-block {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 600;
  color: #fff;
  white-space: nowrap;
}

.cp-exec-block.prefill {
  opacity: 0.8;
  background-image: repeating-linear-gradient(
    45deg,
    transparent,
    transparent 3px,
    rgba(255, 255, 255, 0.15) 3px,
    rgba(255, 255, 255, 0.15) 6px
  );
}
```

- [ ] **Step 3: Verify build and commit**

Run:
```bash
cd frontend && npm run build
git add frontend/src/components/scheduling/ChunkedPrefillViz.tsx frontend/src/components/scheduling/ChunkedPrefillViz.css
git commit -m "feat: add Chunked Prefill timeline visualization component"
```

---

## Task 13: Scheduling — Page Assembly

**Files:**
- Create: `frontend/src/components/scheduling/SchedulingSidebar.tsx`
- Modify: `frontend/src/pages/SchedulingPage.tsx`
- Create: `frontend/src/pages/SchedulingPage.css`

- [ ] **Step 1: Create SchedulingSidebar**

Create `frontend/src/components/scheduling/SchedulingSidebar.tsx`:

```tsx
import type { BatchingMode } from "./SchedulingEngine";

interface Props {
  requestCount: number;
  onRequestCountChange: (n: number) => void;
  maxBatchSize: number;
  onMaxBatchSizeChange: (n: number) => void;
  arrivalSpread: number;
  onArrivalSpreadChange: (n: number) => void;
}

export default function SchedulingSidebar({
  requestCount,
  onRequestCountChange,
  maxBatchSize,
  onMaxBatchSizeChange,
  arrivalSpread,
  onArrivalSpreadChange,
}: Props) {
  return (
    <div className="scheduling-sidebar-controls">
      <label className="control-label">
        Requests: {requestCount}
      </label>
      <input
        type="range"
        min={3}
        max={12}
        value={requestCount}
        onChange={(e) => onRequestCountChange(Number(e.target.value))}
        className="sidebar-range"
      />

      <label className="control-label" style={{ marginTop: 8 }}>
        Max Batch Size: {maxBatchSize}
      </label>
      <input
        type="range"
        min={2}
        max={8}
        value={maxBatchSize}
        onChange={(e) => onMaxBatchSizeChange(Number(e.target.value))}
        className="sidebar-range"
      />

      <label className="control-label" style={{ marginTop: 8 }}>
        Arrival Interval: {arrivalSpread} frames
      </label>
      <input
        type="range"
        min={1}
        max={8}
        value={arrivalSpread}
        onChange={(e) => onArrivalSpreadChange(Number(e.target.value))}
        className="sidebar-range"
      />
    </div>
  );
}
```

- [ ] **Step 2: Implement full SchedulingPage**

Replace `frontend/src/pages/SchedulingPage.tsx`:

```tsx
import { useState, useMemo, useCallback } from "react";
import ContinuousBatchingViz from "../components/scheduling/ContinuousBatchingViz";
import ChunkedPrefillViz from "../components/scheduling/ChunkedPrefillViz";
import SchedulingSidebar from "../components/scheduling/SchedulingSidebar";
import AnimationControls from "../components/shared/AnimationControls";
import ComparisonToggle from "../components/shared/ComparisonToggle";
import MetricsPanel from "../components/shared/MetricsPanel";
import { useAnimation } from "../components/shared/useAnimation";
import {
  generateRequests,
  simulateScheduling,
  simulateChunkedPrefill,
  type BatchingMode,
} from "../components/scheduling/SchedulingEngine";
import "./SchedulingPage.css";

type Section = "batching" | "chunked-prefill";

export default function SchedulingPage() {
  const [activeSection, setActiveSection] = useState<Section>("batching");

  // Batching parameters
  const [requestCount, setRequestCount] = useState(6);
  const [maxBatchSize, setMaxBatchSize] = useState(4);
  const [arrivalSpread, setArrivalSpread] = useState(3);
  const [batchingMode, setBatchingMode] = useState<"a" | "b">("a");

  // Chunked prefill parameters
  const [prefillLength, setPrefillLength] = useState(16);
  const [decodeCount, setDecodeCount] = useState(3);
  const [chunkMode, setChunkMode] = useState<"a" | "b">("a");
  const chunkSize = 4; // tokens per chunk when chunking is enabled

  // Generate requests (memoized, stable across mode toggle)
  const requests = useMemo(
    () => generateRequests(requestCount, [4, 12], [6, 15], arrivalSpread),
    [requestCount, arrivalSpread]
  );

  // Simulate both modes
  const continuousFrames = useMemo(
    () =>
      simulateScheduling({
        requests,
        maxBatchSize,
        mode: "continuous",
        chunkedPrefillSize: 0,
      }),
    [requests, maxBatchSize]
  );

  const staticFrames = useMemo(
    () =>
      simulateScheduling({
        requests,
        maxBatchSize,
        mode: "static",
        chunkedPrefillSize: 0,
      }),
    [requests, maxBatchSize]
  );

  const activeFrames = batchingMode === "a" ? continuousFrames : staticFrames;
  const [batchAnimState, batchAnimControls] = useAnimation(activeFrames.length, 350);
  const currentBatchFrame = activeFrames[batchAnimState.frame] ?? activeFrames[0];

  // Chunked prefill simulation
  const noChunkFrames = useMemo(
    () => simulateChunkedPrefill(prefillLength, decodeCount, 0),
    [prefillLength, decodeCount]
  );
  const chunkedFrames = useMemo(
    () => simulateChunkedPrefill(prefillLength, decodeCount, chunkSize),
    [prefillLength, decodeCount]
  );
  const activePrefillFrames = chunkMode === "a" ? chunkedFrames : noChunkFrames;
  const [prefillAnimState, prefillAnimControls] = useAnimation(activePrefillFrames.length, 500);

  // Compute average GPU utilization for metrics
  const avgGpuUtil =
    activeFrames.length > 0
      ? (
          activeFrames
            .slice(0, batchAnimState.frame + 1)
            .reduce((sum, f) => sum + f.gpuUtilization, 0) /
          (batchAnimState.frame + 1) *
          100
        ).toFixed(0)
      : "0";

  const handleBatchingModeChange = useCallback(
    (mode: "a" | "b") => {
      setBatchingMode(mode);
      batchAnimControls.reset();
    },
    [batchAnimControls]
  );

  return (
    <div className="scheduling-page">
      <div className="scheduling-header">
        <h2>Request Scheduling</h2>
        <p className="page-subtitle">
          Explore how SGLang schedules and batches inference requests for maximum GPU utilization
        </p>
      </div>

      {/* Section tabs */}
      <div className="scheduling-sections">
        <button
          className={`section-tab${activeSection === "batching" ? " active" : ""}`}
          onClick={() => setActiveSection("batching")}
        >
          Continuous Batching
        </button>
        <button
          className={`section-tab${activeSection === "chunked-prefill" ? " active" : ""}`}
          onClick={() => setActiveSection("chunked-prefill")}
        >
          Chunked Prefill
        </button>
      </div>

      {activeSection === "batching" && (
        <>
          <div className="scheduling-toolbar">
            <ComparisonToggle
              labelA="Continuous Batching"
              labelB="Static Batching"
              active={batchingMode}
              onChange={handleBatchingModeChange}
            />
            <SchedulingSidebar
              requestCount={requestCount}
              onRequestCountChange={(n) => { setRequestCount(n); batchAnimControls.reset(); }}
              maxBatchSize={maxBatchSize}
              onMaxBatchSizeChange={(n) => { setMaxBatchSize(n); batchAnimControls.reset(); }}
              arrivalSpread={arrivalSpread}
              onArrivalSpreadChange={(n) => { setArrivalSpread(n); batchAnimControls.reset(); }}
            />
          </div>

          <AnimationControls state={batchAnimState} controls={batchAnimControls} label="Scheduling" />

          <div className="scheduling-message">{currentBatchFrame.message}</div>

          <MetricsPanel
            metrics={[
              { label: "Mode", value: batchingMode === "a" ? "Continuous" : "Static" },
              { label: "GPU Utilization", value: `${avgGpuUtil}%`, color: Number(avgGpuUtil) > 70 ? "var(--green)" : "var(--orange)" },
              { label: "Waiting", value: currentBatchFrame.waitingQueue.length },
              { label: "Running", value: `${currentBatchFrame.runningBatch.length}/${maxBatchSize}` },
              { label: "Completed", value: currentBatchFrame.completedRequests.length, color: "var(--green)" },
            ]}
          />

          <ContinuousBatchingViz frame={currentBatchFrame} maxBatchSize={maxBatchSize} />
        </>
      )}

      {activeSection === "chunked-prefill" && (
        <>
          <div className="scheduling-toolbar">
            <ComparisonToggle
              labelA="Chunked Prefill"
              labelB="No Chunking"
              active={chunkMode}
              onChange={(m) => { setChunkMode(m); prefillAnimControls.reset(); }}
            />
            <div className="scheduling-sidebar-controls">
              <label className="control-label">Prefill Length: {prefillLength}</label>
              <input
                type="range"
                min={8}
                max={32}
                value={prefillLength}
                onChange={(e) => { setPrefillLength(Number(e.target.value)); prefillAnimControls.reset(); }}
                className="sidebar-range"
              />
              <label className="control-label" style={{ marginTop: 8 }}>Decode Requests: {decodeCount}</label>
              <input
                type="range"
                min={1}
                max={6}
                value={decodeCount}
                onChange={(e) => { setDecodeCount(Number(e.target.value)); prefillAnimControls.reset(); }}
                className="sidebar-range"
              />
            </div>
          </div>

          <AnimationControls state={prefillAnimState} controls={prefillAnimControls} label="Prefill" />

          <ChunkedPrefillViz
            frames={activePrefillFrames}
            currentFrame={prefillAnimState.frame}
            totalPrefillLength={prefillLength}
          />
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Create SchedulingPage CSS**

Create `frontend/src/pages/SchedulingPage.css`:

```css
.scheduling-page {
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.scheduling-header h2 {
  font-size: 20px;
  font-weight: 700;
  margin: 0 0 4px;
}

.scheduling-sections {
  display: flex;
  gap: 4px;
  background: var(--bg-hover);
  border-radius: 8px;
  padding: 3px;
  width: fit-content;
}

.section-tab {
  padding: 6px 16px;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
}

.section-tab.active {
  background: var(--bg-card);
  color: var(--text);
  box-shadow: var(--shadow-sm);
}

.scheduling-toolbar {
  display: flex;
  align-items: flex-start;
  gap: 20px;
  flex-wrap: wrap;
}

.scheduling-sidebar-controls {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.scheduling-sidebar-controls .control-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.scheduling-sidebar-controls .sidebar-range {
  width: 180px;
}

.scheduling-message {
  padding: 8px 12px;
  background: var(--bg-hover);
  border-radius: 8px;
  font-size: 13px;
  color: var(--text);
  font-family: "SF Mono", "Fira Code", monospace;
  min-height: 20px;
}
```

- [ ] **Step 4: Verify build + dev**

Run:
```bash
cd frontend && npm run build && npm run dev
```

Expected: Navigate to `/scheduling`. Shows Continuous Batching tab with dual-queue animation and A/B toggle between continuous/static batching. Chunked Prefill tab shows timeline. All animations play/pause/step work correctly.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/SchedulingPage.tsx frontend/src/pages/SchedulingPage.css frontend/src/components/scheduling/SchedulingSidebar.tsx
git commit -m "feat: assemble Scheduling page with continuous batching and chunked prefill visualizations"
```

---

## Task 14: Final Integration & Polish

**Files:**
- Modify: `frontend/src/App.css`
- Verify all routes work end-to-end

- [ ] **Step 1: Clean up App.css**

In `frontend/src/App.css`:
- Change `.app` `padding-top` to `52px` (matching new TopNav height)
- Remove any old `.tab-bar` styles (now in TopNav.css)
- Ensure `.app-body` still works with the new route-based rendering

- [ ] **Step 2: Delete PlaneTabBar files**

Delete `frontend/src/components/controls/PlaneTabBar.tsx` and its CSS import (if any separate CSS file exists for it — based on the codebase analysis, styles are in `Controls.css`). Remove `.tab-bar` and `.tab-btn` styles from `Controls.css` if present, since TopNav replaces them.

- [ ] **Step 3: Add .superpowers to .gitignore**

Append to `frontend/.gitignore` (or root `.gitignore`):
```
.superpowers/
```

- [ ] **Step 4: Full build and verify all routes**

Run:
```bash
cd frontend && npm run build && npm run dev
```

Manually verify:
1. `/compute` — renders existing Compute Plane with model selector, TP viz, GPU memory panel
2. `/control` — renders existing Control Plane with runtime architecture diagram
3. `/kv-cache` — renders RadixAttention tree with scenario selector, animation controls, memory pool
4. `/scheduling` — renders Continuous Batching with A/B toggle, Chunked Prefill timeline
5. Tab navigation highlights the active tab correctly
6. Sidebar dynamically shows relevant sections per page
7. Dark/light theme toggle works on all pages
8. Direct URL navigation works (e.g., refresh on `/kv-cache`)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: finalize P0 integration — navigation, KV Cache, Scheduling pages fully wired"
```

---

## Parallelization Notes

Tasks can be parallelized as follows:

- **Track A (Infrastructure):** Tasks 1 → 2 → 3 (sequential, each depends on previous)
- **Track B (KV Cache):** Tasks 5 → 6 → 7 → 8 → 9 (sequential within track, but 5+6 can start alongside Track A since they're pure logic with no component imports)
- **Track C (Scheduling):** Tasks 10 → 11 → 12 → 13 (sequential within track, but 10 can start alongside Track A)
- **Track D (Shared):** Task 4 (independent, can be done in parallel with anything)
- **Final:** Task 14 depends on all other tasks being complete

Recommended parallel groups:
1. **Batch 1:** Task 1 + Task 4 + Task 5 + Task 6 + Task 10 (all independent)
2. **Batch 2:** Task 2 + Task 3 (depends on Task 1) + Task 7 + Task 11 (depends on 5, 10)
3. **Batch 3:** Task 8 + Task 12 (depends on 7, 11)
4. **Batch 4:** Task 9 + Task 13 (page assembly, depends on all components)
5. **Batch 5:** Task 14 (integration, depends on everything)
