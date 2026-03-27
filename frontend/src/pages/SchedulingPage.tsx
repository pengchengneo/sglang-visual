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
  const chunkSize = 4;

  const requests = useMemo(
    () => generateRequests(requestCount, [4, 12], [6, 15], arrivalSpread),
    [requestCount, arrivalSpread]
  );

  const continuousFrames = useMemo(
    () => simulateScheduling({ requests, maxBatchSize, mode: "continuous", chunkedPrefillSize: 0 }),
    [requests, maxBatchSize]
  );

  const staticFrames = useMemo(
    () => simulateScheduling({ requests, maxBatchSize, mode: "static", chunkedPrefillSize: 0 }),
    [requests, maxBatchSize]
  );

  const activeFrames = batchingMode === "a" ? continuousFrames : staticFrames;
  const [batchAnimState, batchAnimControls] = useAnimation(activeFrames.length, 350);
  const currentBatchFrame = activeFrames[batchAnimState.frame] ?? activeFrames[0];

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
          />
        </>
      )}
    </div>
  );
}
