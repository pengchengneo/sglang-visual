import { useAppContext } from "../contexts/AppContext";
import { PipelineView } from "../components/pipeline/PipelineView";
import { GpuMemoryPanel } from "../components/gpu/GpuMemoryPanel";
import { ErrorBoundary } from "../components/ErrorBoundary";

export default function ComputePlanePage() {
  const ctx = useAppContext();

  if (ctx.modelLoading) {
    return <div className="loading">Loading model data...</div>;
  }

  if (!ctx.model) {
    return (
      <div className="empty-state">
        <p>Select a model to explore inference optimizations.</p>
      </div>
    );
  }

  return (
    <div className="split-layout">
      <div className="panel-left">
        <ErrorBoundary fallbackTitle="Pipeline view error">
          <PipelineView
            key={ctx.model.model_id}
            model={ctx.model}
            tpSize={ctx.tpSize}
            ppSize={ctx.ppSize}
            epSize={ctx.epSize}
            bytesPerParam={ctx.bytesPerParam}
            dtype={ctx.dtype}
            quantization={ctx.quantization}
          />
        </ErrorBoundary>
      </div>
      <div className="panel-right">
        <ErrorBoundary fallbackTitle="GPU memory panel error">
          <GpuMemoryPanel
            config={ctx.model.config}
            tpSize={ctx.tpSize}
            dpSize={ctx.dpSize}
            ppSize={ctx.ppSize}
            epSize={ctx.epSize}
            enableDpAttention={ctx.enableDpAttention}
            perRankParams={ctx.perRankParams}
            gpuMemoryBytes={ctx.gpuMemoryBytes}
            memFractionStatic={ctx.memFractionStatic}
            bytesPerParam={ctx.bytesPerParam}
            kvBytesPerElement={ctx.kvBytesPerElement}
            contextLength={ctx.contextLength}
          />
        </ErrorBoundary>
      </div>
    </div>
  );
}
