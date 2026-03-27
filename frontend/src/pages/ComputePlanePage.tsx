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
