import { useAppContext } from "../contexts/AppContext";
import { ControlPlaneView } from "../components/controlplane/ControlPlaneView";
import { ErrorBoundary } from "../components/ErrorBoundary";

export default function ControlPlanePage() {
  const ctx = useAppContext();

  return (
    <ErrorBoundary fallbackTitle="Control plane error">
      <ControlPlaneView
        tpSize={ctx.tpSize}
        dpSize={ctx.dpSize}
        ppSize={ctx.ppSize}
        epSize={ctx.epSize}
        enableDpAttention={ctx.enableDpAttention}
        modelConfig={ctx.modelConfig}
        schedulePolicy={ctx.schedulePolicy}
        chunkedPrefillSize={ctx.chunkedPrefillSize}
        disableRadixCache={ctx.disableRadixCache}
        specAlgorithm={ctx.specAlgorithm}
        specNumDraftTokens={ctx.specNumDraftTokens}
        cudaGraphMaxBs={ctx.cudaGraphMaxBs}
        disableCudaGraph={ctx.disableCudaGraph}
      />
    </ErrorBoundary>
  );
}
