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
