import { createContext, useContext, useState, useMemo, type ReactNode } from "react";
import { useManifest, useModelData } from "../hooks/useModelData";
import { computePerRankParams, computePerRankParamsForPpStage } from "../utils/tpMath";
import type { ModelArchitecture, ModelConfig, PresetManifestEntry } from "../types/model";

// ── Type exports ──────────────────────────────────────────────────────────────
export type Dtype = "fp16" | "bf16" | "fp32";
export type Quantization = "none" | "int8" | "int4" | "fp8";
export type KvCacheDtype = "fp16" | "fp8";
export type SchedulePolicy = "fcfs" | "lpm" | "random" | "dfs-weight";
export type SpecAlgorithm = "none" | "eagle" | "eagle3" | "nextn" | "ngram";

// ── Constants ─────────────────────────────────────────────────────────────────
const DEFAULT_GPU_BYTES = 80e9; // 80 GB
const DEFAULT_MEM_FRACTION = 0.88;

const BYTES_PER_PARAM: Record<Quantization, number> & Record<Dtype, number> = {
  none: 2, int8: 1, int4: 0.5, fp8: 1,
  fp16: 2, bf16: 2, fp32: 4,
};

// ── Context shape ─────────────────────────────────────────────────────────────
interface AppContextValue {
  // Data
  model: ModelArchitecture | null;
  manifest: PresetManifestEntry[];
  selectedPreset: string | null;
  setSelectedPreset: (id: string | null) => void;
  manifestLoading: boolean;
  manifestError: string | null;

  // Parallelism
  tpSize: number;
  setTpSize: (v: number) => void;
  dpSize: number;
  setDpSize: (v: number) => void;
  ppSize: number;
  setPpSize: (v: number) => void;
  epSize: number;
  setEpSize: (v: number) => void;
  enableDpAttention: boolean;
  setEnableDpAttention: (v: boolean) => void;

  // GPU
  gpuMemoryBytes: number;
  setGpuMemoryBytes: (v: number) => void;
  memFractionStatic: number;
  setMemFractionStatic: (v: number) => void;

  // Quantization
  dtype: Dtype;
  setDtype: (v: Dtype) => void;
  quantization: Quantization;
  setQuantization: (v: Quantization) => void;
  kvCacheDtype: KvCacheDtype;
  setKvCacheDtype: (v: KvCacheDtype) => void;

  // Scheduling
  schedulePolicy: SchedulePolicy;
  setSchedulePolicy: (v: SchedulePolicy) => void;
  chunkedPrefillSize: number;
  setChunkedPrefillSize: (v: number) => void;
  disableRadixCache: boolean;
  setDisableRadixCache: (v: boolean) => void;

  // Context length
  contextLength: number;
  setContextLength: (v: number) => void;

  // Speculative decoding
  specAlgorithm: SpecAlgorithm;
  setSpecAlgorithm: (v: SpecAlgorithm) => void;
  specNumDraftTokens: number;
  setSpecNumDraftTokens: (v: number) => void;

  // CUDA Graph
  cudaGraphMaxBs: number;
  setCudaGraphMaxBs: (v: number) => void;
  disableCudaGraph: boolean;
  setDisableCudaGraph: (v: boolean) => void;

  // Derived
  bytesPerParam: number;
  kvBytesPerElement: number;
  modelConfig: ModelConfig | null;
  perRankParams: number;

  // Loading states
  modelLoading: boolean;
  modelError: string | null;
}

const AppContext = createContext<AppContextValue | null>(null);

// ── Hook ──────────────────────────────────────────────────────────────────────
export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useAppContext must be used within AppProvider");
  return ctx;
}

// ── Provider ──────────────────────────────────────────────────────────────────
export function AppProvider({ children }: { children: ReactNode }) {
  const { manifest, loading: manifestLoading, error: manifestError } = useManifest();
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const { model, loading: modelLoading, error: modelError } = useModelData(selectedPreset);

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

  // Context length
  const [contextLength, setContextLength] = useState(4096);

  // Speculative decoding
  const [specAlgorithm, setSpecAlgorithm] = useState<SpecAlgorithm>("none");
  const [specNumDraftTokens, setSpecNumDraftTokens] = useState(5);

  // CUDA Graph
  const [cudaGraphMaxBs, setCudaGraphMaxBs] = useState(128);
  const [disableCudaGraph, setDisableCudaGraph] = useState(false);

  // Derived
  const bytesPerParam = quantization !== "none"
    ? BYTES_PER_PARAM[quantization]
    : BYTES_PER_PARAM[dtype];
  const kvBytesPerElement = kvCacheDtype === "fp8" ? 1 : 2;
  const modelConfig = model?.config ?? null;

  const perRankParams = useMemo(() => {
    if (!model) return 0;
    if (ppSize <= 1) return computePerRankParams(model, tpSize, epSize);
    let maxParams = 0;
    for (let p = 0; p < ppSize; p++) {
      maxParams = Math.max(maxParams, computePerRankParamsForPpStage(model, tpSize, p, ppSize, epSize));
    }
    return maxParams;
  }, [model, tpSize, ppSize, epSize]);

  const value: AppContextValue = {
    // Data
    model,
    manifest,
    selectedPreset,
    setSelectedPreset,
    manifestLoading,
    manifestError: manifestError ?? null,

    // Parallelism
    tpSize, setTpSize,
    dpSize, setDpSize,
    ppSize, setPpSize,
    epSize, setEpSize,
    enableDpAttention, setEnableDpAttention,

    // GPU
    gpuMemoryBytes, setGpuMemoryBytes,
    memFractionStatic, setMemFractionStatic,

    // Quantization
    dtype, setDtype,
    quantization, setQuantization,
    kvCacheDtype, setKvCacheDtype,

    // Scheduling
    schedulePolicy, setSchedulePolicy,
    chunkedPrefillSize, setChunkedPrefillSize,
    disableRadixCache, setDisableRadixCache,

    // Context length
    contextLength, setContextLength,

    // Speculative decoding
    specAlgorithm, setSpecAlgorithm,
    specNumDraftTokens, setSpecNumDraftTokens,

    // CUDA Graph
    cudaGraphMaxBs, setCudaGraphMaxBs,
    disableCudaGraph, setDisableCudaGraph,

    // Derived
    bytesPerParam,
    kvBytesPerElement,
    modelConfig,
    perRankParams,

    // Loading
    modelLoading,
    modelError: modelError ?? null,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}
