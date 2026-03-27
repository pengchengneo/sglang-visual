/**
 * Control Plane conceptual architecture diagram.
 *
 * Shows SGLang runtime layers: HTTP API -> TokenizerManager -> Scheduler ->
 * TpModelWorker -> ModelRunner -> GPU.
 *
 * When DP Attention is enabled, shows DataParallelController with separate
 * schedulers but shared worker pool. When traditional DP > 1 (no dp_attention),
 * shows side-by-side DP groups.
 */

import type { SchedulePolicy, SpecAlgorithm } from "../../App";
import type { ModelConfig } from "../../types/model";
import "./ControlPlaneView.css";

interface Props {
  tpSize: number;
  dpSize: number;
  ppSize: number;
  epSize: number;
  enableDpAttention: boolean;
  modelConfig: ModelConfig | null;
  schedulePolicy: SchedulePolicy;
  chunkedPrefillSize: number;
  disableRadixCache: boolean;
  specAlgorithm: SpecAlgorithm;
  specNumDraftTokens: number;
  cudaGraphMaxBs: number;
  disableCudaGraph: boolean;
}

/* ── Block color presets (same palette as ArchitectureDiagram) ── */

const S = {
  input:     { fill: "#dbeafe", stroke: "#93c5fd", text: "#1e40af" },
  norm:      { fill: "#fef3c7", stroke: "#fcd34d", text: "#92400e" },
  attention: { fill: "#d1fae5", stroke: "#6ee7b7", text: "#065f46" },
  mlp:       { fill: "#ede9fe", stroke: "#c4b5fd", text: "#5b21b6" },
  comm:      { fill: "#fce7f3", stroke: "#f9a8d4", text: "#9d174d" },
  moe:       { fill: "#cffafe", stroke: "#67e8f9", text: "#155e75" },
  router:    { fill: "#fff7ed", stroke: "#fdba74", text: "#9a3412" },
} as const;

const POLICY_LABELS: Record<SchedulePolicy, string> = {
  fcfs: "FCFS",
  lpm: "LPM",
  random: "Random",
  "dfs-weight": "DFS-Weight",
};

const SPEC_LABELS: Record<SpecAlgorithm, string> = {
  none: "",
  eagle: "EAGLE",
  eagle3: "EAGLE-3",
  nextn: "NextN",
  ngram: "N-Gram",
};

/* ── Helpers ── */

function VArrow({ label }: { label?: string }) {
  return (
    <div className="v-connector">
      <div className="v-connector-line" />
      <div className="v-connector-head" />
      {label && <span className="v-connector-label">{label}</span>}
    </div>
  );
}

interface BlockProps {
  title: string;
  subtitle?: string;
  style: { fill: string; stroke: string; text: string };
  disabled?: boolean;
}

function CpBlock({ title, subtitle, style, disabled }: BlockProps) {
  return (
    <div
      className={`cp-block${disabled ? " cp-block-disabled" : ""}`}
      style={{
        backgroundColor: style.fill,
        borderColor: style.stroke,
        color: style.text,
      }}
    >
      <div className="cp-block-name">{title}</div>
      {subtitle && <div className="cp-block-sub">{subtitle}</div>}
    </div>
  );
}

function CpSubBlock({ title, subtitle, style, disabled }: BlockProps) {
  return (
    <div
      className={`cp-sub-block${disabled ? " cp-block-disabled" : ""}`}
      style={{
        backgroundColor: style.fill,
        borderColor: style.stroke,
        color: style.text,
      }}
    >
      <div className="cp-sub-block-name">{title}</div>
      {subtitle && <div className="cp-sub-block-sub">{subtitle}</div>}
    </div>
  );
}

/* ── Scheduler section box (reusable) ── */

function SchedulerSection({
  schedulePolicy,
  chunkedPrefillSize,
  disableRadixCache,
}: {
  schedulePolicy: SchedulePolicy;
  chunkedPrefillSize: number;
  disableRadixCache: boolean;
}) {
  return (
    <div
      className="section-box cp-section"
      style={{
        borderColor: S.moe.stroke,
        background: "rgba(207,250,254,0.10)",
      }}
    >
      <div className="section-box-title" style={{ color: S.moe.text }}>
        Scheduler
      </div>
      <div className="section-box-body">
        <div className="cp-sub-blocks">
          <CpSubBlock
            title="Waiting Queue"
            subtitle={`policy: ${POLICY_LABELS[schedulePolicy]}`}
            style={S.input}
          />
          <CpSubBlock
            title="Running Batch"
            subtitle="continuous batching"
            style={S.attention}
          />
        </div>

        <VArrow />

        <CpSubBlock
          title="RadixAttention Tree"
          subtitle="prefix caching · automatic reuse"
          style={S.moe}
          disabled={disableRadixCache}
        />

        <VArrow />

        <CpSubBlock
          title="Chunked Prefill"
          subtitle={`chunk size: ${chunkedPrefillSize.toLocaleString()}`}
          style={S.mlp}
        />
      </div>
    </div>
  );
}

/* ── Worker + Runner + GPU stack (reusable) ── */

function WorkerStack({
  tpSize,
  specAlgorithm,
  specNumDraftTokens,
  cudaGraphMaxBs,
  disableCudaGraph,
}: {
  tpSize: number;
  specAlgorithm: SpecAlgorithm;
  specNumDraftTokens: number;
  cudaGraphMaxBs: number;
  disableCudaGraph: boolean;
}) {
  const cudaLabel = disableCudaGraph
    ? "No CUDA Graph"
    : `CUDA Graph (bs ≤ ${cudaGraphMaxBs})`;

  return (
    <>
      {/* TpModelWorker */}
      <CpBlock
        title={`TpModelWorker${tpSize > 1 ? ` × ${tpSize}` : ""}`}
        subtitle={tpSize > 1 ? "one worker per TP rank · NCCL" : "single worker"}
        style={S.attention}
      />

      <VArrow label="forward" />

      {/* ModelRunner */}
      <CpBlock
        title="ModelRunner"
        subtitle={`forward pass · attention backend · sampling · ${cudaLabel}`}
        style={S.mlp}
      />

      {/* Speculative Decoding block */}
      {specAlgorithm !== "none" && (
        <>
          <VArrow label="speculate" />
          <CpBlock
            title="Speculative Decoding"
            subtitle={`${SPEC_LABELS[specAlgorithm]} · ${specNumDraftTokens} draft tokens`}
            style={S.router}
          />
        </>
      )}

      <VArrow />

      {/* GPU */}
      <CpBlock
        title={`GPU${tpSize > 1 ? ` × ${tpSize}` : ""}`}
        subtitle="CUDA kernels · FlashInfer / Triton"
        style={S.comm}
      />
    </>
  );
}

/* ── Single DP group column (Scheduler → Workers → Runner → GPU) ── */

function DpGroupColumn({
  tpSize,
  label,
  schedulePolicy,
  chunkedPrefillSize,
  disableRadixCache,
  specAlgorithm,
  specNumDraftTokens,
  cudaGraphMaxBs,
  disableCudaGraph,
}: {
  tpSize: number;
  label?: string;
  schedulePolicy: SchedulePolicy;
  chunkedPrefillSize: number;
  disableRadixCache: boolean;
  specAlgorithm: SpecAlgorithm;
  specNumDraftTokens: number;
  cudaGraphMaxBs: number;
  disableCudaGraph: boolean;
}) {
  return (
    <div className="cp-dp-group">
      {label && <div className="cp-dp-group-label">{label}</div>}

      <SchedulerSection
        schedulePolicy={schedulePolicy}
        chunkedPrefillSize={chunkedPrefillSize}
        disableRadixCache={disableRadixCache}
      />

      <VArrow label="schedule batch" />

      <WorkerStack
        tpSize={tpSize}
        specAlgorithm={specAlgorithm}
        specNumDraftTokens={specNumDraftTokens}
        cudaGraphMaxBs={cudaGraphMaxBs}
        disableCudaGraph={disableCudaGraph}
      />
    </div>
  );
}

/* ── Main Component ── */

export function ControlPlaneView({
  tpSize,
  dpSize,
  ppSize,
  epSize,
  enableDpAttention,
  modelConfig,
  schedulePolicy,
  chunkedPrefillSize,
  disableRadixCache,
  specAlgorithm,
  specNumDraftTokens,
  cudaGraphMaxBs,
  disableCudaGraph,
}: Props) {
  const useDpAttn = enableDpAttention && dpSize > 1;
  const hasMoe = modelConfig?.n_routed_experts != null;
  const nExperts = modelConfig?.n_routed_experts ?? 0;

  // For traditional DP (non dp_attention), show side-by-side groups
  // For large DP, collapse into a single representative group
  const showCollapsedDp = !useDpAttn && dpSize > 4;
  const dpGroupCount = showCollapsedDp ? 1 : dpSize;

  return (
    <div className="control-plane-view arch-vertical">
      {/* HTTP API */}
      <CpBlock title="HTTP API / Client" subtitle="REST · OpenAI-compatible" style={S.input} />

      <VArrow label="requests" />

      {/* TokenizerManager */}
      <CpBlock title="TokenizerManager" subtitle="tokenize · detokenize · stream" style={S.norm} />

      {useDpAttn ? (
        /* ── DP Attention mode ── */
        <>
          <VArrow label="token ids" />

          <CpBlock
            title="DataParallelController"
            subtitle={`dp_attention · ${dpSize} schedulers`}
            style={S.router}
          />

          <VArrow label="route to scheduler" />

          {/* Multiple schedulers side by side */}
          <div className="cp-dp-fan-out">
            {Array.from({ length: dpSize }, (_, i) => (
              <div key={i} className="cp-dp-group">
                <div className="cp-dp-group-label">Scheduler {i}</div>
                <SchedulerSection
                  schedulePolicy={schedulePolicy}
                  chunkedPrefillSize={chunkedPrefillSize}
                  disableRadixCache={disableRadixCache}
                />
              </div>
            ))}
          </div>

          <VArrow label="all join shared worker pool" />

          {/* Shared worker pool */}
          <WorkerStack
            tpSize={tpSize}
            specAlgorithm={specAlgorithm}
            specNumDraftTokens={specNumDraftTokens}
            cudaGraphMaxBs={cudaGraphMaxBs}
            disableCudaGraph={disableCudaGraph}
          />

          {/* DP Attention annotation */}
          <div className="cp-dp-attn-note">
            Per-layer: Attention (DP sub-groups) → gather → MLP (full TP) → scatter
          </div>
        </>
      ) : dpSize > 1 ? (
        /* ── Traditional DP mode ── */
        <>
          <VArrow label="token ids" />

          <CpBlock
            title="DataParallelRouter"
            subtitle="round-robin / load balance"
            style={S.router}
          />

          <VArrow label="route to DP group" />

          <div className="cp-dp-fan-out">
            {showCollapsedDp ? (
              <DpGroupColumn
                tpSize={tpSize}
                label={`DP Group 0 … ${dpSize - 1}  (× ${dpSize})`}
                schedulePolicy={schedulePolicy}
                chunkedPrefillSize={chunkedPrefillSize}
                disableRadixCache={disableRadixCache}
                specAlgorithm={specAlgorithm}
                specNumDraftTokens={specNumDraftTokens}
                cudaGraphMaxBs={cudaGraphMaxBs}
                disableCudaGraph={disableCudaGraph}
              />
            ) : (
              Array.from({ length: dpGroupCount }, (_, i) => (
                <DpGroupColumn
                  key={i}
                  tpSize={tpSize}
                  label={`DP Group ${i}`}
                  schedulePolicy={schedulePolicy}
                  chunkedPrefillSize={chunkedPrefillSize}
                  disableRadixCache={disableRadixCache}
                  specAlgorithm={specAlgorithm}
                  specNumDraftTokens={specNumDraftTokens}
                  cudaGraphMaxBs={cudaGraphMaxBs}
                  disableCudaGraph={disableCudaGraph}
                />
              ))
            )}
          </div>
        </>
      ) : (
        /* ── Single (no DP) mode ── */
        <>
          <VArrow label="token ids" />

          <SchedulerSection
            schedulePolicy={schedulePolicy}
            chunkedPrefillSize={chunkedPrefillSize}
            disableRadixCache={disableRadixCache}
          />

          <VArrow label="schedule batch" />

          <WorkerStack
            tpSize={tpSize}
            specAlgorithm={specAlgorithm}
            specNumDraftTokens={specNumDraftTokens}
            cudaGraphMaxBs={cudaGraphMaxBs}
            disableCudaGraph={disableCudaGraph}
          />
        </>
      )}

      {/* EP annotation */}
      {epSize > 1 && hasMoe && (
        <div className="cp-ep-note">
          <div className="cp-ep-note-title">Expert Parallelism (EP={epSize})</div>
          <div className="cp-ep-note-detail">
            MoE layers: {nExperts} experts → {Math.floor(nExperts / epSize)} per EP rank · AllToAll dispatch
          </div>
        </div>
      )}

      {/* PP annotation */}
      {ppSize > 1 && (
        <div className="cp-dp-attn-note">
          Pipeline Parallelism: {ppSize} stages × {tpSize} TP = {tpSize * ppSize} GPUs
        </div>
      )}
    </div>
  );
}
