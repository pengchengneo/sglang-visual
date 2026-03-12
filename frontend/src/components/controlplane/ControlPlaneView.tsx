/**
 * Control Plane conceptual architecture diagram.
 *
 * Shows SGLang runtime layers: HTTP API → TokenizerManager → Scheduler →
 * TpModelWorker → ModelRunner → GPU.
 */

interface Props {
  tpSize: number;
}

/* ── Block color presets (same palette as ArchitectureDiagram) ── */

const S = {
  input:     { fill: "#dbeafe", stroke: "#93c5fd", text: "#1e40af" },
  norm:      { fill: "#fef3c7", stroke: "#fcd34d", text: "#92400e" },
  attention: { fill: "#d1fae5", stroke: "#6ee7b7", text: "#065f46" },
  mlp:       { fill: "#ede9fe", stroke: "#c4b5fd", text: "#5b21b6" },
  comm:      { fill: "#fce7f3", stroke: "#f9a8d4", text: "#9d174d" },
  moe:       { fill: "#cffafe", stroke: "#67e8f9", text: "#155e75" },
} as const;

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
}

function CpBlock({ title, subtitle, style }: BlockProps) {
  return (
    <div
      className="cp-block"
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

function CpSubBlock({ title, subtitle, style }: BlockProps) {
  return (
    <div
      className="cp-sub-block"
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

/* ── Main Component ── */

export function ControlPlaneView({ tpSize }: Props) {
  return (
    <div className="control-plane-view arch-vertical">
      {/* HTTP API */}
      <CpBlock title="HTTP API / Client" subtitle="REST · OpenAI-compatible" style={S.input} />

      <VArrow label="requests" />

      {/* TokenizerManager */}
      <CpBlock title="TokenizerManager" subtitle="tokenize · detokenize · stream" style={S.norm} />

      <VArrow label="token ids" />

      {/* Scheduler — section box */}
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
              subtitle="priority · FCFS"
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
          />

          <VArrow />

          <CpSubBlock
            title="Chunked Prefill"
            subtitle="interleave prefill & decode"
            style={S.mlp}
          />
        </div>
      </div>

      <VArrow label="schedule batch" />

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
        subtitle="forward pass · attention backend · sampling"
        style={S.mlp}
      />

      <VArrow />

      {/* GPU */}
      <CpBlock
        title={`GPU${tpSize > 1 ? ` × ${tpSize}` : ""}`}
        subtitle="CUDA kernels · FlashInfer / Triton"
        style={S.comm}
      />
    </div>
  );
}
