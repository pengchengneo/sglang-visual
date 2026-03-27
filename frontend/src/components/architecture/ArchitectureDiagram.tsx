/**
 * Full transformer architecture diagram — vertical layout.
 * Renders a top-to-bottom flowchart with expandable TP partition
 * visualization inline on each weight operator.
 */

import { useMemo } from "react";
import type {
  ModelArchitecture,
  ModelConfig,
  Operator,
  Layer,
} from "../../types/model";
import {
  recomputeOperatorTpShape,
  getRankColor,
  shapeToParams,
  formatParams,
  computePpStageRanges,
  computePerRankParamsForPpStage,
  type PpStageRange,
} from "../../utils/tpMath";
import { getStrategyColor, getStrategyLabel } from "../../utils/layoutMath";
import "./ArchitectureDiagram.css";

/* ── Block type → visual style ── */

const STYLES: Record<
  string,
  { fill: string; stroke: string; text: string }
> = {
  input: { fill: "#dbeafe", stroke: "#93c5fd", text: "#1e40af" },
  norm: { fill: "#fef3c7", stroke: "#fcd34d", text: "#92400e" },
  attention: { fill: "#d1fae5", stroke: "#6ee7b7", text: "#065f46" },
  comm: { fill: "#fce7f3", stroke: "#f9a8d4", text: "#9d174d" },
  residual: { fill: "#f3f4f6", stroke: "#d1d5db", text: "#374151" },
  mlp: { fill: "#ede9fe", stroke: "#c4b5fd", text: "#5b21b6" },
  moe: { fill: "#cffafe", stroke: "#67e8f9", text: "#155e75" },
};

const SECTION_COLORS: Record<
  string,
  { border: string; bg: string; label: string }
> = {
  attention: { border: "#6ee7b7", bg: "rgba(209,250,229,0.15)", label: "#065f46" },
  mlp: { border: "#c4b5fd", bg: "rgba(237,233,254,0.15)", label: "#5b21b6" },
  moe: { border: "#67e8f9", bg: "rgba(207,250,254,0.15)", label: "#155e75" },
  dense_mlp: { border: "#a78bfa", bg: "rgba(237,233,254,0.15)", label: "#7c3aed" },
};

/* ── PP Stage colors ── */

const PP_STAGE_COLORS = [
  { border: "#3b82f6", bg: "rgba(59, 130, 246, 0.04)", label: "#2563eb" },
  { border: "#8b5cf6", bg: "rgba(139, 92, 246, 0.04)", label: "#7c3aed" },
  { border: "#ec4899", bg: "rgba(236, 72, 153, 0.04)", label: "#db2777" },
  { border: "#f97316", bg: "rgba(249, 115, 22, 0.04)", label: "#ea580c" },
  { border: "#14b8a6", bg: "rgba(20, 184, 166, 0.04)", label: "#0d9488" },
  { border: "#6366f1", bg: "rgba(99, 102, 241, 0.04)", label: "#4f46e5" },
  { border: "#84cc16", bg: "rgba(132, 204, 22, 0.04)", label: "#65a30d" },
  { border: "#ef4444", bg: "rgba(239, 68, 68, 0.04)", label: "#dc2626" },
];

function getPpStageColor(stage: number) {
  return PP_STAGE_COLORS[stage % PP_STAGE_COLORS.length];
}

/* ── PP Stage wrapper box ── */

function PpStageBox({
  stage,
  range,
  paramsLabel,
  children,
}: {
  stage: number;
  range: PpStageRange;
  paramsLabel: string;
  children: React.ReactNode;
}) {
  const color = getPpStageColor(stage);
  const endLayer = range.startLayer + range.numLayers - 1;
  return (
    <div
      className="pp-stage-box"
      style={{ borderColor: color.border, backgroundColor: color.bg }}
    >
      <div className="pp-stage-header">
        <span className="pp-stage-title" style={{ color: color.label }}>
          PP Stage {stage}
        </span>
        <span className="pp-stage-info" style={{ color: color.label }}>
          Layers {range.startLayer}–{endLayer} · {paramsLabel}
        </span>
      </div>
      {children}
    </div>
  );
}

/* ── P2P Communication Arrow between PP stages ── */

function P2PArrow() {
  return (
    <div className="pp-p2p-arrow">
      <div className="pp-p2p-line" />
      <span className="pp-p2p-label">P2P Send → Recv</span>
      <div className="pp-p2p-head" />
    </div>
  );
}

/* ── Strategy description ── */

function strategyDescription(strategy: string): string {
  switch (strategy) {
    case "column_parallel":
      return "Column Parallel — output dim split across ranks";
    case "row_parallel":
      return "Row Parallel — input dim split, all-reduce after matmul";
    case "qkv_parallel":
      return "QKV Parallel — Q, K, V heads distributed across ranks";
    case "merged_column_parallel":
      return "Merged Column Parallel — fused sub-matrices, column-split";
    case "replicated":
      return "Replicated — full copy on each rank";
    default:
      return strategy;
  }
}

/* ── Inline TP Partition Visualization ── */

function InlineTpViz({
  op,
  config,
  tpSize,
  epSize,
}: {
  op: Operator;
  config: ModelConfig;
  tpSize: number;
  epSize: number;
}) {
  const strategy = op.partition?.strategy;
  if (
    !strategy ||
    strategy === "replicated" ||
    (tpSize <= 1 && epSize <= 1)
  )
    return null;

  const isExpert = op.full_weight_shape.length === 3;
  const effectiveEp = isExpert ? epSize : 1;
  const tpShape = recomputeOperatorTpShape(op, config, tpSize, effectiveEp);
  const full = op.full_weight_shape;
  const displayTp = isExpert ? tpShape.slice(1) : tpShape;
  const perRankParams = shapeToParams(tpShape);

  const isColumnSplit =
    strategy === "column_parallel" ||
    strategy === "qkv_parallel" ||
    strategy === "merged_column_parallel";

  return (
    <div className="inline-tp-viz">
      {/* Strategy label */}
      <div className="inline-tp-strategy">
        <span
          className="inline-tp-dot"
          style={{ backgroundColor: getStrategyColor(strategy) }}
        />
        <span>{strategyDescription(strategy)}</span>
      </div>

      {/* EP info for expert operators */}
      {isExpert && epSize > 1 && (
        <div className="inline-tp-strategy">
          <span
            className="inline-tp-dot"
            style={{ backgroundColor: "#06b6d4" }}
          />
          <span>EP={epSize}: {full[0]} experts → {tpShape[0]} per EP rank</span>
        </div>
      )}

      {/* Partition slices */}
      <div
        className={`inline-tp-matrix ${isColumnSplit ? "tp-column-split" : "tp-row-split"}`}
      >
        {Array.from({ length: tpSize }, (_, i) => {
          const color = getRankColor(i);
          return (
            <div
              key={i}
              className="inline-tp-slice"
              style={{
                backgroundColor: color + "18",
                borderColor: color,
              }}
            >
              <span className="tp-slice-rank" style={{ color }}>
                R{i}
              </span>
              <span className="tp-slice-shape">
                {isExpert ? `${tpShape[0]}×${displayTp.join(" × ")}` : displayTp.join(" × ")}
              </span>
              <span className="tp-slice-params">
                {formatParams(perRankParams)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Replicated indicator (shown when tp > 1 and strategy is replicated) ── */

function ReplicatedIndicator() {
  return (
    <div className="inline-tp-replicated">
      Replicated — full copy on each rank
    </div>
  );
}

/* ── Vertical Arrow Connector ── */

function VArrow({ label }: { label?: string }) {
  return (
    <div className="v-connector">
      <div className="v-connector-line" />
      <div className="v-connector-head" />
      {label && <span className="v-connector-label">{label}</span>}
    </div>
  );
}

/* ── Horizontal Arrow ── */

function HArrow() {
  return (
    <span className="h-connector">
      <span className="h-connector-line" />
      <span className="h-connector-head" />
    </span>
  );
}

/* ── Operator Block ── */

function OpBlock({
  label,
  sublabel,
  type,
  operatorName,
  op,
  config,
  tpSize,
  epSize,
  isSelected,
  onClick,
  compact,
}: {
  label: string;
  sublabel?: string;
  type: string;
  operatorName?: string;
  op?: Operator;
  config?: ModelConfig;
  tpSize: number;
  epSize: number;
  isSelected: boolean;
  onClick?: () => void;
  compact?: boolean;
}) {
  const style = STYLES[type] ?? STYLES.input;
  const clickable = !!operatorName;
  const hasPartition = op?.partition && op.partition.strategy !== "replicated";
  const showTpViz = isSelected && (tpSize > 1 || epSize > 1) && op && config && hasPartition;
  const showReplicated =
    isSelected &&
    tpSize > 1 &&
    op?.partition?.strategy === "replicated";

  return (
    <div
      className={[
        "op-block",
        clickable ? "op-block-clickable" : "",
        isSelected ? "op-block-selected" : "",
        showTpViz ? "op-block-expanded" : "",
        compact ? "op-block-compact" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      style={{
        backgroundColor: style.fill,
        borderColor: isSelected ? "#6366f1" : style.stroke,
        color: style.text,
      }}
      onClick={clickable ? onClick : undefined}
    >
      <div className="op-block-header">
        <span className="op-block-name">{label}</span>
        {sublabel && (
          <span className="op-block-sublabel">{sublabel}</span>
        )}
        {clickable && !isSelected && (
          <span className="op-block-dot" />
        )}
      </div>

      {/* Shape & strategy meta (only for weight operators) */}
      {op && op.partition && (
        <div className="op-block-meta">
          <span className="op-block-shape">
            {op.full_weight_shape.join("\u00d7")}
          </span>
          <span
            className="op-block-strat"
            style={{ color: getStrategyColor(op.partition.strategy) }}
          >
            {getStrategyLabel(op.partition.strategy)}
          </span>
        </div>
      )}

      {/* Inline TP visualization when selected */}
      {showTpViz && op && config && (
        <InlineTpViz op={op} config={config} tpSize={tpSize} epSize={epSize} />
      )}

      {/* Replicated indicator */}
      {showReplicated && <ReplicatedIndicator />}
    </div>
  );
}

/* ── Section Box (Attention / MLP / MoE) ── */

function SectionBox({
  type,
  title,
  children,
}: {
  type: string;
  title: string;
  children: React.ReactNode;
}) {
  const s = SECTION_COLORS[type] ?? SECTION_COLORS.mlp;
  return (
    <div
      className="section-box"
      style={{ borderColor: s.border, backgroundColor: s.bg }}
    >
      <div className="section-box-title" style={{ color: s.label }}>
        {title}
      </div>
      <div className="section-box-body">{children}</div>
    </div>
  );
}

/* ── Residual Add indicator ── */

function ResidualAdd() {
  return (
    <div className="residual-add">
      <div className="residual-line" />
      <span className="residual-label">+ Residual</span>
      <div className="residual-line" />
    </div>
  );
}

/* ── Component props ── */

interface Props {
  model: ModelArchitecture;
  tpSize: number;
  ppSize: number;
  epSize: number;
  selectedOp: string | null;
  onSelectOp: (operatorName: string | null, layer: Layer | null) => void;
}

/* ── Main component ── */

export function ArchitectureDiagram({
  model,
  tpSize,
  ppSize,
  epSize,
  selectedOp,
  onSelectOp,
}: Props) {
  const { denseLayer, moeLayer } = useMemo(() => {
    const dense =
      model.layers.find((l) => l.layer_type === "dense") ??
      model.layers[0];
    const moe =
      model.layers.find((l) => l.layer_type === "moe") ?? dense;
    return { denseLayer: dense, moeLayer: moe };
  }, [model.layers]);

  const ppStageRanges = useMemo(
    () => computePpStageRanges(model.config.num_hidden_layers, ppSize),
    [model.config.num_hidden_layers, ppSize],
  );

  const ppStageParams = useMemo(
    () => ppStageRanges.map((_, i) =>
      computePerRankParamsForPpStage(model, tpSize, i, ppSize, epSize),
    ),
    [model, tpSize, ppSize, epSize, ppStageRanges],
  );

  const findOp = (layer: Layer, name: string): Operator | undefined =>
    layer.operators.find((o) => o.name === name);

  const handleClick = (operatorName: string, layer: Layer) => {
    if (selectedOp === operatorName) {
      onSelectOp(null, null);
    } else {
      onSelectOp(operatorName, layer);
    }
  };

  /* Helper to render an operator block */
  const renderOp = (
    label: string,
    type: string,
    operatorName?: string,
    layer?: Layer,
    sublabel?: string,
    compact?: boolean,
  ) => {
    const op =
      operatorName && layer
        ? findOp(layer, operatorName)
        : undefined;
    return (
      <OpBlock
        label={label}
        sublabel={sublabel}
        type={type}
        operatorName={operatorName}
        op={op}
        config={model.config}
        tpSize={tpSize}
        epSize={epSize}
        isSelected={selectedOp === operatorName}
        onClick={
          operatorName && layer
            ? () => handleClick(operatorName, layer)
            : undefined
        }
        compact={compact}
      />
    );
  };

  const isLlama = model.model_family !== "deepseek_v2";
  const numLayers = model.config.num_hidden_layers;

  /* ── Render attention section (shared between PP and non-PP modes) ── */
  const renderAttention = () => {
    if (isLlama) {
      return (
        <SectionBox type="attention" title="Self-Attention (MHA)">
          {renderOp("RMSNorm", "norm", undefined, undefined, "input_layernorm")}
          <VArrow />
          {renderOp("QKV Proj", "attention", "qkv_proj", denseLayer)}
          <VArrow />
          <div className="arch-hrow compute-row">
            {renderOp("RoPE", "attention", undefined, undefined, undefined, true)}
            <HArrow />
            {renderOp("Q\u00b7K\u1d40/\u221ad", "attention", undefined, undefined, undefined, true)}
            <HArrow />
            {renderOp("Softmax", "attention", undefined, undefined, undefined, true)}
            <HArrow />
            {renderOp("Score\u00b7V", "attention", undefined, undefined, undefined, true)}
          </div>
          <VArrow />
          {renderOp("O Proj", "attention", "o_proj", denseLayer)}
          <VArrow />
          {renderOp("AllReduce (TP)", "comm")}
        </SectionBox>
      );
    }
    return (
      <SectionBox type="attention" title="Self-Attention (MLA)">
        {renderOp("RMSNorm", "norm", undefined, undefined, "input_layernorm")}
        <VArrow />
        {renderOp("QKV_A Proj", "attention", "fused_qkv_a_proj_with_mqa", denseLayer)}
        <VArrow />
        <div className="arch-hrow">
          {renderOp("Q_B Proj", "attention", "q_b_proj", denseLayer)}
          {renderOp("KV_B Proj", "attention", "kv_b_proj", denseLayer)}
        </div>
        <VArrow />
        <div className="arch-hrow compute-row">
          {renderOp("RoPE", "attention", undefined, undefined, undefined, true)}
          <HArrow />
          {renderOp("Q\u00b7K\u1d40/\u221ad", "attention", undefined, undefined, undefined, true)}
          <HArrow />
          {renderOp("Softmax", "attention", undefined, undefined, undefined, true)}
          <HArrow />
          {renderOp("Score\u00b7V", "attention", undefined, undefined, undefined, true)}
        </div>
        <VArrow />
        {renderOp("O Proj", "attention", "o_proj", denseLayer)}
        <VArrow />
        {renderOp("AllReduce (TP)", "comm")}
      </SectionBox>
    );
  };

  /* ── Render feedforward section ── */
  const renderFeedforward = () => {
    if (isLlama) {
      return (
        <SectionBox type="mlp" title="MLP (Feed-Forward)">
          {renderOp("RMSNorm", "norm", undefined, undefined, "post_attn_norm")}
          <VArrow />
          {renderOp("gate_up_proj", "mlp", "gate_up_proj", denseLayer)}
          <VArrow />
          {renderOp("SiLU", "mlp")}
          <VArrow />
          {renderOp("down_proj", "mlp", "down_proj", denseLayer)}
          <VArrow />
          {renderOp("AllReduce (TP)", "comm")}
        </SectionBox>
      );
    }
    return (
      <>
        {renderOp("RMSNorm", "norm", undefined, undefined, "post_attn_norm")}
        <VArrow />
        <div className="arch-hrow ff-dual-path">
          <SectionBox
            type="dense_mlp"
            title={`Dense MLP (layer < ${model.config.first_k_dense_replace ?? 0})`}
          >
            {renderOp("gate_up_proj", "mlp", "gate_up_proj", denseLayer)}
            <VArrow />
            {renderOp("SiLU", "mlp")}
            <VArrow />
            {renderOp("down_proj", "mlp", "down_proj", denseLayer)}
          </SectionBox>
          <SectionBox
            type="moe"
            title={`MoE (layer \u2265 ${model.config.first_k_dense_replace ?? 0})`}
          >
            {renderOp("Router", "moe", "gate", moeLayer)}
            <VArrow />
            <div className="arch-hrow">
              {renderOp(
                "Experts gate_up",
                "moe",
                "experts_gate_up",
                moeLayer,
                epSize > 1
                  ? `${model.config.n_routed_experts ?? 0} experts → ${Math.floor((model.config.n_routed_experts ?? 0) / epSize)} / EP rank`
                  : `${model.config.n_routed_experts ?? 0} experts`,
              )}
              <HArrow />
              {renderOp("Experts down", "moe", "experts_down", moeLayer)}
            </div>
            <VArrow />
            <div className="arch-hrow">
              {renderOp("Shared gate_up", "moe", "shared_experts_gate_up", moeLayer)}
              <HArrow />
              {renderOp("Shared down", "moe", "shared_experts_down", moeLayer)}
            </div>
          </SectionBox>
        </div>
        <VArrow />
        {renderOp("AllReduce / AllToAll (TP/EP)", "comm")}
      </>
    );
  };

  /* ── Render decoder content (attention + residual + FF + residual) ── */
  const renderDecoderContent = () => (
    <>
      {renderAttention()}
      <ResidualAdd />
      {renderFeedforward()}
      <ResidualAdd />
    </>
  );

  /* ── Render embedding row ── */
  const renderEmbedding = () => (
    <>
      <div className="arch-hrow">
        {renderOp("Embedding", "input", undefined, undefined, "VocabParallelEmbedding")}
        <HArrow />
        {renderOp("AllReduce (TP)", "comm")}
      </div>
      <VArrow />
    </>
  );

  /* ── Render output row ── */
  const renderOutput = () => (
    <>
      <VArrow />
      <div className="arch-hrow">
        {renderOp("Final RMSNorm", "norm")}
        <HArrow />
        {renderOp("LM Head", "input", undefined, undefined, "ParallelLMHead")}
        <HArrow />
        {renderOp("LogitsProcessor", "input")}
        <HArrow />
        {renderOp("Sample", "input")}
      </div>
    </>
  );

  /* ── Compute layer type summary for a PP stage ── */
  const getStageLayerTypes = (range: PpStageRange) => {
    let dense = 0;
    let moe = 0;
    for (let i = 0; i < range.numLayers; i++) {
      const layer = model.layers[range.startLayer + i];
      if (layer?.layer_type === "moe") moe++;
      else dense++;
    }
    return { dense, moe };
  };

  /* ── Legend ── */
  const renderLegend = () => (
    <div className="arch-legend">
      <span className="arch-legend-title">Legend</span>
      {[
        { label: "Input / Output", type: "input" },
        { label: "Normalization", type: "norm" },
        { label: "Attention", type: "attention" },
        { label: "Communication", type: "comm" },
        { label: "Dense MLP", type: "mlp" },
        { label: "MoE / Experts", type: "moe" },
      ].map((item) => {
        const s = STYLES[item.type];
        return (
          <span key={item.type} className="arch-legend-item">
            <span
              className="arch-legend-swatch"
              style={{
                backgroundColor: s.fill,
                borderColor: s.stroke,
              }}
            />
            {item.label}
          </span>
        );
      })}
      <span className="arch-legend-item">
        <span className="arch-legend-dot" />
        Click to view TP partition
      </span>
    </div>
  );

  /* ── PP mode: render stages ── */
  if (ppSize > 1) {
    return (
      <div className="arch-vertical">
        {ppStageRanges.map((range, i) => {
          const layerTypes = getStageLayerTypes(range);
          const isFirstStage = i === 0;
          const isLastStage = i === ppSize - 1;

          return (
            <div key={i}>
              <PpStageBox
                stage={i}
                range={range}
                paramsLabel={formatParams(ppStageParams[i])}
              >
                {/* Embedding only in first stage */}
                {isFirstStage && renderEmbedding()}

                {/* Decoder box for this stage */}
                <div className="decoder-box">
                  <div className="decoder-title">
                    Decoder Layer &times; {range.numLayers}
                    {!isLlama && layerTypes.moe > 0 && (
                      <span className="decoder-layer-types">
                        {layerTypes.dense > 0 && (
                          <span className="layer-type-badge dense">{layerTypes.dense} dense</span>
                        )}
                        <span className="layer-type-badge moe">{layerTypes.moe} MoE</span>
                      </span>
                    )}
                  </div>

                  {/* Show full detail in first stage, compact in others */}
                  {isFirstStage ? (
                    renderDecoderContent()
                  ) : (
                    <div className="pp-stage-compact-content">
                      <span className="pp-stage-compact-label">
                        Self-Attention → {isLlama ? "MLP" : layerTypes.moe > 0 ? "MoE" : "MLP"} → Residual
                      </span>
                      <span className="pp-stage-compact-note">
                        Same structure as Stage 0
                      </span>
                    </div>
                  )}
                </div>

                {/* Output only in last stage */}
                {isLastStage && renderOutput()}
              </PpStageBox>

              {/* P2P communication between stages */}
              {!isLastStage && <P2PArrow />}
            </div>
          );
        })}

        {renderLegend()}
      </div>
    );
  }

  /* ── Normal mode (ppSize <= 1) ── */
  return (
    <div className="arch-vertical">
      {renderEmbedding()}

      <div className="decoder-box">
        <div className="decoder-title">
          Decoder Layer &times; {numLayers}
        </div>
        {renderDecoderContent()}
      </div>

      {renderOutput()}

      {renderLegend()}
    </div>
  );
}
