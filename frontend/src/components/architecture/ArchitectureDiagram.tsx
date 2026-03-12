/**
 * Full transformer architecture diagram.
 * Generates a flowchart-like SVG showing the model's data path,
 * with clickable weight-operator blocks that link to partition viz.
 */

import { useMemo } from "react";
import type { ModelArchitecture, ModelConfig, Operator, Layer } from "../../types/model";

/* ── Block type → visual style ── */

const STYLES: Record<string, { fill: string; stroke: string; text: string }> = {
  input:     { fill: "#dbeafe", stroke: "#93c5fd", text: "#1e40af" },
  norm:      { fill: "#fef3c7", stroke: "#fcd34d", text: "#92400e" },
  attention: { fill: "#d1fae5", stroke: "#6ee7b7", text: "#065f46" },
  comm:      { fill: "#fce7f3", stroke: "#f9a8d4", text: "#9d174d" },
  residual:  { fill: "#f3f4f6", stroke: "#d1d5db", text: "#374151" },
  mlp:       { fill: "#ede9fe", stroke: "#c4b5fd", text: "#5b21b6" },
  moe:       { fill: "#cffafe", stroke: "#67e8f9", text: "#155e75" },
};

/* ── Data structures ── */

interface Block {
  id: string;
  label: string;
  sublabel?: string;
  type: string;
  operatorName?: string; // links to model operator → makes block clickable
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Arrow {
  x1: number; y1: number;
  x2: number; y2: number;
  color?: string;
  dashed?: boolean;
}

interface TextLabel {
  x: number; y: number;
  text: string;
  color: string;
  fontSize?: number;
  anchor?: string;
}

interface BoxOutline {
  x: number; y: number;
  w: number; h: number;
  stroke: string;
  label: string;
  labelColor: string;
}

interface LayoutResult {
  blocks: Block[];
  arrows: Arrow[];
  labels: TextLabel[];
  boxes: BoxOutline[];
  width: number;
  height: number;
}

/* ── Layout helpers ── */

const BH = 40;
const GAP = 14;

function layoutRow(
  defs: Omit<Block, "x" | "y" | "h">[],
  y: number,
  startX: number,
): Block[] {
  let x = startX;
  return defs.map((d) => {
    const b: Block = { ...d, x, y, h: BH };
    x += d.w + GAP;
    return b;
  });
}

function rowArrows(blocks: Block[], color?: string): Arrow[] {
  const out: Arrow[] = [];
  for (let i = 0; i < blocks.length - 1; i++) {
    out.push({
      x1: blocks[i].x + blocks[i].w,
      y1: blocks[i].y + BH / 2,
      x2: blocks[i + 1].x,
      y2: blocks[i + 1].y + BH / 2,
      color,
    });
  }
  return out;
}

function residualPath(
  fromX: number,
  toBlock: Block,
  y: number,
): Arrow {
  return {
    x1: fromX,
    y1: y,
    x2: toBlock.x + toBlock.w / 2,
    y2: y,
    color: "#9ca3af",
    dashed: true,
  };
}

/* ── Llama layout ── */

function llamaLayout(config: ModelConfig, numLayers: number): LayoutResult {
  const PAD = 24;
  const blocks: Block[] = [];
  const arrows: Arrow[] = [];
  const labels: TextLabel[] = [];
  const boxes: BoxOutline[] = [];

  // ── Row 0: Embedding ──
  const r0 = layoutRow([
    { id: "embed", label: "Embedding", sublabel: "VocabParallelEmbedding", type: "input", operatorName: "__embedding__", w: 180 },
    { id: "embed_ar", label: "AllReduce (TP)", type: "comm", w: 130 },
  ], 30, PAD);
  blocks.push(...r0);
  arrows.push(...rowArrows(r0));

  // ── Decoder layer box ──
  const DY = 100;
  const DP = 24; // decoder inner padding
  const leftX = PAD + DP;

  // Attention section label
  labels.push({ x: leftX, y: DY + 20, text: "Attention (MHA)", color: "#065f46", fontSize: 12 });

  // Attention row
  const attnY = DY + 36;
  const r1 = layoutRow([
    { id: "in_norm", label: "RMSNorm", sublabel: "input_layernorm", type: "norm", w: 90 },
    { id: "qkv", label: "QKV Proj", type: "attention", operatorName: "qkv_proj", w: 84 },
    { id: "rope", label: "RoPE", type: "attention", w: 56 },
    { id: "qk", label: "Q·K\u1d40/\u221ad", type: "attention", w: 78 },
    { id: "softmax", label: "Softmax", type: "attention", w: 68 },
    { id: "sv", label: "Score·V", type: "attention", w: 66 },
    { id: "o_proj", label: "O Proj", type: "attention", operatorName: "o_proj", w: 70 },
    { id: "attn_ar", label: "AllReduce (TP)", type: "comm", w: 120 },
    { id: "attn_res", label: "Residual Add (+)", type: "residual", w: 120 },
  ], attnY, leftX);
  blocks.push(...r1);
  arrows.push(...rowArrows(r1));

  // Residual skip (dashed line above attention row)
  const resSkipY1 = attnY - 10;
  arrows.push({ x1: leftX - 8, y1: resSkipY1, x2: r1[r1.length - 1].x + r1[r1.length - 1].w / 2, y2: resSkipY1, color: "#9ca3af", dashed: true });
  arrows.push({ x1: r1[r1.length - 1].x + r1[r1.length - 1].w / 2, y1: resSkipY1, x2: r1[r1.length - 1].x + r1[r1.length - 1].w / 2, y2: r1[r1.length - 1].y, color: "#9ca3af", dashed: true });
  labels.push({ x: (leftX + r1[r1.length - 1].x + r1[r1.length - 1].w / 2) / 2, y: resSkipY1 - 4, text: "residual (skip)", color: "#9ca3af", fontSize: 10, anchor: "middle" });

  // MLP section
  const mlpLabelY = attnY + BH + 30;
  labels.push({ x: leftX, y: mlpLabelY, text: "MLP", color: "#5b21b6", fontSize: 12 });

  const mlpY = mlpLabelY + 16;
  const r2 = layoutRow([
    { id: "mlp_norm", label: "RMSNorm", sublabel: "post_attn_norm", type: "norm", w: 90 },
    { id: "gate_up", label: "gate_up_proj", type: "mlp", operatorName: "gate_up_proj", w: 105 },
    { id: "silu", label: "SiLU", type: "mlp", w: 50 },
    { id: "down", label: "down_proj", type: "mlp", operatorName: "down_proj", w: 90 },
    { id: "mlp_ar", label: "AllReduce (TP)", type: "comm", w: 120 },
    { id: "mlp_res", label: "Residual Add (+)", type: "residual", w: 120 },
  ], mlpY, leftX);
  blocks.push(...r2);
  arrows.push(...rowArrows(r2));

  // Residual skip for MLP
  const resSkipY2 = mlpY - 10;
  arrows.push({ x1: leftX - 8, y1: resSkipY2, x2: r2[r2.length - 1].x + r2[r2.length - 1].w / 2, y2: resSkipY2, color: "#9ca3af", dashed: true });
  arrows.push({ x1: r2[r2.length - 1].x + r2[r2.length - 1].w / 2, y1: resSkipY2, x2: r2[r2.length - 1].x + r2[r2.length - 1].w / 2, y2: r2[r2.length - 1].y, color: "#9ca3af", dashed: true });
  labels.push({ x: (leftX + r2[r2.length - 1].x + r2[r2.length - 1].w / 2) / 2, y: resSkipY2 - 4, text: "residual (skip)", color: "#9ca3af", fontSize: 10, anchor: "middle" });

  // Vertical connector: attention res → MLP norm
  const attnResBlock = r1[r1.length - 1];
  const mlpNormBlock = r2[0];
  arrows.push({
    x1: attnResBlock.x + 20,
    y1: attnResBlock.y + BH,
    x2: mlpNormBlock.x + 20,
    y2: mlpNormBlock.y - 20,
    color: "#9ca3af",
  });

  // Decoder box
  const decoderRight = Math.max(...r1.map(b => b.x + b.w), ...r2.map(b => b.x + b.w)) + DP;
  const decoderBottom = mlpY + BH + DP;
  boxes.push({
    x: PAD, y: DY,
    w: decoderRight - PAD, h: decoderBottom - DY,
    stroke: "#3b82f6",
    label: `Decoder Layer \u00d7 ${numLayers}`,
    labelColor: "#3b82f6",
  });

  // ── Bottom row ──
  const bottomY = decoderBottom + 40;
  const r3 = layoutRow([
    { id: "final_norm", label: "Final RMSNorm", type: "norm", w: 115 },
    { id: "lm_head", label: "LM Head", sublabel: "ParallelLMHead", type: "input", operatorName: "__lm_head__", w: 115 },
    { id: "logits", label: "LogitsProcessor", type: "input", w: 120 },
    { id: "sample", label: "Sample", type: "input", w: 72 },
  ], bottomY, PAD);
  blocks.push(...r3);
  arrows.push(...rowArrows(r3));

  // Connector from decoder bottom to final norm
  arrows.push({
    x1: PAD + DP,
    y1: decoderBottom,
    x2: PAD + DP,
    y2: bottomY + BH / 2,
    color: "#9ca3af",
  });

  const width = decoderRight + PAD;
  const height = bottomY + BH + PAD;

  return { blocks, arrows, labels, boxes, width, height };
}

/* ── DeepSeek V2 layout ── */

function deepseekLayout(config: ModelConfig, numLayers: number): LayoutResult {
  const PAD = 24;
  const blocks: Block[] = [];
  const arrows: Arrow[] = [];
  const labels: TextLabel[] = [];
  const boxes: BoxOutline[] = [];
  const firstK = config.first_k_dense_replace ?? 0;

  // ── Row 0: Embedding ──
  const r0 = layoutRow([
    { id: "embed", label: "Embedding", sublabel: "VocabParallelEmbedding", type: "input", operatorName: "__embedding__", w: 180 },
    { id: "embed_ar", label: "AllReduce (TP)", type: "comm", w: 130 },
  ], 30, PAD);
  blocks.push(...r0);
  arrows.push(...rowArrows(r0));

  // ── Decoder layer ──
  const DY = 100;
  const DP = 24;
  const leftX = PAD + DP;

  // Attention (MLA) label
  labels.push({ x: leftX, y: DY + 20, text: "Attention (MLA)", color: "#065f46", fontSize: 12 });

  const attnY = DY + 36;
  const r1 = layoutRow([
    { id: "in_norm", label: "RMSNorm", sublabel: "input_layernorm", type: "norm", w: 86 },
    { id: "qkv_a", label: "QKV_A Proj", type: "attention", operatorName: "fused_qkv_a_proj_with_mqa", w: 94 },
    { id: "q_b", label: "Q_B Proj", type: "attention", operatorName: "q_b_proj", w: 72 },
    { id: "kv_b", label: "KV_B Proj", type: "attention", operatorName: "kv_b_proj", w: 76 },
    { id: "rope", label: "RoPE", type: "attention", w: 54 },
    { id: "qk", label: "Q·K\u1d40/\u221ad", type: "attention", w: 76 },
    { id: "softmax", label: "Softmax", type: "attention", w: 66 },
    { id: "sv", label: "Score·V", type: "attention", w: 64 },
    { id: "o_proj", label: "O Proj", type: "attention", operatorName: "o_proj", w: 68 },
    { id: "attn_ar", label: "AllReduce (TP)", type: "comm", w: 118 },
    { id: "attn_res", label: "Residual Add (+)", type: "residual", w: 116 },
  ], attnY, leftX);
  blocks.push(...r1);
  arrows.push(...rowArrows(r1));

  // Residual skip for attention
  const resY1 = attnY - 10;
  const attnRes = r1[r1.length - 1];
  arrows.push({ x1: leftX - 8, y1: resY1, x2: attnRes.x + attnRes.w / 2, y2: resY1, color: "#9ca3af", dashed: true });
  arrows.push({ x1: attnRes.x + attnRes.w / 2, y1: resY1, x2: attnRes.x + attnRes.w / 2, y2: attnRes.y, color: "#9ca3af", dashed: true });
  labels.push({ x: (leftX + attnRes.x) / 2, y: resY1 - 4, text: "residual (skip)", color: "#9ca3af", fontSize: 10, anchor: "middle" });

  // ── Feedforward section ──
  const ffLabelY = attnY + BH + 28;

  // Dense MLP path
  const denseLabel = `layer < ${firstK} (dense)`;
  const denseY = ffLabelY + 18;
  labels.push({ x: leftX + 100, y: denseY - 6, text: denseLabel, color: "#7c3aed", fontSize: 10 });

  const rDense = layoutRow([
    { id: "ff_norm", label: "RMSNorm", sublabel: "post_attn_norm", type: "norm", w: 86 },
    { id: "d_gate_up", label: "gate_up_proj", type: "mlp", operatorName: "gate_up_proj", w: 102 },
    { id: "d_silu", label: "SiLU", type: "mlp", w: 48 },
    { id: "d_down", label: "down_proj", type: "mlp", operatorName: "down_proj", w: 86 },
  ], denseY, leftX);
  blocks.push(...rDense);
  arrows.push(...rowArrows(rDense));

  // Dense MLP box
  const denseRight = rDense[rDense.length - 1].x + rDense[rDense.length - 1].w + 8;
  boxes.push({
    x: rDense[1].x - 8, y: denseY - 14,
    w: denseRight - rDense[1].x + 8, h: BH + 20,
    stroke: "#a78bfa",
    label: "",
    labelColor: "#7c3aed",
  });

  // MoE path
  const moeLabel = `layer \u2265 ${firstK} (MoE)`;
  const moeY = denseY + BH + 34;
  labels.push({ x: leftX + 100, y: moeY - 6, text: moeLabel, color: "#0891b2", fontSize: 10 });

  const rMoe = layoutRow([
    { id: "moe_gate", label: "Router", type: "moe", operatorName: "gate", w: 70 },
    { id: "exp_gu", label: "Experts gate_up", sublabel: `${config.n_routed_experts ?? 0} experts`, type: "moe", operatorName: "experts_gate_up", w: 124 },
    { id: "exp_dn", label: "Experts down", type: "moe", operatorName: "experts_down", w: 100 },
    { id: "sh_gu", label: "Shared gate_up", type: "moe", operatorName: "shared_experts_gate_up", w: 112 },
    { id: "sh_dn", label: "Shared down", type: "moe", operatorName: "shared_experts_down", w: 100 },
  ], moeY, leftX + rDense[0].w + GAP);
  blocks.push(...rMoe);
  arrows.push(...rowArrows(rMoe));

  // Arrow from RMSNorm to MoE Router
  arrows.push({
    x1: rDense[0].x + rDense[0].w,
    y1: rDense[0].y + BH / 2,
    x2: rMoe[0].x,
    y2: rMoe[0].y + BH / 2,
    color: "#0891b2",
  });

  // MoE box
  const moeRight = rMoe[rMoe.length - 1].x + rMoe[rMoe.length - 1].w + 8;
  boxes.push({
    x: rMoe[0].x - 8, y: moeY - 14,
    w: moeRight - rMoe[0].x + 8, h: BH + 20,
    stroke: "#67e8f9",
    label: "",
    labelColor: "#0891b2",
  });

  // AllReduce + Residual Add (right of both paths)
  const commX = Math.max(denseRight, moeRight) + GAP + 16;
  const commY = denseY + (moeY - denseY) / 2; // vertically centered

  const ffAr: Block = { id: "ff_ar", label: "AllReduce / AllToAll (TP/EP)", type: "comm", x: commX, y: commY, w: 170, h: BH };
  const ffRes: Block = { id: "ff_res", label: "Residual Add (+)", type: "residual", x: commX + 170 + GAP, y: commY, w: 116, h: BH };
  blocks.push(ffAr, ffRes);

  // Arrows: Dense → AllReduce
  arrows.push({
    x1: denseRight,
    y1: denseY + BH / 2,
    x2: ffAr.x,
    y2: ffAr.y + BH / 2,
    color: "#a78bfa",
  });
  // Arrows: MoE → AllReduce
  arrows.push({
    x1: moeRight,
    y1: moeY + BH / 2,
    x2: ffAr.x,
    y2: ffAr.y + BH / 2,
    color: "#67e8f9",
  });
  // AllReduce → Residual Add
  arrows.push({ x1: ffAr.x + ffAr.w, y1: ffAr.y + BH / 2, x2: ffRes.x, y2: ffRes.y + BH / 2 });

  // Residual skip for feedforward
  const resY2 = denseY - 20;
  arrows.push({ x1: leftX - 8, y1: resY2, x2: ffRes.x + ffRes.w / 2, y2: resY2, color: "#9ca3af", dashed: true });
  arrows.push({ x1: ffRes.x + ffRes.w / 2, y1: resY2, x2: ffRes.x + ffRes.w / 2, y2: ffRes.y, color: "#9ca3af", dashed: true });
  labels.push({ x: (leftX + ffRes.x) / 2, y: resY2 - 4, text: "residual (skip)", color: "#9ca3af", fontSize: 10, anchor: "middle" });

  // Vertical connector: attention → feedforward
  arrows.push({
    x1: attnRes.x + 20,
    y1: attnRes.y + BH,
    x2: rDense[0].x + 20,
    y2: rDense[0].y - 28,
    color: "#9ca3af",
  });

  // Decoder box
  const decoderRight = Math.max(...r1.map(b => b.x + b.w), ffRes.x + ffRes.w) + DP;
  const decoderBottom = moeY + BH + DP + 10;
  boxes.push({
    x: PAD, y: DY,
    w: decoderRight - PAD, h: decoderBottom - DY,
    stroke: "#3b82f6",
    label: `Decoder Layer \u00d7 ${numLayers}`,
    labelColor: "#3b82f6",
  });

  // ── Bottom row ──
  const bottomY = decoderBottom + 40;
  const r3 = layoutRow([
    { id: "final_norm", label: "Final RMSNorm", type: "norm", w: 115 },
    { id: "lm_head", label: "LM Head", sublabel: "ParallelLMHead", type: "input", operatorName: "__lm_head__", w: 115 },
    { id: "logits", label: "LogitsProcessor", type: "input", w: 120 },
    { id: "sample", label: "Sample", type: "input", w: 72 },
  ], bottomY, PAD);
  blocks.push(...r3);
  arrows.push(...rowArrows(r3));

  arrows.push({
    x1: PAD + DP,
    y1: decoderBottom,
    x2: PAD + DP,
    y2: bottomY + BH / 2,
    color: "#9ca3af",
  });

  const width = decoderRight + PAD;
  const height = bottomY + BH + PAD;

  return { blocks, arrows, labels, boxes, width, height };
}

/* ── Legend ── */

const LEGEND: { label: string; type: string }[] = [
  { label: "Input / Output", type: "input" },
  { label: "Normalization", type: "norm" },
  { label: "Attention", type: "attention" },
  { label: "Communication", type: "comm" },
  { label: "Residual Add", type: "residual" },
  { label: "Dense MLP", type: "mlp" },
  { label: "MoE / Experts", type: "moe" },
];

/* ── Component props ── */

interface Props {
  model: ModelArchitecture;
  tpSize: number;
  selectedOp: string | null;
  onSelectOp: (operatorName: string | null, layer: Layer | null) => void;
}

export function ArchitectureDiagram({ model, tpSize, selectedOp, onSelectOp }: Props) {
  const layout = useMemo(() => {
    const family = model.model_family;
    const numLayers = model.config.num_hidden_layers;
    if (family === "deepseek_v2") {
      return deepseekLayout(model.config, numLayers);
    }
    return llamaLayout(model.config, numLayers);
  }, [model]);

  // Find representative layers for operator lookup
  const { denseLayer, moeLayer } = useMemo(() => {
    const dense = model.layers.find(l => l.layer_type === "dense") ?? model.layers[0];
    const moe = model.layers.find(l => l.layer_type === "moe") ?? dense;
    return { denseLayer: dense, moeLayer: moe };
  }, [model.layers]);

  const handleBlockClick = (block: Block) => {
    if (!block.operatorName) return;

    if (selectedOp === block.operatorName) {
      onSelectOp(null, null);
      return;
    }

    // Determine which layer this operator belongs to
    const moOps = ["gate", "experts_gate_up", "experts_down", "shared_experts_gate_up", "shared_experts_down", "moe_output_reduce"];
    const isMoeOp = moOps.includes(block.operatorName);
    const layer = isMoeOp ? moeLayer : denseLayer;

    onSelectOp(block.operatorName, layer);
  };

  // SVG dimensions with some extra for legend
  const svgW = layout.width + 200;
  const svgH = layout.height;

  return (
    <div className="arch-diagram-wrap">
      <svg
        className="arch-diagram-svg"
        viewBox={`0 0 ${svgW} ${svgH}`}
        width="100%"
        preserveAspectRatio="xMinYMin meet"
      >
        {/* Section boxes (decoder layer, MLP/MoE groups) */}
        {layout.boxes.map((box, i) => (
          <g key={`box-${i}`}>
            <rect
              x={box.x} y={box.y}
              width={box.w} height={box.h}
              rx={12} ry={12}
              fill="none"
              stroke={box.stroke}
              strokeWidth={2}
              strokeDasharray="8,4"
              strokeOpacity={0.5}
            />
            {box.label && (
              <text
                x={box.x + 16}
                y={box.y + 18}
                fontSize={14}
                fontWeight={700}
                fill={box.labelColor}
              >
                {box.label}
              </text>
            )}
          </g>
        ))}

        {/* Text labels */}
        {layout.labels.map((lbl, i) => (
          <text
            key={`lbl-${i}`}
            x={lbl.x} y={lbl.y}
            fontSize={lbl.fontSize ?? 12}
            fill={lbl.color}
            fontWeight={500}
            fontStyle="italic"
            textAnchor={lbl.anchor ?? "start"}
          >
            {lbl.text}
          </text>
        ))}

        {/* Arrows */}
        {layout.arrows.map((a, i) => {
          const color = a.color ?? "#6b7280";
          const dx = a.x2 - a.x1;
          const dy = a.y2 - a.y1;
          const isVertical = Math.abs(dy) > Math.abs(dx) * 2;
          const headSize = 4;

          // Arrowhead at end
          let headPoints = "";
          if (isVertical) {
            const dir = dy > 0 ? 1 : -1;
            headPoints = `${a.x2 - headSize},${a.y2 - headSize * dir} ${a.x2},${a.y2} ${a.x2 + headSize},${a.y2 - headSize * dir}`;
          } else {
            const dir = dx > 0 ? 1 : -1;
            headPoints = `${a.x2 - headSize * dir},${a.y2 - headSize} ${a.x2},${a.y2} ${a.x2 - headSize * dir},${a.y2 + headSize}`;
          }

          return (
            <g key={`arr-${i}`}>
              <line
                x1={a.x1} y1={a.y1}
                x2={a.x2} y2={a.y2}
                stroke={color}
                strokeWidth={1.5}
                strokeDasharray={a.dashed ? "4,3" : undefined}
                strokeOpacity={a.dashed ? 0.5 : 0.7}
              />
              {!a.dashed && (
                <polygon points={headPoints} fill={color} fillOpacity={0.7} />
              )}
            </g>
          );
        })}

        {/* Blocks */}
        {layout.blocks.map((block) => {
          const style = STYLES[block.type] ?? STYLES.input;
          const clickable = !!block.operatorName;
          const isSelected = selectedOp === block.operatorName;

          return (
            <g
              key={block.id}
              className={clickable ? "arch-block-clickable" : "arch-block"}
              onClick={clickable ? () => handleBlockClick(block) : undefined}
              style={clickable ? { cursor: "pointer" } : undefined}
            >
              <rect
                x={block.x} y={block.y}
                width={block.w} height={block.h}
                rx={8} ry={8}
                fill={style.fill}
                stroke={isSelected ? "#6366f1" : style.stroke}
                strokeWidth={isSelected ? 2.5 : 1.5}
              />
              {/* Selection glow */}
              {isSelected && (
                <rect
                  x={block.x - 3} y={block.y - 3}
                  width={block.w + 6} height={block.h + 6}
                  rx={11} ry={11}
                  fill="none"
                  stroke="#6366f1"
                  strokeWidth={1}
                  strokeOpacity={0.3}
                />
              )}
              {/* Label */}
              <text
                x={block.x + block.w / 2}
                y={block.y + (block.sublabel ? block.h / 2 - 6 : block.h / 2)}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={11}
                fontWeight={600}
                fill={style.text}
              >
                {block.label}
              </text>
              {/* Sublabel */}
              {block.sublabel && (
                <text
                  x={block.x + block.w / 2}
                  y={block.y + block.h / 2 + 7}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={9}
                  fill={style.text}
                  fillOpacity={0.6}
                >
                  ({block.sublabel})
                </text>
              )}
              {/* Clickable indicator dot */}
              {clickable && !isSelected && (
                <circle
                  cx={block.x + block.w - 8}
                  cy={block.y + 8}
                  r={3}
                  fill="#6366f1"
                  fillOpacity={0.4}
                />
              )}
            </g>
          );
        })}

        {/* Legend */}
        {LEGEND.map((item, i) => {
          const style = STYLES[item.type];
          const lx = layout.width + 20;
          const ly = 200 + i * 28;
          return (
            <g key={item.type}>
              {i === 0 && (
                <text x={lx} y={ly - 20} fontSize={13} fontWeight={700} fill="#374151">
                  Legend
                </text>
              )}
              <rect x={lx} y={ly} width={16} height={16} rx={3} fill={style.fill} stroke={style.stroke} strokeWidth={1.5} />
              <text x={lx + 22} y={ly + 12} fontSize={11} fill="#6b7280">{item.label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
