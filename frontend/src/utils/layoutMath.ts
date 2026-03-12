/**
 * Layout utilities for the pipeline visualization.
 * Strategy colors, operator grouping, layer collapsing, and sizing helpers.
 */

import type { Operator, Layer, PartitionStrategy } from "../types/model";

/* ── Strategy → Color mapping ── */

const STRATEGY_COLORS: Record<string, string> = {
  column_parallel: "#818cf8", // indigo-400
  row_parallel: "#34d399", // emerald-400
  qkv_parallel: "#22d3ee", // cyan-400
  merged_column_parallel: "#a78bfa", // violet-400
  replicated: "#9ca3af", // gray-400
};

export function getStrategyColor(
  strategy: PartitionStrategy | null | undefined,
): string {
  if (!strategy) return STRATEGY_COLORS.replicated;
  return STRATEGY_COLORS[strategy] || STRATEGY_COLORS.replicated;
}

export const COMM_COLOR = "#f87171";

/* ── Sub-block grouping ── */

export type SubBlockType = "attention" | "mlp" | "moe";

export interface SubBlock {
  type: SubBlockType;
  label: string;
  operators: Operator[];
  hasComm: boolean;
}

function categorizeOp(op: Operator): SubBlockType {
  const name = op.name.toLowerCase();
  if (
    name.includes("qkv") ||
    name === "q_proj" ||
    name === "k_proj" ||
    name === "v_proj" ||
    name.includes("q_a_") ||
    name.includes("q_b_") ||
    name.includes("kv_a") ||
    name.includes("kv_b") ||
    name === "o_proj"
  ) {
    return "attention";
  }
  if (
    name.includes("expert") ||
    name.includes("moe_") ||
    op.op_type === "moe_gate"
  ) {
    return "moe";
  }
  return "mlp";
}

export function groupOperatorsIntoSubBlocks(operators: Operator[]): SubBlock[] {
  const groups = new Map<SubBlockType, Operator[]>();

  for (const op of operators) {
    if (op.op_type === "comm") continue;
    const cat = categorizeOp(op);
    if (!groups.has(cat)) groups.set(cat, []);
    groups.get(cat)!.push(op);
  }

  const hasCommOp = operators.some(
    (o) => o.op_type === "comm" && o.comm_after,
  );
  const blocks: SubBlock[] = [];

  if (groups.has("attention")) {
    const ops = groups.get("attention")!;
    blocks.push({
      type: "attention",
      label: "Self-Attention",
      operators: ops,
      hasComm: ops.some((o) => o.comm_after !== null),
    });
  }

  if (groups.has("mlp")) {
    const ops = groups.get("mlp")!;
    blocks.push({
      type: "mlp",
      label: "MLP",
      operators: ops,
      hasComm: ops.some((o) => o.comm_after !== null),
    });
  }

  if (groups.has("moe")) {
    const ops = groups.get("moe")!;
    blocks.push({
      type: "moe",
      label: "MoE Block",
      operators: ops,
      hasComm: ops.some((o) => o.comm_after !== null) || hasCommOp,
    });
  }

  return blocks;
}

/* ── Operator sizing ── */

export function computeOperatorWidth(
  params: number,
  maxParams: number,
  maxWidth: number,
  minWidth: number = 24,
): number {
  if (maxParams === 0) return minWidth;
  const scaled = Math.sqrt(params / maxParams) * maxWidth;
  return Math.max(minWidth, Math.min(scaled, maxWidth));
}

/* ── Layer collapsing ── */

export interface LayerGroup {
  representative: Layer;
  count: number;
  startId: number;
  endId: number;
}

export function collapseLayers(layers: Layer[]): LayerGroup[] {
  if (layers.length === 0) return [];

  const groups: LayerGroup[] = [];
  let current: LayerGroup = {
    representative: layers[0],
    count: 1,
    startId: layers[0].layer_id,
    endId: layers[0].layer_id,
  };

  for (let i = 1; i < layers.length; i++) {
    const prev = current.representative;
    const curr = layers[i];
    const sameType = prev.layer_type === curr.layer_type;
    const sameOps =
      prev.operators.length === curr.operators.length &&
      prev.operators.every((op, j) => op.name === curr.operators[j].name);

    if (sameType && sameOps) {
      current.count++;
      current.endId = curr.layer_id;
    } else {
      groups.push(current);
      current = {
        representative: curr,
        count: 1,
        startId: curr.layer_id,
        endId: curr.layer_id,
      };
    }
  }
  groups.push(current);
  return groups;
}

/* ── Labels ── */

export function getStrategyLabel(
  strategy: PartitionStrategy | null | undefined,
): string {
  switch (strategy) {
    case "column_parallel":
      return "Col \u2225";
    case "row_parallel":
      return "Row \u2225";
    case "qkv_parallel":
      return "QKV \u2225";
    case "merged_column_parallel":
      return "Merged Col \u2225";
    case "replicated":
      return "Replicated";
    default:
      return "\u2014";
  }
}
