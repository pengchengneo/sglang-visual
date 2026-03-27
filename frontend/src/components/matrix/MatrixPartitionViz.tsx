import { useMemo } from "react";
import type { Operator, ModelConfig } from "../../types/model";
import { recomputeOperatorTpShape, getRankColor } from "../../utils/tpMath";
import { getStrategyColor } from "../../utils/layoutMath";
import { GpuSliceView } from "./GpuSliceView";
import "./Matrix.css";

interface Props {
  op: Operator;
  config: ModelConfig;
  tpSize: number;
}

const MATRIX_WIDTH = 340;
const MATRIX_HEIGHT = 200;
const LABEL_MARGIN = 50;

export function MatrixPartitionViz({ op, config, tpSize }: Props) {
  const tpShape = recomputeOperatorTpShape(op, config, tpSize);
  const full = op.full_weight_shape;
  const isExpertTensor = full.length === 3;
  const displayFull = isExpertTensor ? [full[1], full[2]] : full;

  const partitions = useMemo(() => {
    if (
      !op.partition ||
      op.partition.strategy === "replicated" ||
      tpSize === 1
    ) {
      return null;
    }
    return computePartitions(op, tpSize);
  }, [op, tpSize]);

  const strategy = op.partition?.strategy || "replicated";
  const stratColor = getStrategyColor(strategy);

  const isQKV = strategy === "qkv_parallel";
  const qkvRegions = useMemo(() => {
    if (!isQKV) return null;
    return computeQkvRegions(config, tpSize);
  }, [config, tpSize, isQKV]);

  return (
    <div className="matrix-viz">
      <h4>
        Weight Matrix: {op.name}
        {isExpertTensor && (
          <span className="expert-note"> (per expert)</span>
        )}
      </h4>
      <div className="matrix-viz-content">
        {/* Left panel: full matrix partition view */}
        <div className="matrix-panel">
          <div className="matrix-panel-title">Full Matrix</div>
          <svg
            width={MATRIX_WIDTH + LABEL_MARGIN * 2}
            height={MATRIX_HEIGHT + LABEL_MARGIN * 2}
            viewBox={`0 0 ${MATRIX_WIDTH + LABEL_MARGIN * 2} ${MATRIX_HEIGHT + LABEL_MARGIN * 2}`}
          >
            <g transform={`translate(${LABEL_MARGIN}, ${LABEL_MARGIN})`}>
              {/* Full matrix outline */}
              <rect
                x={0}
                y={0}
                width={MATRIX_WIDTH}
                height={MATRIX_HEIGHT}
                fill="#f3f4f6"
                stroke="var(--border)"
                strokeWidth={1.5}
                rx={4}
                ry={4}
              />

              {/* Partition blocks */}
              {partitions &&
                partitions.map((p, i) => (
                  <rect
                    key={i}
                    x={p.x}
                    y={p.y}
                    width={p.w}
                    height={p.h}
                    rx={2}
                    ry={2}
                    fill={getRankColor(p.rank)}
                    fillOpacity={0.25}
                    stroke={getRankColor(p.rank)}
                    strokeWidth={1.5}
                    className="partition-rect"
                  />
                ))}

              {/* QKV sub-regions */}
              {isQKV &&
                qkvRegions &&
                qkvRegions.map((region, i) => (
                  <g key={i}>
                    <line
                      x1={0}
                      y1={region.yEnd}
                      x2={MATRIX_WIDTH}
                      y2={region.yEnd}
                      stroke="var(--text-secondary)"
                      strokeWidth={1}
                      strokeDasharray="4,3"
                      strokeOpacity={0.5}
                    />
                    <text
                      x={MATRIX_WIDTH + 5}
                      y={(region.yStart + region.yEnd) / 2}
                      fontSize={10}
                      fill="var(--text-secondary)"
                      dominantBaseline="middle"
                    >
                      {region.label}
                    </text>
                  </g>
                ))}

              {/* Dimension labels */}
              <text
                x={MATRIX_WIDTH / 2}
                y={-10}
                textAnchor="middle"
                fontSize={11}
                fill="var(--text-secondary)"
              >
                in = {displayFull[1]}
              </text>
              <text
                x={-10}
                y={MATRIX_HEIGHT / 2}
                textAnchor="middle"
                fontSize={11}
                fill="var(--text-secondary)"
                transform={`rotate(-90, -10, ${MATRIX_HEIGHT / 2})`}
              >
                out = {displayFull[0]}
              </text>

              {/* TP shape annotation */}
              {tpSize > 1 && partitions && (
                <text
                  x={MATRIX_WIDTH / 2}
                  y={MATRIX_HEIGHT + 20}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--text-secondary)"
                >
                  Each rank:{" "}
                  {isExpertTensor
                    ? tpShape.slice(1).join(" \u00d7 ")
                    : tpShape.join(" \u00d7 ")}
                </text>
              )}
            </g>
          </svg>

          {/* Legend */}
          {tpSize > 1 && partitions && (
            <div className="rank-legend-inline">
              {Array.from({ length: tpSize }, (_, i) => (
                <span key={i} className="legend-item-inline">
                  <span
                    className="legend-swatch-sm"
                    style={{ backgroundColor: getRankColor(i) }}
                  />
                  R{i}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Right panel: per-GPU slice view */}
        <GpuSliceView op={op} config={config} tpSize={tpSize} />
      </div>

      {/* Strategy description */}
      <div className="partition-description">
        <span
          className="strategy-dot"
          style={{ backgroundColor: stratColor }}
        />
        <strong>
          {strategy === "replicated" && "Replicated"}
          {strategy === "column_parallel" && "Column Parallel"}
          {strategy === "row_parallel" && "Row Parallel"}
          {strategy === "qkv_parallel" && "QKV Parallel"}
          {strategy === "merged_column_parallel" && "Merged Column Parallel"}
        </strong>
        {" \u2014 "}
        {strategy === "replicated" &&
          "Every rank holds the full weight matrix."}
        {strategy === "column_parallel" &&
          "Output dimension split across ranks."}
        {strategy === "row_parallel" &&
          "Input dimension split across ranks. Requires all-reduce after matmul."}
        {strategy === "qkv_parallel" &&
          "Q, K, V heads distributed across ranks."}
        {strategy === "merged_column_parallel" &&
          "Multiple sub-matrices fused and column-split."}
        {op.partition?.details && (
          <span className="partition-detail-text">
            {" "}
            {op.partition.details}
          </span>
        )}
      </div>
    </div>
  );
}

/* ── Helpers ── */

interface Partition {
  x: number;
  y: number;
  w: number;
  h: number;
  rank: number;
}

function computePartitions(op: Operator, tpSize: number): Partition[] {
  const strategy = op.partition?.strategy;
  if (!strategy || strategy === "replicated") return [];

  const partitions: Partition[] = [];

  if (
    strategy === "column_parallel" ||
    strategy === "qkv_parallel" ||
    strategy === "merged_column_parallel"
  ) {
    const sliceHeight = MATRIX_HEIGHT / tpSize;
    for (let r = 0; r < tpSize; r++) {
      partitions.push({
        x: 0,
        y: r * sliceHeight,
        w: MATRIX_WIDTH,
        h: sliceHeight,
        rank: r,
      });
    }
  } else if (strategy === "row_parallel") {
    const sliceWidth = MATRIX_WIDTH / tpSize;
    for (let r = 0; r < tpSize; r++) {
      partitions.push({
        x: r * sliceWidth,
        y: 0,
        w: sliceWidth,
        h: MATRIX_HEIGHT,
        rank: r,
      });
    }
  }

  return partitions;
}

interface QkvRegion {
  label: string;
  yStart: number;
  yEnd: number;
}

function computeQkvRegions(
  config: ModelConfig,
  tpSize: number,
): QkvRegion[] {
  const H = config.num_attention_heads;
  const K = config.num_key_value_heads;
  const d = config.head_dim;
  const totalOut = H * d + 2 * K * d;

  const qFrac = (H * d) / totalOut;
  const kFrac = (K * d) / totalOut;

  const qEnd = qFrac * MATRIX_HEIGHT;
  const kEnd = (qFrac + kFrac) * MATRIX_HEIGHT;

  return [
    {
      label: `Q (${Math.floor(H / tpSize)}h)`,
      yStart: 0,
      yEnd: qEnd,
    },
    {
      label: `K (${Math.floor(K / tpSize)}h)`,
      yStart: qEnd,
      yEnd: kEnd,
    },
    {
      label: `V (${Math.floor(K / tpSize)}h)`,
      yStart: kEnd,
      yEnd: MATRIX_HEIGHT,
    },
  ];
}
