import type { Operator, ModelConfig } from "../../types/model";
import {
  recomputeOperatorTpShape,
  getRankColor,
  shapeToParams,
  formatParams,
} from "../../utils/tpMath";

interface Props {
  op: Operator;
  config: ModelConfig;
  tpSize: number;
}

const SLICE_W = 80;
const SLICE_H = 60;
const SLICE_GAP = 8;
const COLS = 4;

export function GpuSliceView({ op, config, tpSize }: Props) {
  if (tpSize <= 1) return null;

  const tpShape = recomputeOperatorTpShape(op, config, tpSize);
  const isExpert = op.full_weight_shape.length === 3;
  const displayShape = isExpert ? tpShape.slice(1) : tpShape;
  const params = shapeToParams(tpShape);

  const cols = Math.min(tpSize, COLS);
  const rows = Math.ceil(tpSize / cols);
  const svgW = cols * (SLICE_W + SLICE_GAP) - SLICE_GAP + 24;
  const svgH = rows * (SLICE_H + SLICE_GAP) - SLICE_GAP + 40;

  return (
    <div className="gpu-slice-view">
      <div className="gpu-slice-title">Per-GPU View</div>
      <svg
        width={svgW}
        height={svgH}
        viewBox={`0 0 ${svgW} ${svgH}`}
      >
        {Array.from({ length: tpSize }, (_, i) => {
          const col = i % cols;
          const row = Math.floor(i / cols);
          const x = 12 + col * (SLICE_W + SLICE_GAP);
          const y = 8 + row * (SLICE_H + SLICE_GAP);
          const color = getRankColor(i);

          return (
            <g key={i}>
              <rect
                x={x}
                y={y}
                width={SLICE_W}
                height={SLICE_H}
                rx={6}
                ry={6}
                fill={color}
                fillOpacity={0.1}
                stroke={color}
                strokeWidth={1.5}
                strokeOpacity={0.5}
              />
              <text
                x={x + SLICE_W / 2}
                y={y + 16}
                textAnchor="middle"
                fontSize={11}
                fontWeight={600}
                fill={color}
              >
                R{i}
              </text>
              <text
                x={x + SLICE_W / 2}
                y={y + 32}
                textAnchor="middle"
                fontSize={9}
                fontFamily="'JetBrains Mono', monospace"
                fill="var(--text-secondary)"
              >
                {displayShape.join("\u00d7")}
              </text>
              <text
                x={x + SLICE_W / 2}
                y={y + 46}
                textAnchor="middle"
                fontSize={9}
                fill="var(--text-secondary)"
              >
                {formatParams(params)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
