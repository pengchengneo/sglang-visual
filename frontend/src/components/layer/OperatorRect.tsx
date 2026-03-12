import type { Operator, ModelConfig } from "../../types/model";
import { getStrategyColor, getStrategyLabel } from "../../utils/layoutMath";
import {
  recomputeOperatorTpShape,
  shapeToParams,
  formatParams,
} from "../../utils/tpMath";

interface Props {
  op: Operator;
  config: ModelConfig;
  tpSize: number;
  x: number;
  y: number;
  width: number;
  height: number;
  isSelected: boolean;
  onClick: () => void;
}

export function OperatorRect({
  op,
  config,
  tpSize,
  x,
  y,
  width,
  height,
  isSelected,
  onClick,
}: Props) {
  const strategy = op.partition?.strategy;
  const color = getStrategyColor(strategy);
  const tpShape = recomputeOperatorTpShape(op, config, tpSize);
  const params = shapeToParams(tpShape);
  const isExpert = op.full_weight_shape.length === 3;
  const expertCount = isExpert ? op.full_weight_shape[0] : 0;
  const patternId = `expert-grid-${op.name.replace(/\W/g, "")}`;

  return (
    <g
      className="operator-rect"
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      style={{ cursor: "pointer" }}
    >
      {/* Expert grid pattern definition */}
      {isExpert && (
        <defs>
          <pattern
            id={patternId}
            patternUnits="userSpaceOnUse"
            width="14"
            height="14"
          >
            <rect width="14" height="14" fill="none" />
            <line
              x1="14"
              y1="0"
              x2="14"
              y2="14"
              stroke={color}
              strokeWidth="0.5"
              strokeOpacity="0.15"
              strokeDasharray="2,2"
            />
            <line
              x1="0"
              y1="14"
              x2="14"
              y2="14"
              stroke={color}
              strokeWidth="0.5"
              strokeOpacity="0.15"
              strokeDasharray="2,2"
            />
          </pattern>
        </defs>
      )}

      {/* Background rect */}
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={8}
        ry={8}
        fill={color}
        fillOpacity={isSelected ? 0.2 : 0.08}
        stroke={color}
        strokeWidth={isSelected ? 2 : 1}
        strokeOpacity={isSelected ? 1 : 0.3}
        className="op-rect-bg"
      />

      {/* Expert grid overlay */}
      {isExpert && (
        <rect
          x={x + 1}
          y={y + 1}
          width={width - 2}
          height={height - 2}
          rx={7}
          ry={7}
          fill={`url(#${patternId})`}
        />
      )}

      {/* Operator name */}
      <text
        x={x + width / 2}
        y={y + (isExpert ? height / 2 - 12 : height / 2 - 7)}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={11}
        fontWeight={600}
        fill="var(--text)"
      >
        {op.name}
      </text>

      {/* Expert count */}
      {isExpert && (
        <text
          x={x + width / 2}
          y={y + height / 2}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={9}
          fill={color}
          fontWeight={500}
        >
          {expertCount} experts
        </text>
      )}

      {/* Shape + params + strategy label */}
      <text
        x={x + width / 2}
        y={y + (isExpert ? height / 2 + 12 : height / 2 + 7)}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={9}
        fontFamily="'JetBrains Mono', monospace"
        fill="var(--text-secondary)"
      >
        {isExpert
          ? `[${tpShape.slice(1).join("\u00d7")}]/exp`
          : tpShape.join("\u00d7")}
        {" \u00b7 "}
        {formatParams(params)}
        {strategy && strategy !== "replicated" ? ` \u00b7 ${getStrategyLabel(strategy)}` : ""}
      </text>
    </g>
  );
}
