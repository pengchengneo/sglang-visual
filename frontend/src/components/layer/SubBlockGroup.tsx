import type { ModelConfig } from "../../types/model";
import type { SubBlock } from "../../utils/layoutMath";
import { OperatorRect } from "./OperatorRect";
import { FlowArrow, AllReduceDiamond } from "./CommConnector";

interface Props {
  block: SubBlock;
  config: ModelConfig;
  tpSize: number;
  selectedOp: string | null;
  onSelectOp: (name: string | null) => void;
}

const PADDING = 12;
const TITLE_HEIGHT = 22;
const OP_HEIGHT = 52;
const EXPERT_OP_HEIGHT = 64;
const FLOW_GAP = 24;
const COMM_GAP = 24;
const BLOCK_WIDTH = 280;
const OP_WIDTH = BLOCK_WIDTH - PADDING * 2;

function opHeight(isExpert: boolean): number {
  return isExpert ? EXPERT_OP_HEIGHT : OP_HEIGHT;
}

export function computeSubBlockHeight(
  block: SubBlock,
  tpSize: number,
): number {
  let h = PADDING * 2 + TITLE_HEIGHT;
  for (let i = 0; i < block.operators.length; i++) {
    const isExpert = block.operators[i].full_weight_shape.length === 3;
    h += opHeight(isExpert);
    if (i < block.operators.length - 1) h += FLOW_GAP;
  }
  if (block.hasComm && tpSize > 1) h += COMM_GAP;
  return h;
}

export function SubBlockGroup({
  block,
  config,
  tpSize,
  selectedOp,
  onSelectOp,
}: Props) {
  const blockColor =
    block.type === "attention"
      ? "#818cf8"
      : block.type === "moe"
        ? "#a78bfa"
        : "#34d399";

  const svgHeight = computeSubBlockHeight(block, tpSize);
  const centerX = BLOCK_WIDTH / 2;

  // Compute y positions for each operator
  const opPositions: { y: number; h: number }[] = [];
  let cy = PADDING + TITLE_HEIGHT;
  for (let i = 0; i < block.operators.length; i++) {
    const isExpert = block.operators[i].full_weight_shape.length === 3;
    const h = opHeight(isExpert);
    opPositions.push({ y: cy, h });
    cy += h;
    if (i < block.operators.length - 1) cy += FLOW_GAP;
  }

  return (
    <div className="sub-block-group">
      <svg
        width={BLOCK_WIDTH}
        height={svgHeight}
        viewBox={`0 0 ${BLOCK_WIDTH} ${svgHeight}`}
      >
        {/* Sub-block border */}
        <rect
          x={0.5}
          y={0.5}
          width={BLOCK_WIDTH - 1}
          height={svgHeight - 1}
          rx={10}
          ry={10}
          fill="none"
          stroke={blockColor}
          strokeWidth={1}
          strokeOpacity={0.3}
          strokeDasharray="6,3"
        />

        {/* Title */}
        <text
          x={PADDING}
          y={PADDING + 4}
          fontSize={11}
          fontWeight={600}
          fill={blockColor}
          dominantBaseline="hanging"
          letterSpacing="0.5"
        >
          {block.label.toUpperCase()}
        </text>

        {/* Operators */}
        {block.operators.map((op, i) => {
          const { y, h } = opPositions[i];
          const isSelected = selectedOp === op.name;

          return (
            <g key={op.name}>
              <OperatorRect
                op={op}
                config={config}
                tpSize={tpSize}
                x={PADDING}
                y={y}
                width={OP_WIDTH}
                height={h}
                isSelected={isSelected}
                onClick={() => onSelectOp(isSelected ? null : op.name)}
              />

              {/* Flow arrow to next op */}
              {i < block.operators.length - 1 && (
                <FlowArrow
                  x={centerX}
                  y1={y + h}
                  y2={opPositions[i + 1].y}
                />
              )}

              {/* Comm diamond after last op with comm */}
              {op.comm_after &&
                tpSize > 1 &&
                i === block.operators.length - 1 && (
                  <AllReduceDiamond
                    x={centerX}
                    y={y + h + COMM_GAP / 2}
                    label={op.comm_after}
                    show={true}
                  />
                )}
            </g>
          );
        })}

        {/* Block-level comm for MoE (moe_output_reduce) */}
        {block.hasComm &&
          tpSize > 1 &&
          !block.operators.some((o) => o.comm_after) && (
            <AllReduceDiamond
              x={centerX}
              y={
                opPositions[opPositions.length - 1].y +
                opPositions[opPositions.length - 1].h +
                COMM_GAP / 2
              }
              label="all_reduce"
              show={true}
            />
          )}
      </svg>
    </div>
  );
}
