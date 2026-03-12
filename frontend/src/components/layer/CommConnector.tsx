import { COMM_COLOR } from "../../utils/layoutMath";

interface ArrowProps {
  x: number;
  y1: number;
  y2: number;
}

export function FlowArrow({ x, y1, y2 }: ArrowProps) {
  const mid = (y1 + y2) / 2;
  const headSize = 4;
  // Smooth curved path instead of straight line
  const path = `M ${x},${y1} C ${x},${mid} ${x},${mid} ${x},${y2}`;
  return (
    <g>
      <path
        d={path}
        stroke="var(--border)"
        strokeWidth={1.5}
        fill="none"
      />
      <polygon
        points={`${x - headSize},${mid - headSize} ${x},${mid + headSize} ${x + headSize},${mid - headSize}`}
        fill="var(--border)"
      />
    </g>
  );
}

interface DiamondProps {
  x: number;
  y: number;
  label: string;
  show: boolean;
}

export function AllReduceDiamond({ x, y, label, show }: DiamondProps) {
  if (!show) return null;
  const s = 7;
  const points = `${x},${y - s} ${x + s},${y} ${x},${y + s} ${x - s},${y}`;

  return (
    <g>
      <polygon
        points={points}
        fill={COMM_COLOR}
        fillOpacity={0.2}
        stroke={COMM_COLOR}
        strokeWidth={1.5}
        strokeOpacity={0.7}
      />
      <text
        x={x + s + 6}
        y={y}
        dominantBaseline="middle"
        fontSize={10}
        fill={COMM_COLOR}
      >
        {label}
      </text>
    </g>
  );
}
