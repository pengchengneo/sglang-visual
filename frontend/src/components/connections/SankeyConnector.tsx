import { useRef, useState, useLayoutEffect, useCallback } from "react";
import { sankeyBand } from "../../utils/sankeyMath";
import "./Connections.css";

interface ConnectorProps {
  containerRef: React.RefObject<HTMLDivElement | null>;
  stageRefs: React.RefObject<(HTMLDivElement | null)[]>;
}

interface Connection {
  sourceIdx: number;
  targetIdx: number;
}

export function SankeyConnector({ containerRef, stageRefs }: ConnectorProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [paths, setPaths] = useState<
    { d: string; color1: string; color2: string; id: string }[]
  >([]);
  const [size, setSize] = useState({ w: 0, h: 0 });

  const measure = useCallback(() => {
    const container = containerRef.current;
    const stages = stageRefs.current;
    if (!container || !stages) return;

    const containerRect = container.getBoundingClientRect();
    setSize({ w: containerRect.width, h: containerRect.height });

    // Build connections: each consecutive pair of stages
    const connections: Connection[] = [];
    const validIndices: number[] = [];
    for (let i = 0; i < stages.length; i++) {
      if (stages[i]) validIndices.push(i);
    }
    for (let i = 0; i < validIndices.length - 1; i++) {
      connections.push({
        sourceIdx: validIndices[i],
        targetIdx: validIndices[i + 1],
      });
    }

    const accent = "#6366f1";
    const newPaths: typeof paths = [];

    for (const conn of connections) {
      const sourceEl = stages[conn.sourceIdx];
      const targetEl = stages[conn.targetIdx];
      if (!sourceEl || !targetEl) continue;

      const sRect = sourceEl.getBoundingClientRect();
      const tRect = targetEl.getBoundingClientRect();

      const sourceX = sRect.right - containerRect.left;
      const targetX = tRect.left - containerRect.left;

      const sourceTop = sRect.top - containerRect.top + sRect.height * 0.2;
      const sourceBottom = sRect.top - containerRect.top + sRect.height * 0.8;
      const targetTop = tRect.top - containerRect.top + tRect.height * 0.2;
      const targetBottom = tRect.top - containerRect.top + tRect.height * 0.8;

      const d = sankeyBand(
        sourceTop,
        sourceBottom,
        targetTop,
        targetBottom,
        sourceX,
        targetX,
      );

      newPaths.push({
        d,
        color1: accent,
        color2: accent,
        id: `sankey-${conn.sourceIdx}-${conn.targetIdx}`,
      });
    }

    setPaths(newPaths);
  }, [containerRef, stageRefs]);

  useLayoutEffect(() => {
    measure();

    const container = containerRef.current;
    if (!container) return;

    const ro = new ResizeObserver(() => measure());
    ro.observe(container);
    return () => ro.disconnect();
  }, [measure]);

  if (paths.length === 0 || size.w === 0) return null;

  return (
    <svg
      ref={svgRef}
      className="sankey-connector-overlay"
      width={size.w}
      height={size.h}
      viewBox={`0 0 ${size.w} ${size.h}`}
    >
      <defs>
        {paths.map((p) => (
          <linearGradient key={`grad-${p.id}`} id={`grad-${p.id}`}>
            <stop offset="0%" stopColor={p.color1} stopOpacity={0.12} />
            <stop offset="50%" stopColor={p.color1} stopOpacity={0.06} />
            <stop offset="100%" stopColor={p.color2} stopOpacity={0.12} />
          </linearGradient>
        ))}
      </defs>
      {paths.map((p) => (
        <path
          key={p.id}
          d={p.d}
          fill={`url(#grad-${p.id})`}
          stroke={p.color1}
          strokeWidth={0.5}
          strokeOpacity={0.15}
          style={{ pointerEvents: "auto" }}
        />
      ))}
    </svg>
  );
}
