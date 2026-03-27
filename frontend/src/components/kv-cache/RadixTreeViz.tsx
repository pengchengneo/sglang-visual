import { useMemo } from "react";
import type { RadixTree, FlatNode } from "./RadixTreeEngine";
import { flattenTree } from "./RadixTreeEngine";
import "./RadixTreeViz.css";

interface Props {
  tree: RadixTree;
  highlightIds?: Set<string>;
}

const NODE_W = 120;
const NODE_H = 48;
const H_GAP = 20;
const V_GAP = 60;

interface LayoutNode extends FlatNode {
  x: number;
  y: number;
}

function layoutTree(flatNodes: FlatNode[]): LayoutNode[] {
  const byDepth = new Map<number, FlatNode[]>();
  for (const fn of flatNodes) {
    const list = byDepth.get(fn.depth) ?? [];
    list.push(fn);
    byDepth.set(fn.depth, list);
  }
  const positions = new Map<string, { x: number; y: number }>();
  for (const [depth, nodes] of byDepth) {
    const totalWidth = nodes.length * NODE_W + (nodes.length - 1) * H_GAP;
    const startX = -totalWidth / 2;
    nodes.forEach((fn, i) => {
      positions.set(fn.node.id, { x: startX + i * (NODE_W + H_GAP) + NODE_W / 2, y: depth * (NODE_H + V_GAP) });
    });
  }
  return flatNodes.map((fn) => ({ ...fn, ...positions.get(fn.node.id)! }));
}

const STATE_COLORS: Record<string, string> = {
  active: "var(--green)", cached: "var(--accent)", evictable: "var(--text-secondary)",
  inserting: "var(--teal)", evicting: "var(--red)", matching: "var(--yellow)",
};

export default function RadixTreeViz({ tree, highlightIds }: Props) {
  const flatNodes = useMemo(() => flattenTree(tree), [tree]);
  const layout = useMemo(() => layoutTree(flatNodes), [flatNodes]);
  const xs = layout.map((n) => n.x);
  const ys = layout.map((n) => n.y);
  const minX = Math.min(...xs) - NODE_W / 2 - 20;
  const maxX = Math.max(...xs) + NODE_W / 2 + 20;
  const maxY = Math.max(...ys) + NODE_H + 20;
  const viewBox = `${minX} -20 ${maxX - minX} ${maxY + 40}`;
  const posMap = new Map(layout.map((n) => [n.node.id, n]));

  return (
    <svg className="radix-tree-svg" viewBox={viewBox} preserveAspectRatio="xMidYMid meet">
      {layout.filter((n) => n.parentId).map((n) => {
        const parent = posMap.get(n.parentId!);
        if (!parent) return null;
        return (<line key={`edge-${n.node.id}`} x1={parent.x} y1={parent.y + NODE_H / 2} x2={n.x} y2={n.y - NODE_H / 2} className="radix-edge" stroke={STATE_COLORS[n.node.state]} strokeOpacity={0.4} />);
      })}
      {layout.map((n) => {
        const isHighlighted = highlightIds?.has(n.node.id);
        const color = STATE_COLORS[n.node.state];
        const tokenLabel = n.node.tokens.length <= 3 ? n.node.tokens.join(" ") : n.node.tokens.slice(0, 3).join(" ") + "\u2026";
        return (
          <g key={n.node.id} transform={`translate(${n.x - NODE_W / 2}, ${n.y - NODE_H / 2})`} className={`radix-node${isHighlighted ? " highlighted" : ""}`}>
            <rect width={NODE_W} height={NODE_H} rx={8} fill={color} fillOpacity={0.15} stroke={color} strokeWidth={isHighlighted ? 2.5 : 1.5} />
            <text x={NODE_W / 2} y={18} textAnchor="middle" className="radix-node-label" fill={color}>{tokenLabel}</text>
            <text x={NODE_W / 2} y={36} textAnchor="middle" className="radix-node-ref" fill="var(--text-secondary)">ref: {n.node.refCount}</text>
          </g>
        );
      })}
    </svg>
  );
}
