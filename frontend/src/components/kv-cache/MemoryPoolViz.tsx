import type { RadixTree } from "./RadixTreeEngine";
import { flattenTree } from "./RadixTreeEngine";
import { useMemo } from "react";
import "./MemoryPoolViz.css";

interface Props {
  tree: RadixTree;
  hoveredNodeId?: string | null;
  onHoverNode?: (id: string | null) => void;
}

const STATE_COLORS: Record<string, string> = {
  active: "#22c55e",
  cached: "#6366f1",
  evictable: "#9ca3af",
  inserting: "#14b8a6",
  evicting: "#ef4444",
  matching: "#eab308",
};

export default function MemoryPoolViz({ tree, hoveredNodeId, onHoverNode }: Props) {
  const blocks = useMemo(() => {
    const flat = flattenTree(tree);
    return flat.filter((fn) => fn.depth > 0);
  }, [tree]);

  const freeBlocks = tree.maxBlocks - tree.blockCount;
  const usagePercent = ((tree.blockCount / tree.maxBlocks) * 100).toFixed(0);

  return (
    <div className="memory-pool">
      <div className="memory-pool-header">
        <span className="memory-pool-title">KV Cache Memory Pool</span>
        <span className="memory-pool-usage">
          {tree.blockCount} / {tree.maxBlocks} blocks ({usagePercent}%)
        </span>
      </div>
      <div className="memory-pool-grid">
        {blocks.map((b) => (
          <div
            key={b.node.id}
            className={`memory-block${hoveredNodeId === b.node.id ? " hovered" : ""}`}
            style={{
              backgroundColor: STATE_COLORS[b.node.state],
              opacity: hoveredNodeId && hoveredNodeId !== b.node.id ? 0.3 : 1,
            }}
            onMouseEnter={() => onHoverNode?.(b.node.id)}
            onMouseLeave={() => onHoverNode?.(null)}
            title={`${b.node.tokens.join(" ")} | ref: ${b.node.refCount} | ${b.node.state}`}
          />
        ))}
        {Array.from({ length: freeBlocks }).map((_, i) => (
          <div key={`free-${i}`} className="memory-block free" />
        ))}
      </div>
      <div className="memory-pool-legend">
        <span className="legend-item"><span className="legend-dot" style={{ background: "#22c55e" }} /> Active</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: "#6366f1" }} /> Cached</span>
        <span className="legend-item"><span className="legend-dot" style={{ background: "#9ca3af" }} /> Evictable</span>
        <span className="legend-item"><span className="legend-dot free-dot" /> Free</span>
      </div>
    </div>
  );
}
