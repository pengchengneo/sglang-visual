import { useState, useMemo, useCallback } from "react";
import RadixTreeViz from "../components/kv-cache/RadixTreeViz";
import MemoryPoolViz from "../components/kv-cache/MemoryPoolViz";
import KVCacheSidebar from "../components/kv-cache/KVCacheSidebar";
import AnimationControls from "../components/shared/AnimationControls";
import MetricsPanel from "../components/shared/MetricsPanel";
import { useAnimation } from "../components/shared/useAnimation";
import {
  createTree,
  insertSequence,
  releaseSequence,
  evictNodes,
  type RadixTree,
  type RadixNode,
} from "../components/kv-cache/RadixTreeEngine";
import { SCENARIOS } from "../components/kv-cache/scenarioPresets";
import "./KVCachePage.css";

interface FrameState {
  tree: RadixTree;
  highlightIds: Set<string>;
  hits: number;
  misses: number;
  evictions: number;
  message: string;
}

function cloneNode(node: RadixNode, parent: RadixNode | null): RadixNode {
  const clone: RadixNode = {
    id: node.id,
    tokens: [...node.tokens],
    refCount: node.refCount,
    children: new Map(),
    parent,
    state: node.state,
  };
  for (const [key, child] of node.children) {
    clone.children.set(key, cloneNode(child, clone));
  }
  return clone;
}

function cloneTree(t: RadixTree): RadixTree {
  const root = cloneNode(t.root, null);
  return { root, blockCount: t.blockCount, maxBlocks: t.maxBlocks };
}

function generateFrames(scenarioId: string, maxBlocks: number): FrameState[] {
  const scenario = SCENARIOS.find((s) => s.id === scenarioId);
  if (!scenario) return [];

  const frames: FrameState[] = [];
  let tree = createTree(maxBlocks);
  let totalEvictions = 0;

  // Initial frame
  frames.push({
    tree: cloneTree(tree),
    highlightIds: new Set(),
    hits: 0,
    misses: 0,
    evictions: 0,
    message: `Scenario: ${scenario.name} — Press Play to start`,
  });

  const maxFrame = Math.max(
    ...scenario.requests.map((r) => r.arrivalFrame + r.durationFrames)
  );

  for (let frame = 0; frame <= maxFrame; frame++) {
    tree = createTree(maxBlocks);
    const highlights = new Set<string>();
    let frameEvictions = totalEvictions;
    let msg = "";

    for (let ri = 0; ri < scenario.requests.length; ri++) {
      const req = scenario.requests[ri];
      if (frame >= req.arrivalFrame) {
        const isActive = frame < req.arrivalFrame + req.durationFrames;
        const isArriving = frame === req.arrivalFrame;

        while (tree.blockCount >= tree.maxBlocks - 2) {
          const evicted = evictNodes(tree, 1);
          if (evicted.length === 0) break;
          frameEvictions += evicted.length;
        }

        const leaf = insertSequence(tree, req.tokens);

        if (isArriving) {
          msg = `Request ${ri + 1} arrives: "${req.tokens.slice(0, 4).join(" ")}..."`;
          let n: RadixNode | null = leaf;
          while (n) {
            highlights.add(n.id);
            n = n.parent;
          }
        }

        if (!isActive) {
          releaseSequence(tree, leaf);
          if (frame === req.arrivalFrame + req.durationFrames) {
            msg = `Request ${ri + 1} completed — releasing cache references`;
          }
        }
      }
    }

    frames.push({
      tree: cloneTree(tree),
      highlightIds: highlights,
      hits: 0,
      misses: 0,
      evictions: frameEvictions,
      message: msg || `Frame ${frame}`,
    });
  }

  return frames;
}

export default function KVCachePage() {
  const [scenarioId, setScenarioId] = useState(SCENARIOS[0].id);
  const [maxBlocks, setMaxBlocks] = useState(SCENARIOS[0].maxBlocks);

  const frames = useMemo(
    () => generateFrames(scenarioId, maxBlocks),
    [scenarioId, maxBlocks]
  );

  const [animState, animControls] = useAnimation(frames.length, 400);
  const currentFrame = frames[animState.frame] ?? frames[0];

  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  const handleScenarioChange = useCallback((id: string) => {
    setScenarioId(id);
    const scenario = SCENARIOS.find((s) => s.id === id);
    if (scenario) setMaxBlocks(scenario.maxBlocks);
    animControls.reset();
  }, [animControls]);

  return (
    <div className="kv-cache-page">
      <div className="kv-cache-header">
        <h2>KV Cache — RadixAttention</h2>
        <p className="page-subtitle">
          Visualize how SGLang uses a radix tree to share KV cache across requests with common prefixes
        </p>
      </div>

      <div className="kv-cache-controls">
        <KVCacheSidebar
          selectedScenario={scenarioId}
          onSelectScenario={handleScenarioChange}
          maxBlocks={maxBlocks}
          onMaxBlocksChange={setMaxBlocks}
        />
      </div>

      <AnimationControls state={animState} controls={animControls} label="RadixTree" />

      <div className="kv-cache-message">{currentFrame.message}</div>

      <MetricsPanel
        metrics={[
          { label: "Cache Blocks", value: currentFrame.tree.blockCount, unit: ` / ${currentFrame.tree.maxBlocks}` },
          { label: "Evictions", value: currentFrame.evictions, color: "var(--red)" },
        ]}
      />

      <div className="kv-cache-content">
        <div className="kv-cache-tree-panel">
          <RadixTreeViz tree={currentFrame.tree} highlightIds={currentFrame.highlightIds} />
        </div>
        <div className="kv-cache-memory-panel">
          <MemoryPoolViz
            tree={currentFrame.tree}
            hoveredNodeId={hoveredNodeId}
            onHoverNode={setHoveredNodeId}
          />
        </div>
      </div>
    </div>
  );
}
