import { useState, useMemo } from "react";
import type { ModelArchitecture, Layer } from "../../types/model";
import {
  formatParams,
  formatMemory,
  shapeToParams,
  computePerRankParams,
} from "../../utils/tpMath";
import { ArchitectureDiagram } from "../architecture/ArchitectureDiagram";

interface Props {
  model: ModelArchitecture;
  tpSize: number;
}

export function PipelineView({ model, tpSize }: Props) {
  const [selectedOpName, setSelectedOpName] = useState<string | null>(null);

  const totalParams = useMemo(() => computeTotalParams(model), [model]);
  const perRankParams = useMemo(
    () => computePerRankParams(model, tpSize),
    [model, tpSize],
  );
  const commOps = useMemo(
    () => computeTotalCommOps(model, tpSize),
    [model, tpSize],
  );

  const handleSelectOp = (name: string | null, _layer: Layer | null) => {
    setSelectedOpName(name);
  };

  return (
    <div className="pipeline-view">
      {/* Compact stats bar */}
      <div className="stats-bar">
        <div className="stat-item">
          <span className="stat-val">{formatParams(totalParams)}</span>
          <span className="stat-lbl">params</span>
        </div>
        <span className="stat-sep">{"\u00b7"}</span>
        <div className="stat-item">
          <span className="stat-val">{formatParams(perRankParams)}</span>
          <span className="stat-lbl">/ rank</span>
        </div>
        <span className="stat-sep">{"\u00b7"}</span>
        <div className="stat-item">
          <span className="stat-val">{formatMemory(perRankParams)}</span>
          <span className="stat-lbl">FP16</span>
        </div>
        <span className="stat-sep">{"\u00b7"}</span>
        <div className="stat-item">
          <span className="stat-val">{commOps}</span>
          <span className="stat-lbl">comm ops</span>
        </div>
      </div>

      {/* Architecture diagram (TP viz is inline) */}
      <ArchitectureDiagram
        model={model}
        tpSize={tpSize}
        selectedOp={selectedOpName}
        onSelectOp={handleSelectOp}
      />
    </div>
  );
}

/* ── Helpers ── */

function computeTotalParams(model: ModelArchitecture): number {
  let total = 0;
  total += shapeToParams(model.embedding.full_shape);
  total += shapeToParams(model.lm_head.full_shape);
  for (const layer of model.layers) {
    for (const op of layer.operators) {
      if (op.op_type !== "comm") {
        total += shapeToParams(op.full_weight_shape);
      }
    }
  }
  return total;
}

function computeTotalCommOps(
  model: ModelArchitecture,
  tp: number,
): number {
  if (tp === 1) return 0;
  let total = model.embedding.comm_after ? 1 : 0;
  for (const layer of model.layers) {
    for (const op of layer.operators) {
      if (op.comm_after) total++;
    }
  }
  return total;
}
