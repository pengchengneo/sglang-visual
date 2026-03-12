import { useState, useMemo } from "react";
import type { ModelArchitecture, Layer, Operator } from "../../types/model";
import {
  formatParams,
  formatMemory,
  shapeToParams,
  recomputeOperatorTpShape,
  recomputeEmbeddingTpShape,
} from "../../utils/tpMath";
import { ArchitectureDiagram } from "../architecture/ArchitectureDiagram";
import { MatrixPartitionViz } from "../matrix/MatrixPartitionViz";

interface Props {
  model: ModelArchitecture;
  tpSize: number;
}

export function PipelineView({ model, tpSize }: Props) {
  const [selectedOpName, setSelectedOpName] = useState<string | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<Layer | null>(null);

  const totalParams = useMemo(() => computeTotalParams(model), [model]);
  const perRankParams = useMemo(
    () => computePerRankParams(model, tpSize),
    [model, tpSize],
  );
  const commOps = useMemo(
    () => computeTotalCommOps(model, tpSize),
    [model, tpSize],
  );

  const handleSelectOp = (name: string | null, layer: Layer | null) => {
    setSelectedOpName(name);
    setSelectedLayer(layer);
  };

  // Resolve the selected operator for MatrixPartitionViz
  const selectedOperator: Operator | null = useMemo(() => {
    if (!selectedOpName || !selectedLayer) return null;
    return selectedLayer.operators.find((op) => op.name === selectedOpName) ?? null;
  }, [selectedOpName, selectedLayer]);

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

      {/* Architecture diagram */}
      <ArchitectureDiagram
        model={model}
        tpSize={tpSize}
        selectedOp={selectedOpName}
        onSelectOp={handleSelectOp}
      />

      {/* Matrix partition viz for selected operator */}
      {selectedOperator && selectedOperator.op_type !== "comm" && (
        <div className="matrix-viz-container">
          <MatrixPartitionViz
            op={selectedOperator}
            config={model.config}
            tpSize={tpSize}
          />
        </div>
      )}
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

function computePerRankParams(
  model: ModelArchitecture,
  tp: number,
): number {
  let total = 0;
  total += shapeToParams(recomputeEmbeddingTpShape(model.embedding, tp));
  total += shapeToParams(recomputeEmbeddingTpShape(model.lm_head, tp));
  for (const layer of model.layers) {
    for (const op of layer.operators) {
      if (op.op_type !== "comm") {
        const tpShape = recomputeOperatorTpShape(op, model.config, tp);
        total += shapeToParams(tpShape);
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
