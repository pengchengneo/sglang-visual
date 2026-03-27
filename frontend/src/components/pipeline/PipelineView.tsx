import { useState, useMemo } from "react";
import type { ModelArchitecture, Layer } from "../../types/model";
import type { Dtype, Quantization } from "../../App";
import {
  formatParams,
  formatMemory,
  shapeToParams,
  computePerRankParams,
  computePerRankParamsForPpStage,
} from "../../utils/tpMath";
import { ArchitectureDiagram } from "../architecture/ArchitectureDiagram";
import "./Pipeline.css";

interface Props {
  model: ModelArchitecture;
  tpSize: number;
  ppSize: number;
  epSize: number;
  bytesPerParam: number;
  dtype: Dtype;
  quantization: Quantization;
}

export function PipelineView({ model, tpSize, ppSize, epSize, bytesPerParam, dtype, quantization }: Props) {
  const [selectedOpName, setSelectedOpName] = useState<string | null>(null);

  const totalParams = useMemo(() => computeTotalParams(model), [model]);
  const perRankParams = useMemo(
    () => {
      if (ppSize <= 1) return computePerRankParams(model, tpSize, epSize);
      let maxParams = 0;
      for (let p = 0; p < ppSize; p++) {
        maxParams = Math.max(maxParams, computePerRankParamsForPpStage(model, tpSize, p, ppSize, epSize));
      }
      return maxParams;
    },
    [model, tpSize, ppSize, epSize],
  );
  const commOps = useMemo(
    () => computeTotalCommOps(model, tpSize),
    [model, tpSize],
  );

  const dtypeLabel = quantization !== "none"
    ? quantization.toUpperCase()
    : dtype.toUpperCase();

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
          <span className="stat-lbl">/ rank{ppSize > 1 ? " (max)" : ""}</span>
        </div>
        <span className="stat-sep">{"\u00b7"}</span>
        <div className="stat-item">
          <span className="stat-val">{formatMemory(perRankParams, bytesPerParam)}</span>
          <span className="stat-lbl">{dtypeLabel}</span>
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
        ppSize={ppSize}
        epSize={epSize}
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
