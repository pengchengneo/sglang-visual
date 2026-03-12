import { useMemo } from "react";
import type { ModelConfig } from "../../types/model";
import type { LayerGroup } from "../../utils/layoutMath";
import { getStrategyColor, computeOperatorWidth } from "../../utils/layoutMath";
import {
  recomputeOperatorTpShape,
  shapeToParams,
  formatParams,
} from "../../utils/tpMath";
import { LayerBlockSvg, findOperator } from "../layer/LayerBlockSvg";
import { MatrixPartitionViz } from "../matrix/MatrixPartitionViz";

interface Props {
  group: LayerGroup;
  config: ModelConfig;
  tpSize: number;
  isExpanded: boolean;
  selectedOp: string | null;
  onToggle: () => void;
  onSelectOp: (name: string | null) => void;
}

const MAX_THUMB_WIDTH = 120;

export function LayerGroupBlock({
  group,
  config,
  tpSize,
  isExpanded,
  selectedOp,
  onToggle,
  onSelectOp,
}: Props) {
  const { representative: layer, count, startId, endId } = group;
  const layerLabel =
    count === 1 ? `Layer ${startId}` : `Layers ${startId}\u2013${endId}`;

  const linearOps = useMemo(
    () => layer.operators.filter((op) => op.op_type !== "comm"),
    [layer.operators],
  );

  const commCount = useMemo(() => {
    if (tpSize <= 1) return 0;
    return layer.operators.filter((op) => op.comm_after).length;
  }, [layer.operators, tpSize]);

  // Compute max params for sqrt scaling of thumbnails
  const maxParams = useMemo(() => {
    let max = 0;
    for (const op of linearOps) {
      const shape = recomputeOperatorTpShape(op, config, tpSize);
      const p = shapeToParams(shape);
      if (p > max) max = p;
    }
    return max;
  }, [linearOps, config, tpSize]);

  const selectedOperator = useMemo(() => {
    if (!selectedOp) return null;
    return findOperator(layer, selectedOp) ?? null;
  }, [layer, selectedOp]);

  const stackedClass = count > 1 && !isExpanded ? " stacked" : "";

  return (
    <div
      className={`layer-group-block${isExpanded ? " expanded" : ""}${stackedClass} ${layer.layer_type}`}
    >
      {/* Header - always visible */}
      <div className="layer-group-header" onClick={onToggle}>
        <div className="layer-group-title">
          <span className="layer-group-label">{layerLabel}</span>
          {count > 1 && <span className="layer-group-count">\u00d7{count}</span>}
          <span className={`layer-type-badge ${layer.layer_type}`}>
            {layer.layer_type}
          </span>
          {commCount > 0 && (
            <span className="comm-count">{commCount} comm</span>
          )}
        </div>
        <span className="expand-indicator">
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path
              d={isExpanded ? "M1 7L5 3L9 7" : "M1 3L5 7L9 3"}
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </div>

      {/* Collapsed: operator thumbnails */}
      {!isExpanded && (
        <div className="layer-group-thumbs">
          {linearOps.map((op) => {
            const tpShape = recomputeOperatorTpShape(op, config, tpSize);
            const params = shapeToParams(tpShape);
            const color = getStrategyColor(op.partition?.strategy);
            const width = computeOperatorWidth(
              params,
              maxParams,
              MAX_THUMB_WIDTH,
            );

            return (
              <div
                key={op.name}
                className="op-thumb"
                style={{
                  width: `${width}px`,
                  borderLeftColor: color,
                  backgroundColor: `${color}15`,
                }}
                title={`${op.name}: ${tpShape.join("\u00d7")} (${formatParams(params)})`}
              >
                <span className="op-thumb-name">{op.name}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Expanded: layer detail SVG */}
      {isExpanded && (
        <div className="layer-group-detail">
          <LayerBlockSvg
            layer={layer}
            config={config}
            tpSize={tpSize}
            selectedOp={selectedOp}
            onSelectOp={onSelectOp}
          />

          {/* Matrix partition viz for selected operator */}
          {selectedOperator && selectedOperator.op_type !== "comm" && (
            <div className="matrix-viz-container">
              <MatrixPartitionViz
                op={selectedOperator}
                config={config}
                tpSize={tpSize}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
