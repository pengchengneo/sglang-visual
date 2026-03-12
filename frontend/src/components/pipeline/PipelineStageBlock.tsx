import type { EmbeddingInfo } from "../../types/model";
import { recomputeEmbeddingTpShape, shapeToParams, formatParams } from "../../utils/tpMath";
import { getStrategyColor } from "../../utils/layoutMath";

interface Props {
  label: string;
  info: EmbeddingInfo;
  tpSize: number;
}

export function PipelineStageBlock({ label, info, tpSize }: Props) {
  const tpShape = recomputeEmbeddingTpShape(info, tpSize);
  const color = getStrategyColor(info.partition_strategy);
  const params = shapeToParams(tpSize > 1 ? tpShape : info.full_shape);

  return (
    <div className="pipeline-stage-block">
      <div className="stage-color-bar" style={{ backgroundColor: color }} />
      <div className="stage-content">
        <span className="stage-label">{label}</span>
        <span className="stage-shape">
          [{info.full_shape.join(" \u00d7 ")}]
        </span>
        {tpSize > 1 && (
          <span className="stage-shape">
            <span className="stage-tp-shape">
              {"\u2192 ["}
              {tpShape.join(" \u00d7 ")}
              {"]"}
            </span>
          </span>
        )}
        <span className="stage-params">{formatParams(params)}</span>
      </div>
    </div>
  );
}
