import { useMemo } from "react";
import type { Layer, ModelConfig, Operator } from "../../types/model";
import { groupOperatorsIntoSubBlocks } from "../../utils/layoutMath";
import { SubBlockGroup } from "./SubBlockGroup";

interface Props {
  layer: Layer;
  config: ModelConfig;
  tpSize: number;
  selectedOp: string | null;
  onSelectOp: (name: string | null) => void;
}

export function LayerBlockSvg({
  layer,
  config,
  tpSize,
  selectedOp,
  onSelectOp,
}: Props) {
  const blocks = useMemo(
    () => groupOperatorsIntoSubBlocks(layer.operators),
    [layer.operators],
  );

  return (
    <div className="layer-block-svg">
      {blocks.map((block) => (
        <SubBlockGroup
          key={block.type}
          block={block}
          config={config}
          tpSize={tpSize}
          selectedOp={selectedOp}
          onSelectOp={onSelectOp}
        />
      ))}
    </div>
  );
}

/** Find an operator by name in a layer */
export function findOperator(
  layer: Layer,
  name: string,
): Operator | undefined {
  return layer.operators.find((op) => op.name === name);
}
