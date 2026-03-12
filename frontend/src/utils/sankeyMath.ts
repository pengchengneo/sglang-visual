/**
 * Sankey-style path generation for curved connections between pipeline stages.
 */

/** Generate a horizontal cubic bezier path string. */
export function cubicBezierHorizontal(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): string {
  const cx = (x1 + x2) / 2;
  return `M ${x1},${y1} C ${cx},${y1} ${cx},${y2} ${x2},${y2}`;
}

/** Generate a filled Sankey band shape between two vertical edges. */
export function sankeyBand(
  sourceTop: number,
  sourceBottom: number,
  targetTop: number,
  targetBottom: number,
  sourceX: number,
  targetX: number,
): string {
  const cx = (sourceX + targetX) / 2;
  // Top edge: source top -> target top
  const topPath = `M ${sourceX},${sourceTop} C ${cx},${sourceTop} ${cx},${targetTop} ${targetX},${targetTop}`;
  // Bottom edge: target bottom -> source bottom (reverse direction to close shape)
  const bottomPath = `L ${targetX},${targetBottom} C ${cx},${targetBottom} ${cx},${sourceBottom} ${sourceX},${sourceBottom}`;
  return `${topPath} ${bottomPath} Z`;
}
