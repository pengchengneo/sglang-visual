interface Props {
  requestCount: number;
  onRequestCountChange: (n: number) => void;
  maxBatchSize: number;
  onMaxBatchSizeChange: (n: number) => void;
  arrivalSpread: number;
  onArrivalSpreadChange: (n: number) => void;
}

export default function SchedulingSidebar({
  requestCount,
  onRequestCountChange,
  maxBatchSize,
  onMaxBatchSizeChange,
  arrivalSpread,
  onArrivalSpreadChange,
}: Props) {
  return (
    <div className="scheduling-sidebar-controls">
      <label className="control-label">
        Requests: {requestCount}
      </label>
      <input
        type="range"
        min={3}
        max={12}
        value={requestCount}
        onChange={(e) => onRequestCountChange(Number(e.target.value))}
        className="sidebar-range"
      />

      <label className="control-label" style={{ marginTop: 8 }}>
        Max Batch Size: {maxBatchSize}
      </label>
      <input
        type="range"
        min={2}
        max={8}
        value={maxBatchSize}
        onChange={(e) => onMaxBatchSizeChange(Number(e.target.value))}
        className="sidebar-range"
      />

      <label className="control-label" style={{ marginTop: 8 }}>
        Arrival Interval: {arrivalSpread} frames
      </label>
      <input
        type="range"
        min={1}
        max={8}
        value={arrivalSpread}
        onChange={(e) => onArrivalSpreadChange(Number(e.target.value))}
        className="sidebar-range"
      />
    </div>
  );
}
