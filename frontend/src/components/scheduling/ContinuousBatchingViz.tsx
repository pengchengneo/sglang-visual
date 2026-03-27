import type { SchedulerFrame, Request } from "./SchedulingEngine";
import "./ContinuousBatchingViz.css";

interface Props { frame: SchedulerFrame; maxBatchSize: number; }

function RequestBar({ request, progress, total, phase }: { request: Request; progress: number; total: number; phase: "waiting" | "prefill" | "decode" }) {
  const pct = total > 0 ? (progress / total) * 100 : 0;
  return (
    <div className="request-bar" title={`Request ${request.id} — ${phase}`}>
      <div className="request-bar-label">R{request.id}</div>
      <div className="request-bar-track">
        <div className={`request-bar-fill ${phase}`} style={{ width: `${pct}%`, backgroundColor: request.color }} />
      </div>
      <div className="request-bar-info">{progress}/{total}</div>
    </div>
  );
}

export default function ContinuousBatchingViz({ frame, maxBatchSize }: Props) {
  return (
    <div className="cb-viz">
      <div className="cb-queue">
        <div className="cb-queue-header"><span className="cb-queue-title">Waiting Queue</span><span className="cb-queue-count">{frame.waitingQueue.length}</span></div>
        <div className="cb-queue-list">
          {frame.waitingQueue.length === 0 && <div className="cb-empty">Empty</div>}
          {frame.waitingQueue.map((req) => (<RequestBar key={req.id} request={req} progress={0} total={req.prefillLength + req.decodeLength} phase="waiting" />))}
        </div>
      </div>
      <div className="cb-arrow">
        <svg width="40" height="24" viewBox="0 0 40 24"><path d="M4 12 L30 12 M24 6 L30 12 L24 18" fill="none" stroke="var(--accent)" strokeWidth="2" /></svg>
        <span className="cb-arrow-label">Schedule</span>
      </div>
      <div className="cb-batch">
        <div className="cb-queue-header"><span className="cb-queue-title">Running Batch</span><span className="cb-queue-count">{frame.runningBatch.length} / {maxBatchSize}</span></div>
        <div className="cb-queue-list">
          {frame.runningBatch.length === 0 && <div className="cb-empty">Empty</div>}
          {frame.runningBatch.map((rr) => {
            const isPrefill = rr.prefillProgress < rr.request.prefillLength;
            const progress = isPrefill ? rr.prefillProgress : rr.request.prefillLength + rr.decodeProgress;
            const total = rr.request.prefillLength + rr.request.decodeLength;
            return (<RequestBar key={rr.request.id} request={rr.request} progress={progress} total={total} phase={isPrefill ? "prefill" : "decode"} />);
          })}
          {Array.from({ length: maxBatchSize - frame.runningBatch.length }).map((_, i) => (<div key={`empty-${i}`} className="request-bar empty-slot"><div className="request-bar-track" /></div>))}
        </div>
      </div>
      <div className="cb-completed">
        <div className="cb-queue-header"><span className="cb-queue-title">Completed</span><span className="cb-queue-count">{frame.completedRequests.length}</span></div>
        <div className="cb-completed-chips">
          {frame.completedRequests.map((req) => (<span key={req.id} className="cb-completed-chip" style={{ backgroundColor: req.color }}>R{req.id}</span>))}
        </div>
      </div>
    </div>
  );
}
