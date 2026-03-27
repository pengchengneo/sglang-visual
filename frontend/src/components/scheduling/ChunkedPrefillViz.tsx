import type { PrefillFrame } from "./SchedulingEngine";
import "./ChunkedPrefillViz.css";

interface Props { frames: PrefillFrame[]; currentFrame: number; totalPrefillLength: number; }

export default function ChunkedPrefillViz({ frames, currentFrame, totalPrefillLength }: Props) {
  const visibleFrames = frames.slice(0, currentFrame + 1);
  return (
    <div className="cp-viz">
      <div className="cp-timeline-header">
        <span className="cp-timeline-label">Step</span>
        <span className="cp-timeline-label" style={{ flex: 1 }}>Execution Timeline</span>
      </div>
      <div className="cp-timeline">
        {visibleFrames.map((f, i) => (
          <div key={i} className={`cp-step${i === currentFrame ? " current" : ""}`}>
            <span className="cp-step-num">{i}</span>
            <div className="cp-step-bars">
              {f.executing.map((ex, j) => (
                <div key={j} className={`cp-exec-block ${ex.type}`} style={{ backgroundColor: ex.color }}
                  title={ex.type === "prefill" ? `Prefill [${ex.tokenRange?.[0]}:${ex.tokenRange?.[1]}]` : `Decode R${ex.requestId}`}>
                  {ex.type === "prefill" ? <span>P [{ex.tokenRange?.[0]}:{ex.tokenRange?.[1]}]</span> : <span>D{ex.requestId}</span>}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
