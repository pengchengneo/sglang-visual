import type { AnimationState, AnimationControls as Controls } from "./useAnimation";
import "./AnimationControls.css";

interface Props {
  state: AnimationState;
  controls: Controls;
  label?: string;
}

const SPEEDS = [0.5, 1, 2, 4];

export default function AnimationControls({ state, controls, label }: Props) {
  return (
    <div className="anim-controls">
      {label && <span className="anim-label">{label}</span>}
      <div className="anim-buttons">
        <button className="anim-btn" onClick={controls.reset} title="Reset">⏮</button>
        {state.playing ? (
          <button className="anim-btn anim-btn-primary" onClick={controls.pause} title="Pause">⏸</button>
        ) : (
          <button className="anim-btn anim-btn-primary" onClick={controls.play} title="Play">▶</button>
        )}
        <button className="anim-btn" onClick={controls.step} title="Step">⏭</button>
      </div>
      <div className="anim-speed">
        {SPEEDS.map((s) => (
          <button key={s} className={`anim-speed-btn${state.speed === s ? " active" : ""}`} onClick={() => controls.setSpeed(s)}>{s}x</button>
        ))}
      </div>
      <div className="anim-progress">
        <input type="range" min={0} max={state.totalFrames - 1} value={state.frame} onChange={(e) => controls.setFrame(Number(e.target.value))} className="anim-slider" />
        <span className="anim-frame-label">{state.frame + 1} / {state.totalFrames}</span>
      </div>
    </div>
  );
}
