import { useState, useRef, useCallback, useEffect } from "react";

export interface AnimationState {
  frame: number;
  playing: boolean;
  speed: number;
  totalFrames: number;
}

export interface AnimationControls {
  play: () => void;
  pause: () => void;
  step: () => void;
  reset: () => void;
  setSpeed: (speed: number) => void;
  setFrame: (frame: number) => void;
}

export function useAnimation(
  totalFrames: number,
  msPerFrame = 500,
  onFrame?: (frame: number) => void
): [AnimationState, AnimationControls] {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const rafRef = useRef<number>(0);
  const frameRef = useRef(frame);
  frameRef.current = frame;

  useEffect(() => {
    if (!playing) return;
    const interval = msPerFrame / speed;
    let lastTime = performance.now();
    const tick = (now: number) => {
      const delta = now - lastTime;
      if (delta >= interval) {
        lastTime = now - (delta % interval);
        const nextFrame = frameRef.current + 1;
        if (nextFrame >= totalFrames) {
          setPlaying(false);
          return;
        }
        setFrame(nextFrame);
        onFrame?.(nextFrame);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, speed, totalFrames, msPerFrame, onFrame]);

  const controls: AnimationControls = {
    play: useCallback(() => setPlaying(true), []),
    pause: useCallback(() => setPlaying(false), []),
    step: useCallback(() => {
      setPlaying(false);
      setFrame((f) => {
        const next = Math.min(f + 1, totalFrames - 1);
        onFrame?.(next);
        return next;
      });
    }, [totalFrames, onFrame]),
    reset: useCallback(() => {
      setPlaying(false);
      setFrame(0);
      onFrame?.(0);
    }, [onFrame]),
    setSpeed: useCallback((s: number) => setSpeed(s), []),
    setFrame: useCallback(
      (f: number) => {
        setFrame(Math.max(0, Math.min(f, totalFrames - 1)));
        onFrame?.(f);
      },
      [totalFrames, onFrame]
    ),
  };

  return [{ frame, playing, speed, totalFrames }, controls];
}
