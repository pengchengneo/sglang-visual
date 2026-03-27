export interface Request {
  id: number;
  prefillLength: number;
  decodeLength: number;
  arrivalFrame: number;
  color: string;
}

export interface RunningRequest {
  request: Request;
  prefillProgress: number;
  decodeProgress: number;
  startFrame: number;
}

export interface SchedulerFrame {
  frame: number;
  waitingQueue: Request[];
  runningBatch: RunningRequest[];
  completedRequests: Request[];
  gpuUtilization: number;
  message: string;
}

export type BatchingMode = "continuous" | "static";

export interface SchedulingConfig {
  requests: Request[];
  maxBatchSize: number;
  mode: BatchingMode;
  chunkedPrefillSize: number;
}

const REQUEST_COLORS = [
  "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6",
  "#14b8a6", "#f97316", "#ec4899", "#06b6d4", "#84cc16",
];

export function generateRequests(
  count: number,
  prefillRange: [number, number],
  decodeRange: [number, number],
  arrivalSpread: number
): Request[] {
  const requests: Request[] = [];
  for (let i = 0; i < count; i++) {
    const prefillLength = prefillRange[0] + Math.floor(Math.random() * (prefillRange[1] - prefillRange[0]));
    const decodeLength = decodeRange[0] + Math.floor(Math.random() * (decodeRange[1] - decodeRange[0]));
    requests.push({ id: i, prefillLength, decodeLength, arrivalFrame: Math.floor(i * arrivalSpread), color: REQUEST_COLORS[i % REQUEST_COLORS.length] });
  }
  return requests;
}

function isRequestDone(rr: RunningRequest): boolean {
  return rr.prefillProgress >= rr.request.prefillLength && rr.decodeProgress >= rr.request.decodeLength;
}

export function simulateScheduling(config: SchedulingConfig): SchedulerFrame[] {
  const frames: SchedulerFrame[] = [];
  const waiting: Request[] = [];
  let running: RunningRequest[] = [];
  const completed: Request[] = [];

  const maxFrame = Math.max(...config.requests.map((r) => r.arrivalFrame)) + Math.max(...config.requests.map((r) => r.prefillLength + r.decodeLength)) + 20;

  for (let frame = 0; frame <= maxFrame; frame++) {
    let msg = "";

    for (const req of config.requests) {
      if (req.arrivalFrame === frame) {
        waiting.push(req);
        msg += `Request ${req.id} arrives (prefill: ${req.prefillLength}, decode: ${req.decodeLength}). `;
      }
    }

    for (const rr of running) {
      if (rr.prefillProgress < rr.request.prefillLength) {
        const chunkSize = config.chunkedPrefillSize > 0 ? config.chunkedPrefillSize : rr.request.prefillLength;
        rr.prefillProgress = Math.min(rr.prefillProgress + chunkSize, rr.request.prefillLength);
      } else {
        rr.decodeProgress++;
      }
    }

    if (config.mode === "continuous") {
      const justCompleted = running.filter(isRequestDone);
      for (const rr of justCompleted) { completed.push(rr.request); msg += `Request ${rr.request.id} completed! `; }
      running = running.filter((rr) => !isRequestDone(rr));
    } else {
      if (running.length > 0 && running.every(isRequestDone)) {
        for (const rr of running) completed.push(rr.request);
        msg += `Batch completed (${running.length} requests). `;
        running = [];
      }
    }

    if (config.mode === "continuous") {
      while (running.length < config.maxBatchSize && waiting.length > 0) {
        const req = waiting.shift()!;
        running.push({ request: req, prefillProgress: 0, decodeProgress: 0, startFrame: frame });
        msg += `Scheduled request ${req.id}. `;
      }
    } else {
      if (running.length === 0 && waiting.length > 0) {
        const batchSize = Math.min(config.maxBatchSize, waiting.length);
        for (let i = 0; i < batchSize; i++) {
          const req = waiting.shift()!;
          running.push({ request: req, prefillProgress: 0, decodeProgress: 0, startFrame: frame });
        }
        msg += `New batch started (${batchSize} requests). `;
      }
    }

    const gpuUtil = running.length / config.maxBatchSize;
    frames.push({ frame, waitingQueue: [...waiting], runningBatch: running.map((rr) => ({ ...rr })), completedRequests: [...completed], gpuUtilization: gpuUtil, message: msg.trim() || `Frame ${frame}` });

    if (completed.length === config.requests.length && running.length === 0 && waiting.length === 0) break;
  }
  return frames;
}

export interface PrefillFrame {
  frame: number;
  executing: { requestId: number; type: "prefill" | "decode"; tokenRange?: [number, number]; color: string; }[];
  decodeLatencies: number[];
}

export function simulateChunkedPrefill(longPrefillLength: number, decodeRequestCount: number, chunkSize: number): PrefillFrame[] {
  const frames: PrefillFrame[] = [];
  let prefillRemaining = longPrefillLength;
  const decodeTokensGenerated = new Array(decodeRequestCount).fill(0);
  const targetDecodeTokens = 10;

  if (chunkSize === 0) {
    frames.push({ frame: 0, executing: [{ requestId: 0, type: "prefill", tokenRange: [0, longPrefillLength], color: REQUEST_COLORS[0] }], decodeLatencies: [] });
    for (let step = 0; step < targetDecodeTokens; step++) {
      const executing = [];
      for (let i = 0; i < decodeRequestCount; i++) {
        executing.push({ requestId: i + 1, type: "decode" as const, color: REQUEST_COLORS[(i + 1) % REQUEST_COLORS.length] });
      }
      frames.push({ frame: frames.length, executing, decodeLatencies: [longPrefillLength + step] });
    }
  } else {
    let step = 0;
    while (prefillRemaining > 0 || decodeTokensGenerated.some((d) => d < targetDecodeTokens)) {
      const executing = [];
      if (prefillRemaining > 0) {
        const chunk = Math.min(chunkSize, prefillRemaining);
        const start = longPrefillLength - prefillRemaining;
        executing.push({ requestId: 0, type: "prefill" as const, tokenRange: [start, start + chunk] as [number, number], color: REQUEST_COLORS[0] });
        prefillRemaining -= chunk;
      }
      for (let i = 0; i < decodeRequestCount; i++) {
        if (decodeTokensGenerated[i] < targetDecodeTokens) {
          executing.push({ requestId: i + 1, type: "decode" as const, color: REQUEST_COLORS[(i + 1) % REQUEST_COLORS.length] });
          decodeTokensGenerated[i]++;
        }
      }
      frames.push({ frame: step, executing, decodeLatencies: [step] });
      step++;
    }
  }
  return frames;
}
