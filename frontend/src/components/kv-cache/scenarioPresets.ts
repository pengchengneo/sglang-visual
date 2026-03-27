export interface Scenario {
  id: string;
  name: string;
  description: string;
  requests: {
    tokens: string[];
    arrivalFrame: number;
    durationFrames: number;
  }[];
  maxBlocks: number;
}

const SYSTEM_PROMPT = ["[SYS]", "You", "are", "a", "helpful", "assistant", "."];
const FEWSHOT_PREFIX = ["[SYS]", "Translate", "EN", "to", "CN", ":"];
const EXAMPLE_1 = ["Ex1:", "Hello", "→", "你好"];
const EXAMPLE_2 = ["Ex2:", "Thanks", "→", "谢谢"];

export const SCENARIOS: Scenario[] = [
  {
    id: "multi-turn",
    name: "Multi-Turn Dialogue",
    description: "Multiple conversations sharing the same system prompt",
    requests: [
      { tokens: [...SYSTEM_PROMPT, "User:", "What", "is", "AI", "?"], arrivalFrame: 0, durationFrames: 20 },
      { tokens: [...SYSTEM_PROMPT, "User:", "Tell", "me", "a", "joke"], arrivalFrame: 5, durationFrames: 15 },
      { tokens: [...SYSTEM_PROMPT, "User:", "What", "is", "AI", "?", "Bot:", "AI", "is", "...", "User:", "More", "details"], arrivalFrame: 10, durationFrames: 25 },
    ],
    maxBlocks: 20,
  },
  {
    id: "few-shot",
    name: "Few-Shot Sharing",
    description: "Multiple requests sharing the same few-shot examples",
    requests: [
      { tokens: [...FEWSHOT_PREFIX, ...EXAMPLE_1, ...EXAMPLE_2, "Input:", "Good", "morning"], arrivalFrame: 0, durationFrames: 18 },
      { tokens: [...FEWSHOT_PREFIX, ...EXAMPLE_1, ...EXAMPLE_2, "Input:", "Goodbye"], arrivalFrame: 3, durationFrames: 15 },
      { tokens: [...FEWSHOT_PREFIX, ...EXAMPLE_1, "Input:", "Nice", "weather"], arrivalFrame: 8, durationFrames: 20 },
    ],
    maxBlocks: 24,
  },
  {
    id: "no-sharing",
    name: "No Sharing",
    description: "Completely independent requests — no prefix sharing",
    requests: [
      { tokens: ["Summarize", "the", "article", "about", "climate"], arrivalFrame: 0, durationFrames: 15 },
      { tokens: ["Write", "Python", "code", "for", "sorting"], arrivalFrame: 4, durationFrames: 12 },
      { tokens: ["Explain", "quantum", "computing", "basics"], arrivalFrame: 8, durationFrames: 18 },
    ],
    maxBlocks: 20,
  },
  {
    id: "mixed",
    name: "Mixed Traffic",
    description: "Some requests share prefix, others don't",
    requests: [
      { tokens: [...SYSTEM_PROMPT, "User:", "Hello"], arrivalFrame: 0, durationFrames: 10 },
      { tokens: ["Code:", "def", "fib", "(", "n", ")", ":"], arrivalFrame: 3, durationFrames: 14 },
      { tokens: [...SYSTEM_PROMPT, "User:", "Explain", "SGLang"], arrivalFrame: 6, durationFrames: 20 },
      { tokens: ["Translate:", "Bonjour", "le", "monde"], arrivalFrame: 12, durationFrames: 12 },
      { tokens: [...SYSTEM_PROMPT, "User:", "Hello", "again"], arrivalFrame: 15, durationFrames: 16 },
    ],
    maxBlocks: 24,
  },
];
