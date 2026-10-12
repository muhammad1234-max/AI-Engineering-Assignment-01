export type Stats = { tokens: number; tps: number; elapsed: number; latency: number; source: "api" | "demo" };
export type Message = { id: string; role: "user" | "assistant"; content: string; stats?: Stats | undefined };
export type Thread = { id: string; title: string; createdAt: number; messages: Message[] };
export type Settings = {
  temperature: number; maxTokens: number; topK: number; topP: number;
  repPenalty: number; seed: string; apiBase: string;
};

export const DEFAULT_SETTINGS: Settings = {
  temperature: 0.8, maxTokens: 80, topK: 40, topP: 0.9, repPenalty: 1.05, seed: "", apiBase: "http://localhost:8000",
};

export const uid = () => Math.random().toString(36).slice(2, 10);

export const STARTERS = [
  { emoji: "🎭", title: "Romeo's Passion", prompt: "ROMEO:\nLady, by yonder blessed moon I vow," },
  { emoji: "👑", title: "Royal Decree", prompt: "KING CLAUDIUS:\nThough yet of Hamlet our dear brother's death," },
  { emoji: "💀", title: "Existential Soliloquy", prompt: "HAMLET:\nTo be, or not to be: that is the question:" },
  { emoji: "⚔️", title: "A King's Ransom", prompt: "KING RICHARD:\nA horse! a horse! my kingdom for a horse!" },
];

export const CHIPS = ["ROMEO:", "JULIET:", "HAMLET:", "KING:", "MACBETH:", "OTHELLO:"];

const MOCKS: Record<string, string[]> = {
  ROMEO: [
    "\nThat tips with silver all these fruit-tree tops,\nI swear my heart hath ne'er so truly beat.\n\nJULIET:\nO, swear not by the moon, the inconstant moon,\nThat monthly changes in her circled orb,\nLest that thy love prove likewise variable.",
    "\nAnd by the stars that keep their watch above,\nI'll be thy servant till the day doth break.\n\nJULIET:\nWhat say you, good my lord? The night is cold,\nAnd I would have thee gone, and yet not gone.",
  ],
  CLAUDIUS: [
    "\nThe memory be green, and that it us befitted\nTo bear our hearts in grief, and our whole kingdom\nTo be contracted in one brow of woe.\n\nQUEEN GERTRUDE:\nGood Hamlet, cast thy nighted colour off,\nAnd let thine eye look like a friend on Denmark.",
  ],
  HAMLET: [
    "\nWhether 'tis nobler in the mind to suffer\nThe slings and arrows of outrageous fortune,\nOr to take arms against a sea of troubles\nAnd by opposing end them. To die, to sleep,\nNo more; and by a sleep to say we end\nThe heart-ache that the flesh is heir to.\n\nOPHELIA:\nGood my lord, how does your honour for this many a day?",
    "\nWhat is this quintessence of dust to me?\nI have of late lost all my mirth, my lord,\nAnd this most goodly frame, the earth, seems sterile.",
  ],
  RICHARD: [
    "\nSlave, I have set my life upon a cast,\nAnd I will stand the hazard of the die.\nI think there be six Richmonds in the field;\nFive have I slain to-day instead of him.\n\nCATESBY:\nWithdraw, my lord; I'll help you to a horse.",
  ],
  DEFAULT: [
    "\nWhat, ho! my lord, the night is far too spent,\nAnd all the city sleeps upon its fears.\nI pray thee, speak, and let thy tongue be true.\n\nFIRST CITIZEN:\nWe are accounted poor citizens, the patricians good.\nWhat authority surfeits on would relieve us.",
    "\nThou art more lovely than the summer's day,\nAnd yet thy words do wound me like a blade.\nCome, let us hence; the king hath sent for thee.\n\nMACBETH:\nIs this a dagger which I see before me,\nThe handle toward my hand? Come, let me clutch thee.",
  ],
};

function seeded(seed: string) {
  let h = seed ? [...seed].reduce((a, c) => (a * 31 + c.charCodeAt(0)) | 0, 7) : Math.floor(Math.random() * 1e9);
  return () => ((h = (h * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
}

export function mockCompletion(prompt: string, s: Settings) {
  const up = prompt.toUpperCase();
  const key = Object.keys(MOCKS).find((k) => k !== "DEFAULT" && up.includes(k)) ?? "DEFAULT";
  const pool = MOCKS[key] ?? MOCKS["DEFAULT"]!;
  const rnd = seeded(s.seed);
  let text = pool[Math.floor(rnd() * pool.length)] ?? "";
  // higher temperature → append a slightly "wilder" char-level tail, like a tiny char model would
  if (s.temperature > 1.1) text += "\nAnd hor sow, lild sheng bat allee, my liege.";
  const max = Math.max(20, s.maxTokens * 3);
  return text.length > max ? text.slice(0, text.lastIndexOf("\n", max) > 20 ? text.lastIndexOf("\n", max) : max) : text;
}

export async function pingApi(base: string): Promise<boolean> {
  try {
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), 4000);
    const res = await fetch(base.replace(/\/$/, "") + "/health", { signal: c.signal, mode: "cors" });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

/** Fetch completion from backend, fall back to demo text. Returns full text + metadata. */
export async function fetchCompletion(prompt: string, s: Settings, signal: AbortSignal) {
  const started = performance.now();
  try {
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), 30000);
    signal.addEventListener("abort", () => c.abort());
    const res = await fetch(s.apiBase.replace(/\/$/, "") + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: c.signal,
      body: JSON.stringify({
        prompt, max_new_tokens: s.maxTokens, temperature: s.temperature, top_k: s.topK,
        top_p: s.topP, repetition_penalty: s.repPenalty, seed: s.seed ? Number(s.seed) : undefined,
      }),
    });
    clearTimeout(t);
    if (!res.ok) throw new Error("bad status");
    const j = await res.json();
    return { text: String(j.completion ?? ""), latency: performance.now() - started, source: "api" as const };
  } catch {
    if (signal.aborted) throw new DOMException("aborted", "AbortError");
    await new Promise((r) => setTimeout(r, 300 + Math.random() * 250));
    return { text: mockCompletion(prompt, s), latency: performance.now() - started, source: "demo" as const };
  }
}

const day = 86400000;
export function seedThreads(): Thread[] {
  const now = Date.now();
  const mk = (title: string, ago: number, prompt: string, key: string, tps: number): Thread => { const c = MOCKS[key]![0]!; return ({
    id: uid(), title, createdAt: now - ago,
    messages: [
      { id: uid(), role: "user", content: prompt },
      { id: uid(), role: "assistant", content: c, stats: { tokens: c.length, tps, elapsed: c.length / tps, latency: 410, source: "demo" } },
    ],
  }); };
  return [
    mk("🎭 Romeo's Balcony Soliloquy", 1000 * 60 * 20, STARTERS[0]!.prompt, "ROMEO", 118.4),
    mk("👑 Claudius's Royal Address", 1000 * 60 * 180, STARTERS[1]!.prompt, "CLAUDIUS", 112.9),
    mk("💀 Hamlet on Being and Nothingness", day * 2, STARTERS[2]!.prompt, "HAMLET", 121.2),
    mk("⚔️ Richard's Battle Cry", day * 4, STARTERS[3]!.prompt, "RICHARD", 109.7),
  ];
}
