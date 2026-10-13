export type Stats = { tokens: number; tps: number; elapsed: number; latency: number };
export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  stats?: Stats | undefined;
  error?: boolean | undefined;
};
export type Thread = { id: string; title: string; createdAt: number; messages: Message[] };
export type Settings = {
  temperature: number;
  maxTokens: number;
  topK: number;
  topP: number;
  repPenalty: number;
  seed: string;
  apiBase: string;
};

export const DEFAULT_SETTINGS: Settings = {
  temperature: 0.8,
  maxTokens: 80,
  topK: 40,
  topP: 0.9,
  repPenalty: 1.05,
  seed: "",
  apiBase: (import.meta.env.VITE_API_BASE_URL as string) || "http://localhost:8000",
};

export const uid = () => Math.random().toString(36).slice(2, 10);

export const STARTERS = [
  { emoji: "🎭", play: "Romeo & Juliet", act: "Act II, Sc. 2", title: "Romeo's Passion", prompt: "ROMEO:\nLady, by yonder blessed moon I vow," },
  { emoji: "👑", play: "Hamlet", act: "Act I, Sc. 2", title: "Royal Decree", prompt: "KING CLAUDIUS:\nThough yet of Hamlet our dear brother's death," },
  { emoji: "💀", play: "Hamlet", act: "Act III, Sc. 1", title: "Existential Soliloquy", prompt: "HAMLET:\nTo be, or not to be: that is the question:" },
  { emoji: "⚔️", play: "Richard III", act: "Act V, Sc. 4", title: "A King's Ransom", prompt: "KING RICHARD:\nA horse! a horse! my kingdom for a horse!" },
];

export const CHIPS = ["ROMEO:", "JULIET:", "HAMLET:", "KING CLAUDIUS:", "MACBETH:", "OTHELLO:"];

export async function pingApi(base: string): Promise<boolean> {
  try {
    const clean = base.trim().replace(/\/+$/, "");
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), 15000);
    const res = await fetch(clean + "/health", { signal: c.signal, mode: "cors" });
    clearTimeout(t);
    return res.ok;
  } catch {
    return false;
  }
}

/** Fetch completion strictly from the actual backend model. Throws an error if unavailable. */
export async function fetchCompletion(prompt: string, s: Settings, signal: AbortSignal): Promise<{ text: string; latency: number }> {
  const started = performance.now();
  const c = new AbortController();
  const timeoutId = setTimeout(() => c.abort(), 90000);
  const onAbort = () => c.abort();
  signal.addEventListener("abort", onAbort);

  try {
    const clean = s.apiBase.trim().replace(/\/+$/, "");
    const res = await fetch(clean + "/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: c.signal,
      body: JSON.stringify({
        prompt,
        max_new_tokens: s.maxTokens,
        temperature: s.temperature,
        top_k: s.topK,
        top_p: s.topP,
        repetition_penalty: s.repPenalty,
        seed: s.seed ? Number(s.seed) : undefined,
      }),
    });

    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try {
        const j = await res.json();
        if (j?.detail) detail = String(j.detail);
      } catch { /* ignore */ }
      throw new Error(detail);
    }

    const j = await res.json();
    return {
      text: String(j.completion ?? ""),
      latency: performance.now() - started,
    };
  } catch (err: unknown) {
    if (signal.aborted) {
      throw new DOMException("aborted", "AbortError");
    }
    const message = err instanceof Error ? err.message : "Network error";
    throw new Error(`Could not connect to the backend model (${message})`);
  } finally {
    clearTimeout(timeoutId);
    signal.removeEventListener("abort", onAbort);
  }
}

export function seedThreads(): Thread[] {
  return [];
}
