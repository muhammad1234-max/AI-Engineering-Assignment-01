import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  ArrowUp, Check, ChevronDown, Copy, Cpu, Dices, Github, PanelLeft, Pencil, Plus, RefreshCw,
  SlidersHorizontal, Square, Trash2, User, X, Zap,
} from "lucide-react";
import { QuillLogo } from "./Brand";
import {
  CHIPS, DEFAULT_SETTINGS, STARTERS, fetchCompletion, pingApi, seedThreads, uid,
  type Message, type Settings, type Stats, type Thread,
} from "@/lib/zaidlm";

const LS = "zaidlm.threads.v1";
const LS_SET = "zaidlm.settings.v1";

/* ---------- small pieces ---------- */
function Verse({ text, streaming }: { text: string; streaming?: boolean }) {
  const lines = text.split("\n");
  return (
    <div className="font-mono text-[13.5px] leading-7 text-foreground/90">
      {lines.map((l, i) => {
        const m = l.match(/^([A-Z][A-Z .']{1,30}):\s*$/);
        const last = i === lines.length - 1;
        return (
          <div key={i} className={m ? "mt-3 first:mt-0" : "min-h-7 whitespace-pre-wrap"}>
            {m ? (
              <span className="inline-block rounded-md border border-gold/40 bg-gold-soft px-2 py-0.5 text-xs font-semibold tracking-widest text-gold">{m[1]}</span>
            ) : l}
            {streaming && last && <span className="cursor-blink">▋</span>}
          </div>
        );
      })}
    </div>
  );
}

function Wave() {
  return (
    <span className="ml-2 inline-flex h-4 items-center gap-0.5">
      {[0, 1, 2, 3, 4].map((i) => (
        <span key={i} className="wave-bar h-full w-0.5 rounded bg-gold" style={{ animationDelay: `${i * 0.12}s` }} />
      ))}
    </span>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt = (v: number) => String(v) }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string;
}) {
  return (
    <label className="block">
      <div className="mb-2 flex justify-between text-sm"><span>{label}</span><span className="font-mono text-gold">{fmt(value)}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-[var(--gold)]" />
    </label>
  );
}

/* ---------- main ---------- */
export function ChatApp() {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [collapsed, setCollapsed] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [archOpen, setArchOpen] = useState(false);
  const [modelMenu, setModelMenu] = useState(false);
  const [input, setInput] = useState("");
  const [online, setOnline] = useState<boolean | null>(null);
  const [gen, setGen] = useState<{ msgId: string; tokens: number; tps: number; latency: number } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const loaded = useRef(false);

  // load persisted state after mount
  useEffect(() => {
    try {
      const t = JSON.parse(localStorage.getItem(LS) || "null");
      setThreads(Array.isArray(t) && t.length ? t : seedThreads());
      const s = JSON.parse(localStorage.getItem(LS_SET) || "null");
      if (s) setSettings({ ...DEFAULT_SETTINGS, ...s });
    } catch { setThreads(seedThreads()); }
    if (window.innerWidth < 768) setCollapsed(true);
    loaded.current = true;
  }, []);
  useEffect(() => { if (loaded.current) localStorage.setItem(LS, JSON.stringify(threads)); }, [threads]);
  useEffect(() => { if (loaded.current) localStorage.setItem(LS_SET, JSON.stringify(settings)); }, [settings]);
  useEffect(() => { pingApi(settings.apiBase).then(setOnline); }, [settings.apiBase]);

  const active = threads.find((t) => t.id === activeId) ?? null;
  const messages = active?.messages ?? [];

  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight }); }, [messages]);
  useEffect(() => { taRef.current?.focus(); }, [activeId, gen === null]);
  useEffect(() => {
    const ta = taRef.current; if (!ta) return;
    ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [input]);

  const patchMsg = (tid: string, mid: string, p: Partial<Message>) =>
    setThreads((ts) => ts.map((t) => t.id !== tid ? t : { ...t, messages: t.messages.map((m) => (m.id === mid ? { ...m, ...p } : m)) }));

  const run = useCallback(async (tid: string, prompt: string, mid: string) => {
    const ac = new AbortController(); abortRef.current = ac;
    setGen({ msgId: mid, tokens: 0, tps: 0, latency: 0 });
    try {
      const { text = "", latency, source } = await fetchCompletion(prompt, settings, ac.signal);
      if (source === "demo") setOnline(false); else setOnline(true);
      const start = performance.now();
      const targetTps = 95 + Math.random() * 25;
      let i = 0;
      await new Promise<void>((resolve) => {
        const tick = () => {
          if (ac.signal.aborted) return resolve();
          const elapsed = (performance.now() - start) / 1000;
          const want = Math.min(text.length, Math.floor(elapsed * targetTps) + 1);
          if (want > i) {
            i = want;
            patchMsg(tid, mid, { content: text.slice(0, i) });
            setGen({ msgId: mid, tokens: i, tps: i / Math.max(elapsed, 0.01), latency });
          }
          if (i >= text.length) return resolve();
          setTimeout(tick, 8 + Math.random() * 14);
        };
        tick();
      });
      const el = (performance.now() - start) / 1000;
      const stats: Stats = { tokens: i, tps: i / Math.max(el, 0.01), elapsed: el + latency / 1000, latency, source };
      patchMsg(tid, mid, { stats });
    } catch { /* aborted */ }
    setGen(null); abortRef.current = null;
  }, [settings]);

  const send = (text?: string) => {
    const prompt = (text ?? input).replace(/\s+$/, "");
    if (!prompt.trim() || gen) return;
    let tid = activeId;
    const user: Message = { id: uid(), role: "user", content: prompt };
    const asst: Message = { id: uid(), role: "assistant", content: "" };
    if (!tid || !active) {
      tid = uid();
      const title = prompt.split("\n").find((l) => l.trim() && !/:\s*$/.test(l))?.slice(0, 40) || prompt.slice(0, 40);
      setThreads((ts) => [{ id: tid!, title: "✒️ " + title, createdAt: Date.now(), messages: [user, asst] }, ...ts]);
      setActiveId(tid);
    } else {
      setThreads((ts) => ts.map((t) => (t.id === tid ? { ...t, messages: [...t.messages, user, asst] } : t)));
    }
    setInput("");
    run(tid, prompt, asst.id);
  };

  const regenerate = (mid: string) => {
    if (!active || gen) return;
    const idx = active.messages.findIndex((m) => m.id === mid);
    const prompt = active.messages[idx - 1]?.content; if (!prompt) return;
    patchMsg(active.id, mid, { content: "", stats: undefined });
    run(active.id, prompt, mid);
  };

  const stop = () => abortRef.current?.abort();
  const copy = (t: string) => { navigator.clipboard.writeText(t); toast.success("Copied!"); };
  const newChat = () => { stop(); setActiveId(null); setInput(""); };

  const testConn = async () => {
    const ok = await pingApi(settings.apiBase); setOnline(ok);
    ok ? toast.success("API connected") : toast.error("Backend unreachable — using Standalone Demo Mode");
  };

  const dayMs = 86400000;
  const today = threads.filter((t) => Date.now() - t.createdAt < dayMs);
  const prev = threads.filter((t) => Date.now() - t.createdAt >= dayMs);

  const ThreadRow = (t: Thread) => (
    <div key={t.id} className={`group flex items-center gap-1 rounded-lg px-2 py-2 text-sm ${t.id === activeId ? "bg-accent" : "hover:bg-accent/60"}`}>
      {renaming === t.id ? (
        <input autoFocus defaultValue={t.title} className="w-full rounded bg-background px-1 py-0.5 outline-none ring-1 ring-ring"
          onBlur={(e) => { const v = e.target.value.trim(); setThreads((ts) => ts.map((x) => (x.id === t.id ? { ...x, title: v || x.title } : x))); setRenaming(null); }}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()} />
      ) : (
        <button onClick={() => { if (!gen) setActiveId(t.id); }} className="min-w-0 flex-1 truncate text-left">{t.title}</button>
      )}
      <button aria-label="Rename" onClick={() => setRenaming(t.id)} className="hidden p-1 text-muted-foreground hover:text-foreground group-hover:block"><Pencil size={13} /></button>
      <button aria-label="Delete" onClick={() => { setThreads((ts) => ts.filter((x) => x.id !== t.id)); if (activeId === t.id) setActiveId(null); }} className="hidden p-1 text-muted-foreground hover:text-destructive group-hover:block"><Trash2 size={13} /></button>
    </div>
  );

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <aside className={`flex shrink-0 flex-col border-r bg-sidebar transition-all duration-300 ${collapsed ? "w-0 overflow-hidden border-r-0" : "w-72"}`}>
        <div className="flex items-center justify-between p-3">
          <div className="flex items-center gap-2"><QuillLogo size={28} /><span className="font-display text-xl font-semibold text-title-gradient">Shakespearean LLM</span></div>
          <button aria-label="Collapse sidebar" onClick={() => setCollapsed(true)} className="rounded-md p-2 text-muted-foreground hover:bg-accent"><PanelLeft size={18} /></button>
        </div>
        <div className="px-3">
          <button onClick={newChat} className="flex w-full items-center gap-2 rounded-lg border px-3 py-2.5 text-sm hover:border-gold/50 hover:bg-accent"><Plus size={16} /> New Chat</button>
        </div>
        <nav className="mt-4 flex-1 overflow-y-auto px-2">
          {today.length > 0 && <p className="px-2 pb-1 text-xs font-medium text-muted-foreground">Today</p>}
          {today.map(ThreadRow)}
          {prev.length > 0 && <p className="mt-4 px-2 pb-1 text-xs font-medium text-muted-foreground">Previous 7 Days</p>}
          {prev.map(ThreadRow)}
        </nav>
        <div className="m-3 rounded-xl border bg-elevated p-3 text-xs">
          <div className="flex items-center gap-2 font-mono text-gold"><Cpu size={14} /> best_model.pt</div>
          <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-muted-foreground">
            <span>Val Loss <b className="text-foreground">2.3384</b></span><span>PPL <b className="text-foreground">10.36</b></span>
          </div>
          <div className="mt-3 flex gap-2">
            <button onClick={() => setArchOpen(true)} className="flex-1 rounded-md border px-2 py-1.5 hover:border-gold/50">Model Architecture</button>
            <a href="https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01" target="_blank" rel="noreferrer" aria-label="GitHub repository" className="rounded-md border px-2 py-1.5 hover:border-gold/50"><Github size={14} /></a>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="relative flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 px-3 py-2.5">
          {collapsed && <button aria-label="Expand sidebar" onClick={() => setCollapsed(false)} className="rounded-md p-2 text-muted-foreground hover:bg-accent"><PanelLeft size={18} /></button>}
          {collapsed && <button aria-label="New chat" onClick={newChat} className="rounded-md p-2 text-muted-foreground hover:bg-accent"><Plus size={18} /></button>}
          <div className="relative">
            <button onClick={() => setModelMenu((v) => !v)} className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-semibold hover:bg-accent">
              Shakespearean LLM-809k <span className="font-normal text-muted-foreground">(Pre-LN GPT)</span><ChevronDown size={14} />
            </button>
            {modelMenu && (
              <div className="absolute left-0 top-10 z-30 w-72 rounded-xl border bg-popover p-4 text-xs shadow-dock anim-rise" onMouseLeave={() => setModelMenu(false)}>
                <p className="font-semibold text-foreground">Pre-LN Causal Transformer</p>
                <ul className="mt-2 space-y-1 font-mono text-muted-foreground">
                  <li>params · 809,984</li><li>layers · 4 · heads · 4</li><li>context · 128 chars</li><li>data · Tiny Shakespeare</li>
                </ul>
                <div className="mt-3 flex items-center gap-2 text-gold"><Check size={14} /> Selected</div>
              </div>
            )}
          </div>
          <span className={`ml-1 flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${online ? "border-success/40 text-success" : "border-indigo/50 text-indigo"}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${online ? "bg-success" : "bg-indigo"} anim-breathe`} />
            {online === null ? "Checking…" : online ? "API Connected" : "Standalone Demo Mode"}
          </span>
          <button aria-label="Tuning deck" onClick={() => setDrawer(true)} className="ml-auto rounded-md p-2 text-muted-foreground hover:bg-accent hover:text-gold"><SlidersHorizontal size={18} /></button>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          {!active ? (
            <div className="mx-auto flex h-full max-w-3xl flex-col items-center justify-center px-4 pb-40">
              <div className="anim-pop rounded-full p-4 glow-gold"><QuillLogo size={64} /></div>
              <h2 className="anim-rise mt-6 text-center font-display text-4xl font-medium md:text-5xl">What Shakespearean drama shall we summon today?</h2>
              <div className="mt-10 grid w-full gap-3 sm:grid-cols-2">
                {STARTERS.map((s, i) => (
                  <button key={s.title} onClick={() => send(s.prompt)} className="anim-rise group rounded-xl border bg-card p-4 text-left transition hover:-translate-y-0.5 hover:border-gold/50" style={{ animationDelay: `${0.1 + i * 0.07}s` }}>
                    <div className="text-sm font-semibold">{s.emoji} {s.title}</div>
                    <div className="mt-1 line-clamp-2 whitespace-pre-line font-mono text-xs text-muted-foreground group-hover:text-foreground/80">{s.prompt}</div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-8 px-4 pb-56 pt-6">
              {messages.map((m) => m.role === "user" ? (
                <div key={m.id} className="anim-rise flex justify-end gap-3">
                  <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-secondary px-4 py-3 font-mono text-sm">{m.content}</div>
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent"><User size={16} /></div>
                </div>
              ) : (
                <div key={m.id} className="group anim-rise flex gap-3">
                  <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-gold/40 bg-sidebar ${gen?.msgId === m.id ? "anim-breathe" : ""}`}><QuillLogo size={20} /></div>
                  <div className="min-w-0 flex-1 pt-1">
                    {gen?.msgId === m.id && !m.content ? (
                      <div className="flex items-center text-sm text-muted-foreground">Summoning verse<Wave /></div>
                    ) : (
                      <Verse text={m.content} streaming={gen?.msgId === m.id} />
                    )}
                    {m.stats && gen?.msgId !== m.id && (
                      <div className="mt-3 flex items-center gap-1 opacity-60 transition group-hover:opacity-100">
                        <button aria-label="Copy" onClick={() => copy(m.content)} className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"><Copy size={14} /></button>
                        <button aria-label="Regenerate" onClick={() => regenerate(m.id)} className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground"><RefreshCw size={14} /></button>
                        <span className="ml-1 rounded-full border px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                          {m.stats.tps.toFixed(1)} tok/s • {m.stats.elapsed.toFixed(2)}s{m.stats.source === "demo" ? " • demo" : ""}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Dock */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-background via-background to-transparent pt-10">
          <div className="pointer-events-auto mx-auto max-w-3xl px-4 pb-3">
            {gen && (
              <div className="mx-auto mb-2 flex w-fit items-center gap-2 rounded-full border border-gold/30 bg-elevated px-3 py-1 font-mono text-[11px] text-gold anim-rise">
                <Zap size={12} /> Generating: {gen.tokens} tokens • {gen.tps.toFixed(0)} tok/s • Latency: {Math.round(gen.latency)}ms
              </div>
            )}
            <div className="mb-2 flex flex-wrap gap-1.5">
              {CHIPS.map((c) => (
                <button key={c} onClick={() => { setInput((v) => (v && !v.endsWith("\n") ? v + "\n" : v) + c + "\n"); taRef.current?.focus(); }}
                  className="rounded-full border bg-elevated px-2.5 py-1 font-mono text-[11px] text-muted-foreground hover:border-gold/50 hover:text-gold">{c}</button>
              ))}
            </div>
            <div className="flex items-end gap-2 rounded-3xl border bg-elevated p-2 pl-4 shadow-dock focus-within:border-gold/40">
              <textarea ref={taRef} rows={1} value={input} onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder="Speak, my lord, or type a Shakespearean prompt... (Shift+Enter for newline)"
                className="max-h-[200px] flex-1 resize-none bg-transparent py-2 font-mono text-sm outline-none placeholder:text-muted-foreground" />
              {gen ? (
                <button aria-label="Stop" onClick={stop} className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground anim-breathe"><Square size={14} fill="currentColor" /></button>
              ) : (
                <button aria-label="Send" onClick={() => send()} disabled={!input.trim()} className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground transition glow-gold disabled:opacity-30 disabled:shadow-none"><ArrowUp size={18} /></button>
              )}
            </div>
            <p className="mt-2 text-center text-[11px] text-muted-foreground">
              Shakespearean LLM is a compact 809k Pre-LN Transformer trained from scratch on Tiny Shakespeare. Outputs emulate 16th-century theatrical verse.
            </p>
          </div>
        </div>
      </main>

      {/* Drawer */}
      {drawer && <div className="fixed inset-0 z-40 bg-background/60 backdrop-blur-sm" onClick={() => setDrawer(false)} />}
      <aside className={`fixed right-0 top-0 z-50 h-full w-full max-w-sm overflow-y-auto border-l bg-popover p-6 transition-transform duration-300 ${drawer ? "translate-x-0" : "translate-x-full"}`}>
        <div className="mb-6 flex items-center justify-between">
          <h3 className="font-display text-2xl">Tuning Deck</h3>
          <button aria-label="Close" onClick={() => setDrawer(false)} className="rounded-md p-1.5 hover:bg-accent"><X size={18} /></button>
        </div>
        <div className="space-y-6">
          <div>
            <Slider label="Temperature" value={settings.temperature} min={0.1} max={1.5} step={0.05} fmt={(v) => v.toFixed(2)} onChange={(v) => setSettings({ ...settings, temperature: v })} />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {[["Deterministic", 0.3], ["Shakespearean Poetic", 0.8], ["Wild Creative", 1.2]].map(([l, v]) => (
                <button key={l} onClick={() => setSettings({ ...settings, temperature: v as number })}
                  className={`rounded-full border px-2.5 py-1 text-[11px] ${settings.temperature === v ? "border-gold text-gold" : "text-muted-foreground"}`}>{l} ({v})</button>
              ))}
            </div>
          </div>
          <Slider label="Max New Tokens" value={settings.maxTokens} min={20} max={256} step={1} onChange={(v) => setSettings({ ...settings, maxTokens: v })} />
          <Slider label="Top-k" value={settings.topK} min={0} max={65} step={1} onChange={(v) => setSettings({ ...settings, topK: v })} />
          <Slider label="Top-p (Nucleus)" value={settings.topP} min={0.1} max={1} step={0.01} fmt={(v) => v.toFixed(2)} onChange={(v) => setSettings({ ...settings, topP: v })} />
          <Slider label="Repetition Penalty" value={settings.repPenalty} min={1} max={1.5} step={0.01} fmt={(v) => v.toFixed(2)} onChange={(v) => setSettings({ ...settings, repPenalty: v })} />
          <label className="block text-sm">RNG Seed
            <div className="mt-2 flex gap-2">
              <input value={settings.seed} inputMode="numeric" placeholder="random" onChange={(e) => setSettings({ ...settings, seed: e.target.value.replace(/\D/g, "") })} className="flex-1 rounded-md border bg-background px-3 py-2 font-mono text-sm outline-none focus:border-gold/50" />
              <button aria-label="Randomize seed" onClick={() => setSettings({ ...settings, seed: String(Math.floor(Math.random() * 99999)) })} className="rounded-md border px-3 hover:border-gold/50 hover:text-gold"><Dices size={16} /></button>
            </div>
          </label>
          <label className="block text-sm">Backend API Base URL
            <div className="mt-2 flex gap-2">
              <input value={settings.apiBase} onChange={(e) => setSettings({ ...settings, apiBase: e.target.value })} className="flex-1 rounded-md border bg-background px-3 py-2 font-mono text-sm outline-none focus:border-gold/50" />
              <button onClick={testConn} className="rounded-md border px-3 text-xs hover:border-gold/50 hover:text-gold">Test</button>
            </div>
          </label>
          <button onClick={() => setSettings(DEFAULT_SETTINGS)} className="text-xs text-muted-foreground underline">Reset to defaults</button>
        </div>
      </aside>

      {/* Architecture modal */}
      {archOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/70 p-4 backdrop-blur-sm" onClick={() => setArchOpen(false)}>
          <div className="anim-pop w-full max-w-md rounded-2xl border bg-popover p-6" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-3"><QuillLogo size={36} /><h3 className="font-display text-2xl">Model Architecture</h3></div>
            <table className="mt-5 w-full font-mono text-xs">
              <tbody className="[&_td]:py-1.5 [&_tr]:border-b">
                {[["Type", "Pre-LN Causal Transformer"], ["Layers", "4"], ["Attention heads", "4"], ["Context length", "128"], ["Tokenizer", "Character-level"], ["Dataset", "Tiny Shakespeare"], ["Val loss / PPL", "2.3384 / 10.36"]].map(([k, v]) => (
                  <tr key={k}><td className="text-muted-foreground">{k}</td><td className="text-right">{v}</td></tr>
                ))}
                <tr><td className="font-semibold text-gold">Total parameters</td><td className="text-right font-semibold text-gold">809,984</td></tr>
              </tbody>
            </table>
            <button onClick={() => setArchOpen(false)} className="mt-6 w-full rounded-lg bg-primary py-2 text-sm font-semibold text-primary-foreground">Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
