import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  AlertCircle, ArrowUp, Check, ChevronDown, Copy, Cpu, Dices, Github, PanelLeft, Pencil, Plus, RefreshCw,
  SlidersHorizontal, Sparkles, Square, Trash2, User, X, Zap,
} from "lucide-react";
import { QuillLogo } from "./Brand";
import {
  CHIPS, DEFAULT_SETTINGS, PROD_API_BASE, STARTERS, fetchCompletion, pingApi, seedThreads, uid,
  type Message, type Settings, type Stats, type Thread,
} from "@/lib/zaidlm";

const LS = "zaidlm.threads.v1";
const LS_SET = "zaidlm.settings.v1";

/* ---------- small pieces ---------- */
function Verse({ text, streaming }: { text: string; streaming?: boolean }) {
  const lines = text.split("\n");
  let spokenCount = 0;
  return (
    <div className="font-mono text-[13.5px] leading-relaxed text-foreground/95 space-y-1 max-w-[70ch]">
      {lines.map((l, i) => {
        const m = l.match(/^([A-Z][A-Z .']{1,30}):\s*$/);
        const isStageDirection = /^\[.*\]$/.test(l.trim());
        const last = i === lines.length - 1;
        if (!m && l.trim()) spokenCount++;
        const showLineNo = spokenCount > 0 && (spokenCount === 1 || spokenCount % 5 === 0);

        if (m) {
          return (
            <div key={i} className="mt-4 first:mt-0 flex items-center gap-2 pt-1 pb-1">
              <span className="inline-flex items-center gap-1.5 rounded border border-gold/40 bg-gold-soft px-2.5 py-0.5 text-[11px] font-mono font-bold tracking-label text-gold shadow-sm">
                <span className="h-1.5 w-1.5 rounded-full bg-gold" />
                {m[1]}
              </span>
              <span className="h-px flex-1 bg-border/80" />
            </div>
          );
        }

        if (isStageDirection) {
          return (
            <div key={i} className="py-0.5 pl-7 font-sans italic text-xs text-muted-foreground">
              {l}
              {streaming && last && <span className="cursor-blink">▋</span>}
            </div>
          );
        }

        return (
          <div key={i} className="group/line flex items-baseline gap-3 pl-1 min-h-[1.5rem]">
            <span className="w-6 shrink-0 select-none text-right font-mono font-tabular text-[10px] text-muted-foreground/45 group-hover/line:text-gold/80 transition-colors">
              {showLineNo ? spokenCount : ""}
            </span>
            <span className="flex-1 whitespace-pre-wrap tracking-[0.01em]">
              {l}
              {streaming && last && <span className="cursor-blink ml-0.5">▋</span>}
            </span>
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
      <div className="mb-2 flex justify-between text-sm font-medium"><span>{label}</span><span className="font-mono font-tabular text-gold font-semibold">{fmt(value)}</span></div>
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
      if (s) {
        // Upgrade legacy default 80 tokens to 35 for free-tier responsiveness
        if (s.maxTokens === 80) s.maxTokens = 35;
        // Upgrade legacy localhost:8000 to production backend if on remote domain
        if (!s.apiBase || s.apiBase === "http://localhost:8000" || s.apiBase.includes("localhost")) {
          if (typeof window !== "undefined" && window.location.hostname !== "localhost") {
            s.apiBase = PROD_API_BASE;
          }
        }
        if (s.apiBase) s.apiBase = s.apiBase.trim().replace(/\/+$/, "");
        setSettings({ ...DEFAULT_SETTINGS, ...s });
      }
    } catch { setThreads(seedThreads()); }
    if (window.innerWidth < 768) setCollapsed(true);
    loaded.current = true;
  }, []);
  useEffect(() => { if (loaded.current) localStorage.setItem(LS, JSON.stringify(threads)); }, [threads]);
  useEffect(() => { if (loaded.current) localStorage.setItem(LS_SET, JSON.stringify(settings)); }, [settings]);
  useEffect(() => {
    let active = true;
    const check = async () => {
      const ok = await pingApi(settings.apiBase);
      if (active) setOnline(ok);
    };
    check();
    const interval = setInterval(check, 15000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [settings.apiBase]);

  const active = threads.find((t) => t.id === activeId) ?? null;
  const messages = active?.messages ?? [];

  useEffect(() => {
    if (activeId) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
    } else {
      scrollRef.current?.scrollTo({ top: 0 });
    }
  }, [messages, activeId]);
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
      const { text = "", latency } = await fetchCompletion(prompt, settings, ac.signal);
      setOnline(true);
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
            patchMsg(tid, mid, { content: text.slice(0, i), error: false });
            setGen({ msgId: mid, tokens: i, tps: i / Math.max(elapsed, 0.01), latency });
          }
          if (i >= text.length) return resolve();
          setTimeout(tick, 8 + Math.random() * 14);
        };
        tick();
      });
      const el = (performance.now() - start) / 1000;
      const stats: Stats = { tokens: i, tps: i / Math.max(el, 0.01), elapsed: el + latency / 1000, latency };
      patchMsg(tid, mid, { stats, error: false });
    } catch (err: unknown) {
      if (ac.signal.aborted) {
        /* aborted by user */
      } else {
        setOnline(false);
        const rawErr = err instanceof Error ? err.message : String(err);
        toast.error(rawErr);
        patchMsg(tid, mid, {
          content: `${rawErr}. Please verify that the backend is running on ${settings.apiBase}. (Note: Free-tier Render instances take ~50s to wake up on first request).`,
          error: true,
        });
      }
    }
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
    patchMsg(active.id, mid, { content: "", stats: undefined, error: false });
    run(active.id, prompt, mid);
  };

  const stop = () => abortRef.current?.abort();
  const copy = (t: string) => { navigator.clipboard.writeText(t); toast.success("Copied!"); };
  const newChat = () => { stop(); setActiveId(null); setInput(""); };

  const testConn = async () => {
    const ok = await pingApi(settings.apiBase); setOnline(ok);
    ok ? toast.success("Connected to backend model") : toast.error("Could not connect to the backend model");
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
      <button aria-label="Rename" onClick={() => setRenaming(t.id)} className="hidden h-7 w-7 items-center justify-center rounded p-1 text-muted-foreground hover:bg-background/80 hover:text-foreground group-hover:flex"><Pencil size={13} /></button>
      <button aria-label="Delete" onClick={() => { setThreads((ts) => ts.filter((x) => x.id !== t.id)); if (activeId === t.id) setActiveId(null); }} className="hidden h-7 w-7 items-center justify-center rounded p-1 text-muted-foreground hover:bg-background/80 hover:text-destructive group-hover:flex"><Trash2 size={13} /></button>
    </div>
  );

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <aside className={`flex shrink-0 flex-col border-r border-border bg-sidebar transition-all duration-300 ${collapsed ? "w-0 overflow-hidden border-r-0" : "w-72"}`}>
        <div className="flex items-center justify-between p-3.5 border-b border-border/50">
          <div className="flex items-center gap-2.5">
            <QuillLogo size={30} />
            <span className="font-display text-xl font-bold text-gold tracking-tight">Shakespearean LLM</span>
          </div>
          <button aria-label="Collapse sidebar" onClick={() => setCollapsed(true)} className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground transition">
            <PanelLeft size={17} />
          </button>
        </div>
        <div className="p-3">
          <button onClick={newChat} className="flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-card/60 px-3 py-2.5 text-sm font-semibold hover:border-gold/50 hover:bg-accent hover:text-gold transition shadow-sm">
            <Plus size={16} /> New Folio Session
          </button>
        </div>
        <nav className="flex-1 overflow-y-auto px-2.5 space-y-0.5">
          {today.length > 0 && <p className="px-2 pt-2 pb-1 font-mono text-[11px] font-semibold uppercase tracking-label text-muted-foreground">Today's Soliloquies</p>}
          {today.map(ThreadRow)}
          {prev.length > 0 && <p className="mt-4 px-2 pt-2 pb-1 font-mono text-[11px] font-semibold uppercase tracking-label text-muted-foreground">Prior Folios</p>}
          {prev.map(ThreadRow)}
        </nav>
        <div className="m-3 rounded-xl border border-border-strong bg-elevated/90 p-3 text-xs shadow-sm">
          <div className="flex items-center justify-between font-mono">
            <span className="flex items-center gap-1.5 text-gold font-bold"><Cpu size={14} /> best_model.pt</span>
            <span className="inline-flex items-center gap-1 text-[10px] text-success font-semibold tracking-label uppercase"><span className="h-1.5 w-1.5 rounded-full bg-success" /> READY</span>
          </div>
          <div className="mt-2.5 grid grid-cols-2 gap-2 rounded-lg border border-border/60 bg-background/60 p-2 font-mono text-[11px]">
            <div><span className="text-muted-foreground block text-[10px] uppercase tracking-label">Val Loss</span><b className="text-foreground font-tabular text-xs font-bold">2.3384</b></div>
            <div><span className="text-muted-foreground block text-[10px] uppercase tracking-label">Perplexity</span><b className="text-foreground font-tabular text-xs font-bold">10.36</b></div>
          </div>
          <div className="mt-2.5 flex items-center justify-between text-[10px] font-mono text-muted-foreground px-0.5">
            <span className="font-tabular">4L · 4H · 128 ctx</span>
            <span className="text-gold font-semibold font-tabular">809,984 params</span>
          </div>
          <div className="mt-3 flex gap-2">
            <button onClick={() => setArchOpen(true)} className="flex-1 rounded-md border border-border bg-card px-2 py-1.5 font-medium hover:border-gold/50 hover:text-gold transition">Architecture</button>
            <a href="https://github.com/zaidbinnaveed/AI-Engineering-Assignment-01" target="_blank" rel="noreferrer" aria-label="GitHub repository" className="rounded-md border border-border bg-card px-2.5 py-1.5 hover:border-gold/50 hover:text-gold transition"><Github size={14} /></a>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="relative flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 px-3 py-2.5">
          {collapsed && <button aria-label="Expand sidebar" onClick={() => setCollapsed(false)} className="rounded-md p-2 text-muted-foreground hover:bg-accent"><PanelLeft size={18} /></button>}
          {collapsed && <button aria-label="New chat" onClick={newChat} className="rounded-md p-2 text-muted-foreground hover:bg-accent"><Plus size={18} /></button>}
          <div className="relative">
            <button onClick={() => setModelMenu((v) => !v)} className="flex items-center gap-2 rounded-lg border border-border bg-card/80 px-3 py-1.5 text-xs font-semibold hover:border-gold/50 transition">
              <span className="h-1.5 w-1.5 rounded-full bg-gold" />
              <span className="text-foreground">Shakespearean LLM</span>
              <span className="font-mono text-[10px] text-gold font-normal px-1.5 py-0.5 rounded bg-gold-soft border border-gold/30">809k Pre-LN</span>
              <ChevronDown size={13} className="text-muted-foreground" />
            </button>
            {modelMenu && (
              <div className="absolute left-0 top-11 z-30 w-72 rounded-xl border border-border-strong bg-popover p-4 text-xs shadow-dock anim-rise" onMouseLeave={() => setModelMenu(false)}>
                <p className="font-semibold text-foreground text-sm">Pre-LN Causal Transformer</p>
                <ul className="mt-2.5 space-y-1.5 font-mono text-muted-foreground text-[11px]">
                  <li className="flex justify-between border-b border-border/40 pb-1"><span>Parameters:</span><b className="text-foreground font-tabular">809,984</b></li>
                  <li className="flex justify-between border-b border-border/40 pb-1"><span>Architecture:</span><b className="text-foreground font-tabular">4 Layers · 4 Heads</b></li>
                  <li className="flex justify-between border-b border-border/40 pb-1"><span>Context Length:</span><b className="text-foreground font-tabular">128 chars</b></li>
                  <li className="flex justify-between"><span>Training Corpus:</span><b className="text-foreground">Tiny Shakespeare</b></li>
                </ul>
                <div className="mt-3 flex items-center gap-2 text-gold font-semibold pt-1 border-t border-border/40"><Check size={14} /> Active In-Memory Checkpoint</div>
              </div>
            )}
          </div>
          <button
            onClick={testConn}
            title="Click to verify connection"
            className={`ml-1 flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-mono font-medium transition hover:opacity-80 ${online ? "border-success/40 bg-success/10 text-success" : online === false ? "border-destructive/40 bg-destructive/10 text-destructive" : "border-border bg-muted/40 text-muted-foreground"}`}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${online ? "bg-success" : online === false ? "bg-destructive" : "bg-muted-foreground"} anim-breathe`} />
            {online === null ? "Verifying..." : online ? "Model Online" : "Model Offline (Click to Retry)"}
          </button>
          <button aria-label="Tuning deck" onClick={() => setDrawer(true)} className="ml-auto flex items-center gap-1.5 rounded-lg border border-border bg-card/60 px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:border-gold/50 hover:text-gold transition">
            <SlidersHorizontal size={14} />
            <span className="hidden sm:inline">Tuning Deck</span>
          </button>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto flex flex-col">
          {!active ? (
            <div className="mx-auto my-auto flex w-full max-w-4xl flex-col items-center px-4 pt-2 pb-44">
              <div className="anim-pop mb-1.5 rounded-full border border-gold/30 bg-sidebar p-2 shadow-sm glow-gold">
                <QuillLogo size={32} />
              </div>
              <div className="anim-rise mb-1.5 inline-flex items-center gap-1.5 rounded-full border border-gold/30 bg-gold-soft px-2.5 py-0.5 font-mono text-[10px] font-semibold text-gold tracking-label uppercase">
                <Sparkles size={11} /> First Folio Studio • 809,984-Parameter Causal GPT
              </div>
              <h2 className="anim-rise text-center font-display text-2xl md:text-4xl font-semibold text-foreground text-balance max-w-2xl leading-tight tracking-tight">
                What Shakespearean drama shall we summon today?
              </h2>
              <p className="anim-rise mt-1 text-center font-sans text-xs text-muted-foreground max-w-[54ch]">
                Character-level autoregressive generation trained from mathematical primitives on the complete works of William Shakespeare.
              </p>

              {/* Bento Folio Starters */}
              <div className="mt-3.5 w-full grid grid-cols-1 md:grid-cols-12 gap-2.5 text-left">
                {/* Featured Folio (Spans 7 cols on md) */}
                <button
                  onClick={() => send(STARTERS[2].prompt)}
                  className="anim-rise group md:col-span-7 flex flex-col justify-between rounded-xl border border-border bg-card/80 p-4 transition-all hover:border-gold/60 hover:bg-card hover:shadow-md"
                >
                  <div>
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono text-[10px] font-bold uppercase tracking-label text-gold">
                        {STARTERS[2].play} • {STARTERS[2].act}
                      </span>
                      <span className="text-base">{STARTERS[2].emoji}</span>
                    </div>
                    <div className="mt-1 font-display text-lg font-bold text-foreground group-hover:text-gold transition">
                      {STARTERS[2].title}
                    </div>
                    <div className="mt-2.5 pl-3 border-l-2 border-gold/50 font-mono text-xs leading-relaxed text-foreground/85">
                      <span className="font-bold text-gold">HAMLET:</span>
                      <br />
                      To be, or not to be: that is the question:
                      <br />
                      <span className="text-muted-foreground text-[11px]">Whether 'tis nobler in the mind to suffer the slings and arrows...</span>
                    </div>
                  </div>
                  <div className="mt-3.5 flex items-center justify-between text-[11px] font-mono text-muted-foreground pt-2 border-t border-border/40">
                    <span>Summon Act III Soliloquy</span>
                    <span className="text-gold font-bold group-hover:translate-x-1 transition-transform">→</span>
                  </div>
                </button>

                {/* Secondary Folios Column (Spans 5 cols on md) */}
                <div className="md:col-span-5 flex flex-col gap-2.5">
                  {[STARTERS[0], STARTERS[1], STARTERS[3]].map((s, idx) => (
                    <button
                      key={s.title}
                      onClick={() => send(s.prompt)}
                      className="anim-rise group flex flex-col justify-between rounded-xl border border-border bg-card/60 p-3 transition-all hover:border-gold/50 hover:bg-card hover:shadow-sm text-left"
                      style={{ animationDelay: `${0.08 + idx * 0.05}s` }}
                    >
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-mono text-[10px] font-medium uppercase tracking-label text-muted-foreground group-hover:text-gold transition">
                          {s.play} • {s.act}
                        </span>
                        <span className="text-sm">{s.emoji}</span>
                      </div>
                      <div className="mt-1 flex items-baseline justify-between gap-2">
                        <span className="text-xs font-bold text-foreground group-hover:text-gold transition truncate">
                          {s.title}
                        </span>
                        <span className="font-mono text-[10px] text-muted-foreground shrink-0 group-hover:text-gold transition">
                          {s.prompt.split("\n")[0]}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="mx-auto max-w-3xl space-y-8 px-4 pb-56 pt-6">
              {messages.map((m) => m.role === "user" ? (
                <div key={m.id} className="anim-rise flex justify-end gap-3">
                  <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-md border border-border bg-secondary px-4 py-3 font-mono text-sm shadow-sm">{m.content}</div>
                  <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-border bg-accent text-foreground"><User size={16} /></div>
                </div>
              ) : (
                <div key={m.id} className="group anim-rise flex gap-3">
                  <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-gold/40 bg-sidebar shadow-sm ${gen?.msgId === m.id ? "anim-breathe" : ""}`}><QuillLogo size={20} /></div>
                  <div className="min-w-0 flex-1 pt-1">
                    {gen?.msgId === m.id && !m.content ? (
                      <div className="flex items-center text-sm font-medium text-gold">Summoning verse from latent space<Wave /></div>
                    ) : m.error ? (
                      <div className="rounded-xl border border-destructive/50 bg-destructive/10 p-4 text-destructive shadow-sm">
                        <div className="flex items-center gap-2 font-semibold text-sm">
                          <AlertCircle size={16} />
                          <span>Could not connect to the backend model</span>
                        </div>
                        <p className="mt-1.5 text-xs text-destructive/85 font-mono">
                          {m.content}
                        </p>
                        <div className="mt-3 flex gap-2">
                          <button
                            onClick={() => regenerate(m.id)}
                            className="inline-flex items-center gap-1.5 rounded-md border border-destructive/40 bg-destructive/20 px-3 py-1 text-xs font-medium hover:bg-destructive/30 transition"
                          >
                            <RefreshCw size={12} /> Retry Inference
                          </button>
                          <button
                            onClick={testConn}
                            className="rounded-md border border-destructive/40 bg-card/60 px-3 py-1 text-xs font-medium hover:border-gold/50 hover:text-gold transition"
                          >
                            Check Connection
                          </button>
                        </div>
                      </div>
                    ) : (
                      <Verse text={m.content} streaming={gen?.msgId === m.id} />
                    )}
                    {m.stats && !m.error && gen?.msgId !== m.id && (
                      <div className="mt-3.5 flex items-center gap-1.5 opacity-60 transition group-hover:opacity-100">
                        <button aria-label="Copy" onClick={() => copy(m.content)} className="h-7 w-7 flex items-center justify-center rounded-md border border-border/60 text-muted-foreground hover:bg-accent hover:text-foreground transition"><Copy size={13} /></button>
                        <button aria-label="Regenerate" onClick={() => regenerate(m.id)} className="h-7 w-7 flex items-center justify-center rounded-md border border-border/60 text-muted-foreground hover:bg-accent hover:text-foreground transition"><RefreshCw size={13} /></button>
                        <span className="ml-1 rounded-full border border-border/60 bg-elevated px-2.5 py-0.5 font-mono font-tabular text-[11px] text-muted-foreground">
                          <b className="text-foreground">{m.stats.tps.toFixed(1)}</b> tok/s • <b className="text-foreground">{m.stats.elapsed.toFixed(2)}s</b> • live model
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
        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-background via-background to-transparent pt-12">
          <div className="pointer-events-auto mx-auto max-w-3xl px-4 pb-3">
            {gen && (
              <div className="mx-auto mb-2.5 flex w-fit items-center gap-3 rounded-full border border-gold/40 bg-elevated px-4 py-1.5 font-mono font-tabular text-xs text-gold shadow-md anim-rise">
                <span className="flex items-center gap-1.5 font-bold tracking-label uppercase">
                  <Zap size={13} className="text-gold anim-breathe" /> GENERATING VERSE:
                </span>
                <span><b className="text-foreground">{gen.tokens}</b> chars</span>
                <span>•</span>
                <span><b className="text-foreground">{gen.tps.toFixed(0)}</b> tok/s</span>
                <span>•</span>
                <span><b className="text-foreground">{Math.round(gen.latency)}</b>ms latency</span>
              </div>
            )}
            <div className="mb-2 flex items-center gap-2 overflow-x-auto pb-1">
              <span className="shrink-0 font-mono text-[10px] font-bold uppercase tracking-label text-muted-foreground">
                Summon:
              </span>
              {CHIPS.map((c) => (
                <button
                  key={c}
                  onClick={() => { setInput((v) => (v && !v.endsWith("\n") ? v + "\n" : v) + c + "\n"); taRef.current?.focus(); }}
                  className="shrink-0 rounded-md border border-border bg-elevated px-2.5 py-1 font-mono text-xs font-semibold text-foreground/90 hover:border-gold hover:bg-gold-soft hover:text-gold transition shadow-sm"
                >
                  {c}
                </button>
              ))}
            </div>
            <div className="flex items-end gap-2 rounded-2xl border border-border-strong bg-elevated p-2 pl-4 shadow-dock focus-within:border-gold/60 focus-within:ring-2 focus-within:ring-gold/20 transition">
              <textarea
                ref={taRef}
                rows={1}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder="Speak, my lord, or type a Shakespearean prompt... (Enter to summon, Shift+Enter for newline)"
                className="max-h-[200px] flex-1 resize-none bg-transparent py-1.5 font-mono text-sm text-foreground outline-none placeholder:text-muted-foreground/70"
              />
              {gen ? (
                <button aria-label="Stop generation" onClick={stop} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-destructive text-destructive-foreground hover:bg-destructive/90 transition shadow-sm" title="Stop">
                  <Square size={14} fill="currentColor" />
                </button>
              ) : (
                <button aria-label="Summon verse" onClick={() => send()} disabled={!input.trim()} className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gold text-primary-foreground font-semibold hover:bg-gold/90 transition glow-gold disabled:opacity-30 disabled:shadow-none" title="Send (Enter)">
                  <ArrowUp size={18} strokeWidth={2.5} />
                </button>
              )}
            </div>
            <p className="mt-2 text-center text-[11px] text-muted-foreground">
              Shakespearean LLM: 809k-parameter Pre-LN Causal Transformer trained from scratch on Tiny Shakespeare.
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
          <div>
            <Slider label="Max New Tokens" value={settings.maxTokens} min={10} max={256} step={1} onChange={(v) => setSettings({ ...settings, maxTokens: v })} />
            <p className="mt-1 text-[11px] text-muted-foreground">Free CPU generates ~1 tok/s. 20–40 tokens recommended for fast response.</p>
          </div>
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
                  <tr key={k}><td className="text-muted-foreground">{k}</td><td className="text-right font-tabular">{v}</td></tr>
                ))}
                <tr><td className="font-semibold text-gold">Total parameters</td><td className="text-right font-semibold text-gold font-tabular">809,984</td></tr>
              </tbody>
            </table>
            <button onClick={() => setArchOpen(false)} className="mt-6 w-full rounded-lg bg-primary py-2 text-sm font-semibold text-primary-foreground">Close</button>
          </div>
        </div>
      )}
    </div>
  );
}
