import { useEffect, useMemo, useState } from "react";

export function QuillLogo({ size = 40, animate = false }: { size?: number; animate?: boolean }) {
  const d = animate ? "anim-draw" : "";
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" className="shrink-0">
      <defs>
        <linearGradient id="qg" x1="0" y1="0" x2="64" y2="64">
          <stop offset="0" stopColor="var(--gold)" />
          <stop offset="1" stopColor="var(--indigo)" />
        </linearGradient>
      </defs>
      {/* neural attention lines */}
      <g stroke="var(--indigo)" strokeWidth="1" opacity="0.7">
        <path className={d} d="M10 46 Q 24 30 40 14" />
        <path className={d} d="M12 54 Q 30 40 52 24" style={{ animationDelay: ".15s" }} />
        <path className={d} d="M8 34 Q 26 34 46 10" style={{ animationDelay: ".3s" }} />
      </g>
      {[[10, 46], [40, 14], [52, 24], [8, 34], [46, 10], [12, 54]].map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="2" fill="var(--gold)" className={animate ? "anim-pop" : ""} style={{ animationDelay: `${0.4 + i * 0.08}s` }} />
      ))}
      {/* quill */}
      <path className={d} stroke="url(#qg)" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"
        d="M54 8 C 36 12, 22 26, 16 48 L 14 56 M54 8 C 50 24, 38 38, 18 46 M44 16 L 30 30 M48 20 L 36 30" />
    </svg>
  );
}

export function Splash({ onDone }: { onDone: () => void }) {
  const [leaving, setLeaving] = useState(false);
  const particles = useMemo(
    () => Array.from({ length: 40 }, () => ({ l: Math.random() * 100, t: 40 + Math.random() * 60, d: 3 + Math.random() * 4, s: 1 + Math.random() * 2.5, delay: Math.random() * 3 })),
    [],
  );
  const leave = () => { if (!leaving) { setLeaving(true); setTimeout(onDone, 550); } };
  useEffect(() => { const t = setTimeout(leave, 1800); return () => clearTimeout(t); });

  return (
    <div onClick={leave} className={`fixed inset-0 z-50 flex cursor-pointer flex-col items-center justify-center overflow-hidden bg-sidebar ${leaving ? "anim-splash-out" : ""}`}>
      {particles.map((p, i) => (
        <span key={i} className="anim-particle absolute rounded-full bg-gold"
          style={{ left: `${p.l}%`, top: `${p.t}%`, width: p.s, height: p.s, animationDuration: `${p.d}s`, animationDelay: `${p.delay}s`, boxShadow: "var(--glow-gold)" }} />
      ))}
      <div className="anim-pop rounded-full p-6 glow-gold"><QuillLogo size={110} animate /></div>
      <h1 className="anim-rise mt-8 font-display text-5xl md:text-6xl font-semibold text-title-gradient text-center" style={{ animationDelay: ".5s" }}>Shakespearean LLM</h1>
      <p className="anim-rise mt-3 px-6 text-center font-mono text-xs tracking-wide text-muted-foreground" style={{ animationDelay: ".8s" }}>
        Causal Language Modeling from Mathematical Primitives • 809k Parameters
      </p>
      <button onClick={(e) => { e.stopPropagation(); leave(); }} className="absolute bottom-8 right-8 rounded-full border px-4 py-1.5 text-xs text-muted-foreground hover:text-foreground">
        Skip →
      </button>
    </div>
  );
}
