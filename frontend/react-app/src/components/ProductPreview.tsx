import { useEffect, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { Arrow } from "./Brand";

export function DocumentIcon({ small = false }: { small?: boolean }) {
  return (
    <svg
      className={small ? "document-icon small" : "document-icon"}
      viewBox="0 0 32 38"
      fill="none"
      aria-hidden="true"
    >
      <path d="M5 2h14l9 9v24H5z" />
      <path d="M19 2v10h9M10 20h13M10 25h13M10 30h8" />
    </svg>
  );
}

export function HeroVisual() {
  const visualRef = useRef<HTMLDivElement>(null);
  const frameRef = useRef<number | null>(null);
  const pointerRef = useRef<{ x: number; y: number } | null>(null);
  const planetRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const planetMotionRef = useRef<Record<string, { x: number; y: number }>>({});

  const planets = [
    { id: "documents", label: "Documents", detail: "Upload PDFs · Build your knowledge", ring: 0, phase: -1.9, speed: 0.00011, rx: 350, ry: 150, tone: "amber" },
    { id: "retrieval", label: "Retrieval", detail: "Find relevant context", ring: 1, phase: 2.65, speed: -0.00009, rx: 300, ry: 194, tone: "blue" },
    { id: "evidence", label: "Evidence", detail: "See source pages · Verify every claim", ring: 0, phase: -0.25, speed: 0.00008, rx: 350, ry: 150, tone: "pink" },
    { id: "security", label: "Security", detail: "Guardrails · Prompt-injection protection", ring: 2, phase: -2.4, speed: 0.0001, rx: 255, ry: 235, tone: "violet" },
    { id: "human-review", label: "Human Review", detail: "Review and improve", ring: 2, phase: 1.2, speed: 0.00007, rx: 255, ry: 235, tone: "cyan" },
    { id: "evaluation", label: "LLM Evaluation", detail: "Grounding · Source relevance · Quality", ring: 1, phase: -0.75, speed: -0.0001, rx: 300, ry: 194, tone: "magenta" },
    { id: "developers", label: "Developers", detail: "Explore AI reliability", ring: 0, phase: 2.1, speed: -0.00012, rx: 350, ry: 150, tone: "indigo" },
  ] as const;
  const planetTargets: Record<string, string> = {
    documents: "platform",
    retrieval: "platform",
    evidence: "evidence",
    security: "security",
    "human-review": "evaluation",
    evaluation: "evaluation",
    developers: "developers",
  };

  function activatePlanet(id: string) {
    document.getElementById(planetTargets[id])?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (event.pointerType === "touch") return;
    const bounds = event.currentTarget.getBoundingClientRect();
    pointerRef.current = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  }

  function resetPointerPosition() {
    pointerRef.current = null;
  }

  useEffect(() => {
    let active = true;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const render = (time: number) => {
      const visual = visualRef.current;
      if (!active || !visual) return;
      const bounds = visual.getBoundingClientRect();
      const scale = Math.min(1, bounds.width / 900, bounds.height / 500) * 0.82;
      const centerX = bounds.width / 2;
      const centerY = bounds.height / 2;
      const pointer = pointerRef.current;
      let closestInfluence = 0;
      let coreTargetX = 0;
      let coreTargetY = 0;

      planets.forEach((planet) => {
        const angle = planet.phase + (reducedMotion ? 0 : time * planet.speed);
        const orbitX = Math.cos(angle) * planet.rx * scale;
        const orbitY = Math.sin(angle) * planet.ry * scale;
        let targetX = 0;
        let targetY = 0;
        let influence = 0;
        if (pointer && !reducedMotion) {
          const distanceX = pointer.x - (centerX + orbitX);
          const distanceY = pointer.y - (centerY + orbitY);
          const distance = Math.sqrt(distanceX ** 2 + distanceY ** 2);
          influence = Math.max(0, Math.min(1, 1 - distance / (150 * scale + 70))) ** 2;
          targetX = Math.max(-22, Math.min(22, distanceX * influence * 0.16));
          targetY = Math.max(-18, Math.min(18, distanceY * influence * 0.16));
          closestInfluence = Math.max(closestInfluence, influence);
          coreTargetX += (pointer.x - centerX) / bounds.width * influence * 3;
          coreTargetY += (pointer.y - centerY) / bounds.height * influence * 3;
        }
        const current = planetMotionRef.current[planet.id] ?? { x: 0, y: 0 };
        current.x += (targetX - current.x) * 0.12;
        current.y += (targetY - current.y) * 0.12;
        planetMotionRef.current[planet.id] = current;
        const element = planetRefs.current[planet.id];
        if (!element) return;
        element.style.setProperty("--orbit-x", `${orbitX + current.x}px`);
        element.style.setProperty("--orbit-y", `${orbitY + current.y}px`);
        element.style.setProperty("--planet-glow", `${0.16 + influence * 0.75}`);
        element.style.setProperty("--planet-scale", `${1 + influence * 0.06}`);
      });

      const core = planetMotionRef.current.core ?? { x: 0, y: 0 };
      core.x += (coreTargetX - core.x) * 0.08;
      core.y += (coreTargetY - core.y) * 0.08;
      planetMotionRef.current.core = core;
      visual.style.setProperty("--rag-core-x", `${core.x}px`);
      visual.style.setProperty("--rag-core-y", `${core.y}px`);
      visual.style.setProperty("--rag-core-tilt-x", `${core.x * 0.4}deg`);
      visual.style.setProperty("--rag-core-tilt-y", `${core.y * -0.4}deg`);
      visual.style.setProperty("--system-glow", `${0.15 + closestInfluence * 0.25}`);
      if (!reducedMotion) frameRef.current = requestAnimationFrame(render);
    };

    frameRef.current = requestAnimationFrame(render);
    return () => {
      active = false;
      if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
    };
    // Planet definitions are immutable for the lifetime of this visualization.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div
      ref={visualRef}
      className="hero-visual hero-rag-visual"
      aria-label="IntelliGround pipeline from document to grounded answer"
      onPointerMove={handlePointerMove}
      onPointerLeave={resetPointerPosition}
    >
      <div className="rag-ambient" />
      <div className="rag-orbit rag-orbit-a" />
      <div className="rag-orbit rag-orbit-b" />
      <div className="rag-orbit rag-orbit-c" />
      <div className="rag-planets" aria-label="IntelliGround capability planets">
        {planets.map((planet) => <div key={planet.id} ref={(element) => { planetRefs.current[planet.id] = element; }} className={`rag-planet rag-planet-${planet.id} rag-planet-${planet.tone}`} role="button" tabIndex={0} aria-label={`${planet.label}: ${planet.detail}`} onClick={() => activatePlanet(planet.id)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); activatePlanet(planet.id); } }}><span className="planet-index">{String(planets.indexOf(planet) + 1).padStart(2, "0")}</span><b>{planet.label}</b><small>{planet.detail}</small></div>)}
      </div>
      <div className="rag-core"><span className="rag-core-pulse" /><strong>IntelliGround</strong><small>Evidence in Every Answer</small><span className="rag-core-answer">AI GROUNDING CORE</span></div>
      <div className="visual-caption"><span className="tiny-cross">+</span> INTELLIGROUND RAG PIPELINE <span className="caption-line" /></div>
    </div>
  );
}

const examples = {
  answer: {
    title: "An answer with somewhere to go.",
    text: "The guide recommends starting with a clearly defined question, then checking the supporting source before making a decision.",
    label: "Grounded in the example source",
  },
  evidence: {
    title: "The source, right beside the answer.",
    text: "“Start with a clearly defined question. Check the supporting source before using an answer to make a decision.”",
    label: "Illustrative document excerpt",
  },
};

export function EvidencePreview() {
  const [tab, setTab] = useState<"answer" | "evidence">("answer");
  const example = examples[tab];
  return (
    <div className="product-window">
      <div className="window-toolbar">
        <div className="window-dots">
          <i />
          <i />
          <i />
        </div>
        <span>Document intelligence</span>
        <span className="preview-label">ILLUSTRATIVE PREVIEW</span>
      </div>
      <div className="preview-workspace">
        <aside className="preview-sidebar">
          <div className="preview-file">
            <DocumentIcon small />
            <span>
              Example guide.pdf<small>Sample document</small>
            </span>
          </div>
          <span className="sidebar-label">YOUR WORKSPACE</span>
          <span className="sidebar-item active">
            ◈ &nbsp; Ask your document
          </span>
          <span className="sidebar-item">≡ &nbsp; Summary</span>
          <span className="sidebar-item">✧ &nbsp; Suggested questions</span>
          <div className="sidebar-bottom">
            <span className="mini-brand">i</span> Every answer has a source.
          </div>
        </aside>
        <div className="preview-main">
          <div className="preview-question">
            <span className="avatar">Y</span>
            <p>How should I approach a new decision?</p>
          </div>
          <div
            className="preview-tabs"
            role="group"
            aria-label="Explore the illustrative response"
          >
            <button
              aria-pressed={tab === "answer"}
              onClick={() => setTab("answer")}
            >
              The answer
            </button>
            <button
              aria-pressed={tab === "evidence"}
              onClick={() => setTab("evidence")}
            >
              The evidence <span>↗</span>
            </button>
          </div>
          <div className="preview-answer" aria-live="polite">
            <span className="eyebrow">{example.label}</span>
            <h3>{example.title}</h3>
            <p>{example.text}</p>
          </div>
          <button
            className="source-link"
            onClick={() => setTab(tab === "answer" ? "evidence" : "answer")}
          >
            <DocumentIcon small />
            <span>
              {tab === "answer"
                ? "Explore the supporting excerpt"
                : "Return to the grounded answer"}
              <small>Example guide · illustrative content</small>
            </span>
            <Arrow />
          </button>
          <div className="preview-input" aria-hidden="true">
            <span>Follow your curiosity. Check the evidence.</span>
            <span>↑</span>
          </div>
        </div>
      </div>
    </div>
  );
}
