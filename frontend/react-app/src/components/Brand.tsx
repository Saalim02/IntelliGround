export function Brand({ inverted = false }: { inverted?: boolean }) {
  return (
    <a
      className={`brand${inverted ? " brand-inverted" : ""}`}
      href="#top"
      aria-label="IntelliGround home"
    >
      <span className="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32" fill="none">
          <path d="m16 5 11 6-11 6-11-6Z" />
          <path d="m5 17 11 6 11-6M5 23l11 6 11-6" />
        </svg>
      </span>
      <span>
        intelli<span className="brand-weight">ground</span>
        <span className="brand-dot">.</span>
      </span>
    </a>
  );
}

export function Arrow({ diagonal = false }: { diagonal?: boolean }) {
  return (
    <svg
      className="arrow-icon"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d={diagonal ? "M6 18 18 6M6 6h12v12" : "M4 12h15m-6-6 6 6-6 6"}
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
