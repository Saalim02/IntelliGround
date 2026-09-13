import { useState } from "react";
import { Arrow, Brand } from "./Brand";

const links = [
  ["Platform", "platform"],
  ["Security", "security"],
  ["Evaluation", "evaluation"],
  ["Developers", "developers"],
];

export function Navigation({ onStart }: { onStart: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <header className="site-header">
      <div className="nav-inner">
        <Brand />
        <button
          className="menu-toggle"
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
          aria-controls="main-navigation"
          onClick={() => setOpen(!open)}
        >
          <span />
          <span className={open ? "menu-line-open" : ""} />
        </button>
        <nav
          id="main-navigation"
          aria-label="Main navigation"
          className={open ? "navigation is-open" : "navigation"}
          onKeyDown={(event) => {
            if (event.key === "Escape") setOpen(false);
          }}
        >
          <div className="nav-links">
            {links.map(([label, id]) => (
              <a key={id} href={`#${id}`} onClick={() => setOpen(false)}>
                {label}
              </a>
            ))}
          </div>
          <div className="nav-actions">
            <button
              className="sign-in"
              onClick={() => {
                setOpen(false);
                onStart();
              }}
            >
              Sign in
            </button>
            <button
              className="button button-dark button-small"
              onClick={() => {
                setOpen(false);
                onStart();
              }}
            >
              Get started <Arrow />
            </button>
          </div>
        </nav>
      </div>
    </header>
  );
}
