import { Arrow, Brand } from "../components/Brand";
import {
  DocumentIcon,
  EvidencePreview,
  HeroVisual,
} from "../components/ProductPreview";

const steps = [
  {
    number: "01",
    title: "Upload",
    text: "Bring a text-selectable PDF. Your knowledge starts here.",
    icon: "↥",
  },
  {
    number: "02",
    title: "Process",
    text: "Extract pages, create chunks, and build embeddings.",
    icon: "▤",
  },
  {
    number: "03",
    title: "Retrieve",
    text: "Find relevant passages with semantic vector search.",
    icon: "⌕",
  },
  {
    number: "04",
    title: "Generate",
    text: "Turn the retrieved context into a grounded response.",
    icon: "✧",
  },
  {
    number: "05",
    title: "Verify",
    text: "Explore the evidence, page references, and evaluation.",
    icon: "✓",
  },
];

export function LandingPage({ onStart }: { onStart: () => void }) {
  return (
    <>
      <main id="main">
        <section
          className="hero section-wrap"
          id="top"
          aria-labelledby="hero-title"
        >
          <div className="hero-copy">
            <div className="hero-kicker">
              <span className="status-dot" /> KNOWLEDGE, WITH A FOUNDATION.
            </div>
            <h1 id="hero-title">
              Evidence-Based AI
              <br />
              You Can <span className="serif-emphasis">Verify.</span>
              <span className="title-star" aria-hidden="true">
                ✳
              </span>
            </h1>
            <p className="hero-description">
              Upload your documents. Ask better questions.
              <br className="desktop-break" /> Get grounded answers backed by
              evidence.
              <br className="desktop-break" /> Because confidence should come
              with a source.
            </p>
            <div className="hero-actions">
              <button className="button button-primary" onClick={onStart}>
                Get started <Arrow />
              </button>
              <a className="text-link" href="#platform">
                Explore the platform <span aria-hidden="true">↗</span>
              </a>
            </div>
            <div className="hero-footnote">
              <span className="outline-check">✓</span> Your documents. Your
              context. Verifiable answers.
            </div>
          </div>
          <HeroVisual />
          <div className="hero-bottom">
            <span>LESS GUESSWORK. MORE GROUND TRUTH.</span>
            <a href="#why">
              A different kind of AI <span aria-hidden="true">↓</span>
            </a>
          </div>
        </section>

        <section
          className="problem-section section-wrap"
          id="why"
          aria-labelledby="problem-title"
        >
          <div className="section-heading">
            <span className="eyebrow">
              <span className="section-index">01 /</span> THE CASE FOR EVIDENCE
            </span>
            <h2 id="problem-title">
              An AI answer
              <br />
              isn’t <span className="serif-emphasis">enough.</span>
            </h2>
          </div>
          <div className="problem-copy">
            <p className="large-copy">Fluent doesn’t always mean factual.</p>
            <p>
              When an answer matters, you need more than a convincing sentence.
              You need to know where it came from, what supports it, and when
              the evidence falls short.
            </p>
            <div className="principle-row">
              <span>
                IntelliGround connects the answer
                <br />
                <strong>back to its foundation.</strong>
              </span>
              <span className="principle-mark" aria-hidden="true">
                ↳
              </span>
            </div>
          </div>
        </section>

        <section
          className="platform-section"
          id="platform"
          aria-labelledby="platform-title"
        >
          <div className="section-wrap">
            <div className="center-heading">
              <span className="eyebrow">FROM INFORMATION TO UNDERSTANDING</span>
              <h2 id="platform-title">
                Your documents.
                <br />A more grounded{" "}
                <span className="serif-emphasis">conversation.</span>
              </h2>
              <p>
                A thoughtful path from the PDF you upload to the answer you can
                inspect.
              </p>
            </div>
            <div className="pipeline">
              {steps.map((step) => (
                <div className="pipeline-step" key={step.number}>
                  <div className="step-top">
                    <span className="step-icon" aria-hidden="true">
                      {step.icon}
                    </span>
                    <span className="step-number">{step.number}</span>
                  </div>
                  <h3>{step.title}</h3>
                  <p>{step.text}</p>
                </div>
              ))}
            </div>
            <div className="pipeline-note">
              <span className="status-dot" /> Purpose-built for evidence-based
              PDF question answering.
            </div>
          </div>
        </section>

        <section
          className="evidence-section section-wrap"
          id="evidence"
          aria-labelledby="evidence-title"
        >
          <div className="split-heading">
            <div>
              <span className="eyebrow">
                <span className="section-index">02 /</span> FOLLOW THE SOURCE
              </span>
              <h2 id="evidence-title">
                Don’t just get the answer.
                <br />
                <span className="serif-emphasis">See the evidence.</span>
              </h2>
            </div>
            <p>
              Move from a response to its supporting passages and page previews.
              Keep the source within reach, and form your own judgment.
            </p>
          </div>
          <EvidencePreview />
          <div className="evidence-benefits">
            <span>
              <span aria-hidden="true">↗</span> Page-level references
            </span>
            <span>
              <span aria-hidden="true">≡</span> Supporting passages
            </span>
            <span>
              <span aria-hidden="true">◈</span> Protected page previews
            </span>
          </div>
        </section>

        <section
          className="unanswerable-section"
          aria-labelledby="unanswerable-title"
        >
          <div className="section-wrap unanswerable-inner">
            <div className="unanswerable-visual">
              <div className="boundary-rings">
                <span>?</span>
              </div>
              <span className="state-label">UNANSWERABLE</span>
              <span className="boundary-caption">
                A deliberate boundary. A better answer.
              </span>
            </div>
            <div>
              <span className="eyebrow">HONESTY IS A FEATURE</span>
              <h2 id="unanswerable-title">
                Knowing when
                <br />
                <span className="serif-emphasis">not to answer.</span>
              </h2>
              <p>
                When the evidence isn’t strong enough, IntelliGround doesn’t
                pretend. It can return an unanswerable state instead of an
                unsupported response.
              </p>
              <div className="refusal-example">
                <span aria-hidden="true">↳</span>
                <span>
                  “Answer not found in uploaded PDF.”
                  <small>THE SYSTEM’S INTENTIONAL FALLBACK</small>
                </span>
              </div>
            </div>
          </div>
        </section>

        <section
          className="security-section section-wrap"
          id="security"
          aria-labelledby="security-title"
        >
          <div className="security-copy">
            <span className="eyebrow">
              <span className="section-index">03 /</span> SECURITY BY DESIGN
            </span>
            <h2 id="security-title">
              A question goes in.
              <br />
              <span className="serif-emphasis">Guardrails stand up.</span>
            </h2>
            <p>
              AI security at the query layer. Deterministic checks and semantic
              classification inspect requests before retrieval and generation.
            </p>
            <ul className="feature-list">
              <li>
                <span>✓</span> Prompt-injection and instruction-probing checks
              </li>
              <li>
                <span>✓</span> Retrieved content treated as untrusted evidence
              </li>
              <li>
                <span>✓</span> Output checks for potential secret leakage
              </li>
            </ul>
            <p className="security-note">
              Uncertain security decisions fail closed. A{" "}
              <strong>REVIEW</strong> decision stops the request; it doesn’t
              grant permission.
            </p>
          </div>
          <div
            className="security-diagram"
            aria-label="Security pipeline concept: input checks, semantic classification, allow, review or block"
          >
            <div className="diagram-header">
              <span className="eyebrow">THE QUERY BOUNDARY</span>
              <span className="diagram-symbol" aria-hidden="true">
                ⌘
              </span>
            </div>
            <div className="query-in">
              An incoming question <span>↓</span>
            </div>
            <div className="security-layer">
              <span className="layer-number">01</span>
              <div>
                <strong>Input guardrails</strong>
                <small>Validate · detect · inspect</small>
              </div>
              <span aria-hidden="true">⌁</span>
            </div>
            <div className="connector" />
            <div className="security-layer semantic-layer">
              <span className="layer-number">02</span>
              <div>
                <strong>Semantic classification</strong>
                <small>Understand intent, beyond keywords</small>
              </div>
              <span aria-hidden="true">✧</span>
            </div>
            <div className="decision-tree">
              <div>
                <span className="decision allow">ALLOW</span>
                <small>Continue</small>
              </div>
              <div>
                <span className="decision review">REVIEW</span>
                <small>Stop · uncertain</small>
              </div>
              <div>
                <span className="decision block">BLOCK</span>
                <small>Stop · unsafe</small>
              </div>
            </div>
            <div className="diagram-footer">
              PIPELINE CONCEPT <span>Decisions belong to the backend.</span>
            </div>
          </div>
        </section>

        <section
          className="evaluation-section"
          id="evaluation"
          aria-labelledby="evaluation-title"
        >
          <div className="section-wrap">
            <div className="split-heading">
              <div>
                <span className="eyebrow">BEYOND THE FIRST RESPONSE</span>
                <h2 id="evaluation-title">
                  Don’t just generate.
                  <br />
                  <span className="serif-emphasis">Evaluate.</span>
                </h2>
              </div>
              <p>
                Look at the answer from more than one angle. Inspect evaluation
                signals and add human judgment where it matters.
              </p>
            </div>
            <div className="evaluation-grid">
              <div className="evaluation-statement">
                <span className="large-asterisk" aria-hidden="true">
                  ✳
                </span>
                <h3>
                  Confidence.
                  <br />
                  With context.
                </h3>
                <p>
                  Evaluation combines retrieval, relevance, grounding,
                  correctness, and citation signals. These help you assess an
                  answer—not guarantee it.
                </p>
                <a className="text-link" href="#developers">
                  Meet the reliability lab <Arrow />
                </a>
              </div>
              <div className="evaluation-dimensions">
                {[
                  [
                    "01",
                    "Retrieval quality",
                    "How strong is the retrieved context?",
                  ],
                  [
                    "02",
                    "Source relevance",
                    "Does the evidence address the question?",
                  ],
                  [
                    "03",
                    "Grounding",
                    "Is the response supported by the source?",
                  ],
                  [
                    "04",
                    "Answer correctness",
                    "Does the answer follow from the evidence?",
                  ],
                  [
                    "05",
                    "Citation quality",
                    "Do the references match retrieved pages?",
                  ],
                ].map(([number, title, text]) => (
                  <div className="dimension" key={number}>
                    <span>{number}</span>
                    <div>
                      <h3>{title}</h3>
                      <p>{text}</p>
                    </div>
                    <span aria-hidden="true">↗</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section
          className="developers-section section-wrap"
          id="developers"
          aria-labelledby="developers-title"
        >
          <div className="developer-visual">
            <div className="lab-topbar">
              <span>
                <span className="lab-status" /> INTELLIGROUND / LAB
              </span>
              <span>WORKSPACE CONCEPT</span>
            </div>
            <div className="lab-body">
              <div className="lab-nav">
                <span className="selected">Security</span>
                <span>Retrieval</span>
                <span>Confidence</span>
                <span>Evaluation</span>
                <span>Human review</span>
              </div>
              <div className="lab-content">
                <span className="code-comment">
                  // Look beneath the answer.
                </span>
                <div className="code-line">
                  <span>question</span>
                  <b>→</b> input checks
                </div>
                <div className="code-line">
                  <span>evidence</span>
                  <b>→</b> retrieved chunks
                </div>
                <div className="code-line">
                  <span>answer</span>
                  <b>→</b> evaluation
                </div>
                <div className="code-line">
                  <span>judgment</span>
                  <b>→</b> human review
                </div>
                <div className="lab-divider" />
                <span className="code-comment">
                  // Inspect what the backend returns.
                </span>
                <div className="lab-prompt">
                  <span>›</span> Make the reasoning inspectable.
                  <i />
                </div>
              </div>
            </div>
            <div className="lab-bottom">
              OBSERVE. QUESTION. VERIFY.<span>↗</span>
            </div>
          </div>
          <div className="developer-copy">
            <span className="eyebrow">
              <span className="section-index">04 /</span> BUILT FOR THE CURIOUS
            </span>
            <h2 id="developers-title">
              Understand the AI
              <br />
              <span className="serif-emphasis">underneath.</span>
            </h2>
            <p>
              An AI Reliability Lab for developers who want to inspect the real
              pipeline. Explore security responses, retrieved chunks,
              confidence, and evaluation.
            </p>
            <p>
              Bring human judgment into the loop with feedback and corrected
              answers attached to an evaluation.
            </p>
            <a href="#workspaces" className="text-link">
              Find your workspace <Arrow />
            </a>
          </div>
        </section>

        <section
          className="workspaces-section"
          id="workspaces"
          aria-labelledby="workspaces-title"
        >
          <div className="section-wrap">
            <div className="center-heading">
              <span className="eyebrow">
                SIMPLE ON THE SURFACE. THOUGHTFUL UNDERNEATH.
              </span>
              <h2 id="workspaces-title">
                One platform.
                <br />
                Two ways to <span className="serif-emphasis">work.</span>
              </h2>
            </div>
            <div className="workspace-grid">
              <div className="workspace-option user-option">
                <span className="workspace-audience">FOR THE EXPLORERS</span>
                <div className="workspace-icon">
                  <DocumentIcon />
                </div>
                <h3>Document intelligence.</h3>
                <p>
                  Spend less time searching your PDF and more time understanding
                  it. Start with a summary, follow suggested questions, and
                  explore the evidence.
                </p>
                <div className="workspace-flow">
                  <span>Upload</span>
                  <span>→</span>
                  <span>Ask</span>
                  <span>→</span>
                  <span>Verify</span>
                </div>
                <button onClick={onStart} className="workspace-cta">
                  Explore the User workspace <Arrow diagonal />
                </button>
              </div>
              <div className="workspace-option developer-option">
                <span className="workspace-audience">FOR THE BUILDERS</span>
                <div className="workspace-icon code-icon" aria-hidden="true">
                  &lt;/&gt;
                </div>
                <h3>AI reliability lab.</h3>
                <p>
                  Go deeper into the response. Test security prompts, inspect
                  retrieval, examine evaluation signals, and submit human
                  review.
                </p>
                <div className="workspace-flow">
                  <span>Test</span>
                  <span>→</span>
                  <span>Inspect</span>
                  <span>→</span>
                  <span>Review</span>
                </div>
                <button onClick={onStart} className="workspace-cta">
                  Explore the Developer workspace <Arrow diagonal />
                </button>
              </div>
            </div>
            <p className="workspace-note">
              Two perspectives on the same evidence-based system. Workspace
              access is coming in the next phase.
            </p>
          </div>
        </section>

        <section
          className="final-section section-wrap"
          aria-labelledby="final-title"
        >
          <div className="final-decoration" aria-hidden="true">
            ✳
          </div>
          <span className="eyebrow">
            A STRONGER FOUNDATION FOR EVERY QUESTION
          </span>
          <h2 id="final-title">
            Make AI answers
            <br />
            <span className="serif-emphasis">verifiable.</span>
          </h2>
          <p>Your knowledge deserves more than a guess.</p>
          <button className="button button-primary" onClick={onStart}>
            Get started with IntelliGround <Arrow />
          </button>
          <span className="final-note">
            Built around your documents. Grounded in the evidence.
          </span>
        </section>
      </main>
      <footer className="site-footer">
        <div className="section-wrap">
          <div className="footer-main">
            <div>
              <Brand />
              <p>Evidence-based AI you can verify.</p>
            </div>
            <nav aria-label="Footer navigation">
              <a href="#platform">Platform</a>
              <a href="#security">Security</a>
              <a href="#evaluation">Evaluation</a>
              <a href="#developers">Developers</a>
            </nav>
            <a className="back-top" href="#top">
              Back to top <span aria-hidden="true">↑</span>
            </a>
          </div>
          <div className="footer-bottom">
            <span>© {new Date().getFullYear()} IntelliGround</span>
            <span>Built with curiosity. Backed by evidence.</span>
            <span className="footer-version">LANDING PREVIEW / 01</span>
          </div>
        </div>
      </footer>
    </>
  );
}
