import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { DragEvent, FormEvent } from "react";
import { Brand } from "../components/Brand";
import { ApiError, documentApi } from "../lib/api";
import type { AccessConfig, UploadResult } from "../lib/api";

type OnboardingStep = "access-setup" | "upload" | "processing";

function Progress({ step }: { step: OnboardingStep }) {
  const current = step === "access-setup" ? 0 : step === "upload" ? 1 : 2;
  return <div className="onboarding-progress" aria-label={`Onboarding step ${current + 1} of 3`}>
    {(["Access Setup", "Upload Documents", "Start Exploring"] as const).map((label, index) => <div className={`onboarding-progress-step${index < current ? " complete" : index === current ? " active" : ""}`} key={label}><span>{index < current ? "✓" : String(index + 1).padStart(2, "0")}</span><small>{label}</small></div>)}
  </div>;
}

function OnboardingShell({ step, eyebrow, title, description, children }: { step: OnboardingStep; eyebrow: string; title: string; description: string; children: ReactNode }) {
  return <main className="onboarding-page"><div className="onboarding-ambient" /><header className="onboarding-header"><Brand /><span className="onboarding-session">SECURE ONBOARDING / {step === "access-setup" ? "01" : step === "upload" ? "02" : "03"}</span></header><div className="onboarding-layout"><aside className="onboarding-aside"><span className="eyebrow">INTELLIGROUND / JOURNEY</span><h1>Build a more grounded way to work with AI.</h1><p>Move from your access method to an evidence-backed workspace in a few focused steps.</p><div className="onboarding-aside-orbit"><span>◉</span><small>AI GROUNDING CORE</small></div></aside><section className="onboarding-content"><Progress step={step} /><div className="onboarding-heading"><span className="eyebrow">{eyebrow}</span><h2>{title}</h2><p>{description}</p></div>{children}</section></div></main>;
}

export function AccessSetupPage({ onComplete }: { onComplete: (access: AccessConfig) => void }) {
  const [method, setMethod] = useState<"system" | "openai">("system");
  const [password, setPassword] = useState("");
  const [openAiKey, setOpenAiKey] = useState("");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const access = method === "system" ? { accessPassword: password, userOpenAiApiKey: "" } : { accessPassword: "", userOpenAiApiKey: openAiKey };
    if (access.accessPassword || access.userOpenAiApiKey) onComplete(access);
  }

  return <OnboardingShell step="access-setup" eyebrow="01 / ACCESS SETUP" title="Choose how IntelliGround can work." description="Provide system access or use your own OpenAI API key for this session."><form className="onboarding-form access-choice-form" onSubmit={submit}><div className="method-switch" role="tablist" aria-label="Access method"><button type="button" role="tab" aria-selected={method === "system"} onClick={() => setMethod("system")}>Use system password</button><button type="button" role="tab" aria-selected={method === "openai"} onClick={() => setMethod("openai")}>Use your own API key</button></div>{method === "system" ? <label>Enter system password<span>Enter the project access password provided by your team.</span><input type="password" autoComplete="off" value={password} onChange={(event) => setPassword(event.target.value)} required /></label> : <label>OpenAI API key<span>Provide your own OpenAI API key for this session.</span><input type="password" placeholder="sk-…" autoComplete="off" value={openAiKey} onChange={(event) => setOpenAiKey(event.target.value)} required /></label>}<p className="onboarding-note">Your credentials are kept in memory only and are never saved after your session.</p><button className="button button-primary" type="submit">Continue <span aria-hidden="true">→</span></button></form></OnboardingShell>;
}

export function UploadDocumentsPage({ access, onComplete }: { access: AccessConfig; onComplete: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [state, setState] = useState<"idle" | "uploading" | "error">("idle");
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  function chooseFile(candidate: File | undefined) {
    if (!candidate) return;
    if (candidate.type !== "application/pdf" || !candidate.name.toLowerCase().endsWith(".pdf")) { setError("Only PDF files are accepted by the document service."); setState("error"); return; }
    setFile(candidate); setError(""); setState("idle");
  }
  function onDrop(event: DragEvent<HTMLDivElement>) { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0]); }
  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || state === "uploading") return;
    setState("uploading"); setError("");
    try {
      const result: UploadResult = await documentApi.upload(file, access);
      if (result.status !== "success") throw new Error(result.message ?? "The document could not be processed.");
      onComplete();
    } catch (caught: unknown) {
      setState("error");
      setError(caught instanceof ApiError && caught.status >= 500 ? "The document service is temporarily unavailable. Please try again." : caught instanceof Error ? caught.message : "The document could not be uploaded.");
    }
  }

  return <OnboardingShell step="upload" eyebrow="02 / DOCUMENT SOURCE" title="Upload your documents." description="Add a PDF to build your knowledge base and start asking grounded questions."><form className="onboarding-form upload-onboarding-form" onSubmit={(event) => void upload(event)}><div className={`onboarding-dropzone${dragging ? " is-dragging" : ""}${file ? " has-file" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop} onClick={() => fileInput.current?.click()} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInput.current?.click(); }}><input ref={fileInput} type="file" accept="application/pdf" hidden onChange={(event) => chooseFile(event.target.files?.[0])} />{file ? <><span className="upload-file-icon">PDF</span><strong>{file.name}</strong><small>Ready to process · {Math.max(1, Math.round(file.size / 1024))} KB</small></> : <><span className="upload-arrow">↑</span><strong>Drag & drop your file here</strong><small>or choose a PDF from your device</small><em>Supported format: PDF</em></>}</div>{error && <div className="form-alert error" role="alert">{error}</div>}<button className="button button-primary" type="submit" disabled={!file || state === "uploading"}>{state === "uploading" ? "Processing document…" : "Continue"} <span aria-hidden="true">→</span></button></form></OnboardingShell>;
}

export function ProcessingPage({ onComplete }: { onComplete: () => void }) {
  useEffect(() => { const timer = window.setTimeout(onComplete, 900); return () => window.clearTimeout(timer); }, [onComplete]);
  return <OnboardingShell step="processing" eyebrow="03 / PREPARING WORKSPACE" title="Your knowledge base is taking shape." description="The document upload completed. IntelliGround is preparing your evidence-backed workspace."><div className="processing-card"><div className="processing-core"><span /><strong>IG</strong></div><h3>Preparing your workspace…</h3><p>Document processing completed successfully. Loading your grounded workspace.</p><div className="processing-checks"><span>✓ Upload received</span><span>✓ Text extraction complete</span><span>✓ Knowledge context ready</span></div><div className="processing-bar"><span /></div></div></OnboardingShell>;
}
