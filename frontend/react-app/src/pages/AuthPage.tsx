import { useState } from "react";
import type { FormEvent } from "react";
import { API_BASE_URL, ApiError, authApi } from "../lib/api";
import { Brand } from "../components/Brand";

type Mode = "login" | "register" | "forgot" | "reset";

function authErrorMessage(error: unknown) {
  if (error instanceof ApiError) {
    if (error.status === 401) return "Invalid email or password.";
    if (error.status === 429) return "Too many authentication attempts. Please wait a moment and try again.";
    if (error.status >= 500) return "The IntelliGround service is unavailable right now. Please try again shortly.";
    return error.message;
  }
  if (error instanceof TypeError) return `IntelliGround could not reach the backend at ${API_BASE_URL}. Check that the API is running and that the browser can access this address.`;
  return "We could not complete that request.";
}

export function AuthPage({ initialMode = "login", onAuthenticated }: { initialMode?: Mode; onAuthenticated: () => void }) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const [email, setEmail] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [password, setPassword] = useState("");
  const [resetToken, setResetToken] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const isReset = mode === "reset";
  const title = mode === "login" ? "Welcome back." : mode === "register" ? "Build a grounded workspace." : mode === "forgot" ? "Recover your access." : "Set a new password.";

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true); setError(""); setMessage("");
    try {
      if (mode === "login") { await authApi.login(email, password); onAuthenticated(); }
      else if (mode === "register") { await authApi.register(firstName, lastName, email, password); setMode("login"); setMessage("Account created. Sign in to continue."); }
      else if (mode === "forgot") { const result = await authApi.forgotPassword(email); setMessage(result.message); }
      else { const result = await authApi.resetPassword(resetToken, password); setMessage(result.message); setMode("login"); }
    } catch (caught) {
      setError(authErrorMessage(caught));
    } finally { setBusy(false); }
  }

  return <main className="auth-page">
    <div className="auth-brand"><Brand /></div>
    <section className="auth-card" aria-labelledby="auth-title">
      <div className="auth-card-intro"><span className="eyebrow">INTELLIGROUND WORKSPACE</span><h1 id="auth-title">{title}</h1><p>{mode === "login" ? "Sign in to work with your documents and inspect every answer." : "Evidence-based answers begin with a secure workspace."}</p></div>
      {error && <div className="form-alert error" role="alert">{error}</div>}
      {message && <div className="form-alert success" role="status">{message}</div>}
      <form className="auth-form" onSubmit={submit}>
        {mode === "register" && <><label>First name<input type="text" autoComplete="given-name" value={firstName} onChange={(event) => setFirstName(event.target.value)} required /></label><label>Last name<input type="text" autoComplete="family-name" value={lastName} onChange={(event) => setLastName(event.target.value)} required /></label></>}
        {!isReset && mode !== "forgot" && <label>Email address<input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>}
        {mode === "forgot" && <label>Email address<input type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>}
        {isReset && <label>Reset token<input type="text" value={resetToken} onChange={(event) => setResetToken(event.target.value)} required /></label>}
        {(mode === "login" || mode === "register" || isReset) && <label>{isReset ? "New password" : "Password"}<input type="password" autoComplete={isReset ? "new-password" : mode === "login" ? "current-password" : "new-password"} minLength={8} value={password} onChange={(event) => setPassword(event.target.value)} required /><span className="field-hint">At least 8 characters.</span></label>}
        <button className="button button-primary auth-submit" disabled={busy}>{busy ? "Working…" : mode === "login" ? "Sign in" : mode === "register" ? "Create account" : mode === "forgot" ? "Send recovery link" : "Reset password"}<span aria-hidden="true">↗</span></button>
      </form>
      {mode === "login" && <><div className="auth-divider"><span>or</span></div><a className="google-button" href={authApi.googleLoginUrl()} onClick={() => sessionStorage.setItem("intelliground_google_pending", "1")}>Continue with Google <span aria-hidden="true">↗</span></a></>}
      <div className="auth-links">
        {mode === "login" && <><button onClick={() => setMode("register")}>Create an account</button><button onClick={() => setMode("forgot")}>Forgot password?</button></>}
        {mode !== "login" && <button onClick={() => setMode("login")}>Back to sign in</button>}
      </div>
      <p className="auth-footnote">Your API credentials stay in memory for the active workspace session.</p>
    </section>
  </main>;
}
