import { useEffect, useRef, useState } from "react";
import { LandingPage } from "./pages/LandingPage";
import { AuthPage } from "./pages/AuthPage";
import { WorkspacePage } from "./pages/WorkspacePage";
import { authApi, hasSession } from "./lib/api";
import type { User } from "./lib/api";
import { Navigation } from "./components/Navigation";
import { AccessSetupPage, ProcessingPage, UploadDocumentsPage } from "./pages/OnboardingPages";

export default function App() {
  const [view, setView] = useState<"landing" | "auth" | "access-setup" | "upload" | "processing" | "workspace">(hasSession() ? "access-setup" : "landing");
  const [user, setUser] = useState<User | null>(null);
  const [access, setAccess] = useState<import("./lib/api").AccessConfig | null>(null);
  const sessionRestoreStarted = useRef(false);

  useEffect(() => {
    const oauthReturn = sessionStorage.getItem("intelliground_google_pending") === "1";
    if (!hasSession() && !oauthReturn) return;
    if (sessionRestoreStarted.current) return;
    sessionRestoreStarted.current = true;
    authApi.me().then((currentUser) => { sessionStorage.removeItem("intelliground_google_pending"); setUser(currentUser); setView("access-setup"); }).catch(() => { sessionStorage.removeItem("intelliground_google_pending"); void authApi.logout(); setView("landing"); });
  }, []);

  async function openWorkspace() {
    if (hasSession()) {
      try { setUser(await authApi.me()); setView("access-setup"); } catch { void authApi.logout(); setView("auth"); }
    } else setView("auth");
  }

  if (view === "auth") return <AuthPage onAuthenticated={openWorkspace} />;
  if (view === "access-setup" && user) return <AccessSetupPage onComplete={(nextAccess) => { setAccess(nextAccess); setView("upload"); }} />;
  if (view === "upload" && user && access) return <UploadDocumentsPage access={access} onComplete={() => setView("processing")} />;
  if (view === "processing" && user) return <ProcessingPage onComplete={() => setView("workspace")} />;
  if (view === "workspace" && user && access) return <WorkspacePage user={user} initialAccess={access} onLogout={() => { setUser(null); setAccess(null); setView("landing"); }} />;
  return <><a className="skip-link" href="#main">Skip to content</a><Navigation onStart={openWorkspace} /><LandingPage onStart={openWorkspace} /></>;
}
