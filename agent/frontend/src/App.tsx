import { useEffect, useState } from "react";
import { api, selectPatient } from "./api";
import type { Account } from "./types";
import { Login } from "./components/Login";
import { PatientPortal } from "./components/PatientPortal";
import { DoctorPortal } from "./components/DoctorPortal";
import "./App.css";
import "./portals.css";

export default function App() {
  const [user, setUser] = useState<Account | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [path, setPath] = useState(window.location.pathname);
  const role = path.startsWith("/doctor") ? "doctor" : "patient";

  function navigate(next: string) {
    window.history.pushState(null, "", next);
    setPath(next);
  }
  useEffect(() => {
    let active = true;
    api.me().then((account) => {
      if (!active) return;
      setUser(account);
      if (window.location.pathname === "/" || window.location.pathname.endsWith("/login")) {
        window.history.replaceState(null, "", `/${account.role}`);
        setPath(`/${account.role}`);
      }
    })
      .catch(() => {}).finally(() => { if (active) setLoading(false); });
    const expired = () => { selectPatient(""); setUser(null); };
    const pop = () => setPath(window.location.pathname);
    window.addEventListener("session-expired", expired);
    window.addEventListener("popstate", pop);
    return () => {
      active = false;
      window.removeEventListener("session-expired", expired);
      window.removeEventListener("popstate", pop);
    };
  }, []);


  async function logout() {
    try {
      await api.logout();
      selectPatient("");
      setUser(null);
      setError("");
      navigate(`/${role}/login`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not sign out."); }
  }
  return <div className="app-shell">
    <div className="ambient ambient-one" aria-hidden="true" />
    <div className="ambient ambient-two" aria-hidden="true" />
    <header className="app-header">
      <a className="brand portal-brand" href={user ? `/${user.role}` : "/"}>
        <span className="brand-mark" aria-hidden="true">♡</span>
        <span><strong>SentiVera</strong><small>AI · {role === "doctor" ? "Doctor portal" : "Your wellbeing space"}</small></span>
      </a>
      {user && <div className="portal-account"><span>{user.name}</span><button onClick={logout}>Sign out</button></div>}
    </header>
    {error && <p className="error" role="alert">{error}</p>}
    <main className={`app-main ${user?.role === "doctor" ? "doctor-main" : ""}`}>
      {loading ? <p role="status">Opening your space…</p> : !user ?
        <Login key={role} role={role} onLogin={(account) => { setUser(account); navigate(`/${account.role}`); }} /> :
        role !== user.role ? <section className="card portal-panel"><h1>This is the {role} portal</h1><p>You are signed in as a {user.role}.</p><a href={`/${user.role}`}>Go to your {user.role} page →</a></section> :
        user.role === "doctor" ? <DoctorPortal key={user.id} /> : <PatientPortal key={user.id} user={user} onUpdate={setUser} />}
    </main>
    <footer className="app-footer">{role === "doctor" ? "Research estimates support review; they are not diagnoses." : "A gentle space to pause, reflect, and feel heard."}</footer>
  </div>;
}
