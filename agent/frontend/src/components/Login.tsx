import { useState, type FormEvent } from "react";
import { api } from "../api";
import type { Account } from "../types";

export function Login({ role, onLogin }: { role: Account["role"]; onLogin: (user: Account) => void }) {
  const [register, setRegister] = useState(false);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true); setError("");
    try {
      if (register) await api.register(name, email, password);
      onLogin(await api.login(email, password, role));
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not sign in."); }
    finally { setBusy(false); }
  }
  return <section className="card login-card">
    <span className="eyebrow">{role === "doctor" ? "SENTIVERA FOR DOCTORS" : "A LITTLE SPACE FOR YOU"}</span>
    <h1>{role === "doctor" ? "Care starts with understanding." : register ? "Make yourself at home." : "Welcome back."}</h1>
    <p className="subtitle">{role === "doctor" ? "Sign in to review your patients’ check-ins, sensor signals, and AI explanations." : "Your conversations and check-ins, in one calm place."}</p>
    <nav className="portal-switch" aria-label="Sign-in portal"><a href="/patient/login" aria-current={role === "patient" ? "page" : undefined}>Patient</a><a href="/doctor/login" aria-current={role === "doctor" ? "page" : undefined}>Doctor</a></nav>
    <form onSubmit={submit} className="portal-form">
      {register && <label>Your name<input autoComplete="name" value={name} onChange={e => setName(e.target.value)} required maxLength={100} /></label>}
      <label>Email address<input type="email" autoComplete="username" value={email} onChange={e => setEmail(e.target.value)} required maxLength={254} /></label>
      <label>Password<input type="password" autoComplete={register ? "new-password" : "current-password"} value={password} onChange={e => setPassword(e.target.value)} minLength={register ? 10 : 1} maxLength={128} required /></label>
      {register && <small>Use at least 10 characters.</small>}
      {error && <p className="error" role="alert">{error}</p>}
      <button className="portal-primary" disabled={busy}>{busy ? "Please wait…" : register ? "Create account" : `Sign in as ${role}`}</button>
    </form>
    {role === "patient" ? <button className="portal-link" disabled={busy} onClick={() => { setRegister(!register); setError(""); }}>{register ? "Already have an account? Sign in" : "New here? Create your account"}</button> : <p className="portal-note">Use the doctor account provided by your administrator.</p>}
  </section>;
}
