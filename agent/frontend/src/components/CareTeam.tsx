import { useEffect, useState } from "react";
import { api } from "../api";
import type { Account, DoctorSummary } from "../types";

export function CareTeam({ user, onUpdate }: { user: Account; onUpdate: (user: Account) => void }) {
  const [tab, setTab] = useState<"current" | "available">(user.doctor_id ? "current" : "available");
  const [doctors, setDoctors] = useState<DoctorSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [directoryError, setDirectoryError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const [search, setSearch] = useState("");
  const [doctorEmail, setDoctorEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    async function load() {
      try {
        const result = await api.doctors();
        if (active) { setDoctors(result); setDirectoryError(""); }
      } catch {
        if (active) setDirectoryError("Could not refresh the doctors list. Please try again.");
      } finally {
        if (active) { setLoading(false); timer = window.setTimeout(load, 15000); }
      }
    }
    void load();
    return () => { active = false; window.clearTimeout(timer); };
  }, [refresh]);

  const currentDoctor = doctors.find(doctor => doctor.id === user.doctor_id);
  const query = search.trim().toLowerCase();
  const matches = doctors.filter(doctor => `${doctor.name} ${doctor.email}`.toLowerCase().includes(query));
  const selectedDoctor = doctors.find(doctor => doctor.email.toLowerCase() === doctorEmail.trim().toLowerCase());
  const sameDoctor = !!selectedDoctor && selectedDoctor.id === user.doctor_id;

  async function share(remove = false) {
    setBusy(true); setNotice(""); setError("");
    try {
      if (remove) {
        await api.unlinkDoctor();
        onUpdate(await api.me());
        setNotice("Sharing has stopped. Your former doctor can no longer access your activity.");
        setDoctorEmail(""); setTab("available");
      } else {
        const linked = await api.linkDoctor(doctorEmail);
        onUpdate(await api.me());
        setNotice(`Your activity is now shared with ${linked.name}.`);
        setDoctorEmail(""); setTab("current");
      }
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not update sharing."); }
    finally { setBusy(false); }
  }

  return <details className="card sharing-panel">
    <summary>Your care team · {user.doctor_id ? "Sharing enabled" : "Not sharing"}</summary>
    <p>Choose whether a doctor can review your conversations, assessments, wearable readings, and AI insights. Selecting a name does not share anything until you confirm below.</p>
    <div className="care-team-tabs" role="tablist" aria-label="Your doctors">
      <button id="my-doctor-tab" role="tab" aria-selected={tab === "current"} aria-controls="my-doctor-panel" onClick={() => setTab("current")}>My doctor</button>
      <button id="available-doctors-tab" role="tab" aria-selected={tab === "available"} aria-controls="available-doctors-panel" onClick={() => setTab("available")}>Available doctors ({doctors.length})</button>
    </div>
    <div className="directory-toolbar"><small>{loading ? "Loading doctors…" : `${doctors.length} registered doctor${doctors.length === 1 ? "" : "s"}`}</small><button disabled={loading} onClick={() => { setLoading(true); setRefresh(refresh + 1); }}>Refresh doctors</button></div>
    {directoryError && <p className="error" role="alert">{directoryError}</p>}
    {tab === "current" ? <section id="my-doctor-panel" role="tabpanel" aria-labelledby="my-doctor-tab">
      {currentDoctor ? <article className="directory-card"><strong>{currentDoctor.name}</strong><span>{currentDoctor.email}</span><small>Currently linked · Your activity is shared with this doctor</small></article> : user.doctor_id ? <p>{loading ? "Loading your doctor's details…" : "Sharing is enabled. Refresh the list to load your doctor's details."}</p> : <p>You have not linked a doctor yet.</p>}
      <button disabled={busy} onClick={() => setTab("available")}>{user.doctor_id ? "Choose another doctor" : "Find a doctor"}</button>
      {user.doctor_id && <button className="portal-link" disabled={busy} onClick={() => void share(true)}>Stop sharing</button>}
    </section> : <section id="available-doctors-panel" role="tabpanel" aria-labelledby="available-doctors-tab">
      <label className="patient-search">Find a doctor<input type="search" value={search} onChange={event => setSearch(event.target.value)} placeholder="Search name or email" /></label>
      {!loading && doctors.length === 0 && !directoryError && <p>No doctor accounts are available yet.</p>}
      {!loading && doctors.length > 0 && matches.length === 0 && <p>No doctor matches your search.</p>}
      <div className="doctor-directory">{matches.map(doctor => <article className={`directory-card ${selectedDoctor?.id === doctor.id ? "is-selected" : ""}`} key={doctor.id}>
        <div><strong>{doctor.name}</strong><span>{doctor.email}</span>{user.doctor_id === doctor.id && <small>Your current doctor</small>}</div>
        <button disabled={busy || user.doctor_id === doctor.id} onClick={() => { setDoctorEmail(doctor.email); setError(""); setNotice(""); }}>{user.doctor_id === doctor.id ? "Currently linked" : selectedDoctor?.id === doctor.id ? "Selected" : "Select doctor"}</button>
      </article>)}</div>
      <form className="portal-form" onSubmit={event => { event.preventDefault(); void share(); }}>
        <label>Doctor's email<input type="email" required value={doctorEmail} onChange={event => { setDoctorEmail(event.target.value); setError(""); }} placeholder="Select a doctor above or enter their email" /></label>
        <p>{selectedDoctor ? `Selected: ${selectedDoctor.name}. Confirm to share your retained activity.` : "Choose a doctor above to fill the email automatically, or enter an existing doctor's email."}</p>
        <button disabled={busy || !doctorEmail.trim() || sameDoctor}>{busy ? "Updating sharing…" : user.doctor_id ? "Change doctor and share" : "Share with selected doctor"}</button>
      </form>
    </section>}
    {notice && <p role="status">{notice}</p>}
    {error && <p className="error" role="alert">{error}</p>}
  </details>;
}
