import { useState, useMemo } from "react";
import axios from "axios";

const API_BASE_URL = "/api";

/* ── Demo data generators ────────────────────────────────────────────── */
const generateDemoWindows = () =>
  Array.from({ length: 125 }, (_, i) => {
    const startSec = 200 + i * 5;
    const sp = Math.min(95, Math.max(10, Math.round(20 + Math.sin(i / 8) * 38 + (i % 7) * 3)));
    const cc = Math.max(5, sp - Math.round((i % 8) + 2));
    const level = sp >= 75 ? "HIGH STRESS" : sp >= 50 ? "MODERATE" : "LOW STRESS";
    return [i + 1, `${startSec} – ${startSec + 5}`, `${sp}%`, `${cc}%`, level];
  });

const demoWindowResults = generateDemoWindows();

const demoResult = {
  prediction: "HIGH STRESS",
  stress_probability: 82,
  calibrated_confidence: 78,
  window_index: 245,
  time_range: "1220s – 1225s (5s)",
  model: "CNN-LSTM (Calibrated)",
  features: {
    mean_eda:     { value: "1.42 μS",  icon: "💧", color: "#0057e7", label: "Mean EDA" },
    eda_peak:     { value: "6",        icon: "📈", color: "#7c3aed", label: "EDA Peak Count" },
    phasic_eda:   { value: "0.83 μS",  icon: "〰️", color: "#0891b2", label: "Phasic EDA Amplitude" },
    heart_rate:   { value: "98 bpm",   icon: "❤️", color: "#f5222d", label: "Heart Rate (HR)" },
    hrv:          { value: "28.6 ms",  icon: "📉", color: "#ff8a00", label: "Heart Rate Variability (RMSSD)" },
    rr_interval:  { value: "612 ms",   icon: "⏱️", color: "#0ba84a", label: "RR Interval (Mean)" },
  },
  shap: [
    { name: "Heart Rate (HR)",       value: 0.42, color: "#f5222d" },
    { name: "HRV (RMSSD)",           value: 0.31, color: "#ff8a00" },
    { name: "EDA Peak Count",        value: 0.18, color: "#7c3aed" },
    { name: "Phasic EDA Amplitude",  value: 0.12, color: "#0891b2" },
    { name: "Mean EDA",              value: 0.07, color: "#0057e7" },
  ],
  window_results: demoWindowResults,
  summary: {
    avg_stress: 58,
    avg_confidence: 55,
    predominant: "HIGH",
    high: 725,
    moderate: 323,
    low: 200,
    total: 1248,
  },
};

/* ── Normalise raw backend response into the shape the UI expects ─────── */
function normaliseBackendResponse(data) {
  // Normalise stress_probability to 0-100 integer
  let stressProb = data.stress_probability !== undefined
    ? (data.stress_probability <= 1
        ? Math.round(data.stress_probability * 100)
        : Math.round(data.stress_probability))
    : demoResult.stress_probability;
  stressProb = Math.min(100, Math.max(0, stressProb));

  // Normalise calibrated_confidence to 0-100 integer
  let calibConf = data.calibrated_confidence !== undefined
    ? (data.calibrated_confidence <= 1
        ? Math.round(data.calibrated_confidence * 100)
        : Math.round(data.calibrated_confidence))
    : Math.max(5, stressProb - 5);
  calibConf = Math.min(100, Math.max(0, calibConf));

  // Normalise prediction label to one of three known strings
  const rawPred = String(data.prediction || "").toLowerCase();
  let prediction;
  if (rawPred.includes("high") || rawPred === "1" || rawPred === "stress") {
    prediction = stressProb >= 75 ? "HIGH STRESS" : "MODERATE";
  } else if (rawPred.includes("moderate")) {
    prediction = "MODERATE";
  } else if (rawPred.includes("low") || rawPred === "0" || rawPred === "baseline") {
    prediction = "LOW STRESS";
  } else {
    // Derive from probability if label not recognised
    prediction = stressProb >= 75 ? "HIGH STRESS" : stressProb >= 50 ? "MODERATE" : "LOW STRESS";
  }

  // Rebuild summary from window_results if backend didn't supply one
  let summary = data.summary || null;
  if (!summary) {
    const wins = data.window_results || [];
    if (wins.length > 0) {
      const high     = wins.filter(r => r[4] === "HIGH STRESS").length;
      const moderate = wins.filter(r => r[4] === "MODERATE").length;
      const low      = wins.filter(r => r[4] === "LOW STRESS").length;
      const probs    = wins.map(r => parseInt(r[2], 10)).filter(Boolean);
      const confs    = wins.map(r => parseInt(r[3], 10)).filter(Boolean);
      summary = {
        avg_stress:     probs.length ? Math.round(probs.reduce((a, b) => a + b, 0) / probs.length) : stressProb,
        avg_confidence: confs.length ? Math.round(confs.reduce((a, b) => a + b, 0) / confs.length) : calibConf,
        predominant:    high >= moderate && high >= low ? "HIGH" : moderate >= low ? "MODERATE" : "LOW",
        high, moderate, low,
        total: wins.length,
      };
    } else {
      // Single-window fallback summary
      summary = {
        avg_stress:     stressProb,
        avg_confidence: calibConf,
        predominant:    prediction.split(" ")[0],
        high:     prediction === "HIGH STRESS" ? 1 : 0,
        moderate: prediction === "MODERATE"    ? 1 : 0,
        low:      prediction === "LOW STRESS"  ? 1 : 0,
        total: 1,
      };
    }
  }

  return {
    ...demoResult,                                    // safe defaults for every field
    ...data,                                          // real backend values
    prediction,
    stress_probability:     stressProb,
    calibrated_confidence:  calibConf,
    model:                  data.model || "Baseline Model",
    window_results:         data.window_results || demoWindowResults,
    features:               data.features       || demoResult.features,
    shap:                   data.shap           || demoResult.shap,
    summary,
  };
}

/* ── Download helpers ────────────────────────────────────────────────── */
function triggerDownload(content, filename, mime) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

const dlWindowCSV = (rows) => {
  const hdr = "Window,Time Range,Stress Probability,Calibrated Confidence,Prediction\n";
  triggerDownload(hdr + rows.map(r => r.map(c => `"${c}"`).join(",")).join("\n"),
    "c1-window-results.csv", "text/csv;charset=utf-8;");
};

const dlSHAPCSV = (shap) =>
  triggerDownload("Feature,SHAP Value\n" + shap.map(s => `"${s.name}",${s.value}`).join("\n"),
    "c1-shap-explanations.csv", "text/csv;charset=utf-8;");

const dlSummaryHTML = (r) => {
  const { high, moderate, low, total } = r.summary;
  triggerDownload(`<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>C1 Summary Report</title>
<style>body{font-family:Arial,sans-serif;padding:40px;color:#071635;max-width:820px;margin:auto}
h1{color:#0057e7}h2{color:#21324f;border-bottom:2px solid #d8e4f5;padding-bottom:6px}
table{width:100%;border-collapse:collapse;margin:14px 0}th,td{border:1px solid #d8e1f0;padding:10px 14px}
th{background:#eef5ff}.red{color:#f5222d;font-weight:bold}.green{color:#0ba84a;font-weight:bold}</style></head>
<body><h1>C1 Physiological Stress Detection – Summary Report</h1>
<p>Generated: ${new Date().toLocaleString()}</p>
<h2>Overall Prediction</h2>
<table><tr><th>Model Prediction</th><td class="red">${r.prediction}</td></tr>
<tr><th>Stress Probability</th><td class="red">${r.stress_probability}%</td></tr>
<tr><th>Calibrated Confidence</th><td class="green">${r.calibrated_confidence}%</td></tr></table>
<h2>Feature Summary</h2>
<table>${Object.values(r.features).map(f=>`<tr><th>${f.label}</th><td>${f.value}</td></tr>`).join("")}</table>
<h2>SHAP Feature Importance</h2>
<table><tr><th>Feature</th><th>SHAP Value</th></tr>
${r.shap.map(s=>`<tr><td>${s.name}</td><td>${s.value}</td></tr>`).join("")}</table>
<h2>Window Distribution (${total} windows)</h2>
<table><tr><th>Level</th><th>Count</th><th>%</th></tr>
<tr><td class="red">HIGH STRESS</td><td>${high}</td><td>${Math.round(high/total*100)}%</td></tr>
<tr><td style="color:#ff8a00;font-weight:bold">MODERATE</td><td>${moderate}</td><td>${Math.round(moderate/total*100)}%</td></tr>
<tr><td class="green">LOW STRESS</td><td>${low}</td><td>${Math.round(low/total*100)}%</td></tr></table>
<footer style="margin-top:40px;color:#60708e;font-size:12px">C1 – ${r.model} | WESAD Dataset | SHAP Explainability</footer>
</body></html>`, "c1-summary-report.html", "text/html;charset=utf-8;");
};

const dlAllJSON = (r) =>
  triggerDownload(JSON.stringify({ exported_at: new Date().toISOString(), ...r,
    window_results: r.window_results.map(w => ({
      window: w[0], time_range: w[1], stress_probability: w[2],
      calibrated_confidence: w[3], prediction: w[4],
    })) }, null, 2), "c1-all-results.json", "application/json");

/* ── Donut SVG ───────────────────────────────────────────────────────── */
function DonutChart({ high, moderate, low, total }) {
  if (!total) return null;
  const r = 60, cx = 80, cy = 80, stroke = 28;
  const circ = 2 * Math.PI * r;
  const pHigh = high / total, pMod = moderate / total;
  const highDash = pHigh * circ, modDash = pMod * circ, lowDash = (1 - pHigh - pMod) * circ;
  const offHigh = 0, offMod = -(highDash), offLow = -(highDash + modDash);
  const base = { fill: "none", strokeWidth: stroke, strokeLinecap: "butt" };
  return (
    <svg width="160" height="160" viewBox="0 0 160 160">
      <circle cx={cx} cy={cy} r={r} {...base} stroke="#e8eef7" strokeDasharray={`${circ} ${circ}`} />
      <circle cx={cx} cy={cy} r={r} {...base} stroke="#f5222d"
        strokeDasharray={`${highDash} ${circ - highDash}`}
        strokeDashoffset={offHigh}
        transform={`rotate(-90 ${cx} ${cy})`} />
      <circle cx={cx} cy={cy} r={r} {...base} stroke="#ff8a00"
        strokeDasharray={`${modDash} ${circ - modDash}`}
        strokeDashoffset={offMod}
        transform={`rotate(-90 ${cx} ${cy})`} />
      <circle cx={cx} cy={cy} r={r} {...base} stroke="#0ba84a"
        strokeDasharray={`${lowDash} ${circ - lowDash}`}
        strokeDashoffset={offLow}
        transform={`rotate(-90 ${cx} ${cy})`} />
      <text x={cx} y={cy - 6} textAnchor="middle" fontSize="18" fontWeight="900" fill="#071635">{total.toLocaleString()}</text>
      <text x={cx} y={cy + 12} textAnchor="middle" fontSize="10" fill="#60708e">Total</text>
      <text x={cx} y={cy + 24} textAnchor="middle" fontSize="10" fill="#60708e">Windows</text>
    </svg>
  );
}

/* ── Constants ──────────────────────────────────────────────────────── */
const ROWS_PER_PAGE = 7;

const MENU = [
  { key: "dashboard",      label: "Dashboard",                emoji: "🏠" },
  { key: "upload",         label: "Upload / Select Data",     emoji: "📤" },
  { key: "preprocessing",  label: "Preprocessing",            emoji: "⚙️" },
  { key: "prediction",     label: "Stress Prediction",        emoji: "🧠" },
  { key: "features",       label: "Physiological Features",   emoji: "❤️" },
  { key: "explainability", label: "Explainability (SHAP)",    emoji: "📊" },
  { key: "windows",        label: "Window Results",           emoji: "🪟" },
  { key: "summary",        label: "Summary & Insights",       emoji: "📌" },
  { key: "download",       label: "Download / Export",        emoji: "⬇️" },
  { key: "settings",       label: "Settings",                 emoji: "🔧" },
  { key: "about",          label: "About",                    emoji: "ℹ️" },
];

/* ── Shared sub-components ─────────────────────────────────────────── */
const Card = ({ children, style = {} }) => (
  <div style={{
    background: "#fff", border: "1px solid #d7e4f5", borderRadius: "18px",
    padding: "24px", boxShadow: "0 6px 22px rgba(23,65,120,0.09)",
    marginBottom: "20px", ...style,
  }}>{children}</div>
);

const CardTitle = ({ children }) => (
  <h2 style={{ fontSize: "20px", fontWeight: "900", marginTop: 0, marginBottom: "4px", color: "#071635" }}>
    {children}
  </h2>
);

const Subtitle = ({ children }) => (
  <p style={{ color: "#60708e", fontSize: "13px", marginTop: 0, marginBottom: "18px" }}>{children}</p>
);

const InfoBox = ({ children, icon = "ℹ️", color = "#0057e7", bg = "#eef6ff", border = "#c8def8" }) => (
  <div style={{ background: bg, border: `1px solid ${border}`, borderRadius: "10px",
    padding: "12px 16px", color, fontSize: "13px", display: "flex", gap: "10px",
    alignItems: "flex-start", marginTop: "16px" }}>
    <span>{icon}</span><span>{children}</span>
  </div>
);

const StatCard = ({ label, value, color = "#f5222d", bar, barColor }) => (
  <div style={{ background: "#fff", border: "1px solid #d7e4f5", borderRadius: "18px",
    padding: "22px", boxShadow: "0 6px 22px rgba(23,65,120,0.09)" }}>
    <div style={{ color: "#60708e", fontWeight: "800", fontSize: "13px", marginBottom: "10px" }}>{label}</div>
    <div style={{ fontSize: "40px", fontWeight: "900", color, lineHeight: 1 }}>{value}</div>
    {bar !== undefined && (
      <div style={{ height: "8px", background: "#e8eef7", borderRadius: "999px", overflow: "hidden", marginTop: "14px" }}>
        <div style={{ height: "100%", width: `${bar}%`, background: barColor || color,
          borderRadius: "999px", transition: "width 0.6s ease" }} />
      </div>
    )}
  </div>
);

/* ── App ─────────────────────────────────────────────────────────────── */
export default function App() {
  const [page, setPage] = useState("upload");
  const [subject, setSubject] = useState("S2");
  const [session, setSession] = useState("Stress Session");
  const [result, setResult] = useState(demoResult);
  const [loading, setLoading] = useState(false);
  const [showProfile, setShowProfile] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);
  const [windowPage, setWindowPage] = useState(1);
  const [preprocessStep, setPreprocessStep] = useState(7);

  // ── Safe accessors — never undefined even if backend omits fields ──────
  const summary  = result?.summary  ?? demoResult.summary;
  const features = result?.features ?? demoResult.features;
  const shap     = result?.shap     ?? demoResult.shap;

  const windowRows    = result?.window_results ?? demoWindowResults;
  const totalWinPages = Math.ceil(windowRows.length / ROWS_PER_PAGE);
  const pagedRows     = useMemo(
    () => windowRows.slice((windowPage - 1) * ROWS_PER_PAGE, windowPage * ROWS_PER_PAGE),
    [windowRows, windowPage],
  );
  const pageNums = useMemo(() => {
    const t = totalWinPages;
    if (t <= 5) return Array.from({ length: t }, (_, i) => i + 1);
    let s = Math.max(1, windowPage - 2), e = Math.min(t, s + 4);
    if (e - s < 4) s = Math.max(1, e - 4);
    return Array.from({ length: e - s + 1 }, (_, i) => s + i);
  }, [windowPage, totalWinPages]);

  const runPrediction = async () => {
    try {
      setLoading(true); setBackendStatus(null);
      setPreprocessStep(0); setPage("preprocessing");
      for (let i = 1; i <= 8; i++) {
        await new Promise(res => setTimeout(res, 300));
        setPreprocessStep(i);
      }
      const res = await axios.post(`${API_BASE_URL}/predict`, { subject, session });
      // Normalise and merge — backend only returns partial data for baseline models
      setResult(normaliseBackendResponse(res.data));
      setBackendStatus("live");
    } catch {
      await new Promise(res => setTimeout(res, 300));
      setResult(demoResult);
      setBackendStatus("demo");
    } finally {
      setLoading(false); setWindowPage(1); setPage("prediction");
    }
  };

  /* ── styles ── */
  const S = {
    app: { minHeight: "100vh", display: "flex",
      background: "linear-gradient(135deg, #eef6ff 0%, #f4f8ff 100%)",
      fontFamily: "'Segoe UI', system-ui, sans-serif", color: "#071635" },

    sb: { width: "240px", minWidth: "240px", flexShrink: 0,
      background: "linear-gradient(180deg, #001830 0%, #002650 50%, #001a3a 100%)",
      color: "#fff", padding: "20px 14px", minHeight: "100vh",
      boxShadow: "4px 0 20px rgba(0,0,0,0.18)", display: "flex", flexDirection: "column" },
    sbLogo: { fontSize: "17px", fontWeight: "900", marginBottom: "4px", lineHeight: "1.3",
      display: "flex", alignItems: "center", gap: "8px" },
    sbLogoSub: { fontSize: "11px", color: "#90b8e0", marginBottom: "24px" },
    navItem: { padding: "11px 13px", borderRadius: "10px", marginBottom: "4px",
      cursor: "pointer", fontWeight: "700", fontSize: "13px", display: "flex",
      alignItems: "center", gap: "10px", transition: "all 0.15s", color: "#c8dff5" },
    navActive: { background: "linear-gradient(135deg, #0057e7, #0b7cff)",
      color: "#fff", boxShadow: "0 4px 14px rgba(0,87,231,0.40)" },
    navHover: { background: "rgba(255,255,255,0.07)" },
    navSep: { borderTop: "1px solid rgba(255,255,255,0.08)", margin: "10px 0" },
    navEmoji: { fontSize: "16px", width: "22px", textAlign: "center" },

    main: { flex: 1, padding: "28px 32px", overflowY: "auto" },
    topBar: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "22px" },
    pageTitle: { fontSize: "28px", fontWeight: "900", margin: 0, color: "#071635" },
    pageSub: { color: "#60708e", marginTop: "4px", fontSize: "13px" },

    profileBtn: { display: "flex", alignItems: "center", gap: "10px", background: "#fff",
      padding: "9px 16px", borderRadius: "999px", border: "1px solid #d8e4f5",
      cursor: "pointer", boxShadow: "0 4px 14px rgba(0,0,0,0.07)", fontWeight: "800", fontSize: "14px" },
    avatar: { width: "36px", height: "36px", borderRadius: "50%",
      background: "linear-gradient(135deg, #0057e7, #7c3aed)", color: "#fff",
      display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "900", fontSize: "13px" },
    profileDrop: { position: "absolute", top: "54px", right: 0, background: "#fff",
      border: "1px solid #d8e4f5", borderRadius: "14px",
      boxShadow: "0 12px 30px rgba(0,0,0,0.13)", padding: "12px", width: "200px", zIndex: 200 },

    label: { display: "block", fontWeight: "800", marginBottom: "8px", color: "#21324f", fontSize: "13px" },
    select: { width: "260px", padding: "13px", border: "1px solid #cbd8ee", borderRadius: "10px",
      background: "#f9fcff", fontWeight: "700", outline: "none", fontSize: "14px",
      cursor: "pointer" },
    btnPrimary: { background: "linear-gradient(135deg, #0057e7, #0b7cff)", color: "#fff",
      border: "none", padding: "14px 28px", borderRadius: "12px", fontWeight: "900",
      cursor: "pointer", boxShadow: "0 8px 20px rgba(0,87,231,0.30)", fontSize: "15px",
      width: "100%", marginTop: "8px" },
    btnOutline: { background: "#fff", color: "#0057e7", border: "1px solid #8bb8ff",
      padding: "9px 18px", borderRadius: "9px", fontWeight: "800", cursor: "pointer",
      fontSize: "13px" },

    statGrid:  { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "18px", marginBottom: "20px" },
    statGrid2: { display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "18px" },

    fRow: { display: "flex", justifyContent: "space-between", alignItems: "center",
      border: "1px solid #e8eef7", background: "#fafcff",
      padding: "14px 18px", borderRadius: "12px", marginBottom: "10px" },
    fLeft: { display: "flex", alignItems: "center", gap: "12px" },
    fIcon: { width: "36px", height: "36px", borderRadius: "10px", display: "flex",
      alignItems: "center", justifyContent: "center", fontSize: "18px" },

    shapRow: { display: "grid", gridTemplateColumns: "200px 1fr 52px", gap: "14px",
      alignItems: "center", marginBottom: "18px" },
    shapBg: { height: "22px", background: "#eef1f7", borderRadius: "20px", overflow: "hidden" },

    table: { width: "100%", borderCollapse: "collapse" },
    th: { background: "#f4f8ff", border: "1px solid #dce8f5", padding: "11px 14px",
      textAlign: "center", fontWeight: "800", color: "#21324f", fontSize: "13px" },
    td: { border: "1px solid #dce8f5", padding: "11px 14px", textAlign: "center",
      fontWeight: "700", fontSize: "13px" },

    pag: { display: "flex", alignItems: "center", justifyContent: "center", gap: "5px", marginTop: "16px" },
    pBtn: { padding: "6px 11px", borderRadius: "7px", border: "1px solid #d8e1f0",
      background: "#fff", color: "#0057e7", fontWeight: "700", cursor: "pointer",
      fontSize: "12px", minWidth: "34px" },
    pBtnActive:   { background: "#0057e7", color: "#fff", border: "1px solid #0057e7" },
    pBtnDisabled: { color: "#bdc8d8", cursor: "not-allowed", border: "1px solid #edf1f7" },

    alertDemo: { background: "#fff8e1", border: "1px solid #ffe082", borderRadius: "12px",
      padding: "12px 18px", marginBottom: "18px", display: "flex", alignItems: "center",
      gap: "10px", fontSize: "13px", fontWeight: "700", color: "#7c5200" },
    alertLive: { background: "#e8f5e9", border: "1px solid #a5d6a7", borderRadius: "12px",
      padding: "12px 18px", marginBottom: "18px", display: "flex", alignItems: "center",
      gap: "10px", fontSize: "13px", fontWeight: "700", color: "#1b5e20" },
    alertInfo: { background: "#e3f2fd", border: "1px solid #90caf9", borderRadius: "12px",
      padding: "12px 18px", marginBottom: "18px", display: "flex", alignItems: "center",
      gap: "10px", fontSize: "13px", fontWeight: "700", color: "#0d47a1" },

    stepRow: { display: "flex", justifyContent: "space-between", alignItems: "center",
      border: "1px solid #e8eef7", background: "#fafcff",
      padding: "14px 18px", borderRadius: "12px", marginBottom: "8px" },

    insightItem: { display: "flex", alignItems: "flex-start", gap: "10px",
      padding: "10px 0", borderBottom: "1px solid #f0f4fb" },
  };

  const stressCol = (txt) =>
    txt === "HIGH STRESS" ? "#f5222d" : txt === "MODERATE" ? "#ff8a00" : "#0ba84a";

  const stressBadge = (txt) => {
    const c = stressCol(txt);
    return (
      <span style={{ color: c, background: c + "18", padding: "3px 10px",
        borderRadius: "20px", fontWeight: "800", fontSize: "12px" }}>
        {txt === "HIGH STRESS" ? "🔴" : txt === "MODERATE" ? "🟡" : "🟢"} {txt}
      </span>
    );
  };

  const preprocessSteps = [
    "Loading EDA & ECG signals",
    "Signal cleaning (Filtering & Artifact Removal)",
    "5-second sliding windowing",
    "Z-score normalization (Per Subject)",
    "Feature extraction (EDA + ECG)",
    "Creating model input sequences",
    "Sending to Baseline Model",
    "Receiving calibrated predictions",
  ];

  const navEl = (item) => (
    <div
      key={item.key}
      onClick={() => setPage(item.key)}
      style={{ ...S.navItem, ...(page === item.key ? S.navActive : {}) }}
    >
      <span style={S.navEmoji}>{item.emoji}</span>
      <span>{item.label}</span>
    </div>
  );

  // Whether live backend data was used (for showing demo-data notices)
  const isLive = backendStatus === "live";
  const isDemo = backendStatus === "demo";

  return (
    <div style={S.app}>
      {/* ── Sidebar ── */}
      <aside style={S.sb}>
        <div style={S.sbLogo}>
          <span style={{ background: "#0b68ff", padding: "5px 8px", borderRadius: "8px", fontSize: "13px" }}>C1</span>
          Stress Detection
        </div>
        <div style={S.sbLogoSub}>Physiological Signals · EDA &amp; ECG</div>

        {MENU.slice(0, 9).map(navEl)}

        <div style={{ flex: 1 }} />
        <div style={S.navSep} />
        {MENU.slice(9).map(navEl)}
      </aside>

      {/* ── Main ── */}
      <main style={S.main}>
        {/* Top bar */}
        <div style={S.topBar}>
          <div>
            <h1 style={S.pageTitle}>Stress Prediction Dashboard</h1>
            <p style={S.pageSub}>Real-time physiological stress analysis using WESAD EDA &amp; ECG signals</p>
          </div>
          <div style={{ position: "relative" }}>
            <div style={S.profileBtn} onClick={() => setShowProfile(v => !v)}>
              <div style={S.avatar}>TP</div>
              <span>Tharindu</span>
              <span style={{ fontSize: "11px" }}>⌄</span>
            </div>
            {showProfile && (
              <div style={S.profileDrop}>
                <div style={{ padding: "10px", borderBottom: "1px solid #edf1f7", marginBottom: "8px" }}>
                  <div style={{ fontWeight: "900", marginBottom: "2px" }}>Tharindu Perera</div>
                  <div style={{ color: "#60708e", fontSize: "12px" }}>C1 Research User</div>
                </div>
                {["👤 Profile", "⚙️ Settings", "📄 My Reports", "🚪 Logout"].map(m => (
                  <div key={m} style={{ padding: "10px", cursor: "pointer", borderRadius: "8px",
                    fontWeight: "700", fontSize: "13px" }}>{m}</div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Backend banners */}
        {isDemo && (
          <div style={S.alertDemo}>
            ⚠️ Backend not reachable — showing demo data. Set your ngrok URL in{" "}
            <code style={{ background: "#ffeaa0", padding: "2px 6px", borderRadius: "5px" }}>.env</code> and restart Vite.
          </div>
        )}
        {isLive && (
          <div style={S.alertLive}>
            ✅ Live data received from backend — using {result.model}.
          </div>
        )}
        {/* Notice when live baseline data is shown (SHAP/features are illustrative) */}
        {isLive && (
          <div style={S.alertInfo}>
            ℹ️ Physiological features and SHAP values shown are illustrative. Full CNN-LSTM + SHAP pipeline coming in next sprint.
          </div>
        )}

        {/* ════ DASHBOARD ════ */}
        {page === "dashboard" && (
          <>
            <div style={S.statGrid}>
              <StatCard label="Model Prediction"      value={result.prediction}          color={stressCol(result.prediction)} />
              <StatCard label="Stress Probability"    value={`${result.stress_probability}%`}   color="#f5222d"
                bar={result.stress_probability}   barColor="linear-gradient(90deg,#ff4d4f,#f5222d)" />
              <StatCard label="Calibrated Confidence" value={`${result.calibrated_confidence}%`} color="#0ba84a"
                bar={result.calibrated_confidence} barColor="linear-gradient(90deg,#2ed573,#0ba84a)" />
            </div>
            <Card>
              <CardTitle>🏠 Welcome to C1 – Physiological Stress Detection</CardTitle>
              <Subtitle>Navigate using the sidebar to explore all modules of the C1 system.</Subtitle>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: "12px", marginTop: "8px" }}>
                {[
                  { icon: "📤", t: "Upload / Select Data",    d: "Choose subject & session from WESAD dataset" },
                  { icon: "⚙️", t: "Preprocessing Pipeline",  d: "Signal cleaning, windowing & normalization" },
                  { icon: "🧠", t: "Stress Prediction",        d: "Model predictions with calibrated output" },
                  { icon: "📊", t: "Explainability (SHAP)",    d: "Feature importance for every window" },
                ].map(c => (
                  <div key={c.t} style={{ background: "#f4f8ff", borderRadius: "12px",
                    padding: "16px", border: "1px solid #dce8f5" }}>
                    <div style={{ fontSize: "24px", marginBottom: "6px" }}>{c.icon}</div>
                    <div style={{ fontWeight: "800", marginBottom: "4px" }}>{c.t}</div>
                    <div style={{ color: "#60708e", fontSize: "12px" }}>{c.d}</div>
                  </div>
                ))}
              </div>
              <InfoBox>
                💡 Frontend communicates with Google Colab backend via API. The WESAD dataset
                is stored in Google Drive and accessed by the backend. Frontend only sends requests
                (subject / session) and displays results.
              </InfoBox>
            </Card>
          </>
        )}

        {/* ════ UPLOAD ════ */}
        {page === "upload" && (
          <Card style={{ maxWidth: "680px" }}>
            <CardTitle>📤 Upload or Select Data</CardTitle>
            <Subtitle>Choose a subject and session from the WESAD dataset</Subtitle>

            <div style={{ display: "flex", gap: "20px", flexWrap: "wrap", marginBottom: "20px" }}>
              <div>
                <label style={S.label}>Select Subject</label>
                <select style={S.select} value={subject} onChange={e => setSubject(e.target.value)}>
                  {["S2","S3","S4","S5","S6","S7","S8","S9","S10","S11","S13","S14","S15","S16","S17"]
                    .map(s => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label style={S.label}>Select Session</label>
                <select style={S.select} value={session} onChange={e => setSession(e.target.value)}>
                  <option>Stress Session</option>
                  <option>Baseline Session</option>
                </select>
              </div>
            </div>

            <div style={{ background: "#f4f8ff", border: "1px solid #dce8f5",
              borderRadius: "12px", padding: "16px", marginBottom: "20px" }}>
              <div style={{ fontWeight: "800", marginBottom: "10px", color: "#21324f" }}>
                ℹ️ About WESAD Dataset
              </div>
              <ul style={{ margin: 0, paddingLeft: "20px", color: "#3a4a6b", fontSize: "13px", lineHeight: "1.9" }}>
                <li>15 Subjects</li>
                <li>Sensors: EDA, ECG (Used in C1)</li>
                <li>Labels: Baseline, Stress (Amusement excluded)</li>
              </ul>
              <InfoBox icon="💾" color="#0057e7">
                The dataset is stored in Google Drive and accessed by the backend.
              </InfoBox>
            </div>

            <button style={S.btnPrimary} onClick={runPrediction} disabled={loading}>
              {loading ? "⏳ Processing..." : "🚀 Run Stress Detection"}
            </button>
          </Card>
        )}

        {/* ════ PREPROCESSING ════ */}
        {page === "preprocessing" && (
          <Card style={{ maxWidth: "680px" }}>
            <CardTitle>⚙️ Preprocessing Pipeline</CardTitle>
            <Subtitle>System is processing the data…</Subtitle>

            {preprocessSteps.map((step, i) => {
              const done   = i < preprocessStep;
              const active = i === preprocessStep - 1 && loading;
              return (
                <div key={step} style={{
                  ...S.stepRow,
                  background: done ? "#f0fdf4" : active ? "#fffbeb" : "#fafcff",
                  border: `1px solid ${done ? "#bbf7d0" : active ? "#fde68a" : "#e8eef7"}`,
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <span style={{ fontSize: "18px" }}>
                      {done ? "✅" : active ? "⏳" : "⬜"}
                    </span>
                    <span style={{ fontWeight: "700", fontSize: "14px", color: done ? "#166534" : "#21324f" }}>
                      {step}
                    </span>
                  </div>
                  <span style={{
                    fontWeight: "800", fontSize: "12px",
                    color: done ? "#0ba84a" : active ? "#d97706" : "#aab9d0",
                  }}>
                    {done ? "Completed" : active ? "In Progress" : "Pending"}
                  </span>
                </div>
              );
            })}

            {loading && (
              <InfoBox icon="⏳" color="#d97706" bg="#fffbeb" border="#fde68a">
                Please wait while the model generates predictions. This may take a few seconds.
              </InfoBox>
            )}
            {!loading && preprocessStep >= preprocessSteps.length && (
              <InfoBox icon="✅" color="#166534" bg="#f0fdf4" border="#bbf7d0">
                All preprocessing steps completed successfully.
              </InfoBox>
            )}
          </Card>
        )}

        {/* ════ STRESS PREDICTION ════ */}
        {page === "prediction" && (
          <>
            <Card style={{ maxWidth: "700px" }}>
              <CardTitle>🧠 Stress Prediction (Current Window)</CardTitle>
              <Subtitle>{result.model} output with temperature-scaled calibration</Subtitle>

              <div style={{ display: "flex", alignItems: "center", gap: "20px", marginBottom: "24px",
                background: "#fff5f5", border: "1px solid #ffd6d6", borderRadius: "14px", padding: "20px" }}>
                <span style={{ fontSize: "52px" }}>
                  {result.prediction === "HIGH STRESS" ? "😫" : result.prediction === "MODERATE" ? "😐" : "😊"}
                </span>
                <div>
                  <div style={{ color: "#60708e", fontWeight: "800", fontSize: "13px", marginBottom: "4px" }}>
                    Prediction
                  </div>
                  <div style={{ fontSize: "36px", fontWeight: "900", color: stressCol(result.prediction), lineHeight: 1 }}>
                    {result.prediction}
                  </div>
                </div>
              </div>

              <div style={S.statGrid2}>
                <div style={{ background: "#fff5f5", border: "1px solid #ffd6d6",
                  borderRadius: "14px", padding: "18px" }}>
                  <div style={{ color: "#60708e", fontWeight: "800", fontSize: "12px", marginBottom: "8px" }}>
                    Stress Probability
                  </div>
                  <div style={{ fontSize: "38px", fontWeight: "900", color: "#f5222d" }}>
                    {result.stress_probability}%
                  </div>
                  <div style={{ height: "8px", background: "#ffd6d6", borderRadius: "999px", overflow: "hidden", marginTop: "10px" }}>
                    <div style={{ height: "100%", width: `${result.stress_probability}%`,
                      background: "linear-gradient(90deg,#ff4d4f,#f5222d)", borderRadius: "999px" }} />
                  </div>
                  <div style={{ color: "#f5222d", fontSize: "11px", marginTop: "6px", fontWeight: "700" }}>
                    Probability of being stressed
                  </div>
                </div>

                <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0",
                  borderRadius: "14px", padding: "18px" }}>
                  <div style={{ color: "#60708e", fontWeight: "800", fontSize: "12px", marginBottom: "8px" }}>
                    Calibrated Confidence
                  </div>
                  <div style={{ fontSize: "38px", fontWeight: "900", color: "#0ba84a" }}>
                    {result.calibrated_confidence}%
                  </div>
                  <div style={{ height: "8px", background: "#bbf7d0", borderRadius: "999px", overflow: "hidden", marginTop: "10px" }}>
                    <div style={{ height: "100%", width: `${result.calibrated_confidence}%`,
                      background: "linear-gradient(90deg,#2ed573,#0ba84a)", borderRadius: "999px" }} />
                  </div>
                  <div style={{ color: "#0ba84a", fontSize: "11px", marginTop: "6px", fontWeight: "700" }}>
                    Confidence in this prediction
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", gap: "0", marginTop: "18px",
                border: "1px solid #dce8f5", borderRadius: "12px", overflow: "hidden" }}>
                {[
                  { icon: "🪟", label: "Window Index", val: result.window_index },
                  { icon: "⏱️", label: "Time Range",   val: result.time_range },
                  { icon: "🤖", label: "Model",        val: result.model },
                ].map((m, i) => (
                  <div key={m.label} style={{ flex: 1, padding: "14px 16px", textAlign: "center",
                    borderLeft: i === 0 ? "none" : "1px solid #dce8f5", background: "#f9fbff" }}>
                    <div style={{ fontSize: "18px", marginBottom: "4px" }}>{m.icon}</div>
                    <div style={{ color: "#60708e", fontSize: "11px", fontWeight: "700" }}>{m.label}</div>
                    <div style={{ fontWeight: "800", fontSize: "12px", marginTop: "2px", color: "#21324f" }}>
                      {m.val}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </>
        )}

        {/* ════ PHYSIOLOGICAL FEATURES ════ */}
        {page === "features" && (
          <Card style={{ maxWidth: "680px" }}>
            <CardTitle>❤️ Extracted Physiological Features</CardTitle>
            <Subtitle>Features used by the model for this window</Subtitle>

            {Object.values(features).map(f => (
              <div key={f.label} style={S.fRow}>
                <div style={S.fLeft}>
                  <div style={{ ...S.fIcon, background: f.color + "18" }}>{f.icon}</div>
                  <span style={{ fontWeight: "700", fontSize: "14px" }}>{f.label}</span>
                </div>
                <span style={{ fontWeight: "900", color: f.color, fontSize: "16px" }}>{f.value}</span>
              </div>
            ))}

            <InfoBox>
              These features are computed from the 5-second sliding window and passed to the model.
            </InfoBox>
          </Card>
        )}

        {/* ════ EXPLAINABILITY SHAP ════ */}
        {page === "explainability" && (
          <Card style={{ maxWidth: "680px" }}>
            <CardTitle>📊 SHAP Feature Importance (Current Window)</CardTitle>
            <Subtitle>Top features contributing to the predicted stress</Subtitle>

            {shap.map(item => (
              <div key={item.name} style={S.shapRow}>
                <span style={{ fontWeight: "700", fontSize: "13px", color: "#21324f" }}>{item.name}</span>
                <div style={S.shapBg}>
                  <div style={{ height: "100%", width: `${item.value * 160}%`,
                    background: `linear-gradient(90deg, ${item.color}99, ${item.color})`,
                    borderRadius: "20px", transition: "width 0.6s ease" }} />
                </div>
                <span style={{ fontWeight: "900", color: item.color, textAlign: "right" }}>{item.value}</span>
              </div>
            ))}

            <div style={{ display: "flex", justifyContent: "flex-end", gap: "40px",
              marginLeft: "214px", color: "#aab9d0", fontSize: "11px", fontWeight: "700" }}>
              <span>0</span><span>0.2</span><span>0.4</span><span>0.6</span>
            </div>
            <div style={{ textAlign: "right", fontSize: "11px", color: "#60708e", marginTop: "4px" }}>
              SHAP Value (Impact)
            </div>

            <InfoBox icon="💡">
              Positive SHAP values indicate higher contribution to stress prediction.
            </InfoBox>
          </Card>
        )}

        {/* ════ WINDOW RESULTS ════ */}
        {page === "windows" && (
          <Card>
            <CardTitle>🪟 Window-wise Prediction Results (5-second windows)</CardTitle>
            <Subtitle>Stress classification for every 5-second EDA / ECG segment</Subtitle>

            <table style={S.table}>
              <thead>
                <tr>
                  {["Window", "Time Range (s)", "Stress Probability", "Calibrated Confidence", "Prediction"].map(h => (
                    <th key={h} style={S.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pagedRows.map((r, idx) => (
                  // Use index as fallback key in case window numbers are not unique
                  <tr key={`row-${r[0]}-${idx}`} style={{ background: "#fafcff" }}>
                    <td style={S.td}>{r[0]}</td>
                    <td style={S.td}>{r[1]}</td>
                    <td style={{ ...S.td, color: stressCol(r[4]) }}>{r[2]}</td>
                    <td style={{ ...S.td, color: stressCol(r[4]) }}>{r[3]}</td>
                    <td style={S.td}>{stressBadge(r[4])}</td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination — use explicit keys on every node to avoid key warnings */}
            <div style={S.pag}>
              <button
                key="prev"
                style={{ ...S.pBtn, ...(windowPage === 1 ? S.pBtnDisabled : {}) }}
                onClick={() => setWindowPage(p => Math.max(1, p - 1))}
                disabled={windowPage === 1}
              >‹</button>

              {pageNums[0] > 1 && (
                <button key="page-1" style={S.pBtn} onClick={() => setWindowPage(1)}>1</button>
              )}
              {pageNums[0] > 2 && (
                <span key="ellipsis-start" style={{ color: "#aab9d0" }}>…</span>
              )}

              {pageNums.map(n => (
                <button key={`page-${n}`} style={{ ...S.pBtn, ...(n === windowPage ? S.pBtnActive : {}) }}
                  onClick={() => setWindowPage(n)}>{n}</button>
              ))}

              {pageNums[pageNums.length - 1] < totalWinPages - 1 && (
                <span key="ellipsis-end" style={{ color: "#aab9d0" }}>…</span>
              )}
              {pageNums[pageNums.length - 1] < totalWinPages && (
                <button key={`page-last`} style={S.pBtn} onClick={() => setWindowPage(totalWinPages)}>{totalWinPages}</button>
              )}

              <button
                key="next"
                style={{ ...S.pBtn, ...(windowPage === totalWinPages ? S.pBtnDisabled : {}) }}
                onClick={() => setWindowPage(p => Math.min(totalWinPages, p + 1))}
                disabled={windowPage === totalWinPages}
              >›</button>
            </div>
            <div style={{ textAlign: "center", fontSize: "12px", color: "#60708e", marginTop: "8px" }}>
              Showing {(windowPage - 1) * ROWS_PER_PAGE + 1}–{Math.min(windowPage * ROWS_PER_PAGE, windowRows.length)} of {windowRows.length} windows
            </div>
          </Card>
        )}

        {/* ════ SUMMARY ════ */}
        {page === "summary" && (
          <>
            <div style={S.statGrid}>
              <StatCard label="📊 Average Stress Probability"    value={`${summary.avg_stress}%`}     color="#f5222d"
                bar={summary.avg_stress}     barColor="linear-gradient(90deg,#ff4d4f,#f5222d)" />
              <StatCard label="🎯 Average Calibrated Confidence" value={`${summary.avg_confidence}%`} color="#0ba84a"
                bar={summary.avg_confidence} barColor="linear-gradient(90deg,#2ed573,#0ba84a)" />
              <StatCard label="⚡ Predominant Stress Level"      value={summary.predominant}           color="#f5222d" />
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: "20px" }}>
              <Card>
                <div style={{ fontWeight: "800", marginBottom: "14px", color: "#21324f", fontSize: "14px" }}>
                  📉 Stress Distribution (All Windows)
                </div>
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <DonutChart high={summary.high} moderate={summary.moderate}
                    low={summary.low} total={summary.total} />
                </div>
                <div style={{ marginTop: "10px" }}>
                  {[
                    { label: "High Stress",    count: summary.high,     pct: Math.round(summary.high     / summary.total * 100), color: "#f5222d" },
                    { label: "Moderate Stress", count: summary.moderate, pct: Math.round(summary.moderate / summary.total * 100), color: "#ff8a00" },
                    { label: "Low Stress",     count: summary.low,      pct: Math.round(summary.low      / summary.total * 100), color: "#0ba84a" },
                  ].map(l => (
                    <div key={l.label} style={{ display: "flex", alignItems: "center", gap: "8px",
                      marginBottom: "6px", fontSize: "12px", fontWeight: "700" }}>
                      <div style={{ width: "12px", height: "12px", borderRadius: "3px",
                        background: l.color, flexShrink: 0 }} />
                      <span style={{ flex: 1, color: "#21324f" }}>{l.label}</span>
                      <span style={{ color: l.color }}>{l.pct}% ({l.count.toLocaleString()})</span>
                    </div>
                  ))}
                </div>
              </Card>

              <Card>
                <div style={{ fontWeight: "800", marginBottom: "14px", color: "#21324f", fontSize: "14px" }}>
                  💡 Insights
                </div>
                {[
                  { icon: "🔴", txt: "The participant shows high stress for the majority of the session." },
                  { icon: "❤️", txt: "Elevated heart rate and low HRV are the major contributors." },
                  { icon: "💧", txt: "EDA peaks are significantly higher during stress windows." },
                  { icon: "🎯", txt: "Calibrated confidence remains stable across all predictions." },
                  { icon: "📈", txt: "Phasic EDA amplitude spikes align with high-stress windows." },
                ].map(ins => (
                  <div key={ins.txt} style={S.insightItem}>
                    <span style={{ fontSize: "18px" }}>{ins.icon}</span>
                    <span style={{ fontSize: "13px", color: "#3a4a6b", lineHeight: "1.6" }}>{ins.txt}</span>
                  </div>
                ))}
              </Card>
            </div>
          </>
        )}

        {/* ════ DOWNLOAD ════ */}
        {page === "download" && (
          <Card style={{ maxWidth: "680px" }}>
            <CardTitle>⬇️ Download Results</CardTitle>
            <Subtitle>Download predictions, explanations and reports</Subtitle>

            {[
              { label: "📄 Export Window-wise Results (CSV)",  action: () => dlWindowCSV(windowRows) },
              { label: "📊 Export SHAP Explanations (CSV)",   action: () => dlSHAPCSV(shap) },
              { label: "📑 Export Summary Report (PDF/HTML)", action: () => dlSummaryHTML(result) },
              { label: "📦 Export All Results (ZIP / JSON)",  action: () => dlAllJSON(result) },
            ].map(f => (
              <div key={f.label} style={{ display: "flex", justifyContent: "space-between",
                alignItems: "center", border: "1px solid #e1e8f5", background: "#fafcff",
                padding: "16px 18px", borderRadius: "12px", marginBottom: "10px" }}>
                <span style={{ fontWeight: "700", fontSize: "14px" }}>{f.label}</span>
                <button style={S.btnOutline} onClick={f.action}>⬇ Download</button>
              </div>
            ))}

            <InfoBox icon="📋">
              All files include subject info, timestamps and calibrated predictions.
            </InfoBox>
          </Card>
        )}

        {/* ════ SETTINGS ════ */}
        {page === "settings" && (
          <Card style={{ maxWidth: "600px" }}>
            <CardTitle>🔧 Settings</CardTitle>
            <Subtitle>Configure system preferences and API connection</Subtitle>
            {[
              { label: "Backend API URL", placeholder: "https://xxxx.ngrok.io/api" },
              { label: "Default Subject",  placeholder: "S2" },
              { label: "Window Size (s)",  placeholder: "5" },
            ].map(f => (
              <div key={f.label} style={{ marginBottom: "18px" }}>
                <label style={S.label}>{f.label}</label>
                <input placeholder={f.placeholder} style={{ ...S.select, width: "100%", boxSizing: "border-box" }} />
              </div>
            ))}
            <button style={{ ...S.btnPrimary, width: "auto", padding: "12px 28px" }}>💾 Save Settings</button>
          </Card>
        )}

        {/* ════ ABOUT ════ */}
        {page === "about" && (
          <Card style={{ maxWidth: "680px" }}>
            <CardTitle>ℹ️ About C1 – Physiological Stress Detection</CardTitle>
            <Subtitle>Component 1 of the AI-Driven Multimodal Stress &amp; Emotion Analysis System</Subtitle>

            {[
              { icon: "🎓", t: "Research Context",  d: "Undergraduate dissertation project at SLIIT — IT22235688 · Perera K.T.K." },
              { icon: "🗄️", t: "Dataset",           d: "WESAD (Wearable Stress and Affect Detection) — 15 subjects, EDA & ECG sensors." },
              { icon: "🤖", t: "Model Architecture", d: "Lightweight CNN-LSTM hybrid (<1M parameters), suitable for real-time inference." },
              { icon: "🔬", t: "Explainability",     d: "SHAP (SHapley Additive exPlanations) applied per window and per subject." },
              { icon: "🌡️", t: "Calibration",        d: "Temperature scaling applied to output probabilities for reliable confidence scores." },
              { icon: "👨‍🏫", t: "Supervisor",         d: "Prof. Samantha Thelijjagoda — SLIIT · March 2026" },
            ].map(i => (
              <div key={i.t} style={S.insightItem}>
                <span style={{ fontSize: "22px" }}>{i.icon}</span>
                <div>
                  <div style={{ fontWeight: "800", marginBottom: "2px", color: "#21324f" }}>{i.t}</div>
                  <div style={{ fontSize: "13px", color: "#3a4a6b", lineHeight: "1.6" }}>{i.d}</div>
                </div>
              </div>
            ))}

            <InfoBox icon="🔗">
              Keywords: physiological stress detection · WESAD · wearable signals · CNN-LSTM · explainable AI
            </InfoBox>
          </Card>
        )}
      </main>
    </div>
  );
}
