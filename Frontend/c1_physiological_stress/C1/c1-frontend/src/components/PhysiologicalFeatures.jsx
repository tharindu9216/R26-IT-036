import Sidebar from "./Sidebar";
import Section from "./Section";

function PhysiologicalFeatures({ result }) {
  const f = result?.features || {
    mean_eda: "1.42 μS",
    eda_peak_count: 6,
    phasic_eda: "0.83 μS",
    heart_rate: "98 bpm",
    hrv: "28.6 ms",
    rr_interval: "612 ms",
  };

  const rows = [
    ["💧", "Mean EDA", f.mean_eda],
    ["〽️", "EDA Peak Count", f.eda_peak_count],
    ["〽️", "Phasic EDA Amplitude", f.phasic_eda || "0.83 μS"],
    ["❤️", "Heart Rate (HR)", f.heart_rate],
    ["💚", "Heart Rate Variability (RMSSD)", f.hrv],
    ["📈", "RR Interval (Mean)", f.rr_interval || "612 ms"],
  ];

  const styles = {
    content: {
      flex: 1,
      padding: "18px",
      fontSize: "13px",
    },
    h3: {
      margin: "0 0 4px",
      fontSize: "18px",
    },
    small: {
      color: "#33456d",
      fontSize: "12px",
      marginBottom: "12px",
    },
    row: {
      border: "1px solid #e1e8f5",
      padding: "10px",
      borderRadius: "6px",
      margin: "8px 0",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    },
    left: {
      display: "flex",
      gap: "10px",
      alignItems: "center",
    },
    value: {
      color: "#0057ff",
      fontWeight: "800",
    },
    info: {
      marginTop: "10px",
      background: "#eef6ff",
      border: "1px solid #c8def8",
      padding: "8px",
      borderRadius: "6px",
      color: "#0057d9",
      fontSize: "12px",
    },
  };

  return (
    <Section title="4. Physiological Features (Current Window)">
      <Sidebar active="Physiological Features" />

      <div style={styles.content}>
        <h3 style={styles.h3}>Extracted Physiological Features</h3>
        <div style={styles.small}>Features used by the model for this window</div>

        {rows.map(([icon, name, value]) => (
          <div style={styles.row} key={name}>
            <div style={styles.left}>
              <span>{icon}</span>
              <b>{name}</b>
            </div>
            <span style={styles.value}>{value}</span>
          </div>
        ))}

        <div style={styles.info}>
          ⓘ These features are computed from the 5-second window.
        </div>
      </div>
    </Section>
  );
}

export default PhysiologicalFeatures;