import Sidebar from "./Sidebar";
import Section from "./Section";

function Preprocessing({ loading, result }) {
  const steps = [
    "Loading EDA & ECG signals",
    "Signal cleaning (Filtering & Artifact Removal)",
    "5-second sliding windowing",
    "Z-score normalization (Per Subject)",
    "Feature extraction (EDA + ECG)",
    "Creating model input sequences",
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
    status: {
      border: "1px solid #e1e8f5",
      padding: "8px",
      borderRadius: "6px",
      margin: "7px 0",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    },
    completed: {
      color: "#0b9f3a",
      fontWeight: "700",
    },
    progress: {
      color: "#0057ff",
      fontWeight: "700",
    },
    info: {
      marginTop: "14px",
      background: "#eef6ff",
      border: "1px solid #c8def8",
      padding: "10px",
      borderRadius: "6px",
      color: "#0057d9",
      fontSize: "12px",
    },
  };

  return (
    <Section title="2. Preprocessing Status">
      <Sidebar active="Preprocessing" />

      <div style={styles.content}>
        <h3 style={styles.h3}>Preprocessing Pipeline</h3>
        <div style={styles.small}>System is processing the data...</div>

        {steps.map((step) => (
          <div style={styles.status} key={step}>
            <span>✅ {step}</span>
            <span style={styles.completed}>Completed</span>
          </div>
        ))}

        <div style={styles.status}>
          <span>⏳ Sending to CNN-LSTM Model</span>
          <span style={loading ? styles.progress : styles.completed}>
            {loading ? "In Progress" : result ? "Completed" : "Waiting"}
          </span>
        </div>

        <div style={styles.info}>
          ⓘ Please wait while the model generates predictions. This may take a few seconds.
        </div>
      </div>
    </Section>
  );
}

export default Preprocessing;