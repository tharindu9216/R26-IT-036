import Sidebar from "./Sidebar";
import Section from "./Section";

function Explainability({ result }) {
  const shap = result?.shap || [
    { name: "Heart Rate (HR)", value: 0.42 },
    { name: "HRV (RMSSD)", value: 0.31 },
    { name: "EDA Peak Count", value: 0.18 },
    { name: "Phasic EDA Amplitude", value: 0.12 },
    { name: "Mean EDA", value: 0.07 },
  ];

  const max = Math.max(...shap.map((x) => x.value));

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
      marginBottom: "20px",
    },
    row: {
      display: "grid",
      gridTemplateColumns: "140px 1fr 40px",
      alignItems: "center",
      gap: "10px",
      margin: "13px 0",
    },
    label: {
      textAlign: "right",
      fontSize: "12px",
    },
    barBg: {
      height: "16px",
      background: "#edf1f7",
      borderRadius: "10px",
      overflow: "hidden",
    },
    bar: {
      height: "100%",
      background: "#f5222d",
      borderRadius: "10px",
    },
    info: {
      marginTop: "20px",
      background: "#fff8df",
      border: "1px solid #f3d58a",
      padding: "8px",
      borderRadius: "6px",
      color: "#8a5b00",
      fontSize: "12px",
    },
  };

  return (
    <Section title="5. Explainability (SHAP)">
      <Sidebar active="Explainability (SHAP)" />

      <div style={styles.content}>
        <h3 style={styles.h3}>SHAP Feature Importance (Current Window)</h3>
        <div style={styles.small}>Top features contributing to the predicted stress.</div>

        {shap.map((item) => (
          <div style={styles.row} key={item.name}>
            <span style={styles.label}>{item.name}</span>
            <div style={styles.barBg}>
              <div
                style={{
                  ...styles.bar,
                  width: `${(item.value / max) * 100}%`,
                }}
              ></div>
            </div>
            <b>{item.value}</b>
          </div>
        ))}

        <div style={styles.info}>
          ⓘ Positive SHAP values indicate higher contribution to stress prediction.
        </div>
      </div>
    </Section>
  );
}

export default Explainability;