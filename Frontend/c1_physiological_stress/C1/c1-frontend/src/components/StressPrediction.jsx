import Sidebar from "./Sidebar";
import Section from "./Section";

function StressPrediction({ result, loading }) {
  const prediction = result?.prediction || "HIGH STRESS";
  const probability = result?.stress_probability || 82;
  const confidence = result?.calibrated_confidence || 78;

  const styles = {
    content: {
      flex: 1,
      padding: "18px",
      fontSize: "13px",
    },
    h3: {
      margin: "0 0 16px",
      fontSize: "18px",
    },
    predictionBox: {
      border: "1px solid #e1e8f5",
      borderRadius: "8px",
      padding: "22px",
      marginBottom: "14px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    },
    label: {
      fontSize: "12px",
      color: "#33456d",
      fontWeight: "700",
    },
    prediction: {
      color: "#f5222d",
      fontSize: "28px",
      fontWeight: "800",
      marginTop: "8px",
    },
    sad: {
      fontSize: "42px",
      color: "#f5222d",
    },
    cards: {
      display: "flex",
      gap: "12px",
    },
    card: {
      flex: 1,
      border: "1px solid #e1e8f5",
      borderRadius: "8px",
      padding: "14px",
    },
    red: {
      color: "#f5222d",
      fontSize: "28px",
      margin: "8px 0",
    },
    green: {
      color: "#0ba84a",
      fontSize: "28px",
      margin: "8px 0",
    },
    barBg: {
      height: "7px",
      background: "#e9edf5",
      borderRadius: "8px",
      overflow: "hidden",
    },
    redBar: {
      height: "100%",
      width: `${probability}%`,
      background: "#f5222d",
    },
    greenBar: {
      height: "100%",
      width: `${confidence}%`,
      background: "#0ba84a",
    },
    bottom: {
      display: "flex",
      gap: "8px",
      marginTop: "14px",
    },
    mini: {
      flex: 1,
      background: "#f4f8ff",
      border: "1px solid #d6e6ff",
      padding: "9px",
      borderRadius: "6px",
      fontSize: "11px",
      fontWeight: "700",
    },
  };

  return (
    <Section title="3. Stress Prediction Result">
      <Sidebar active="Stress Prediction" />

      <div style={styles.content}>
        <h3 style={styles.h3}>Stress Prediction (Current Window)</h3>

        <div style={styles.predictionBox}>
          <div>
            <div style={styles.label}>Prediction</div>
            <div style={styles.prediction}>
              {loading ? "PROCESSING..." : prediction}
            </div>
          </div>
          <div style={styles.sad}>☹</div>
        </div>

        <div style={styles.cards}>
          <div style={styles.card}>
            <b>Stress Probability</b>
            <h2 style={styles.red}>{probability}%</h2>
            <div style={styles.barBg}>
              <div style={styles.redBar}></div>
            </div>
            <p>Probability of being stressed</p>
          </div>

          <div style={styles.card}>
            <b>Calibrated Confidence</b>
            <h2 style={styles.green}>{confidence}%</h2>
            <div style={styles.barBg}>
              <div style={styles.greenBar}></div>
            </div>
            <p>Confidence in this prediction</p>
          </div>
        </div>

        <div style={styles.bottom}>
          <div style={styles.mini}>Window Index<br />245 / 1248</div>
          <div style={styles.mini}>Time Range<br />1220s - 1225s</div>
          <div style={styles.mini}>Model<br />CNN-LSTM</div>
        </div>
      </div>
    </Section>
  );
}

export default StressPrediction;