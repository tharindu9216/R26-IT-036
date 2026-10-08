import Sidebar from "./Sidebar";
import Section from "./Section";

function Summary({ result }) {
  const summary = result?.summary || {
    average_stress: 58,
    average_confidence: 55,
    predominant_level: "HIGH",
    high: 58,
    moderate: 26,
    low: 16,
    total_windows: 1248,
  };

  const styles = {
    content: {
      flex: 1,
      padding: "18px",
      fontSize: "13px",
    },
    h3: {
      margin: "0 0 12px",
      fontSize: "18px",
    },
    cards: {
      display: "flex",
      gap: "10px",
      marginBottom: "14px",
    },
    card: {
      flex: 1,
      border: "1px solid #e1e8f5",
      borderRadius: "8px",
      padding: "10px",
      textAlign: "center",
    },
    small: {
      fontSize: "10px",
      fontWeight: "700",
      color: "#33456d",
    },
    red: {
      color: "#f5222d",
      fontSize: "22px",
      fontWeight: "800",
      marginTop: "4px",
    },
    green: {
      color: "#0ba84a",
      fontSize: "22px",
      fontWeight: "800",
      marginTop: "4px",
    },
    body: {
      display: "flex",
      gap: "14px",
      alignItems: "center",
    },
    donut: {
      width: "120px",
      height: "120px",
      borderRadius: "50%",
      background: `conic-gradient(#f5222d 0 ${summary.high}%, #ff8a00 ${summary.high}% ${
        summary.high + summary.moderate
      }%, #22b866 ${summary.high + summary.moderate}% 100%)`,
      display: "grid",
      placeItems: "center",
      textAlign: "center",
      fontSize: "12px",
      fontWeight: "700",
    },
    legend: {
      flex: 1,
      fontSize: "12px",
    },
    insight: {
      background: "#f4f8ff",
      border: "1px solid #d6e6ff",
      padding: "10px",
      borderRadius: "6px",
      marginTop: "10px",
      fontSize: "11px",
    },
  };

  return (
    <Section title="7. Summary & Insights">
      <Sidebar active="Summary & Insights" />

      <div style={styles.content}>
        <h3 style={styles.h3}>Session Summary</h3>

        <div style={styles.cards}>
          <div style={styles.card}>
            <div style={styles.small}>Average Stress Probability</div>
            <div style={styles.red}>{summary.average_stress}%</div>
          </div>

          <div style={styles.card}>
            <div style={styles.small}>Average Calibrated Confidence</div>
            <div style={styles.green}>{summary.average_confidence}%</div>
          </div>

          <div style={styles.card}>
            <div style={styles.small}>Predominant Stress Level</div>
            <div style={styles.red}>{summary.predominant_level}</div>
          </div>
        </div>

        <div style={styles.body}>
          <div style={styles.donut}>
            Total<br />Windows<br />{summary.total_windows}
          </div>

          <div style={styles.legend}>
            <p>🔴 High Stress {summary.high}%</p>
            <p>🟠 Moderate Stress {summary.moderate}%</p>
            <p>🟢 Low Stress {summary.low}%</p>
          </div>
        </div>

        <div style={styles.insight}>
          <b>Insights</b>
          <ul>
            <li>The participant shows high stress for the majority of the session.</li>
            <li>HRV and Heart Rate are major contributors.</li>
            <li>EDA peaks are higher during stress windows.</li>
          </ul>
        </div>
      </div>
    </Section>
  );
}

export default Summary;