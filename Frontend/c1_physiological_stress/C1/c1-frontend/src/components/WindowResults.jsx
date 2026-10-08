import Sidebar from "./Sidebar";
import Section from "./Section";

function WindowResults({ result }) {
  const rows = result?.window_results || [
    [241, "1205 - 1210", "34%", "32%", "LOW STRESS"],
    [242, "1210 - 1215", "62%", "58%", "MODERATE"],
    [243, "1215 - 1220", "76%", "72%", "HIGH STRESS"],
    [244, "1220 - 1225", "82%", "78%", "HIGH STRESS"],
    [245, "1225 - 1230", "84%", "79%", "HIGH STRESS"],
    [246, "1230 - 1235", "65%", "61%", "MODERATE"],
    [247, "1235 - 1240", "28%", "28%", "LOW STRESS"],
  ];

  const styles = {
    content: {
      flex: 1,
      padding: "18px",
      fontSize: "12px",
    },
    h3: {
      margin: "0 0 12px",
      fontSize: "18px",
    },
    table: {
      width: "100%",
      borderCollapse: "collapse",
      fontSize: "11px",
    },
    th: {
      background: "#f4f7fb",
      border: "1px solid #d8e1f0",
      padding: "7px",
      textAlign: "center",
    },
    td: {
      border: "1px solid #d8e1f0",
      padding: "7px",
      textAlign: "center",
      fontWeight: "700",
    },
    red: {
      color: "#f5222d",
    },
    green: {
      color: "#0ba84a",
    },
    orange: {
      color: "#ff8a00",
    },
    pages: {
      textAlign: "center",
      marginTop: "13px",
      color: "#0057ff",
      fontWeight: "700",
    },
  };

  const color = (text) => {
    if (text.includes("HIGH")) return styles.red;
    if (text.includes("LOW")) return styles.green;
    return styles.orange;
  };

  return (
    <Section title="6. Window-wise Results">
      <Sidebar active="Window Results" />

      <div style={styles.content}>
        <h3 style={styles.h3}>Window-wise Prediction Results (5-second windows)</h3>

        <table style={styles.table}>
          <thead>
            <tr>
              <th style={styles.th}>Window</th>
              <th style={styles.th}>Time Range</th>
              <th style={styles.th}>Stress Probability</th>
              <th style={styles.th}>Calibrated Confidence</th>
              <th style={styles.th}>Prediction</th>
            </tr>
          </thead>

          <tbody>
            {rows.map((r) => (
              <tr key={r[0]}>
                <td style={styles.td}>{r[0]}</td>
                <td style={styles.td}>{r[1]}</td>
                <td style={styles.td}>{r[2]}</td>
                <td style={styles.td}>{r[3]}</td>
                <td style={{ ...styles.td, ...color(r[4]) }}>{r[4]}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div style={styles.pages}>‹ &nbsp; 1 &nbsp; 2 &nbsp; 3 &nbsp; 4 &nbsp; 5 &nbsp; ... &nbsp; 125 &nbsp; ›</div>
      </div>
    </Section>
  );
}

export default WindowResults;