import Sidebar from "./Sidebar";
import Section from "./Section";

function DownloadExport({ result }) {
  const downloadJSON = () => {
    const data = result || {
      message: "Demo result. Connect backend to download real output.",
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "c1-stress-result.json";
    a.click();
  };

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
      marginBottom: "14px",
    },
    row: {
      border: "1px solid #e1e8f5",
      padding: "9px",
      borderRadius: "6px",
      margin: "8px 0",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
    },
    button: {
      background: "#ffffff",
      color: "#0057e7",
      border: "1px solid #8bb8ff",
      padding: "6px 12px",
      borderRadius: "5px",
      cursor: "pointer",
      fontWeight: "700",
    },
    info: {
      background: "#eef6ff",
      border: "1px solid #c8def8",
      padding: "10px",
      borderRadius: "6px",
      color: "#0057d9",
      fontSize: "12px",
      marginTop: "14px",
    },
  };

  const files = [
    "Export Window-wise Results (CSV)",
    "Export SHAP Explanations (CSV)",
    "Export Summary Report (PDF)",
    "Export All Results (ZIP)",
  ];

  return (
    <Section title="8. Download / Export">
      <Sidebar active="Download / Export" />

      <div style={styles.content}>
        <h3 style={styles.h3}>Download Results</h3>
        <div style={styles.small}>Download predictions, explanations and reports.</div>

        {files.map((file) => (
          <div style={styles.row} key={file}>
            <span>{file}</span>
            <button style={styles.button} onClick={downloadJSON}>
              Download
            </button>
          </div>
        ))}

        <div style={styles.info}>
          All files include subject info, timestamps and calibrated predictions.
        </div>
      </div>
    </Section>
  );
}

export default DownloadExport;