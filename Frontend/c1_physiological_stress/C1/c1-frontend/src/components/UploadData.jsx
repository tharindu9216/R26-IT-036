import { useState } from "react";
import axios from "axios";
import Sidebar from "./Sidebar";
import Section from "./Section";

function UploadData({ apiBaseUrl, setResult, setLoading }) {
  const [subject, setSubject] = useState("S2");
  const [session, setSession] = useState("Stress Session");
  const [error, setError] = useState("");

  const runPrediction = async () => {
    try {
      setError("");
      setLoading(true);

      const response = await axios.post(`${apiBaseUrl}/predict`, {
        subject,
        session,
      });

      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError("Backend connection failed. Check Colab/ngrok URL.");
    } finally {
      setLoading(false);
    }
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
    p: {
      margin: "0 0 14px",
      color: "#33456d",
      fontSize: "12px",
    },
    row: {
      display: "flex",
      gap: "12px",
      marginBottom: "12px",
    },
    label: {
      fontWeight: "700",
      fontSize: "12px",
      display: "block",
      marginBottom: "5px",
    },
    select: {
      width: "150px",
      padding: "8px",
      border: "1px solid #cbd8ee",
      borderRadius: "5px",
      background: "#ffffff",
    },
    box: {
      border: "1px solid #d6e1f3",
      borderRadius: "7px",
      padding: "12px",
      margin: "12px 0",
      background: "#f8fbff",
    },
    info: {
      background: "#eef6ff",
      border: "1px solid #c8def8",
      color: "#0057d9",
      padding: "9px",
      borderRadius: "5px",
      fontSize: "12px",
      marginTop: "10px",
      fontWeight: "600",
    },
    button: {
      width: "100%",
      background: "#0057e7",
      color: "#ffffff",
      border: "none",
      padding: "10px",
      borderRadius: "5px",
      cursor: "pointer",
      fontWeight: "700",
    },
    error: {
      color: "red",
      marginTop: "8px",
      fontSize: "12px",
    },
  };

  return (
    <Section title="1. Upload / Select Data">
      <Sidebar active="Upload / Select Data" />

      <div style={styles.content}>
        <h3 style={styles.h3}>Upload or Select Data</h3>
        <p style={styles.p}>Choose a subject and session from the WESAD dataset</p>

        <div style={styles.row}>
          <div>
            <label style={styles.label}>Select Subject</label>
            <select
              style={styles.select}
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
            >
              <option>S2</option>
              <option>S3</option>
              <option>S4</option>
              <option>S5</option>
              <option>S6</option>
            </select>
          </div>

          <div>
            <label style={styles.label}>Select Session</label>
            <select
              style={styles.select}
              value={session}
              onChange={(e) => setSession(e.target.value)}
            >
              <option>Stress Session</option>
              <option>Baseline Session</option>
            </select>
          </div>
        </div>

        <div style={styles.box}>
          <b>About WESAD Dataset</b>
          <ul>
            <li>15 Subjects</li>
            <li>Sensors: EDA, ECG used in C1</li>
            <li>Labels: Baseline, Stress</li>
          </ul>

          <div style={styles.info}>
            ⓘ Dataset is stored in Google Drive and accessed by backend.
          </div>
        </div>

        <button style={styles.button} onClick={runPrediction}>
          Run Stress Detection
        </button>

        {error && <div style={styles.error}>{error}</div>}
      </div>
    </Section>
  );
}

export default UploadData;