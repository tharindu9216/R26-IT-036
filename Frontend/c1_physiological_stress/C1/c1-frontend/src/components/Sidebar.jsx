function Sidebar({ active }) {
  const items = [
    "Dashboard",
    "Upload / Select Data",
    "Preprocessing",
    "Stress Prediction",
    "Physiological Features",
    "Explainability (SHAP)",
    "Window Results",
    "Summary & Insights",
    "Download / Export",
    "Settings",
    "About",
  ];

  const styles = {
    sidebar: {
      width: "132px",
      background: "#001a38",
      color: "#ffffff",
      padding: "12px 8px",
      fontSize: "11px",
    },
    logo: {
      fontWeight: "800",
      fontSize: "12px",
      marginBottom: "14px",
      lineHeight: "1.4",
    },
    item: {
      padding: "6px 7px",
      margin: "3px 0",
      borderRadius: "4px",
      cursor: "pointer",
      whiteSpace: "nowrap",
    },
    active: {
      background: "#0057d9",
      fontWeight: "700",
    },
  };

  return (
    <div style={styles.sidebar}>
      <div style={styles.logo}>▣ C1 Stress Detection</div>

      {items.map((item) => (
        <div
          key={item}
          style={{
            ...styles.item,
            ...(item === active ? styles.active : {}),
          }}
        >
          {item}
        </div>
      ))}
    </div>
  );
}

export default Sidebar;