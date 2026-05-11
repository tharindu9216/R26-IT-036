function Section({ title, children }) {
  const styles = {
    wrapper: {},
    title: {
      textAlign: "center",
      color: "#0057ff",
      fontSize: "18px",
      margin: "10px 0 6px",
      fontWeight: "800",
    },
    panel: {
      height: "320px",
      border: "1px solid #cbd8ee",
      borderRadius: "8px",
      display: "flex",
      overflow: "hidden",
      background: "#ffffff",
      boxShadow: "0 2px 8px #dce5f2",
    },
  };

  return (
    <div style={styles.wrapper}>
      <h2 style={styles.title}>{title}</h2>
      <div style={styles.panel}>{children}</div>
    </div>
  );
}

export default Section;