import type { FaceCamera } from "./useFaceCamera";

export function CameraConsent({ camera }: { camera: FaceCamera }) {
  if (!camera.askConsent) return null;
  return (
    <div className="camera-consent-backdrop">
      <div
        className="camera-consent"
        role="dialog"
        aria-modal="true"
        aria-labelledby="camera-consent-title"
      >
        <span className="camera-consent-icon" aria-hidden="true">◉</span>
        <h3 id="camera-consent-title">Turn on your camera?</h3>
        <p>
          While you talk or type, I can also notice your facial expression as one
          gentle extra cue for how you might be feeling.
        </p>
        <ul>
          <li>Only used while you are recording or typing a message</li>
          <li>Pictures are never saved</li>
          <li>You can turn it off at any time</li>
        </ul>
        <div className="camera-consent-actions">
          <button className="camera-ui camera-consent-allow" onClick={() => void camera.start()}>
            Turn on camera
          </button>
          <button onClick={camera.decline}>Not now</button>
        </div>
      </div>
    </div>
  );
}

export function FacePreview({ camera }: { camera: FaceCamera }) {
  const { available, active, capturing, faceInView, attachVideo, start, turnOff } = camera;
  if (!available) return null;
  if (!active) {
    return (
      <button className="camera-ui camera-toggle" onClick={() => void start()} title="Turn camera on">
        <span aria-hidden="true">◉</span> Camera
      </button>
    );
  }
  const hint = capturing
    ? "Reading expression"
    : faceInView === false
      ? "Center your face"
      : faceInView
        ? "Face in view"
        : "Camera on";
  return (
    <div className={`face-preview${capturing ? " capturing" : ""}`}>
      <video ref={attachVideo} muted playsInline aria-label="Your camera preview" />
      <span className={`face-preview-hint${faceInView === false && !capturing ? " warn" : ""}`}>
        {hint}
      </span>
      <button className="camera-ui face-preview-close" onClick={turnOff} aria-label="Turn camera off">
        ×
      </button>
    </div>
  );
}
