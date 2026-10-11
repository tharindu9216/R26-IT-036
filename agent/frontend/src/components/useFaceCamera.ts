import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

// One still every 0.7 s while the user speaks or types, at most MAX_FRAMES per
// turn (the backend's facial_stress.max_frames), spread evenly across the turn.
const FRAME_INTERVAL_MS = 700;
const MAX_FRAMES = 40;
const FRAME_WIDTH = 480;
const PREVIEW_CHECK_MS = 2000;
const DECISION_KEY = "sentivera-camera-decision";

type Decision = "granted" | "declined";

function readDecision(): Decision | null {
  try {
    const value = window.sessionStorage.getItem(DECISION_KEY);
    return value === "granted" || value === "declined" ? value : null;
  } catch {
    return null;
  }
}

function saveDecision(value: Decision) {
  try {
    window.sessionStorage.setItem(DECISION_KEY, value);
  } catch {
    // Private mode or blocked storage: the popup simply asks again next visit.
  }
}

function thin<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  const step = (items.length - 1) / (limit - 1);
  return Array.from({ length: limit }, (_, index) => items[Math.round(index * step)]);
}

/**
 * Optional webcam for voice and text mode: asks once, shows a live preview, and
 * captures JPEG stills only between beginCapture() and endCapture() -- i.e.
 * while the user is recording or composing a message -- so the frames line up
 * with what they said.
 */
export function useFaceCamera(paused: boolean) {
  const [available, setAvailable] = useState(false);
  const [askConsent, setAskConsent] = useState(false);
  const [active, setActive] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [faceInView, setFaceInView] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const framesRef = useRef<Blob[]>([]);
  const captureTimerRef = useRef<number | undefined>(undefined);
  const mountedRef = useRef(true);

  const grabFrame = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.readyState < 2 || !video.videoWidth) return null;
    const canvas = canvasRef.current ?? (canvasRef.current = document.createElement("canvas"));
    const scale = Math.min(1, FRAME_WIDTH / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
  }, []);

  const stop = useCallback(() => {
    window.clearInterval(captureTimerRef.current);
    captureTimerRef.current = undefined;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    framesRef.current = [];
    setCapturing(false);
    setActive(false);
    setFaceInView(null);
  }, []);

  const start = useCallback(async () => {
    setAskConsent(false);
    setError(null);
    saveDecision("granted");
    if (streamRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser cannot open the camera.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      // Left the chat while the permission prompt was open: release it.
      if (!mountedRef.current || streamRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      setActive(true);
    } catch {
      setError("The camera could not be opened. You can carry on without it.");
    }
  }, []);

  const decline = useCallback(() => {
    saveDecision("declined");
    setAskConsent(false);
  }, []);

  const turnOff = useCallback(() => {
    saveDecision("declined");
    stop();
  }, [stop]);

  const beginCapture = useCallback(() => {
    if (!streamRef.current || captureTimerRef.current !== undefined) return;
    framesRef.current = [];
    const capture = () => {
      void grabFrame().then((frame) => {
        if (!frame || !streamRef.current) return;
        framesRef.current.push(frame);
        // A message can take minutes to type; keep memory bounded while still
        // covering the whole turn.
        if (framesRef.current.length >= MAX_FRAMES * 2) {
          framesRef.current = thin(framesRef.current, MAX_FRAMES);
        }
      });
    };
    capture();
    captureTimerRef.current = window.setInterval(capture, FRAME_INTERVAL_MS);
    setCapturing(true);
  }, [grabFrame]);

  const endCapture = useCallback((): Blob[] => {
    window.clearInterval(captureTimerRef.current);
    captureTimerRef.current = undefined;
    setCapturing(false);
    const frames = thin(framesRef.current, MAX_FRAMES);
    framesRef.current = [];
    return frames;
  }, []);

  // Offer the camera only when the server can actually use it.
  useEffect(() => {
    let alive = true;
    api.faceModelStatus()
      .then((status) => {
        if (!alive || !status.available) return;
        setAvailable(true);
        const decision = readDecision();
        if (decision === "granted") void start();
        else if (decision === null) setAskConsent(true);
      })
      .catch(() => undefined);
    return () => { alive = false; };
  }, [start]);

  // Callback ref: the <video> mounts with the preview, after the stream exists.
  const attachVideo = useCallback((video: HTMLVideoElement | null) => {
    videoRef.current = video;
    if (!video || !streamRef.current) return;
    video.srcObject = streamRef.current;
    void video.play().catch(() => undefined);
  }, []);

  // "Is my face in view?" hint, while idle only -- never during a turn.
  useEffect(() => {
    if (!active || paused || capturing) return;
    let alive = true;
    let inFlight = false;
    const check = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const frame = await grabFrame();
        if (frame && alive) {
          const result = await api.faceCheck(frame);
          if (alive) setFaceInView(result.face_detected);
        }
      } catch {
        // A missed hint is harmless; the next tick tries again.
      } finally {
        inFlight = false;
      }
    };
    void check();
    const timer = window.setInterval(check, PREVIEW_CHECK_MS);
    return () => { alive = false; window.clearInterval(timer); };
  }, [active, paused, capturing, grabFrame]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      stop();
    };
  }, [stop]);

  return {
    available,
    askConsent,
    active,
    capturing,
    faceInView,
    error,
    attachVideo,
    start,
    decline,
    turnOff,
    beginCapture,
    endCapture,
  };
}

export type FaceCamera = ReturnType<typeof useFaceCamera>;
