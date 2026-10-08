import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { ChatMessage } from "../types";
import { SupportContacts } from "./SupportContacts";
import { VoiceWaveMessage } from "./VoiceWaveMessage";


function recordingMimeType(): string | undefined {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/ogg;codecs=opus",
    "audio/webm",
  ];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
}

function extensionForMime(mime: string): string {
  return mime.includes("ogg") ? "ogg" : "webm";
}

export function VoiceChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [recording, setRecording] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const localAudioUrlsRef = useRef<string[]>([]);



  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  useEffect(
    () => () => {
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.onstop = null;
        recorder.stop();
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
      localAudioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    },
    [],
  );

  useEffect(() => {
    let active = true;
    api.chatHistory().then((history) => { if (active) setMessages(history); })
      .catch((error: unknown) => { if (active) setError(error instanceof Error ? error.message : "Could not load conversation."); });
    return () => { active = false; };
  }, []);

  async function submitAudio(audio: Blob, filename: string) {
    if (sending || audio.size === 0) return;
    setSending(true);
    setError(null);
    const localAudioUrl = URL.createObjectURL(audio);
    try {
      const result = await api.sendVoice(audio, filename);

      localAudioUrlsRef.current.push(localAudioUrl);
      setMessages((previous) => [
        ...previous,
        {
          role: "user",
          text: result.transcript,
          meta: {
            audio_url: localAudioUrl,
            audio_duration: result.duration_seconds,
          },
        },
        {
          role: "assistant",
          text: result.reply,
          meta: {
            support_contacts: result.support_contacts,
            input_text: result.transcript,
            turn_id: result.turn_id,
            audio_url: result.reply_audio_url,

          },
        },
      ]);
    } catch (caught) {
      URL.revokeObjectURL(localAudioUrl);
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setSending(false);
    }
  }

  async function startRecording() {
    if (recording || sending) return;
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("This browser does not support microphone recording. Upload an audio file instead.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = recordingMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onerror = () => setError("Microphone recording failed.");
      recorder.onstop = () => {
        const actualMime = recorder.mimeType || mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: actualMime });
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        recorderRef.current = null;
        setRecording(false);
        void submitAudio(blob, `voice-input.${extensionForMime(actualMime)}`);
      };
      recorder.start(250);
      setRecording(true);
    } catch (caught) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  function stopRecording() {
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }

  return (
    <div className="card chat-card voice-chat-card">
      <div className="chat-header">
        <div className="chat-heading">
          <span className="companion-avatar" aria-hidden="true">♡</span>
          <div>
            <h2>Voice check-in</h2>
            <span className="online-status"><i /> Listening when you’re ready</span>
          </div>
        </div>
      </div>
      <div className="chat-log">
        {messages.length === 0 && (
          <div className="empty-state">
            <span className="empty-state-icon" aria-hidden="true">♫</span>
            <h3>Take your time. I’m listening.</h3>
            <p>Tap record when you’re ready and speak naturally about what’s on your mind.</p>
          </div>
        )}
        {messages.map((message, index) => (
          <div key={index} className={`voice-message ${message.role}`}>
            {message.meta?.audio_url && (
              <VoiceWaveMessage
                audioUrl={message.meta.audio_url}
                transcript={message.text}
                role={message.role}
                knownDuration={message.meta.audio_duration}
                autoPlay={message.role === "assistant"}
              />
            )}
            {!message.meta?.audio_url && <p>{message.text}</p>}
            <SupportContacts contacts={message.meta?.support_contacts} />

          </div>
        ))}
        {sending && (
          <div className="bubble assistant thinking voice-thinking">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span>Taking a moment to reply…</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      {error && <p className="error">{error}</p>}
      <div className="voice-controls">
        <button
          className={recording ? "record-button recording" : "record-button"}
          onClick={recording ? stopRecording : startRecording}
          disabled={sending}
        >
          <span aria-hidden="true">{recording ? "■" : "●"}</span>
          {recording ? "Stop & send" : "Start recording"}
        </button>
        <label className="ghost upload-audio">
          Upload audio
          <input
            type="file"
            accept="audio/*,.webm,.m4a"
            disabled={recording || sending}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void submitAudio(file, file.name);
              event.currentTarget.value = "";
            }}
          />
        </label>
      </div>
      {recording && <p className="recording-status">Recording… speak naturally, then press Stop &amp; send.</p>}
    </div>
  );
}
