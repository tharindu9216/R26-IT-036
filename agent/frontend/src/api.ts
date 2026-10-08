import type {
  Account,
  PatientSummary,
  ChatMessage,
  ChatResponse,
  QuestionnaireStatus,
  SensorStatus,
  TextModelStatus,
  VoiceModelStatus,
  VoiceChatResponse,
  XAIResponse,
} from "./types";

const BASE = "/api";

// Patient scope is always checked against the signed-in account on the server.
let selectedPatient = "";
export function selectPatient(id: string) { selectedPatient = id; }
export function sessionId() { return selectedPatient; }
function headers() {
  return { "X-SentiVera-Client": "web", ...(selectedPatient ? { "X-Patient-ID": selectedPatient } : {}) };
}
function failure(response: Response, detail: unknown): Error {
  if (response.status === 401 && !response.url.endsWith("/auth/login")) {
    window.dispatchEvent(new Event("session-expired"));
  }
  const message = (detail as { detail?: unknown } | null)?.detail;
  return new Error(typeof message === "string" ? message : `Request failed: ${response.status}`);
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...headers() },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw failure(response, detail);
  }
  return response.json() as Promise<T>;
}

async function requestForm<T>(path: string, form: FormData): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { method: "POST", body: form, credentials: "same-origin", headers: headers() });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw failure(response, detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  me: () => request<Account>("/auth/me"),
  login: (email: string, password: string, role: Account["role"]) => request<Account>("/auth/login", {
    method: "POST", body: JSON.stringify({ email, password, role }),
  }),
  register: (name: string, email: string, password: string) => request<Account>("/auth/register", {
    method: "POST", body: JSON.stringify({ name, email, password }),
  }),
  logout: () => request<{ ok: boolean }>("/auth/logout", { method: "POST" }),
  linkDoctor: (email: string) => request<{ name: string; email: string }>("/auth/doctor", {
    method: "POST", body: JSON.stringify({ email }),
  }),
  unlinkDoctor: () => request<{ ok: boolean }>("/auth/doctor", { method: "DELETE" }),
  patients: () => request<PatientSummary[]>("/doctor/patients"),
  activity: () => request<ChatMessage[]>("/doctor/activity"),
  health: () => request<{ ok: boolean }>("/health"),
  textModelStatus: () => request<TextModelStatus>("/models/text/status"),
  prepareTextModels: () =>
    request<TextModelStatus>("/models/text/prepare", { method: "POST" }),
  voiceModelStatus: () => request<VoiceModelStatus>("/models/voice/status"),
  prepareVoiceModels: () =>
    request<VoiceModelStatus>("/models/voice/prepare", { method: "POST" }),
  sensorStatus: (includeWaveform = false) =>
    request<SensorStatus>(
      `/sensor/status?session_id=${encodeURIComponent(sessionId())}&include_waveform=${includeWaveform}`,
    ),
  questionnaireStatus: () =>
    request<QuestionnaireStatus>(
      `/questionnaire/status?session_id=${encodeURIComponent(sessionId())}`,
    ),
  submitQuestionnaire: (answers: Record<string, number>) =>
    request<QuestionnaireStatus>("/questionnaire", {
      method: "POST",
      body: JSON.stringify({ answers, session_id: sessionId() }),
    }),
  startSensor: (port: string) =>
    request<SensorStatus>("/sensor/start", {
      method: "POST",
      body: JSON.stringify({ port }),
    }),
  uploadSensorRecording: (files: File[]) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return requestForm<SensorStatus>("/sensor/upload", form);
  },
  skipSensor: () => request<SensorStatus>("/sensor/skip", { method: "POST" }),
  disconnectSensor: () =>
    request<SensorStatus>("/sensor/disconnect", { method: "POST" }),
  restartSensor: () => request<SensorStatus>("/sensor/restart", { method: "POST" }),
  chatHistory: () =>
    request<ChatMessage[]>(
      `/chat/history?session_id=${encodeURIComponent(sessionId())}`,
    ),
  sendChat: (message: string) =>
    request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId() }),
    }),
  resetChat: () =>
    request<{ ok: boolean }>("/chat/reset", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId() }),
    }),
  sendVoice: (audio: Blob, filename = "voice-input.webm") => {
    const form = new FormData();
    form.append("audio", audio, filename);
    form.append("session_id", sessionId());
    return requestForm<VoiceChatResponse>("/voice/chat", form);
  },
  explainChat: (message: string, turnId: string) =>
    request<XAIResponse>("/xai", {
      method: "POST",
      body: JSON.stringify({ message, turn_id: turnId }),
    }),
};
